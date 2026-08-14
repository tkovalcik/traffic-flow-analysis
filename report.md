# Traffic Flow Analysis
### A reproducible streaming data product for highway vehicle counting

**MSDS 682 - Data Stream Processing · Final Project**\
Tom Kovalcik and Christopher Monzon · August 14, 2026

---

## 1. Problem, target user, and useful result

Traffic engineers plan signal timing, ramp metering, and capacity studies from
**15-minute directional volume counts** - the standard interval in the field.
Getting those counts usually means either paying for pneumatic tube counts and
manual studies, or waiting on loop-detector infrastructure that exists only on
instrumented segments. Meanwhile, state DOTs already operate hundreds of public
CCTV cameras pointed at exactly the roadway a count would cover. Those cameras
are watched by humans, or not watched at all, and the vehicles they see are
never counted.

**Target user:** a traffic engineer or transportation analyst who needs
directional volume for a corridor that has a public camera but no counting
hardware, and who also needs to know when a camera has gone dark so they do not
silently plan against missing data.

**Useful, observable result:** the system produces a **15-minute volume table**
broken down by camera, direction, and vehicle class, plus a **live alert stream**
for congestion anomalies and camera health. Both are real artifacts a user
consumes, not intermediate state:

| Artifact | Form | Consumer |
|---|---|---|
| Volume table | CSV, one row per window × camera × direction × class | Analysis, spreadsheets, dashboard |
| Alerts | JSONL **and** the `traffic.alerts` Kafka topic | Operators, downstream subscribers |

A representative row and alert, both from the committed demo run:

```
window_start,window_end,camera_id,direction,vehicle_class,count
2026-08-09T18:46:00+00:00,2026-08-09T18:47:00+00:00,tva43,EB,car,39

{"alert_type":"camera_stale","camera_id":"tva43_mirror",
 "message":"tva43_mirror: no vehicle events for 359s", ...}
```

The scope is deliberately one corridor and a handful of cameras. The goal was
one complete path that another person can run and verify, not breadth.

---

## 2. Data source and event contract

### 2.1 Source and limitations

Live public highway CCTV from **Caltrans District 4**, accessed through the
public traffic-camera map. The validation camera for everything in this report
is **`tva43` - I-580 at Grand/Lakeshore, Oakland**; a second camera, `tv516`
(I-80 west of Ashby, Emeryville), was calibrated and used for lane-geometry
work. Full documentation is in `DATA_SOURCE.md`; the essentials:

- **Access:** public web streams, no API key, no documented rate limit. We
  sample 2–4 cameras at native stream rate - comparable load to a person
  watching the public page. **Stream URLs are not committed**; they are supplied
  through `.env` (see `.env.example`).
- **Rights:** Caltrans' Conditions of Use (accessed 2026-08-09) state website
  content is generally public domain and may be distributed or copied as
  permitted by law, with no camera-feed-specific restriction or attribution
  requirement stated. Our use is educational and non-commercial. We do not
  rebroadcast streams.
- **Privacy:** no PII is extracted or stored. Video is processed in memory on
  the perception node and discarded; only detection metadata leaves the node.
  Faces and plates are not resolvable at these resolutions.
- **Measured stream behavior (2026-08-09):** HLS over HTTPS from a Wowza server,
  10-second segments with a 3-segment live window, giving **~30 s of live-edge
  latency** (verified against the cameras' burned-in clocks). `tva43` is
  720×480 H.264 @ 30 fps, ~570 kbps.
- **Continuity:** a 15-minute continuous capture completed with **0 reconnects,
  0 read failures**, all 27,000 expected frames, and a worst inter-frame stall
  of 9.7 s absorbed by buffering with no frame loss. 15-minute continuous
  recording - the interval the product is built around - is therefore supported
  rather than assumed.
- **Honest limitation:** these are best-effort public feeds. Cameras go down,
  change URLs, and serve stale images. The pipeline treats a silent camera as an
  alert condition, not an error.

### 2.2 The validated event contract

The unit of the system is a **vehicle crossing event**: one vehicle crossed one
calibrated counting line in one direction. The canonical contract is an Avro
schema in `src/streaming/schemas/` registered in Schema Registry, mirrored by a
Pydantic model used for validation. Topic `vehicle.events`, **key =
`camera_id`**.

| Field | Type | Role |
|---|---|---|
| `event_id` | string (uuid) | unique per crossing |
| `camera_id` | string | **partition key**, stable per camera |
| `ts_event` | timestamp-millis | frame time - **the event-time clock all windowing uses** |
| `ts_publish` | timestamp-millis | producer wall clock, for latency measurement |
| `track_id` | long | ByteTrack persistent id |
| `vehicle_class` | enum | car / truck / bus / motorcycle |
| `direction` | enum | which counting line and travel direction |
| `confidence` | float | detector confidence at the crossing |

A real event, from `data/sample/vehicle_events_sample.jsonl`:

```json
{"event_id": "bfdd553d-11a5-4439-8dfb-88fb99b20dbb", "camera_id": "tva43",
 "ts_event": 1786264793966, "ts_publish": 1786267097220, "track_id": 21,
 "vehicle_class": "car", "direction": "EB", "confidence": 0.3776901364326477}
```

Two contract decisions carry real weight:

**Keying by `camera_id`** puts every event from one camera on one partition, so
a camera's events stay ordered and its windows can close on a per-camera
watermark. The demo proves this holds rather than asserting it: `tva43` lands on
partition 0, `tva43_mirror` on partition 2, and partition 1 stays empty.

**Carrying both `ts_event` and `ts_publish`** separates when a vehicle passed
from when we managed to tell Kafka about it. Windowing uses `ts_event` only.
This is what makes a replay at 60× produce the same counts as a replay at
real time - the arrival rate is not part of the result.

---

## 3. Architecture and implementation

### 3.1 The path

```
Live CCTV (Caltrans D4, 2-4 cameras)        [or] recorded detection JSONL
        │                                         │
        ▼                                         │
Perception node: decode → YOLO11n → ByteTrack     │  (replay producer,
  → counting-line crossing  (video stays here)    │   original timestamps)
        │                                         │
        └────────────────┬────────────────────────┘
                         ▼   Avro + Schema Registry
          Kafka topic: vehicle.events   (key = camera_id, 3 partitions)
                         │
                         ▼
     Stream processor: event-time tumbling windows, per-camera
       watermarks, EWMA baseline, alert rules
                         │
        ┌────────────────┼─────────────────┐
        ▼                ▼                 ▼
  volume CSV        alerts JSONL     Kafka: traffic.alerts
        │                │
        └──── Dashboard (FastAPI) ────┘
```

### 3.2 Why this design

**Detection metadata crosses the network, never video.** A crossing event is a
few hundred bytes; the 570 kbps stream it came from is not. Putting perception
at the edge means the Kafka path scales with vehicles, not pixels, and the
privacy story becomes structural rather than a promise - the video is discarded
on the node that decoded it.

**Event time, not processing time.** Windows close on a per-camera watermark
derived from `ts_event`. This was the single most consequential choice: it makes
the recorded path and the live path produce identical results from identical
input, which is what makes the whole system testable at all. Processing-time
windows would have made every run a different run.

**Per-camera watermarks.** One silent camera must not stall another camera's
windows. A separate cross-camera `stream_time` is what the staleness rule
compares against - which is precisely how a camera going dark becomes
detectable.

**Late events are dropped and tallied, not silently discarded.** A window that
has closed will not reopen; anything arriving for it increments a visible
counter. A reviewer can therefore see that the demo drops **0** events as late,
rather than trusting that it doesn't.

**A pretrained detector, with no training.** YOLO11n on pretrained COCO weights
plus ByteTrack - see §4.2. The project's contribution is the streaming path, and
introducing a training loop would have added a large unverifiable surface for no
gain to the stated problem.

### 3.3 Components

| Module | Responsibility |
|---|---|
| `src/perception/` | capture, detection, tracking, counting-line crossing logic |
| `src/streaming/` | event contracts, windows, consumer loop, alert rules, output writers |
| `src/replay/` | session recorder and deterministic replay producer |
| `src/dashboard/` | FastAPI dashboard over the output files |
| `src/triage/` | camera-inventory scanner (which public cameras actually work) |

Counting lines are calibrated per camera and **motion-gated**: each line carries
a calibrated flow vector, and a track moving against it cannot fire that line.
This was not cosmetic - ungated per-flow lines overcounted roughly 2× on `tva43`
because the opposite flow's vehicles crossed the line in the far field.

---

## 4. Evidence and reproducibility

### 4.1 Exact review steps

No GPU, no cloud, no camera access. Docker and Python 3.11 are the only
prerequisites.

```bash
uv venv --python 3.11
uv pip install "confluent-kafka[avro,schemaregistry]" pydantic python-dotenv
cp .env.example .env      # local defaults work as-is
./scripts/demo.sh
```

The script brings up local Kafka, recreates both topics at 3 partitions, replays
the recorded 15-minute capture at 60×, and processes it. **Expected output**,
reproduced verbatim from a real run (`evaluation/results/`):

```
volume table : outputs/volume_demo_60s.csv (62 rows)
alert log    : outputs/alerts.jsonl (1 alerts)

counted per camera:
  tva43          2401
  tva43_mirror    794

alerts raised (also published to traffic.alerts):
  [camera_stale] tva43_mirror: no vehicle events for 359s

Partition spread (key=camera_id keeps a camera's events ordered on one partition):
  vehicle.events:0:2401
  vehicle.events:1:0
  vehicle.events:2:794
```

Cold start end to end is ~73 s. A cold broker prints one
`Coordinator load in progress` warning - expected, and documented: Kafka reports
its container healthy before the transaction coordinator loads, and the load is
triggered *by* the first producer request.

**The counts are hand-checkable against the input.** `wc -l
data/sample/replay_tva43_15min.jsonl` returns 2401, and the demo counts 2401 on
`tva43`. Every event in, every event counted, none dropped as late.

### 4.2 The bounded AI element and how it is verified

**What AI owns:** per-frame vehicle detection (YOLO11n, pretrained COCO weights,
pinned, no training) and multi-object association (ByteTrack). The boundary is
narrow and deliberate - sampled frames in, schema-validated crossing events out.
Everything downstream is deterministic code.

**Accepted / rejected:** COCO classes 2/3/5/7 (car, motorcycle, bus, truck) at
confidence ≥ 0.35, and a detection must cross a calibrated, motion-gated
counting line to become an event. Thresholds were fixed at calibration time and
not retuned afterwards.

**Verified:** that emitted events are schema-valid, correctly keyed and
partitioned, and that they window and count deterministically - 177 automated
tests in CI, plus the end-to-end reproduction below.

**Not verified - detection accuracy against ground truth.** No count MAE, no
precision/recall. We state this rather than let it be discovered. Measuring it
needs hand-labeled frames, and labeling needs the source clips plus a working
perception environment; `torch`, `opencv-python` and `ultralytics` publish no
x86_64 macOS wheels and cannot be installed on the machine that did this work.
Labeling ~100 frames was scoped and not finished before the deadline.

**Known limitation, disclosed:** **night scenes fail outright.** On ~2 AM
captures the pretrained detector missed the large majority of passing vehicles
(headlight glare, low contrast). Accuracy is claimed for daylight only. A
no-training mitigation path is documented as future work; fine-tuning is
explicitly out of scope.

**Fallback:** recorded detection JSONL drives the identical downstream pipeline,
so the system is fully demonstrable without live inference - which is exactly
what the graded review path uses.

**AI-assisted development (disclosed):** Claude Code was used for scaffolding,
boilerplate, and non-core components under human direction. The core Kafka path
- producer setup, consumer loop, window logic - was hand-written by the team.
Architecture, scope, and design decisions are ours. AI-assisted code was
verified by team review, the pytest suite, and the deterministic replay check.
Full disclosure in `AI_USAGE.md`.

### 4.3 Validation

| Evidence | Result |
|---|---|
| Automated suite (CI, `ubuntu-latest`) | **177 tests pass** |
| Automated suite (Intel Mac, streaming subset) | **118 tests pass** |
| Lint + format gate on every PR | `ruff check` → `ruff format --check` → `pytest`; red CI never merges |
| End-to-end demo | 3195 events published, **0 failed**, 44 windows, 62 rows, **0 late dropped**, 1 alert |

The suite concentrates on parts where correctness is not obvious by inspection:
window assignment and late-event handling, per-camera watermarks, the poll loop
and offset handling, Avro/Pydantic agreement and rejection of malformed events,
keying, EWMA baseline and threshold rules, crossing logic and double-count
suppression. Breakdown in `evaluation/test-evidence.md`.

**Reproducibility was verified by destroying the environment, not by re-running
in place.** The path has been reproduced from three fresh clones (two local, one
from GitHub `main`) and against a broker destroyed with `docker compose down -v`
and rebuilt from nothing. **The first fresh-clone attempt failed** - the
documented setup recipe had never been executed on a machine without a prepared
`.venv`, and it was wrong. That failure is the reason the check exists, and the
recipe in §4.1 is the corrected one, re-verified from a second clean clone.

### 4.4 Bonus extension - controlled reproducibility comparison

*Clearly labeled as the optional extension; additional to §4.3, not a substitute.*

One input held fixed, conditions varied one at a time, one named metric compared
across all of them.

| | |
|---|---|
| **Input (fixed)** | `data/sample/replay_tva43_15min.jsonl` - 2401 events, `tva43`, 2026-08-09 18:46:30Z–19:01:30Z |
| **Conditions varied** | arrival pacing (`--speed 0` vs `--speed 60`); broker warm vs destroyed and rebuilt; fault injection present vs absent |
| **Metric** | SHA-1 of the sorted volume table, plus per-camera counts, row count, alert count |
| **Implementation** | `evaluation/extension/validate.sh` - 16 assertions, exits non-zero on any failure |
| **Saved output** | `evaluation/extension/results/` - transcripts, volume tables, alert logs, summaries |

```bash
./evaluation/extension/validate.sh                  # warm broker,  ~2 min
./evaluation/extension/validate.sh --include-cold   # rebuilds broker, ~4 min
```

**Result - 16 checks passed, 0 failed:**

| Condition | Sorted-table SHA-1 |
|---|---|
| `--speed 0` | `1bd6cba1c010d82951c9c340428dab30dbcc40ff` |
| `--speed 60` | `1bd6cba1c010d82951c9c340428dab30dbcc40ff` |
| no-fault control | `c08531859140dc98ae60a660e96b93bc241fe31d` |

The first two are identical, which is the claim under test: the pipeline windows
on event time, so **the wall-clock rate at which events reach the broker does not
change the result**. The third run is the control - it removes the injected
fault and shows the alert is caused by the injected silence and not by the
recorded data, while `tva43`'s counts stay identical whether the synthetic
camera is present or not (2401 events, 32 windows, 46 rows, 0 alerts).

### 4.5 How the alert evidence was produced - and why that way

A single-camera recording **cannot** exercise a staleness rule: a lone camera's
stream time is its own last event, so its silence gap is always zero. Two
options were rejected before the one we used:

- **Lowering the spike/drop thresholds** until something fired. At ~0.5 the only
  thing that fires is the end-of-file truncation artifact, which is not a signal.
- **Recapturing at commute hour.** Correct, but it needed camera access hours
  before the deadline.

**Chosen:** `--mirror-camera` replays the capture under a second, clearly
synthetic id (`tva43_mirror`) and `--drop-after` silences it 300 s in. The
surviving real camera keeps advancing `stream_time`, which is the only way a
recording can produce a genuine silence gap. **The mirror dies, never the real
camera** - `--drop-after` is rejected without `--mirror-camera` - so `tva43`
holds the exact 2401-event / 32-window baseline that was verified before the
fault injection existed, and the injection cannot be accused of perturbing the
real result.

The cut point was **not** tuned to force a second alert. We expected the
mirror's truncated tail might also fire `volume_drop`; at a round 300 s it does
not (the tail holds ~half a window, ratio ~0.56 against a 0.35 threshold).
Moving the cut to ~280 s would have fired it. We deliberately did not - choosing
a cut point to manufacture an alert is the same sin as loosening thresholds.

### 4.6 Limitations, collected

- **Detection accuracy is unmeasured** (§4.2), and night scenes fail outright.
- **`volume_spike` and `volume_drop` have never fired on real data.** This
  corridor's volume does not move enough at any window size. **The demo
  exercises 1 of 3 alert rules.** Thresholds were not loosened and the fault cut
  point was not tuned to change that.
- **`camera_stale` fires only against an injected synthetic camera**, for the
  structural reason in §4.5. The gap it reports varies with consumption progress
  (359 s / 840 s observed).
- **Volume-table row order is not byte-stable** once two cameras close windows on
  independent watermarks and their rows interleave by arrival. Counts are exact;
  reproducibility is claimed over the **sorted** table. The original determinism
  claim was written and tested for one camera and did not survive a second - a
  claim holding only under the conditions it was written for.
- **The demo runs 60-second windows, not the 900-second windows the architecture
  describes.** A 15-minute capture yields only two 15-minute windows, one
  truncated - a nearly empty table. The code default is unchanged
  (`DEFAULT_WINDOW_SECONDS`); the demo passes `--window-seconds` and writes
  `volume_demo_60s.csv`, so no artifact is mislabelled.
- **The live Confluent Cloud path is not part of the graded run.** Cloud credits
  ran out; local Kafka covers the path end to end with identical code, selected
  by environment configuration.

### 4.7 What we would do differently

Three findings came from running things, not reasoning about them - which is
itself the lesson:

1. **A documented setup recipe that was never executed was wrong.** It only
   worked because of a prepared `.venv` no reviewer would have. The fresh-clone
   test is what converted "works" into "works for someone else," and it failed
   the first time.
2. **A container reporting `healthy` is not a readiness signal.** Kafka goes
   green before its transaction coordinator loads. The obvious fix - sleeping
   before producing - does *not* work, because the load is triggered by the
   producer's first request. Waiting for something nothing has asked for yet is
   a no-op.
3. **A determinism claim held only under the conditions it was written for.**
   True for one camera; a second broke byte-stability while leaving every count
   exact.

Given more time, the top priority is the unmeasured accuracy gap: label ~100
frames and report count MAE and detection precision/recall, then the night
detection profile (higher inference resolution, lower confidence, contrast
preprocessing - all without training).

---

## 5. Contributions

Both team members can explain the complete architecture, the code path, the AI
usage, and the evaluation. Work was tracked task-by-task in `TEAM_PLAN.md`
throughout, which doubles as the contribution record. Split is approximately
50-50.

**Tom Kovalcik - perception, infrastructure, and camera science**
Repo scaffold and submission layout; CI (ruff + pytest) and branch protection;
camera triage scanner and the District 4 inventory scan; GCP project, capture VM, 
and storage bucket; capture module with HLS decode and reconnect handling; 
YOLO11n + ByteTrack integration; counting-line configuration and crossing logic; 
the Avro event contract and Pydantic model; session recorder and the golden 
15-minute capture; local Kafka compose; volume/alert output writers; alert rule 
definitions; FastAPI dashboard; staged demo renders; and the lane-geometry and 
scene-segmentation work behind the speed-estimation design.

**Christopher Monzon - the streaming path and reproducibility**
Kafka producer (keying, idempotent config, local/Confluent selection);
consumer loop and event-time tumbling windows with per-camera watermarks and
late-event handling; alert wiring and the `traffic.alerts` producer; the
deterministic replay producer including pacing, late-event injection, and the
mirror-camera fault injection; the one-command reviewer demo; fresh-clone
verification of the review path; the pytest suite for the streaming path; and
the evaluation package including the bonus reproducibility extension.

---

## 6. Submission map

| Required item | Location |
|---|---|
| Setup, review path, expected output, cleanup | `README.md` |
| Data source, rights, schema, rate limits, replay | `DATA_SOURCE.md` |
| AI task, evidence, decisions, verification, limits | `AI_USAGE.md` |
| Pinned dependencies | `requirements.txt`, `uv.lock`, `pyproject.toml` |
| Environment template (no secrets) | `.env.example` |
| Ingestion, contracts, Kafka processing, output | `src/` |
| Sample / replay data | `data/sample/` |
| Representative output artifact | `outputs/` |
| Validation artifact | `evaluation/` |
| Bonus extension | `evaluation/extension/` |
| Contribution record | `TEAM_PLAN.md`, §5 above |

No credentials, `.env`, virtual environments, or caches are included in the
submitted package.
