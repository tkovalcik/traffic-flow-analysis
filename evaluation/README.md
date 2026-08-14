# Validation

What has been verified about this system, how to reproduce each claim, and —
equally important — what has *not* been verified.

| Artifact | What it establishes |
|---|---|
| [`test-evidence.md`](test-evidence.md) | Automated suite: 177 tests green in CI, 118 locally |
| [`results/`](results/) | Output of the reviewer demo, as a TA would see it |
| [`extension/`](extension/) | **Bonus extension** — controlled reproducibility comparison (separate and additional to this base validation) |

## 1. The logic is covered by tests

177 tests pass in GitHub Actions on `main`, covering window semantics, the
consumer loop, the event contract, alert rules, crossing logic and the replay
producer. Lint and format checks gate every pull request; red CI never merges.
Breakdown, coverage map, and reproduction commands: [`test-evidence.md`](test-evidence.md).

## 2. The end-to-end path produces the documented result

The reviewer demo runs the full path — local Kafka, replayed recorded events
through `vehicle.events`, event-time windowing, alert evaluation, output writers
— and prints the result against the numbers this capture is known to produce.

```bash
cp .env.example .env
./scripts/demo.sh
```

Expected:

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

Saved output from a real run is in [`results/`](results/).

The counts are checkable against the input by hand: the committed replay file
holds 2401 events, and `wc -l data/sample/replay_tva43_15min.jsonl` confirms it.
Every event in, every event counted, none dropped as late.

## 3. Reproducibility was verified by destroying the environment

Not by re-running in place. The path has been reproduced from three fresh
clones (two local, one from GitHub `main`), and against a broker destroyed with
`docker compose down -v` and rebuilt from nothing. The first fresh-clone attempt
**failed**, which is the reason the check exists: the documented setup recipe had
never been executed on a machine without a prepared `.venv`, and it was wrong.

The formal version of this — same input, varied conditions, checksummed output
— is the bonus extension in [`extension/`](extension/).

## 4. The bounded AI element, and how far it is verified

The AI element is pretrained YOLO11n detection plus ByteTrack association, with
no training or fine-tuning. Its boundary is narrow by design: sampled frames in,
schema-validated crossing events out. Everything downstream — windowing,
counting, baselines, alerts — is deterministic code covered by the test suite.

**What is verified:** that the events it emits are schema-valid, that they are
keyed and partitioned correctly, that they window and count deterministically,
and that the whole path reproduces exactly from the recorded output.

**What is not verified: detection accuracy against ground truth.** No count MAE,
no precision/recall. This is the honest gap in this project's evaluation and we
would rather state it than let it be discovered.

The reason is specific rather than general: measuring it requires hand-labeled
frames, and labeling requires the source clips (retained in a private bucket)
plus a working perception environment. The perception stack — `torch`,
`opencv-python`, `ultralytics` — has no x86_64 macOS wheels and cannot be
installed on the machine that did this work at all. Labeling ~100 frames was
scoped (TEAM_PLAN 2.9) and not completed before the deadline.

What is known about detector quality is therefore qualitative and is disclosed
in [`../AI_USAGE.md`](../AI_USAGE.md): daylight performance was inspected on
rendered overlays and looked sound; **night scenes fail outright**, with the
large majority of vehicles missed on ~2 AM captures, so no accuracy claim is
made outside daylight hours.

## Limitations, collected

- Detection accuracy is unmeasured (above).
- `volume_spike` and `volume_drop` have never fired on real data — this
  corridor's volume does not move enough at any window size. The demo exercises
  1 of 3 alert rules. Thresholds were deliberately not loosened, and the fault
  cut point deliberately not tuned, to manufacture one.
- `camera_stale` fires only against an injected synthetic camera; a
  single-camera recording cannot exercise the rule.
- The staleness gap the alert reports varies with consumption progress
  (359 s / 840 s observed). See [`extension/README.md`](extension/README.md).
- Volume-table row order is not byte-stable once two cameras are involved;
  counts are exact. Reproducibility is claimed over the sorted table.
- The demo runs 60-second windows, not the 900-second windows the architecture
  describes, because a 15-minute capture yields only two 15-minute windows. The
  code default is unchanged; the artifact is named `volume_demo_60s.csv` so
  nothing is mislabelled.
