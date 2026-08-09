# Team Plan — Traffic Flow Analysis (MSDS 682 Final Project)

**How this works:** pick any `open` task, put your initials in Owner and set Status to
`WIP`. When finished, set `done`. Add tasks freely. Keep it honest — this doubles as
our contribution record for the rubric.

**Deadlines:** presentation **Thu Aug 13, 5:30 PM** · report + code ZIP **Fri Aug 14,
11:59 PM** (late = −10%/day, zero at day 3, so effectively done by Aug 12).

**Grading (20 pts + 3 bonus):** problem & observable result · data doc & event
contract · architecture & working Kafka path · evidence & reproducibility. The graded
review path is: clone repo → one command → local Kafka in Docker → replay recorded
detections → 15-min volume tables + alerts. Everything else is demo/portfolio layer.

Statuses: `open` → `WIP` → `done` (or `skip` with a note).

## Phase 0 — Scaffold & unblock (Sat Aug 9)

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 0.1 | Repo scaffold: gitignore, skeleton, pyproject, submission layout | TK | done | |
| 0.2 | Course docs downloaded for reference | TK | done | internal-docs (local only) |
| 0.3 | Camera triage script: scan Caltrans D4 inventory, test streams, thumbnails, CSV report | TK | done | stream URLs stay out of git |
| 0.4 | Run D4 scan, shortlist 2-4 cameras on one corridor | TK | done | 97/194 streams usable; top corridor: I-80 Emeryville/Berkeley (2×1080p) — confirm final picks together |
| 0.5 | GCP: project set up, Chris on IAM (editor), billing linked, video bucket created | TK | done | spot T4 usable now (us-west1 quota=1); on-demand needs GPUS_ALL_REGIONS request |
| 0.9 | CONTRIBUTING guide, GitHub Actions CI (ruff+pytest), branch protection on main | TK | done | see CONTRIBUTING.md |
| 0.10 | GCS bucket for video-clip dataset (us-west1) | TK | done | name in .env (GCS_BUCKET) |
| 0.11 | 15-min capture continuity research | TK | done | VERDICT: fully feasible (0 reconnects over 15 min); evidence in DATA_SOURCE.md + internal-docs/research/ |
| 0.12 | Scheduled collection: tfa-capture VM records tva43 15 min 6×/day → bucket | TK | done | `scripts/gcp/provision_capture_vm.sh`; ~$13/mo VM + $0.72/mo storage |
| 0.13 | Direction mapping (EB/WB) verified from daylight frames, both cameras | TK | done | evidence: outputs/review/direction-verification/ |
| 0.6 | Confluent Cloud: cluster + Schema Registry + API keys | TK | hold | credits ran out; Tom creating fresh account per instructor guidance — local Kafka covers the graded path meanwhile |
| 0.7 | Add Chris as GitHub collaborator | TK | done | |
| 0.8 | Verify + document Caltrans usage terms in DATA_SOURCE.md | TK | done | Conditions of Use cited; content generally public domain, no attribution requirement stated |

## Phase 1 — Perception + producer + first golden data (Sun Aug 10)

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 1.1 | Camera capture module: HLS decode, frame sampling, reconnect handling | TK | done | FrameSource + camera registry, tested |
| 1.2 | YOLO11n + ByteTrack integration, class filter (car/truck/bus/motorcycle) | TK | done | detect_track.py; --stats mode for calibration |
| 1.3 | Counting-line config + crossing-event logic (per camera, direction) | TK | WIP | crossing logic done+tested; per-camera line calibration pending (use --stats + thumbnails) |
| 1.4 | Avro schema for `vehicle.events` + Pydantic model + sample event doc | TK | done | the event contract — graded |
| 1.5 | Kafka producer (key=camera_id, Schema Registry, idempotent) | TK | open | Tom hand-writes (learning) |
| 1.6 | Local Kafka docker compose (broker + Schema Registry), smoke test | TK | done | verified: broker healthy, SR responds |
| 1.7 | Recorder: frames + detection JSONL to disk during capture | TK | done | record.py (clips+metadata) + detect_track --out (events JSONL) |
| 1.8 | **Golden capture session #1, afternoon commute (~3 PM)** | | open | one command ready: `uv run python -m src.replay.session --minutes 15 --upload` |
| 1.9 | Clip uploader: capture sessions push video clips to the GCS bucket (dataset building) | TK | done | record.py --upload; first clips archived 2026-08-09 |

## Phase 2 — The graded path end-to-end (Mon Aug 11)

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 2.1 | Consumer loop + event-time 15-min tumbling windows (+1-min demo windows) | TK | open | Tom hand-writes core (learning) |
| 2.2 | EWMA baseline + congestion alerts → `traffic.alerts` | TK | WIP | rules+contract done+tested (src/streaming/{baseline,alerts}.py); wiring into consumer awaits 2.1 |
| 2.3 | Camera-health staleness alert (no events N min) | TK | WIP | rule done+tested; wiring awaits 2.1 |
| 2.4 | Volume table CSV + alerts JSONL writers | TK | done | src/streaming/outputs.py, tested |
| 2.5 | Replay producer: recorded JSONL → Kafka with original timestamps | | open | |
| 2.6 | **One-command reviewer demo** (compose up + replay + processor + expected output) | | open | THE graded artifact |
| 2.7 | Verify reviewer path from a fresh clone in a temp dir | | open | |
| 2.8 | pytest: crossing logic, window semantics (incl. late events), schema validation | | open | |
| 2.9 | Eval labeling: split ~100 frames (50/50 TK/CM) | | open | |
| 2.10 | Deploy perception to GCP VM, scheduled 3:00-7:30 PM captures | | open | |

## Phase 3 — Live layer, evidence, report (Tue Aug 12 — target: everything done)

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 3.1 | Confluent Cloud live path (same code, env config swap) | | open | |
| 3.2 | Dashboard: 15-min counts table + alert feed + camera health (FastAPI) | TK | done | minimal version reads output files (`uvicorn src.dashboard.app:app`); corridor strip stays 3.9 |
| 3.3 | Latency instrumentation: capture→publish→alert, p50/p95 report script | | open | eval evidence + resume |
| 3.4 | Eval metrics script: count MAE, precision/recall vs labels | | open | |
| 3.5 | Demo videos rendered offline: annotated vs detections-only side-by-side | | open | presentation punchline |
| 3.6 | Report draft (structure per rubric) | | open | |
| 3.7 | Pick + label the +3 bonus extension with repro steps | | open | candidate: benchmark table or replay determinism demo |
| 3.8 | Contribution documentation (this file → report section) | | open | |
| 3.9 | *Nice-to-have:* corridor status strip on dashboard | | open | only if ahead |
| 3.11 | *Nice-to-have:* enrich clip metadata with Caltrans RWIS road-weather feed (nearest station per camera) | | open | weather on camera pages is burned into video, not structured; RWIS is the structured source |
| 3.10 | *Nice-to-have:* TensorRT/ONNX benchmark table on T4 | | open | time-boxed; only if graded items done |

## Phase 4 — Present & submit (Wed Aug 13 – Thu Aug 14)

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 4.1 | Presentation slides + demo dry-run (replay-driven, sped-up event-time demo) | | open | check for instructor's presentation handout — unpublished as of Aug 9 |
| 4.2 | Both partners can explain every component (walkthrough session) | | open | rubric requirement |
| 4.3 | **Present — Thu Aug 13, 5:30 PM** | | open | |
| 4.4 | report.pdf final, ZIP per required structure, submit on Canvas | | open | both upload if not linked as group |
| 4.5 | Cloud cleanup: stop VM, document teardown | | open | |
