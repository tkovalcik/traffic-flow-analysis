# Test suite evidence

The automated suite is the primary validation of the logic behind the graded
Kafka path. It runs on every pull request and every push to `main`
(`.github/workflows/ci.yml`: `ruff check` → `ruff format --check` → `pytest`).
Red CI never merges.

## Current state

| Where | Collected | Result |
|---|---|---|
| GitHub Actions (`main`, run 31732975689) | **177** | all passed, 1 warning |
| This machine (Intel Mac) | **118** | all passed |
| Difference | 59 | not collectable locally — see below |

## Why the local number is lower

Nine test modules import `numpy` / `cv2` and cover the perception stage:

```
tests/test_cameras.py        tests/test_render.py
tests/test_capture.py        tests/test_render_stages.py
tests/test_lane_map.py       tests/test_scene_mask.py
tests/test_record.py         tests/test_session.py
tests/test_speed_tools.py
```

`torch`, `opencv-python` and `ultralytics` publish no x86_64 macOS wheels, so
`uv sync` cannot complete on this machine and those modules fail collection.
CI runs on `ubuntu-latest`, where the full 177 collect and pass.

This is itself a reproducibility finding worth recording: the environment that
runs the graded review path is deliberately *smaller* than the one that runs
the full suite. The reviewer demo needs three packages
(`confluent-kafka`, `pydantic`, `python-dotenv`) and none of the perception
stack, which is why it runs on hardware that cannot build the project's own
dependency set.

## Reproducing each number

Full suite, as CI runs it:

```bash
uv sync --frozen
uv run ruff check . && uv run ruff format --check .
uv run pytest -q                                        # 177 passed
```

Streaming-only subset, on a machine without the perception stack:

```bash
.venv/bin/python -m pytest -q \
  --ignore=tests/test_cameras.py --ignore=tests/test_capture.py \
  --ignore=tests/test_lane_map.py --ignore=tests/test_record.py \
  --ignore=tests/test_render.py --ignore=tests/test_render_stages.py \
  --ignore=tests/test_scene_mask.py --ignore=tests/test_speed_tools.py \
  --ignore=tests/test_session.py                        # 118 passed
```

## What the suite covers

Concentrated on the parts where correctness is not obvious by inspection:

| Area | Module | Covers |
|---|---|---|
| Window semantics | `tests/test_windows.py` | tumbling-window assignment, per-camera watermarks, late-event handling, drop-and-tally |
| Consumer loop | `tests/test_consumer.py` | poll loop, offset handling, window close on watermark, idle timeout |
| Replay producer | `tests/test_replay_producer.py` | pacing, `--speed`, late-event injection, mirror camera, `--drop-after` |
| Event contract | `tests/test_contracts.py` | Avro/Pydantic agreement, required fields, enum validation, rejection of malformed events |
| Producer | `tests/test_producer.py` | keying by `camera_id`, idempotent config, local vs Confluent selection |
| Alert rules | `tests/test_alerts.py`, `tests/test_baseline.py` | EWMA baseline, spike/drop thresholds, `camera_stale` gap logic |
| Crossing logic | `tests/test_crossing.py` | counting-line crossing, direction assignment, double-count suppression |
| Output writers | `tests/test_outputs.py` | volume CSV shape, alerts JSONL shape |

## Gap

The suite validates logic, not detection accuracy. No test asserts that the
detector counted the vehicles that were actually present — that would require
labeled ground truth, which does not exist for this project. See
[`README.md`](README.md).
