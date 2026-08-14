# Traffic Flow Analysis

Real-time vehicle counting from public highway CCTV. A perception node (YOLO11n +
ByteTrack) watches live traffic camera streams, emits structured vehicle **crossing
events** into Kafka, and a stream processor turns them into the standard **15-minute
volume tables** used in traffic engineering, plus congestion and camera-health alerts
and a dashboard. Video never leaves the perception node — only detection metadata
flows downstream.

Built as the final project for MSDS 682 (Data Streaming) by Tom Kovalcik and Chris
Monzon.

## Architecture

```
Live CCTV streams (2-4 cameras, one corridor)
        │
        ▼
Edge perception node  (Docker; decode → YOLO11n → ByteTrack → counting-line
        │              crossing events; video stays on the node)
        ▼   Avro, Schema Registry
Kafka topic: vehicle.events  (key = camera_id)
        │
        ▼
Stream processor  (event-time 15-min tumbling windows; counts by
        │          camera/direction/class; EWMA baseline; health checks)
        ▼
Kafka topic: traffic.alerts  +  volume tables (CSV)  +  alerts (JSONL)
        │
        ▼
Dashboard (FastAPI)          Evaluation (labeled frames → MAE, precision/recall)
```

Two ways to run it:

- **Replay path (reviewer demo):** one command starts local Kafka in Docker and
  replays recorded detection events through the full pipeline — deterministic,
  no cloud, no GPU, no camera access needed:

  ```
  cp .env.example .env      # local defaults work as-is for this demo
  ./scripts/demo.sh
  ```

  It brings up the broker, recreates both topics with 3 partitions, replays the
  recorded 15-minute capture at 60x, and processes it into 60-second event-time
  windows, writing `outputs/volume_demo_60s.csv` and `outputs/alerts.jsonl`.
  Expect 2401 events on camera `tva43` across 32 windows. Counts are
  reproducible across runs and pacings — the printed checksum is over the sorted
  table, since two cameras close windows on independent watermarks and their
  rows interleave by arrival order. Useful flags: `--speed 0` (replay as fast as
  possible), `--no-fault`, `--window-seconds N`.

  The demo also injects a second, clearly synthetic camera (`tva43_mirror`, a
  copy of the capture under a different id) that goes silent five minutes in, so
  the `camera_stale` rule has something to fire on and alerts actually flow
  through `traffic.alerts`. A single-camera recording cannot exercise that rule:
  a lone camera's stream time is its own last event, so its silence gap is
  always zero. Pass `--no-fault` for a run over the recorded data alone, which
  raises no alerts — this corridor's volume never moves enough to trip the
  spike/drop thresholds.
- **Live path:** the perception container runs on a GPU/CPU VM against live camera
  streams and produces to Confluent Cloud.

## Data source

We use publicly viewable state-DOT traffic cameras (our corridor comes from the
Caltrans District 4 public camera map). Camera stream endpoints are **not** hardcoded
in this repo — they are supplied via `.env` (see `.env.example`). If you want to run
the live path yourself, pick cameras from your state DOT's public camera site (e.g.
Caltrans, Iowa DOT 511) and set the URLs in your `.env`. Details, access rules, and
schema: [DATA_SOURCE.md](DATA_SOURCE.md).

## Repository layout

```
src/perception/   capture, detection, tracking, crossing events, Kafka producer
src/streaming/    event contracts (Avro/Pydantic), window processor, alerts
src/replay/       session recorder + deterministic replay producer
src/dashboard/    FastAPI dashboard
src/triage/       camera-inventory scanner (which public cams actually work)
docker/           local Kafka compose
scripts/          capture-session & cloud VM helpers
data/sample/      small committed sample/replay data
outputs/          generated volume tables & alerts (gitignored except samples)
evaluation/       labeled frames, metrics scripts, latency reports
tests/            pytest suite (crossing logic, windowing, contracts)
```

`TEAM_PLAN.md` tracks who is building what. `DATA_SOURCE.md` and `AI_USAGE.md` are
course-required documentation.

## Setup

For the reviewer demo you only need the streaming path — three packages, no
perception stack:

```bash
uv venv --python 3.11
uv pip install "confluent-kafka[avro,schemaregistry]" pydantic python-dotenv
cp .env.example .env    # local Kafka defaults work as-is
```

On an Intel Mac add `"cryptography==46.0.3"` to that install: `confluent-kafka`
pulls `cryptography` in through `authlib`, and the pinned version builds from
source there rather than installing a wheel.

The full environment, needed only for perception (capture, YOLO, tracking):

```bash
uv sync                 # or: pip install -r requirements.txt  (Python 3.11)
cp .env.example .env    # fill in your values
```

`uv sync` resolves `torch`, `opencv-python` and `ultralytics`, which have no
x86_64 macOS wheels — it cannot complete on an Intel Mac. The reviewer demo
above does not need any of them.

## Cleanup

Everything the demo creates is local:

```bash
docker compose -f docker/compose.local-kafka.yml down -v   # stop broker + Schema Registry, delete volumes
rm -rf outputs/volume_demo_60s.csv outputs/alerts.jsonl   # generated demo outputs (optional)
```

`down -v` removes the Kafka data volumes, so the next run starts from a cold
broker (expect the one documented coordinator warning again). No cloud
resources are part of the review path.
