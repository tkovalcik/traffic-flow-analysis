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
  no cloud, no GPU, no camera access needed. *(Run instructions land here as soon
  as the pipeline is merged — see TEAM_PLAN.md for status.)*
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
docker/           local Kafka compose, perception Dockerfile
scripts/          capture-session & cloud VM helpers
data/sample/      small committed sample/replay data
outputs/          generated volume tables & alerts (gitignored except samples)
evaluation/       labeled frames, metrics scripts, latency reports
tests/            pytest suite (crossing logic, windowing, contracts)
```

`TEAM_PLAN.md` tracks who is building what. `DATA_SOURCE.md` and `AI_USAGE.md` are
course-required documentation.

## Setup

```bash
uv sync                 # or: pip install -r requirements.txt  (Python 3.11)
cp .env.example .env    # fill in your values
```
