# Data Source

## Source

- **What:** Live public highway CCTV streams (California DOT District 4; public
  camera map at the Caltrans "live traffic cameras" site). Fallback sources: other
  state DOT public camera programs (e.g. Iowa 511).
- **Owner:** California Department of Transportation (Caltrans).
- **Access:** Public web streams, no API key. Per-camera stream endpoint URLs are
  configured via `.env` and intentionally not committed to this repository.
- **Classification:** Realtime (frames sampled continuously from live streams).
  The replay path re-enters recorded events record-by-record (deterministic).

## Usage terms & limitations

- **Terms:** Caltrans' Conditions of Use (dot.ca.gov "Conditions of Use" page,
  accessed 2026-08-09) state that website content is generally public domain and
  "may be distributed or copied as permitted by law," while individually
  copyrighted items (e.g., photographs) "may require additional permissions."
  No camera-feed-specific restrictions or attribution requirements are stated.
- **Our use:** educational/non-commercial coursework. Live video is processed in
  memory and discarded; we do not rebroadcast streams. A small number of short
  clips are retained in a private storage bucket solely for evaluation and
  reproducibility of this project. Published artifacts contain detection
  metadata (counts, classes, timestamps), not video.
- Streams are best-effort public infrastructure: cameras go down, change URLs, or
  serve stale images. The pipeline treats a silent camera as a health-alert
  condition, not an error.
- No PII is extracted or stored: video is processed on the perception node and
  discarded; only vehicle detection metadata (class, track id, direction,
  timestamps) leaves the node. Faces/plates are not detectable at these
  resolutions.

## Event schema (contract)

The canonical contract is the Avro schema in `src/streaming/schemas/` (registered
in Schema Registry) with a mirrored Pydantic model for validation. Summary of
`vehicle.events` (key = `camera_id`):

| Field | Type | Notes |
|---|---|---|
| `event_id` | string (uuid) | unique per crossing event |
| `camera_id` | string | partition key; stable per camera |
| `ts_event` | timestamp-millis | frame timestamp (event time) |
| `ts_publish` | timestamp-millis | producer wall clock (latency measurement) |
| `track_id` | long | ByteTrack persistent id |
| `vehicle_class` | enum | car / truck / bus / motorcycle |
| `direction` | enum | which counting line + travel direction |
| `confidence` | float | detector confidence at crossing |

(Representative sample events: `data/sample/`.)

## Rate limits & replay

- No documented rate limits; we sample frames at the stream's native rate on 2-4
  cameras only, comparable to a person watching the public page.
- **Replay:** recorded detection JSONL from capture sessions is committed as
  sample data and drives the deterministic reviewer demo (see README).
