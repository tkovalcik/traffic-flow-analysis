# AI Usage

This file satisfies the course AI-disclosure requirement: the bounded AI element in
the product, plus disclosed AI-assisted development.

## 1. Bounded AI element: pretrained vehicle detection + tracking

- **Task AI owns:** per-frame vehicle detection (YOLO11n, pretrained COCO weights,
  pinned version — no training) and multi-object association (ByteTrack). Boundary:
  sampled frames in → schema-validated vehicle crossing events out. Everything
  downstream (windowing, counts, alerts) is deterministic code.
- **Representative input/output:** see `data/sample/` — `vehicle_events_sample.jsonl`
  (a short excerpt for reading) and `replay_tva43_15min.jsonl` (2401 crossing events,
  the full 15-minute capture that drives the reviewer demo), both produced by this
  detector on our 2026-08-09 session. Source video is not committed: clips are
  retained privately and only detection metadata is published.
- **Accepted/rejected:** we accept only COCO classes 2/3/5/7 (car, motorcycle, bus,
  truck) at confidence ≥ 0.35 (`CONFIDENCE_THRESHOLD`, model `yolo11n.pt`), and a
  detection must cross a calibrated counting line to become an event. Crossings are
  additionally motion-gated: each line carries a calibrated flow vector and a track
  moving against it cannot fire that line. This was not a cosmetic filter —
  ungated per-flow lines overcounted roughly 2× on `tva43` because the opposite
  flow's vehicles crossed the line in the far field. Thresholds were fixed at
  calibration time (2026-08-09) and not retuned afterwards.
- **Verification:** what we verified is that emitted events are schema-valid,
  correctly keyed and partitioned, and that they window and count
  deterministically — 177 automated tests in CI plus the end-to-end replay
  reproduction; artifacts in `evaluation/`. Detection accuracy against ground
  truth (hand-labeled frames, per-frame count MAE, detection precision/recall)
  was scoped but **not completed within this project** and is documented as
  future work; we state that boundary explicitly rather than let it be
  discovered. See `report.pdf` §4.2 for the same statement and the reason
  (labeling requires the retained source clips plus a working perception
  environment).
- **Known limitations:** low-resolution cameras, occlusion at congested moments,
  class confusion (car vs truck). Counting-line logic mitigates double-counting
  but depends on tracker id stability. **Night scenes fail outright**: on ~2 AM
  captures (2026-08-09) the pretrained detector missed the large majority of
  passing vehicles (headlight glare, low contrast), so count accuracy is
  claimed for daylight hours only. A no-training mitigation path is documented
  as future work (lower confidence threshold, larger inference resolution,
  larger pretrained variant, contrast preprocessing — TEAM_PLAN 3.13); model
  fine-tuning on night data is explicitly out of scope for this project.
- **Fallback:** recorded detection JSONL replay drives the identical downstream
  pipeline (also the graded review path), so the system is fully demonstrable
  without live inference.

## 2. AI-assisted development (disclosure)

- **Tool & task:** Claude Code was used as a development assistant for scaffolding,
  boilerplate, and implementation of non-core components, under human direction.
  The core Kafka path (producer setup, consumer loop, window logic) was
  hand-written by the team; architecture, scope, and all design decisions are the
  team's.
- **Verification of AI-assisted code:** code review by the team, pytest suite
  (crossing logic, window semantics incl. late events, schema validation), and the
  deterministic replay check that reproduces expected outputs end-to-end.
- **Ownership:** both team members can explain every component; work assignments
  are tracked in `TEAM_PLAN.md`.
