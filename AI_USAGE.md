# AI Usage

This file satisfies the course AI-disclosure requirement: the bounded AI element in
the product, plus disclosed AI-assisted development.

## 1. Bounded AI element: pretrained vehicle detection + tracking

- **Task AI owns:** per-frame vehicle detection (YOLO11n, pretrained COCO weights,
  pinned version — no training) and multi-object association (ByteTrack). Boundary:
  sampled frames in → schema-validated vehicle crossing events out. Everything
  downstream (windowing, counts, alerts) is deterministic code.
- **Representative input/output:** see `data/sample/` (sample frame + the crossing
  events it produced) — populated from our first capture session.
- **Accepted/rejected:** we filter detections to 4 vehicle classes and a confidence
  threshold; detections outside the counting corridor are ignored. TODO: document
  final thresholds after evaluation.
- **Verification:** ~100 hand-labeled frames; per-frame count MAE and detection
  precision/recall; alert sanity checks on a known-busy segment. Artifacts in
  `evaluation/`.
- **Known limitations:** low-resolution cameras, night/rain degradation, occlusion
  at congested moments, class confusion (car vs truck). Counting-line logic
  mitigates double-counting but depends on tracker id stability.
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
