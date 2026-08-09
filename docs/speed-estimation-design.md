# Design: Vehicle Speed Estimation from Lane-Marking Geometry

Status: **design only** — stretch feature, build after the graded streaming path
is complete. No code exists yet; this documents the architecture so the build
is mechanical when we green-light it.

## Goal

Per-vehicle speed estimates (mph) using only what we already have — tracked
vehicle positions and the camera image — by calibrating a pixel→road-distance
mapping per camera from the standardized lane-line dashes.

## The core idea

A tracked vehicle gives us displacement in *pixels per second*. To convert to
*feet per second* we need the pixel→feet scale — but that scale changes across
the image (perspective: far things are small). Lane-line dashes solve this
elegantly: **every dash is a known real-world length painted directly on the
road surface**, and dashes appear all along the roadway at every depth. Each
detected dash is therefore a local measurement of the pixel-per-foot scale at
its own position, and the full set of dashes constrains a global mapping
between the image and the (flat) road plane.

## Dash geometry — RESOLVED (Tom's research, 2026-08-09)

| Context | Segment | Gap | Cycle |
|---|---|---|---|
| **California, multi-lane highway, design speed > 45 mph** (our cameras) | **12 ft** | **36 ft** | **48 ft** |
| MUTCD national default (Sec. 3A) — fallback for other geographies | 10 ft | 30 ft | 40 ft |

Tom verified the California-specific standard: 12-foot stripes with 36-foot
gaps on multi-lane highways designed for over 45 mph. Both our cameras (I-80,
I-580) qualify, so **48 ft/cycle is our calibration constant**.

Design consequence: dash geometry is a **per-camera configuration**, not a
constant — `dash_ft` / `gap_ft` live in the camera's ground-plane config with
geography-based defaults (CA >45 mph multilane → 12/36; generic MUTCD → 10/30),
so adding an Iowa DOT camera later is a config entry, not a code change.

**No satellite imagery in the calibration path.** The mapping is derived
entirely from the camera image itself: every visible dash (and dash-to-dash
cycle) is a 12 ft (48 ft) ruler *at its own image position*, and those rulers
shrink with distance exactly as perspective dictates — that shrinkage is the
signal the homography fit consumes. Aerial measurement is demoted to an
optional diagnostic, used only if the in-image fit residuals suggest the
striping deviates from standard.

## Architecture (`src/perception/speed/`)

```
clip frames ──► 1. median_frame.py     temporal median of ~500 frames
                    │                  = empty road (moving cars average away)
                    ▼
                2. dash_detect.py      white-line mask (adaptive threshold +
                    │                  morphology) → elongated contours →
                    │                  dash endpoints in pixels
                    ▼
                3. ground_plane.py     fit 3×3 homography H : road plane → image
                    │                  from dash constraints (known segment/cycle
                    │                  lengths along each lane line), least
                    │                  squares + RANSAC; save per-camera
                    │                  configs/ground_plane/<camera>.json
                    ▼
tracks ───────► 4. speed.py            map track centers image→road through H⁻¹,
                    │                  speed = Δroad-distance / Δevent-time,
                    │                  smoothed ~1 s (median filter)
                    ▼
                5. events + QC         optional speed_mph on VehicleEvent;
                                       speed overlay in render.py; histogram
                                       QC script per camera/direction
```

Key design points:

1. **Median frame** (step 1) is the trick that makes dash detection easy: with
   enough frames, moving vehicles vanish and we get clean pavement. Cached per
   camera; recomputed when the camera view changes.
2. **Homography, not per-row scale factors** (step 3): a flat road maps to the
   image by a 3×3 projective transform. Fitting one global H over dozens of
   dash measurements averages out paint wear and detection noise, and gives us
   principled residuals (bad dash detections get RANSAC-rejected). Known
   limitation: our corridors **curve**, and a homography assumes a plane with
   straight-line preservation — it still holds on a plane with curved lanes,
   but lane-arc-length ≠ straight-line distance. Mitigation: compute speed
   from short displacement steps (~1 s apart) where chord ≈ arc, and restrict
   to the near-field zone.
3. **Schema evolution** (step 5): `speed_mph` joins `vehicle.events` as an
   optional nullable field with a default — the textbook backward-compatible
   Avro change (old consumers keep working; a nice live demo of the Schema
   Registry compatibility rules from lecture).

## Error budget — what limits accuracy

- **Box-center jitter** (±a few px/frame) dominates instantaneous speed →
  never report single-frame speeds; median over ~1 s of steps.
- **Parallax**: the box center sits on the vehicle body (~3-5 ft above the
  road), not the road plane, so it maps to a slightly shifted road point. The
  shift is nearly constant over a short step, so it mostly cancels in
  differences — worst near the image edges.
- **Far-field collapse**: pixels-per-foot shrinks with distance; a 1 px error
  at the horizon is many feet. Report speeds only where the local scale
  exceeds a threshold (near-field), and mark the zone on the QC overlay.
- **PTZ drift**: operators re-aim these cameras. Detect via similarity (SSIM)
  between the live median frame and the calibration median; recalibrate when
  it drops. (Same trigger should flag counting-line recalibration.)

## Validation plan (deterministic)

1. Aerial dash measurement vs standards table (above).
2. Manual ground truth: for ~10 vehicles in a clip, time their travel between
   two identified dashes by frame count → hand-computed speed vs pipeline.
3. Sanity distributions: free-flow p50 ≈ 60-70 mph on I-80/I-580, congested
   windows lower; EB/WB distributions should differ during commute peaks.
4. Traffic-engineering deliverable: per-window **p85 speed** (the standard
   design metric) alongside volumes — one more "real product" artifact.

## Effort and sequencing

- Phase A — calibration tooling (median frame, dash detect, homography, QC):
  ~4-8 h, the fiddly part is dash segmentation robustness.
- Phase B — speed attachment + contract evolution + render overlay: ~2-3 h.
- **Recommendation:** build after Aug 12 (nothing in the rubric needs it), or
  as post-course portfolio polish. Prerequisite: the aerial verification.

## Open questions for Tom

1. Dash dimensions: how do your research numbers compare to the table above?
2. Output preference: per-event `speed_mph`, per-window aggregates (mean +
   p85), or both?
3. OK to restrict speeds to the near-field zone where calibration is solid?
