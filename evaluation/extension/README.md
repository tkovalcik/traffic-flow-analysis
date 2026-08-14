# Bonus extension — controlled reproducibility comparison

**This directory is the optional +3 extension. It is additional to the base
validation in [`../README.md`](../README.md), not a substitute for it.**

A controlled comparison in the sense the rubric describes: one input, held
fixed; several conditions, varied one at a time; a single named metric compared
across all of them.

| | |
|---|---|
| **Input (fixed)** | `data/sample/replay_tva43_15min.jsonl` — 2401 recorded crossing events, camera `tva43`, 2026-08-09 18:46:30Z–19:01:30Z |
| **Conditions (varied)** | arrival pacing (`--speed 0` vs `--speed 60`); broker state (warm vs destroyed and rebuilt); fault injection present vs absent |
| **Metric** | SHA-1 of the sorted volume table, plus per-camera counts, row count, and alert count |
| **Implementation** | [`validate.sh`](validate.sh) — 16 assertions, exits non-zero on any failure |
| **Saved output** | [`results/`](results/) — full transcripts, volume tables, alert logs, summaries |

## What it is testing

The pipeline windows on **event time**, so the wall-clock rate at which events
reach the broker must not change the result. That is a claim, and a claim is
worth an experiment. The harness replays the identical file as fast as the
broker will accept it and again paced at 60× event time, then compares.

The third run removes the injected fault. It is the control: it shows that the
alert in the other two runs is caused by the injected silence and not by the
recorded data, and that the real camera's counts are identical whether the
synthetic camera is present or not.

## Exact steps to run it

Requires Docker running. No GPU, no cloud, no camera access. Uses the same
three-package streaming environment as the reviewer demo — see the repo README
if `.venv` is not already present.

```bash
./evaluation/extension/validate.sh                  # warm broker,  ~2 min
./evaluation/extension/validate.sh --include-cold   # rebuilds broker, ~4 min
```

Each writes to `results/warm-broker/` or `results/cold-broker/` respectively, so
the two conditions do not overwrite each other.

## Expected output

```
Controlled comparison — identical input, varied arrival pacing:
  PASS  speed 0 checksum                       1bd6cba1c010d82951c9c340428dab30dbcc40ff
  PASS  speed 60 checksum                      1bd6cba1c010d82951c9c340428dab30dbcc40ff
  PASS  speed 0 == speed 60                    1bd6cba1c010d82951c9c340428dab30dbcc40ff

Counts unchanged across pacing:
  PASS  speed 0 rows                           62
  PASS  speed 60 rows                          62
  PASS  speed 0 tva43 count                    2401
  PASS  speed 60 tva43 count                   2401
  PASS  speed 0 mirror count                   794
  PASS  speed 60 mirror count                  794

Fault detection — the injected silence is what raises the alert:
  PASS  alerts with fault                      1
  PASS  alerts without fault                   0
  PASS  speed 0 alert identity                 camera_stale:tva43_mirror
  PASS  speed 60 alert identity                camera_stale:tva43_mirror
  note: the alert's reported gap duration is deliberately not asserted
        (359s here, 840s cold at --speed 0) — see README 'Limitations'.

Control run — removing the fault leaves the real camera untouched:
  PASS  no-fault checksum                      c08531859140dc98ae60a660e96b93bc241fe31d
  PASS  no-fault rows                          46
  PASS  no-fault tva43 count                   2401

All 16 checks passed.
```

## Results

Both checksums reproduced exactly under every condition tested:

| Condition | Fault-run checksum | Control checksum |
|---|---|---|
| Warm broker, `--speed 0` | `1bd6cba1…` | — |
| Warm broker, `--speed 60` | `1bd6cba1…` | — |
| Warm broker, `--no-fault` | — | `c0853185…` |
| Cold broker (`compose down -v`, rebuilt) | `1bd6cba1…` | `c0853185…` |

Previously verified by hand under two further conditions not re-run by this
harness: three fresh clones (two local, one from GitHub `main`).

The fault run also demonstrates that `key=camera_id` keeps each camera's events
on a single partition — which a single-camera recording could never show:

```
vehicle.events:0:2401     <- tva43
vehicle.events:1:0        <- empty
vehicle.events:2:794      <- tva43_mirror
```

And the alert that fires:

```
[camera_stale] tva43_mirror: no vehicle events for 359s
```

The alert's **type and camera are reproducible; its reported duration is not.**
Measured across the four fault runs above: 359 s in three of them, 840 s on a
cold broker at `--speed 0`. This is a real property of the rule, found by
running the harness rather than by reading the code, and it is disclosed rather
than smoothed over — see Limitations.

## Why the checksum is over the *sorted* table

Counts are exact and reproducible. Row *order* is not, once two cameras are
involved: each camera closes its windows on its own watermark, so their rows
interleave according to arrival order, which pacing does change. Checksumming
the sorted table isolates the claim we actually make — that the same input
produces the same counts — from an ordering property we never claimed and do
not need. This was found by testing, not by reasoning: the determinism claim in
`windows.py` was written for one camera and held only there.

## Limitations

State these rather than let them be discovered:

- **This validates the pipeline against known-good baselines, not against
  ground truth.** It shows the system computes the same answer every time from
  the same input. It says nothing about whether the detector counted the real
  vehicles correctly — see [`../README.md`](../README.md) for that gap.
- **`camera_stale` fires only against an injected synthetic camera.** A
  single-camera recording cannot exercise the rule at all: a lone camera's
  stream time is its own last event, so its silence gap is always zero.
- **The staleness gap the alert reports is not reproducible.** `_staleness_alerts`
  computes it as `windows.stream_time` — the maximum event time seen across
  *all* cameras — minus the last event seen for the silent camera, both as of
  whatever the consumer has consumed at that moment. Cross-partition read skew
  therefore moves the number: when a cold broker at `--speed 0` let partition 0
  (`tva43`) run far ahead of partition 2 (`tva43_mirror`), the check fired late
  and reported 840 s instead of 359 s. The alert is correct in both cases — the
  camera genuinely was silent — but the duration is a measurement of consumption
  progress as much as of the fault, and should not be read as a precise outage
  length. The harness asserts the alert's type and camera and deliberately does
  not assert the duration.
- **`alert_id` is a fresh UUID per run**, so `alerts.jsonl` is never
  byte-identical between runs even when every other field matches. Only the
  volume table is checksummed for that reason.
- **This is failure *detection*, not failure recovery.** The silenced camera
  stays silent to end of file. Nothing reconnects or heals.
- **`volume_spike` and `volume_drop` have never fired on real data.** This
  corridor's volume does not move enough at any window size. The demo therefore
  exercises 1 of 3 alert rules. The cut point was deliberately *not* tuned to
  manufacture a second alert.
- **Replay pacing drifts ~19%** — `--speed 60` on a 15.0-minute capture takes
  ~17.5 s rather than 15 s, because per-event publish cost accumulates against
  a sleep-per-gap loop. It does not affect any result here, since windows close
  on event time.
