"""Tests for motion-based counting-line proposal (synthetic tracks, no YOLO)."""

import math
from collections import namedtuple

import pytest

from src.perception.calibrate_line import (
    collect_paths,
    propose_line,
    split_flows,
)
from src.perception.crossing import LineCrossingCounter, parse_line_spec
from src.streaming.contracts import TravelDirection

Obs = namedtuple("Obs", ["track_id", "center"])


def flow_observations(track_ids, x_band, y_path, jitter=0.0):
    """Tracks moving along y_path (a list of y positions), spread across x_band."""
    observations = []
    for i, tid in enumerate(track_ids):
        x = x_band[0] + (x_band[1] - x_band[0]) * (i / max(len(track_ids) - 1, 1))
        for j, y in enumerate(y_path):
            observations.append(Obs(tid, (x + jitter * (j % 2), y)))
    return observations


# EB flow: right band, moving UP the frame; WB flow: left band, moving DOWN.
EB_UP = flow_observations([1, 2, 3, 4], (0.55, 0.9), [0.9, 0.75, 0.6, 0.45, 0.3])
WB_DOWN = flow_observations([10, 11, 12, 13], (0.1, 0.45), [0.3, 0.45, 0.6, 0.75, 0.9])


def test_collect_paths_filters_short_and_static_tracks():
    static = [Obs(99, (0.5, 0.5))] * 8  # 8 points, zero displacement
    short = [Obs(98, (0.1, 0.1)), Obs(98, (0.9, 0.9))]  # huge motion, 2 points
    paths = collect_paths(EB_UP + static + short)
    assert sorted(p.track_id for p in paths) == [1, 2, 3, 4]


def test_split_flows_separates_opposing_directions():
    a, b = split_flows(collect_paths(EB_UP + WB_DOWN))
    ids_a = frozenset(p.track_id for p in a)
    ids_b = frozenset(p.track_id for p in b)
    assert {ids_a, ids_b} == {frozenset({1, 2, 3, 4}), frozenset({10, 11, 12, 13})}


@pytest.mark.parametrize(
    ("observations", "expected"),
    [(EB_UP, TravelDirection.EB), (WB_DOWN, TravelDirection.WB)],
)
def test_proposed_line_emits_correct_compass_direction(observations, expected):
    """End-to-end: tracks crossing their own proposed line get the right label."""
    (flow,) = [f for f in split_flows(collect_paths(observations)) if f]
    proposal = propose_line(flow, up_frame=TravelDirection.EB)
    line = parse_line_spec(proposal.spec)

    counter = LineCrossingCounter([line])
    directions = []
    for path in flow:
        for point in path.points:
            directions.extend(c.direction for c in counter.update(path.track_id, point))
    assert directions, "every track should cross its own flow's line"
    assert set(directions) == {expected}


def test_line_is_perpendicular_to_motion_and_spans_band():
    (flow,) = [f for f in split_flows(collect_paths(EB_UP)) if f]
    proposal = propose_line(flow, up_frame=TravelDirection.EB)
    line = parse_line_spec(proposal.spec)
    (x1, y1), (x2, y2) = line.p1, line.p2

    # Vertical motion → near-horizontal line (|dot(line_dir, motion)| ~ 0).
    ldx, ldy = x2 - x1, y2 - y1
    llen = math.hypot(ldx, ldy)
    assert abs(ldy / llen) < 0.05

    xs = sorted([x1, x2])
    assert xs[0] <= 0.55 and xs[1] >= 0.9  # covers the whole EB band + margin


def test_diagonal_flow_gets_diagonal_line():
    diagonal = []
    for tid, x0 in enumerate((0.2, 0.3, 0.4, 0.5)):
        # moving down-right at 45 degrees
        diagonal.extend(Obs(tid, (x0 + 0.08 * j, 0.3 + 0.08 * j)) for j in range(6))
    (flow,) = [f for f in split_flows(collect_paths(diagonal)) if f]
    proposal = propose_line(flow, up_frame=TravelDirection.EB)
    line = parse_line_spec(proposal.spec)
    ldx = line.p2[0] - line.p1[0]
    ldy = line.p2[1] - line.p1[1]
    # Perpendicular to (1,1)/sqrt2 → direction ~ (-1,1)/sqrt2: slope ≈ -1.
    assert ldy / ldx == pytest.approx(-1.0, abs=0.15)
    assert proposal.mean_motion_deg == pytest.approx(45.0, abs=3.0)
