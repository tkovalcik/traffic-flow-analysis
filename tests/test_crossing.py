"""Tests for counting-line crossing geometry and hysteresis."""

from src.perception.crossing import CountingLine, LineCrossingCounter, parse_line_spec
from src.streaming.contracts import TravelDirection

# Horizontal line across the frame at y=0.5; downward crossing (top→bottom in
# image coords) lands on the positive side per the cross-product convention.
LINE = CountingLine(
    name="mid",
    p1=(0.0, 0.5),
    p2=(1.0, 0.5),
    positive_direction=TravelDirection.EB,
    negative_direction=TravelDirection.WB,
)


def walk(counter, track_id, ys):
    events = []
    for y in ys:
        events.extend(counter.update(track_id, (0.5, y)))
    return events


def test_downward_crossing_emits_one_positive_event():
    counter = LineCrossingCounter([LINE])
    events = walk(counter, 1, [0.2, 0.4, 0.6, 0.8])
    assert len(events) == 1
    assert events[0].direction == TravelDirection.EB
    assert events[0].track_id == 1


def test_upward_crossing_emits_negative_direction():
    counter = LineCrossingCounter([LINE])
    events = walk(counter, 2, [0.8, 0.6, 0.4, 0.2])
    assert [e.direction for e in events] == [TravelDirection.WB]


def test_jitter_on_the_line_does_not_double_count():
    counter = LineCrossingCounter([LINE])
    # Approach, then jitter within epsilon of the line, then continue through.
    ys = [0.3, 0.499, 0.5, 0.501, 0.4999, 0.502, 0.7]
    events = walk(counter, 3, ys)
    assert len(events) == 1  # one real crossing despite the wobble


def test_no_event_without_confident_positions_on_both_sides():
    counter = LineCrossingCounter([LINE])
    assert walk(counter, 4, [0.3, 0.35, 0.4]) == []  # never crossed


def test_recrossing_counts_again():
    counter = LineCrossingCounter([LINE])
    events = walk(counter, 5, [0.3, 0.7, 0.3])
    assert [e.direction for e in events] == [TravelDirection.EB, TravelDirection.WB]


def test_tracks_are_independent():
    counter = LineCrossingCounter([LINE])
    walk(counter, 6, [0.3])  # track 6 sits above the line
    events = walk(counter, 7, [0.7, 0.3])  # track 7 crosses upward
    assert len(events) == 1
    assert events[0].track_id == 7


def test_parse_line_spec_roundtrip():
    line = parse_line_spec("0.05,0.55,0.95,0.55:EB:WB")
    assert line.p1 == (0.05, 0.55)
    assert line.p2 == (0.95, 0.55)
    assert line.positive_direction == TravelDirection.EB
    assert line.negative_direction == TravelDirection.WB
    assert line.expected_motion is None


def test_parse_line_spec_with_motion():
    line = parse_line_spec("0.65,0.38,0.95,0.64:WB:EB:0.643,-0.766")
    assert line.expected_motion == (0.643, -0.766)


def test_motion_gate_blocks_opposing_flow():
    # Line calibrated for an UP-frame flow (motion 0,-1): a track moving DOWN
    # across it must not fire, but an up-moving track must.
    gated = parse_line_spec("0.0,0.5,1.0,0.5:WB:EB:0.0,-1.0")
    counter = LineCrossingCounter([gated])
    assert walk(counter, 1, [0.3, 0.45, 0.6, 0.8]) == []  # downward: rejected
    events = walk(counter, 2, [0.8, 0.6, 0.4, 0.2])  # upward: counted
    assert [e.direction for e in events] == [TravelDirection.EB]


def test_motion_gate_without_vector_counts_both_ways():
    ungated = parse_line_spec("0.0,0.5,1.0,0.5:WB:EB")
    counter = LineCrossingCounter([ungated])
    assert len(walk(counter, 3, [0.3, 0.7])) == 1
    assert len(walk(counter, 4, [0.7, 0.3])) == 1
