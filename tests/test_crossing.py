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
