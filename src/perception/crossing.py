"""Counting-line crossing detection from tracked vehicle centers.

A CountingLine is a virtual segment in normalized image coordinates. Each track's
center is classified to one side of the line (sign of the cross product); when a
track's side flips between two *confidently* off-line positions, that's one
crossing event in the direction the flip implies. The epsilon hysteresis keeps
box jitter on the line itself from double-counting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.streaming.contracts import TravelDirection

# Minimum |signed area| for a position to count as clearly off the line.
# Normalized coords: 0.004 ≈ a few pixels at 720p — jitter stays below it.
SIDE_EPSILON = 0.004


@dataclass(frozen=True)
class CountingLine:
    """Directed segment p1→p2 in normalized [0,1]² image coordinates.

    A track crossing from the line's negative half-plane to the positive one is
    labeled `positive_direction`, the reverse `negative_direction`.
    """

    name: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    positive_direction: TravelDirection
    negative_direction: TravelDirection

    def side(self, point: tuple[float, float]) -> float:
        """Signed cross product: >0 left of p1→p2, <0 right, ~0 on the line."""
        (x1, y1), (x2, y2) = self.p1, self.p2
        px, py = point
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


@dataclass
class Crossing:
    track_id: int
    line_name: str
    direction: TravelDirection


@dataclass
class LineCrossingCounter:
    """Stateful crossing detector for one camera's set of counting lines."""

    lines: list[CountingLine]
    epsilon: float = SIDE_EPSILON
    # (line_name, track_id) -> last confident side sign (+1.0 / -1.0)
    _last_side: dict[tuple[str, int], float] = field(default_factory=dict)

    def update(self, track_id: int, center: tuple[float, float]) -> list[Crossing]:
        """Feed one tracked position; return crossings it completed (usually 0-1)."""
        crossings = []
        for line in self.lines:
            s = line.side(center)
            if abs(s) < self.epsilon:
                continue  # on/near the line — keep previous confident side
            sign = 1.0 if s > 0 else -1.0
            key = (line.name, track_id)
            prev = self._last_side.get(key)
            if prev is not None and prev != sign:
                direction = line.positive_direction if sign > 0 else line.negative_direction
                crossings.append(Crossing(track_id, line.name, direction))
            self._last_side[key] = sign
        return crossings


def parse_line_spec(spec: str) -> CountingLine:
    """Parse 'x1,y1,x2,y2:POS:NEG' (e.g. '0.05,0.55,0.95,0.55:EB:WB')."""
    coords, pos, neg = spec.split(":")
    x1, y1, x2, y2 = (float(v) for v in coords.split(","))
    return CountingLine(
        name=f"line_{pos}_{neg}",
        p1=(x1, y1),
        p2=(x2, y2),
        positive_direction=TravelDirection(pos),
        negative_direction=TravelDirection(neg),
    )
