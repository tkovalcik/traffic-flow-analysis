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

    `expected_motion` (optional unit-ish vector) restricts the line to the flow
    it was calibrated for: crossings by tracks moving against it are ignored.
    Per-flow lines need this — the far-away opposite flow can geometrically
    cross a line meant for the near flow and would otherwise miscount.
    """

    name: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    positive_direction: TravelDirection
    negative_direction: TravelDirection
    expected_motion: tuple[float, float] | None = None

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
    # track_id -> previous position, for motion-gating
    _last_pos: dict[int, tuple[float, float]] = field(default_factory=dict)

    def update(self, track_id: int, center: tuple[float, float]) -> list[Crossing]:
        """Feed one tracked position; return crossings it completed (usually 0-1)."""
        prev_pos = self._last_pos.get(track_id)
        self._last_pos[track_id] = center
        crossings = []
        for line in self.lines:
            s = line.side(center)
            if abs(s) < self.epsilon:
                continue  # on/near the line — keep previous confident side
            sign = 1.0 if s > 0 else -1.0
            key = (line.name, track_id)
            prev = self._last_side.get(key)
            if prev is not None and prev != sign and self._motion_ok(line, prev_pos, center):
                direction = line.positive_direction if sign > 0 else line.negative_direction
                crossings.append(Crossing(track_id, line.name, direction))
            self._last_side[key] = sign
        return crossings

    @staticmethod
    def _motion_ok(
        line: CountingLine,
        prev_pos: tuple[float, float] | None,
        center: tuple[float, float],
    ) -> bool:
        """Motion gate: the step must not oppose the line's calibrated flow."""
        if line.expected_motion is None or prev_pos is None:
            return True
        step = (center[0] - prev_pos[0], center[1] - prev_pos[1])
        mx, my = line.expected_motion
        return step[0] * mx + step[1] * my > 0


def parse_line_spec(spec: str) -> CountingLine:
    """Parse 'x1,y1,x2,y2:POS:NEG[:mx,my]'.

    The optional trailing mx,my is the expected flow motion for motion-gating
    (e.g. '0.65,0.38,0.95,0.64:WB:EB:0.64,-0.77').
    """
    parts = spec.split(":")
    coords, pos, neg = parts[0], parts[1], parts[2]
    x1, y1, x2, y2 = (float(v) for v in coords.split(","))
    motion = None
    if len(parts) > 3:
        mx, my = (float(v) for v in parts[3].split(","))
        motion = (mx, my)
    return CountingLine(
        name=f"line_{pos}_{neg}",
        p1=(x1, y1),
        p2=(x2, y2),
        positive_direction=TravelDirection(pos),
        negative_direction=TravelDirection(neg),
        expected_motion=motion,
    )
