"""The staged-timeline, trail, and crossing-pop logic behind presentation renders."""

import numpy as np
import pytest

from src.perception.render import (
    POP_SECONDS,
    TRAIL_FLOOR,
    StagePlan,
    TrailCanvas,
    parse_stage_spec,
    pop_strength,
    trail_decay,
)


class TestStagePlan:
    def test_raw_stage_shows_video_only(self):
        plan = StagePlan(raw=8, overlay=14, dark=10, fade=1.5)
        assert plan.gains(0.0) == (1.0, 0.0)
        assert plan.gains(6.4) == (1.0, 0.0)  # before the overlay ramp begins

    def test_overlay_stage_shows_both(self):
        plan = StagePlan(raw=8, overlay=14, dark=10, fade=1.5)
        assert plan.gains(8.0) == (1.0, 1.0)  # ramp completes exactly at the boundary
        assert plan.gains(15.0) == (1.0, 1.0)

    def test_dark_stage_keeps_only_the_overlay(self):
        plan = StagePlan(raw=8, overlay=14, dark=10, fade=1.5)
        video_gain, overlay_gain = plan.gains(22.0)
        assert video_gain == 0.0
        assert overlay_gain == 1.0
        assert plan.gains(plan.total) == (0.0, 1.0)

    def test_transitions_ramp_linearly_before_each_boundary(self):
        plan = StagePlan(raw=8, overlay=14, dark=10, fade=2.0)
        _, overlay_mid = plan.gains(7.0)  # halfway up the overlay fade-in
        assert overlay_mid == pytest.approx(0.5)
        video_mid, _ = plan.gains(21.0)  # halfway down the video fade-out
        assert video_mid == pytest.approx(0.5)

    def test_zero_fade_switches_instantly(self):
        plan = StagePlan(raw=5, overlay=5, dark=5, fade=0.0)
        assert plan.gains(4.999) == (1.0, 0.0)
        assert plan.gains(5.0) == (1.0, 1.0)

    def test_total_is_the_render_length(self):
        assert StagePlan(raw=8, overlay=14, dark=10).total == 32


class TestParseStageSpec:
    def test_parses_the_documented_spec(self):
        plan = parse_stage_spec("raw=8,overlay=14,dark=10", fade=1.5)
        assert (plan.raw, plan.overlay, plan.dark, plan.fade) == (8.0, 14.0, 10.0, 1.5)

    def test_rejects_unknown_stage_names(self):
        with pytest.raises(ValueError, match="unknown stage"):
            parse_stage_spec("raw=8,fog=3,dark=10", fade=1.0)

    def test_rejects_missing_stages(self):
        with pytest.raises(ValueError, match="missing"):
            parse_stage_spec("raw=8,overlay=14", fade=1.0)


class TestTrailMath:
    def test_decay_reaches_the_floor_after_trail_seconds(self):
        fps, seconds = 30.0, 2.5
        decay = trail_decay(fps, seconds)
        assert 0.0 < decay < 1.0
        assert decay ** (fps * seconds) == pytest.approx(TRAIL_FLOOR)

    def test_canvas_draws_between_consecutive_centers_and_fades(self):
        from src.perception.detect_track import TrackObservation
        from src.streaming.contracts import VehicleClass

        def obs(track_id, x):
            return TrackObservation(
                frame_index=0,
                track_id=track_id,
                vehicle_class=VehicleClass.car,
                confidence=0.9,
                center=(x, 0.5),
                box=(x - 0.05, 0.4, x + 0.05, 0.6),
            )

        canvas = TrailCanvas((40, 40, 3), decay=0.5)
        canvas.step([obs(1, 0.25)], (40, 40))  # first sighting: nothing to join yet
        assert canvas._canvas.sum() == 0
        canvas.step([obs(1, 0.75)], (40, 40))  # second sighting draws the segment
        lit_after_draw = canvas._canvas.sum()
        assert lit_after_draw > 0
        canvas.step([], (40, 40))  # decay only
        assert canvas._canvas.sum() == pytest.approx(lit_after_draw * 0.5)
        # The vanished track was forgotten: a re-used id must not join old points.
        assert canvas._last_center == {}

    def test_composite_saturates_instead_of_wrapping(self):
        canvas = TrailCanvas((4, 4, 3), decay=1.0)
        canvas._canvas[:] = 200.0
        frame = np.full((4, 4, 3), 200, dtype=np.uint8)
        assert canvas.composite(frame).max() == 255


class TestPopStrength:
    def test_full_at_birth_gone_at_lifetime(self):
        assert pop_strength(0.0) == 1.0
        assert pop_strength(POP_SECONDS) == 0.0
        assert pop_strength(POP_SECONDS * 10) == 0.0

    def test_fades_linearly(self):
        assert pop_strength(POP_SECONDS / 2) == pytest.approx(0.5)
