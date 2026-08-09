"""Tests for EWMA baseline math."""

import pytest

from src.streaming.baseline import EwmaBaseline


def test_first_observation_seeds_baseline():
    b = EwmaBaseline(alpha=0.3)
    before = b.observe("cam1", "EB", 100)
    assert before.windows_seen == 0  # nothing known before the first window
    after = b.stats_for("cam1", "EB")
    assert after.ewma == 100.0
    assert after.windows_seen == 1


def test_ewma_math_over_sequence():
    b = EwmaBaseline(alpha=0.5)
    for count in (100, 200):
        b.observe("cam1", "EB", count)
    # 0.5*200 + 0.5*100
    assert b.stats_for("cam1", "EB").ewma == pytest.approx(150.0)
    b.observe("cam1", "EB", 50)
    assert b.stats_for("cam1", "EB").ewma == pytest.approx(100.0)  # 0.5*50 + 0.5*150


def test_observe_returns_pre_observation_stats():
    b = EwmaBaseline(alpha=0.5)
    b.observe("cam1", "EB", 100)
    before = b.observe("cam1", "EB", 900)
    assert before.ewma == pytest.approx(100.0)  # the spike isn't in its own baseline
    assert before.windows_seen == 1


def test_keys_are_independent():
    b = EwmaBaseline()
    b.observe("cam1", "EB", 100)
    b.observe("cam1", "WB", 5)
    b.observe("cam2", "EB", 42)
    assert b.stats_for("cam1", "EB").ewma == 100.0
    assert b.stats_for("cam1", "WB").ewma == 5.0
    assert b.stats_for("cam2", "EB").ewma == 42.0
    assert b.stats_for("cam9", "EB").windows_seen == 0
