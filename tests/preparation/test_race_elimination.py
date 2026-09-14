"""Rounds after the first re-time only the candidates near the leader.

The race protocol lives in ``b12x.preparation._measurement``; these cases drive
it with timers that report fixed latencies, so the survivor set, the round
budget and the reported medians are exact rather than device-dependent.
"""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from b12x.preparation import _measurement
from b12x.preparation._measurement import (
    ELIMINATION_MARGIN, PreparedRace, SURVIVOR_ROUNDS, measure_race_steps,
)


class _Timer:
    """A timer whose successive replays report the given latencies in order."""

    def __init__(self, *latencies_us):
        self.latencies_us = latencies_us
        self.replays = 0
        self.call = SimpleNamespace(capture_safe=False)

    def replay(self):
        self.replays += 1

    def samples(self):
        # Two calibration laps precede round one and report the first latency.
        index = min(max(self.replays - 3, 0), len(self.latencies_us) - 1)
        return (self.latencies_us[index],)


@pytest.fixture
def host_timing(monkeypatch):
    """Run the protocol on the host: no device selection, no compilation guard."""
    @contextmanager
    def scope(*_args, **_kwargs):
        yield None

    monkeypatch.setattr(torch.cuda, "device", scope)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(_measurement, "no_compilation", scope)


def _race(timers):
    return PreparedRace(tuple(timers), None, 1)


def _measure(prepared, *, eliminate=True, **kwargs):
    steps = measure_race_steps(prepared, device_ordinal=0, eliminate=eliminate, **kwargs)
    while True:
        try:
            next(steps)
        except StopIteration as finished:
            return finished.value


def _rounds(timer):
    """Replays beyond the two calibration laps."""
    return timer.replays - 2


def test_trailing_candidates_stop_after_one_round_and_keep_their_median(host_timing):
    leader, near, behind, far = _Timer(10.0), _Timer(10.9), _Timer(12.0), _Timer(40.0)
    prepared = _race((leader, near, behind, far))

    measurement = _measure(prepared, rounds=7)

    assert [_rounds(timer) for timer in (leader, near, behind, far)] == [3, 3, 1, 1]
    assert measurement.latencies_us == (10.0, 10.9, 12.0, 40.0)
    assert prepared.completed_rounds == prepared.planned_rounds == SURVIVOR_ROUNDS
    assert prepared.active_count == 2


def test_a_candidate_at_the_margin_is_measured_to_the_last_round(host_timing):
    leader, edge = _Timer(10.0), _Timer(10.0 * ELIMINATION_MARGIN)
    prepared = _race((leader, edge))

    _measure(prepared, rounds=7)

    assert _rounds(edge) == SURVIVOR_ROUNDS
    assert prepared.active_count == 2


def test_an_improving_leader_removes_a_candidate_after_a_later_round(host_timing):
    improving, trailing = _Timer(10.0, 8.0, 8.0), _Timer(10.5)
    prepared = _race((improving, trailing))

    measurement = _measure(prepared, rounds=7)

    assert [_rounds(timer) for timer in (improving, trailing)] == [3, 2]
    assert measurement.latencies_us == (8.0, 10.5)
    assert prepared.active_count == 1


def test_the_carried_champion_is_re_timed_against_every_batch(host_timing):
    champion, leader = _Timer(40.0), _Timer(10.0)
    prepared = _race((champion, leader))

    measurement = _measure(prepared, rounds=7, champion=True)

    assert _rounds(champion) == SURVIVOR_ROUNDS
    assert measurement.latencies_us == (40.0, 10.0)
    assert prepared.active_count == 2


def test_a_first_timer_that_is_not_a_champion_is_eliminated_like_any_other(host_timing):
    first, leader = _Timer(40.0), _Timer(10.0)
    prepared = _race((first, leader))

    _measure(prepared, rounds=7)

    assert [_rounds(timer) for timer in (first, leader)] == [1, 3]
    assert prepared.active_count == 1


def test_a_lower_session_round_count_still_bounds_the_survivors(host_timing):
    leader, near = _Timer(10.0), _Timer(10.5)
    prepared = _race((leader, near))

    _measure(prepared, rounds=2)

    assert prepared.planned_rounds == 2
    assert [_rounds(timer) for timer in (leader, near)] == [2, 2]


def test_an_exhaustive_race_re_times_every_candidate_every_round(host_timing):
    """The acceptance recheck compares candidates over the rounds it asks for."""
    leader, far = _Timer(10.0), _Timer(40.0)
    prepared = _race((leader, far))

    measurement = _measure(prepared, rounds=5, eliminate=False)

    assert [_rounds(timer) for timer in (leader, far)] == [5, 5]
    assert prepared.planned_rounds == 5
    assert measurement.latencies_us == (10.0, 40.0)


def test_overlapping_compilation_counts_every_replayed_sample(host_timing):
    leader, behind = _Timer(10.0), _Timer(40.0)
    prepared = PreparedRace((leader, behind), None, 8)

    measurement = _measure(prepared, rounds=7, compilation_active=lambda: True)

    assert measurement.overlapped_samples == 8 * (_rounds(leader) + _rounds(behind))
