"""Representative activation-producing races owned by preparation."""
from __future__ import annotations

import gc
import math
import statistics
from contextlib import contextmanager
from dataclasses import dataclass

from b12x._lib.compile_plan import (
    ProgramKey, forbid_lowering, observe_programs, record_program, retain_compiled_programs,
)
from .types import PreparedCall, _prime, _close_all

# Rounds each surviving candidate completes after calibration, and the factor by
# which a candidate may trail the leader and still be re-timed. Across recorded
# races the round-to-round spread of one candidate stays under 4% and the
# eventual winner never trailed the first round's leader by more than 0.2%.
SURVIVOR_ROUNDS = 3
ELIMINATION_MARGIN = 1.10


class _ParentCompilations:
    """Scoped actual-compilation counters and program keys, not launch tracing."""

    def __init__(self, *, cache_only):
        self.cache_only = cache_only
        self.cute = self.triton = 0

    def __enter__(self):
        from triton import knobs
        from b12x._lib import compiler
        self.compiler, self.knobs = compiler, knobs
        self.before = int(compiler.compile_cache_info()["compile_misses"])
        self.original_compile = compiler._call_cute_compile
        self.previous_listener = knobs.compilation.listener
        self.observation = observe_programs()
        self.programs = self.observation.__enter__()

        def listener(*args, **kwargs):
            metadata = kwargs.get("metadata", args[1] if len(args) > 1 else {})
            record_program(ProgramKey("triton", metadata["hash"], metadata.get("name", "")))
            if not kwargs.get("cache_hit", args[4] if len(args) > 4 else False):
                self.triton += 1
            if self.previous_listener is not None:
                self.previous_listener(*args, **kwargs)

        def reject_compile(*_args, **_kwargs):
            name = getattr(_kwargs.get("compile_spec"), "kernel_id", "unknown")
            raise RuntimeError(
                f"no-compilation phase encountered an unplanned CuTe program: {name} {_kwargs.get('cache_key')}"
            )

        knobs.compilation.listener = listener
        if self.cache_only:
            compiler._call_cute_compile = reject_compile
        return self

    def check(self):
        self.cute = int(self.compiler.compile_cache_info()["compile_misses"]) - self.before
        if self.cache_only and (self.cute or self.triton):
            raise RuntimeError(f"no-compilation phase compiled CuTe={self.cute}, Triton={self.triton}")

    def __exit__(self, kind, value, traceback):
        self.cute = int(self.compiler.compile_cache_info()["compile_misses"]) - self.before
        self.knobs.compilation.listener = self.previous_listener
        if self.cache_only:
            self.compiler._call_cute_compile = self.original_compile
        self.observation.__exit__(kind, value, traceback)


@contextmanager
def no_compilation():
    """Permit already-built object loads, but no new CuTe/Triton compilation."""
    with _ParentCompilations(cache_only=True) as observed, forbid_lowering():
        yield observed
        observed.check()


class _TimedCall:
    def __init__(self, call, eviction, samples):
        import torch
        self.call, self.eviction = call, eviction
        self.events = tuple((torch.cuda.Event(enable_timing=True, external=True),
                             torch.cuda.Event(enable_timing=True, external=True)) for _ in range(samples))
        self.graph = None
        with retain_compiled_programs() as self._retained:
            if call.capture_safe:
                self.graph = torch.cuda.CUDAGraph()
                # A discarded CuTe executor can own a cyclic reference to its CUDA
                # library. Finalizing that cycle during capture invalidates it.
                collecting = gc.isenabled()
                gc.disable()
                try:
                    with torch.cuda.graph(self.graph):
                        self._invoke()
                finally:
                    if collecting:
                        gc.enable()

    def _invoke(self):
        from .types import call_scope

        with call_scope():
            for start, end in self.events:
                self.eviction()
                if self.call.reset is not None:
                    self.call.reset()
                if self.call.produce is not None:
                    self.call.produce()
                start.record()
                self.call.invoke()
                end.record()

    def replay(self):
        if self.graph is None:
            self._invoke()
        else:
            self.graph.replay()

    def samples(self):
        return tuple(start.elapsed_time(end) * 1000.0 for start, end in self.events)

    def close(self):
        if self.graph is not None:
            self.graph.reset()
        # CUDA graph nodes do not own their compiled libraries. Drop these
        # references only after destroying the graph executable.
        self._retained = None


@dataclass
class PreparedRace:
    timers: tuple[_TimedCall, ...]
    eviction: object
    sample_count: int
    completed_rounds: int = 0
    planned_rounds: int = 0
    active_count: int = 0
    latest_round_us: tuple[float, ...] = ()

    def close(self):
        _close_all(timer.close for timer in self.timers)


@dataclass(frozen=True)
class RaceMeasurements:
    latencies_us: tuple[float, ...]
    overlapped_samples: int



def _l2_flush_fn(device: object, *, enabled: bool):
    if not enabled:
        return None
    import torch

    properties = torch.cuda.get_device_properties(device)
    flush_bytes = max(2 * int(properties.L2_cache_size), 64 << 20)
    buffer = torch.ones(
        (flush_bytes + 3) // 4,
        dtype=torch.float32,
        device=device,
    )
    reduction = torch.empty((), dtype=torch.float32, device=device)

    def flush() -> None:
        torch.sum(buffer, dim=0, out=reduction)

    return flush


def prepare_race_steps(calls, *, device_ordinal, samples=8):
    """Yield only outside GPU/compilation scopes, once per candidate bracket."""
    import torch
    if not calls or type(samples) is not int or samples <= 0:
        raise ValueError("a race requires candidates and positive samples")
    if any(call.produce is None for call in calls):
        raise ValueError("candidate races require an activation-producing context")
    timers = []
    completed = False
    try:
        with torch.cuda.device(device_ordinal), no_compilation():
            eviction = _l2_flush_fn(torch.device("cuda", device_ordinal), enabled=True)
            eviction()
            torch.cuda.synchronize(device_ordinal)
        yield
        for call in calls:
            with torch.cuda.device(device_ordinal), no_compilation():
                _prime(call)
                torch.cuda.synchronize(device_ordinal)
            yield
        for call in calls:
            with torch.cuda.device(device_ordinal), no_compilation():
                timers.append(_TimedCall(call, eviction, samples))
                torch.cuda.synchronize(device_ordinal)
            yield
        completed = True
        return PreparedRace(tuple(timers), eviction, samples)
    finally:
        if not completed:
            _close_all(timer.close for timer in timers)


def measure_race_steps(
    prepared, *, device_ordinal, rounds=7, compilation_active=None,
    eliminate=False, champion=False,
):
    """Balanced comparison; cancellation discards this generator's result.

    A selection race sets ``eliminate``: a timer whose best round so far trails
    the leader by more than ELIMINATION_MARGIN stops being re-timed and keeps the
    median of the rounds it completed, survivors run at most SURVIVOR_ROUNDS
    rounds, and ``champion`` exempts timer 0, which carries the previous batch's
    winner. Left unset, every timer completes every round.
    """
    import torch
    if type(rounds) is not int or rounds <= 0:
        raise ValueError("race rounds must be positive")
    if eliminate:
        rounds = min(rounds, SURVIVOR_ROUNDS)
    prepared.planned_rounds = rounds
    prepared.active_count = len(prepared.timers)
    values = [[] for _ in prepared.timers]
    active = list(range(len(prepared.timers)))
    overlaps = 0
    for _ in range(2):
        for timer in prepared.timers:
            with torch.cuda.device(device_ordinal), no_compilation():
                timer.replay()
                torch.cuda.synchronize(device_ordinal)
            yield
    pilots = tuple(statistics.fmean(timer.samples()) for timer in prepared.timers)
    if any(not math.isfinite(value) or value <= 0 for value in pilots):
        raise RuntimeError("candidate calibration produced an invalid latency")
    repeats = tuple(
        2 * max(1, math.ceil(1024.0 / (2 * prepared.sample_count * pilot)))
        if timer.call.capture_safe else 1
        for timer, pilot in zip(prepared.timers, pilots)
    )
    for turn in range(rounds):
        order = list(active)
        if turn % 2:
            order.reverse()
        offset = (turn // 2) % len(order)
        order = order[offset:] + order[:offset]
        totals = [0.0] * len(prepared.timers)
        for repetition in range(max(repeats[index] for index in order)):
            for index in (order if repetition % 2 == 0 else reversed(order)):
                if repetition >= repeats[index]:
                    continue
                with torch.cuda.device(device_ordinal), no_compilation():
                    if compilation_active is not None and compilation_active():
                        overlaps += prepared.sample_count
                    timer = prepared.timers[index]
                    timer.replay()
                    torch.cuda.synchronize(device_ordinal)
                    totals[index] += statistics.fmean(timer.samples())
                yield
        for index in order:
            values[index].append(totals[index] / repeats[index])
        # Timers that sat out this round report no latency, so a reader of the
        # round feed does not mistake a stale entry for a fresh measurement.
        timed = frozenset(order)
        prepared.latest_round_us = tuple(
            series[-1] if index in timed else math.nan
            for index, series in enumerate(values)
        )
        prepared.completed_rounds = turn + 1
        if eliminate:
            best = [min(series) for series in values]
            leader = min(best[index] for index in active)
            # A timer outside the leader's margin stops being re-timed and keeps
            # the median of the rounds it completed; the champion at position 0
            # is re-timed against every batch.
            active = [
                index for index in active
                if best[index] <= ELIMINATION_MARGIN * leader or (champion and index == 0)
            ]
            prepared.active_count = len(active)
    latencies = tuple(statistics.median(series) for series in values)
    if any(not math.isfinite(value) or value <= 0 for value in latencies):
        raise RuntimeError("candidate race produced an invalid latency")
    return RaceMeasurements(latencies, overlaps)


def _consume(steps):
    while True:
        try:
            next(steps)
        except StopIteration as finished:
            return finished.value


def _prepare_race(calls, *, device_ordinal, samples=8):
    return _consume(prepare_race_steps(calls, device_ordinal=device_ordinal, samples=samples))


def _measure_race(prepared, *, device_ordinal, rounds=7, compilation_active=None):
    return _consume(measure_race_steps(
        prepared, device_ordinal=device_ordinal, rounds=rounds,
        compilation_active=compilation_active,
    ))
