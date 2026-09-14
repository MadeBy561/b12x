"""Rank-zero live dashboard for kernel preparation, with plain milestone lines for pipes.

The dashboard is a six-line panel: a stage rail, an overall request bar, an
activity line, and one detail line (a race track with a marker per candidate,
a staging grid, the compile forge, or a phase hint).
All accounting happens on the caller's thread inside ``update`` and is published
as one immutable frame; rich's refresh thread only reads that frame and the clock,
so it never touches a session, CUDA, a compiler, or a measurement object.
"""
from __future__ import annotations

import math
import statistics
import sys
import time
import zlib
from dataclasses import dataclass, replace

from .types import PreparationProgress

_STAGES = ("PLAN", "BUILD", "TUNE", "PRIME", "READY")
_PHASE_STAGE = {
    "planning": "PLAN", "selecting": "PLAN", "compiling": "BUILD",
    "preparing candidates": "TUNE", "calibrating": "TUNE", "autotuning": "TUNE",
    "priming": "PRIME", "finishing": "PRIME", "ready": "READY", "waiting for ranks": "WAIT",
}
# Activity copy: one line per phase, picked per request and rotated in long phases.
_PHASE_VERBS = {
    "planning": (
        "charting the requests", "scanning the kernel catalog", "mapping the request graph",
        "reading persisted selections", "indexing the tuning contracts",
        "resolving request dependencies", "sizing the workspace",
    ),
    "selecting": (
        "choosing a configuration", "checking the selection cache", "picking the contenders",
        "reading the tuning contract", "weighing the knobs", "matching persisted selections",
    ),
    "compiling": (
        "forging programs", "lowering CuTe to PTX", "feeding the compile pool",
        "hammering out kernels", "spinning up the forge", "assembling machine code",
        "compiling in worker processes", "building the program set",
    ),
    "preparing candidates": (
        "staging the contenders", "capturing CUDA graphs", "lining up the racers",
        "loading the contenders", "warming the contenders", "priming every lane",
    ),
    "calibrating": (
        "warming up the track", "running pilot laps", "sizing the repeat count",
        "measuring pilot latency", "flushing the L2", "settling the clocks",
    ),
    "autotuning": (
        "racing the contenders", "timing every lane", "running balanced rounds",
        "letting them race", "measuring the contenders", "watching the lanes",
        "collecting round medians", "trading laps between lanes",
    ),
    "priming": (
        "priming the kernels", "admitting the winner", "first real launch",
        "loading the winner", "warming the winner", "binding real inputs",
    ),
    "finishing": (
        "cooling the forge", "draining the compile pool", "evicting unretained programs",
        "tidying the program cache", "sealing the selections",
    ),
    "ready": ("all kernels ready",),
    "waiting for ranks": (
        "waiting for the other ranks", "holding for the collective", "waiting at the rendezvous",
        "the other ranks are still loading",
    ),
}
_PHASE_HINT = {
    "planning": "reading the tuning catalog and persisted selections",
    "selecting": "a persisted selection skips the race; a single-point space is fixed",
    "compiling": "programs compile in parallel worker processes; cached ones load instantly",
    "preparing candidates": "each contender is compiled, primed, and captured into a CUDA graph",
    "calibrating": "two pilot laps size the repeat count so every contender runs about 1 ms per round",
    "autotuning": "rounds alternate direction and rotate the start lane to cancel ordering bias",
    "priming": "the winner runs once with real inputs before it is admitted for serving",
    "finishing": "draining the compiler and evicting programs nothing retained",
    "waiting for ranks": "collective kernels wait until every rank reaches the same request",
}
# Phosphor palette: brightness carries meaning, hue stays green except for failure.
_INK = "#c8ffd4"
_SIGNAL = "#00ff41"
_GLOW = "#b7ffcb"
_LEAF = "#39d353"
_MOSS = "#26a641"
_DIM = "#3fa34d"
_TRACK = "#0f3d1a"
_FLASH = "bold #e6ffe9"
_STAGE_STYLE = {
    "PLAN": _DIM, "BUILD": _MOSS, "TUNE": _SIGNAL, "PRIME": _LEAF,
    "READY": _SIGNAL, "WAIT": _GLOW, "FAILED": "#ff5252",
}
_FAMILY_STYLE = {
    "attention": _SIGNAL, "gemm": _LEAF, "moe": "#7cfc9a", "norm": _MOSS,
    "sequence": "#5efc82", "quantization": _GLOW, "comm": "#00c853",
}
_OUTCOME_GLYPH = {
    "tuned": ("◆", _SIGNAL), "cached": ("●", "#5efc82"),
    "fixed": ("▪", "#2e7d3a"), "default": ("▲", _GLOW),
}
_OUTCOME_PRIORITY = ("tuned", "default", "cached", "fixed")
_SPINNERS = {
    "PLAN": (80, "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"), "BUILD": (120, "▖▘▝▗"), "TUNE": (100, "◜◠◝◞◡◟"),
    "PRIME": (90, "▁▂▃▄▅▆▇█▇▆▅▄▃▂"), "WAIT": (350, "◐◓◑◒"), "READY": (1000, "✓"),
    "FAILED": (1000, "✗"),
}
_BLOCKS = "▁▂▃▄▅▆▇█"
_EIGHTHS = "▏▎▍▌▋▊▉█"
_RACE_PHASES = frozenset({"preparing candidates", "calibrating", "autotuning"})


def _ramp(anchors, steps=48):
    def channel(index, position):
        scaled = position * (len(anchors) - 1)
        low = min(len(anchors) - 2, int(scaled))
        a, b = anchors[low], anchors[low + 1]
        t = scaled - low
        return round(int(a[index:index + 2], 16) * (1 - t) + int(b[index:index + 2], 16) * t)

    anchors = tuple(anchor.lstrip("#") for anchor in anchors)
    return tuple(
        "#" + "".join(f"{channel(index, step / (steps - 1)):02x}" for index in (0, 2, 4))
        for step in range(steps)
    )


_PHOSPHOR = _ramp(("#0b5d1e", "#00b32c", _SIGNAL, _GLOW))
_LANE_RAMPS = (
    _ramp(("#00c231", _GLOW)), _ramp(("#0a7d2a", _LEAF)),
    _ramp(("#0a5c22", _MOSS)), _ramp((_TRACK, "#1f6f34")),
)
_LANE_LABEL = (f"bold {_GLOW}", _LEAF, _MOSS, _DIM)
_TRAIL = (_FLASH, _GLOW, _LEAF)


def _plain(value):
    return "".join(character if character.isprintable() else " " for character in value)


def _duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}:{seconds:02d}"


def _brief(seconds):
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    return _duration(seconds)


def _us(value):
    if value >= 1000:
        return f"{value / 1000:.2f} ms"
    if value >= 100:
        return f"{value:.0f} µs"
    return f"{value:.1f} µs"


def _spark(values, width=8):
    values = tuple(values)
    if not values:
        return ""
    if any(not math.isfinite(value) for value in values):
        return "?" * min(width, len(values))
    count = min(width, len(values))
    bins = []
    for index in range(count):
        begin, end = index * len(values) // count, (index + 1) * len(values) // count
        bins.append(sum(values[begin:end]) / (end - begin))
    lower, upper = min(bins), max(bins)
    if upper <= lower:
        return _BLOCKS[3] * count
    return "".join(_BLOCKS[min(7, int(7 * (value - lower) / (upper - lower)))] for value in bins)


def _verb(progress, seconds_on_request):
    verbs = _PHASE_VERBS.get(progress.phase)
    if not verbs:
        return progress.phase
    seed = zlib.crc32(f"{progress.component_id}|{progress.request_name}|{progress.phase}".encode())
    return verbs[(seed + int(seconds_on_request / 6)) % len(verbs)]


def _spinner(stage, now):
    interval, frames = _SPINNERS.get(stage, _SPINNERS["PLAN"])
    return frames[int(now * 1000 / interval) % len(frames)]


def _ease(t):
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


def _gradient_bar(fraction, cells, ramp, *, shimmer=None, track=_TRACK, track_glyph="·"):
    from rich.text import Text

    text = Text(no_wrap=True)
    cells = max(1, cells)
    eighths = int(min(1.0, max(0.0, fraction)) * cells * 8 + 0.5)
    full, remainder = divmod(eighths, 8)
    full = min(full, cells)
    last = len(ramp) - 1
    for index in range(full):
        offset = None if shimmer is None else shimmer - index
        style = _TRAIL[offset] if offset is not None and 0 <= offset < len(_TRAIL) else ramp[index * last // max(1, cells - 1)]
        text.append("█", style=style)
    if remainder and full < cells:
        text.append(_EIGHTHS[remainder - 1], style=ramp[full * last // max(1, cells - 1)])
        full += 1
    for index in range(full, cells):
        offset = None if shimmer is None else shimmer - index
        if offset is not None and 0 <= offset < len(_TRAIL):
            text.append("•", style=_TRAIL[offset])
        else:
            text.append(track_glyph, style=track)
    return text


@dataclass(frozen=True)
class _Lane:
    index: int
    median_us: float
    history: tuple[float, ...]


@dataclass(frozen=True)
class _Frame:
    progress: PreparationProgress
    stage: str
    lanes: tuple[_Lane, ...] = ()
    previous_medians: tuple[tuple[int, float], ...] = ()
    lanes_changed_at: float = 0.0
    journey: tuple[str, ...] = ()
    compile_rate: tuple[int, ...] = ()
    peak_active: int = 0
    widest: tuple[float, str] | None = None
    request_started: float = 0.0
    eta: tuple[float, float] | None = None
    failed: bool = False


class PreparationDisplay:
    """Live dashboard for global rank zero; plain milestone lines when piped.

    Only the coordinator for global rank zero creates a live renderer. The
    refresh thread reads immutable frames, never a session, CUDA, a compiler,
    or a measurement object. Pipe output is plain milestone text.
    """

    def __init__(self, *, global_rank: int, stream=None):
        if type(global_rank) is not int or global_rank < 0:
            raise ValueError("progress display requires a nonnegative global rank")
        self._enabled = global_rank == 0
        self._stream = sys.stderr if stream is None else stream
        self._frame = _Frame(PreparationProgress(False, False, (), False), "PLAN")
        self._live = None
        self._console = None
        self._started = None
        self._ended = None
        self._closed = False
        self._last_log = 0.0
        self._logged = set()
        self._last_stop = False
        self._signature = None
        self._frame_at = 0.0
        self._completed_seen = 0
        self._measured_seen = 0
        self._request_key = None
        self._request_started = 0.0
        self._request_candidates = 0
        self._request_raced = False
        self._request_cache_hits = 0
        self._journey = []
        self._race_history = []
        self._race_rounds = 0
        self._lanes = ()
        self._previous_medians = ()
        self._lanes_changed_at = 0.0
        self._compile_seen = 0
        self._compile_buckets = {}
        self._peak_active = 0
        self._widest = None
        self._eta = None

    def __enter__(self):
        if self._started is not None or self._closed:
            raise RuntimeError("preparation display is single-use")
        self._started = time.monotonic()
        self._request_started = self._started
        if self._enabled and self._stream.isatty():
            from rich.console import Console

            self._attach(Console(file=self._stream))
        return self

    def _attach(self, console):
        from rich.live import Live

        self._console = console
        self._live = Live(
            console=console, get_renderable=self._render, refresh_per_second=10,
            screen=False, transient=False, redirect_stdout=True, redirect_stderr=True,
            vertical_overflow="crop",
        )
        self._live.start(refresh=True)

    def update(self, progress: PreparationProgress):
        if self._started is None or self._closed:
            raise RuntimeError("progress updates require an active display context")
        if not isinstance(progress, PreparationProgress):
            raise TypeError("display requires a PreparationProgress snapshot")
        if not self._enabled:
            return
        now = time.monotonic()
        signature = (
            progress.phase, progress.component_id, progress.request_name, progress.completed_requests,
            progress.candidate_count, progress.candidates_prepared, progress.completed_rounds,
            progress.measured_candidates, progress.cache_hits, progress.compilations,
            progress.active_compilations, progress.tuning_stopped, progress.done,
        )
        if signature == self._signature and now - self._frame_at < 0.25 and not progress.done:
            return
        self._signature = signature
        self._frame_at = now
        self._account(progress, now)
        self._frame = _Frame(
            progress, _PHASE_STAGE.get(progress.phase, "PLAN"),
            lanes=self._lanes, previous_medians=self._previous_medians,
            lanes_changed_at=self._lanes_changed_at, journey=tuple(self._journey),
            compile_rate=self._rate(now), peak_active=self._peak_active, widest=self._widest,
            request_started=self._request_started, eta=self._eta,
        )
        if progress.done:
            self._ended = now
        if self._live is not None:
            if progress.done:
                self._live.stop()
            return
        milestone = (progress.component_id, progress.phase)
        if (milestone not in self._logged or progress.done
                or progress.tuning_stopped != self._last_stop or now - self._last_log >= 10):
            self._logged.add(milestone)
            self._last_log = now
            self._last_stop = progress.tuning_stopped
            self._write_milestone()

    def _account(self, progress, now):
        previous = self._frame.progress
        completed = progress.completed_requests - self._completed_seen
        if completed > 0:
            measured = progress.measured_candidates - self._measured_seen
            if self._request_raced:
                outcome = "tuned" if measured > 0 else "default"
            elif self._request_candidates == 1:
                outcome = "fixed"
            elif self._request_candidates == 0 or previous.cache_hits > self._request_cache_hits:
                outcome = "cached"
            else:
                outcome = "default"
            self._journey.extend(["fixed"] * (completed - 1))
            self._journey.append(outcome)
            if outcome == "tuned" and len(self._lanes) > 1:
                spread = self._lanes[-1].median_us / self._lanes[0].median_us
                if self._widest is None or spread > self._widest[0]:
                    self._widest = (spread, previous.component_id)
            self._completed_seen = progress.completed_requests
            self._measured_seen = progress.measured_candidates
            if progress.completed_requests >= 3 and progress.total_requests > progress.completed_requests:
                estimate = (now - self._started) / progress.completed_requests
                estimate *= progress.total_requests - progress.completed_requests
                smoothed = estimate if self._eta is None else 0.7 * max(0.0, self._eta[0] - (now - self._eta[1])) + 0.3 * estimate
                self._eta = (smoothed, now)
        key = (progress.component_id, progress.request_name)
        if key != self._request_key:
            self._request_key = key
            self._request_started = now
            self._request_candidates = 0
            self._request_raced = False
            self._request_cache_hits = previous.cache_hits
            self._reset_race()
        self._request_candidates = max(self._request_candidates, progress.candidate_count)
        self._request_raced = self._request_raced or progress.phase in _RACE_PHASES
        if progress.completed_rounds < self._race_rounds:
            self._reset_race()
        if progress.completed_rounds > self._race_rounds:
            values = progress.latest_round_us
            if len(values) != len(self._race_history):
                self._race_history = [[] for _ in values]
            for series, value in zip(self._race_history, values, strict=True):
                if math.isfinite(value):
                    series.append(value)
            self._race_rounds = progress.completed_rounds
            self._previous_medians = tuple((lane.index, lane.median_us) for lane in self._lanes)
            lanes = []
            for index, series in enumerate(self._race_history):
                if series and all(math.isfinite(value) and value > 0 for value in series):
                    lanes.append(_Lane(index, statistics.median(series), tuple(series[-8:])))
            self._lanes = tuple(sorted(lanes, key=lambda lane: (lane.median_us, lane.index)))
            self._lanes_changed_at = now
        built = progress.compilations - self._compile_seen
        if built > 0:
            bucket = int((now - self._started) / 2)
            self._compile_buckets[bucket] = self._compile_buckets.get(bucket, 0) + built
            self._compile_seen = progress.compilations
            for stale in [key for key in self._compile_buckets if key < bucket - 24]:
                del self._compile_buckets[stale]
        self._peak_active = max(self._peak_active, progress.active_compilations)

    def _reset_race(self):
        self._race_history = []
        self._race_rounds = 0
        self._lanes = ()
        self._previous_medians = ()

    def _rate(self, now):
        if not self._compile_buckets:
            return ()
        bucket = int((now - self._started) / 2)
        return tuple(self._compile_buckets.get(index, 0) for index in range(bucket - 23, bucket + 1))

    def _elapsed(self):
        return (self._ended if self._ended is not None else time.monotonic()) - self._started

    def _write_milestone(self):
        p = self._frame.progress
        phase = "failed" if self._frame.failed else p.phase
        component = f" {p.component_id}" if p.component_id else ""
        candidates = f", candidates {p.candidates_prepared}/{p.candidate_count} prepared" if p.candidate_count else ""
        rounds = f", round {p.completed_rounds}/{p.total_rounds}" if p.completed_rounds else ""
        stopped = ", tuning stopped; completing required preparation" if p.tuning_stopped and not p.done else ""
        line = (
            f"b12x {phase}{component}: {p.completed_requests}/{p.total_requests} ready"
            f"{candidates}{rounds}, {p.measured_candidates} measured, {p.cache_hits} cached, "
            f"{p.compilations} compilations, {_duration(self._elapsed())}{stopped}"
        )
        self._stream.write(_plain(line) + "\n")
        self._stream.flush()

    # Rendering runs on rich's refresh thread and reads only the published frame.

    def _render(self):
        from rich import box
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        frame = self._frame
        now = time.monotonic()
        width, height = self._console.width, self._console.height
        if width < 84 or height < 8:
            return self._render_line(frame, now, width)
        p = frame.progress
        stage = "FAILED" if frame.failed else frame.stage
        color = _STAGE_STYLE[stage]
        inner = width - 4
        rows = [
            self._rail_row(frame, now, color),
            self._progress_row(frame, now, inner, color),
            self._activity_row(frame, now, color),
            self._summary_row(frame) if p.done or frame.failed else self._detail_row(frame, now, inner, color),
        ]
        grid = Table.grid(expand=True)
        grid.add_column(no_wrap=True, overflow="crop")
        for row in rows:
            grid.add_row(row)
        if frame.failed:
            title = "b12x · preparation failed"
        elif p.done:
            title = "✦ b12x · kernels ready ✦"
        else:
            title = "b12x · kernel preparation"
        return Panel(
            grid, box=box.ROUNDED, border_style=color, padding=(0, 1), expand=True,
            title=Text(title, style=f"bold {color}"), title_align="left",
        )

    def _two(self, left, right):
        from rich.table import Table

        grid = Table.grid(padding=(0, 1), expand=True)
        grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        grid.add_column(no_wrap=True, justify="right")
        grid.add_row(left, right)
        return grid

    def _component(self, component_id, request_name="", *, limit=28):
        from rich.text import Text

        text = Text(no_wrap=True)
        if not component_id:
            return text
        family, _, name = _plain(component_id).partition(".")
        style = _FAMILY_STYLE.get(family, _INK)
        text.append(family, style=style)
        if name:
            text.append("·", style=_DIM)
            text.append(name[:limit], style=f"bold {style}")
        if request_name:
            text.append(f"  {_plain(request_name)[:limit]}", style=_DIM)
        return text

    def _rail_row(self, frame, now, color):
        from rich.text import Text

        p = frame.progress
        rail = Text(no_wrap=True)
        active = None if frame.failed else frame.stage
        for position, stage in enumerate(_STAGES):
            if position:
                rail.append("  ")
            if p.done and stage == "READY" and not frame.failed:
                rail.append("✓ READY", style=f"bold {_SIGNAL}")
            elif stage == active and not p.done:
                marker = "◉" if int(now * 2) % 2 == 0 else "◎"
                rail.append(f"{marker} {stage}", style=f"bold {color}")
            else:
                rail.append(f"○ {stage}", style=_DIM)
        if active == "WAIT":
            rail.append(f"  {_spinner('WAIT', now)} WAIT", style=f"bold {_GLOW}")
        if frame.failed:
            rail.append("  ✗ FAILED", style="bold #ff5252")
        if p.component_id and not p.done:
            rail.append("   ")
            rail.append_text(self._component(p.component_id, p.request_name))
        clock = Text(no_wrap=True)
        clock.append(_duration(self._elapsed()), style=f"bold {_INK}")
        if frame.eta is not None and not p.done and not frame.failed and p.completed_requests < p.total_requests:
            remaining = max(0.0, frame.eta[0] - (now - frame.eta[1]))
            clock.append(f"  ≈ {_duration(remaining)} left", style=_DIM)
        return self._two(rail, clock)

    def _progress_row(self, frame, now, inner, color):
        from rich.text import Text

        p = frame.progress
        label = Text(no_wrap=True)
        label.append(f"{p.completed_requests}/{p.total_requests}", style=f"bold {color}")
        label.append(" requests", style=_DIM)
        cells = max(10, inner - label.cell_len - 2)
        fraction = (p.completed_requests / p.total_requests) if p.total_requests else 0.0
        shimmer = None
        if not p.done and not frame.failed:
            shimmer = int((now * 18) % (cells + 12)) - 6
        bar = _gradient_bar(fraction, cells, _PHOSPHOR, shimmer=shimmer)
        return self._two(bar, label)

    def _activity_row(self, frame, now, color):
        from rich.text import Text

        p = frame.progress
        stage = "FAILED" if frame.failed else frame.stage
        text = Text(no_wrap=True)
        text.append(f"{_spinner(stage, now)} ", style=f"bold {color}")
        if frame.failed:
            text.append("preparation failed", style="bold #ff5252")
            text.append(f" during {p.phase}", style=_DIM)
            return text
        text.append(_verb(p, now - frame.request_started), style=f"bold {_INK}")
        if p.phase == "autotuning" and p.total_rounds:
            text.append(f"  {p.candidate_count} configs  ", style=_DIM)
            for turn in range(p.total_rounds):
                text.append("●" if turn < p.completed_rounds else "○", style=color if turn < p.completed_rounds else _TRACK)
            text.append(f" {p.completed_rounds}/{p.total_rounds}", style=_DIM)
            if frame.lanes:
                leader = frame.lanes[0]
                text.append("  leader ", style=_DIM)
                text.append(f"#{leader.index}", style=f"bold {_GLOW}")
                text.append(f" {_us(leader.median_us)}", style=_GLOW)
                if len(frame.lanes) > 1:
                    spread = frame.lanes[-1].median_us / leader.median_us
                    text.append(f"  spread {spread:.2f}×", style=_DIM)
                if p.active_count and p.active_count < len(frame.lanes):
                    text.append(f"  {p.active_count}/{len(frame.lanes)} timed", style=_DIM)
        elif p.phase == "preparing candidates":
            text.append(f"  {p.candidates_prepared}/{p.candidate_count} staged", style=_DIM)
        elif p.phase == "calibrating":
            text.append(f"  {p.candidate_count} configs · pilot laps", style=_DIM)
        elif p.phase == "waiting for ranks":
            keys = ", ".join(_plain(str(requirement.key))[:24] for requirement in p.ready_collectives[:2])
            if keys:
                text.append(f"  {keys}", style=_DIM)
        elif p.phase == "compiling" and p.candidate_count >= 2 and p.request_name:
            text.append(f"  {p.candidate_count} contenders", style=_DIM)
        if p.tuning_stopped and not p.done:
            text.append("  ⚠ tuning stopped · defaults for the rest", style=f"bold {_GLOW}")
        side = Text(no_wrap=True)
        if not p.done and p.request_name:
            side.append(f"{_brief(now - frame.request_started)} on request", style=_DIM)
        if p.active_compilations:
            side.append("  ⚙ ", style=_LEAF)
            side.append(f"{p.active_compilations} compiling", style=_LEAF)
        return self._two(text, side)

    def _detail_row(self, frame, now, inner, color):
        p = frame.progress
        if frame.lanes and p.phase in ("autotuning", "compiling") and len(frame.lanes) <= p.candidate_count:
            return self._race_row(frame, now, inner)
        if p.candidate_count >= 2 and p.phase in _RACE_PHASES:
            return self._staging_row(frame, now, inner, color)
        if p.active_compilations or p.phase == "compiling":
            return self._forge_row(frame, now)
        return self._hint_row(frame)

    def _race_row(self, frame, now, inner):
        """One track: each candidate sits at leader/median of the way to the finish line."""
        from rich.text import Text

        lanes = frame.lanes
        previous = dict(frame.previous_medians)
        eased = _ease((now - frame.lanes_changed_at) / 0.55)
        values = {}
        for lane in lanes:
            before = previous.get(lane.index, lane.median_us)
            values[lane.index] = before + (lane.median_us - before) * eased
        leader_value = min(values.values())
        label = Text(no_wrap=True)
        label.append(f"★ #{lanes[0].index} {_us(lanes[0].median_us)}", style=f"bold {_GLOW}")
        for rank, lane in enumerate(lanes[1:3], start=1):
            label.append(f"  #{lane.index} {100 * (lane.median_us / lanes[0].median_us - 1):+.0f}%", style=_LANE_LABEL[rank])
        if len(lanes) > 3:
            label.append(f"  +{len(lanes) - 3} more", style=_DIM)
        cells = max(16, inner - label.cell_len - 9)
        placed = [None] * cells
        for rank in reversed(range(len(lanes))):
            position = int(round((cells - 1) * leader_value / values[lanes[rank].index]))
            placed[min(cells - 1, max(0, position))] = rank
        track = Text(no_wrap=True)
        track.append("race ", style=_DIM)
        for rank in placed:
            if rank is None:
                track.append("┈", style=_TRACK)
            elif rank == 0:
                track.append("★", style=f"bold {_GLOW}")
            elif rank < 3:
                track.append("◆", style=_LANE_LABEL[rank])
            else:
                track.append("●", style=_DIM)
        track.append("┃", style=_GLOW)
        return self._two(track, label)

    def _staging_row(self, frame, now, inner, color):
        from rich.text import Text

        p = frame.progress
        slots = max(1, (inner - 10) // 8)
        text = Text(no_wrap=True)
        for index in range(min(slots, p.candidate_count)):
            if index < p.candidates_prepared:
                text.append(f"#{index:<3}✓  ", style="#5efc82")
            elif index == p.candidates_prepared and p.phase == "preparing candidates":
                text.append(f"#{index:<3}{_spinner('PLAN', now)}  ", style=f"bold {color}")
            elif p.phase == "calibrating":
                text.append(f"#{index:<3}{'◠◡'[(int(now * 4) + index) % 2]}  ", style=color)
            else:
                text.append(f"#{index:<3}○  ", style=_TRACK)
        if p.candidate_count > slots:
            text.append(f"+{p.candidate_count - slots} more", style=_DIM)
        return text

    def _forge_row(self, frame, now):
        from rich.text import Text

        p = frame.progress
        text = Text(no_wrap=True)
        text.append("⚙ ", style=_LEAF)
        slots = max(frame.peak_active, p.active_compilations, 1)
        for index in range(min(slots, 16)):
            if index < p.active_compilations:
                text.append(_SPINNERS["PLAN"][1][(int(now * 12) + index * 3) % 10], style=f"bold {_LEAF}")
            else:
                text.append("○", style=_TRACK)
            text.append(" ")
        if p.active_compilations:
            text.append(f"{p.active_compilations} forging", style=_LEAF)
        elif p.compilations:
            text.append("forge idle", style=_DIM)
        else:
            text.append("forge cold", style=_DIM)
        text.append(f"  {p.compilations} built", style=_INK)
        text.append(f" · {p.cache_hits} cached", style="#5efc82")
        if frame.compile_rate:
            text.append("  rate ", style=_DIM)
            text.append(_spark(frame.compile_rate, width=16), style=_LEAF)
            text.append(" /2s", style=_DIM)
        return text

    def _hint_row(self, frame):
        from rich.text import Text

        text = Text(no_wrap=True)
        hint = _PHASE_HINT.get(frame.progress.phase)
        if hint:
            text.append(f"  {hint}", style=_DIM)
        return text

    def _summary_row(self, frame):
        from rich.text import Text

        p = frame.progress
        counts = {outcome: frame.journey.count(outcome) for outcome in _OUTCOME_PRIORITY}
        text = Text(no_wrap=True)
        if frame.failed:
            text.append("✗ ", style="bold #ff5252")
            text.append(f"stopped after {p.completed_requests}/{p.total_requests} requests", style="#ff5252")
        else:
            text.append("✓ ", style=f"bold {_SIGNAL}")
            text.append(f"{p.total_requests} requests", style=f"bold {_INK}")
        text.append(f" in {_duration(self._elapsed())}", style=_INK)
        for outcome in _OUTCOME_PRIORITY:
            if counts[outcome]:
                glyph, style = _OUTCOME_GLYPH[outcome]
                text.append(f"  {glyph} {counts[outcome]} {outcome}", style=style)
        text.append(f"  ⚙ {p.compilations} built", style=_LEAF)
        text.append(f"  {p.measured_candidates} candidates measured", style=_DIM)
        if frame.widest is not None and not frame.failed:
            text.append(f"  widest race {frame.widest[0]:.2f}× ", style=_DIM)
            text.append_text(self._component(frame.widest[1], limit=20))
        return text

    def _render_line(self, frame, now, width):
        from rich.table import Table
        from rich.text import Text

        p = frame.progress
        stage = "FAILED" if frame.failed else frame.stage
        color = _STAGE_STYLE[stage]
        line = Text("b12x ", style="bold", no_wrap=True, overflow="ellipsis")
        line.append(stage, style=f"bold {color}")
        if p.component_id and not p.done:
            line.append("  ")
            line.append_text(self._component(p.component_id, limit=20))
        if width >= 60:
            line.append("  ")
            line.append_text(_gradient_bar(
                (p.completed_requests / p.total_requests) if p.total_requests else 0.0, 10, _PHOSPHOR,
                shimmer=None if p.done else int((now * 18) % 22) - 6,
            ))
        line.append(f" {p.completed_requests}/{p.total_requests}", style=color)
        if p.phase == "autotuning" and p.total_rounds:
            line.append(f"  r{p.completed_rounds}/{p.total_rounds}", style=_DIM)
            if frame.lanes:
                line.append(f" ★#{frame.lanes[0].index} {_us(frame.lanes[0].median_us)}", style=_GLOW)
        elif p.candidate_count and p.phase in _RACE_PHASES:
            line.append(f"  c{p.candidates_prepared}/{p.candidate_count}", style=_DIM)
        if p.active_compilations:
            line.append(f"  ⚙{p.active_compilations}", style=_LEAF)
        if p.tuning_stopped and not p.done:
            line.append("  defaults", style=_GLOW)
        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(width=1, no_wrap=True)
        table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        table.add_column(no_wrap=True, justify="right")
        marker = Text(_spinner(stage, now), style=f"bold {color}")
        table.add_row(marker, line, Text(_duration(self._elapsed()), style=_DIM))
        return table

    def close(self, *, failed=False):
        if self._closed:
            return
        if failed:
            self._frame = replace(self._frame, failed=True)
        if self._started is not None and self._ended is None:
            self._ended = time.monotonic()
        try:
            if self._live is not None:
                self._live.stop()
            elif self._enabled and failed and self._started is not None:
                self._write_milestone()
        finally:
            self._closed = True

    def __exit__(self, kind, value, traceback):
        self.close(failed=kind is not None)
        return False
