"""Host state-machine boundaries, without substituting a serving kernel."""
import gc
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.preparation import (
    CollectiveRequirement, DetectedDevice, MemoryRequirements,
    PersistentMemory, Plan, PreparationSession, PreparedCall, current_plan,
    plan_from_handle, require_prepared,
)
from b12x.preparation._cache import SelectionCache
from b12x.preparation.types import _CompositePlan
from b12x._lib.scratch import ScratchBufferSpec
from b12x._lib.runtime_control import KernelResolutionFrozenError, kernel_resolution_guard
from .test_defaults import Config, Query, contract


def session(tmp_path, **kwargs):
    value = PreparationSession(device=DetectedDevice(None, None), **kwargs)
    value._cache = SelectionCache(tmp_path, {"schema_version": 4})
    return value


def declaration(*, tuning=None, pin=None, shared=False):
    tuning = contract(values=(2,)) if tuning is None else tuning
    return Plan(
        contract=tuning, query=Query(3), override=pin, shared=shared,
        _compile_jobs=lambda config, device: (),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=lambda selection, device: SimpleNamespace(value=selection.config.width * 3),
    )


def request(*, name, tuning=None, pin=None, calls=None, close=None, benchmark=None, dependencies=(), collective=None, shared=False):
    calls = [] if calls is None else calls
    return declaration(tuning=tuning, pin=pin, shared=shared).request(
        name=name,
        prepare_call=lambda state: PreparedCall(run=lambda: calls.append(state.value), close=close),
        benchmark_call=benchmark, dependencies=dependencies, collective=collective,
    )


def test_prepare_fills_plans_in_place_and_release_runs_closers(tmp_path):
    calls, closed = [], []
    a = request(name="a", calls=calls, close=lambda: closed.append("a"))
    b = request(name="b", calls=calls, close=lambda: closed.append("b"))
    engine = session(tmp_path)
    result = engine.prepare((a, b))
    assert calls == [6, 6]
    assert result.plans["a"] is a.plan
    assert require_prepared(a.plan, "test.arithmetic").value == 6
    assert a.plan.selection.source == "fixed"
    engine.freeze()
    engine.prepare((a, b))
    assert calls == [6, 6]
    engine.release(a.plan)
    assert closed == ["a"]
    assert a.plan.prepared is None
    with pytest.raises(RuntimeError, match="frozen"):
        require_prepared(a.plan, "test.arithmetic")
    engine.close()
    assert sorted(closed) == ["a", "b"]
    assert b.plan.prepared is None


def test_plan_scoped_persistent_owners_reserve_independent_buffers(tmp_path):
    buffers = {}

    def memory(config, device):
        return MemoryRequirements(persistent=(
            PersistentMemory(("scratch-owner", current_plan()), 16),
            PersistentMemory("shared-readonly", 8, 8),
        ))

    def materialize(selection, device):
        buffers[current_plan()] = bytearray(16)
        return SimpleNamespace(buffer=buffers[current_plan()])

    def make(name):
        plan = Plan(
            contract=contract(values=(2,)), query=Query(3),
            _compile_jobs=lambda config, device: (),
            _memory_requirements=memory, _materialize=materialize,
        )
        return plan.request(
            name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.buffer.__setitem__(0, 1)),
        )

    requests = (make("target"), make("draft"))
    with session(tmp_path) as engine:
        assert engine.candidate_memory_envelope(requests).pending_persistent_nbytes == 32
        job = engine.begin(requests)
        while True:
            progress = job.advance()
            with pytest.raises(RuntimeError):
                current_plan()
            if progress.done:
                break
        job.result()
        target = require_prepared(requests[0].plan, "test.arithmetic").buffer
        draft = require_prepared(requests[1].plan, "test.arithmetic").buffer
        target[0] = 9
        assert draft[0] == 1


def test_sticky_stop_before_enumeration_prepares_default_without_winner(tmp_path):
    def no_optional(query, device, assignment):
        raise AssertionError("stopped session enumerated an optional candidate")

    tuning = replace(contract(), materialize=no_optional)
    calls = []
    with session(tmp_path) as engine:
        engine.cancel_tuning()
        result = engine.prepare((request(name="a", tuning=tuning, calls=calls),))
        assert result.selections["a"].source == "default"
        result = engine.prepare((request(name="b", tuning=tuning, calls=calls),))
        assert result.selections["b"].source == "default"
        assert engine._cache.records == {}
    assert calls == [21, 21]


def test_cache_only_effective_singleton_and_explicit_pin_need_no_selection_record(tmp_path):
    tuning = replace(contract(), equivalence_key=lambda query, device, config: {"same": True})
    calls = []
    with session(tmp_path, cache_only=True) as engine:
        result = engine.prepare((request(name="singleton", tuning=tuning, calls=calls),))
        assert result.selections["singleton"].source == "fixed"
        result = engine.prepare((request(name="pinned", tuning=contract(), pin=Config(9), calls=calls),))
        assert result.selections["pinned"].source == "override"
        with pytest.raises(LookupError):
            engine.prepare((request(name="missing", tuning=contract()),))
    assert calls == [3, 27]


def test_collective_requires_explicit_matching_authorization(tmp_path):
    calls = []
    requirement = CollectiveRequirement("group/prime", (0, 1))
    req = request(name="collective", calls=calls, collective=requirement)
    with session(tmp_path) as engine:
        with pytest.raises(ValueError):
            engine.prepare((req,))
        job = engine.begin((req,))
        progress = job.advance()
        assert progress.ready_collectives == (requirement,)
        assert calls == []
        progress = job.advance(collective_key="wrong/group")
        assert progress.ready_collectives == (requirement,)
        assert calls == []
        progress = job.advance(collective_key=requirement.key)
        while not progress.done:
            progress = job.advance()
        assert calls == [6]
        job.result()


def test_new_obligation_fails_after_freeze(tmp_path):
    with session(tmp_path) as engine:
        engine.prepare((request(name="ready"),))
        engine.freeze()
        with pytest.raises(KernelResolutionFrozenError):
            engine.prepare((request(name="not-ready"),))


def test_second_prepare_is_incremental_for_prepared_plans(tmp_path):
    calls = []
    a, b = request(name="a", calls=calls), request(name="b", calls=calls)
    with session(tmp_path) as engine:
        engine.prepare((a, b))
        first = a.plan.prepared
        c = request(name="c", calls=calls)
        engine.prepare((a, b, c))
        assert calls == [6, 6, 6]
        assert a.plan.prepared is first
        assert c.plan.prepared is not None


def test_failure_restores_and_closes_all_while_preserving_primary_error(tmp_path):
    restored = []

    def factory(state):
        def fail():
            raise ValueError("primary failure")

        def restore():
            restored.append("restore")
            raise RuntimeError("cleanup failure")

        return PreparedCall(run=fail, restore=restore, close=lambda: restored.append("close"))

    req = request(name="failed")
    req = replace(req, prepare_call=factory)
    with session(tmp_path) as engine:
        with pytest.raises(ValueError, match="primary failure"):
            engine.prepare((req,))
    assert restored == ["restore", "close"]
    assert req.plan.prepared is None


def test_persistent_memory_counts_shared_keys_once_and_rejects_conflicts():
    requirements = MemoryRequirements(persistent=(
        PersistentMemory("weights-side-state", 40, 10),
        PersistentMemory("weights-side-state", 40, 10),
    ))
    assert requirements.pending_persistent_nbytes == 30
    assert MemoryRequirements(persistent=(PersistentMemory("state", 40, 40),)).pending_persistent_nbytes == 0
    with pytest.raises(ValueError):
        MemoryRequirements.sequential((requirements, MemoryRequirements(persistent=(
            PersistentMemory("weights-side-state", 40, 20),
        ))))


def _deterministic_timer(monkeypatch, *, stop=None, batches=None):
    from b12x.preparation import _measurement

    def prepare(calls, **kwargs):
        if batches is not None:
            batches.append(tuple(call.output for call in calls))
        yield
        return SimpleNamespace(calls=calls, close=lambda: None, completed_rounds=0,
                               planned_rounds=0, active_count=len(calls), latest_round_us=())

    def measure(race, **kwargs):
        if stop is not None:
            stop()
        yield
        return _measurement.RaceMeasurements(
            tuple(abs(call.output - 6) + 1 for call in race.calls), 0,
        )

    monkeypatch.setattr(_measurement, "prepare_race_steps", prepare)
    monkeypatch.setattr(_measurement, "measure_race_steps", measure)


def test_complete_race_cached_restart_and_disabled_precedence(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)
    trial_closed, calls = [], []

    def benchmark(state):
        return PreparedCall(
            run=lambda: state.value, produce=lambda: None,
            close=lambda: trial_closed.append(state.value),
        )

    with session(tmp_path) as engine:
        result = engine.prepare((request(name="first", tuning=contract(), calls=calls, benchmark=benchmark),))
        assert result.selections["first"].source == "tuned"
        assert result.selections["first"].config.width == 2
        assert result.benchmarked_candidates == 3
        assert sorted(trial_closed) == [3, 6, 12]
    with session(tmp_path, autotune=False) as engine:
        result = engine.prepare((request(name="different-owner", tuning=contract(), calls=calls),))
        assert result.selections["different-owner"].source == "cached"
        result = engine.prepare((request(name="pin", tuning=contract(), pin=Config(9), calls=calls),))
        assert result.selections["pin"].source == "override"
    assert calls == [6, 6, 27]


def test_race_batches_bound_residency_and_carry_the_champion(tmp_path, monkeypatch):
    batches, trial_closed = [], []
    _deterministic_timer(monkeypatch, batches=batches)

    def benchmark(state):
        return PreparedCall(
            run=lambda: state.value, produce=lambda: None,
            close=lambda: trial_closed.append(state.value),
        )

    tuning = contract(values=(1, 2, 4, 8))
    with session(tmp_path, race_batch=2) as engine:
        result = engine.prepare((request(name="batched", tuning=tuning, benchmark=benchmark),))
        assert result.selections["batched"].config.width == 2
        assert result.benchmarked_candidates == 4
        assert result.coverage["batched"]["measured_count"] == 4
    assert batches == [(3, 6), (6, 12, 24)]
    assert sorted(trial_closed) == [3, 6, 12, 24]


def test_two_ranks_measure_disjoint_candidate_halves_and_install_global_winner(
    tmp_path, monkeypatch
):
    _deterministic_timer(monkeypatch)
    tuning = contract(values=(1, 2, 4, 8))
    engines = [session(tmp_path / f"rank-{rank}") for rank in range(2)]
    requests = []
    jobs = []
    for rank, engine in enumerate(engines):
        engine.configure_tuning_shard(rank, (0, 1))
        req = request(
            name="shared",
            tuning=tuning,
            benchmark=lambda state: PreparedCall(
                run=lambda: state.value,
                produce=lambda: None,
            ),
        )
        requests.append(req)
        jobs.append(engine.begin((req,)))

    authorizations = [None, None]
    progress = [None, None]
    for _ in range(100):
        progress = [
            job.advance(tuning=authorization)
            for job, authorization in zip(jobs, authorizations)
        ]
        authorizations = [None, None]
        contributions = [
            item
            for state in progress
            for item in state.ready_tuning
        ]
        if contributions:
            assert len(contributions) == 2
            assert {item.candidate_index for item in contributions} == {0, 1}
            winner = min(
                contributions,
                key=lambda item: (item.latency_us, item.candidate_index),
            )
            authorizations = [winner, winner]
        if all(state.done for state in progress):
            break
    else:
        pytest.fail("distributed tuning did not complete")

    results = [job.result() for job in jobs]
    try:
        assert [result.benchmarked_candidates for result in results] == [2, 2]
        assert [result.selections["shared"].config.width for result in results] == [2, 2]
        assert [result.coverage["shared"]["measured_count"] for result in results] == [4, 4]
        assert [req.plan.selection.config.width for req in requests] == [2, 2]
    finally:
        for engine in engines:
            engine.close()


def test_job_autotune_override_upgrades_live_default_and_reuses_winner(
    tmp_path, monkeypatch
):
    _deterministic_timer(monkeypatch)
    calls = []

    def benchmark(state):
        return PreparedCall(
            run=lambda: state.value,
            produce=lambda: None,
        )

    with session(tmp_path) as engine:
        early = request(
            name="early",
            tuning=contract(),
            calls=calls,
            benchmark=benchmark,
        )
        result = engine.prepare((early,), autotune=False)
        assert result.selections["early"].source == "default"
        assert early.plan.selection.source == "default"

        result = engine.prepare((early,))
        assert result.selections["early"].source == "tuned"
        assert result.selections["early"].config.width == 2
        assert early.plan.selection.source == "tuned"

        result = engine.prepare((early,))
        assert result.selections["early"].source == "tuned"

    assert calls == [21, 6]


def test_candidate_memory_envelope_covers_every_legal_config(tmp_path):
    tuning = contract(default=1, values=(256, 512, 1024))
    plan = Plan(
        contract=tuning,
        query=Query(3),
        _compile_jobs=lambda config, device: (),
        _memory_requirements=lambda config, device: MemoryRequirements(
            scratch=(
                ScratchBufferSpec(
                    name="candidate",
                    shape=(config.width,),
                    dtype=torch.uint8,
                    device=torch.device("cpu"),
                ),
            )
        ),
        _materialize=lambda selection, device: object(),
    )
    candidate = plan.request(
        name="candidate",
        prepare_call=lambda state: PreparedCall(run=lambda: None),
    )

    with session(tmp_path) as engine:
        envelope = engine.candidate_memory_envelope((candidate,))

    assert envelope.scratch[0].shape == (1024,)
    assert plan.scratch_specs()[0].shape == (1,)


def test_stop_mid_race_discards_partial_winner_and_restores_trials(tmp_path, monkeypatch):
    calls, restored = [], []
    with session(tmp_path) as engine:
        _deterministic_timer(monkeypatch, stop=engine.cancel_tuning)

        def benchmark(state):
            return PreparedCall(
                run=lambda: state.value, produce=lambda: None,
                restore=lambda: restored.append(state.value),
            )

        result = engine.prepare((request(name="stopped", tuning=contract(), calls=calls, benchmark=benchmark),))
        assert result.selections["stopped"].source == "default"
        assert result.benchmarked_candidates == 0
        assert engine._cache.records == {}
    assert calls == [21]
    assert sorted(restored) == [3, 6, 12]


def test_equal_declarations_enumerate_their_candidates_once(tmp_path, monkeypatch):
    from b12x.preparation.tuning import TuningContract

    enumerated = []
    original = TuningContract.iterate

    def counting(self, configuration):
        enumerated.append(configuration.encoded_query)
        return original(self, configuration)

    monkeypatch.setattr(TuningContract, "iterate", counting)
    _deterministic_timer(monkeypatch)
    tuning = contract()

    def benchmark(state):
        return PreparedCall(run=lambda: state.value, produce=lambda: None)

    def duplicate(name, rows):
        plan = Plan(
            contract=tuning, query=Query(rows),
            _compile_jobs=lambda config, device: (),
            _memory_requirements=lambda config, device: MemoryRequirements(),
            _materialize=lambda selection, device: SimpleNamespace(value=selection.config.width),
        )
        return plan.request(
            name=name, prepare_call=lambda state: PreparedCall(run=lambda: None),
            benchmark_call=benchmark,
        )

    with session(tmp_path) as engine:
        result = engine.prepare((
            duplicate("first", 3), duplicate("second", 3), duplicate("wider", 5),
        ))
    assert [query["rows"] for query in enumerated] == [3, 5]
    assert result.coverage["first"]["effective_count"] == 3
    assert result.coverage["second"]["effective_count"] == 3
    assert result.selections["second"].config == result.selections["first"].config


def test_duplicate_choice_dependency_order_still_prepares_both_plans(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)
    calls = []
    producer = request(name="producer")

    def benchmark(state):
        return PreparedCall(run=lambda: state.value, produce=lambda: None)

    a = request(name="a", tuning=contract(), calls=calls,
                benchmark=benchmark, dependencies=("producer",))
    b = request(name="b", tuning=contract(), calls=calls,
                benchmark=benchmark, dependencies=("producer",))
    with session(tmp_path) as engine:
        result = engine.prepare((b, a, producer))
        assert result.benchmarked_candidates == 3
        assert a.plan.prepared is not b.plan.prepared
    assert calls == [6, 6]


def test_shared_declarations_alias_one_prepared_state(tmp_path):
    calls, closed = [], []
    a = request(name="a", calls=calls, shared=True, close=lambda: closed.append("closed"))
    b = request(name="b", calls=calls, shared=True, close=lambda: closed.append("closed"))
    c = request(name="c", calls=calls, shared=True)
    with session(tmp_path) as engine:
        engine.prepare((a, b))
        assert calls == [6]
        assert a.plan.prepared is b.plan.prepared
        engine.prepare((c,))
        assert calls == [6]
        assert c.plan.prepared is a.plan.prepared
        engine.release(a.plan)
        assert a.plan.prepared is None
        assert b.plan.prepared is not None
        assert closed == []
        engine.release(b.plan)
        engine.release(c.plan)
        assert closed == ["closed"]
    with pytest.raises(ValueError, match="shared plan"):
        with session(tmp_path) as engine:
            engine.prepare((request(name="d", shared=True, dependencies=("e",)), request(name="e")))


def test_composite_prepares_children_and_assembles_their_states(tmp_path):
    assembled = []

    def child(width):
        return Plan(
            contract=contract(values=(width,)), query=Query(3),
            _compile_jobs=lambda config, device: (),
            _memory_requirements=lambda config, device: MemoryRequirements(),
            _materialize=lambda selection, device: SimpleNamespace(value=selection.config.width * 3),
        )

    children = {1: child(1), 2: child(2)}
    root = _CompositePlan(
        component_id="test.arithmetic", capacity_metadata={}, variants=children,
        _assemble=lambda states, device: assembled.append(dict(states)) or SimpleNamespace(states=dict(states)),
    )
    calls = []
    req = root.request(
        name="root",
        prepare_calls={count: (lambda state: PreparedCall(run=lambda: calls.append(state.value))) for count in (1, 2)},
    )
    with session(tmp_path) as engine:
        result = engine.prepare((req,))
        assert result.plans["root"] is root
        assert sorted(calls) == [3, 6]
        state = require_prepared(root, "test.arithmetic")
        assert state.states == {1: children[1].prepared.state, 2: children[2].prepared.state}
        assert root.prepared.variants[2] is children[2]
        assert root.token_counts == (1, 2)
        engine.prepare((req,))
        assert len(assembled) == 1
        engine.release(root)
        assert root.prepared is None and children[1].prepared is None


def test_plan_handles_are_stable_and_resolve_only_live_plans():
    req = request(name="handle")
    handle = req.plan.handle
    assert plan_from_handle(handle) is req.plan
    with pytest.raises(RuntimeError):
        plan_from_handle(handle + 1_000_000)
    del req
    gc.collect()
    with pytest.raises(RuntimeError):
        plan_from_handle(handle)


def test_unprepared_plan_materializes_its_default_with_a_warning_before_freeze_only(
    tmp_path, caplog,
):
    calls = []
    req = request(name="lazy", calls=calls)
    with caplog.at_level("WARNING", logger="b12x"):
        state = require_prepared(req.plan, "test.arithmetic")
    assert state.value == 6
    assert req.plan.selection.source == "fixed"
    assert calls == []
    messages = [r.getMessage() for r in caplog.records if "not prepared before its first use" in r.getMessage()]
    assert len(messages) == 1 and "test.arithmetic" in messages[0]
    with pytest.raises(ValueError, match="belongs to"):
        require_prepared(req.plan, "test.other")
    other = request(name="later")
    with session(tmp_path) as engine:
        engine.prepare((request(name="ready"),))
        engine.freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            require_prepared(other.plan, "test.arithmetic")


def test_priming_closures_release_transients_and_restore_before_readiness(tmp_path):
    import weakref

    class Temporary:
        pass

    references, lifetime = [], []

    def factory(state):
        source = Temporary()
        references.append(weakref.ref(source))

        def run():
            assert source is not None
            output = Temporary()
            references.append(weakref.ref(output))
            return output

        return PreparedCall(
            run=run, restore=lambda: lifetime.append("restored"),
            close=lambda: lifetime.append("resources closed"),
        )

    req = replace(request(name="temporary"), prepare_call=factory)
    with session(tmp_path) as engine:
        engine.prepare((req,))
        gc.collect()
        assert lifetime == ["restored"]
        assert all(reference() is None for reference in references)
    assert lifetime == ["restored", "resources closed"]


def test_prepared_resources_remain_mutable_after_inference_mode_priming(tmp_path):
    observed = {}

    def materialize(selection, device):
        observed["materialize_inference"] = torch.is_inference_mode_enabled()
        return SimpleNamespace(buffer=torch.zeros(1))

    def factory(state):
        observed["factory_inference"] = torch.is_inference_mode_enabled()
        observed["buffer"] = state.buffer

        def produce():
            observed["prime_inference"] = torch.is_inference_mode_enabled()
            state.buffer.fill_(1)

        return PreparedCall(
            run=lambda: state.buffer.add_(1),
            produce=produce,
            owners=(state.buffer,),
        )

    plan = Plan(
        contract=contract(values=(2,)),
        query=Query(3),
        _compile_jobs=lambda config, device: (),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize,
    )
    prepared = plan.request(
        name="inference-profile",
        prepare_call=factory,
    )

    with session(tmp_path) as engine, torch.inference_mode():
        engine.prepare((prepared,))
        assert observed["buffer"].item() == 2

    assert observed == {
        "materialize_inference": False,
        "factory_inference": False,
        "buffer": observed["buffer"],
        "prime_inference": True,
    }
    assert not observed["buffer"].is_inference()
    observed["buffer"].add_(1)
    assert observed["buffer"].item() == 3


def test_frozen_reuse_primes_independent_benchmark_trial(tmp_path):
    production, trials, closed = [], [], []

    def benchmark(state):
        token = object()
        return PreparedCall(
            run=lambda: trials.append((token, state.value)),
            close=lambda: closed.append(token),
        )

    req = replace(
        request(name="retained", calls=production, benchmark=benchmark),
        retain_benchmark_call=True,
    )
    with session(tmp_path) as engine:
        with engine.prepare((req,)) as first:
            first.benchmark_calls["retained"].invoke()
        assert len(closed) == 1
        engine.freeze()
        with engine.prepare((req,)) as second:
            second.benchmark_calls["retained"].invoke()
            assert production == [6]
            assert [value for _, value in trials] == [6, 6, 6, 6]
            assert trials[0][0] is not trials[2][0]
            assert len(closed) == 1
        assert len(closed) == 2


def test_frozen_reuse_rejects_changed_declared_device(tmp_path):
    req = request(name="device")
    with session(tmp_path) as engine:
        engine.prepare((req,))
        engine.freeze()
        changed = replace(req, plan=replace(req.plan, _device="cuda:1"))
        with pytest.raises(KernelResolutionFrozenError):
            engine.prepare((changed,))



def test_plan_state_omits_the_prepared_payload(tmp_path):
    """Compiler caches serialize closed-over plans; only the declaration travels."""
    req = request(name="pickled")
    plan = req.plan
    prepared_session = session(tmp_path, autotune=False)
    prepared_session.prepare((req,))
    assert plan.prepared is not None
    state = plan.__getstate__()
    assert state["_prepared"] is None
    copy = object.__new__(type(plan))
    copy.__setstate__(state)
    assert copy.prepared is None
    assert copy.handle == plan.handle
    assert copy.query == plan.query
    assert plan_from_handle(plan.handle) is plan
    assert plan.prepared is not None
    prepared_session.close()


def test_prepare_default_primes_with_the_request_call_and_refuses_after_freeze_or_under_capture(
    tmp_path, monkeypatch,
):
    import torch

    from b12x.preparation import prepare_default
    from b12x.preparation import session as session_module

    monkeypatch.setitem(session_module._LAZY_SESSIONS, None, session(tmp_path, autotune=False))
    calls = []
    req = request(name="on-demand", tuning=contract(values=(1, 2, 4)), calls=calls)
    prepared = prepare_default(req)
    assert prepared is req.plan.prepared and prepared is not None
    assert calls == [prepared.selection.config.width * 3]
    assert req.plan.selection.source == "default"
    with kernel_resolution_guard("frozen"), pytest.raises(RuntimeError, match="frozen"):
        prepare_default(request(name="late"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="under CUDA graph capture"):
        prepare_default(request(name="captured"))
