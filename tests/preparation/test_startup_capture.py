"""Prepared launchers remain valid after compiler caches release their copies."""

import gc

import torch

from b12x._lib import compiler
from b12x.preparation import PreparationSession, PreparedCall
from ..conftest import require_b12x


def test_graph_replay_survives_compiler_cache_eviction(monkeypatch):
    from b12x.gemm import bf16_gemv

    device = require_b12x()
    source = torch.randn(2, 2048, device=device, dtype=torch.bfloat16)
    weight = torch.randn(96, 2048, device=device, dtype=torch.bfloat16)
    plan = bf16_gemv.plan(bf16_gemv.query_from_call(source, weight))
    request = plan.request(
        name="gemv",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(source, weight)
        ),
    )

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare((request,))
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                output = bf16_gemv.mm(source, weight, plan=plan)

            compiler.clear_compile_cache()
            gc.collect()

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    "prepared graph replay reached compiler or loader"
                )

            monkeypatch.setattr(compiler, "compile", forbidden)
            monkeypatch.setattr(
                compiler, "_load_cute_compile_from_disk", forbidden
            )
            changed = torch.randn_like(source)
            source.copy_(changed)
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)

            expected = changed.float() @ weight.float().T
            assert torch.isfinite(output).all()
            assert torch.count_nonzero(output) > 0
            torch.testing.assert_close(
                output, expected.bfloat16(), rtol=2e-2, atol=2e-2
            )
        finally:
            graph.reset()
