"""Prepared WO projection numerical behavior."""
from __future__ import annotations

from contextlib import ExitStack

import pytest
import torch

from b12x.gemm import wo_projection as wo
from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch, quantize_wo_projection_weights_mxfp8_torch
from b12x.preparation import PreparedCall, PreparationSession
from ..conftest import require_b12x


def _prepared(caps, source, weights):
    resources = ExitStack()
    specs = ()
    declaration = wo.plan(caps)

    def prepare(state):
        nonlocal specs
        specs = state._scratch_state.scratch_specs()
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=caps.device) for spec in specs)
        binding = state.bind(scratch=scratch, source_tgd=source, weights=weights)
        return PreparedCall(run=lambda: state.run(binding))

    session = resources.enter_context(PreparationSession(device=caps.device, autotune=False, compile_workers=2))
    resources.enter_context(session.prepare((declaration.request(
        name="wo", prepare_call=prepare,
    ),)))
    plan = declaration
    scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=caps.device) for spec in specs)
    return resources, plan, wo.bind(plan, scratch=scratch, source_tgd=source, weights=weights)


def test_plan_bind_run_singleton_group_matches_quantized_reference() -> None:
    require_b12x()
    torch.manual_seed(31005)
    tokens, groups, group_width, rank, hidden = 3, 1, 512, 128, 128
    source = torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16) / 4
    wo_a = torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16) / group_width**0.5
    wo_b = torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16) / (groups * rank) ** 0.5
    weights = quantize_wo_projection_weights_mxfp8_torch(wo_a, wo_b)
    caps = wo.Caps(device=source.device, max_tokens=tokens, groups=groups, group_width=group_width, rank=rank, hidden=hidden)
    resources, plan, binding = _prepared(caps, source, weights)
    with resources:
        actual = wo.run(binding=binding, plan=plan)
        x = wo.quantize_input(source, plan=plan)
        tmp = (dequantize_mxfp8_rows_torch(x.values, x.scale_rows) @ dequantize_mxfp8_rows_torch(weights.wo_a.values, weights.wo_a.scale_rows).T).to(torch.bfloat16).unsqueeze(-1)
        tmp_q = wo.quantize_input_b(tmp, plan=plan)
        expected = dequantize_mxfp8_rows_torch(tmp_q.values, tmp_q.scale_rows) @ dequantize_mxfp8_rows_torch(weights.wo_b.values, weights.wo_b.scale_rows).T
        torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)

def test_inverse_rope_prepared_execution_rejects_mismatched_runtime_pointer_dtype() -> None:
    require_b12x()
    tokens, groups, heads_per_group, nope_dim, rope_dim, rank, hidden = 1, 1, 1, 96, 32, 128, 128
    o = torch.randn((tokens, groups * heads_per_group, nope_dim + rope_dim), device="cuda", dtype=torch.bfloat16)
    positions = torch.zeros((tokens,), device="cuda", dtype=torch.int64)
    cos_sin_cache = torch.randn((4, rope_dim), device="cuda", dtype=torch.bfloat16)
    weights = quantize_wo_projection_weights_mxfp8_torch(
        torch.randn((groups, rank, heads_per_group * (nope_dim + rope_dim)), device="cuda", dtype=torch.bfloat16),
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16),
    )
    caps = wo.Caps(
        device=o.device, max_tokens=tokens, groups=groups, group_width=heads_per_group * (nope_dim + rope_dim),
        rank=rank, hidden=hidden,
    )
    declaration = wo.plan(
        caps,
        invocation={"operation": "inv_rope", "heads_per_group": heads_per_group,
                    "nope_dim": nope_dim, "rope_dim": rope_dim},
    )
    resources = ExitStack()
    specs = ()

    def prepare(state):
        nonlocal specs
        specs = state._scratch_state.scratch_specs()
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=o.device) for spec in specs)
        binding = state.bind_inv_rope(
            scratch=scratch, o=o, positions=positions, cos_sin_cache=cos_sin_cache, weights=weights,
            heads_per_group=heads_per_group, nope_dim=nope_dim, rope_dim=rope_dim,
        )
        return PreparedCall(run=lambda: state.run_inv_rope(binding))

    session = resources.enter_context(PreparationSession(device=o.device, autotune=False, compile_workers=2))
    resources.enter_context(session.prepare((declaration.request(
        name="wo-inv", prepare_call=prepare,
    ),)))
    scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=o.device) for spec in specs)
    with resources:
        plan = declaration
        with pytest.raises(ValueError, match="dtypes differ"):
            wo.bind_inv_rope(
                plan, scratch=scratch, o=o, positions=positions.to(torch.int32),
                cos_sin_cache=cos_sin_cache, weights=weights, heads_per_group=heads_per_group,
                nope_dim=nope_dim, rope_dim=rope_dim,
            )


