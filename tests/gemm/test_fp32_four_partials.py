"""Prepared four-partial FP32 projection reduction with dynamic rows."""

import pytest
import torch

from b12x.preparation import PreparationSession, PreparedCall
from b12x.gemm import block_fp8_linear as bfl
from tests._reference.helpers import require_b12x
from tests.gemm.test_gemm_block_fp8_linear import (
    _assert_v41_accumulation_matches_reference,
    _make_block_fp8_weight,
)


@pytest.mark.parametrize("n", (1152, 1792))
@pytest.mark.parametrize("capacity", (2, 4, 8))
def test_fp32_four_partials_frozen_dynamic_rows(n, capacity, monkeypatch):
    """Every live slice is written; source mutation does not allocate on replay."""
    import b12x._lib.dense_gemm as dense_module

    require_b12x()
    monkeypatch.setattr(dense_module, "_B12X_DENSE_SPLITK_TURBO", True)
    torch.manual_seed(42313 + n + capacity)
    device = torch.device("cuda", torch.cuda.current_device())
    k = 5120
    source = torch.randn((capacity, k), dtype=torch.bfloat16, device=device)
    weight, scales = _make_block_fp8_weight(n, k, block_size=32)
    packed = bfl.pack_weight(weight, scales, block_size=(32, 32))
    config = bfl.BlockFp8LinearConfig(backend="mxfp8_split4_fp32", tile_m=16, tile_n=64)
    plan = bfl.plan(bfl.Caps(device=device, max_tokens=capacity, in_features=k,
                            out_features=n, block_size=(32, 32)), override=config)
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    output = torch.empty((capacity, n, 1), dtype=torch.bfloat16, device=device)

    def prepare(state):
        spec, = state.scratch.scratch_specs()
        trial_scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        trial_output = torch.empty_like(output)
        binding = state.bind(scratch=trial_scratch, source=source, packed_weight=packed, output=trial_output)
        return PreparedCall(run=lambda: state.run_binding(binding), owners=(trial_scratch, trial_output, binding))

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name="four-partials", prepare_call=prepare),))
        full = bfl.bind(plan, scratch=scratch, source=source, packed_weight=packed, output=output)
        partials = full.workspace.view(4, capacity, n)
        pointers = tuple(t.data_ptr() for t in (source, scratch, partials, output))
        session.freeze()
        for rows in sorted({1, max(1, capacity - 1), capacity}):
            binding = bfl.bind(plan, scratch=scratch, source=source[:rows],
                               packed_weight=packed, output=output[:rows])
            run = binding.run
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for _ in range(3):
                source.normal_()
                source[:, :32].mul_(1e-5)
                scratch.fill_(255)
                partials.fill_(float("nan"))
                output.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                assert pointers == tuple(t.data_ptr() for t in (source, scratch, partials, output))
                assert torch.isfinite(partials.view(-1)[:4 * rows * n]).all()
                assert torch.isnan(partials.view(-1)[4 * rows * n:]).all()
                _assert_v41_accumulation_matches_reference(
                    source[:rows], weight, scales, output[:rows, :, 0],
                )
            graph.reset()
