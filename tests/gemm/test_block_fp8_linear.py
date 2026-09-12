"""gemm.block_fp8_linear: quantized-reference parity + the planned lifecycle
(plan -> bind -> run) replaying under CUDA-graph capture.

Curated from b12x tests/test_gemm_block_fp8_linear.py; kernel-reuse and
regime-policy tests stay in the b12x repo.
"""

from __future__ import annotations

import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.gemm import block_fp8_linear as bfl
from b12x.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
)
from tests.gemm.test_gemm_block_fp8_linear import (
    _assert_v41_accumulation_matches_reference,
    _make_block_fp8_weight as _make_v41_block_fp8_weight,
)

from ..conftest import require_b12x


def _make_block_fp8_weight(
    out_features: int, in_features: int
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        / 8
    ).to(torch.float8_e4m3fn)
    scale_u8 = (
        torch.arange(
            (out_features // 128) * (in_features // 128),
            device="cuda",
            dtype=torch.int32,
        )
        % 3
        + 126
    ).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).reshape(
        out_features // 128, in_features // 128
    )
    return weight, scale


def _reference(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
    x_q = bfl.quantize_input(x)
    w_q = bfl.pack_weight(weight, scale)
    x_deq = dequantize_mxfp8_rows_torch(x_q.values, x_q.scale_rows)
    w_deq = dequantize_mxfp8_rows_torch(w_q.weight.values, w_q.weight.scale_rows)
    return x_deq @ w_deq.T


def test_run_matches_quantized_reference() -> None:
    require_b12x()
    torch.manual_seed(20260523)

    tokens, in_features, out_features = 7, 256, 384
    x = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = bfl.pack_weight(weight, scale)

    actual = bfl.run(x, packed)
    expected = _reference(x, weight, scale)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.float(), expected.to(actual.dtype).float(), rtol=0, atol=0
    )


def test_plan_bind_run_replays_under_cuda_graph() -> None:
    require_b12x()
    torch.manual_seed(20260526)

    tokens, in_features, out_features = 1, 128, 256
    x = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight, scale = _make_block_fp8_weight(out_features, in_features)
    packed = bfl.pack_weight(weight, scale)

    plan = bfl.plan(
        bfl.Caps(
            device=x.device,
            max_tokens=tokens,
            in_features=in_features,
            out_features=out_features,
            output_dtype=x.dtype,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=x.device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    output = torch.empty((tokens, out_features, 1), dtype=x.dtype, device=x.device)
    binding = bfl.bind(
        plan, scratch=scratch, source=x, packed_weight=packed, output=output
    )

    def run_once() -> torch.Tensor:
        return bfl.run(binding=binding)

    eager = run_once().clone()
    torch.cuda.synchronize()

    run_once()  # warm before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run_once()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, eager, rtol=0, atol=0)


def test_v41_k576_padding_preserves_bound_and_functional_graph_replay() -> None:
    """A logical K576 V4.1 projection uses its zero-padded K640 native operand."""
    require_b12x()
    torch.manual_seed(20260911)

    capacity, in_features, out_features = 8, 576, 160
    source = torch.randn(
        (capacity, in_features), device="cuda", dtype=torch.bfloat16
    ).mul_(0.25)
    weight, scale = _make_v41_block_fp8_weight(
        out_features, in_features, block_size=32
    )
    packed = bfl.pack_weight(weight, scale, block_size=(32, 32))
    plan = bfl.plan(
        bfl.Caps(
            device=source.device,
            max_tokens=capacity,
            in_features=in_features,
            out_features=out_features,
            output_dtype=source.dtype,
            block_size=(32, 32),
        )
    )
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    output = torch.empty(
        (capacity, out_features, 1), dtype=source.dtype, device=source.device
    )
    pointers = (source.data_ptr(), scratch.data_ptr(), output.data_ptr())

    # Prewarm both public forms before freezing resolution.  The scale grid is
    # deliberately nonuniform, so an invalid padded K32 group cannot borrow a
    # neighboring scale without changing the independent K32 oracle below.
    bfl.prewarm(packed, (1,), output_dtype=source.dtype, expected_m=capacity)
    freeze_kernel_resolution("V4.1 K576/K640 tail padding")
    try:
        for rows in (1, 3, 7):
            binding = bfl.bind(
                plan,
                scratch=scratch,
                source=source[:rows],
                packed_weight=packed,
                output=output[:rows],
            )
            for bound in (True, False):
                def run() -> torch.Tensor:
                    if bound:
                        return bfl.run(binding=binding)
                    return bfl.run(source[:rows], packed, expected_m=capacity)

                # Bind/run owns poisoned caller scratch; the functional form
                # independently exercises its allocated native K640 path.
                run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = run()

                source.normal_().mul_(0.25)
                source[:, :32].mul_(1e-5)
                scratch.fill_(255)
                actual.fill_(float("nan"))
                actual_ptr = actual.data_ptr()
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()

                for key in (
                    "allocation.all.allocated",
                    "allocated_bytes.all.allocated",
                ):
                    assert before[key] == after[key]
                assert actual.data_ptr() == actual_ptr
                assert pointers == (
                    source.data_ptr(),
                    scratch.data_ptr(),
                    output.data_ptr(),
                )
                assert torch.isfinite(actual).all()
                assert torch.count_nonzero(actual) > 0
                _assert_v41_accumulation_matches_reference(
                    source[:rows], weight, scale, actual
                )
    finally:
        unfreeze_kernel_resolution()
