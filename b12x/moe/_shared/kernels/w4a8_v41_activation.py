"""Fused V4.1 FC1 boundary activation and MXFP8 materialization."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint32

from b12x._lib.intrinsics import (
    FLOAT8_E4M3_MAX,
    cvt_f32x4_to_e4m3x4,
    div_rn_f32,
    fabs_f32,
    fmax_f32,
    pow2_ceil_ue8m0,
    ue8m0_to_output_scale,
)


class V41MicroActivationKernel:
    """Apply routed SwiGLU to BF16 FC1 projections and materialize MXFP8 rows.

    One CTA owns one routed pair and one N128 projection tile.  Its four
    eight-lane subgroups independently quantize the tile's four K32 groups.
    Each routed pair owns one compact materialized row.
    """

    def __init__(self, n: int, num_experts: int):
        self.n = int(n)
        self.num_experts = int(num_experts)
        if self.n < 128 or self.n % 128:
            raise ValueError("V4.1 micro activation requires N divisible by 128")
        if self.num_experts < 1:
            raise ValueError("V4.1 micro activation requires at least one expert")
        self.n_tiles = self.n // 128

    @cute.jit
    def __call__(
        self,
        projections: cute.Tensor,
        intermediate: cute.Tensor,
        topk_ids: cute.Tensor,
        route_weights: cute.Tensor,
        num_pairs: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            projections, intermediate, topk_ids, route_weights, num_pairs
        ).launch(
            grid=(num_pairs * Int32(self.n_tiles), 1, 1),
            block=[32, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        projections: cute.Tensor,
        intermediate: cute.Tensor,
        topk_ids: cute.Tensor,
        route_weights: cute.Tensor,
        num_pairs: Int32,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = Int32(tid)
        pair = Int32(bid) // Int32(self.n_tiles)
        output_tile = Int32(bid) % Int32(self.n_tiles)

        # This predicate is uniform for the CTA.  In particular, invalid routes
        # must not inspect their (intentionally uninitialized) FC1 boundary.
        if pair < num_pairs:
            expert = Int64(topk_ids[pair])
            if expert >= Int64(0) and expert < Int64(self.num_experts):
                col = output_tile * Int32(128) + lane * Int32(4)
                gate = cute.make_rmem_tensor((4,), cutlass.Float32)
                up = cute.make_rmem_tensor((4,), cutlass.Float32)
                value = cute.make_rmem_tensor((4,), cutlass.Float32)
                for i in cutlass.range_constexpr(4):
                    gate[i] = projections[pair, col + Int32(i)].to(cutlass.Float32)
                    up[i] = projections[
                        pair, Int32(self.n) + col + Int32(i)
                    ].to(cutlass.Float32)
                    gate[i] = cutlass.min(gate[i], cutlass.Float32(10.0))
                    up[i] = cutlass.max(
                        cutlass.min(up[i], cutlass.Float32(10.0)),
                        cutlass.Float32(-10.0),
                    )

                route_weight = route_weights[pair].to(cutlass.Float32)
                for i in cutlass.range_constexpr(4):
                    silu = div_rn_f32(
                        gate[i],
                        cutlass.Float32(1.0)
                        + cute.math.exp(-gate[i], fastmath=False),
                    )
                    # The FP32 route product crosses the required BF16
                    # activation boundary before the E4M3 conversion.
                    value[i] = cutlass.BFloat16(silu * up[i] * route_weight).to(
                        cutlass.Float32
                    )

                amax = fabs_f32(value[0])
                for i in cutlass.range_constexpr(1, 4):
                    amax = fmax_f32(amax, fabs_f32(value[i]))
                for shift in cutlass.range_constexpr(3):
                    amax = fmax_f32(
                        amax,
                        cute.arch.shuffle_sync_bfly(amax, offset=1 << shift),
                    )
                amax = fmax_f32(amax, cutlass.Float32(1.0e-4))
                _, scale_byte = pow2_ceil_ue8m0(
                    amax * cutlass.Float32(1.0 / FLOAT8_E4M3_MAX)
                )
                inv_scale = ue8m0_to_output_scale(scale_byte)

                rows = Int64(projections.shape[0])
                words_per_row = Int64(self.n // 4)
                physical_row = Int64(pair)
                intermediate[
                    physical_row * words_per_row
                    + output_tile * Int32(32)
                    + lane
                ] = cvt_f32x4_to_e4m3x4(
                    value[0] * inv_scale,
                    value[1] * inv_scale,
                    value[2] * inv_scale,
                    value[3] * inv_scale,
                )

                # The N128 scale plane has one packed word per physical row.
                # Each subgroup leader writes its raw UE8M0 byte directly.
                if lane % Int32(8) == Int32(0):
                    intermediate_u8 = cute.recast_tensor(intermediate, Uint8)
                    scale_byte_index = (
                        (
                            rows * words_per_row
                            + output_tile * rows
                            + physical_row
                        )
                        * Int32(4)
                        + lane // Int32(8)
                    )
                    intermediate_u8[scale_byte_index] = (
                        scale_byte & Uint32(0xFF)
                    ).to(Uint8)


__all__ = ["V41MicroActivationKernel"]
