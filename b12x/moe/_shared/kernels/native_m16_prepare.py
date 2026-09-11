"""Bounded native V4.1 M16 routing and MXFP8 input preparation.

Two stream-ordered kernels replace the cooperative resident-grid preparation.
The first publishes integer grouped-route counts/prefixes; the second packs
one input per token, emits stable token-major route maps, clears all route
outputs (including masked routes), and publishes materialized FC1/FC2 tasks.
The original FC1, route-weight-before-quantization, FC2 and top-k sum remain
unchanged. Live token counts are runtime extents, never compilation keys.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint32, Uint64

from b12x._lib.intrinsics import (
    atomic_add_shared_i32,
    bfloat2_to_float2_scaled,
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    ld_global_v4_u32,
    quantize_block_fp8_mx,
    shared_ptr_to_u32,
    st_global_u64,
    st_global_v4_u32,
)


@cute.jit
def _load_native_block(a: cute.Tensor, offset: Int64):
    # Same four vector loads, conversion, maximum order and quantizer as the
    # original dynamic front-end. Widen before every global row-stride product.
    values = cute.make_rmem_tensor((32,), cutlass.Float32)
    maximum = cutlass.Float32(0.0)
    address = get_ptr_as_int64(a, offset)
    for vec in cutlass.range_constexpr(4):
        w0, w1, w2, w3 = ld_global_v4_u32(address + Int64(vec * 16))
        v0, v1 = bfloat2_to_float2_scaled(w0, cutlass.Float32(1.0))
        v2, v3 = bfloat2_to_float2_scaled(w1, cutlass.Float32(1.0))
        v4, v5 = bfloat2_to_float2_scaled(w2, cutlass.Float32(1.0))
        v6, v7 = bfloat2_to_float2_scaled(w3, cutlass.Float32(1.0))
        values[vec * 8] = v0
        values[vec * 8 + 1] = v1
        values[vec * 8 + 2] = v2
        values[vec * 8 + 3] = v3
        values[vec * 8 + 4] = v4
        values[vec * 8 + 5] = v5
        values[vec * 8 + 6] = v6
        values[vec * 8 + 7] = v7
        maximum = fmax_f32(maximum, fabs_f32(v0))
        maximum = fmax_f32(maximum, fabs_f32(v1))
        maximum = fmax_f32(maximum, fabs_f32(v2))
        maximum = fmax_f32(maximum, fabs_f32(v3))
        maximum = fmax_f32(maximum, fabs_f32(v4))
        maximum = fmax_f32(maximum, fabs_f32(v5))
        maximum = fmax_f32(maximum, fabs_f32(v6))
        maximum = fmax_f32(maximum, fabs_f32(v7))
    return values, maximum


class NativeM16Prepare:
    """Native materialized M16 only; at most 512 experts, static K/top-k."""

    def __init__(self, num_topk: int):
        self.num_topk = int(num_topk)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        packed: cute.Tensor,
        scales: cute.Tensor,
        row_counts: cute.Tensor,
        expert_bases: cute.Tensor,
        route_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        all_published: cute.Tensor,
        gate_tiles: Int32,
        stream: cuda.CUstream,
    ):
        self.plan(ids, row_counts, expert_bases).launch(
            grid=(1, 1, 1), block=(512, 1, 1), stream=stream,
        )
        self.pack(
            a, ids, weights, packed, scales, row_counts, expert_bases,
            route_output, token_map, token_weights, task_expert, task_valid,
            task_head, task_tail, all_published, gate_tiles,
        ).launch(
            grid=(a.shape[0], 1, 1), block=(256, 1, 1), stream=stream,
        )

    @cute.kernel
    def plan(self, ids: cute.Tensor, counts: cute.Tensor, bases: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        tid = Int32(tid)
        experts = Int32(counts.shape[0])
        total = Int32(ids.shape[0])
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            histogram: cute.struct.MemRange[cutlass.Int32, 512]

        storage = smem.allocate(Storage)
        hist = storage.histogram.get_tensor(cute.make_layout(512))
        hist[tid] = Int32(0)
        cute.arch.sync_threads()
        idx = tid
        while idx < total:
            expert = ids[idx].to(Int32)
            if expert >= Int32(0) and expert < experts:
                atomic_add_shared_i32(
                    shared_ptr_to_u32(storage.histogram.data_ptr()) + expert * Int32(4),
                    Int32(1),
                )
            idx += Int32(512)
        cute.arch.sync_threads()
        count = hist[tid]
        if tid < experts:
            counts[tid] = count
        own_tiles = (count + Int32(15)) // Int32(16)
        value = own_tiles
        hist[tid] = value
        cute.arch.sync_threads()
        for bit in cutlass.range_constexpr(9):
            previous = Int32(0)
            if tid >= Int32(1 << bit):
                previous = hist[tid - Int32(1 << bit)]
            cute.arch.sync_threads()
            value += previous
            hist[tid] = value
            cute.arch.sync_threads()
        if tid < experts:
            bases[tid] = value - own_tiles
        if tid == Int32(0):
            bases[experts] = hist[experts - Int32(1)]

    @cute.kernel
    def pack(
        self,
        a: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        packed: cute.Tensor,
        scales: cute.Tensor,
        counts: cute.Tensor,
        bases: cute.Tensor,
        route_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        all_published: cute.Tensor,
        gate_tiles: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        blocks, _, _ = cute.arch.grid_dim()
        tid, token, blocks = Int32(tid), Int32(token), Int32(blocks)
        hidden = Int32(a.shape[1])
        experts = Int32(counts.shape[0])
        groups = hidden // Int32(32)

        if tid < Int32(self.num_topk):
            pair = Int64(token) * Int64(self.num_topk) + Int64(tid)
            expert = ids[pair].to(Int32)
            if expert >= Int32(0) and expert < experts:
                rank = Int32(0)
                prior = Int64(0)
                while prior < pair:
                    if ids[prior].to(Int32) == expert:
                        rank += Int32(1)
                    prior += Int64(1)
                physical = (Int64(bases[expert]) + Int64(rank // Int32(16))) * Int64(16)
                physical += Int64(rank % Int32(16))
                token_map[physical] = Int32(pair)
                token_weights[physical] = weights[pair].to(cutlass.Float32)

        group = tid
        while group < groups:
            offset = Int64(token) * Int64(hidden) + Int64(group) * Int64(32)
            values, maximum = _load_native_block(a, offset)
            maximum = cutlass.max(maximum, cutlass.Float32(1.0e-4))
            payload, scale = quantize_block_fp8_mx(values, maximum)
            for pair_word in cutlass.range_constexpr(4):
                word = (Uint64(payload[pair_word * 2 + 1]) << Uint64(32))
                word |= Uint64(payload[pair_word * 2])
                st_global_u64(
                    get_ptr_as_int64(packed, offset + Int64(pair_word * 8)), word,
                )
            scales[Int64(token) * Int64(groups) + Int64(group)] = Uint8(scale)
            group += Int32(256)

        # BF16 route outputs: every valid route is later overwritten by FC2;
        # inactive routes must remain exact zero for the fixed-order sum.
        vectors_per_token = Int64(self.num_topk) * Int64(hidden) // Int64(8)
        vector = Int64(tid)
        base = route_output.iterator.toint()
        while vector < vectors_per_token:
            offset_bytes = (Int64(token) * vectors_per_token + vector) * Int64(16)
            st_global_v4_u32(base + offset_bytes, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
            vector += Int64(256)

        expert = token * Int32(256) + tid
        while expert < experts:
            remaining = counts[expert]
            tile = bases[expert]
            while remaining > Int32(0):
                valid = cutlass.min(remaining, Int32(16))
                group = Int32(0)
                while group < gate_tiles:
                    slot = Int64(tile) * Int64(gate_tiles) + Int64(group)
                    task_expert[slot] = expert
                    task_valid[slot] = valid
                    group += Int32(1)
                remaining -= Int32(16)
                tile += Int32(1)
            expert += blocks * Int32(256)
        if token == Int32(0) and tid == Int32(0):
            task_head[Int32(0)] = Int32(0)
            task_tail[Int32(0)] = bases[experts] * gate_tiles
            all_published[Int32(0)] = Int32(1)


__all__ = ["NativeM16Prepare"]
