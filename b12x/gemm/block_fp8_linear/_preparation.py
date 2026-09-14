"""Prepared native block-FP8 linear launchers."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compile_pool import CompileJob
from b12x.preparation import FrozenMapping, MemoryRequirements, PersistentMemory, Plan
from b12x.preparation.types import require_prepared
from ._tuning import BlockFp8LinearConfig, BlockFp8LinearQuery, TUNING, split_k_slices
from .._shared.block_fp8 import (
    BlockFP8LinearBinding,
    BlockFP8LinearScratchCaps,
    BlockFP8LinearWeight,
    _scratch_plan,
    _physical_mxfp8_k,
    _source_2d,
)
from .._shared.wo_mxfp8 import (
    MXFP8Rows,
    _check_mxfp8_rows_storage,
    empty_dense_gemm_mnl_view,
    empty_mxfp8_rows_bases,
    mxfp8_rows_from_bases,
)

def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _dense_lowering(query: BlockFp8LinearQuery, config: BlockFp8LinearConfig, device):
    """Lower the actual dense branch using metadata-only native operands."""
    from b12x._lib.dense_gemm import _lower_dense_gemm

    m, k, n = query.max_tokens, _physical_mxfp8_k(query.in_features), query.out_features
    values = lambda rows: torch.empty_strided(
        (rows, k, 1), (k, 1, rows * k), dtype=torch.float8_e4m3fn, device="meta"
    )
    scales = lambda rows: torch.empty_strided(
        (32, 4, math.ceil(rows / 128), 4, math.ceil((k // 32) / 4), 1),
        (16, 4, math.ceil((k // 32) / 4) * 512, 1, 512,
         math.ceil(rows / 128) * math.ceil((k // 32) / 4) * 512),
        dtype=torch.float8_e8m0fnu, device="meta",
    )
    out = torch.empty_strided(
        (m, n, 1), (n, 1, m * n), dtype=_dtype(query.output_dtype), device="meta"
    )
    return _lower_dense_gemm(
        (values(m), scales(m)), (values(n), scales(n)), out,
        ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
        c_dtype=query.output_dtype, sf_vec_size=32, sm_count=device.sm_count,
        mma_tiler_mn=(config.tile_m, config.tile_n), expected_m=m,
        sfb_k_replicated=query.weight_block_size == 128,
        _split_k_slices_override=split_k_slices(config) if split_k_slices(config) > 1 else None,
        _split_k_atomic_bf16_override=False if split_k_slices(config) > 1 else None,
        _split_k_workspace=(
            torch.empty((split_k_slices(config), m, n), dtype=torch.float32, device="meta")
            if split_k_slices(config) > 1 else None
        ),
    )


def _fused_route(query):
    return (
        query.max_tokens <= 8 and query.source_dtype == query.output_dtype == "bfloat16"
        and query.weight_block_size == 128 and query.in_features % 128 == 0
    )


def _fused_workspace_nbytes(query, config, sm_count):
    if not _fused_route(query):
        from types import SimpleNamespace
        lowering = _dense_lowering(query, config, SimpleNamespace(sm_count=sm_count))
        policy = lowering.policy
        return (policy.split_k_slices * query.max_tokens * query.out_features * 4
                if policy.split_k_slices > 1 and not policy.split_k_atomic_bf16 else 0)
    from b12x._lib import dense_gemm as dense
    source = torch.empty((query.max_tokens, query.in_features), dtype=torch.bfloat16, device="meta")
    weight = torch.empty_strided(
        (query.out_features, query.in_features, 1),
        (query.in_features, 1, query.out_features * query.in_features),
        dtype=torch.float8_e4m3fn, device="meta",
    )
    scales = torch.empty_strided(
        (32, 4, math.ceil(query.out_features / 128), 4, math.ceil((query.in_features // 32) / 4), 1),
        (16, 4, math.ceil((query.in_features // 32) / 4) * 512, 1, 512,
         math.ceil(query.out_features / 128) * math.ceil((query.in_features // 32) / 4) * 512),
        dtype=torch.float8_e8m0fnu, device="meta",
    )
    out = torch.empty_strided(
        (query.max_tokens, query.out_features, 1),
        (query.out_features, 1, query.max_tokens * query.out_features),
        dtype=torch.bfloat16, device="meta",
    )
    return dense._lower_dense_gemm_fused_quant_a(
        source, weight, scales, sm_count=sm_count, out=out,
        expected_m=query.max_tokens, sfb_k_replicated=True,
        mma_tiler_mn=(config.tile_m, config.tile_n),
    ).workspace_nbytes


@dataclass(frozen=True)
class _BlockFP8Compiled:
    """Exact compiled launchers plus their lowering metadata."""

    lowering: object
    quantize: object
    dense: object | None = None
    fused: object | None = None


def compile_block_fp8(query_payload, config_payload, ordinal, sm_count):
    """Compile the exact selected native branch from metadata."""
    from b12x._lib import dense_gemm as dense
    from b12x._lib.quant.mxfp8_rows import _get_compiled_mxfp8_rows_quant

    query = BlockFp8LinearQuery(**dict(query_payload))
    config = BlockFp8LinearConfig.from_config(FrozenMapping(config_payload))

    class _Device:
        identity = None

        def __init__(self):
            self.sm_count = sm_count

    if _fused_route(query):
        source = torch.empty(
            (query.max_tokens, query.in_features),
            dtype=torch.bfloat16,
            device="meta",
        )
        weight = torch.empty_strided(
            (query.out_features, query.in_features, 1),
            (query.in_features, 1, query.out_features * query.in_features),
            dtype=torch.float8_e4m3fn,
            device="meta",
        )
        scales = torch.empty_strided(
            (32, 4, math.ceil(query.out_features / 128), 4,
             math.ceil((query.in_features // 32) / 4), 1),
            (16, 4, math.ceil((query.in_features // 32) / 4) * 512, 1, 512,
             math.ceil(query.out_features / 128) * math.ceil((query.in_features // 32) / 4) * 512),
            dtype=torch.float8_e8m0fnu,
            device="meta",
        )
        out = torch.empty_strided(
            (query.max_tokens, query.out_features, 1),
            (query.out_features, 1, query.max_tokens * query.out_features),
            dtype=torch.bfloat16,
            device="meta",
        )
        lowering = dense._lower_dense_gemm_fused_quant_a(
            source, weight, scales, sm_count=sm_count, out=out,
            expected_m=query.max_tokens, sfb_k_replicated=True,
            mma_tiler_mn=(config.tile_m, config.tile_n),
        )
        with torch.cuda.device(ordinal):
            quantize = _get_compiled_mxfp8_rows_quant(
                query.in_features, torch.bfloat16, 8, 128, "linear",
                device_ordinal=ordinal, sm_count=sm_count,
            )
            fused = dense._compile_dense_fused_quant_lowering(
                lowering.to_dict(), ordinal
            )
        return attach_programs(
            _BlockFP8Compiled(lowering=lowering, quantize=quantize, fused=fused),
            quantize,
            fused,
        )
    lowering = _dense_lowering(query, config, _Device())
    subgroup_width, threads = ((8, 128) if query.max_tokens <= 8 else (4, 256))
    with torch.cuda.device(ordinal):
        quantize = _get_compiled_mxfp8_rows_quant(
            _physical_mxfp8_k(query.in_features), _dtype(query.source_dtype), subgroup_width, threads,
            "linear", 1e-4 if query.weight_block_size == 32 else 0.0,
            device_ordinal=ordinal, sm_count=sm_count,
        )
        dense_programs = dense._compile_dense_lowering(lowering.to_dict(), ordinal)
    return attach_programs(
        _BlockFP8Compiled(
            lowering=lowering,
            quantize=quantize,
            dense=dense_programs,
        ),
        quantize,
        dense_programs,
    )

@dataclass(frozen=True)
class _BlockFP8ExecutionState:
    query: BlockFp8LinearQuery
    config: BlockFp8LinearConfig
    device: torch.device
    scratch: object
    quantize: object | None
    dense: object | None
    fused: object | None

    def _check_source(self, source):
        source_2d = _source_2d(source)
        if (source.device != self.device or source_2d.dtype != _dtype(self.query.source_dtype)
                or source_2d.shape[0] > self.query.max_tokens
                or source_2d.shape[1] != self.query.in_features
                or not source_2d.is_contiguous()):
            raise ValueError("block-FP8 source differs from prepared capacity")
        return source_2d

    def quantize_input(self, source, *, out: MXFP8Rows | None = None, _initialize_scales=True):
        if self.quantize is None:
            raise ValueError("the prepared fused block-FP8 route has no standalone quantizer")
        source_2d = self._check_source(source)
        physical_k = _physical_mxfp8_k(self.query.in_features)
        if out is None:
            bases = empty_mxfp8_rows_bases(
                source_2d.shape[0], physical_k, num_groups=1,
                device=self.device, initialize_scales=_initialize_scales,
            )
            out = mxfp8_rows_from_bases(
                *bases, source_2d.shape[0], physical_k, num_groups=1,
            )
        _check_mxfp8_rows_storage(
            out,
            m=source_2d.shape[0],
            k=physical_k,
            num_groups=1,
        )
        self.quantize(source_2d, out.values, out.scale_rows, out.scale_mma)
        return out

    def _run(self, source, packed_weight, x_q, output, workspace, bias, stream):
        source_2d = self._check_source(source)
        if (not isinstance(packed_weight, BlockFP8LinearWeight)
                or packed_weight.in_features != self.query.in_features
                or packed_weight.out_features != self.query.out_features
                or packed_weight.block_size != (self.query.weight_block_size,) * 2):
            raise ValueError("block-FP8 weight geometry differs from preparation")
        weight = packed_weight.weight
        physical_k = _physical_mxfp8_k(self.query.in_features)
        if (output.shape != (source_2d.shape[0], self.query.out_features, 1)
                or output.dtype != _dtype(self.query.output_dtype)
                or output.device != self.device):
            raise ValueError("block-FP8 output differs from prepared capacity")
        if self.fused is not None:
            self.fused.run(
                source_2d, weight.values.view(self.query.out_features, physical_k, 1),
                weight.scale_mma, output, split_k_workspace=workspace, stream=stream,
            )
        else:
            x_q = self.quantize_input(source_2d, out=x_q, _initialize_scales=False)
            self.dense.run(
                (x_q.values.view(source_2d.shape[0], physical_k, 1), x_q.scale_mma),
                (weight.values.view(self.query.out_features, physical_k, 1), weight.scale_mma),
                out=output, stream=stream, split_k_workspace=workspace,
            )
        result = output[:, :, 0]
        if bias is not None:
            if (bias.device != self.device or bias.dtype != output.dtype
                    or bias.shape != (self.query.out_features,)):
                raise ValueError("block-FP8 bias differs from prepared output")
            result += bias
        return result.view(*source.shape[:-1], self.query.out_features)

    def bind(self, *, plan=None, **kwargs):
        if self.query.output_mode != "provided":
            raise ValueError("functional block-FP8 plan does not accept a caller output binding")
        return self.scratch.bind(plan=plan, **kwargs)

    def run_binding(self, binding: BlockFP8LinearBinding, *, stream=None):
        if binding.plan is not None and require_prepared(binding.plan, "gemm.block_fp8_linear") is not self:
            raise ValueError("binding belongs to another prepared block-FP8 plan")
        return self._run(binding.source, binding.packed_weight, binding.x_q, binding.output,
                         binding.workspace, binding.bias, stream)

    def run(self, source, packed_weight, *, bias=None, workspace=None, stream=None):
        if self.query.output_mode != "functional":
            raise ValueError("prepared block-FP8 plan requires caller-provided output binding")
        self._check_source(source)
        output = empty_dense_gemm_mnl_view(
            _source_2d(source).shape[0], self.query.out_features, 1, device=self.device,
            dtype=_dtype(self.query.output_dtype),
        )
        return self._run(source, packed_weight, None, output, workspace, bias, stream)

def plan(caps: BlockFP8LinearScratchCaps, *, invocation=FrozenMapping(), override=None) -> Plan:
    if not isinstance(caps, BlockFP8LinearScratchCaps):
        raise TypeError("plan requires BlockFP8LinearScratchCaps")
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("block-FP8 invocation semantics belong in Caps")
    query = BlockFp8LinearQuery(
        max_tokens=caps.max_tokens, in_features=caps.in_features,
        out_features=caps.out_features,
        source_dtype=str(caps.source_dtype).removeprefix("torch."),
        output_dtype=str(caps.output_dtype).removeprefix("torch."),
        output_mode=caps.output_mode,
        weight_block_size=caps.block_size[0],
    )

    def compile_jobs(config, device):
        return (CompileJob.create(
            "b12x.gemm.block_fp8_linear._preparation:compile_block_fp8",
            TUNING.encode_query(query), TUNING.encode_config(config),
            device.ordinal, device.identity.sm_count,
        ),)

    def memory(config, device):
        from b12x._lib import dense_gemm as dense
        alpha = dense._ALPHA_ONE_CACHE.get(("cuda", device.ordinal))
        resident = 0 if alpha is None else alpha.numel() * alpha.element_size()
        workspace_nbytes = _fused_workspace_nbytes(query, config, device.identity.sm_count)
        return MemoryRequirements(
            scratch=_scratch_plan(
                caps, (config.tile_m, config.tile_n), workspace_nbytes=workspace_nbytes,
            ).scratch_specs(),
            persistent=(PersistentMemory(("dense.alpha_one", device.ordinal), 4, resident),),
        )

    def materialize(selection, device):
        from b12x._lib import dense_gemm as dense
        config = selection.config
        programs = compile_block_fp8(
            TUNING.encode_query(query), TUNING.encode_config(config), device.ordinal,
            device.identity.sm_count,
        )
        resolved_device = torch.device("cuda", device.ordinal)
        if programs.fused is not None:
            fused = dense._materialize_dense_fused_quant(
                programs.lowering, resolved_device
            )
            quantize, core = programs.quantize, None
            workspace_nbytes = programs.lowering.workspace_nbytes
        else:
            lowering = programs.lowering
            core = dense._DenseExecutionState(
                lowering, resolved_device, programs.dense["gemm"],
                programs.dense.get("reduce"),
                dense._cached_alpha_one(resolved_device) if lowering.alpha_is_one else None,
            )
            quantize, fused = programs.quantize, None
            policy = lowering.policy
            workspace_nbytes = (
                policy.split_k_slices * query.max_tokens * query.out_features * 4
                if policy.split_k_slices > 1 and not policy.split_k_atomic_bf16 else 0
            )
        return _BlockFP8ExecutionState(
            query, config, resolved_device,
            _scratch_plan(
                caps, (config.tile_m, config.tile_n), workspace_nbytes=workspace_nbytes,
            ),
            quantize, core, fused,
        )

    return Plan(contract=TUNING, query=query, invocation=invocation, override=override,
                _compile_jobs=compile_jobs, _memory_requirements=memory,
                _materialize=materialize, _device=caps.device, shared=True)


__all__ = ["plan", "compile_block_fp8"]
