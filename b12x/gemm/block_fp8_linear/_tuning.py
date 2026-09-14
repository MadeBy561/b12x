"""Configuration contract for block-FP8 linear preparation."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.preparation import (
    DeviceIdentity,
    FrozenMapping,
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)


@dataclass(frozen=True, kw_only=True)
class BlockFp8LinearQuery:
    max_tokens: int
    in_features: int
    out_features: int
    source_dtype: str
    output_dtype: str
    output_mode: str
    weight_block_size: int = 128

@dataclass(frozen=True, kw_only=True)
class BlockFp8LinearConfig:
    backend: str
    tile_m: int
    tile_n: int

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "BlockFp8LinearConfig":
        if set(payload) != {"backend", "tile_m", "tile_n"}:
            raise ValueError(
                "block-FP8 linear configs require backend, tile_m, and tile_n"
            )
        return cls(
            backend=str(payload["backend"]),
            tile_m=int(payload["tile_m"]),
            tile_n=int(payload["tile_n"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
        }


def _encode(query: BlockFp8LinearQuery) -> dict[str, object]:
    return {
        name: getattr(query, name) for name in BlockFp8LinearQuery.__dataclass_fields__
    }


def _default_config(
    query: BlockFp8LinearQuery,
    device: DeviceIdentity | None,
) -> BlockFp8LinearConfig:
    backend = "mxfp8"
    if query.max_tokens == 1:
        tile = (16, 64)
    elif query.max_tokens <= 8:
        tile = (16, 128)
        if (
            device is not None and device.compute_capability == (12, 0)
            and device.sm_count == 188 and query.weight_block_size == 32
            and query.output_dtype == "bfloat16"
        ):
            shape = (query.out_features, query.in_features)
            if shape in {(1792, 5120), (1152, 5120)}:
                tile, backend = (16, 64), "mxfp8_split4_fp32"
            elif shape == (4096, 1280):
                tile = (16, 64)
    elif query.max_tokens <= 128 and query.out_features > 1_536:
        tile = (32, 128)
    elif query.max_tokens <= 128:
        tile = (64, 64)
    else:
        tile = (64, 128)
    return BlockFp8LinearConfig(backend=backend, tile_m=tile[0], tile_n=tile[1])


def _validate_query(
    query: BlockFp8LinearQuery,
    _device: DeviceIdentity | None,
) -> None:
    if not isinstance(query, BlockFp8LinearQuery):
        raise TypeError("query must be BlockFp8LinearQuery")
    if query.source_dtype not in ("bfloat16", "float16"):
        raise ValueError(f"unsupported source dtype {query.source_dtype!r}")
    if query.output_dtype not in ("bfloat16", "float16"):
        raise ValueError(f"unsupported output dtype {query.output_dtype!r}")
    if query.output_mode not in ("functional", "provided"):
        raise ValueError(f"unsupported block-FP8 output mode {query.output_mode!r}")
    if query.max_tokens <= 0 or query.in_features <= 0 or query.out_features <= 0:
        raise ValueError("block-FP8 dimensions must be positive")
    if query.in_features % 32:
        raise ValueError("block-FP8 in_features must be a multiple of 32")
    if query.weight_block_size not in (32, 128):
        raise ValueError("block-FP8 weight block size must be 32 or 128")


def _validate_config(
    query: BlockFp8LinearQuery,
    config: BlockFp8LinearConfig,
    _device: DeviceIdentity | None,
) -> None:
    if not isinstance(config, BlockFp8LinearConfig):
        raise TypeError("config must be BlockFp8LinearConfig")
    if config.backend not in ("mxfp8", "mxfp8_split2_fp32", "mxfp8_split4_fp32"):
        raise ValueError(f"unsupported block-FP8 backend {config.backend!r}")
    if (config.tile_m, config.tile_n) not in {
        (16, 64),
        (16, 128),
        (32, 64),
        (32, 128),
        (64, 64),
        (64, 128),
        (128, 64),
        (128, 128),
    }:
        raise ValueError("unsupported block-FP8 MMA tile")


    slices = split_k_slices(config)
    if slices > 1 and not _split_eligible(query, slices):
        raise ValueError("FP32 split-K requires BF16 block32, M2..8 and K divisible by 256*slices")
    if slices > 1 and config.tile_m != 16:
        raise ValueError("FP32 split-K requires a 16-row tile")


def split_k_slices(config):
    return {"mxfp8_split2_fp32": 2, "mxfp8_split4_fp32": 4}.get(config.backend, 1)


def _split_eligible(query, slices):
    return (2 <= query.max_tokens <= 8 and query.weight_block_size == 32
            and query.output_dtype == "bfloat16"
            and query.in_features % (256 * slices) == 0)


def _parameters(query, device):
    backends = ("mxfp8",) + tuple(
        f"mxfp8_split{slices}_fp32" for slices in (2, 4) if _split_eligible(query, slices)
    )
    return ParameterSpace.create(
        TUNING.knobs, values={"backend": backends},
        predicates=(lambda choice: choice["backend"] == "mxfp8" or choice["tile_m"] == 16,),
    )


TUNING = TuningContract(
    component_id="gemm.block_fp8_linear",
    query_schema_version=4,
    config_schema_version=3,
    query_fields=frozenset(BlockFp8LinearQuery.__dataclass_fields__),
    config_fields=frozenset(BlockFp8LinearConfig.__dataclass_fields__),
    encode_query=_encode,
    encode_config=BlockFp8LinearConfig.to_dict,
    decode_config=BlockFp8LinearConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default_config,
    candidate_contract_version=2,
    parameters=_parameters,
    materialize=lambda query, device, choice: BlockFp8LinearConfig.from_config(choice),
    knobs=(
        Knob(name="backend", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="tile_m", values=(16, 32, 64, 128), binding=ParameterBinding.COMPILE),
        Knob(name="tile_n", values=(64, 128), binding=ParameterBinding.COMPILE),
    ),
)


__all__ = ["TUNING", "BlockFp8LinearConfig", "BlockFp8LinearQuery"]
