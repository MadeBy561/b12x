"""V4.1 benchmark rank geometry and unbiased global routing contracts."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from benchmarks.benchmark_moe import (
    MODEL_PROFILES,
    ModelSpec,
    build_model_spec,
    compute_model_gate_routing,
)


def test_v41_checkpoint_partitions_experts_not_intermediate(tmp_path):
    config = {
        "model_type": "deepseek_v41_text", "hidden_size": 5120,
        "moe_intermediate_size": 2304, "n_routed_experts": 384,
        "num_experts_per_tok": 6,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"text_config": config}))
    profile = MODEL_PROFILES["deepseek-v4.1-flash"]
    for rank in range(4):
        spec = build_model_spec(tmp_path, profile, tp_rank=rank)
        assert (spec.num_experts, spec.I_tp) == (96, 2304)
        assert spec.tp_rank == rank
    with pytest.raises(ValueError):
        build_model_spec(tmp_path, profile, tp_rank=4)
    config["model_type"] = "deepseek_v4"
    path.write_text(json.dumps({"text_config": config}))
    with pytest.raises(ValueError, match="V4.1"):
        build_model_spec(tmp_path, profile)


def test_v41_ep_mapping_preserves_global_unbiased_route_weights():
    spec = ModelSpec(2, 128, 2, 2, 4, 0, expert_parallel=True)
    gate = torch.tensor([[1., 0.], [2., 0.], [3., 0.], [4., 0.],
                         [5., 0.], [6., 0.], [7., 0.], [8., 0.]])
    x = torch.tensor([[1., 0.], [-1., 0.]])
    bias = torch.tensor([0., 10., 0., 0., 0., 0., 0., 0.])
    weights = SimpleNamespace(
        spec=spec, numerical_recipe="deepseek_v41", gate_weight=gate,
        gate_temperature=2., gate_score_func="sqrtsoftplus", gate_bias=bias,
        gate_tid2eid=None, gate_norm_topk_prob=True, gate_route_scale=1.5,
    )
    unbiased = torch.nn.functional.softplus(x @ gate.T / 2.).sqrt()
    expected_ids = torch.tensor([[1, 7], [1, 0]])
    expected_weights = unbiased.gather(1, expected_ids)
    expected_weights *= 1.5 / expected_weights.sum(-1, keepdim=True)
    owners = torch.zeros_like(expected_ids)
    for rank in range(4):
        weights.spec = replace(spec, tp_rank=rank)
        ids, route_weights = compute_model_gate_routing(weights, x, seed=0)
        local = ids >= 0
        torch.testing.assert_close(route_weights, expected_weights)
        torch.testing.assert_close((ids + rank * 2)[local].long(), expected_ids[local])
        torch.testing.assert_close(ids[~local], torch.full_like(ids[~local], -1))
        owners += local.long()
    torch.testing.assert_close(owners, torch.ones_like(owners))
