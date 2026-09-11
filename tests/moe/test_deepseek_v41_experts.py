"""Published V4.1 expert boundaries, including inactive local routes and replay."""

import pytest
import torch

from b12x.moe import fused_moe
from b12x._lib.runtime_control import (
    freeze_kernel_resolution,
    unfreeze_kernel_resolution,
)
from tests._reference.deepseek_v41_moe import moe_reference_deepseek_v41 as _oracle


def _setup(experts, hidden, intermediate, topk, capacity, *, policy=None):
    generator = torch.Generator(device="cuda").manual_seed(41083)
    w13 = torch.randint(0, 256, (experts, 2 * intermediate, hidden // 2), dtype=torch.uint8, device="cuda", generator=generator)
    w2 = torch.randint(0, 256, (experts, hidden, intermediate // 2), dtype=torch.uint8, device="cuda", generator=generator)
    sf13 = torch.full((experts, 2 * intermediate, hidden // 32), 122, dtype=torch.uint8, device="cuda")
    sf2 = torch.full((experts, hidden, intermediate // 32), 122, dtype=torch.uint8, device="cuda")
    checkpoint = tuple(t.clone() for t in (w13, sf13, w2, sf2))
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
        activation=fused_moe.ActivationSpec(mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16, numerical_recipe="deepseek_v41"),
        geometry=fused_moe.MoEGeometry(num_experts=experts, hidden_size=hidden, intermediate_size=intermediate),
    )
    unit = torch.ones(experts, device="cuda")
    prepared = fused_moe.prepare_weights(plan=weight_plan, weights=fused_moe.PackedWeights(
        w13=w13, w2=w2, w13_block_scales=sf13, w2_block_scales=sf2,
        w13_global_scales=unit, w2_global_scales=unit,
    ))
    plan = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(max_tokens=capacity, top_k=topk),
        policy=policy,
    )
    fused_moe.prewarm(plan)
    scratch = {spec.name: torch.empty(spec.shape, dtype=spec.dtype, device=spec.device) for spec in plan.scratch_specs()}
    x = torch.randn((capacity, hidden), generator=generator, device="cuda").bfloat16()
    # Only three experts receive work, and -1 denotes routes owned by other ranks.
    ids = torch.arange(capacity * topk, device="cuda", dtype=torch.int32).reshape(capacity, topk) % min(experts, 3)
    ids[:, -1] = -1
    weights = torch.rand((capacity, topk), generator=generator, device="cuda") * .7 + .07
    output = torch.empty((capacity, hidden), device="cuda", dtype=torch.float32)
    return plan, prepared, scratch, checkpoint, x, ids, weights, output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("geometry,counts", [
    ((384, 640, 192), (1, 6, 16, 32, 33, 192)),
    ((96, 2304, 32), (1, 6, 16, 32)),
])
def test_v41_m16_matches_m64_after_poison_and_graph_route_changes(geometry, counts):
    """Every consumed route slot must be rewritten without a barrier reset."""
    from b12x.moe.fused_moe._policy import MoeDecodeConfig
    from b12x.policy import PolicyContext

    policy = PolicyContext.for_device("cuda").with_override(
        "moe.decode",
        MoeDecodeConfig(
            backend="dynamic", route_planner="internal", max_active_clusters=None,
            dynamic_tile_m=16, dynamic_route_mode="grouped",
        ),
    )
    expert_count, intermediate, capacity = geometry
    plan, experts, scratch, _, x, ids, weights, output = _setup(
        expert_count, 5120, intermediate, 6, capacity, policy=policy,
    )
    control = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=capacity, top_k=6),
    )
    fused_moe.prewarm(control)
    control_scratch = {
        s.name: torch.empty(s.shape, dtype=s.dtype, device=s.device)
        for s in control.scratch_specs()
    }
    expected = torch.empty_like(output)

    def bind(p, storage, out, count):
        return fused_moe.bind(
            p, scratch=storage, experts=experts, a=x[:count],
            topk_ids=ids[:count], topk_weights=weights[:count],
            output=out[:count], input_scales_static=True,
        )

    freeze_kernel_resolution("V4.1 M16 planned capacity reuse")
    try:
        for count in counts:
            for storage in scratch.values():
                storage.view(torch.uint8).fill_(0xA5)
            candidate = bind(plan, scratch, output, count)
            reference = bind(control, control_scratch, expected, count)
            fused_moe.run(binding=reference)
            fused_moe.run(binding=candidate)
            torch.testing.assert_close(output[:count], expected[:count], rtol=0, atol=0)
        candidate = bind(plan, scratch, output, capacity)
        reference = bind(control, control_scratch, expected, capacity)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused_moe.run(binding=candidate)
        ids.fill_(-1)
        ids[::2, 0] = expert_count - 1
        weights.mul_(.8)
        x.mul_(.5)
        graph.replay()
        fused_moe.run(binding=reference)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        ids.fill_(-1)
        graph.replay()
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deepseek_v41_rounding_and_router_placement():
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        plan, experts, scratch, checkpoint, x, ids, weights, output = _setup(4, 256, 128, 3, 7)
        expected = _oracle(x, ids, weights, checkpoint)
        unrounded = _oracle(x, ids, weights, checkpoint, round_fc1=False)
        postweighted = _oracle(x, ids, weights, checkpoint, weight_before_fc2=False)
        # The adversarial oracle must distinguish both historical semantics.
        assert (expected - unrounded).abs().max().item() > .01
        assert (expected - postweighted).abs().max().item() > .01
        binding = fused_moe.bind(plan, scratch=scratch, experts=experts, a=x, topk_ids=ids, topk_weights=weights, output=output, input_scales_static=True)
        fused_moe.run(binding=binding)
        torch.testing.assert_close(output, expected, rtol=0, atol=.002)
        assert (output - expected).abs().sum() < (output - unrounded).abs().sum()
        assert (output - expected).abs().sum() < (output - postweighted).abs().sum()
        bf16_output = torch.empty_like(x)
        bf16_binding = fused_moe.bind(plan, scratch=scratch, experts=experts, a=x, topk_ids=ids, topk_weights=weights, output=bf16_output, input_scales_static=True)
        fused_moe.run(binding=bf16_binding)
        torch.testing.assert_close(bf16_output, expected.bfloat16(), rtol=0, atol=.002)
        x.mul_(1.0e-6)
        fused_moe.run(binding=binding)
        tiny_expected = _oracle(x, ids, weights, checkpoint)
        torch.testing.assert_close(output, tiny_expected, rtol=0, atol=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("local_experts,topk,capacity", [(48, 6, 65), (16, 3, 65), (96, 6, 8), (16, 3, 8)])
def test_deepseek_v41_local_experts_live_counts_and_graph(local_experts, topk, capacity):
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        plan, experts, scratch, checkpoint, x, ids, weights, output = _setup(local_experts, 5120, 2304, topk, capacity)
        for storage in scratch.values():
            storage.fill_(0xA5)
        freeze_kernel_resolution("V4.1 expert capacity reuse")
        try:
            for count in (1, 7, capacity):
                binding = fused_moe.bind(plan, scratch=scratch, experts=experts, a=x[:count], topk_ids=ids[:count], topk_weights=weights[:count], output=output[:count], input_scales_static=True)
                allocated_before = torch.cuda.memory_allocated()
                fused_moe.run(binding=binding)
                assert torch.cuda.memory_allocated() == allocated_before
                expected = _oracle(x[:count], ids[:count], weights[:count], checkpoint)
                torch.testing.assert_close(output[:count], expected, rtol=.01, atol=.08)
            binding = fused_moe.bind(plan, scratch=scratch, experts=experts, a=x, topk_ids=ids, topk_weights=weights, output=output, input_scales_static=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused_moe.run(binding=binding)
            # Replay transitions from three active experts to one, then none;
            # stale route scratch must never survive inactive local routes.
            ids.fill_(-1)
            ids[::2, 0] = local_experts - 1
            x.mul_(.5)
            weights.mul_(.8)
            graph.replay()
            torch.testing.assert_close(output, _oracle(x, ids, weights, checkpoint), rtol=.01, atol=.08)
            ids.fill_(-1)
            graph.replay()
            torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
        finally:
            unfreeze_kernel_resolution()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("capacity", [8, 65])
def test_deepseek_v41_rejects_bf16_route_weights_at_bind(capacity):
    plan, experts, scratch, _, x, ids, weights, output = _setup(4, 256, 128, 3, capacity)
    with pytest.raises(TypeError, match="FP32"):
        fused_moe.bind(
            plan, scratch=scratch, experts=experts, a=x,
            topk_ids=ids, topk_weights=weights.bfloat16(), output=output,
            input_scales_static=True,
        )
