"""Published DeepSeek V4.1 W4A8 expert numerical reference."""

import torch


def _mxfp8(x):
    blocks = x.float().reshape(*x.shape[:-1], -1, 32)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1.0e-4) / 448.0)))
    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape)


def _decode(packed, scales):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=packed.device)
    scale = torch.exp2(scales.view(torch.uint8).float() - 127).repeat_interleave(32, dim=-1)
    return lut[codes] * scale


def _gemm_k32(activations, weights):
    """Published kernel.py:534-554 uses FP32 accumulation of K32 partials.

    MX scales are exact powers of two, so decoding each K32 operand group
    commutes with its contraction; summing all K at once does not preserve
    the reference's intermediate FP32 rounding.
    """
    result = torch.zeros(
        (activations.shape[0], weights.shape[0]),
        dtype=torch.float32,
        device=activations.device,
    )
    for k in range(0, activations.shape[1], 32):
        result.add_(activations[:, k:k + 32] @ weights[:, k:k + 32].T)
    return result


def moe_reference_deepseek_v41(x, ids, weights, checkpoint, *, round_fc1=True, weight_before_fc2=True):
    """Return the FP32 route sum with the published K32/BF16 boundaries.

    ``checkpoint`` is ``(w13, sf13, w2, sf2)`` in source gate/up order.
    Scale tensors contain raw E8M0 exponent bytes, either uint8 or E8M0 typed.
    Routes outside the local expert range contribute zero. The keyword
    switches retain adversarial comparisons against historical semantics;
    the defaults round FC1 to BF16 and apply route weights before FC2.
    """
    w13, sf13, w2, sf2 = checkpoint
    result = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    x8 = _mxfp8(x)
    for expert in range(w13.shape[0]):
        token, slot = torch.where(ids == expert)
        if not token.numel():
            continue
        gate, up = _gemm_k32(x8[token], _decode(w13[expert], sf13[expert])).chunk(2, dim=-1)
        if round_fc1:
            gate, up = gate.bfloat16().float(), up.bfloat16().float()
        mid = torch.nn.functional.silu(gate.clamp(max=10.)) * up.clamp(-10., 10.)
        route = weights[token, slot, None]
        if weight_before_fc2:
            mid = mid * route
        down = _gemm_k32(_mxfp8(mid.bfloat16()), _decode(w2[expert], sf2[expert])).bfloat16().float()
        if not weight_before_fc2:
            down = down * route
        result.index_add_(0, token, down)
    return result
