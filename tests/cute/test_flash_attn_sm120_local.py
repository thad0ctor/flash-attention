"""SM120 local-window regression coverage for consumer Blackwell paths."""

from __future__ import annotations

import importlib
import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def _ensure_worktree_cute_loaded():
    worktree_root = Path(__file__).resolve().parents[2]
    worktree_cute = worktree_root / "flash_attn" / "cute"
    try:
        finder = importlib.import_module("__editable___flash_attn_4_0_0_0_finder")
        if finder.MAPPING.get("flash_attn.cute") != str(worktree_cute):
            finder.MAPPING["flash_attn.cute"] = str(worktree_cute)
            for name in list(sys.modules):
                if name == "flash_attn" or name.startswith("flash_attn."):
                    del sys.modules[name]
    except ModuleNotFoundError:
        for name in list(sys.modules):
            if name == "flash_attn" or name.startswith("flash_attn."):
                del sys.modules[name]
        pkg = types.ModuleType("flash_attn")
        pkg.__path__ = [str(worktree_root / "flash_attn")]
        pkg.__package__ = "flash_attn"
        sys.modules["flash_attn"] = pkg


_ensure_worktree_cute_loaded()


def _sm120_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cc = torch.cuda.get_device_capability(0)
    if cc != (12, 0):
        pytest.skip(f"SM120-only test (got sm_{cc[0]}{cc[1]})")


def _sliding_ref(q, k, v, window_left):
    b, s, hq, d = q.shape
    hkv = k.shape[2]
    qpkv = hq // hkv
    qf = q.float().transpose(1, 2)
    kf = k.float().repeat_interleave(qpkv, dim=2).transpose(1, 2)
    vf = v.float().repeat_interleave(qpkv, dim=2).transpose(1, 2)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * (1.0 / math.sqrt(d))
    q_idx = torch.arange(s, device=q.device)[:, None]
    k_idx = torch.arange(s, device=q.device)[None, :]
    mask = (k_idx <= q_idx) & (k_idx >= q_idx - window_left)
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vf).transpose(1, 2).to(q.dtype)


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "h_q,h_kv,window_left",
    [
        (8, 1, 64),   # Gemma E2B-style qpkv=8
        (8, 2, 64),   # Gemma E4B-style qpkv=4
        (32, 16, 96), # Gemma 31B-style qpkv=2
    ],
)
def test_sm120_hd256_local_forward_matches_reference(h_q, h_kv, window_left):
    _sm120_only()
    from flash_attn.cute import flash_attn_func

    torch.manual_seed(0)
    q = torch.randn(1, 256, h_q, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 256, h_kv, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 256, h_kv, 256, device="cuda", dtype=torch.bfloat16)

    out = flash_attn_func(q, k, v, causal=True, window_size=(window_left, 0))
    out = out[0] if isinstance(out, tuple) else out
    ref = _sliding_ref(q, k, v, window_left)
    max_diff = float((out.float() - ref.float()).abs().max())
    assert max_diff < 0.05


@pytest.mark.timeout(60)
@pytest.mark.parametrize("hook_mode", ["k", "v", "both"])
def test_sm120_qpkv5_d128_hook_forward_matches_sdpa(monkeypatch, hook_mode):
    _sm120_only()
    from flash_attn.cute import flash_attn_func

    monkeypatch.setenv("FLASH_ATTENTION_SM120_QPKV5_HOOKS", hook_mode)
    torch.manual_seed(0)
    q = torch.randn(1, 256, 40, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 256, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 256, 8, 128, device="cuda", dtype=torch.bfloat16)

    out = flash_attn_func(q, k, v, causal=True)
    out = out[0] if isinstance(out, tuple) else out

    q_ref = q.float().transpose(1, 2)
    k_ref = k.float().repeat_interleave(5, dim=2).transpose(1, 2)
    v_ref = v.float().repeat_interleave(5, dim=2).transpose(1, 2)
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True).transpose(1, 2)
    assert (out.float() - ref).abs().max().item() < 0.05


@pytest.mark.timeout(90)
@pytest.mark.parametrize("hook_mode", ["off", "v", "both"])
def test_sm120_qpkv6_d256_hook_forward_matches_sdpa(monkeypatch, hook_mode):
    _sm120_only()
    from flash_attn.cute import flash_attn_func

    monkeypatch.setenv("FLASH_ATTENTION_SM120_QPKV6_D256_HOOKS", hook_mode)
    torch.manual_seed(0)
    q = torch.randn(1, 128, 24, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 128, 4, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 128, 4, 256, device="cuda", dtype=torch.bfloat16)

    out = flash_attn_func(q, k, v, causal=False)
    out = out[0] if isinstance(out, tuple) else out

    q_ref = q.float().transpose(1, 2)
    k_ref = k.float().repeat_interleave(6, dim=2).transpose(1, 2)
    v_ref = v.float().repeat_interleave(6, dim=2).transpose(1, 2)
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref).transpose(1, 2)
    assert (out.float() - ref).abs().max().item() < 0.05


@pytest.mark.timeout(90)
def test_sm120_qpkv6_d256_static_causal_blocks_matches_sdpa(monkeypatch):
    _sm120_only()
    from flash_attn.cute import flash_attn_func

    monkeypatch.setenv("FLASH_ATTENTION_SM120_QPKV6_D256_STATIC_CAUSAL_BLOCKS", "on")
    torch.manual_seed(0)
    q = torch.randn(1, 256, 24, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 256, 4, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 256, 4, 256, device="cuda", dtype=torch.bfloat16)

    out = flash_attn_func(q, k, v, causal=True)
    out = out[0] if isinstance(out, tuple) else out

    q_ref = q.float().transpose(1, 2)
    k_ref = k.float().repeat_interleave(6, dim=2).transpose(1, 2)
    v_ref = v.float().repeat_interleave(6, dim=2).transpose(1, 2)
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True).transpose(1, 2)
    assert (out.float() - ref).abs().max().item() < 0.05


@pytest.mark.timeout(90)
def test_sm120_qpkv8_d256_causal_qregs_matches_sdpa(monkeypatch):
    _sm120_only()
    from flash_attn.cute import flash_attn_func

    monkeypatch.setenv("FLASH_ATTENTION_SM120_D256_QPKV8_CAUSAL_QREGS", "128x64_t256")
    torch.manual_seed(0)
    q = torch.randn(1, 256, 16, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 256, 2, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 256, 2, 256, device="cuda", dtype=torch.bfloat16)

    out = flash_attn_func(q, k, v, causal=True, pack_gqa=True)
    out = out[0] if isinstance(out, tuple) else out

    q_ref = q.float().transpose(1, 2)
    k_ref = k.float().repeat_interleave(8, dim=2).transpose(1, 2)
    v_ref = v.float().repeat_interleave(8, dim=2).transpose(1, 2)
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True).transpose(1, 2)
    assert (out.float() - ref).abs().max().item() < 0.05


def test_sm120_bwd_qpkv4_s1024_causal_pack_split_policy(monkeypatch):
    from flash_attn.cute.interface import _sm120_bwd_pack_gqa_m_splits

    monkeypatch.delenv("FLASH_ATTENTION_SM120_BWD_PACK_GQA_M_SPLITS", raising=False)
    common = dict(
        arch=120,
        pack_gqa=True,
        qhead_per_kvhead=4,
        num_head=8,
        num_head_kv=2,
        causal=True,
        local=False,
        seqlen_k=1024,
        head_dim=256,
        head_dim_v=256,
        m_block_size=64,
        n_block_size=64,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
    )
    assert _sm120_bwd_pack_gqa_m_splits(seqlen_q=1024, **common) == 16
    assert _sm120_bwd_pack_gqa_m_splits(
        seqlen_q=1024,
        **{**common, "num_head": 16, "num_head_kv": 4},
    ) == 8
    assert _sm120_bwd_pack_gqa_m_splits(seqlen_q=2048, **{**common, "seqlen_k": 2048}) == 4


def test_sm120_bwd_qpkv8_s1024_causal_fused_dkv_policy(monkeypatch):
    from flash_attn.cute import interface

    monkeypatch.delenv("FLASH_ATTENTION_SM120_FUSED_DKV", raising=False)
    common = dict(
        arch=120,
        dtype=interface.cutlass.BFloat16,
        dkv_postprocess=True,
        pack_gqa=False,
        pack_gqa_m_splits=1,
        qhead_per_kvhead=8,
        causal=True,
        local=False,
        seqlen_k=1024,
        cu_seqlens_k=None,
        seqused_k=None,
        head_dim=256,
        head_dim_v=256,
        dKV_swapAB=False,
    )
    assert interface._sm120_use_fused_dkv_postprocess(seqlen_q=1024, **common)
    assert not interface._sm120_use_fused_dkv_postprocess(seqlen_q=2048, **{**common, "seqlen_k": 2048})
    assert not interface._sm120_use_fused_dkv_postprocess(
        seqlen_q=1024, **{**common, "dtype": interface.cutlass.Float16}
    )


@pytest.mark.timeout(60)
def test_sm120_d128_fused_dkv_backward_matches_sdpa(monkeypatch):
    _sm120_only()
    from flash_attn.cute import flash_attn_func

    monkeypatch.setenv("FLASH_ATTENTION_SM120_FUSED_DKV", "on")
    torch.manual_seed(0)
    q = torch.randn(1, 128, 32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = flash_attn_func(q, k, v, causal=False)
    out = out[0] if isinstance(out, tuple) else out
    dout = torch.randn_like(out)
    out.backward(dout)

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().repeat_interleave(8, dim=2).requires_grad_(True)
    v_ref = v.detach().float().repeat_interleave(8, dim=2).requires_grad_(True)
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(
            q_ref.transpose(1, 2),
            k_ref.transpose(1, 2),
            v_ref.transpose(1, 2),
        ).transpose(1, 2)
    ref.backward(dout.float())

    dk_ref = k_ref.grad.view(1, 128, 4, 8, 128).sum(dim=3)
    dv_ref = v_ref.grad.view(1, 128, 4, 8, 128).sum(dim=3)
    assert (q.grad.float() - q_ref.grad).abs().max().item() < 0.05
    assert (k.grad.float() - dk_ref).abs().max().item() < 0.05
    assert (v.grad.float() - dv_ref).abs().max().item() < 0.05


@pytest.mark.timeout(60)
def test_sm120_d256_fused_dkv_backward_matches_sdpa(monkeypatch):
    _sm120_only()
    from flash_attn.cute import flash_attn_func

    monkeypatch.setenv("FLASH_ATTENTION_SM120_FUSED_DKV", "on")
    torch.manual_seed(0)
    q = torch.randn(1, 128, 8, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 128, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 128, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = flash_attn_func(q, k, v, causal=True)
    out = out[0] if isinstance(out, tuple) else out
    dout = torch.randn_like(out)
    out.backward(dout)

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().repeat_interleave(8, dim=2).requires_grad_(True)
    v_ref = v.detach().float().repeat_interleave(8, dim=2).requires_grad_(True)
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(
            q_ref.transpose(1, 2),
            k_ref.transpose(1, 2),
            v_ref.transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2)
    ref.backward(dout.float())

    dk_ref = k_ref.grad.view(1, 128, 1, 8, 256).sum(dim=3)
    dv_ref = v_ref.grad.view(1, 128, 1, 8, 256).sum(dim=3)
    assert (q.grad.float() - q_ref.grad).abs().max().item() < 0.05
    assert (k.grad.float() - dk_ref).abs().max().item() < 0.05
    assert (v.grad.float() - dv_ref).abs().max().item() < 0.05


@pytest.mark.timeout(60)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "h_q,h_kv,pack_gqa",
    [
        (4, 2, False),
        (8, 2, False),
        (8, 2, True),
        (16, 4, True),
        (24, 4, True),
        (32, 2, True),
        (32, 16, True),
        (8, 1, False),
        (8, 1, None),
    ],
)
def test_sm120_hd256_backward_matches_sdpa(causal, h_q, h_kv, pack_gqa):
    _sm120_only()
    from flash_attn.cute import flash_attn_func

    torch.manual_seed(0)
    q = torch.randn(1, 128, h_q, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 128, h_kv, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 128, h_kv, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = flash_attn_func(q, k, v, causal=causal, pack_gqa=pack_gqa)
    out = out[0] if isinstance(out, tuple) else out
    dout = torch.randn_like(out)
    out.backward(dout)

    repeat = q.shape[2] // k.shape[2]
    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().repeat_interleave(repeat, dim=2).requires_grad_(True)
    v_ref = v.detach().float().repeat_interleave(repeat, dim=2).requires_grad_(True)
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(
            q_ref.transpose(1, 2),
            k_ref.transpose(1, 2),
            v_ref.transpose(1, 2),
            is_causal=causal,
        ).transpose(1, 2)
    ref.backward(dout.float())

    dk_ref = k_ref.grad.view(1, 128, h_kv, repeat, 256).sum(dim=3)
    dv_ref = v_ref.grad.view(1, 128, h_kv, repeat, 256).sum(dim=3)
    assert (q.grad.float() - q_ref.grad).abs().max().item() < 0.05
    assert (k.grad.float() - dk_ref).abs().max().item() < 0.05
    assert (v.grad.float() - dv_ref).abs().max().item() < 0.05
