"""SM120 local-window regression coverage for consumer Blackwell paths."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest
import torch


def _ensure_worktree_cute_loaded():
    worktree_cute = Path(__file__).resolve().parents[2] / "flash_attn" / "cute"
    try:
        finder = importlib.import_module("__editable___flash_attn_4_0_0_0_finder")
        if finder.MAPPING.get("flash_attn.cute") != str(worktree_cute):
            finder.MAPPING["flash_attn.cute"] = str(worktree_cute)
            for name in list(sys.modules):
                if name == "flash_attn" or name.startswith("flash_attn."):
                    del sys.modules[name]
    except ModuleNotFoundError:
        pass


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


@pytest.mark.timeout(30)
def test_sm120_hd256_backward_rejects_before_bad_launch():
    _sm120_only()
    from flash_attn.cute import flash_attn_func

    torch.manual_seed(0)
    q = torch.randn(1, 128, 8, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 128, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 128, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = flash_attn_func(q, k, v, causal=True)
    out = out[0] if isinstance(out, tuple) else out
    with pytest.raises(NotImplementedError, match="SM120 FA4 backward.*head_dim=head_dim_v=256"):
        out.backward(torch.randn_like(out))
