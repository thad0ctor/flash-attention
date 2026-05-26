"""SM120 backward PackGQA regression coverage."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def _route_flash_attn_cute_to_this_worktree():
    worktree_root = Path(__file__).resolve().parents[2]
    cute_dir = worktree_root / "flash_attn" / "cute"
    try:
        finder = importlib.import_module("__editable___flash_attn_4_0_0_0_finder")
    except ModuleNotFoundError:
        if str(worktree_root) not in sys.path:
            sys.path.insert(0, str(worktree_root))
        return
    if finder.MAPPING.get("flash_attn.cute") == str(cute_dir):
        return
    finder.MAPPING["flash_attn.cute"] = str(cute_dir)
    for name in list(sys.modules):
        if name == "flash_attn" or name.startswith("flash_attn."):
            del sys.modules[name]


_route_flash_attn_cute_to_this_worktree()

try:
    from flash_attn.cute import flash_attn_func
except ImportError as _e:
    pytest.skip(f"flash_attn.cute not importable: {_e}", allow_module_level=True)


def _sm120_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cc = torch.cuda.get_device_capability(0)
    if cc != (12, 0):
        pytest.skip(f"SM120-only test (got sm_{cc[0]}{cc[1]})")


def _sdpa_ref_grads(q, k, v, dout, causal):
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    repeat = q.shape[2] // k.shape[2]
    qh = q_ref.transpose(1, 2)
    kh = k_ref.repeat_interleave(repeat, dim=2).transpose(1, 2)
    vh = v_ref.repeat_interleave(repeat, dim=2).transpose(1, 2)
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            qh.float(), kh.float(), vh.float(), is_causal=causal,
        ).transpose(1, 2).to(q.dtype)
    out.backward(dout)
    return q_ref.grad, k_ref.grad, v_ref.grad


@pytest.mark.parametrize("causal", [False, True])
def test_sm120_bwd_pack_gqa_odd_seqlen(causal):
    """Odd seqlen leaves OOB packed rows in the final m-block.

    The backward PackGQA Q/dO loads must zero those rows; otherwise stale smem
    pollutes dK/dV.  Use qh=7 to cover non-divisible packed row groups.
    """
    _sm120_only()
    torch.manual_seed(1700 + int(causal))
    batch, seqlen, nheads, nheads_kv, head_dim = 1, 65, 28, 4, 64
    dtype = torch.bfloat16
    q = torch.randn(batch, seqlen, nheads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(batch, seqlen, nheads_kv, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(batch, seqlen, nheads_kv, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    dout = torch.randn_like(q)

    out = flash_attn_func(q, k, v, causal=causal, pack_gqa=True)
    if isinstance(out, tuple):
        out = out[0]
    out.backward(dout)
    ref_dq, ref_dk, ref_dv = _sdpa_ref_grads(q, k, v, dout, causal)

    max_diff = max(
        (q.grad.float() - ref_dq.float()).abs().max().item(),
        (k.grad.float() - ref_dk.float()).abs().max().item(),
        (v.grad.float() - ref_dv.float()).abs().max().item(),
    )
    assert max_diff < 0.12
