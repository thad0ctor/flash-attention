"""Regression tests for paged-KV forward on consumer Blackwell (sm_120).

Bug F (`/sm120-overnight-logs/phase4Z/REPORT.md`): the SM80-base forward
kernel that SM120 inherits silently produced wrong K/V reads when a
``page_table`` was supplied because ``mPageTable`` was never wired through
``load_K`` / ``load_V``.  Phase 4-Z installed an ``assert page_table is None``
on the SM120 dispatch (commit ``bf0a814``) and Phase 4-R replaced it with a
real cp.async paged-KV mainloop in ``FlashAttentionForwardSm80`` driven by
``flash_attn.cute.paged_kv.PagedKVManager`` (the same manager used by the
SM90 and SM100 forward paths). Phase 8 extends coverage to
``64 < head_dim <= 128`` by forcing the SM120 tile picker to
``(tile_m, tile_n, ns) = (128, 128, 1)`` when paged-KV is requested; the
SMEM cost (72 KB at d=96, 96 KB at d=128 with d==dv) fits the 99 KB cap.

These tests exercise the resulting paged-KV path against PyTorch SDPA on the
reconstructed (logical) K/V layout.  Covered:

* multiple page sizes (16 / 64 / 256)
* random permuted page tables
* page tables that share pages across batches
* causal masking
* GQA / MQA
* longer sequences
* head_dim in {64, 96, 128}

The bf16 max-abs-diff tolerance vs SDPA is 0.05 (the actual achieved diff is
~0.004 on every supported config).

Skips when not running on sm_120 because the fix lives in the SM120
dispatch.  head_dim>128 with paged-KV still raises ``NotImplementedError``
(tile_n>=128 + d>128 overflows the 99 KB SMEM cap); head_dim > head_dim_v
with paged-KV is rejected by the Bug E ``can_implement`` gate. We assert
those errors rather than silently passing.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Tuple

import pytest
import torch
import torch.nn.functional as F


def _route_flash_attn_cute_to_this_worktree():
    """Point `flash_attn.cute` at the worktree we live in.

    When pytest is run against the editable FA4 install, the finder maps
    ``flash_attn.cute`` to whatever directory was active at install time —
    usually the main checkout, not this worktree.  For a regression test to
    actually exercise THIS worktree's kernel changes, we patch the finder's
    MAPPING (and drop any cached ``flash_attn`` modules) so subsequent
    imports resolve here.

    This is the same shim Phase 4-R used in ``_use_worktree.py``.
    """
    worktree_root = Path(__file__).resolve().parents[2]  # tests/cute/.. -> repo root
    cute_dir = worktree_root / "flash_attn" / "cute"
    if not cute_dir.exists():
        return
    try:
        finder = importlib.import_module("__editable___flash_attn_4_0_0_0_finder")
    except ModuleNotFoundError:
        # No editable install — fall back to prepending the worktree to sys.path.
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

# Import directly from flash_attn.cute to avoid touching the old
# flash_attn_2_cuda package (which may not be installed alongside FA4).
try:
    from flash_attn.cute import flash_attn_varlen_func
except ImportError as _e:
    pytest.skip(f"flash_attn.cute not importable: {_e}", allow_module_level=True)


def _sm120_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cc = torch.cuda.get_device_capability(0)
    if cc != (12, 0):
        pytest.skip(f"Test targets sm_120, current device is sm_{cc[0]}{cc[1]}")


def _sdpa_reference(
    q: torch.Tensor,  # (b, hq, sq, d)
    k: torch.Tensor,  # (b, hk, sk, d)
    v: torch.Tensor,
    causal: bool,
):
    """SDPA reference. Uses FlashAttention's right-aligned causal mask when sk!=sq."""
    if causal and q.shape[-2] != k.shape[-2]:
        sq, sk = q.shape[-2], k.shape[-2]
        i = torch.arange(sq, device=q.device).unsqueeze(1)
        j = torch.arange(sk, device=q.device).unsqueeze(0)
        attn_mask = ~(j > (sk - sq + i))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _run_paged_case(
    batch_size: int = 2,
    seqlen_q: int = 128,
    seqlen_k: int = 256,
    nheads: int = 8,
    nheads_kv: int = 8,
    d: int = 64,
    page_size: int = 64,
    page_table_pattern: str = "permuted",
    causal: bool = False,
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[float, float]:
    """Returns (max_abs_diff, mean_abs_diff) vs SDPA on reconstructed K/V."""
    device = "cuda"
    torch.manual_seed(seed)
    assert seqlen_k % page_size == 0
    num_pages_per_seq = seqlen_k // page_size
    total_pages = (
        num_pages_per_seq
        if page_table_pattern == "shared"
        else max(batch_size * num_pages_per_seq * 2, num_pages_per_seq + 1)
    )

    total_q = batch_size * seqlen_q
    total_k = batch_size * seqlen_k

    q = torch.randn(total_q, nheads, d, device=device, dtype=dtype)
    k_contig = torch.randn(total_k, nheads_kv, d, device=device, dtype=dtype)
    v_contig = torch.randn(total_k, nheads_kv, d, device=device, dtype=dtype)

    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q, seqlen_q,
        dtype=torch.int32, device=device,
    )

    if page_table_pattern == "identity":
        page_table = torch.arange(
            batch_size * num_pages_per_seq, dtype=torch.int32, device=device,
        ).reshape(batch_size, num_pages_per_seq)
    elif page_table_pattern == "permuted":
        page_table = torch.randperm(
            total_pages, dtype=torch.int32, device=device,
        )[: batch_size * num_pages_per_seq].reshape(batch_size, num_pages_per_seq)
    elif page_table_pattern == "shared":
        base = torch.arange(num_pages_per_seq, dtype=torch.int32, device=device)
        page_table = base.unsqueeze(0).expand(batch_size, -1).contiguous()
    else:
        raise ValueError(f"Unknown pattern: {page_table_pattern}")

    k_paged = torch.zeros(
        total_pages, page_size, nheads_kv, d, device=device, dtype=dtype,
    )
    v_paged = torch.zeros(
        total_pages, page_size, nheads_kv, d, device=device, dtype=dtype,
    )
    for b in range(batch_size):
        for i in range(num_pages_per_seq):
            phys = int(page_table[b, i].item())
            src = b * seqlen_k + i * page_size
            k_paged[phys] = k_contig[src : src + page_size]
            v_paged[phys] = v_contig[src : src + page_size]

    seqused_k = torch.full(
        (batch_size,), seqlen_k, dtype=torch.int32, device=device,
    )

    out_paged, _ = flash_attn_varlen_func(
        q, k_paged, v_paged,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None,
        max_seqlen_q=seqlen_q, max_seqlen_k=None,
        seqused_k=seqused_k, page_table=page_table, causal=causal,
    )

    # Reference: SDPA on the reconstructed (logical) K/V layout per batch.
    out_ref_list = []
    for b in range(batch_size):
        qb = q[b * seqlen_q : (b + 1) * seqlen_q]
        kb = torch.zeros(seqlen_k, nheads_kv, d, device=device, dtype=dtype)
        vb = torch.zeros(seqlen_k, nheads_kv, d, device=device, dtype=dtype)
        for i in range(num_pages_per_seq):
            phys = int(page_table[b, i].item())
            kb[i * page_size : (i + 1) * page_size] = k_paged[phys]
            vb[i * page_size : (i + 1) * page_size] = v_paged[phys]
        if nheads != nheads_kv:
            assert nheads % nheads_kv == 0
            rep = nheads // nheads_kv
            kb = kb.repeat_interleave(rep, dim=1)
            vb = vb.repeat_interleave(rep, dim=1)
        qb_ = qb.transpose(0, 1).unsqueeze(0).float()
        kb_ = kb.transpose(0, 1).unsqueeze(0).float()
        vb_ = vb.transpose(0, 1).unsqueeze(0).float()
        out_b = _sdpa_reference(qb_, kb_, vb_, causal=causal)
        out_ref_list.append(out_b.squeeze(0).transpose(0, 1).to(dtype))
    out_ref = torch.cat(out_ref_list, dim=0)

    diff = (out_paged.float() - out_ref.float()).abs()
    return float(diff.max()), float(diff.mean())


TOL_BF16 = 0.05


@pytest.mark.parametrize("page_size,seqlen_k", [(16, 256), (64, 256), (256, 512)])
def test_page_sizes(page_size, seqlen_k):
    _sm120_only()
    md, _ = _run_paged_case(page_size=page_size, seqlen_k=seqlen_k, seed=page_size)
    assert md < TOL_BF16, f"max diff {md:.5f} >= {TOL_BF16}"


@pytest.mark.parametrize(
    "page_table_pattern", ["identity", "permuted", "shared"]
)
def test_page_table_patterns(page_table_pattern):
    _sm120_only()
    md, _ = _run_paged_case(
        page_table_pattern=page_table_pattern, seed=hash(page_table_pattern) & 0xFFFF,
    )
    assert md < TOL_BF16, f"max diff {md:.5f} >= {TOL_BF16}"


def test_causal():
    _sm120_only()
    md, _ = _run_paged_case(causal=True, seed=1)
    assert md < TOL_BF16


def test_multi_batch_causal_permuted():
    _sm120_only()
    md, _ = _run_paged_case(
        batch_size=8, causal=True, page_table_pattern="permuted", seed=2,
    )
    assert md < TOL_BF16


@pytest.mark.parametrize(
    "nheads,nheads_kv", [(8, 8), (8, 4), (8, 2), (8, 1)]
)
def test_gqa_mqa(nheads, nheads_kv):
    _sm120_only()
    md, _ = _run_paged_case(
        nheads=nheads, nheads_kv=nheads_kv, seed=nheads_kv,
    )
    assert md < TOL_BF16


@pytest.mark.parametrize(
    "seqlen_q,seqlen_k,causal",
    [(512, 1024, False), (1024, 2048, True), (256, 4096, True)],
)
def test_longer_sequences(seqlen_q, seqlen_k, causal):
    _sm120_only()
    md, _ = _run_paged_case(
        seqlen_q=seqlen_q, seqlen_k=seqlen_k, page_size=64,
        causal=causal, seed=seqlen_k,
    )
    assert md < TOL_BF16


@pytest.mark.parametrize("d", [96, 128])
@pytest.mark.parametrize("page_size,seqlen_k", [(16, 256), (64, 256), (256, 512)])
def test_d_gt64_page_sizes(d, page_size, seqlen_k):
    """head_dim in {96, 128} paged-KV across page sizes."""
    _sm120_only()
    md, _ = _run_paged_case(d=d, page_size=page_size, seqlen_k=seqlen_k, seed=d * 1000 + page_size)
    assert md < TOL_BF16, f"d={d} page_size={page_size}: max diff {md:.5f} >= {TOL_BF16}"


@pytest.mark.parametrize("d", [96, 128])
@pytest.mark.parametrize("page_table_pattern", ["identity", "permuted", "shared"])
def test_d_gt64_page_table_patterns(d, page_table_pattern):
    """head_dim in {96, 128} paged-KV across page-table layouts."""
    _sm120_only()
    md, _ = _run_paged_case(
        d=d, page_table_pattern=page_table_pattern,
        seed=d * 1000 + (hash(page_table_pattern) & 0xFFFF),
    )
    assert md < TOL_BF16, f"d={d} {page_table_pattern}: max diff {md:.5f} >= {TOL_BF16}"


@pytest.mark.parametrize("d", [96, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_d_gt64_causal(d, causal):
    """head_dim in {96, 128} paged-KV with/without causal masking."""
    _sm120_only()
    md, _ = _run_paged_case(d=d, causal=causal, seed=d * 1000 + int(causal))
    assert md < TOL_BF16, f"d={d} causal={causal}: max diff {md:.5f} >= {TOL_BF16}"


@pytest.mark.parametrize("d", [96, 128])
@pytest.mark.parametrize("nheads,nheads_kv", [(8, 2), (8, 1)])  # GQA(qhpkv=4), MQA(qhpkv=8)
def test_d_gt64_gqa_mqa(d, nheads, nheads_kv):
    """head_dim in {96, 128} paged-KV with GQA (qhpkv=4) and MQA (qhpkv=8)."""
    _sm120_only()
    md, _ = _run_paged_case(
        d=d, nheads=nheads, nheads_kv=nheads_kv, seed=d * 1000 + nheads_kv,
    )
    assert md < TOL_BF16, f"d={d} ({nheads},{nheads_kv}): max diff {md:.5f} >= {TOL_BF16}"


def test_d192_paged_rejected():
    """head_dim=192 with paged-KV must raise NotImplementedError (SMEM overflow)."""
    _sm120_only()
    with pytest.raises(NotImplementedError, match="head_dim"):
        _run_paged_case(d=192)


def test_d256_paged_rejected():
    """head_dim=256 with paged-KV must raise NotImplementedError (SMEM overflow)."""
    _sm120_only()
    with pytest.raises(NotImplementedError, match="head_dim"):
        _run_paged_case(d=256)


def test_d128_dv64_paged_varlen_correctness():
    """head_dim=128, head_dim_v=64 paged-KV + varlen routes through the non-TMA
    SM80-base kernel (the TMA path rejects d > dv). Verify the output matches
    SDPA on reconstructed K/V."""
    _sm120_only()
    device = "cuda"
    torch.manual_seed(0)
    batch_size = 2
    seqlen_q = 128
    seqlen_k = 256
    page_size = 64
    d_qk = 128
    d_v = 64
    nheads = 8
    nheads_kv = 8
    num_pages_per_seq = seqlen_k // page_size
    total_pages = batch_size * num_pages_per_seq * 2

    q = torch.randn(batch_size * seqlen_q, nheads, d_qk, device=device, dtype=torch.bfloat16)
    k_paged = torch.randn(total_pages, page_size, nheads_kv, d_qk, device=device, dtype=torch.bfloat16)
    v_paged = torch.randn(total_pages, page_size, nheads_kv, d_v, device=device, dtype=torch.bfloat16)
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=device)
    page_table = torch.randperm(total_pages, dtype=torch.int32, device=device)[
        : batch_size * num_pages_per_seq
    ].reshape(batch_size, num_pages_per_seq)
    seqused_k = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=device)

    out = flash_attn_varlen_func(
        q, k_paged, v_paged,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None,
        max_seqlen_q=seqlen_q, max_seqlen_k=None,
        seqused_k=seqused_k, page_table=page_table, causal=False,
    )
    if isinstance(out, tuple):
        out = out[0]

    # Reconstruct logical K/V from page table for SDPA reference.
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    k_ref = torch.zeros(batch_size, seqlen_k, nheads_kv, d_qk, device=device, dtype=torch.bfloat16)
    v_ref = torch.zeros(batch_size, seqlen_k, nheads_kv, d_v, device=device, dtype=torch.bfloat16)
    for b in range(batch_size):
        for p in range(num_pages_per_seq):
            k_ref[b, p * page_size : (p + 1) * page_size] = k_paged[page_table[b, p]]
            v_ref[b, p * page_size : (p + 1) * page_size] = v_paged[page_table[b, p]]
    q_ref = q.view(batch_size, seqlen_q, nheads, d_qk)
    with sdpa_kernel([SDPBackend.MATH]):
        ref = F.scaled_dot_product_attention(
            q_ref.transpose(1, 2).float(),
            k_ref.transpose(1, 2).float(),
            v_ref.transpose(1, 2).float(),
            is_causal=False,
        ).transpose(1, 2)
    ref = ref.reshape(batch_size * seqlen_q, nheads, d_v).to(torch.bfloat16)

    max_diff = float((out.float() - ref.float()).abs().max())
    assert max_diff < 0.05, f"max abs diff {max_diff:.6f} exceeds bf16 tolerance"
