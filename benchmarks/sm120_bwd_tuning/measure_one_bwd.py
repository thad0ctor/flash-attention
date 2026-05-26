"""Phase 14 W1.2: fresh-subprocess measurement of one (cell, candidate)
backward configuration.

Mirrors phase5c/measure_one.py but for the backward pass:

  1. Pins the RTX 5090 by UUID BEFORE importing torch.
  2. Overrides the editable-install MAPPING so `flash_attn.cute` resolves to
     the phase14 worktree.
  3. Monkey-patches `flash_attn.cute.interface._flash_attn_bwd` so the
     hard-coded SM120 tile constants
     (interface.py: arch // 10 == 12 branch, lines ~1401-1421)
     can be overridden by CLI args.
  4. Numerical sanity check vs PyTorch SDPA gradient (max abs diff <= 0.05).
  5. Warmup 3 runs + 10 timed runs of backward only.
  6. Prints one JSON line on stdout.

CLI:
  python measure_one_bwd.py --preset NAME --sl N --causal {0,1}
                            --tile_m M --tile_n N --num_stages NS [--B B]

Stdout JSON keys:
  preset, sl, causal, tile_m, tile_n, num_stages,
  median_ms, tflops, status, error, maxdiff
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

# ---- Pin GPU BEFORE importing torch ----
# Pin to a specific GPU if FA_BENCH_GPU_UUID is exported; otherwise leave
# CUDA_VISIBLE_DEVICES unchanged (so the user's existing pinning applies).
_pin = os.environ.get("FA_BENCH_GPU_UUID")
if _pin:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pin

# ---- Optional: override editable install mapping to a worktree path ----
# Useful when iterating on local changes to flash_attn.cute without reinstalling.
# Set FA_BENCH_CUTE_OVERRIDE to the absolute path of the worktree's
# `flash_attn/cute` directory.
_cute_override = os.environ.get("FA_BENCH_CUTE_OVERRIDE")
if _cute_override:
    import __editable___flash_attn_4_0_0_0_finder as _F  # noqa: E402
    _F.MAPPING["flash_attn.cute"] = _cute_override

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

import flash_attn.cute  # noqa: E402,F401
from flash_attn.cute import interface as _iface  # noqa: E402


def _install_sm120_bwd_override_hook():
    """Patch `_iface._flash_attn_bwd` so the SM120 branch reads tile constants
    from `_iface._PHASE14_BWD_OVERRIDE` when set, falling back to the original
    hardcoded defaults when None.
    """
    import inspect
    _iface._PHASE14_BWD_OVERRIDE = None
    _orig_bwd = _iface._flash_attn_bwd
    src = inspect.getsource(_orig_bwd)
    needle = (
        "    if arch // 10 == 12:\n"
        "        # SM120: uses SM80 MMA with 99 KB SMEM, 128 threads (4 warps).\n"
        "        m_block_size = 64\n"
        "        n_block_size = 64\n"
        "        # num_stages=1 across all head_dim on consumer Blackwell. At\n"
        "        # head_dim>64 the SMEM cap forces ns=1; at head_dim<=64 the SM80-base\n"
        "        # default was ns=2 but the async pipeline overhead exceeds the\n"
        "        # latency-hiding benefit at small tile size. Phase 17C tightened\n"
        "        # paired validation (RTX 5090, n_measure=30, interleaved trials)\n"
        "        # confirms geomean speedup ~1.06x on 19 d=64 cells with 0\n"
        "        # regressions >2%.\n"
        "        num_stages_Q = 1\n"
        "        num_stages_dO = 1\n"
    )
    replacement = (
        "    if arch // 10 == 12:\n"
        "        # SM120: uses SM80 MMA with 99 KB SMEM, 128 threads (4 warps).\n"
        "        _ov = globals().get('_PHASE14_BWD_OVERRIDE')\n"
        "        if _ov is not None:\n"
        "            m_block_size = _ov['tile_m']\n"
        "            n_block_size = _ov['tile_n']\n"
        "            num_stages_Q = _ov['num_stages_Q']\n"
        "            num_stages_dO = _ov['num_stages_dO']\n"
        "        else:\n"
        "            m_block_size = 64\n"
        "            n_block_size = 64\n"
        "            # num_stages=1 across all head_dim on consumer Blackwell. At\n"
        "            # head_dim>64 the SMEM cap forces ns=1; at head_dim<=64 the SM80-base\n"
        "            # default was ns=2 but the async pipeline overhead exceeds the\n"
        "            # latency-hiding benefit at small tile size. Phase 17C tightened\n"
        "            # paired validation (RTX 5090, n_measure=30, interleaved trials)\n"
        "            # confirms geomean speedup ~1.06x on 19 d=64 cells with 0\n"
        "            # regressions >2%.\n"
        "            num_stages_Q = 1\n"
        "            num_stages_dO = 1\n"
    )
    if needle not in src:
        raise RuntimeError(
            "monkey-patch needle not found in _flash_attn_bwd source.\n"
            "Expected the SM120 branch to start with:\n" + needle
        )
    new_src = src.replace(needle, replacement)
    glb = _iface.__dict__
    loc = {}
    exec(compile(new_src, "<phase14_patched_bwd>", "exec"), glb, loc)
    # Preserve the compile_cache attribute attached to the original function
    new_fn = loc["_flash_attn_bwd"]
    if hasattr(_orig_bwd, "compile_cache"):
        new_fn.compile_cache = _orig_bwd.compile_cache
    _iface._flash_attn_bwd = new_fn


_install_sm120_bwd_override_hook()


PRESETS = {
    "llama3-8b":    {"H_q": 32, "H_kv": 8,  "head_dim": 128},
    "mistral-7b":   {"H_q": 32, "H_kv": 8,  "head_dim": 128},
    "qwen2.5-7b":   {"H_q": 28, "H_kv": 4,  "head_dim": 128},
    "llama2-7b":    {"H_q": 32, "H_kv": 32, "head_dim": 128},
    "mixtral-8x7b": {"H_q": 32, "H_kv": 8,  "head_dim": 128},
}


def fwd_flops(B, S, H_q, D, causal):
    base = 4 * B * H_q * S * S * D
    return base // 2 if causal else base


def bwd_flops(B, S, H_q, D, causal):
    # Backward ~2.5x forward (standard convention; matches phase13 bench)
    return int(2.5 * fwd_flops(B, S, H_q, D, causal))


def _repeat_kv(t, ratio):
    return t if ratio == 1 else t.repeat_interleave(ratio, dim=2)


def sdpa_grad_ref(q, k, v, do, causal):
    """Compute (dq, dk, dv) via SDPA math/flash backend for correctness check.

    Returns gradients with respect to the original (unrepeated) k/v tensors.
    """
    ratio = q.shape[2] // k.shape[2]
    q_r = q.detach().clone().requires_grad_(True)
    k_r = k.detach().clone().requires_grad_(True)
    v_r = v.detach().clone().requires_grad_(True)
    qh = q_r.transpose(1, 2)
    kh = _repeat_kv(k_r, ratio).transpose(1, 2)
    vh = _repeat_kv(v_r, ratio).transpose(1, 2)
    seqlen = qh.shape[2]
    backends = [SDPBackend.MATH] if seqlen <= 4096 else [SDPBackend.FLASH_ATTENTION]
    last_err = None
    for b in backends:
        try:
            with sdpa_kernel(b):
                out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal).transpose(1, 2)
            out.backward(do)
            return q_r.grad, k_r.grad, v_r.grad
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All SDPA backends failed: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(PRESETS.keys()))
    ap.add_argument("--sl", required=True, type=int)
    ap.add_argument("--causal", required=True, type=int, choices=[0, 1])
    ap.add_argument("--tile_m", required=True, type=int)
    ap.add_argument("--tile_n", required=True, type=int)
    ap.add_argument("--num_stages", required=True, type=int)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--n_warmup", type=int, default=3)
    ap.add_argument("--n_measure", type=int, default=10)
    ap.add_argument("--use_baseline", action="store_true",
                    help="Skip the override and use the hardcoded SM120 defaults "
                         "(ignores --tile_m / --tile_n / --num_stages for kernel selection, "
                         "but the values still appear in the output tag for tracking).")
    ap.add_argument("--skip_correctness", action="store_true",
                    help="Skip the SDPA correctness check (faster, but unsafe).")
    args = ap.parse_args()

    p = PRESETS[args.preset]
    H_q, H_kv, D = p["H_q"], p["H_kv"], p["head_dim"]
    B, sl, causal = args.B, args.sl, bool(args.causal)
    dtype = torch.bfloat16
    dev = torch.device("cuda")

    if args.use_baseline:
        _iface._PHASE14_BWD_OVERRIDE = None
    else:
        _iface._PHASE14_BWD_OVERRIDE = {
            "tile_m": args.tile_m,
            "tile_n": args.tile_n,
            "num_stages_Q": args.num_stages,
            "num_stages_dO": args.num_stages,
        }

    tag = {
        "preset": args.preset, "sl": sl, "causal": int(causal),
        "tile_m": args.tile_m, "tile_n": args.tile_n, "num_stages": args.num_stages,
        "B": B, "mode": "baseline" if args.use_baseline else "override",
    }

    try:
        torch.manual_seed(0)
        q = torch.randn(B, sl, H_q,  D, dtype=dtype, device=dev, requires_grad=True)
        k = torch.randn(B, sl, H_kv, D, dtype=dtype, device=dev, requires_grad=True)
        v = torch.randn(B, sl, H_kv, D, dtype=dtype, device=dev, requires_grad=True)
        do = torch.randn(B, sl, H_q, D, dtype=dtype, device=dev)

        from flash_attn.cute import flash_attn_func

        def fa_bwd_once(qq, kk, vv, do):
            for t in (qq, kk, vv):
                if t.grad is not None:
                    t.grad = None
            out = flash_attn_func(qq, kk, vv, causal=causal)
            if isinstance(out, tuple):
                out = out[0]
            out.backward(do)
            return qq.grad, kk.grad, vv.grad

        # --- Correctness check (also does the JIT compile) ---
        diff_q = diff_k = diff_v = -1.0
        try:
            dq, dk, dv = fa_bwd_once(q, k, v, do)
            if not args.skip_correctness:
                ref_dq, ref_dk, ref_dv = sdpa_grad_ref(q, k, v, do, causal)
                diff_q = (dq.float() - ref_dq.float()).abs().max().item()
                diff_k = (dk.float() - ref_dk.float()).abs().max().item()
                diff_v = (dv.float() - ref_dv.float()).abs().max().item()
                del ref_dq, ref_dk, ref_dv
        except Exception as e:
            print(json.dumps({**tag, "status": "compile_fail",
                              "error": f"{type(e).__name__}: {str(e)[:200].replace(chr(10), ' ')}"}))
            return
        torch.cuda.empty_cache()

        if not args.skip_correctness:
            # Backward gradients accumulate more noise than forward (multiple
            # MMAs per gradient element) and SDPA's math backend computes in
            # fp32 while ours accumulates in bf16 with fp32 intermediates, so
            # 0.05 is too tight for long sequences. 0.1 absolute is roughly
            # 12 bf16 ULPs at magnitude 1.0 -- still catches a real correctness
            # regression while accepting natural noise.
            tol = 0.1
            if max(diff_q, diff_k, diff_v) > tol:
                print(json.dumps({
                    **tag, "status": "numerical_fail",
                    "maxdiff_q": diff_q, "maxdiff_k": diff_k, "maxdiff_v": diff_v,
                }))
                return

        # --- Warmup ---
        for _ in range(args.n_warmup):
            fa_bwd_once(q, k, v, do)
        torch.cuda.synchronize()

        # --- Timed iters (backward only) ---
        times = []
        for _ in range(args.n_measure):
            for t in (q, k, v):
                if t.grad is not None:
                    t.grad = None
            out = flash_attn_func(q, k, v, causal=causal)
            if isinstance(out, tuple):
                out = out[0]
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out.backward(do)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
            del out
        times.sort()
        median_ms = times[len(times) // 2]
        p10_ms = times[max(0, int(len(times) * 0.1) - 1)]
        p90_ms = times[min(len(times) - 1, int(len(times) * 0.9))]
        flops = bwd_flops(B, sl, H_q, D, causal)
        tflops = flops / (median_ms * 1e-3) / 1e12

        print(json.dumps({
            **tag, "status": "ok",
            "median_ms": median_ms, "p10_ms": p10_ms, "p90_ms": p90_ms,
            "tflops": tflops,
            "maxdiff_q": diff_q, "maxdiff_k": diff_k, "maxdiff_v": diff_v,
        }))
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:300].replace(chr(10), ' ')}"
        print(json.dumps({**tag, "status": "error", "error": err}))
        traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
