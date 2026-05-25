# SM120 backward tile-tuning harness

Subprocess-isolated benchmark for the FA4 CuTe SM120 backward kernel.

For each cell (preset, seqlen, causal), spawns a fresh Python process per
`(tile_m, tile_n, num_stages)` candidate. Picks the per-cell winner by median
TFLOPS over multiple repeated runs and emits a Python dict suitable for use
as the SM120 backward tile lookup in `flash_attn/cute/interface.py`.

## Why subprocesses

Inside a single process the CuTe JIT cache is shared across configurations,
which contaminates timings (the first compile in each batch pays a hidden
cost, follow-on compiles hit warm caches). Spawning a fresh subprocess per
measurement ensures every cell starts from a cold JIT cache and a clean CUDA
context.

## Files

- `measure_one_bwd.py` — runs one cell + one tile config, prints a JSON
  result line on stdout. Monkey-patches the SM120 branch of
  `flash_attn.cute.interface._flash_attn_bwd` to honor the chosen tile.
- `bench_master_bwd.py` — orchestrator: sweeps the candidate grid, gathers
  results, computes per-cell winners, emits a lookup dict.

## Quick start (RTX 5090)

```bash
export FA_BENCH_GPU_UUID="GPU-..."   # nvidia-smi -L to find your card's UUID
export FA_BENCH_OUT_DIR=/tmp/sm120_bwd_tune
mkdir -p "$FA_BENCH_OUT_DIR"

# 1. Main sweep: 5 presets x 4 seqlens x 2 causal x up to 8 candidates each
python bench_master_bwd.py --phase main

# 2. Repro top-3 per cell with 2 extra runs for variance gating
python bench_master_bwd.py --phase repro

# 3. Pick winners, write bwd_lookup_candidate.py
python bench_master_bwd.py --phase analyze

# 4. Paired validation: baseline vs tuned, N runs each per cell
python bench_master_bwd.py --phase validate --N 3
```

## Environment variables

- `FA_BENCH_GPU_UUID` — pins both this harness and child subprocesses to a
  specific GPU. Use `nvidia-smi -L` to find the UUID. If unset, the
  harness inherits the caller's `CUDA_VISIBLE_DEVICES`.
- `FA_BENCH_OUT_DIR` — where to write `sweep_main.jsonl`, `sweep_repro.jsonl`,
  `winners.json`, `validation_report.json`. Defaults to the script's
  directory.
- `FA_BENCH_PYTHON` — Python interpreter to use for child subprocesses.
  Defaults to `sys.executable`.
- `FA_BENCH_CUTE_OVERRIDE` — absolute path of an alternative `flash_attn/cute`
  directory; the child process will rebind the editable-install MAPPING to
  this path. Use this when iterating on local kernel changes in a worktree
  without reinstalling the package.

## Output JSON schema (`sweep_main.jsonl` etc.)

```
{
  "preset": "llama3-8b",
  "sl": 4096,
  "causal": 1,
  "tile_m": 64,
  "tile_n": 64,
  "num_stages": 1,
  "B": 2,
  "mode": "override",            // "override" or "baseline"
  "status": "ok",                // "ok" | "compile_fail" | "numerical_fail" | "timeout" | ...
  "median_ms": 7.84,
  "p10_ms": 7.78,
  "p90_ms": 7.92,
  "tflops": 121.4,
  "maxdiff_q": 0.0039,           // vs PyTorch SDPA reference grad
  "maxdiff_k": 0.0039,
  "maxdiff_v": 0.0039,
  "elapsed_s": "12.3"
}
```

## Editing the candidate grid

`gen_candidates()` in `bench_master_bwd.py` returns up to 8 perturbations
around the (64, 64, baseline_ns) baseline. Adjust there to explore other
shapes. The SMEM filter (`smem_ok`) uses a conservative 92 KB budget
(actual SM120 cap is 99 KB; we leave headroom for PdS staging + alignment).
Candidates that pass `smem_ok` but still launch-fail at runtime are caught
as `status=compile_fail` and excluded from analysis.

## Paired-validation acceptance gate

The recommended gate before shipping a backward lookup as a source change is:

1. Zero per-cell regressions > 2% in `validation_report.json`.
2. Geomean tuned/baseline ratio ≥ 1.02× (otherwise the lookup isn't worth
   the source code complexity).

`bench_master_bwd.py --phase validate` checks both and writes the report to
`validation_report.json`.
