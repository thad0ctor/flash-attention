"""Phase 14 W1.2: backward tile-tuning sweep harness for SM120.

For each (preset, sl, causal) cell (= 5 * 4 * 2 = 40 cells), spawns up to 8
subprocesses each running `measure_one_bwd.py` with a different
(tile_m, tile_n, num_stages) candidate. Picks the best per cell
(highest TFLOPS, status==ok), then re-measures the top-3 candidates twice
more for variance gating, and emits `bwd_lookup_candidate.py`.

The same harness is also used for W1.4 paired-validation: pass `--validate`
to run both baseline (--use_baseline) and tuned (lookup-based) at each cell
N=3 times and compute the paired ratio + regression count.

Run:
  python bench_master_bwd.py --phase main
  python bench_master_bwd.py --phase repro
  python bench_master_bwd.py --phase analyze
  python bench_master_bwd.py --phase validate --N 3
  python bench_master_bwd.py --phase all
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict

# Output directory: env-overridable. Defaults to the directory holding this
# script so the harness is self-contained when invoked from a clone.
ROOT = os.environ.get("FA_BENCH_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "raw")
# measure_one_bwd.py lives next to this script.
MEASURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measure_one_bwd.py")
PYTHON = os.environ.get("FA_BENCH_PYTHON", sys.executable)
# GPU pin (UUID). Empty string -> let the caller's CUDA_VISIBLE_DEVICES win.
GPU_UUID = os.environ.get("FA_BENCH_GPU_UUID", "")

PRESETS = [
    # (name, H_q, H_kv, head_dim)
    ("llama3-8b",    32, 8,  128),
    ("mistral-7b",   32, 8,  128),
    ("qwen2.5-7b",   28, 4,  128),
    ("llama2-7b",    32, 32, 128),
    ("mixtral-8x7b", 32, 8,  128),
]
SEQLENS = [1024, 2048, 4096, 8192]
CAUSAL_OPTS = [0, 1]
B = 2  # match phase13 bench batch size


def sm120_baseline_for(head_dim: int):
    """Mirror interface.py SM120 backward hard-coded defaults."""
    return (64, 64, 1)


def smem_ok(d: int, tile_m: int, tile_n: int, num_stages: int,
            head_dim_v: int | None = None) -> bool:
    """Conservative SMEM check. Includes a fudge factor for PdS + alignment.

    Real SM120 cap is 101376 B. The kernel uses Q, K, V, dO, PdS staging,
    plus alignment overhead. The actual `can_implement` only checks
    Q + V + dO + K and SM120 cap, but launches still fail with extra overhead
    (PdS, padding). Use 92 KB as a conservative budget.
    """
    head_dim_v = head_dim_v if head_dim_v is not None else d
    smem_Q = tile_m * d * num_stages * 2
    smem_dO = tile_m * head_dim_v * num_stages * 2
    smem_K = tile_n * d * 2
    smem_V = tile_n * head_dim_v * 2
    smem_PdS = tile_m * tile_n * 2  # rough
    total = smem_Q + smem_dO + smem_K + smem_V + smem_PdS
    return total <= 92 * 1024


def gen_candidates(head_dim: int):
    """Return up to 8 distinct (tm, tn, ns) candidates per cell. Conservative
    perturbations around (64, 64, baseline_ns). Filtered by smem_ok().

    For backward, SMEM is the dominant constraint. Most useful explorations:
      - num_stages variants (1 vs 2)
      - asymmetric tiles (wider N, narrower N)
      - keep M=64 (M can grow but the M dimension shares scheduling with the
        forward, so M deviation has limited isolated impact).
    """
    baseline = sm120_baseline_for(head_dim)
    cands_raw = [
        baseline,
        (64, 64, 1),
        (64, 64, 2),
        (64, 96, 1),
        (64, 48, 1),
        (64, 32, 1),
        (64, 96, 2),
        (64, 48, 2),
    ]
    seen, out = set(), []
    for c in cands_raw:
        if c in seen:
            continue
        if not smem_ok(head_dim, *c):
            continue
        seen.add(c)
        out.append(c)
    return out[:8]


def run_measure(preset, sl, causal, tm, tn, ns, *, use_baseline=False,
                skip_correctness=False, timeout=240):
    env = os.environ.copy()
    if GPU_UUID:
        env["CUDA_VISIBLE_DEVICES"] = GPU_UUID
        env["FA_BENCH_GPU_UUID"] = GPU_UUID
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    cmd = [
        PYTHON, MEASURE,
        "--preset", preset, "--sl", str(sl), "--causal", str(causal),
        "--tile_m", str(tm), "--tile_n", str(tn), "--num_stages", str(ns),
        "--B", str(B),
    ]
    if use_baseline:
        cmd.append("--use_baseline")
    if skip_correctness:
        cmd.append("--skip_correctness")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, text=True,
        )
        elapsed = time.time() - t0
        # The measure script prints a single JSON object on stdout
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            return {
                "preset": preset, "sl": sl, "causal": causal,
                "tile_m": tm, "tile_n": tn, "num_stages": ns,
                "mode": "baseline" if use_baseline else "override",
                "status": "subprocess_no_output",
                "error": (proc.stderr[-200:] if proc.stderr else "").replace("\n", " "),
                "elapsed_s": f"{elapsed:.1f}",
            }
        try:
            row = json.loads(lines[-1])
        except json.JSONDecodeError:
            return {
                "preset": preset, "sl": sl, "causal": causal,
                "tile_m": tm, "tile_n": tn, "num_stages": ns,
                "mode": "baseline" if use_baseline else "override",
                "status": "subprocess_bad_output",
                "error": lines[-1][:200],
                "elapsed_s": f"{elapsed:.1f}",
            }
        row["elapsed_s"] = f"{elapsed:.1f}"
        return row
    except subprocess.TimeoutExpired:
        return {
            "preset": preset, "sl": sl, "causal": causal,
            "tile_m": tm, "tile_n": tn, "num_stages": ns,
            "mode": "baseline" if use_baseline else "override",
            "status": "timeout", "error": f"timeout after {timeout}s",
            "elapsed_s": f"{timeout}",
        }


def cells_iter():
    for (name, H_q, H_kv, D) in PRESETS:
        for sl in SEQLENS:
            for causal in CAUSAL_OPTS:
                yield (name, H_q, H_kv, D, sl, causal)


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def phase_main(args):
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(ROOT, "sweep_main.jsonl")
    if os.path.exists(out_path) and not args.force:
        print("sweep_main.jsonl exists; resuming. Use --force to restart.")
    elif args.force and os.path.exists(out_path):
        os.remove(out_path)
    existing = {(r["preset"], r["sl"], r["causal"], r["tile_m"], r["tile_n"], r["num_stages"])
                for r in read_jsonl(out_path)}

    cells = list(cells_iter())
    t0 = time.time()
    for idx, (name, H_q, H_kv, D, sl, causal) in enumerate(cells, 1):
        cands = gen_candidates(D)
        for (tm, tn, ns) in cands:
            key = (name, sl, causal, tm, tn, ns)
            if key in existing:
                continue
            r = run_measure(name, sl, causal, tm, tn, ns, skip_correctness=False)
            r["phase"] = "main"
            r["head_dim"] = D
            r["qhead_per_kvhead"] = H_q // H_kv
            append_jsonl(out_path, r)
            status = r.get("status", "")
            tflops = r.get("tflops", "")
            ms = r.get("median_ms", "")
            print(f"  [{idx:2d}/{len(cells)}] {name:14s} sl={sl:5d} c={causal} "
                  f"({tm:3d},{tn:3d},ns={ns}) {ms!s:>10} ms {tflops!s:>8} TF "
                  f"{status:18s} [{r['elapsed_s']}s]", flush=True)
    dt = time.time() - t0
    print(f"\nphase=main done in {dt/60:.1f} min  -> {out_path}")


def phase_repro(args):
    """Re-run top-3 per cell twice more for variance gating."""
    main_path = os.path.join(ROOT, "sweep_main.jsonl")
    repro_path = os.path.join(ROOT, "sweep_repro.jsonl")
    if not os.path.exists(main_path):
        print(f"ERROR: missing {main_path}; run --phase main first")
        sys.exit(1)
    # Idempotency: unlike phase_main, this phase appends to sweep_repro.jsonl
    # without clearing it, so a re-run would add duplicate repeats and bias
    # phase_analyze toward whichever candidates were rerun more often. Clear on
    # --force, otherwise skip already-recorded (cell, tile, repeat) entries.
    if args.force and os.path.exists(repro_path):
        os.remove(repro_path)
    existing = {
        (
            r["preset"], r["sl"], r["causal"],
            r["tile_m"], r["tile_n"], r["num_stages"],
            r.get("repeat", 0),
        )
        for r in read_jsonl(repro_path)
    }
    main_rows = read_jsonl(main_path)
    by_cell = defaultdict(list)
    for r in main_rows:
        if r.get("status") != "ok":
            continue
        cell = (r["preset"], r["sl"], r["causal"])
        by_cell[cell].append(r)

    cells = list(cells_iter())
    t0 = time.time()
    for idx, (name, H_q, H_kv, D, sl, causal) in enumerate(cells, 1):
        cell = (name, sl, causal)
        rows = sorted(by_cell.get(cell, []), key=lambda r: r["tflops"], reverse=True)[:3]
        for r in rows:
            tm, tn, ns = r["tile_m"], r["tile_n"], r["num_stages"]
            for repeat in range(2):
                rkey = (name, sl, causal, tm, tn, ns, repeat)
                if rkey in existing:
                    continue
                rr = run_measure(name, sl, causal, tm, tn, ns,
                                 skip_correctness=True)  # already validated in main
                rr["phase"] = "repro"
                rr["repeat"] = repeat
                rr["head_dim"] = D
                rr["qhead_per_kvhead"] = H_q // H_kv
                append_jsonl(repro_path, rr)
                status = rr.get("status", "")
                tflops = rr.get("tflops", "")
                ms = rr.get("median_ms", "")
                print(f"  [{idx:2d}/{len(cells)}] {name:14s} sl={sl:5d} c={causal} "
                      f"({tm:3d},{tn:3d},ns={ns}) [r{repeat}] {ms!s:>10} ms "
                      f"{tflops!s:>8} TF {status:18s} [{rr['elapsed_s']}s]",
                      flush=True)
    dt = time.time() - t0
    print(f"\nphase=repro done in {dt/60:.1f} min  -> {repro_path}")


def phase_analyze(args):
    """Combine main + repro, pick winners, emit bwd_lookup_candidate.py."""
    main_path = os.path.join(ROOT, "sweep_main.jsonl")
    repro_path = os.path.join(ROOT, "sweep_repro.jsonl")

    by_cand = defaultdict(list)  # (preset, sl, causal, tm, tn, ns) -> [tflops...]
    for path in [main_path, repro_path]:
        for r in read_jsonl(path):
            if r.get("status") != "ok" or "tflops" not in r:
                continue
            k = (r["preset"], r["sl"], r["causal"],
                 r["tile_m"], r["tile_n"], r["num_stages"])
            by_cand[k].append(float(r["tflops"]))

    # Per-cell winner
    by_cell = defaultdict(list)
    for k, tflops_list in by_cand.items():
        preset, sl, causal, tm, tn, ns = k
        mean = statistics.mean(tflops_list)
        if len(tflops_list) >= 2:
            stdev = statistics.stdev(tflops_list)
            cv_pct = stdev / mean * 100.0
        else:
            cv_pct = None
        by_cell[(preset, sl, causal)].append({
            "tile_m": tm, "tile_n": tn, "num_stages": ns,
            "mean_tflops": mean, "median_tflops": statistics.median(tflops_list),
            "n_runs": len(tflops_list), "cv_pct": cv_pct,
            "tflops_runs": tflops_list,
        })

    winners = []
    BASELINE_PREFER_MARGIN = 0.02  # within 2% of baseline -> keep baseline
    HIGH_VAR_THRESHOLD = 10.0      # %

    for (name, H_q, H_kv, D) in PRESETS:
        for sl in SEQLENS:
            for causal in CAUSAL_OPTS:
                cell = (name, sl, causal)
                rows = by_cell.get(cell, [])
                if not rows:
                    continue
                baseline_tile = sm120_baseline_for(D)
                baseline_row = next((r for r in rows
                                     if (r["tile_m"], r["tile_n"], r["num_stages"]) == baseline_tile),
                                    None)
                baseline_mean = baseline_row["mean_tflops"] if baseline_row else None

                trusted = [r for r in rows if r["n_runs"] >= 2]
                trusted.sort(key=lambda r: r["mean_tflops"], reverse=True)

                flag_high_var = False
                if trusted and trusted[0]["cv_pct"] is not None and trusted[0]["cv_pct"] > HIGH_VAR_THRESHOLD:
                    flag_high_var = True

                if not trusted:
                    winner = baseline_row if baseline_row else (rows[0] if rows else None)
                elif flag_high_var:
                    winner = baseline_row if baseline_row else trusted[0]
                else:
                    cand = trusted[0]
                    if baseline_mean is not None and cand["mean_tflops"] < baseline_mean * (1.0 + BASELINE_PREFER_MARGIN):
                        winner = baseline_row
                    else:
                        winner = cand
                if winner is None:
                    continue
                winners.append({
                    "preset": name, "sl": sl, "causal": causal,
                    "head_dim": D, "qhead_per_kvhead": H_q // H_kv,
                    "best_tile_m": winner["tile_m"], "best_tile_n": winner["tile_n"],
                    "best_num_stages": winner["num_stages"],
                    "best_mean_tflops": winner["mean_tflops"],
                    "best_cv_pct": winner["cv_pct"],
                    "baseline_mean_tflops": baseline_mean,
                    "speedup_vs_baseline": (winner["mean_tflops"] / baseline_mean) if baseline_mean else None,
                    "high_variance_flag": flag_high_var,
                })

    winners_path = os.path.join(ROOT, "winners.json")
    with open(winners_path, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"Wrote {len(winners)} winners to {winners_path}")

    # Emit a python lookup table
    lookup = {}
    for w in winners:
        if w["speedup_vs_baseline"] is None:
            continue
        # Only include cells where override BEATS baseline by margin
        if w["speedup_vs_baseline"] <= 1.0 + BASELINE_PREFER_MARGIN:
            continue
        key = (w["head_dim"], w["qhead_per_kvhead"], w["sl"], w["causal"])
        val = (w["best_tile_m"], w["best_tile_n"], w["best_num_stages"])
        lookup[key] = val

    cand_path = os.path.join(ROOT, "bwd_lookup_candidate.py")
    with open(cand_path, "w") as f:
        f.write("# Auto-generated by bench_master_bwd.py.\n")
        f.write("# (head_dim, qhead_per_kvhead, seqlen, causal): (tile_m, tile_n, num_stages)\n")
        f.write("_SM120_BWD_TILE_LOOKUP = {\n")
        for key, val in sorted(lookup.items()):
            f.write(f"    {key}: {val},\n")
        f.write("}\n")
    print(f"Wrote {len(lookup)} tuned entries to {cand_path}")

    # Quick summary
    n_improved = sum(1 for w in winners
                     if w["speedup_vs_baseline"] is not None
                     and w["speedup_vs_baseline"] > 1.02)
    print(f"\nSummary: improved (>2%): {n_improved}/{len(winners)}")
    speedups = [w["speedup_vs_baseline"] for w in winners
                if w["speedup_vs_baseline"] is not None]
    if speedups:
        import math
        geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"  Geomean speedup: {geomean:.4f}x")


def phase_validate(args):
    """Paired validation: run baseline AND tuned (lookup-based) N times each per
    cell, compute per-cell ratio and overall geomean + regression count.

    Loads the tuned tiles from bwd_lookup_candidate.py. Cells not in the
    lookup use the baseline tile for *both* runs (so their ratio is ~1.0 by
    construction). Cells in the lookup use baseline vs tuned.
    """
    import importlib.util
    cand_path = os.path.join(ROOT, "bwd_lookup_candidate.py")
    if not os.path.exists(cand_path):
        print(f"ERROR: missing {cand_path}; run --phase analyze first")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("bwd_lookup_candidate", cand_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    LOOKUP = mod._SM120_BWD_TILE_LOOKUP

    val_path = os.path.join(ROOT, "validate.jsonl")
    if args.force and os.path.exists(val_path):
        os.remove(val_path)
    # Include the tuned tile in the resume key: if bwd_lookup_candidate.py
    # changes between runs, "tuned" rows with a different (tile_m, tile_n,
    # num_stages) must NOT be treated as cache hits, or the report would
    # compare the new lookup against stale measurements.
    existing = {
        (
            r["preset"],
            r["sl"],
            r["causal"],
            r.get("mode_label") or r.get("mode"),
            r["tile_m"],
            r["tile_n"],
            r["num_stages"],
            r.get("repeat", 0),
        )
        for r in read_jsonl(val_path)
    }

    cells = list(cells_iter())
    t0 = time.time()
    for idx, (name, H_q, H_kv, D, sl, causal) in enumerate(cells, 1):
        qpkv = H_q // H_kv
        key = (D, qpkv, sl, causal)
        baseline_tile = sm120_baseline_for(D)
        tuned_tile = LOOKUP.get(key, baseline_tile)

        for mode in ("baseline", "tuned"):
            tile = baseline_tile if mode == "baseline" else tuned_tile
            tm, tn, ns = tile
            for repeat in range(args.N):
                rkey = (name, sl, causal, mode, tm, tn, ns, repeat)
                if rkey in existing:
                    continue
                r = run_measure(name, sl, causal, tm, tn, ns,
                                use_baseline=(mode == "baseline"),
                                skip_correctness=True)
                r["phase"] = "validate"
                r["mode_label"] = mode
                r["repeat"] = repeat
                r["head_dim"] = D
                r["qhead_per_kvhead"] = qpkv
                append_jsonl(val_path, r)
                status = r.get("status", "")
                ms = r.get("median_ms", "")
                tflops = r.get("tflops", "")
                print(f"  [{idx:2d}/{len(cells)}] {name:14s} sl={sl:5d} c={causal} "
                      f"{mode:8s} ({tm:3d},{tn:3d},ns={ns}) [r{repeat}] {ms!s:>10} ms "
                      f"{tflops!s:>8} TF {status:18s} [{r['elapsed_s']}s]",
                      flush=True)
    dt = time.time() - t0
    print(f"\nphase=validate done in {dt/60:.1f} min  -> {val_path}")
    _analyze_validation(val_path, LOOKUP, args)


def _analyze_validation(val_path, lookup, args):
    rows = read_jsonl(val_path)
    by_cell_mode = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("status") != "ok":
            continue
        cell = (r["preset"], r["sl"], r["causal"])
        by_cell_mode[cell][r["mode_label"]].append(float(r["tflops"]))

    import math
    per_cell = []
    for (name, H_q, H_kv, D) in PRESETS:
        for sl in SEQLENS:
            for causal in CAUSAL_OPTS:
                cell = (name, sl, causal)
                if cell not in by_cell_mode:
                    continue
                base = by_cell_mode[cell].get("baseline", [])
                tuned = by_cell_mode[cell].get("tuned", [])
                if not base or not tuned:
                    continue
                base_mean = statistics.mean(base)
                tuned_mean = statistics.mean(tuned)
                ratio = tuned_mean / base_mean
                key = (D, H_q // H_kv, sl, causal)
                in_lookup = key in lookup
                per_cell.append({
                    "preset": name, "sl": sl, "causal": causal,
                    "in_lookup": in_lookup,
                    "baseline_tflops_mean": base_mean,
                    "tuned_tflops_mean": tuned_mean,
                    "ratio_tuned_over_baseline": ratio,
                })

    geomean = math.exp(sum(math.log(r["ratio_tuned_over_baseline"]) for r in per_cell) / len(per_cell)) if per_cell else float("nan")
    regressions_2pct = [r for r in per_cell if r["ratio_tuned_over_baseline"] < 0.98]
    lookup_cells = [r for r in per_cell if r["in_lookup"]]
    lookup_geomean = (
        math.exp(sum(math.log(r["ratio_tuned_over_baseline"]) for r in lookup_cells) / len(lookup_cells))
        if lookup_cells else float("nan")
    )
    out = {
        "n_cells_total": len(per_cell),
        "n_cells_in_lookup": len(lookup_cells),
        "geomean_tuned_over_baseline_ALL": geomean,
        "geomean_tuned_over_baseline_LOOKUP_ONLY": lookup_geomean,
        "n_regressions_gt_2pct": len(regressions_2pct),
        "regressions": [
            {"preset": r["preset"], "sl": r["sl"], "causal": r["causal"],
             "ratio": r["ratio_tuned_over_baseline"]}
            for r in regressions_2pct
        ],
        "per_cell": per_cell,
    }
    rep_path = os.path.join(ROOT, "validation_report.json")
    with open(rep_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nValidation summary:")
    print(f"  cells:           {out['n_cells_total']}")
    print(f"  in lookup:       {out['n_cells_in_lookup']}")
    print(f"  geomean (all):   {geomean:.4f}x")
    print(f"  geomean (lookup): {lookup_geomean:.4f}x")
    print(f"  regressions >2%: {out['n_regressions_gt_2pct']}")
    if regressions_2pct:
        for r in regressions_2pct:
            print(f"    {r['preset']:14s} sl={r['sl']:5d} c={r['causal']}  "
                  f"ratio={r['ratio_tuned_over_baseline']:.4f}")
    print(f"  full report:     {rep_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["main", "repro", "analyze", "validate", "all"],
                    default="all")
    ap.add_argument("--force", action="store_true", help="overwrite existing results")
    ap.add_argument("--N", type=int, default=3, help="repeats for --phase validate")
    args = ap.parse_args()
    if args.phase in ("main", "all"):
        phase_main(args)
    if args.phase in ("repro", "all"):
        phase_repro(args)
    if args.phase in ("analyze", "all"):
        phase_analyze(args)
    if args.phase in ("validate", "all"):
        phase_validate(args)
