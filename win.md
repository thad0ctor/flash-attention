# SM120 Win Ledger

Date started: 2026-05-28

This file tracks the best known SM120 commits, dispatch configs, benchmark
artifacts, and rejected experiments. Keep it updated as tuning continues so
future agents compare against the right winner instead of rediscovering noisy
or reverted paths.

## Update Protocol

- Add new entries with commit SHA, shape family, dispatch config, benchmark
  command or artifact path, FA4/FA2 ratio, and reliability label.
- Do not replace a winner from a single noisy run. Mark it as `candidate` until
  it survives paired repeats or an interleaved A/B run.
- Mark rejected paths explicitly, including the artifact or PR comment that
  rejected them.
- If a broad sweep predates a targeted follow-up, keep both and say which rows
  are superseded.
- For any GPU benchmark, pin the intended card first:
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  - `CUDA_VISIBLE_DEVICES=GPU-f423fa54-c41a-719d-89ca-e09ae9c1826a`
  - verify Python sees exactly one GPU: RTX 5090, UUID
    `GPU-f423fa54-c41a-719d-89ca-e09ae9c1826a`
  - check `nvidia-smi pmon -c 1` and do not benchmark if active training is
    using the machine unless the user explicitly clears it.

## Current Reference Points

| Scope | Commit/artifact | Result | Notes |
|---|---|---:|---|
| 80-cell fwd+bwd broad snapshot | `9b46c42`, `/tmp/sm120_fa2_fa4_d2d0ec2_20260526_0800`, `AI/SM120_CURRENT_PERF_2026_05_26.md` | overall 1.0226x, fwd 1.0248x, bwd 1.0204x | Current winner marker after restoring packed GQA forward path. |
| Latest broad forward table in repo | `agent_space/sm120_full_sweep_20260528_fwd_table.md` | 130 pairs, geomean 1.025512, median 1.016456, wins 83/130 | Good trend table, but at least qwen3-14b S=16384 causal is superseded by `532331f` targeted data below. |
| Focused Qwen current-vs-history check | `/tmp/sm120_qwen_commit_compare_focused_20260528` | current geomean 1.030644, median 1.029464, wins 6/7 | Compared `e65b67a`, `6b77ace`, and current. |
| Short/mid Qwen current-vs-history check | `/tmp/sm120_qwen_commit_compare_shortmid_20260528` | current geomean 1.034586, median 1.020806, wins 41/54 | Current beat earlier candidate commits in aggregate. |
| Current long qpkv5 targeted check | `532331f`, `/tmp/sm120_qwen_commit_compare_long_qpkv5_64x128_20260528` | geomean 1.048624, median 1.038903, wins 4/4 | Current best for qwen3-14B D128 qpkv5 causal S>=16384. |

## Best Known Winners

| Family | Best commit/config | Evidence | Reliability |
|---|---|---|---|
| qwen3-14B D128 qpkv5 causal S=16384 | `532331f`, lookup `(128, 5, 16384, 1): (64, 128, 1)` | Targeted long qpkv5 repeat: S=16384 FA4/FA2 1.088882, long set 4/4 wins, geomean 1.048624. Correctness max abs 0.00390625; ruff, diff check, 15 pytest passed. | keeper |
| qwen3-14B D128 qpkv5 causal S>=32768 | `48c7d4d`, `128x128`, 256 threads, `Q_in_regs=True` | PR repeat: S32768 1.0223, S65536 1.0048, S131072 1.0306; qpkv5 causal geomean 1.0173. Current `532331f` keeps this path and improves S=16384. | keeper |
| qwen3-14B D128 qpkv5 causal S=8192 | `6b77ace`, `128x128`, 256 threads, `Q_in_regs=True` | Env-gated A/B: baseline 0.953/0.986 vs candidate 1.053/1.050. Dispatcher validation 1.055. Full 78-cell run after patch geomean 1.0526, touched row 1.019. | keeper, but rerun if broad noise changes |
| qwen3.5/qwen3.6 27B D256 qpkv6 dense | `6459f7d`, narrow SM120 load-overlap hooks, `64x64`, `num_stages=1`, nonlocal, no score/mask mod | Focused PR rows: S4096 c 1.0495, S4096 nc 1.0546, S8192 c 1.0241, S8192 nc 1.0165, S16384 c 1.0402, S16384 nc 1.0141. NCU S4096 causal removed 2,082,240 spill inst and beat FA2. | keeper |
| Gemma D256 local qpkv4/qpkv8 | `3dc3f22`, local `64x16` narrow-N path | Local-only Gemma sweep geomean 1.0654, median 1.0793, wins 5/6. e4b S4096 1.108, S8192 1.147; e2b S4096 1.083, S8192 1.076. | keeper for local rows; broad aggregate noisy |
| qwen3-30B D128 qpkv8 short/mid causal | Current retained qpkv8 path, with `4c0b291`/`6b77ace` as historical comparison points | Old repeats: S1024 causal 1.288-1.301, S4096 causal 1.039-1.050, S8192 causal 1.010-1.029. Commit compare qpkv8-only was mixed: `4c0b291` 1.0107, `6b77ace` 1.0165, pre-patch current 0.9930. Later strict r41 current run was positive: geomean 1.026586, wins 5/6. | noisy candidate; do not change dispatch without strict paired repeat |
| qwen2.5 D128 dense noncausal TMA mask skip | `0af9a4c`, static noncausal TMA seqlen-mask skip | NCU Qwen2.5 S8192 noncausal: 10.56 ms -> 10.18 ms, instructions 2.064B -> 1.866B. Repeats median 0.975, mean 0.987, range 0.971-1.031. | historical, partly superseded |
| backward d<=64 | `4d59090`, SM120 backward default `num_stages` 2 -> 1 | Phase 17C reported +5.6% on d<=64 cells, arch-gated. | keeper |
| backward broad Phase 17 | `362a65a` + `55ab672`, 8 warps/block and v4 atomic dQ/dK/dV | Phase 17 backward 40-cell FA4/FA2 geomean 1.017x, 29/40 wins, peak 180.8 TFLOPS; was 0.93x and 10/40 wins before. | keeper |

## Rejected Or Noisy Paths

| Path | Outcome |
|---|---|
| qpkv6 D256 causal `64x48` and Q-in-reg variants | One run looked good, repeats rejected. Prior experiment had qpkv6 S4096 causal 0.929 despite a noisy 78-cell aggregate win. Do not restore without fresh paired evidence. |
| qwen3-14B qpkv5 S8192 noncausal `64x128` / `64x112` candidates | Interleaved A/B rejected; current S8192 causal keeper is `6b77ace`, noncausal remains sensitive/noisy. |
| D256 qpkv6 exact-shape TMA `kv_stages=1` | Ran around 341.1 ms on qwen3.5/qwen3.6 S65536 causal, slower than CpAsync and FA2. |
| Stream-PV prototype avoiding full bf16 `rP` | Compiled and ran, but 3 paired reps geomean 0.958x. Reverted. |
| TMA stage retune commits `4e895b5`, `ec6dd92`, `ca7f03e`, `08c184f` | Reverted by `6a8f9ab` after long-regression/noise. Do not restore wholesale. |
| Broad qpkv8 old-commit restore | Current strict qpkv8 causal remains positive, but noncausal/noisy rows do not justify a broad revert. Profile exact rows before changing dispatch. |
| Backward tile sweep `a0dd865`/`47aa883` | Negative by design: d=128 only viable tile was already default `(64,64,1)`. No `_SM120_BWD_TILE_LOOKUP` shipped. |
| Dense noncausal `check_inf` skip | Tried after `48c7d4d`; qpkv6 S131072 noncausal stayed around 0.966x, so patch was reverted. |

## Review Risks To Keep Separate From Perf Winners

These are CodeRabbit/PR-trail risks noted during the performance pass. They are
not evidence against the dense Qwen/Gemma forward winners above, but future work
should not lose track of them.

- SM120 TMA should avoid `learnable_sink` unless correctness is proven.
- Varlen autograd is missing `mask_mod` coverage.
- Varlen pack-GQA dQ atomics may need per-batch offset handling.
- Paged-KV local masking has `n_block_min` / tail-loop concerns.
- SM120 TMA reuses K copy byte count for V when `D != Dv`; keep this separate
  from dense `D == Dv` tuning claims.

## Open Targets

- Re-run a full 130-cell forward sweep after `532331f`; current broad table is useful
  but not fully current for qwen3-14B S=16384 causal.
- qpkv8 D128: strict probe exists in `agent_space/sm120_qpkv8_probe.py`; test
  exact noncausal rows plus S16384/S32768/S65536 causal before changing lookup.
- D128/qpkv5 small and mid noncausal rows: repeat before patching; several
  apparent misses have flipped with run order and FA2 variance.
- Gemma local qpkv4/qpkv8: keep `64x16`, but rerun broader repeats before
  extending to qpkv2 or dense Gemma shapes.
- D256 qpkv6 long causal/noncausal: current load-overlap path is the best
  committed approach. Further wins likely require source/SASS-level instruction
  pressure reduction, not tile lookup sweeps.
