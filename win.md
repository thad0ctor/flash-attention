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
  - `CUDA_VISIBLE_DEVICES=<RTX_5090_GPU_UUID>`
  - verify Python sees exactly one GPU: RTX 5090, UUID
    `<RTX_5090_GPU_UUID>`
  - check `nvidia-smi pmon -c 1` and do not benchmark if active training is
    using the machine unless the user explicitly clears it.

## Current Reference Points

| Scope | Commit/artifact | Result | Notes |
|---|---|---:|---|
| 80-cell fwd+bwd broad snapshot | `9b46c42`, `/tmp/sm120_fa2_fa4_d2d0ec2_20260526_0800`, `AI/SM120_CURRENT_PERF_2026_05_26.md` | overall 1.0226x, fwd 1.0248x, bwd 1.0204x | Current winner marker after restoring packed GQA forward path. |
| Post-qpkv-update model forward sweep | `3c34c6a`, `/tmp/sm120_model_variants_after_qpkv_updates_20260528b`, `/tmp/sm120_longseq_qwen_gemma_after_qpkv_updates_20260528b` | 130 pairs, geomean 1.041438, median 1.026890, wins/ties/losses 88/6/36 | Current clean combined Qwen/Gemma short-mid + long reference. Short/mid 78-cell geomean 1.061524; long 52-cell geomean 1.012019. |
| Latest broad forward table in repo | `agent_space/sm120_full_sweep_20260528_fwd_table.md` | 130 pairs, geomean 1.025512, median 1.016456, wins 83/130 | Good trend table, but at least qwen3-14b S=16384 causal is superseded by `532331f` targeted data below. |
| Focused Qwen current-vs-history check | `/tmp/sm120_qwen_commit_compare_focused_20260528` | current geomean 1.030644, median 1.029464, wins 6/7 | Compared `e65b67a`, `6b77ace`, and current. |
| Short/mid Qwen current-vs-history check | `/tmp/sm120_qwen_commit_compare_shortmid_20260528` | current geomean 1.034586, median 1.020806, wins 41/54 | Current beat earlier candidate commits in aggregate. |
| Current long qpkv5 targeted check | `532331f`, `/tmp/sm120_qwen_commit_compare_long_qpkv5_64x128_20260528` | geomean 1.048624, median 1.038903, wins 4/4 | Current best for qwen3-14B D128 qpkv5 causal S>=16384. |
| Current 5-repeat broad forward rerun | `b9261e6`, `/tmp/sm120_model_variants_qpkv4_5x_rerun_b9261e6_20260528` | 78 cells, geomean 1.070998, median 1.037421, wins 67/78 | Rerun after rejecting qpkv2 dense probe. Versus prior 5-repeat run: FA4-time geomean old/new 1.011005, FA4/FA2 ratio geomean new/old 1.016971. |

## Best Known Winners

| Family | Best commit/config | Evidence | Reliability |
|---|---|---|---|
| qwen3-14B D128 qpkv5 causal S=16384 | `532331f`, lookup `(128, 5, 16384, 1): (64, 128, 1)` | Targeted long qpkv5 repeat: S=16384 FA4/FA2 1.088882, long set 4/4 wins, geomean 1.048624. Correctness max abs 0.00390625; ruff, diff check, 15 pytest passed. | keeper |
| qwen3-14B D128 qpkv5 causal S>=32768 | `48c7d4d`, `128x128`, 256 threads, `Q_in_regs=True` | PR repeat: S32768 1.0223, S65536 1.0048, S131072 1.0306; qpkv5 causal geomean 1.0173. Current `532331f` keeps this path and improves S=16384. | keeper |
| qwen3-14B D128 qpkv5 causal S=8192 | `6b77ace`, `128x128`, 256 threads, `Q_in_regs=True` | Env-gated A/B: baseline 0.953/0.986 vs candidate 1.053/1.050. Dispatcher validation 1.055. Full 78-cell run after patch geomean 1.0526, touched row 1.019. | keeper, but rerun if broad noise changes |
| qwen3.5/qwen3.6 27B D256 qpkv6 dense | `6459f7d`, narrow SM120 load-overlap hooks, `64x64`, `num_stages=1`, nonlocal, no score/mask mod | Focused PR rows: S4096 c 1.0495, S4096 nc 1.0546, S8192 c 1.0241, S8192 nc 1.0165, S16384 c 1.0402, S16384 nc 1.0141. NCU S4096 causal removed 2,082,240 spill inst and beat FA2. | keeper |
| Gemma D256 local qpkv4/qpkv8 | `3dc3f22`, local `64x16` narrow-N path | Local-only Gemma sweep geomean 1.0654, median 1.0793, wins 5/6. e4b S4096 1.108, S8192 1.147; e2b S4096 1.083, S8192 1.076. | keeper for local rows; broad aggregate noisy |
| qwen3-30B D128 qpkv8 short/mid | post-`c6051aa` local retune: S8192 noncausal `128x32`, S8192 causal `128x64`; `4c0b291`/`6b77ace` remain historical comparison points | Patch validation: `/tmp/sm120_qwen_qpkv8_patch_128x64c_dispatch_20260528b` geomean 1.025994, wins 5/6; repeat `/tmp/sm120_qwen_qpkv8_patch_128x64c_dispatch_r2_20260528b` geomean 1.038794, wins 5/6. S8192 noncausal 1.024/1.039; S8192 causal 1.012/1.013. Old mixed compare: `4c0b291` 1.0107, `6b77ace` 1.0165, pre-patch current 0.9930. | keeper for S8192 qpkv8; S1024 noncausal remains tiny/noisy |
| Qwen D128 qpkv4 short/mid | `ced523f`: S1024 noncausal/causal `64x64`; S8192 noncausal `128x32`; S8192 causal `128x64` | Subprocess public-API validation `/tmp/sm120_qpkv4_lookup_patch_narrow_validation_20260528`: 12 affected qwen3-embedding/qwen3-vl rows, geomean 1.081840 vs FA2, wins 12/12. FA4 time geomean vs saved post-qpkv artifact: 1.057589; correctness max_abs <= 0.00390625 on changed S1024/S8192 paths. Broad 78-cell 5-repeat rerun `/tmp/sm120_model_variants_qpkv4_5x_rerun_b9261e6_20260528`: qpkv4 D128 S1024/S8192 rows all mean wins; qwen3-vl S8192 causal 1.039250 and 4/5 wins. | keeper; S4096 was explicitly reverted to prior lookup |
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
| SM80-base qpkv6 hook-path `utils.cvt_f16(acc_S, rP)` conversion | Tried after `60354fd` and reverted locally. Focused dirty run `/tmp/sm120_qwen_focused_qpkv6_cvt_hook_dirty_20260528b` dropped D256 qpkv6 causal geomean to 1.015849 vs prior current reference 1.025739, mainly hurting S16384 causal. |
| D256 qpkv6 hook scheduling K-only / V-only / off | Profiling-only env selector tested then reverted. NCU at S4096 causal: V-only was fastest in one profile (2.312 ms vs both 2.327 ms), but focused timing rejected changing the default: D256 qpkv6 causal geomean both-hooks 1.079767, V-only 1.049291, K-only 1.036348, hooks-off 0.985875. Artifacts: `/tmp/sm120_qpkv6_hook_ab_ncu_20260528b`, `/tmp/sm120_qwen_focused_hook_{both,v,off,k}_dirty_20260528b`. |
| D256 qpkv6 first-load predicate / causal first-tile mask / SM120 softmax reductions | Rejected after dirty probes and reverted. Combined predicate+mask improved one S4096 run but regressed S8192/S16384; load-predicate-only had a tiny NCU duration win (2.328 ms -> 2.323 ms) but repeated timing regressed S4096; softmax-120 reduced instructions but added writeback and did not improve NCU duration. Artifacts: `/tmp/sm120_qwen_focused_predmask_dirty_20260528b`, `/tmp/sm120_qwen_focused_loadpred_dirty_20260528b`, `/tmp/sm120_qwen_focused_softmax120_dirty_20260528b`, `/tmp/sm120_qpkv6_s4096_loadpred_ncu_20260528b`. |
| Qwen D128 qpkv4 S4096 lookup changes | Rejected in the qpkv4 lookup patch. Interleaved explicit-tile timing suggested S4096 wins, but subprocess public-API validation showed FA4-time regressions versus the saved post-qpkv artifact, so S4096 noncausal stayed `64x64` and S4096 causal stayed `64x96`. |
| Gemma31 D256 qpkv2 dense causal `64x48` | Interleaved timing looked mildly positive for S4096/S8192 causal, but public-API 5-repeat validation rejected shipping it: S4096 causal mean 0.989660 and S8192 causal median 0.979706. Noncausal must remain `64x64` for dense mask-skip behavior. Artifacts: `/tmp/sm120_gemma31_qpkv2_dense_condition_ab_20260528`, `/tmp/sm120_gemma31_qpkv2_patch_validation_20260528`. |
| qwen3.5-9B S4096 noncausal D256 qpkv4 and gemma4-e2b S8192 noncausal D256 qpkv8 broad-repeat misses | No dispatch patch. A later public-API 10-repeat check flipped both apparent 5-repeat misses into wins: qwen3.5-9B S4096 noncausal mean 1.015043 / median 1.024537, wins 6/10; gemma4-e2b S8192 noncausal mean 1.020579 / median 1.029481, wins 8/10. The matching interleaved A/B run mostly favored the existing `64x64` path; qwen S4096 `64x48` was only a tiny distinct hint and would disable dense mask-skip divisibility. Artifacts: `/tmp/sm120_broad_miss_qpkv4_qpkv8_condition_ab_20260528`, `/tmp/sm120_broad_miss_public_repeat_20260528`. |

## Review Risks To Keep Separate From Perf Winners

These are CodeRabbit/PR-trail risks noted during the performance pass. They are
not evidence against the dense Qwen/Gemma forward winners above, but future work
should not lose track of them.

- SM120 TMA `learnable_sink` dispatch is gated off in the CodeRabbit fix set
  after `3c34c6a`.
- Varlen autograd now propagates `mask_mod` in the CodeRabbit fix set after
  `3c34c6a`.
- Varlen pack-GQA dQ atomics now include the per-batch padded Q offset, and
  SM120 backward varlen/seqused explicit pack falls back to the nonpacked GQA
  path until the full packed varlen path is validated.
- Paged-KV local masking now stops the paged unmasked loop at `n_block_min` in
  the CodeRabbit fix set after `3c34c6a`.
- SM120 TMA now tracks V copy byte count separately from K when `D != Dv`; keep
  this separate from dense `D == Dv` tuning claims.

## Open Targets

- Current broad reference is the post-qpkv-update 130-cell sweep at `3c34c6a`.
  Re-run it after any further dispatch/kernel changes, not before.
- qpkv8 D128: S8192 dispatch is retuned; if continuing this family, test
  S16384/S32768/S65536 causal and avoid broad old-commit restores.
- D128/qpkv5 small and mid noncausal rows: repeat before patching; several
  apparent misses have flipped with run order and FA2 variance.
- Gemma local qpkv4/qpkv8: keep `64x16`, but rerun broader repeats before
  extending to qpkv2 or dense Gemma shapes.
- D256 qpkv6 long causal/noncausal: current load-overlap path is the best
  committed approach. Further wins likely require source/SASS-level instruction
  pressure reduction, not tile lookup sweeps. Current post-`60354fd` NCU for
  qwen3.5/qwen3.6 D256 qpkv6 S4096 causal is under
  `/tmp/sm120_qpkv6_ncu_after_qpkv8_20260528b`: FA4 2.328 ms / 419.4M SM
  instructions / 255 regs/thread, FA2 2.365 ms / 323.8M SM instructions /
  255 regs/thread.
