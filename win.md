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
| Post-qpkv5-hook 5-repeat forward sweep | `effed34`, `/tmp/sm120_model_variants_after_qpkv5_hooks_5x_20260529` | 78 cells, mean-ms geomean 1.048271, median-ms geomean 1.051726, median-ms wins 64/78 | Current-state reference after qpkv5 hook policy. Contains obvious one-repeat outliers, so use row medians/repeat ranges before treating a miss as actionable. |
| Current long Qwen/Gemma forward sweep | `a11ddad` + qpkv8 long patches, `/tmp/sm120_longseq_qwen_gemma_after_qpkv8_long_20260529` | 52 cells, geomean 1.022723, median 1.010010, wins 31/52. Qwen geomean 1.005650, Gemma geomean 1.081755. qwen3-30B qpkv8 causal improved to S16384 1.052, S32768 1.048, S65536 1.023; S131072 remains a tiny 0.994 miss. | current long forward reference; one-repeat table, rerun targeted misses before patching |
| Post-qpkv5-S16384-qregs long Qwen/Gemma sweep | `d6dc1e9`, `/tmp/sm120_longseq_qwen_gemma_after_qregs_20260529` | 52 cells, geomean 1.013534, median 1.014381, wins 33/52. Qwen geomean 1.008702, Gemma geomean 1.029809. S=16384/32768/65536/131072 geomeans 1.005973/1.019813/1.011184/1.017223. | qpkv5 S16384 causal is a strong 1.093x win; qpkv8 S32768/S65536 causal broad misses required focused repeats before changing dispatch |
| Post-qpkv8-long 5-repeat broad forward sweep | `0d709fb`, `/tmp/sm120_model_variants_current_after_long_5x_20260529` | 78 cells, geomean 1.063193, median 1.037063, wins 66/78. D128 1.052319, D256 1.068062, qwen 1.044741, gemma 1.105912. | current short/mid sanity check after qpkv8 long keepers; comparable to the `d6bdf7e` 10-repeat keeper, but use targeted repeats for the remaining qpkv5/qpkv8 S4096 misses |
| Post-D256-qregs long Qwen/Gemma sweep | current patch, `/tmp/sm120_longseq_qwen_gemma_after_d256_qregs_20260529` | 52 cells, geomean 1.031068, median 1.016116, wins 38/52. Qwen geomean 1.021863, Gemma geomean 1.062352. S=16384/32768/65536/131072 geomeans 1.024305/1.024756/1.031990/1.043334. | superseded as current long reference by `246a798`, but still the pre-qpkv8-NC comparison point and D256 qregs evidence |
| Post-D128-qpkv8-NC long Qwen/Gemma sweep | `246a798`, `/tmp/sm120_longseq_qwen_gemma_after_d128_qpkv8_nc_20260529` | 52 cells, geomean 1.020915, median 1.010156, wins 34/52. Qwen geomean 1.018785, Gemma geomean 1.028046. D128 1.009449, D256 1.026053. S=16384/32768/65536/131072 geomeans 1.022460/1.015963/1.019362/1.025901. qwen3-30B D128 qpkv8 noncausal S16384/S32768/S65536/S131072 = 1.062/1.020/1.011/1.009. | newest long forward reference after qpkv8 long-noncausal `128x32`; lower geomean vs D256-qregs is mostly FA2/run variance because common-row FA4 medians improved 1.003971x while FA2 medians improved 1.013956x |
| Post-qpkv6-static long Qwen/Gemma sweep | `fcb6352`, `/tmp/sm120_longseq_qwen_gemma_after_static_causal_20260529` | 52 cells, geomean 1.037399, median 1.028160, wins 41/52. Qwen geomean 1.026698, Gemma geomean 1.073882. S=16384/32768/65536/131072 geomeans 1.039001/1.046225/1.025527/1.038951. qpkv6-only long FA2/FA4 after static: `/tmp/sm120_qpkv6_long_fa2fa4_after_static_20260529`, geomean 1.006064, wins 6/8. | current long forward reference after qpkv6 causal static-block keeper; one-repeat long table, validate any new row with paired A/B before defaulting |
| Post-qpkv8-D256-qregs long Qwen/Gemma sweep | `517282c`, `/tmp/sm120_longseq_qwen_gemma_after_qpkv8_d256_qregs_20260529` | 52 cells, geomean 1.045304, median 1.023615, wins 43/52. Qwen geomean 1.027649, Gemma geomean 1.106376. S=16384/32768/65536/131072 geomeans 1.038543/1.044506/1.028850/1.069754. | current pre-qpkv6-qregs long reference; qpkv6 D256 rows below are superseded by the new qpkv6 Q-regs keeper |
| Post-qpkv6-Qregs long Qwen/Gemma sweep | `77a7cb4`, `/tmp/sm120_longseq_qwen_gemma_after_qpkv6_qregs_20260529` | 52 cells, geomean 1.037726, median 1.049108, wins 39/52. qpkv6 D256 long rows all beat FA2 after the Q-in-regs keeper. | use targeted repeats for apparent local/qpkv8 one-repeat outliers; qpkv6 rows are superseded winners |
| Focused outlier repeat after qpkv8 causal extension | `fcc906b`, `/tmp/sm120_focused_outlier_repeat_after_qpkv8_qregs_20260529` | 9 rows, geomean 1.018156, wins 5/9. Gemma e2b S131072 local qpkv8 flipped to 1.1377x; qpkv8 D256 causal S65536/S131072 stayed at 1.0177x/1.0767x. | S32768 qpkv8 D256 causal was an outlier in this script; same public auto path validated separately at about 54.1 ms vs 56.5 ms forced-off |
| Current long Qwen/Gemma forward sweep | `1d2d944`, `/tmp/sm120_longseq_qwen_gemma_after_qpkv16_qregs_20260529` | 52 cells, geomean 1.042562, median 1.046580, wins 41/52. Qwen geomean 1.043973, Gemma geomean 1.037872. S=16384/32768/65536/131072 geomeans 1.042475/1.044059/1.011395/1.073239. | latest long forward reference after qpkv6/qpkv8/qpkv16 D256 Q-in-regs keepers; repeat apparent misses before patching |
| Current long miss focused repeat | `1d2d944`, `/tmp/sm120_current_long_miss_repeat_after_qpkv16_20260529` | 6 rows, 5 repeats each, geomean 1.036052, median 1.028226, wins 22/30. Gemma e2b S65536 local 1.071x geomean; qpkv8 D256 S16384/S32768 causal 1.052x/1.051x geomean; D128 qpkv8 S16384/S65536 causal is parity/noise at 0.998x/0.999x geomean. | rejects the broad local/qpkv8-D256 misses as one-repeat noise; only D128 qpkv8 remains near parity |
| Post-qpkv16-Qregs 5-repeat broad forward sweep | `93d200b`, `/tmp/sm120_model_variants_after_qpkv16_5x_20260529` | 78 cells, mean-ms geomean 1.052380, median 1.029268, wins 64/78. D128 1.058423, D256 1.049705, qwen 1.041908, gemma 1.076328. Stable-ish mean-ratio rows below 0.99 with non-tiny time: qpkv6 S8192 causal 0.940x, qpkv5 S4096 causal 0.981x, qpkv4 S4096 noncausal 0.982x. | latest short/mid reference after qpkv16 Q-regs keeper. Use repeat range and medians before patching; qpkv6 S8192 causal is the clearest source-level follow-up |
| Post-qpkv6-B2-Qregs 5-repeat broad forward sweep | `43f1ea9`, `/tmp/sm120_model_variants_after_qpkv6_b2_qregs_5x_20260529` | 78 cells, mean-ms geomean 1.046904, median 1.026764, wins 59/78. D128 1.036949, D256 1.051360, qwen 1.038590, gemma 1.065855. Touched qpkv6 rows improved vs the prior 5x: S4096 noncausal 1.004x -> 1.076x; S8192 causal 0.940x -> 1.035x. | current short/mid validation after exact B=2 qpkv6 Q-regs+V keeper; broad geomean is lower due D128/tiny-shape run variance, while D256 is slightly higher |
| Post-qpkv5-nonTMA 5-repeat broad forward sweep | `8320cf4`, `/tmp/sm120_model_variants_after_qpkv5_notma_5x_20260529` | 78 cells, geomean 1.060928, median 1.026603, wins 65/78. D128 1.061520, D256 1.060665, qwen 1.053161, gemma 1.078615. Touched qwen3-14B qpkv5 S4096 noncausal improved from the prior 0.967620x broad miss to 1.013488x by mean ratio. | current short/mid reference after exact qpkv5 S4096 non-TMA gate; broad geomean moved within noise while qwen geomean improved |
| Pause-point 5-repeat broad forward sweep | `cf0a7f0`, `/tmp/sm120_pause_full_fwd_5x_20260529_cf0a7f0` | 78 cells, geomean 1.051918, median 1.021741, wins 47/78. D128 1.025795, D256 1.063741, qwen 1.038774, gemma 1.082105. Stable-ish misses by current artifact: qpkv6 S1024 causal 0.775x, qpkv8 S8192 noncausal 0.899x, qpkv5 S4096 causal 0.919x, qpkv4 S4096 noncausal 0.931x. | noisy pause-point snapshot after rejected follow-up probes; D256/gemma are slightly above the `8320cf4` run, but D128/qwen are lower. Use prior 5x plus this artifact's repeat ranges before declaring a regression. |
| Packed-row fast path and fused-dKV guard focused check | `6125775`, `/tmp/sm120_packgqa_valid_rows_nonc_95x_20260529`, `/tmp/sm120_bwd_d256_s1024_causal_after_dkv_guard_20260529` | D128 packed noncausal fast-valid-row A/B: qpkv4 S4096 +0.67% vs forced-off, qpkv8 S8192 +0.57%, output diff 0. D256 S1024 causal backward smoke now has 24/24 FA2/FA4 pairs with no fused-dKV compile errors; geomean 1.046385, wins 17/24. | focused source-level keeper; broad forward result recorded below |
| Post-valid-rows 5-repeat broad forward sweep | `6125775`, `/tmp/sm120_model_variants_after_validrows_5x_6125775_20260529` | 78 cells, geomean 1.053112, median 1.026819, wins 48/78. D128 1.027592, D256 1.064657, qwen 1.030358, gemma 1.106164. | current pause-point after the fast-valid-row source keeper; overall positive and Gemma/D256 are strong, but `8320cf4` remains the cleaner short/mid broad reference for qwen/D128 and win count. Remaining actionable misses are mostly qpkv6 D256 short causal/noncausal and D128/qpkv4 outliers with large repeat ranges. |
| Current FA2/FA4/SDPA forward sweep | this commit, `/tmp/sm120_sdpa_fa2_fa4_forward_qpkv4_patch_20260528` | 60 cells, FA4/FA2 geomean 1.092207, wins 50/60; FA4/SDPA geomean 0.927178, wins 15/60 | D64 + qpkv4 lookup patches improved the old 60-cell reference from 1.080762 geomean and 41/60 wins. Peak FA4 189.82 TFLOPS in this run. |
| SM120 paged-KV D192/D256 | this commit, `/tmp/sm120_paged_kv_hdgt128_bench_packoff_20260528` | New paged-KV coverage for head_dim 192/256. Full SM120 paged-KV suite: 51/51 pass. D256 B=2 qpkv4 paged/contiguous median ratio: S1024 noncausal 1.015x, S1024 causal 1.032x, S4096 noncausal 0.981x, S4096 causal 0.974x. | feature keeper; performance is near-contiguous without unpacking KV |
| SM120 D256 backward functional baseline | current D256 alias path, `/tmp/sm120_bwd_d256_alias_overlap_3x_20260528`, `/tmp/sm120_bwd_d256_ncu_20260528` | Dense D256 backward now validates for qpkv2/qpkv4/qpkv8 causal and noncausal. 3-repeat S1024 causal smoke vs FA2: geomean 0.901647, wins 2/8. Qwen geomean 0.926900, Gemma geomean 0.861077. NCU qwen3.5-9B S1024 causal main kernel: FA4 479.5 us vs FA2 420.5 us; FA4 uses fewer instructions but launches half the CTA grid. | feature baseline, not a perf winner |
| D256 backward smoke after qpkv8 fused-dKV | `e943809`, `/tmp/sm120_bwd_d256_smoke_after_qpkv8_fused_20260529`, `/tmp/sm120_bwd_d256_smoke_e943809_5x_20260529` | 3-repeat smoke: geomean 0.965548 vs FA2, wins 1/8 by mean. 5-repeat follow-up: geomean 0.932977 by mean, 3/8 wins, median-from-mean 0.950970; dominated by qpkv6/qpkv4 outliers. qpkv16 wins strongly; qpkv4 is near parity by median; qpkv6/qpkv8 remain the short causal drag. | superseded by nonpacked split2 reference below |
| Current D256 backward smoke after nonpacked split2/split3 | `3b276ae`, `/tmp/sm120_bwd_nonpack_split_default_validate_20260529`, `/tmp/sm120_bwd_d256_after_split2_10x_89fe103_20260529`, `/tmp/sm120_bwd_d256_after_split3_10x_3b276ae_20260529` | Exact nonpacked bf16 D256 qpkv6/qpkv8 S1024 causal uses split2 by default, except the small-H Gemma-style qpkv8 Hq8/Hkv1 row now uses split3. Post-split3 10-repeat S1024 causal smoke vs FA2: geomean 1.023134, median 1.026959, wins 49/80; qwen geomean 1.041951, gemma 0.992525. Pre-split3 was geomean 1.018130, median 1.026598, wins 52/80; qwen 1.038795, gemma 0.984598. | current D256 short-causal reference; qpkv8/qpkv6 mainloop CTA gap mostly addressed, unrelated rows remain noisy |
| D256 backward after causal full-valid mask skip | current patch, `/tmp/sm120_bwd_skip_full_causal_mask_default_validate_10x_20260529`, `/tmp/sm120_bwd_d256_s1024_after_maskskip_5x_20260529` | Exact default/off timing: qpkv4 Hq8/Hkv2 +9.1% geomean, 7/10 wins; qpkv6 Hq24/Hkv4 +3.3% geomean, 7/10 wins. 5-repeat S1024 causal FA2/FA4 smoke: geomean 1.016781, median 1.014463, wins 24/40; qwen geomean 1.025748, gemma 1.002008. | current targeted D256 backward source-level keeper; broad smoke remains noisy, use exact default/off artifacts for row-level evidence |
| D256 backward qpkv2 causal mask-skip extension | current patch, `/tmp/sm120_bwd_qpkv2_s1024_maskskip_ab_20260529`, `/tmp/sm120_bwd_d256_s1024_qpkv2_maskskip_3x_20260529`, `/tmp/sm120_bwd_gemma31_qpkv2_s1024_fa2fa4_maskskip_10x_20260529b` | Exact default/off timing for Gemma31 qpkv2 Hq32/Hkv16 S1024 causal: +8.2% geomean, +5.3% median ratio, 7/10 wins. Exact FA2/FA4 10x was noisy: geomean 1.009508, median 0.973922, wins 4/10. 3-repeat S1024 causal smoke stayed noisy overall but moved Gemma31 median above parity. SDPA correctness for exact B=2 S1024 qpkv2: dq/dk/dv max 0.0117/0.0169/0.0271; D256 SDPA subset 18/18 pass. | current qpkv2 self-improvement keeper; do not present as a clean FA2-row win |
| D256 backward qpkv4 mask-skip expansion | current patch, `/tmp/sm120_bwd_qpkv4_h8_b1_maskskip_default_validate_20260529`, `/tmp/sm120_bwd_qpkv4_h16_b2_maskskip_default_validate_20260529`, `/tmp/sm120_bwd_d256_s1024_causal_after_qpkv4_maskskip_20260529` | Exact default/off timing through the public auto-pack route: B=1 Hq8/Hkv2 +5.67% median / +7.62% trimmed mean, 44/64 wins; B=2 Hq16/Hkv4 +1.17% median / +2.19% trimmed mean, 44/64 wins. 5-repeat S1024 causal FA2/FA4 smoke improved to geomean 1.043000, median 1.039382, wins 30/40; qwen geomean 1.055430, gemma 1.022607. | current D256 short-causal reference after qpkv4 mask-skip expansion |
| D256 backward S4096/S2048 pause-point repeats | `5dc126d`, `/tmp/sm120_bwd_d256_s4096_noncausal_current_5x_5dc126d_20260529`, `/tmp/sm120_bwd_d256_s2048_causal_current_5x_5dc126d_20260529`, `/tmp/ncu_sm120_bwd_qpkv4_h8_s2048_causal_fa4_5dc126d_raw.csv`, `/tmp/ncu_sm120_bwd_qpkv4_h8_s2048_causal_fa2_5dc126d_raw.csv` | S4096 noncausal is healthy: 40 pairs, geomean 1.076071, median 1.084512, wins 38/40. S2048 causal is the current D256 backward gap: 40 pairs, geomean 0.983272, median 0.991065, wins 19/40; qwen3.5 qpkv4 Hq8/Hkv2 geomean 0.870047 and Hq16/Hkv4 geomean 0.970827. NCU on qpkv4 Hq8/Hkv2 S2048 causal shows FA4 main about 0.872 ms vs FA2 0.762 ms, with half the CTA grid, lower tensor utilization, and higher long-scoreboard stalls despite fewer instructions/spills. | pause-point reference; S4096 is not the blocker, S2048 qpkv4 needs source-level mainloop work |
| D256 backward qpkv4 S2048 packed split16 keeper | current patch, `/tmp/sm120_bwd_qpkv4_s2048_pack_oversplit_dirty_cd1b3e3_20260529`, `/tmp/sm120_bwd_qpkv4_s2048_auto_split16_validate_20260529`, `/tmp/sm120_bwd_d256_s2048_causal_after_qpkv4_split16_5x_20260529` | Exact B=2 qpkv4 S2048 causal now auto-enables packed split16 for Hq8/Hkv2 and Hq16/Hkv4. Default/off timing: Hq8 +11.1% median / +10.1% mean, Hq16 +10.2% median / +6.2% mean. 5-repeat S2048 causal FA2/FA4 sweep improved from the pause-point 0.983272 geomean and 19/40 wins to 1.034150 geomean, median 1.018967, wins 26/40. qpkv4 Hq8/Hkv2 moved from 0.870047 geomean to 1.068165; Hq16/Hkv4 moved from 0.970827 to 1.009983. Correctness default vs forced-off max diffs on B=2 S2048: out 0, dq 0.00098, dk <= 0.00782, dv <= 0.01563. | current S2048 qpkv4 backward keeper; B=2 exact rows only |
| B=1 D256 split2 validation | `229f87f`, `/tmp/sm120_bwd_nonpack_split_b1_validate_20260529` | B=1 default split2 also beats forced split1 by medians and paired wins: qpkv6 H24 0.544128 ms vs 0.595552 ms with split1 winning only 3/32; qpkv8 H8 0.365056 ms vs 0.414432 ms with split1 winning 4/32; qpkv8 H16 0.467504 ms vs 0.503792 ms with split1 winning 7/32. qpkv8 H16 mean had large default outliers, so keep this as median/pair-win evidence rather than a mean-speedup claim. | confirms split2 is not only a B=2 artifact |

## Best Known Winners

| Family | Best commit/config | Evidence | Reliability |
|---|---|---|---|
| qwen3-14B D128 qpkv5 causal S=16384 | this commit, exact dense bf16 `128x128`, 256 threads, `Q_in_regs=True`, V-only hook | B=1 paired A/B `/tmp/sm120_qpkv5_s16384_qregs_ab_20260529`: qregs/base geomean 1.062964, median 1.069898, wins 8/8; qregs/FA2 geomean 1.036819, median 1.040758, wins 8/8. B=1 default/off `/tmp/sm120_qpkv5_s16384_qregs_default_off_20260529`: default/old geomean 1.031669, median 1.035725, wins 3/4. B=2 default/off `/tmp/sm120_qpkv5_s16384_b2_qregs_default_off_20260529`: default/old geomean 1.060897, median 1.053861, wins 4/4; default/FA2 geomean 1.009635. B=4 forced qregs `/tmp/sm120_qpkv5_s16384_b4_qregs_probe_20260529`: qregs/old geomean 1.060646, wins 4/4; qregs/FA2 geomean 1.031007. B=4 default/off `/tmp/sm120_qpkv5_s16384_b4_qregs_default_off_20260529`: default/old geomean 1.051926, wins 2/2. Output vs prior path max abs 0 on B=1. | keeper, exact fixed dense row; `FLASH_ATTENTION_SM120_QPKV5_S16384_QREGS=off` forces old path for validation |
| qwen3-14B D128 qpkv5 causal S=16384 historical lookup | `532331f`, lookup `(128, 5, 16384, 1): (64, 128, 1)` | Targeted long qpkv5 repeat: S=16384 FA4/FA2 1.088882, long set 4/4 wins, geomean 1.048624. Correctness max abs 0.00390625; ruff, diff check, 15 pytest passed. Superseded for exact B=1 by the qregs row above; still applies to broader/non-exact S16384 cases. | keeper |
| qwen3-14B D128 qpkv5 causal S>=32768 | `48c7d4d`, `128x128`, 256 threads, `Q_in_regs=True` | PR repeat: S32768 1.0223, S65536 1.0048, S131072 1.0306; qpkv5 causal geomean 1.0173. Current `532331f` keeps this path and improves S=16384. | keeper |
| qwen3-14B D128 qpkv5 causal S=8192 | `6b77ace`, `128x128`, 256 threads, `Q_in_regs=True` | Env-gated A/B: baseline 0.953/0.986 vs candidate 1.053/1.050. Dispatcher validation 1.055. Full 78-cell run after patch geomean 1.0526, touched row 1.019. | keeper, but rerun if broad noise changes |
| qwen3-14B D128 qpkv5 causal load-overlap hooks | current patch: exact SM120 bf16 fixed dense qpkv5 D128 causal hook policy, S8192 both hooks, S16384 V-only, S32768/S65536 both hooks, S>=131072 V-only | Paired auto-vs-forced-off timing on RTX 5090: `/tmp/sm120_qpkv5_hook_ab_20260529` showed forced hook wins up to +3.8% on B=2 rows; `/tmp/sm120_qpkv5_hook_policy_b1_long_20260529` and two-label repeat showed shipped auto/off geomean +1.0113 across B=1/B=2 S8192-S131072. New B=1 S16384 repeat `/tmp/sm120_qpkv5_s16384_b1_hook_default_v_validate_20260529`: patched auto 1.010x vs forced-off median, V-only 1.023x vs forced-off, output diff 0. Public check after patch: B1 S16384 causal FA4 15.065 ms vs FA2 15.078 ms (1.001x). Public long qpkv5 old/new run `/tmp/sm120_qwen_qpkv5_hooks_long_compare_20260529`: new FA4/FA2 geomean 1.0252 vs old 1.0113, 3/4 wins. Correctness: forced K/V/both hook test passes against SDPA. | keeper, narrow |
| qwen3-14B D128 qpkv5 S4096 noncausal non-TMA | `8320cf4`: exact fixed dense bf16 B=2 S4096 noncausal qpkv5 routes to SM80-base non-TMA; `FLASH_ATTENTION_SM120_QPKV5_S4096_NC_TMA=stage2` forces old TMA for profiling | Dirty A/B `/tmp/sm120_qpkv5_s4096_nc_tma_ab_20260529`: non-TMA vs TMA +1.05% median and +3.02% mean, FA4/FA2 1.019x median. Repeat `/tmp/sm120_qpkv5_s4096_nc_tma_ab_r2_20260529`: +1.36% median and +4.82% mean, FA4/FA2 1.026x median. Default validation `/tmp/sm120_qpkv5_s4096_nc_default_notma_validate_20260529`: default non-TMA vs forced old TMA +2.62% median and +2.66% mean; public subprocess repeat `/tmp/sm120_qpkv5_s4096_nc_public_default_notma_20260529` is parity vs FA2, 0.9999x geomean. Broad 5x `/tmp/sm120_model_variants_after_qpkv5_notma_5x_20260529`: touched row 1.013488x mean ratio, up from 0.967620x before. | keeper, exact row; fixes the broad qpkv5 S4096 miss without changing qpkv5 tiles |
| qwen3.5/qwen3.6 27B D256 qpkv6 dense | `6459f7d`, narrow SM120 load-overlap hooks, `64x64`, `num_stages=1`, nonlocal, no score/mask mod | Focused PR rows: S4096 c 1.0495, S4096 nc 1.0546, S8192 c 1.0241, S8192 nc 1.0165, S16384 c 1.0402, S16384 nc 1.0141. NCU S4096 causal removed 2,082,240 spill inst and beat FA2. | keeper |
| qwen3.5/qwen3.6 27B D256 qpkv6 long hook split | current patch: exact SM120 bf16 B=1 qpkv6 D256 fixed dense nonlocal, `64x64`, `num_stages=1`; V-only hook for noncausal S16384/S32768/S65536 and causal S32768/S65536; S16384 causal and S131072 stay on the base schedule | Hook-mode A/B `/tmp/sm120_qpkv6_d256_nc_hook_ab_20260529`: V-only/off geomean 1.02067 across S16384/S32768/S65536 noncausal, output diff 0. Causal A/B `/tmp/sm120_qpkv6_d256_causal_hook_ab_20260529` supported V-only for S32768/S65536 causal. S131072 check `/tmp/sm120_qpkv6_d256_s131072_hook_ab_20260529` was mixed, and broad sweep `/tmp/sm120_longseq_qwen_gemma_after_qpkv6_hook_split_20260529` rejected shipping V-only there; final S131072 repeat `/tmp/sm120_qpkv6_d256_hook_split_default_validate3_20260529` keeps default on the base schedule. Default validation `/tmp/sm120_qpkv6_d256_hook_split_default_validate2_20260529`: shipped auto/off geomean 1.01757 over five touched rows; S16384 causal left effectively off after the `both` exception flipped. `FLASH_ATTENTION_SM120_QPKV6_D256_HOOKS={off,k,v,both}` remains available for profiling. | keeper, exact long qpkv6 rows only |
| qwen3.5/qwen3.6 27B D256 qpkv6 causal static block bounds | current patch: exact fixed dense equal-length causal qpkv6 D256 S32768/S65536 uses specialized `n_block_min=0`, `n_block_max=m_block+1`, `n_block_min_causal_local_mask=m_block`; `FLASH_ATTENTION_SM120_QPKV6_D256_STATIC_CAUSAL_BLOCKS=off` forces the old generic `BlockInfo` path | Static-only A/B `/tmp/sm120_qpkv6_d256_static_causal_off_2label_20260529`: S16384/S32768/S65536 median-speedup geomean 1.00736, with all three positive by mean and median. Final default/off validation `/tmp/sm120_qpkv6_d256_static_causal_default_validate_20260529` kept S32768 and S65536 positive: S32768 median +1.76%, mean +0.82%; S65536 median +0.14%, mean +0.42%. S16384 flipped negative in the final validation and is not defaulted. Correctness: `test_sm120_qpkv6_d256_static_causal_blocks_matches_sdpa` passes. | keeper, S32768/S65536 causal only |
| qwen3.5/qwen3.6 27B D256 qpkv6 long Q-regs | current patch: exact SM120 bf16 B=1 Hq24/Hkv4 qpkv6 D256 fixed dense equal-length nonlocal rows S16384/S32768/S65536/S131072 use `128x64`, 256 threads, `Q_in_regs=True`; `FLASH_ATTENTION_SM120_D256_QPKV6_QREGS=off` forces the old path | Focused forced-mode A/B: `/tmp/sm120_d256_qpkv6_shortmid_qregs_ab_20260529` showed `128x64_t256` beating off on S16384/S32768/S65536 causal and noncausal by +2.3% to +5.2%, all output diffs 0. Long rows: `/tmp/sm120_d256_qpkv6_s131072_c_qregs_ab_20260529` causal +9.95% vs off, 31/31 paired wins, +8.56% vs FA2; `/tmp/sm120_d256_qpkv6_s131072_nc_qregs_ab_20260529` noncausal +18.13% vs off, 31/31 paired wins, +16.04% vs FA2. Shipped default validation `/tmp/sm120_d256_qpkv6_qregs_default_auto_off_20260529`: auto beat off on all 8 rows by +1.9% to +18.7% and beat FA2 on all 8 rows. Correctness: `test_sm120_qpkv6_d256_qregs_matches_sdpa` passes causal and noncausal. | keeper; supersedes the qpkv6 long hook/static path for these B=1 long rows |
| qwen3.5/qwen3.6 27B D256 qpkv6 B=2 short/mid Q-regs | current patch: exact SM120 bf16 B=2 Hq24/Hkv4 qpkv6 D256 fixed dense nonlocal rows S4096 noncausal, S8192 noncausal, and S8192 causal use `128x64`, 256 threads, `Q_in_regs=True`; S4096 noncausal and S8192 causal additionally use the V-only load hook. `FLASH_ATTENTION_SM120_D256_QPKV6_QREGS=off` plus `FLASH_ATTENTION_SM120_QPKV6_D256_HOOKS=off` forces the old path. | Initial forced A/B `/tmp/sm120_qpkv6_b2_qregs_plain_ab_20260529`: qregs+V vs off was +3.37% for S4096 noncausal and +9.18% for S8192 causal, output diff 0. Stricter two-label validation `/tmp/sm120_qpkv6_b2_qregs_default_auto_off_20260529`: S4096 noncausal auto/off +3.97%, 34/61 paired wins, FA4/FA2 1.078x; S8192 causal auto/off +4.04%, 41/61 paired wins, FA4/FA2 1.039x. New S8192 noncausal repeat `/tmp/sm120_qpkv6_b2_s8192_nc_default_qregs_validate_20260529`: default/off +4.43% median, 50/63 paired wins, output diff 0; public subprocess repeat `/tmp/sm120_qpkv6_b2_public_after_s8192_nc_qregs_20260529`: FA4/FA2 geomean 1.026004, 5/5 wins. Correctness: `test_sm120_qpkv6_d256_b2_qregs_hook_matches_sdpa` and `test_sm120_qpkv6_d256_b2_s8192_noncausal_default_qregs` pass. | keeper, exact B=2 rows only; S1024/S4096 causal remain tiny/noisy and are not defaulted |
| Qwen D256 qpkv8/qpkv16 noncausal long | current patch: exact SM120 bf16 B=1, Hkv=2, qpkv8/qpkv16, S=16384/32768/65536/131072, `128x64`, 256 threads, `Q_in_regs=True`; `FLASH_ATTENTION_SM120_D256_QREGS128=off` forces old path | Env A/B `/tmp/sm120_d256_qregs128_ab_20260529`: qpkv8 S16384 +5.7% over old path and qpkv8 S131072 +13.3%; qpkv16 S16384 +4.7% and S131072 +19.5%; all output diffs 0. Mid run `/tmp/sm120_d256_qregs128_mid_ab_20260529`: qpkv8/qpkv16 S32768/S65536 +5.6% to +8.8% over old path. Default validation `/tmp/sm120_d256_qregs128_default_validate_20260529`: default/off wins 7/8 rows, with qpkv8 S16384 resolved by strict repeat `/tmp/sm120_d256_qregs128_qpkv8_s16384_repeat_20260529` at +2.6% over off. Long sweep `/tmp/sm120_longseq_qwen_gemma_after_d256_qregs_20260529`: qpkv8/qpkv16 noncausal long rows all beat FA2, up to 1.168x. | keeper, exact Qwen-style noncausal rows only; 128-thread Q-in-regs was rejected as much slower |
| qwen3.5-122B-style D256 qpkv16 causal long Q-regs | current patch: exact SM120 bf16 B=1 Hq32/Hkv2 qpkv16 D256 causal S16384/S32768/S65536/S131072 uses `128x64`, 256 threads, `Q_in_regs=True`; `FLASH_ATTENTION_SM120_D256_QPKV16_CAUSAL_QREGS=off` forces the old `64x64` path | Focused A/B: `/tmp/sm120_d256_qpkv16_causal_qregs_s16_s32_ab_20260529` showed S16384 +3.94% vs off and 1.062x vs FA2, S32768 +3.16% vs off and 1.060x vs FA2. Long A/B `/tmp/sm120_d256_qpkv16_causal_qregs_s64_s128_ab_20260529` showed S65536 +5.73% vs off and 1.072x vs FA2, S131072 +8.76% vs off and 1.107x vs FA2. Shipped default validation `/tmp/sm120_d256_qpkv16_causal_qregs_default_auto_off_20260529`: auto beat forced off by +3.92%/+5.33%/+5.36%/+9.08% for S16384/S32768/S65536/S131072 and beat FA2 by 1.046x/1.056x/1.068x/1.115x. All output diffs vs off were 0. Correctness: `test_sm120_qpkv16_d256_causal_qregs_matches_sdpa` passes with the forced mode. | keeper, exact qwen3.5-122B-style causal long rows only; S8192 was previously rejected as tiny/noisy |
| qwen3.6-35B-style D256 qpkv8 causal Q-regs | current patch: exact SM120 bf16 B=1 Hq16/Hkv2 D256 causal S16384/S32768/S65536/S131072 uses `128x64`, 256 threads, `Q_in_regs=True`; `FLASH_ATTENTION_SM120_D256_QPKV8_CAUSAL_QREGS=off` forces the old `64x64` no-Q-regs path | S16384 evidence: tile-only repeat `/tmp/sm120_d256_qpkv8_causal_s16384_tile_31x_20260529` rejected narrow-N tiles; Q-in-regs A/B `/tmp/sm120_d256_qpkv8_s16384_causal_qregs_ab_r2_20260529`: `128x64_t256` vs off +4.45% median, +3.61% mean, 45/63 paired wins, output diff 0; default two-label validation `/tmp/sm120_d256_qpkv8_s16384_causal_qregs_auto_off_95x_20260529`: auto/off +1.51% median, 56/95 paired wins, FA4/FA2 1.038x. Extension A/B `/tmp/sm120_d256_qpkv8_causal_qregs_extend_ab_20260529`: S32768 +3.52% vs off and +3.75% vs FA2, S65536 +3.87% vs off and +4.24% vs FA2, S131072 +7.41% vs off and +9.24% vs FA2; all output diffs 0. Shipped default validation `/tmp/sm120_d256_qpkv8_causal_qregs_default_auto_off_20260529`: S32768/S65536/S131072 auto beat off by +9.36%/+5.96%/+6.72% and beat FA2 by 1.079x/1.050x/1.099x. Correctness: `test_sm120_qpkv8_d256_causal_qregs_matches_sdpa` passes with the forced mode. | keeper, exact qwen3.6-style causal rows only |
| Broad Qwen/Gemma short-mid forward | `d6bdf7e`, `/tmp/sm120_model_variants_qpkv4_10x_d6bdf7e_20260528` | 78 cells, 10 repeats, geomean 1.061977, median 1.027039, wins 67/78. D128 1.050017, D256 1.067336, qwen 1.047141, gemma 1.096131. Versus prior 5-repeat rerun: ratio geomean 0.991577 with the same 67/78 win count. | current broad keeper |
| Gemma D256 local qpkv4/qpkv8 | `3dc3f22`, local `64x16` narrow-N path | Local-only Gemma sweep geomean 1.0654, median 1.0793, wins 5/6. e4b S4096 1.108, S8192 1.147; e2b S4096 1.083, S8192 1.076. | keeper for local rows; broad aggregate noisy |
| qwen3-30B D128 qpkv8 short/mid | S4096 noncausal `64x64`; S8192 noncausal `128x32`; S8192 causal `128x64`; `4c0b291`/`6b77ace` remain historical comparison points | S4096 noncausal patch: `/tmp/sm120_qpkv8_s4096_nc_patch_validate_20260529` auto 1.1199x vs FA2 after dispatch, and `/tmp/sm120_qpkv8_short_patch_validate_20260529` auto 1.1429x on the changed row. Earlier S8192 validation: `/tmp/sm120_qwen_qpkv8_patch_128x64c_dispatch_20260528b` geomean 1.025994, wins 5/6; repeat `/tmp/sm120_qwen_qpkv8_patch_128x64c_dispatch_r2_20260528b` geomean 1.038794, wins 5/6. | keeper for S4096 noncausal and S8192 qpkv8; S1024 noncausal remains tiny/noisy |
| qwen3-30B D128 qpkv8 long noncausal | current patch: exact SM120 bf16 B=1 Hq32/Hkv4 qpkv8 noncausal S=16384/32768/65536/131072 uses `128x32` | Candidate probe `/tmp/sm120_d128_qpkv8_long_noncausal_probe_20260529`: S16384 auto 0.9665x vs FA2, `128x32` 1.0015x; S65536 auto 0.9986x, `128x32` 1.0055x; S131072 auto 0.9913x, `128x32` 1.0082x. S32768 check `/tmp/sm120_d128_qpkv8_s32768_noncausal_128x32_20260529`: `128x32` 1.0126x vs FA2. Default validation `/tmp/sm120_d128_qpkv8_long_nc_default_validate_20260529`: default vs old `128x64` +6.8%/+0.7%/+0.9%/+1.0% for S16384/S32768/S65536/S131072. Long sweep confirmation `/tmp/sm120_longseq_qwen_gemma_after_d128_qpkv8_nc_20260529`: qpkv8 noncausal long ratios S16384/S32768/S65536/S131072 = 1.062473/1.019958/1.011293/1.009368; versus the D256-qregs sweep this group moved 0.988481 -> 1.025550, with FA4 medians 1.028162x faster. | keeper, exact qwen3-30B-style noncausal rows only; S32768 remains near-parity/noisy but old/new positive |
| qwen3-30B D128 qpkv8 long causal | current patch: exact dense bf16 B=1; S32768 `128x64` with 256 threads, S65536 `128x32` with default 128 threads | S32768 pre-patch repeat `/tmp/sm120_qpkv8_s32768_repeat2_20260529`: auto 1.0083x vs FA2, explicit `128x64_t256` 1.0261x. S32768 patched validation `/tmp/sm120_qpkv8_s32768_patch_validate_20260529`: auto 1.0193x and best in-run. Post-qregs repeat `/tmp/sm120_qpkv8_s32768_after_s65536_patch_20260529`: auto 1.0486x vs FA2. S65536 prior validation `/tmp/sm120_qpkv8_s65536_patch_validate_20260529`: auto 1.0176x; focused repeats after qregs showed current `128x64` falling to 0.986-0.998x while `128x32` held 1.020-1.028x. Patched default `/tmp/sm120_qpkv8_s65536_default_after_128x32_patch_20260529`: auto 1.0281x and best in-run. | keeper, exact long-causal qpkv8 rows only |
| Qwen D128 qpkv4 short/mid | `ced523f`: S1024 noncausal/causal `64x64`; S8192 noncausal `128x32`; S8192 causal `128x64` | Subprocess public-API validation `/tmp/sm120_qpkv4_lookup_patch_narrow_validation_20260528`: 12 affected qwen3-embedding/qwen3-vl rows, geomean 1.081840 vs FA2, wins 12/12. FA4 time geomean vs saved post-qpkv artifact: 1.057589; correctness max_abs <= 0.00390625 on changed S1024/S8192 paths. Broad 78-cell 5-repeat rerun `/tmp/sm120_model_variants_qpkv4_5x_rerun_b9261e6_20260528`: qpkv4 D128 S1024/S8192 rows all mean wins; qwen3-vl S8192 causal 1.039250 and 4/5 wins. | keeper; S4096 was explicitly reverted to prior lookup |
| SM120 D128 packed-GQA noncausal full-valid-row fast path | current patch: exact nonlocal bf16 D128 qpkv4/qpkv8 noncausal rows with fully valid packed M tiles skip redundant Q/O/LSE row-bound checks; `FLASH_ATTENTION_SM120_PACK_GQA_VALID_ROWS_FAST=off` forces old path | Stricter 95-round A/B `/tmp/sm120_packgqa_valid_rows_nonc_95x_20260529`: qpkv4 S4096 fast/on vs off +0.67%, qpkv8 S8192 +0.57%, output max diff 0. Earlier 31-round probe showed the same source path positive on average but causal qpkv4 S8192 regressed, so causal/local are excluded. Correctness: new SDPA/off-path test plus full `tests/cute/test_flash_attn_sm120_local.py` 40/40 pass. | keeper, narrow source-level speedup; do not broaden to causal without a fresh artifact |
| D64 MHA old 60-cell misses | `e93f8ef`: `(64,1,S,causal)` lookup now S1024 nc `64x64`, S2048 nc `128x32`, S4096 nc `64x64`, S8192/S16384 nc `128x48`, S16384 c `128x48` | Fresh public 60-cell run `/tmp/sm120_sdpa_fa2_fa4_forward_d64_patch_20260528`: hd64 MHA old/new FA4/FA2 ratios S1024 nc 0.849 -> 0.998, S2048 nc 0.977 -> 1.020, S4096 nc 1.017 -> 1.047, S8192 nc 0.946 -> 0.973, S16384 nc 0.967 -> 0.991, S16384 c 0.937 -> 1.000. | keeper for old 60-cell matrix; D64 remains noisy in full public sweeps |
| D128 qpkv4 old 60-cell long rows | this commit: S8192 causal B=1 `64x64`; S16384 noncausal `128x32`; S16384 causal B=1 `128x64`, B>1 `128x48` | Public 60-cell run `/tmp/sm120_sdpa_fa2_fa4_forward_qpkv4_patch_20260528`: FA4/FA2 geomean 1.092207, wins 50/60. Touched qpkv4 ratios from prior fresh D64 run: llama S8192 c 0.954 -> 1.021, llama S16384 c 0.996 -> 1.037, mistral S16384 nc 0.991 -> 1.006, mistral S16384 c 0.982 -> 1.016. B=2 qwen3-vl S16384 direct public checks stayed above FA2 for causal and noncausal. | keeper; qpkv7 rows were probed and left unchanged |
| SM120 paged-KV head_dim > 128 | this commit: PagedKVManager uses `ceil(tile_n / num_threads)` page-table entries; SM120 D192/D256 use non-TMA `64x64` | Correctness: `tests/cute/test_paged_kv_sm120.py` 51/51 pass, including D192/D256 page sizes, causal, GQA, MQA. Bench: `/tmp/sm120_paged_kv_hdgt128_bench_packoff_20260528`, D192/D256 exact match vs unpacked nonpacked FA4 baseline; D256 S1024 paged slightly faster, S4096 within 2-3% of contiguous. | keeper |
| SM120 D256 backward support | current D256 alias path: Q/dO and K/V shared-memory reuse with Q/K reload | Correctness: `tests/cute/test_flash_attn_sm120_local.py` validates D256 qpkv2/qpkv4/qpkv8 causal and noncausal against SDPA. Perf is below FA2 in the current S1024 smoke, so this is a coverage baseline rather than a winner. | feature baseline |
| Pre-expansion D256 backward smoke after pack keepers | `74c0b4d`, qpkv2/qpkv4/qpkv8 D256 S1024/S4096/S8192 smoke | `/tmp/sm120_bwd_d256_current_smoke_20260529`: 20 cells, FA4/FA2 median-ratio geomean 1.000220, wins 10/20. qpkv2 rows are mostly wins, qpkv4 S8192 benefits from the qpkv4 auto-pack keeper, and remaining qpkv4/qpkv8 S4096 misses are not stable enough for a dispatch-only patch. Superseded for qpkv2 S8192 by the qpkv2/qpkv6/qpkv16 exact-row keeper below. | historical status, not a winner |
| SM120 D256 qpkv8 noncausal backward auto-pack | auto-enable packed backward only for fixed dense bf16 D256 qpkv8 noncausal | Targeted A/B `/tmp/sm120_bwd_d256_qpkv8_autopack_ab_20260528`: Gemma-e2b qpkv8 S1024 noncausal auto vs forced old pack-off +3.6% (FA4/FA2 1.0439 vs 1.0076); Qwen35-style qpkv8 S1024 noncausal auto vs forced old pack-off +1.0% (0.9922 vs 0.9819). Correctness: qpkv8 `pack_gqa=None` causal/noncausal vs SDPA; local SM120 pytest 11/11. | keeper, narrow |
| SM120 D256 qpkv8 S1024 causal backward fused-dKV | auto-enable fused dK+dV postprocess only for fixed dense bf16 D256 qpkv8 S1024 causal | Forced on/off A/B `/tmp/sm120_bwd_fused_dkv_qpkv8_causal_40x_20260529`: Hq8/Hkv1 median +1.8%, 31/40 wins, clipped mean +2.6%; Hq16/Hkv2 median +1.6%, 31/40 wins, clipped mean +1.7%. Patched default/off validation `/tmp/sm120_bwd_fused_dkv_qpkv8_causal_default_validate_20260529`: Hq8 median +1.3%, 24/40 wins, clipped mean +3.4%; Hq16 median +1.2%, 26/40 wins, clipped mean +0.2%. Correctness: D256 fused-dKV backward checked against SDPA; `tests/cute/test_flash_attn_sm120_local.py` 28/28 pass on RTX 5090. | keeper, exact S1024 causal qpkv8 only |
| SM120 fused dK+dV postprocess compile guard | current patch removes the redundant dynamic mdK/mdV head-dim `const_expr` check; the Python wrapper keeps the static same-D validation | Previous pause smoke had 15 FA4 errors from the dynamic `const_expr` guard. After patch `/tmp/sm120_bwd_d256_s1024_causal_after_dkv_guard_20260529`: 24/24 FA2/FA4 pairs succeeded, geomean 1.046385, median 1.022133, wins 17/24; qwen geomean 1.101264, gemma 0.960929. Correctness: full `tests/cute/test_flash_attn_sm120_local.py` 40/40 pass. | correctness/coverage keeper; perf smoke is noisy, but rows are measurable again |
| SM120 D256 qpkv6/qpkv8 S1024 causal nonpacked M-split | auto-enable 2 M-splits only for fixed dense bf16 D256 qpkv6/qpkv8 S1024 causal when `pack_gqa=False` | Env-forced A/B `/tmp/sm120_bwd_nonpack_split_32x_20260529`: split2 vs default/off median qpkv6 +4.8% with 29/32 paired wins, qpkv8 H8 +9.1% with 32/32 wins, qpkv8 H16 +6.7% with 28/32 wins. Final default-vs-split1 validation `/tmp/sm120_bwd_nonpack_split_default_validate_20260529`: default split2 median faster on all three rows, while split1 won only 7/32, 4/32, and 5/32. Correctness: forced split2 passed `tests/cute/test_flash_attn_sm120_local.py::test_sm120_hd256_backward_matches_sdpa` 18/18; default split2 passed the full SM120 local suite 28/28. | keeper, exact qpkv6/qpkv8 S1024 causal only |
| SM120 D256 qpkv8 Hq8/Hkv1 S1024 causal nonpacked split3 | current patch: use 3 nonpacked M-splits only for fixed dense bf16 D256 qpkv8 Hq8/Hkv1 S1024 causal | Split2/split3/split4 probe `/tmp/sm120_bwd_qpkv8_h8_split3_probe_20260529`: split3 vs split2 median speedup 1.015x, geomean 1.022x, 42/64 paired wins, clipped mean 0.4342 ms vs 0.4435 ms. B=1 probe `/tmp/sm120_bwd_qpkv8_h8_split3_b1_probe_20260529`: median 1.035x, 51/64 wins. Patched default-vs-forced-split2: B=2 `/tmp/sm120_bwd_qpkv8_h8_split3_default_b2_20260529` median 1.033x, 48/64 wins, clipped mean +1.4%; B=1 `/tmp/sm120_bwd_qpkv8_h8_split3_default_b1_20260529` median 1.025x, 45/64 wins. FA2 check `/tmp/sm120_bwd_gemma_e2b_split3_default_vs_fa2_10x_20260529`: 1.066x geomean, 1.042x median, 9/10 wins. Correctness: S1024 SDPA max errors dq 0.0119, dk 0.0191, dv 0.0378. | keeper, exact small-H Gemma qpkv8 row |
| SM120 D256 qpkv4 S8192 noncausal backward auto-pack | auto-enable packed backward only for fixed dense bf16 D256 qpkv4 S8192 noncausal | Targeted A/B `/tmp/sm120_bwd_d256_qpkv4_pack_condition_clean_20260529`: Hq8/Hkv2 S8192 forced pack-on vs off +5.9% median / +2.3% mean, Hq16/Hkv4 S8192 auto vs off +1.9% median / +2.0% mean. Focused repeat `/tmp/sm120_bwd_d256_qpkv4_s8192_pack_repeat2_20260529`: Hq8/Hkv2 S8192 forced pack-on vs off +2.4% median / +2.8% mean, Hq16/Hkv4 S8192 auto vs off +1.3% median / +1.8% mean. Final dispatch check `/tmp/sm120_bwd_d256_qpkv4_s8192_final_dispatch_ab_20260529`: S8192 auto/off +1.7% Hq8 and +3.5% Hq16 median; S4096 stays effectively pack-off/parity. Correctness: explicit qpkv4 packed D256 backward added to the SM120 local SDPA test. | keeper, S8192-only |
| SM120 D256 qpkv4 S1024 causal backward auto-pack | auto-enable packed backward only for fixed dense bf16 D256 qpkv4 S1024 causal; Hq8/Hkv2 now uses 16 packed-M splits, wider qpkv4 stays on split8 | Original pack8 keeper: `/tmp/sm120_bwd_packgqa_causal_oversplit_qpkv4_20260529` and `/tmp/sm120_bwd_qpkv4_s1024_causal_auto_pack8_validate_20260529` showed auto-pack beating forced pack-off. Hq8/Hkv2 split16 refinement: `/tmp/sm120_bwd_qpkv4_s1024_split16_h8only_auto_split8_128x_20260529` auto split16 vs forced split8 +2.9% median, +6-8% clipped/mean, 86/128 paired wins; B=1 `/tmp/sm120_bwd_qpkv4_s1024_split16_h8only_b1_96x_20260529` +3.7% median, 70/96 paired wins. D256 SDPA subset passed 18/18. Public 10x smoke `/tmp/sm120_bwd_d256_split16_h8only_10x_20260529` improved qwen3.5-0.8B qpkv4 from the `cf39944` reference by mean/median, while same-shape Gemma remained noisy vs FA2. | keeper, exact S1024 causal qpkv4; split16 only for Hq8/Hkv2 |
| SM120 D256 qpkv4/qpkv6 S1024 causal full-valid mask skip | current patch: skip causal mask work only on full-valid m-blocks for fixed dense bf16 B=2 S1024 qpkv4 Hq8/Hkv2 and qpkv6 Hq24/Hkv4; `FLASH_ATTENTION_SM120_BWD_SKIP_FULL_CAUSAL_MASK=0` forces old path, `=1` forces broader profiling eligibility | Default/off 10x `/tmp/sm120_bwd_skip_full_causal_mask_default_validate_10x_20260529`: qpkv4 Hq8 +9.1% geomean, median ratio +7.1%, 7/10 wins; qpkv6 Hq24 +3.3% geomean, median ratio +1.9%, 7/10 wins. Exact S1024 correctness vs SDPA: qpkv4 max dq/dk/dv 0.0118/0.0168/0.0214; qpkv6 0.0141/0.0244/0.0347. Full SM120 local suite: 28/28 pass. | keeper, exact rows only; qpkv8 broadening rejected |
| SM120 D256 qpkv4 S1024 causal mask-skip expansion | current patch: additionally default full-valid mask skip for exact B=1 Hq8/Hkv2 and B=2 Hq16/Hkv4 qpkv4 S1024 causal rows | Default/off validation: `/tmp/sm120_bwd_qpkv4_h8_b1_maskskip_default_validate_20260529` B=1 Hq8/Hkv2 +5.67% median / +7.62% trimmed mean, 44/64 wins; `/tmp/sm120_bwd_qpkv4_h16_b2_maskskip_default_validate_20260529` B=2 Hq16/Hkv4 +1.17% median / +2.19% trimmed mean, 44/64 wins. Broad smoke `/tmp/sm120_bwd_d256_s1024_causal_after_qpkv4_maskskip_20260529`: 40 pairs, geomean 1.043000, median 1.039382, wins 30/40. Correctness: full `tests/cute/test_flash_attn_sm120_local.py` 43/43 pass. | keeper, exact qpkv4 rows only; B=1 qpkv2 and B=1 Hq16 were rejected by trimmed-mean instability |
| SM120 D256 qpkv2 S1024 causal full-valid mask skip | current patch: add Gemma31-style Hq32/Hkv16 qpkv2 B=2 S1024 causal to the mask-skip default gate | Default/off 10x `/tmp/sm120_bwd_qpkv2_s1024_maskskip_ab_20260529`: +8.2% geomean, +5.3% median ratio, 7/10 wins. Exact FA2/FA4 timing remained noisy (`/tmp/sm120_bwd_gemma31_qpkv2_s1024_fa2fa4_maskskip_10x_20260529b`: 1.0095 geomean, 0.9739 median), so this is an FA4 old/new keeper rather than a clean FA2 claim. Correctness: exact B=2 S1024 SDPA max dq/dk/dv 0.0117/0.0169/0.0271; D256 SDPA subset 18/18 pass. | keeper, exact qpkv2 row only |
| SM120 D256 qpkv2/qpkv6/qpkv16 noncausal backward auto-pack | auto-enable packed backward only for fixed dense bf16 D256 qpkv2 Gemma31 S4096/S8192/S16384, qpkv6 Qwen27 S4096, and qpkv16 Qwen122 S4096/S8192 | Interleaved explicit-pack A/B `/tmp/sm120_bwd_packgqa_condition_ab_20260529`: qpkv16 S4096 +2.9% median, qpkv16 S8192 +1.8%, qpkv2 S8192 +1.4%, qpkv6 S4096 +15.8% median but noisy. Final patched auto vs forced pack-off `/tmp/sm120_bwd_packgqa_patch_false_auto_20260529`: qpkv16 S4096 +2.0% median, qpkv16 S8192 +1.6%, qpkv2 S8192 +3.0%, qpkv6 S4096 +3.3%; all four improved by mean. qpkv2 extended final dispatch `/tmp/sm120_bwd_d256_qpkv2_final_dispatch2_20260529`: auto/off +3.2% S4096, +1.5% S8192, +1.2% S16384 by median. Correctness: D256 packed qpkv2/qpkv6/qpkv16 added to the SM120 local SDPA test; `tests/cute/test_flash_attn_sm120_local.py` 21/21 pass on RTX 5090. Broad affected-seqlen checks: S4096 noncausal `/tmp/sm120_bwd_d256_s4096_noncausal_pack_expand_20260529` geomean 1.041x, 13/16 wins; S8192 noncausal `/tmp/sm120_bwd_d256_s8192_noncausal_pack_expand_20260529` geomean 1.111x, 13/16 wins. | keeper, exact rows only |
| SM120 D256 qpkv4 S2048 causal packed split16 | current patch: B=2 fixed dense bf16 D256 qpkv4 Hq8/Hkv2 and Hq16/Hkv4 S2048 causal auto-pack with split16 | Default/off validation `/tmp/sm120_bwd_qpkv4_s2048_auto_split16_validate_20260529`: Hq8 +11.1% median / +10.1% mean; Hq16 +10.2% median / +6.2% mean. FA2/FA4 5-repeat S2048 causal `/tmp/sm120_bwd_d256_s2048_causal_after_qpkv4_split16_5x_20260529`: geomean 1.034150, median 1.018967, wins 26/40; qpkv4 Hq8 1.068165 geomean, Hq16 1.009983. Correctness against forced pack-off on exact B=2 S2048 rows: output max diff 0, gradient max diffs within bf16 tolerance. | keeper, exact B=2 S2048 qpkv4 rows |
| qwen2.5 D128 dense noncausal TMA mask skip | `0af9a4c`, static noncausal TMA seqlen-mask skip | NCU Qwen2.5 S8192 noncausal: 10.56 ms -> 10.18 ms, instructions 2.064B -> 1.866B. Repeats median 0.975, mean 0.987, range 0.971-1.031. | historical, partly superseded |
| backward d<=64 | `4d59090`, SM120 backward default `num_stages` 2 -> 1 | Phase 17C reported +5.6% on d<=64 cells, arch-gated. | keeper |
| backward broad Phase 17 | `362a65a` + `55ab672`, 8 warps/block and v4 atomic dQ/dK/dV | Phase 17 backward 40-cell FA4/FA2 geomean 1.017x, 29/40 wins, peak 180.8 TFLOPS; was 0.93x and 10/40 wins before. | keeper |

## Rejected Or Noisy Paths

| Path | Outcome |
|---|---|
| qpkv6 D256 causal `64x48` and Q-in-reg variants | One run looked good, repeats rejected. Prior experiment had qpkv6 S4096 causal 0.929 despite a noisy 78-cell aggregate win. Do not restore without fresh paired evidence. |
| qpkv6 D256 full-head-dim predicate elision | Rejected/no default after source probe. The first 4-row A/B `/tmp/sm120_qpkv6_d256_source_ab_20260529` had small hints, but the two-label default/off run `/tmp/sm120_qpkv6_d256_source_default_off_2label_20260529` was flat overall and regressed S16384/S32768 noncausal by median. Keep the predicate setup path unchanged; only the static causal block-bound specialization survived. |
| qpkv6 D256 static causal bounds at S16384 | Rejected for default. Static-only A/B initially looked positive, but final auto/off validation `/tmp/sm120_qpkv6_d256_static_causal_default_validate_20260529` flipped S16384 causal to 0.988x by median. Keep S16384 causal on generic `BlockInfo` unless a stricter repeat reverses this. |
| qpkv6 D256 Q-regs plus static causal bounds | Rejected/correctness failure. Extending `FLASH_ATTENTION_SM120_QPKV6_D256_STATIC_CAUSAL_BLOCKS=on` to the Q-regs path cut runtime roughly in half in `/tmp/sm120_qpkv6_b2_qregs_source_ab_20260529`, but output was wrong (`max_abs_vs_base` about 1.1-1.3). Keep static causal bounds limited to the existing non-Q-regs `64x64` path. |
| qpkv8 D128 pack-aware static causal bounds | Rejected/no product patch. The quick two-shape run `/tmp/sm120_d128_qpkv8_static_causal_ab_quick_20260529` looked positive, but the full long run `/tmp/sm120_d128_qpkv8_static_causal_ab_full_20260529` regressed S32768/S65536/S131072 by median (0.979x/0.981x/0.973x on/off). A stricter S16384-only 31x repeat `/tmp/sm120_d128_qpkv8_static_causal_s16384_31x_20260529` was only +0.5% median. Keep the existing qpkv8 long-causal tile gates instead. |
| qpkv8 D128 causal seqlen-mask elision | Rejected/no product patch. The env-gated source probe preserved outputs and initially helped S16384/S32768 in `/tmp/sm120_d128_qpkv8_skip_causal_seqlen_ab_full_20260529`, but the stricter positive-row repeat `/tmp/sm120_d128_qpkv8_skip_causal_seqlen_s16_s32_31x_20260529` flipped to flat/slower: S16384 0.9997x, S32768 0.9912x on/off. S65536/S131072 were already negative in the full run. Keep causal `mask_seqlen=True`. |
| qpkv8 D128 S32768 causal Q-in-regs | Rejected/no product patch. Env-gated `FLASH_ATTENTION_SM120_D128_QPKV8_QREGS=on` was exact, but `/tmp/sm120_d128_qpkv8_s32768_causal_qregs_ab_20260529` only gave +0.99% median with 33/63 paired wins. That is below the bar for a known-variant row. Keep the existing `128x64`, 256-thread tile gate. |
| qpkv8 D128 S65536 causal 256-thread recheck | Rejected/no product patch. `/tmp/sm120_current_long_miss_repeat_after_qpkv16_20260529` put current auto at parity/noise (0.9989x geomean vs FA2, 2/5 wins). Tile recheck `/tmp/sm120_qpkv8_s65536_causal_tile_recheck_after_qpkv16_20260529` found `128x32_t256` only +0.26% vs auto median in one 9-round probe. Keep the existing auto path. |
| qwen3.6-35B-style D256 qpkv8 causal S=16384 narrow-N tiles | Rejected. `/tmp/sm120_d256_qpkv8_causal_tile_ab_s16_s32_20260529` had a first-run S16384 `64x16` hint, but the stricter repeat `/tmp/sm120_d256_qpkv8_causal_s16384_tile_31x_20260529` rejected it: `64x16` was 0.93x vs auto, and `64x48`/`64x64` were effectively tied. The accepted fix is the exact Q-in-regs row above, not a tile-only change. |
| qwen3.6-35B-style D256 qpkv8 causal S=16384 K/V hooks | Rejected/no product patch. Hook modes on top of the accepted Q-in-regs row had only a small first-run hint in `/tmp/sm120_qpkv8_d256_s16384_hook_ab_20260529`; the clean V-only repeat `/tmp/sm120_qpkv8_d256_s16384_hook_v_off_95x_20260529` flipped negative at 0.996x median vs off. Keep hook scheduling limited to the existing qpkv5/qpkv6 gates. |
| qwen3-14B qpkv5 S8192 noncausal `64x128` / `64x112` candidates | Interleaved A/B rejected; current S8192 causal keeper is `6b77ace`, noncausal remains sensitive/noisy. |
| qwen3-14B qpkv5 S8192 noncausal non-TMA extension | Rejected/no product patch. Dirty generalized S4096/S8192 non-TMA gate had a weak first S8192 hint in `/tmp/sm120_qpkv5_s8192_nc_tma_ab_20260529` (+0.58% median vs TMA but slightly worse by mean). Stricter `/tmp/sm120_qpkv5_s8192_nc_tma_ab_r2_20260529` favored non-TMA by about +1.3% vs TMA, but FA4 still did not beat FA2 and the generalized S4096 recheck `/tmp/sm120_qpkv5_s4096_nc_tma_recheck_after_general_20260529` favored the old TMA label in that run. Keep the committed exact S4096 policy and do not extend to S8192 without a cleaner artifact. |
| qwen3-14B qpkv5 S8192 noncausal latest recheck/profile | Rejected/no product patch. Current exact tile recheck `/tmp/sm120_qpkv5_s8192_nc_tile_recheck_20260529` had auto best by mean/median geomean; `128x32`, `64x96`, `64x128`, `64x112`, `64x80`, and `64x64` all regressed. NCU `/tmp/sm120_ncu_qpkv5_s8192_nc_current` showed FA4 TMA and FA2 main kernels effectively tied under profile (14.32 ms vs 14.23 ms), same 5120-CTA grid, no spills; the public FA2 gap is run-order sensitive, not a simple dispatch miss. |
| qwen3-14B qpkv5 S16384 noncausal `128x32` lookup | Rejected. `/tmp/sm120_qpkv5_s16384_nc_tile_exact_20260529` showed `128x32` slower than current auto for both B=1 and B=2: B=1 auto 30.39 ms vs `128x32` 31.88 ms; B=2 auto 60.98 ms vs `128x32` 63.62 ms. Current auto/`128x64` is already the safe tile family for this row. |
| qwen3-14B qpkv5 S16384 noncausal non-TMA fallback | Rejected. A dirty exact B=1 gate had only a small/non-repeatable edge over TMA: `/tmp/sm120_qpkv5_s16384_nc_notma_paired_20260529` non-TMA +0.69% vs forced TMA, repeat `/tmp/sm120_qpkv5_s16384_nc_notma_paired_r2_20260529` +0.20%. That is below the bar for switching kernel architecture on this row. |
| qwen3-14B qpkv5 S32768 noncausal tile alternatives | Rejected/no patch. `/tmp/sm120_qpkv5_s32768_nc_tile_probe_20260529` did not reproduce the broad-sweep miss: current auto was 1.0045x vs FA2, `128x32` regressed badly, `64x128` regressed, and explicit `128x64` was only a tiny same-family/noise improvement over auto. |
| qwen3-14B qpkv5 S4096 noncausal tile alternatives | Rejected/no patch. `/tmp/sm120_qpkv5_s4096_nc_tile_ab_20260529` did not reproduce the post-qpkv8 broad miss: current auto was already 1.007x by mean-geomean and 1.016x by median-geomean vs FA2 across three rounds. `64x96` was the closest alternate but still slower than auto by mean; `64x128`, `128x32`, `128x48`, and `64x64` regressed. |
| qwen3-14B D128 qpkv5 S4096 causal tile/profile follow-up | Rejected/no product patch. The pause sweep's 0.919x mean row was outlier-driven; focused `/tmp/sm120_qpkv5_s4096_causal_tile_recheck_20260529` had current auto near FA2 parity by median and no stable explicit-tile win. Corrected 128-thread repeat `/tmp/sm120_qpkv5_s4096_causal_tile_t128_recheck_20260529` rejected `64x32`, `64x96`, `64x112`, `64x128`, and `128x64`: auto was best by mean and 0.996x by median-geomean vs FA2. NCU `/tmp/sm120_ncu_qpkv5_s4096_causal_20260529` showed FA4 and FA2 both at 1.99 ms under profile; FA4 has 2x CTAs and more instructions (379M vs 328M) but no spills, while FA2 has 6.1M spill instructions. No simple dispatch/source knob is indicated. |
| qwen3-14B D128 qpkv5 S65536 noncausal tile alternatives | Rejected/no patch. `/tmp/sm120_qpkv5_s65536_nc_tile_ab_after_246a798_20260529` reproduced only a near-parity miss: auto median-ratio geomean 0.989743 vs FA2. Explicit `128x64` was same-family/noise at 0.990544 and only 1/3 median wins vs auto; `64x128`, `64x96`, and `128x32` regressed to 0.949530/0.944481/0.937210. Keep current auto. |
| qwen3-14B D128 qpkv5 long noncausal TMA/non-TMA microprobes | Rejected/no patch. `/tmp/sm120_qpkv5_long_nc_tma_microprobe_20260529` rejected fixed-tile TMA `kv_stages=1` and 2-MMA-warp variants: stage1 was slower at S32768/S65536, and 2-warps collapsed throughput. Non-TMA had a tempting S32768 hint in `/tmp/sm120_qpkv5_s32768_notma_ab_20260529`, but stricter default-vs-forced-TMA validation `/tmp/sm120_qpkv5_s32768_nc_default_notma_validate_20260529` flipped to forced TMA faster. S65536 rejected non-TMA in `/tmp/sm120_qpkv5_s65536_notma_ab_20260529`; S131072 was only tied in `/tmp/sm120_qpkv5_s131072_notma_ab_20260529`. Non-TMA Q-in-regs also lost to plain non-TMA in `/tmp/sm120_qpkv5_s32768_nc_notma_qregs_ab_20260529`. |
| qwen3-30B qpkv8 S4096 noncausal `128x32` lookup | Rejected/no patch. A first exact probe `/tmp/sm120_qpkv8_s4096_nc_tile_ab_20260529` favored `128x32`, but stricter validation after a temporary lookup change rejected it: `/tmp/sm120_qpkv8_s4096_nc_patch_validate_r2_20260529` had current auto/`128x32` at 1.021x mean-geomean vs FA2, while old `64x64` was slightly better at 1.024x and won 5/8 mean rounds versus auto. Keep the existing `64x64` lookup. |
| qwen3-30B D128 qpkv8 S8192 noncausal pause-sweep miss | Rejected/no product patch. The pause sweep's 0.899x row did not reproduce. Focused tile probe `/tmp/sm120_qpkv8_s8192_nc_tile_recheck_20260529` put current auto at 1.075x mean-ratio geomean and 1.029x median-ratio geomean vs FA2, faster than explicit tile alternatives. Corrected 128-thread repeat `/tmp/sm120_qpkv8_s8192_nc_tile_t128_recheck_20260529` again had auto/`128x32` around parity-to-win; same-kernel explicit `128x32_t128` was only noise. Pack-off/TMA validation `/tmp/sm120_qpkv8_s8192_nc_packoff_probe_20260529` rejected pack-off (`128x64` pack-off 0.997x vs FA2; `128x32` pack-off 0.938x). Current packed `128x32` remains the right path. |
| qwen3-30B D128 qpkv8 S65536 causal latest tile recheck | Rejected/no patch. `/tmp/sm120_qpkv8_d128_s65536_causal_tile_recheck_20260529` put current auto above FA2 (1.044x mean-ratio geomean, 1.052x median-ratio geomean) and faster than `128x48`, `128x64`, `64x64`, `64x96`, and `64x128`. This reinforces the earlier NCU/repeat conclusion that the row is parity/noise, not a tile target. |
| D256 qpkv8 noncausal pack/tile gates | Rejected/no patch. Gemma e2b Hq8/Hkv1 S4096/S8192 focused repeat `/tmp/sm120_qpkv8_noncausal_pack_tile_validate_20260529` produced only same-kernel/noisy hints: S8192 best was the same logical packed `64x64` path as auto, and S4096 pack-off was only +0.5% in-process. Wider qwen3.6 Hq16/Hkv2 S4096 `/tmp/sm120_qwen36_qpkv8_s4096_nc_pack_ab_20260529` had current auto best and 1.048x vs FA2. |
| qwen3-30B qpkv8 S131072 causal long tile alternatives | Rejected/no patch. `/tmp/sm120_qpkv8_s131072_probe_20260529` showed auto already above FA2 at 1.0093x; best explicit `128x64_t256` was only 1.0132x vs FA2, about +0.4% over auto. Keep the current S131072 lookup until a larger repeatable gap appears. |
| qwen3.6 D256 qpkv8 long causal pack/tile alternatives | Rejected/no patch. `/tmp/sm120_d256_qpkv8_long_causal_pack_tile_probe_20260529` did not reproduce the long-sweep causal misses: focused auto beat FA2 at both S32768 and S65536. Pack-off, `64x48`, and `64x32` all regressed versus auto. |
| D256 qpkv4 Hq8/Hkv2 S4096 noncausal pack-off/TMA exact gate | Rejected/no patch. A first same-process probe `/tmp/sm120_qpkv4_s4096_nc_candidates_validate_1ca5ca7` favored explicit pack-off on the Hq8/Hkv2 row, but the patched default vs forced-old packed subprocess A/B rejected it: `/tmp/sm120_qpkv4_h8_s4096_nc_default_vs_old_pack_20260529` default pack-off/old packed geomean 0.994x with 5/10 paired wins. Keep the packed default. |
| D256 qpkv4 current forward gap probes | Rejected/no patch. `/tmp/sm120_qpkv4_d256_current_gap_condition_ab_20260529` did not reproduce qwen3.5-0.8B/2B S4096 causal as a miss: auto was best and 1.076x vs FA2. qwen3.5-9B S8192 noncausal stayed near parity at 0.988x, but auto was still the fastest label. A dirty Q-in-regs qpkv4 probe had a first-run qwen0.8 hint, but the stricter repeat `/tmp/sm120_d256_qpkv4_qregs_qwen08_repeat_20260529` rejected it: qregs64/off 0.977x by median. |
| SM120 D256 qpkv8 S1024 causal full-valid mask skip | Rejected for default dispatch. Env-forced broad timing `/tmp/sm120_bwd_skip_full_causal_mask_ab_20260529` kept qpkv4/qpkv6 positive but did not justify qpkv8: Hq8/Hkv1 regressed to 0.956x geomean with only 2/6 wins, and Hq16/Hkv2 was neutral/noisy at 1.001x with 3/6 wins. Keep the mask-skip gate limited to qpkv4 Hq8 and qpkv6 Hq24. |
| qwen3.5/qwen3.6 D256 qpkv6 S16384 noncausal tile alternatives | Rejected/no patch. `/tmp/sm120_qpkv6_s16384_nc_tile_probe_20260529` did not reproduce the long-sweep miss: focused auto was 1.042x vs FA2. `64x48`, `64x32`, `64x16`, `128x32`, and `128x16` all regressed versus auto/current `64x64`. |
| Gemma4 e4b D256 qpkv4 S131072 local tile/pack alternatives | Rejected/no patch. `/tmp/sm120_gemma_e4b_s131072_local_tile_probe_20260529` did not reproduce the broad long-sweep miss: focused auto was 1.066x vs FA2. Pack-off regressed, `64x64` regressed, and `64x32` was effectively tied with current auto/`64x16`. |
| D256 qpkv6 exact-shape TMA `kv_stages=1` | Ran around 341.1 ms on qwen3.5/qwen3.6 S65536 causal, slower than CpAsync and FA2. |
| D64 qpkv1 noncausal paired-KV TMA barrier | Rejected and removed. Env-gated prototype used one K+V TMA barrier for S8192/S16384 and matched output exactly, but timing regressed: `/tmp/sm120_d64_paired_kv_ab_20260529` paired-on/off was 0.9088x at S8192 and 0.9353x at S16384. Keep separate K and V TMA pipelines. |
| SM120 D256 qpkv8/qpkv16 Q-in-regs with 128 threads | Rejected. The same Q-in-regs `128x64` tile needs 256 threads; the 128-thread variant was correct but far slower in `/tmp/sm120_d256_qregs128_ab_20260529` (about 0.21-0.30x vs FA2 on the sampled rows). |
| qwen3.5/qwen3.6 qpkv6 S16384 noncausal tile/pack alternatives | No dispatch patch. The broad post-qpkv8-S65536 sweep showed a 0.973x miss, but focused repeat `/tmp/sm120_d256_s16384_noncausal_candidates_20260529` put auto/explicit same-kernel paths at parity-to-win, and the stripped same-kernel repeat `/tmp/sm120_qpkv6_s16384_same_kernel_repeat_20260529` had auto 1.009x vs FA2 with explicit `64x64` only a 0.6% same-output timing split. |
| Stream-PV prototype avoiding full bf16 `rP` | Compiled and ran, but 3 paired reps geomean 0.958x. Reverted. |
| TMA stage retune commits `4e895b5`, `ec6dd92`, `ca7f03e`, `08c184f` | Reverted by `6a8f9ab` after long-regression/noise. Do not restore wholesale. |
| Broad qpkv8 old-commit restore | Current strict qpkv8 causal remains positive, but noncausal/noisy rows do not justify a broad revert. Profile exact rows before changing dispatch. |
| Backward tile sweep `a0dd865`/`47aa883` | Negative by design: d=128 only viable tile was already default `(64,64,1)`. No `_SM120_BWD_TILE_LOOKUP` shipped. |
| Dense noncausal `check_inf` skip | Tried after `48c7d4d`; qpkv6 S131072 noncausal stayed around 0.966x, so patch was reverted. |
| SM80-base qpkv6 hook-path `utils.cvt_f16(acc_S, rP)` conversion | Tried after `60354fd` and reverted locally. Focused dirty run `/tmp/sm120_qwen_focused_qpkv6_cvt_hook_dirty_20260528b` dropped D256 qpkv6 causal geomean to 1.015849 vs prior current reference 1.025739, mainly hurting S16384 causal. |
| D256 qpkv6 broad hook replacement | Partially superseded by the exact long-row V-only hook keeper above. The old broad replacement remains rejected: profiling-only env selector at S4096 causal had V-only fastest in one profile, but broad focused timing rejected changing the default for all qpkv6 rows. Keep the new policy limited to exact B=1 long qpkv6 rows; do not restore a broad K/V/off hook rule. Old artifacts: `/tmp/sm120_qpkv6_hook_ab_ncu_20260528b`, `/tmp/sm120_qwen_focused_hook_{both,v,off,k}_dirty_20260528b`. |
| D256 qpkv6 first-load predicate / causal first-tile mask / SM120 softmax reductions | Rejected after dirty probes and reverted. Combined predicate+mask improved one S4096 run but regressed S8192/S16384; load-predicate-only had a tiny NCU duration win (2.328 ms -> 2.323 ms) but repeated timing regressed S4096; softmax-120 reduced instructions but added writeback and did not improve NCU duration. Artifacts: `/tmp/sm120_qwen_focused_predmask_dirty_20260528b`, `/tmp/sm120_qwen_focused_loadpred_dirty_20260528b`, `/tmp/sm120_qwen_focused_softmax120_dirty_20260528b`, `/tmp/sm120_qpkv6_s4096_loadpred_ncu_20260528b`. |
| Current 10-repeat forward tail rows: qwen122 qpkv16 S8192 causal, Gemma31 qpkv2 S4096 local, qwen27 qpkv6 S1024 causal | No dispatch patch. Targeted reruns showed these are not repeatable misses: qwen122 qpkv16 S8192 causal subprocess repeat is parity (mean 1.001314, median 1.001489, 5/10 wins; `/tmp/sm120_forward_miss_repeat_qwen122_current_20260528`) and interleaved tile A/B favored current auto over explicit alternates (auto 1.054302x; `/tmp/sm120_qwen122_qpkv16_s8192_causal_ab_20260528`). Gemma31 qpkv2 S4096 local auto rerun was already 1.087680x vs FA2, with best alternate only +0.05% over auto; S8192 local's best was the current logical `64x64` path (`/tmp/sm120_current_gap_condition_ab_20260528_qpkv2_local_rerun`). qwen27 qpkv6 S1024 causal has median >1 in the 10-repeat broad sweep and extreme subprocess outliers, so treat it as tiny/noisy. |
| qwen27 D256 qpkv6 S1024 causal Q-in-reg/static/hook forward variants | Rejected/no patch. Controlled B=2 Hq24/Hkv4 A/B `/tmp/sm120_qpkv6_b2_s1024_causal_qregs_source_ab_b782823_20260529` showed current auto already fastest: auto 0.240784 ms median, 1.039338x vs FA2, 1.010898x vs forced-off, 36/64 wins vs off, output diff vs off 0. Q-in-regs/static/hook variants all regressed to roughly 0.936-0.948x vs FA2. Keep the current auto path. |
| Qwen D128 qpkv4 S4096 lookup changes | Rejected in the qpkv4 lookup patch. Interleaved explicit-tile timing suggested S4096 wins, but subprocess public-API validation showed FA4-time regressions versus the saved post-qpkv artifact, so S4096 noncausal stayed `64x64` and S4096 causal stayed `64x96`. |
| Gemma31 D256 qpkv2 dense causal `64x48` | Interleaved timing looked mildly positive for S4096/S8192 causal, but public-API 5-repeat validation rejected shipping it: S4096 causal mean 0.989660 and S8192 causal median 0.979706. Noncausal must remain `64x64` for dense mask-skip behavior. Artifacts: `/tmp/sm120_gemma31_qpkv2_dense_condition_ab_20260528`, `/tmp/sm120_gemma31_qpkv2_patch_validation_20260528`. |
| qwen3.5-9B S4096 noncausal D256 qpkv4 and gemma4-e2b S8192 noncausal D256 qpkv8 broad-repeat misses | No dispatch patch. A later public-API 10-repeat check flipped both apparent 5-repeat misses into wins: qwen3.5-9B S4096 noncausal mean 1.015043 / median 1.024537, wins 6/10; gemma4-e2b S8192 noncausal mean 1.020579 / median 1.029481, wins 8/10. The matching interleaved A/B run mostly favored the existing `64x64` path; qwen S4096 `64x48` was only a tiny distinct hint and would disable dense mask-skip divisibility. Artifacts: `/tmp/sm120_broad_miss_qpkv4_qpkv8_condition_ab_20260528`, `/tmp/sm120_broad_miss_public_repeat_20260528`. |
| qwen3.5-9B D256 qpkv4 S1024 causal | No dispatch patch after `/tmp/sm120_qwen35_9b_s1024_causal_qpkv4_ab_20260529` and direct 61-round timing. The focused row already wins vs FA2, and explicit `64x64` is the same logical auto path; `64x32/64x48` hints were tiny/noisy and had larger output deltas. |
| Gemma31 D256 qpkv2 local extension | No dispatch patch after `/tmp/sm120_gemma_qpkv2_local_ab_20260528` and `/tmp/sm120_gemma31_qpkv2_s16384_local_tile_ab_20260529`. The new S16384 local repeat has current auto at mean-ratio geomean 1.047905 and median-ratio geomean 1.005957 across 5 rounds; alternates were slower than auto (`64x64` speedup 0.985224 mean / 0.977660 median, `64x16` 0.971649 / 0.981906, `64x32` 0.948836 / 0.984960, `64x48` 0.941193 / 0.963363). This supersedes the broad one-repeat S16384 miss as an actionable target. |
| Gemma D256 qpkv4/qpkv8 local wider tiles | Rejected after `/tmp/sm120_gemma_local_tile_ab_20260529` and direct auto-vs-pack-on timing. Wider `64x32/48/64` local tiles lost to current `64x16`; explicit `pack_on_64x16` was only small ordering noise versus auto and does not imply a product-code change. Long e2b B=1 S32768/S65536 repeat `/tmp/sm120_gemma_e2b_b1_s32768_s65536_local_tile_ab_20260529` confirms the broad local misses are not actionable: auto is already 1.145938x and 1.123497x vs FA2. `pack_on_64x16` is only +0.6%/+1.7% over auto, while wider and pack-off variants are worse; do not patch from this without a stricter repeat. |
| Gemma e2b D256 qpkv8 S65536 local outlier | Rejected as noise. The long sweep's 0.741x row did not reproduce: `/tmp/sm120_gemma_e2b_s65536_local_repeat_current_20260529` was 1.092x geomean, 5/5 wins, and the sidecar repeat `/tmp/sm120_gemma_e2b_s65536_local_qpkv8_repeat_20260529_sidecar` was 1.114x geomean. Keep the current `64x16` local path. |
| qwen3-14B D128 qpkv5 S4096 causal hook extension | No dispatch patch after `/tmp/sm120_qpkv5_hook_s4096_probe_20260529` and `/tmp/sm120_qpkv5_hook_s1024_s4096_repeat_20260529`. A first run favored V/both hooks, but the stricter repeat had auto at least as fast and showed the auto/off same-kernel variance dominating. Keep qpkv5 hook policy at the current S8192+ exact rows. |
| qwen3.6-35B D256 qpkv8 S4096 causal | No dispatch patch after direct 51-round timing. Auto was already 1.069x vs FA2 and explicit `64x64` was only +1.0% over the same logical path; `64x48` did not beat auto. |
| D128 qpkv7 old 60-cell rows | No dispatch patch after `/tmp/sm120_d128_old60_condition_ab_20260528`. The best qwen2.5 qpkv7 S8192/S16384 causal/noncausal candidates matched the current lookup or were too small/noisy to justify a change. |
| D64 MHA exact B1/H16 override for S1024/S8192/S16384 | Rejected after `/tmp/sm120_d64_mha_h16_exact_condition_ab_20260529`, `/tmp/sm120_d64_mha_h16_s16384_override_ab_20260529`, and a direct 61-round S16384 causal timing. S1024 and S8192 did not justify changing the current lookup. The apparent S16384 causal `128x32` win did not repeat; direct timing favored the existing `128x48` path (3.2247 ms) over patched auto/`128x32` (3.2658/3.3576 ms). |
| SM120 D256 backward N=32 path | Rejected for now. NCU showed FA2 uses a D256 N=32 main kernel and launches twice the CTA grid, so this was the right structural target to test. Verified N32/AtomLayoutNdKV=1 and N32/AtomLayoutNdKV=2 probes passed S128 correctness for qpkv2/qpkv4/qpkv6/qpkv8/qpkv16, causal and noncausal, but timing rejected them as defaults or row-specific dispatches. The verified S1024 causal targeted rerun showed Atom1 0.969x geomean vs current default and Atom2 0.964x, with qpkv6/qpkv4/qpkv2 regressions; only small Gemma qpkv8/qpkv4 hints remained. Keep N64 default; revisit only with a real source-level D256 variant, not a dispatcher tile flip. Artifacts: `/tmp/sm120_bwd_d256_ncu_20260528`, `/tmp/sm120_bwd_d256_n32_correctness_20260529`, `/tmp/sm120_bwd_d256_n32_targeted_verified_20260529`. |
| SM120 D256 backward qpkv2 no-overlap reload | Rejected. Disabling K-reload overlap for qpkv2 improved one noisy Gemma timing, but correctness failed for qpkv2 noncausal dK with max error around 2.3. Keep K-reload overlap enabled for all D256 GQA groups until a correct qpkv2-specific path exists. |
| SM120 D256 backward qpkv6 pack extensions beyond S4096 noncausal | Rejected for now. `/tmp/sm120_bwd_packgqa_qpkv6_ab_20260529` showed qpkv6 S8192 noncausal only +0.28% median and qpkv6 S4096 causal -1.1% median. A stricter S1024 noncausal repeat `/tmp/sm120_bwd_packgqa_qpkv6_s1024_false_true_20260529` had +1.3% median but mean was flat/slightly negative, so the row stays pack-off until a cleaner artifact appears. |
| SM120 D256 backward qpkv6 packed M-splits=1 | Rejected. `/tmp/sm120_bwd_qpkv6_split_policy_20260529` showed forcing `FLASH_ATTENTION_SM120_BWD_PACK_GQA_M_SPLITS=1` was much slower than the default split policy: qpkv6 S1024 noncausal 11.87 ms vs 1.20 ms default and S4096 noncausal 46.21 ms vs 15.48 ms default. |
| SM120 D256 backward qpkv16 S1024 causal explicit PackGQA | Rejected/no dispatch patch. `/tmp/sm120_bwd_qpkv16_s1024_c_pack_ab_b782823_20260529` showed explicit packed backward was only +0.22% by median and worse by mean/trimmed mean than current auto/nonpacked; true-pack won 27/48, not enough to default. |
| SM120 D256 backward qpkv4 S2048 causal packed split4/split8 | Superseded by the split16 keeper. The earlier `/tmp/sm120_bwd_qpkv4_s2048_c_pack_split_ab_b782823_20260529` only tested the old capped split policy: Hq8/Hkv2 split4 was +0.32% median with mean slightly worse and Hq16/Hkv4 split4 was +0.72% median/+1.12% mean. After unlocking true oversplits, split16 validated and is now the exact B=2 default for Hq8/Hkv2 and Hq16/Hkv4. |
| SM120 D256 backward qpkv4 S2048 causal mask-skip/nonpacked split probes | Rejected/no product patch. Current repeat `/tmp/sm120_bwd_d256_s2048_causal_current_5x_5dc126d_20260529` reproduced S2048 causal as the active D256 backward gap, but quick knobs did not close it. Mask-skip A/B `/tmp/sm120_bwd_s2048_causal_maskskip_ab_5dc126d_20260529`: Hq8/Hkv2 gained only +0.48% median while remaining about 7% behind FA2, Hq16/Hkv4 regressed -1.57% median, qpkv2/qpkv8 hints were too small/noisy. Nonpacked split dirty probe `/tmp/sm120_bwd_s2048_causal_nonpack_split_probe_dirty_5dc126d_20260529` had a tempting Hq16 split2 result, but exact validation `/tmp/sm120_bwd_qpkv4_h16_s2048_split2_default_validate_5dc126d_20260529` did not hold: old split1 had slightly better mean and 37/64 wins vs patched default. NCU points to mainloop underfill/long-scoreboard on Hq8, not mask/postprocess overhead. |
| SM120 D256 backward RS dKV layout switch for qpkv4 S2048 | Rejected. A dirty env-only source probe set `SdP_swapAB=True`, `AtomLayoutMSdP=1`, and `AtomLayoutNdKV=8` to try the existing `Mma_dKV_is_RS` branch. The exact B=2 Hq8/Hkv2 S2048 correctness compile failed before launch at `flash_bwd.py`'s swapped-SdP LSE shape assertion. A real RS variant would need LSE/layout work, not just a dispatch switch. |
| SM120 D256 qpkv8 noncausal length gating | No dispatch patch after `/tmp/sm120_bwd_d256_qpkv8_pack_length_ab_20260529`. The current qpkv8 auto-pack path stayed above FA2 at S1024/S4096/S8192/S16384 in the focused run (median ratios 1.073/1.012/1.009/1.075). Pack-off was only slightly faster than auto at S4096 and did not justify weakening the broad qpkv8 keeper. |
| SM120 D256 qpkv4 S4096 backward tile/Atom probe | No dispatch patch after `/tmp/sm120_bwd_d256_qpkv4_s4096_tile_atom_probe_20260529`. N32 AtomLayoutNdKV=1/2 regressed every qpkv4 S4096 cell, N48/N96 failed MMA partitioning, and the default path was already near parity in the targeted rerun. Remaining S4096 variance points back to structural D256 backward work, not a simple tile/Atom condition. |
| SM120 fused dK+dV postprocess expansion beyond exact keepers | No broader default-dispatch patch beyond packed split paths and the exact D256 qpkv8 S1024 causal keeper above. Forced fused dKV remains layout-safe for fixed non-varlen equal-D GQA and is available through `FLASH_ATTENTION_SM120_FUSED_DKV=on` for profiling, but broad timing was not stable enough to ship. Artifacts: `/tmp/sm120_fused_dkv_exact_rows_ab_20260529`, `/tmp/sm120_fused_dkv_exact_rows_repeat_20260529`, `/tmp/sm120_fused_dkv_qpkv8_s4096_confirm_20260529`, `/tmp/sm120_fused_dkv_qpkv8_s4096_default_validate_20260529`. The qwen3-30B D128 qpkv8 S4096 noncausal hint confirmed once (+0.7% mean) but then default validation regressed auto vs forced-off by mean, so keep it off by default. |
| SM120 D256 backward N32 recheck after fused-dKV gate | Still rejected. `/tmp/sm120_bwd_d256_n32_recheck_20260529` repeated the old Gemma hint, but it was not stable enough to ship: qwen3.5-0.8B and qwen3.5-9B qpkv4 S1024 causal regressed by mean, gemma4-e2b was flat, and only gemma4-e4b showed a small mean win. The later pack8 causal qpkv4 path captures the useful short-shape win without changing N to 32. |
| SM120 D256 qpkv8 S1024 causal N32 retry after fused-dKV | Still rejected. NCU showed qpkv8 H16 S1024 causal FA4 has fewer instructions/spills than FA2 but half the main-kernel grid, so N32/AtomLayoutNdKV=1/2 was retried. Interleaved A/B `/tmp/sm120_bwd_n32_qpkv8_s1024_causal_ab_20260529`: H16 regressed by about 7% median; H8 was parity/noise. Keep N64. |
| SM120 D256 qpkv6/qpkv8 S1024 causal explicit PackGQA after fused-dKV | Rejected. `/tmp/sm120_bwd_causal_pack_qpkv6_qpkv8_after_fused_20260529`: qpkv6 true-pack median 0.962x vs auto, qpkv8 H16 true-pack median 0.981x vs auto, qpkv8 H8 was mean-positive but median-negative/noisy. Do not add causal qpkv6/qpkv8 to auto-pack. |
| SM120 D256 qpkv6 S1024 causal fused-dKV | Rejected for default dispatch. `/tmp/sm120_bwd_fused_dkv_qpkv6_s1024_causal_20260529`: forced fused had a tiny median hint but mean regressed vs default/off and included large outliers. Keep qpkv6 fused-dKV off by default. |
| SM120 D256 qpkv6/qpkv8 S1024 causal nonpacked M-split=4 | Rejected as the default. `/tmp/sm120_bwd_nonpack_split_32x_20260529`: split4 had similar medians to split2 on some qpkv8 rows but more large outliers and worse mean stability; qpkv6 split4 won fewer paired rounds than split2. Keep split2 for qpkv6/wider qpkv8; only the exact Hq8/Hkv1 qpkv8 row uses split3. |
| SM120 D256 qpkv6/qpkv8 wider nonpacked M-split=3 | Rejected. `/tmp/sm120_bwd_nonpack_split3_wide_b2_after_541cd4b_20260529` and `/tmp/sm120_bwd_nonpack_split3_wide_b1_after_541cd4b_20260529` tested split3 for qpkv6 Hq24/Hkv4 and qpkv8 Hq16/Hkv2 S1024 causal after the qpkv4 split16 keeper. qpkv6 B=2 regressed by median/mean with only 22/64 paired wins; B=1 was median-flat/slightly worse. qpkv8 H16 had a B=1 hint, but B=2 was only +0.5% median with worse mean and 35/64 paired wins. Keep split3 limited to qpkv8 Hq8/Hkv1. |
| SM120 D256 qpkv6/qpkv8 S2048 causal nonpacked M-split=2 | Rejected/no product patch. Dirty source probe widened `FLASH_ATTENTION_SM120_BWD_NONPACK_M_SPLITS` eligibility to S2048, then compared split2 vs split1 in `/tmp/sm120_bwd_nonpack_s2048_split2_probe_dirty_20260529`. Split2 was slower by median and mean for qpkv6 Hq24/Hkv4, qpkv8 Hq8/Hkv1, and qpkv8 Hq16/Hkv2. Keep nonpacked split eligibility at S1024. |
| SM120 D256 split empty-epilogue skip | Rejected. A dynamic early `return` for empty M-split CTAs is not legal in CuTe kernels. A narrower epilogue-only guard passed the 18-row forced-split2 D256 SDPA subset, but timing did not justify shipping it: dirty-vs-base default split2 A/B artifacts `/tmp/sm120_bwd_empty_epilogue_base_2d7ce45_20260529` and `/tmp/sm120_bwd_empty_epilogue_dirty_20260529` had qpkv6 median faster but qpkv8 H8 and H16 median slower. Keep the simpler committed split2 path. |
| SM120 D256 backward K/V full-tile load predicate elision | Rejected. The env-gated source probe passed D256 SDPA correctness but timing did not hold: `/tmp/sm120_bwd_skip_kv_load_pred_ab_20260529` had qpkv6 S1024 causal median parity and mean regression, qpkv8 H8 parity/noise, qpkv8 H16 median-only +1.4% with mean regression, and qpkv6 S4096 noncausal median/mean regression. Do not skip K/V load predicates without a deeper SASS-level reason. |
| SM120 D256 backward Q/dO full-tile load predicate elision | Rejected and reverted. Env-gated `FLASH_ATTENTION_SM120_BWD_SKIP_FULL_M_LOAD_PRED=1` passed the D256 SDPA subset, but timing did not hold. Broad A/B `/tmp/sm120_bwd_full_m_load_pred_ab_20260529` regressed qpkv6 S1024 causal badly (0.890x geomean) and was flat overall (0.998x). Narrow repeat `/tmp/sm120_bwd_full_m_load_pred_exact_repeat_20260529` flattened the candidate rows: qpkv2 H32/H16 1.034x geomean but 0.999x median and 8/16 wins; qpkv8 H16 1.004x geomean, 1.022x median, 10/16 wins. Do not ship without SASS-level evidence. |
| SM120 D256 backward `acc_S_pre` score-mod guard | Rejected/no codegen change. Guarding the shadow `acc_S_pre` fragment behind `score_mod_bwd is not None` passed correctness, but NCU qpkv6 S1024 causal stayed identical to base: 255 regs, 83.968 KB smem, 51,981,312 instructions, 1,456,128 register-spill instructions. Artifacts: `/tmp/sm120_bwd_scorepre_guard_ncu_20260529`, prior noisy timing `/tmp/sm120_bwd_scorepre_only_ab_20260529`. |
| SM120 D256 backward K-reload overlap disable | Rejected. A source probe making the D256 alias path skip overlap of the K reload for dQ failed correctness immediately: `FLASH_ATTENTION_SM120_BWD_OVERLAP_K_FOR_DQ=off` hit dK max error 1.336 on `test_sm120_hd256_backward_matches_sdpa[4-2-False-False]`. A later qpkv6 S1024 causal-only retry after NCU long-scoreboard analysis also failed correctness with dK max error about 3.34. Keep the current overlap schedule. |
| SM120 D256 backward dO/V prefetch after S GEMM | Rejected. A compile-keyed dirty probe moved the aliased dO/V loads immediately after the S GEMM to hide latency under scalar softmax work. Correctness passed the full D256 SDPA subset with the flag forced on, but paired qpkv6 S1024 causal timing `/tmp/sm120_bwd_qpkv6_prefetch_dov_ab_20260529` was effectively flat: prefetch/base 1.0007. |
| SM120 D256 backward early K reload after dV | Rejected. A dirty probe launching the K reload alongside the Q reload after dV failed correctness immediately on `test_sm120_hd256_backward_matches_sdpa[4-2-False-False]` with dK max error about 1.33. The existing dK-GEMM hook remains the safe K-overlap point. |
| Qwen27 qpkv6 S1024 causal D256 backward NCU after mask-skip | Profiled, but not a source patch by itself. Current reports: `/tmp/sm120_qpkv6_s1024_bwd_current_29ad06c_raw.csv`, `/tmp/sm120_qpkv6_s1024_bwd_fa2_29ad06c_raw.csv`. FA4 main is now faster than FA2: 606.9 us vs 622.2 us, same 1536-CTA grid, 255 regs/thread, lower dynamic smem (82.9 KB vs 98.3 KB), fewer executed instructions (65.8M vs 67.2M), and far fewer local spilling requests (55K vs 897K). Remaining FA4 weakness is higher long-scoreboard stalls and L2 read sectors, not CTA count or simple predicate/mask overhead. |
| Qwen3-30B D128 qpkv8 S65536 causal NCU after qpkv16 keeper | Profiled, but not a source patch by itself. `/tmp/sm120_ncu_qpkv8_d128_s65536_after_qpkv16`: FA4 profiled faster than FA2 in the NCU pair (168.8 ms vs 177.8 ms) with the same 16,384-CTA grid, 128-thread blocks, 255 regs/thread, lower dynamic smem (49.2 KB vs 65.5 KB), and higher SM throughput (96.1% vs 90.0%). FA4 also executes more instructions (42.8B vs 31.2B) and has far higher shared-load bank conflicts (37.3M vs 17K), while L2 read sectors are essentially the same. This supports treating the repeat-level parity as source/SASS pressure/noise, not a tile or TMA dispatch miss. |
| Gemma e2b qpkv8 S1024 causal NCU after split2 | Profiled, but not a source patch by itself. `/tmp/sm120_ncu_gemma_e2b_bwd_20260529`: FA4 main 262.3 us vs FA2 main 247.9 us, same 512-CTA grid; FA4 postprocess is faster (dKV post 7.5 us vs FA2 reductions about 19.7 us), but the main kernel has more local-spill requests (661k vs 299k) and more L2 read sectors despite fewer instructions. This motivated the exact split3 row above; deeper remaining work is source/SASS D256 live-range pressure, not another broad tile lookup. |
| Gemma/Qwen-style qpkv8 H16 S1024 causal D256 backward NCU after split2 | Profiled, but not a source patch by itself. Current reports: `/tmp/sm120_ncu_qpkv8_h16_current_fb21d97/fa4_raw.csv`, `/tmp/sm120_ncu_qpkv8_h16_current_fb21d97/fa2_raw.csv`. The old underfilled-grid diagnosis is gone: FA4 and FA2 both launch 1024 CTAs. FA4 main is slightly slower in the NCU pair (440.0 us vs 430.7 us) despite fewer instructions (44.0M vs 44.8M), lower dynamic smem (82.9 KB vs 98.3 KB), fewer local-spill requests (462K vs 598K), no shared-spill requests, and far fewer shared-bank conflicts (673 vs 946K). Remaining issue is long-scoreboard/L2-read pressure: FA4 tex/global-read sectors are about 26.8M vs FA2 about 19.1M. |
| SM120 D256 split dK/dV combined zero allocation | Rejected. Broadening the existing packed-split combined `dkv_accum` zero buffer to nonpacked split rows passed the S1024 qpkv8 H8 SDPA check, but timing versus the `17ce38a` worktree did not hold: `/tmp/sm120_bwd_combined_zero_base_17ce38a_20260529` vs `/tmp/sm120_bwd_combined_zero_dirty_20260529` had slower dirty medians on qpkv6 H24, qpkv8 H8, and qpkv8 H16, with qpkv8 H8 only 14/48 paired wins. Keep separate dK/dV zero buffers for nonpacked split rows. |
| SM120 D256 qpkv4 S1024 causal packed split count below/above keepers | Partially rejected. `/tmp/sm120_bwd_qpkv4_s1024_pack_split_ab_20260529` rejected replacing pack8 with split2/4/6 or pack-off. Later oversplit probes accepted split16 only for Hq8/Hkv2. Broad split16 over all qpkv4 was not shipped because Hq16/Hkv4 had mean/outlier instability in `/tmp/sm120_bwd_qpkv4_s1024_split16_default_validate_20260529`; keep Hq16/Hkv4 on split8. |
| SM120 D128 fused dK+dV postprocess broadening | Rejected. Forced `FLASH_ATTENTION_SM120_FUSED_DKV=on` over five D128 GQA rows in `/tmp/sm120_bwd_d128_fused_dkv_ab_20260529` was mixed: qwen3-14B qpkv5 S8192 noncausal had only a tiny median hint, qwen3-30B qpkv8 S4096 noncausal regressed, qwen3-embedding qpkv4 S8192 causal regressed by median, and default/off/on variance was large on the tiny S1024 row. Do not broaden the fused-dKV default to D128 without a cleaner exact-row artifact. |
| SM120 D256 qpkv4 S1024 causal fused-dKV off gate | Rejected. `/tmp/sm120_bwd_qpkv4_s1024_fused_exact_20260529` compared current default, forced off, and forced on for Hq8/Hkv2 plus Hq16/Hkv4. Hq8 current default/fused stayed best (median 0.449 ms vs off 0.457, off only 18/64 paired wins). Hq16 forced-on was a same-path/noisy median hint, while forced-off did not help. Keep the existing packed-split fused-dKV policy. |
| SM120 D256 backward PackGQA power-of-two row-map | Rejected. PTX inspection already showed no real `div`/`rem` in qpkv2/qpkv4/qpkv8 PackGQA backward; the compiler strength-reduces the existing constexpr division. An env-gated shift/mask source probe passed the 18-row D256 SDPA subset, but timing did not hold. Broad A/B `/tmp/sm120_bwd_pow2_rowmap_ab_20260529`: median-geomean 0.990x and mean-geomean 0.979x for on/off. Candidate repeat `/tmp/sm120_bwd_pow2_rowmap_candidates_20260529`: median-geomean 0.985x, mean-geomean 0.991x; qpkv4 H16 S1024 causal was only +0.3% median with split wins. Do not replace the existing `idx // qhead_per_kvhead` map. |
| SM120 D256 backward non-alias `V_in_regs` schedule | Rejected. A dirty source probe disabled the SM120 Q/K+dO/V alias path by forcing the existing `V_in_regs` schedule for D256. This was the direct response to current qpkv8/qpkv6 L2-read pressure, but it is not launchable on SM120: the first D256 SDPA case failed with `cudaErrorInvalidValue`, reporting 148,480 bytes dynamic shared memory requested vs the 101,376 byte SM120 cap. Keep the alias/reload schedule unless a real smaller-smem D256 variant is written. |
| SM120 D256 backward cp.async cache-always policy | Rejected. An env-gated source probe changed the D256 SM120 alias path G2S copy atom from `LoadCacheMode.GLOBAL` to `ALWAYS`, targeting the qpkv8/qpkv6 L2-read/long-scoreboard gap. Correctness passed the 18-row D256 SDPA subset, but timing did not hold. qpkv6/qpkv8 H16 A/B `/tmp/sm120_bwd_cp_async_cache_ab_20260529`: qpkv6 regressed by median and trimmed mean; qpkv8 H16 showed a first-run +1.8% median hint. Repeat `/tmp/sm120_bwd_cp_async_cache_qpkv8_repeat_20260529` rejected it: qpkv8 H16 was flat at 0.9996x vs base and qpkv8 H8 regressed to 0.988x. Keep `LoadCacheMode.GLOBAL`. |
| SM120 D256 qpkv6 B=1 S1024 causal full-valid mask skip | Rejected for default dispatch. The committed full-causal-mask skip helps the exact B=2 qpkv6 S1024 causal row, but the B=1 extension did not hold. Env-forced B=1 A/B `/tmp/sm120_bwd_maskskip_b1_qpkv6_20260529`: on/off median 0.999x and trimmed-mean 0.971x despite 39/64 paired wins. Keep the default gate limited to the current B=2 rows. |
| SM120 TMA dense noncausal `check_inf=False` | Rejected. A dirty env-gated source probe skipped the softmax `row_max == -inf` guard when the existing dense noncausal seqlen-mask skip was active and no score/mask mods were present. Correctness smoke passed D64 MHA and D128 qpkv5 noncausal, but timing rejected it. `/tmp/sm120_fwd_tma_infcheck_ab_20260529`: overall on/off geomean 0.979x, median 0.976x, wins 3/18. D64 S8192 noncausal geomean 0.982x, D64 S16384 noncausal 0.983x, qpkv5 S16384 noncausal 0.973x with 0/6 wins. Keep TMA `check_inf=True`; these forward misses are not caused by this guard. |
| SM120 D256 qpkv2 S1024 causal nonpacked M-split=2 | Rejected for default dispatch. Env-gated qpkv2 split2 had a median hint, but did not meet the stability bar. Initial probe `/tmp/sm120_bwd_qpkv2_nonpack_split_probe_20260529`: split2 median 1.1329 ms vs default 1.1412 ms, 23/32 wins. Stricter repeat `/tmp/sm120_bwd_qpkv2_nonpack_split2_repeat_20260529`: split2 median 1.1064 ms vs default 1.1372 ms and 71/96 paired wins, but split2 mean regressed 1.2684 ms vs 1.2225 ms and paired geomean was 0.9865 because split2 had larger outliers. Keep qpkv2 on the current unsplit path. |
| Gemma4 D256 local S131072 tile widening | Rejected/no patch because the focused miss did not reproduce. `/tmp/sm120_gemma_local_s131072_tile_ab_20260529`: current auto was best for both B=1 long-local rows and beat FA2: e4b 1.1386x, e2b 1.0316x. Wider N tiles and pack-off variants were slower. |

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
- SM80/SM120 forward now rejects block-sparse `cu_seqlens` varlen explicitly;
  the SM80-base block-sparse mainloop uses non-varlen block indices.

## Open Targets

- Current broad reference is the post-qpkv-update 130-cell sweep at `3c34c6a`.
  Re-run it after any further dispatch/kernel changes, not before.
- qpkv8 D128: S8192 dispatch is retuned. S16384/S65536 causal are now
  parity/noise in focused repeats and S32768 Q-in-regs/static/mask probes are
  rejected; avoid broad old-commit restores.
- D128/qpkv5 small and mid noncausal rows: repeat before patching; several
  apparent misses have flipped with run order and FA2 variance. The current
  post-qpkv8-long 5-repeat broad sweep made qwen3-14B S4096 noncausal look
  stable, but the exact tile probe under
  `/tmp/sm120_qpkv5_s4096_nc_tile_ab_20260529` flipped it back into a current
  auto win. The exact non-TMA gate now addresses this row; do not add a
  qpkv5 S4096 noncausal tile rule without a new paired artifact that
  reproduces a tile-specific win.
- Gemma local qpkv4/qpkv8: keep `64x16`, but rerun broader repeats before
  extending to qpkv2 or dense Gemma shapes.
- D64 MHA after the current lookup patch: S8192 noncausal and S16384
  noncausal are improved but still close to parity in the public 60-cell run
  (0.973x and 0.991x). Further work should profile those exact public rows
  before another lookup change.
- D256 qpkv6 long causal/noncausal: exact B=1 long rows are now handled by
  the Q-in-regs keeper above. Further qpkv6 work should target source/SASS
  instruction pressure outside that exact gate, not more tile lookup sweeps.
- D256 qpkv6 S8192 causal and noncausal: both are now handled by the exact B=2
  Q-regs keeper above. The causal row uses the V-hook; the noncausal row keeps
  plain Q-regs because the V-hook repeat had lower median but worse mean/outlier
  behavior. Q-regs static causal bounds were rejected for correctness, so do
  not retry that route without a new mask/bounds proof.
- D256 backward performance: the Q/dO + K/V alias path fits SM120 and is
  correct. The exact qpkv6/qpkv8 S1024 causal CTA-count gap is now mostly
  addressed by the nonpacked split2 keeper, and qpkv4 short-causal rows were
  improved by the mask-skip expansion. S2048 causal qpkv4 is now addressed by
  the exact B=2 packed split16 keeper, moving the 5-repeat S2048 causal sweep
  to 1.034150 geomean vs FA2. S4096 noncausal is healthy in the current 5x
  repeat. Remaining D256 misses are qpkv8/qpkv2-style rows such as qwen3.6
  Hq16/Hkv2 and Gemma31; further work likely needs a dedicated D256 variant or
  per-Q-head dK/dV reduction, not another simple tile flip.
- D256 qpkv6/qpkv8 S1024 causal: N32, explicit causal PackGQA, qpkv6
  fused-dKV, K/V load-predicate elision, and `acc_S_pre` guard were all
  rejected. Keep nonpacked split2 as the current structural fix; split4 is
  rejected for default due outliers.
- Current 10-repeat short-mid sweep's three mean-ratio rows below 0.98 were
  rechecked and are not actionable. qwen3.5/qwen3.6-27B S1024 causal D256
  qpkv6 is tiny/noisy; gemma4-31B S4096 local D256 qpkv2 and qwen3.5-122B
  S8192 causal D256 qpkv16 both flipped to parity or wins in targeted reruns.
  Do not add a dispatch condition for these rows without a new paired artifact
  that reproduces a stable miss.

## 2026-06-02 — RTX PRO 6000 Blackwell migration

Target GPU is now the **RTX PRO 6000 Blackwell Workstation Edition** (sm_120,
**188 SMs**, 102 GB), not the 170-SM RTX 5090 (occupied by llama-server). Prior
win.md numbers were 5090-tuned and do NOT transfer: FA2 scales better on the
larger part, so backward FA4/FA2 is worse here (S2048 bwd geomean ~0.98 vs the
5090's 1.034). Bench scripts read `SM120_BENCH_GPU` (GPU-33a7e490 = GPU 0,
GPU-a861102e = GPU 3); 5090 asserts relaxed to any sm_120. Cross-impl repeats
are very noisy at S1024 (clock-boost variance between FA2/FA4 subprocesses);
trust the interleaved in-process A/B (within-round speedup_vs_auto).

Commit 7a6e8c8 — D128 forward tile retune + qpkv2 backward split:
- Forward broad map (3x, B=2) showed a systematic D128 forward loss
  (**geomean 0.955** vs FA2; D256 fwd wins at 1.027). The `_SM120_TILE_LOOKUP`
  D128 entries (5090-tuned) mostly used `(64,64,1)`, worse than even the generic
  `(128,64)` fallback here. Retuned 7 entries (each matches auto within bf16 tol):
  (128,4,1024,0)/(128,4,2048,0)/(128,4,4096,0) 64x64->128x64;
  (128,4,4096,1) 64x96->128x48; (128,8,1024,1) 64x64->64x128;
  (128,8,4096,0) 64x64->128x64 (1.28x vs auto); (128,8,4096,1) 64x64->128x64.
  Net: **D128 fwd 0.955 -> ~1.00**, overall fwd 1.004 -> 1.021, D256 unchanged.
- Backward: extended nonpacked-M-split eligibility to qpkv2 Gemma31
  (Hq32/Hkv16) D256 S1024 causal -> split2, **+7.3% FA4** (was a 5090 regression).

Follow-up D256 forward fixes:
- qpkv6 D256 causal: qwen3.5-27b (Hq24/Hkv4) S4096 causal B=2 flips 0.976 ->
  **1.067** vs FA2 by enabling the existing Q-in-regs `128x64_t256` schedule
  (was gated to causal S8192 only; now S4096 too). S1024 causal is already
  1.034 (the map's 0.959 was subprocess noise).
- Gemma local D256: qpkv8 (Gemma e2b, window 512) local N tile 16 -> 32, ~+7%
  among shuffled-tile A/B (S4096 1.08x, S8192 ~tied); qpkv4 (e4b) stays N=16
  (it is competitive at N=16, only marginally better at N=64). These rows were
  already >=1.04 vs FA2 in-process; the map's sub-1.0 was subprocess noise.

Methodology note: sm120_fwd_exact_tile_ab.py measures `auto` first every round
(not shuffled), so its speedup_vs_auto is cold-clock-biased upward; trust
comparisons AMONG the shuffled explicit tiles, and validate net via the
cross-impl re-map (which moved D128 fwd 0.955 -> ~1.00).

Confirmed exhausted (RTX 6000, dispatch-level): D256 S2048 backward gaps are a
hard 1-CTA/SM occupancy wall — 255 reg/thread is dominated by acc_dK/acc_dV
(shape n_block x 256, independent of m_block); n_block=32 is rejected, so
2 CTAs/SM is unreachable. qpkv4 S2048 split16 is already FA4-optimal here;
qpkv8/qpkv2 S2048 splitting is flat/harmful. These need a kernel redesign,
not tile/split dispatch.

## 2026-06-02 — overnight backward campaign: full lever sweep on RTX 6000

Backward maps (5x cross-impl): the backward problem is CAUSAL-specific.
S2048 noncausal geomean 1.046 (28/40 wins, healthy); S2048 causal 0.979 and
S4096 causal 0.980 (12/40 wins). Consistent causal losers across S2048+S4096:
qpkv4 (0.93 Hq8 / 0.965 Hq16), qpkv8 (0.974), qpkv2 (0.977); winners are
qpkv16/qpkv6. (Only noncausal loser: qwen3.5-9b qpkv4 S2048 nc 0.967.)

Every safe lever for the causal S2048/S4096 gaps was tested and REJECTED on the
RTX 6000:
- nonpack M-split at S2048/S4096 (all qpkv): flat-to-harmful (split hurts ~2-4%);
  the underfill that makes split help only exists at S1024. Keep split S1024-only.
- packed-GQA split for qpkv4 S4096 causal: all splits slower than unpacked
  (off/mode 0.95-1.0). Keep qpkv4 packed only at S1024/S2048.
- mask-skip at S2048/S4096 causal (all qpkv): "on" is ~4% SLOWER than off
  (speed_vs_off 0.96-0.98, wins 0-9/40). Keep mask-skip S1024-only.
- RS-dKV (Mma_dKV_is_RS) kernel change: REJECTED by analysis — holding P/dS in
  registers would add ~+512 reg/thread (registers are already the binding cap),
  sP/sdS smem isn't even conditionally freed, and it needs a deep softmax-loop
  rewrite. Would worsen the occupancy wall, not fix it.
- cp.async cache policy (GLOBAL->ALWAYS): rejected on reasoning — K/V/Q/dO are
  read ONCE from gmem into smem (reused from smem, not gmem), so the L1 cache
  hint can't help; the long-scoreboard stall is on smem->reg dependency chains,
  not the already-async gmem->smem copy. (Consistent with the 5090 rejection.)

NEW backward wins via the grid-underfill principle (split helps only when the
backward grid ~ceil(S/64)*B*Hq underfills the 188 SMs, <~3 waves):
- B>=2 S2048: only the smallest grid, qpkv8 Hq8/Hkv1 (gemma-e2b, ~2.7 waves),
  still underfills -> nonpack split3 (+6%). Larger qpkv8 Hq16/Hkv2, qpkv6,
  qpkv2 are filled at B>=2 S2048 (split flat/harmful, unchanged).
- B=1 S2048 (grid halved): the small-grid rows (num_head<=24) all underfill ->
  split4 (+4% to +9%): qpkv6 Hq24/Hkv4, qpkv8 Hq8/Hkv1 + Hq16/Hkv2, and qpkv4
  Hq8/Hkv2 (+9.3%) + Hq16/Hkv4 (+7.7%). qpkv4 at B=1 is NOT auto-packed (pack
  needs B=2) and prefers nonpack split4 over packed split16 here. Larger-Hq
  rows (e.g. qpkv4 Hq32/Hkv8, ~5 waves) are filled and left unsplit.
- B=1 S4096: the two smallest grids still underfill -> qpkv8 Hq8/Hkv1 split6
  (+10%), qpkv4 Hq8/Hkv2 split4 (+7%). qpkv8 Hq16/Hkv2, qpkv6, and qpkv4
  Hq16/Hkv4 are filled by S4096 (flat, unchanged).
- B=1 S1024: the tiny B=1 grid wants MORE splits than the B>=2-tuned defaults:
  qpkv8 Hq8/Hkv1 split3->split4 (+6%), qpkv8 Hq16/Hkv2 split2->split4 (+9%).
  qpkv6/qpkv2 B=1 S1024 are marginal (kept at split2). B>=2 S1024 keeps 3/2/2.
All gradients match split1 within bf16 tol. The nonpack policy was refactored
into a single batch/seqlen/shape-gated block; the env override
FLASH_ATTENTION_SM120_BWD_NONPACK_M_SPLITS now works for any D256-causal-nonpack
shape (no dirty edits needed to probe new shapes).

CONCLUSION: the D256 causal backward (~0.93-0.98 vs FA2 on the 188-SM RTX 6000)
is at its practical limit for dispatch/knob tuning. The only remaining lever is
a ground-up D256 backward redesign to break the 1-CTA/SM occupancy wall, e.g.
a Blackwell-native UMMA/tcgen05 backward (cf. flash_bwd_sm100) or a head_dim-
split dK/dV accumulation to cut register pressure. That is a multi-day project,
not overnight tuning. The one overnight backward win was qpkv2 Gemma31 S1024
causal -> split2 (committed 7a6e8c8).

## 2026-06-02 — multi-agent across-the-board pass

Parallel design/analysis agents + GPU validation. Outcomes:
- FORWARD is fully dispatch-tuned. qpkv5 D128 confirmed kernel-limited (no tile
  beats the current auto; S4096 causal 0.959 is the FA4 best); the other forward
  "losers" (qpkv5/qpkv6 S1024) are cross-impl map noise (>=1.0 in-process). No
  forward dispatch wins remain.
- D256 backward KERNEL-body changes all rejected by design analysis: head_dim-
  split dK/dV lands ~223 reg/thread (need <=128 for 2 CTAs/SM); can't free enough
  smem for a 2nd stage (sK/sQ are 32KB each, unavoidable; 2nd stage >99KB);
  cp.async reorder is alias-bound (the exposed Q-reload before MMA dK can't be
  deferred — dK reads it immediately). See memory sm120-backward-kernel-changes-rejected.
- D128 BACKWARD map (B=2): geomean 0.986. Reliable losses are at S8192
  NONCAUSAL (qpkv4 0.83, qpkv8 0.84) — a FILLED grid, so M-split does not apply;
  the lever would be num_stages=2, but D128 stages=2 EXCEEDS the 99KB smem cap
  (cudaErrorInvalidValue at launch). Closing it needs the D256 reuse_qk_dov_smem
  alias extended to D128 to free stage room — a kernel change with uncertain
  payoff (alias Q-reload overhead vs pipelining benefit); deferred to supervised
  work. D128 short-causal backward is healthy (>=1.0).

NET: dispatch-level tuning is exhausted across forward AND backward on the RTX
6000. Both residual gap classes (D256 causal large-grid bwd; D128 S8192-nc bwd)
are smem/occupancy-walled and need a kernel redesign, not a dispatch knob.

## 2026-06-02 — D128 long-seq backward asymmetric pipeline stage (kernel-config)

Kernel-config win for the D128 S8192-nc backward gap: D128 (no smem alias) has
room for a 2nd Q-load stage. Symmetric ns=2 overflows the 99KB cap (verified:
cudaErrorInvalidValue at launch), but ASYMMETRIC num_stages_Q=2 / num_stages_dO=1
fits (~98KB) and pipelines the Q loads. Gated to D128 dense S>=8192 (short seq
regresses ~14% from async overhead, matching the original SM120 ns=1 rationale).
Controlled in-process A/B (stages1 vs stages2, same inputs, interleaved): a
consistent **+2.0% (nc) / +1.5% (causal)** FA4 speedup at S8192 across qpkv4/5/8,
no regressions; gradients match fp32 SDPA. (The single-shot +26% and cross-impl
+6% were clock noise; +2% is the controlled truth.) Modest but real and safe;
the D128 nc backward still trails FA2 (~0.85-0.88) — fully closing it needs the
same kernel redesign as D256.

## 2026-06-02 — DEFINITIVE: sm_120 has no tcgen05/TMEM/WGMMA (hardware limit)

flash_fwd_sm120_tma.py header: sm_120 (consumer Blackwell, RTX PRO 6000 / 5090)
has "No WGMMA, no tcgen05, no TMEM" — only SM80-era mma.sync.m16n8k16 tensor
cores. Both fwd and bwd subclass the SM80 kernels (arch=80 for MMA selection).
CONSEQUENCE: the flash_bwd_sm100 (tcgen05/UMMA, tensor-memory accumulators)
approach CANNOT run on sm_120 — the instructions don't exist. With only mma.sync,
the D256 backward accumulators (dK/dV/dQ) must be register-resident, so the
255-reg / 1-CTA-per-SM occupancy wall is INTRINSIC to the hardware. There is no
FA4 kernel redesign that breaks it on this GPU. The D256 causal large-grid
backward (~0.93-0.98 vs FA2) is at its true hardware limit; all achievable wins
are dispatch-level + the one D128 long-seq stage config. Campaign complete.

## 2026-06-02 — history-mining pass: recover overlooked shape wins

Mined the full reject/supersede commit history + win.md for 5090-dropped configs
that won on a specific shape (re-test on 188-SM RTX 6000). Outcomes:
- N32 backward path (small-grid Gemma qpkv8/qpkv6, B3): re-REJECTED — 10-17%
  slower than the current N64+nonpack-split default (split beats N32 for underfill).
- qpkv4 Hq16/Hkv4 S1024 packed split16 (B4): TIED with current split8 (+0.09%);
  the 5090 outlier instability is just parity here. No change.
- qpkv6 D256 S16384 causal FORWARD static causal block bounds (F1): RECOVERED.
  Was a regression on the 5090 (rejected at S16384) but a clean GPU-specific
  flip here: +1.6% (controlled A/B; output bit-identical, max_abs_diff=0). Gain
  scales with S (+0.65% S8192, +1.6% S16384, +2.8% S32768). Added S16384 to the
  default (S32768/65536 already shipped). S8192 excluded (qregs is on there;
  static+qregs is the known wrong-output combo). qpkv6 S16384 causal fwd FA4/FA2
  0.956 -> 0.967.
Mining confirms the earlier tuning was thorough — little overlooked speed remained.
- fused-dKV broadening (B5): re-tested qpkv6 S1024 causal + D128 S8192-nc rows;
  flat/noisy (qpkv6 median +0.9% but mean regressed on outliers; D128 nc tied).
  No change. Net mining result: 1 of ~5 candidates recovered (F1, +1.6%);
  the rest confirm the prior tuning was thorough.

## 2026-06-02 — double-check (correctness audit of all campaign changes)

Holistic correctness sweep (FA4 vs fp32 SDPA / FA2) across every shape touched
this campaign: 11/12 PASS within bf16 tol (out_rel ~2-3e-3, grad_rel ~3-5e-3) —
D128 fwd tiles, D128 S8192 stages2, D256 qpkv6 qregs, qpkv6 S16384 static
(out bit-identical), all backward splits (qpkv2/4/6/8, B=1 & B=2). The gemma
qpkv8 D256 LOCAL row: my change was the FORWARD tile (64x16->64x32) and its
FORWARD output is correct (matches FA2, out_rel 1e-3).

DISCOVERED — PRE-EXISTING (NOT from this campaign): FA4-cute local/sliding-window
BACKWARD gradients diverge ~1490x from FA2/SDPA (forward is correct) for the
gemma local shape (qpkv8 Hq8/Hkv1 D256, window 512), for both causal=True+ws and
causal=False+ws invocations. CONFIRMED present at dfb7a24 (campaign baseline) with
identical dq_rel=1.49e3, so it predates all campaign commits. My changes never
touch the local backward path (nonpack splits are `not local`-gated, the stages
change is D128-only, the tile changes are forward-only). Flagged for separate
investigation — likely FA4-cute local backward never applies the K/V window mask
to dK/dV/dQ. The benchmark suite never caught it (gemma was benched local-FORWARD
and full-causal-BACKWARD only). Not a regression introduced here.

## 2026-06-02 — FIXED: SM120/SM80 local/sliding-window backward (correctness + speed)

The discovered pre-existing local-backward bug is now FIXED (2 commits):
- c29cc0e (correctness): backward never applied the window (mask_causal only,
  window params dead). For a local request causal resolves False, so it
  recomputed FULL attention vs the windowed-forward LSE -> garbage grads. Added
  is_local + threaded window_size into the device kernel + apply the window in
  the recompute mask. gemma-local backward grad_rel ~3e-3 (was ~1490x) vs SDPA;
  causal/full byte-identical (no-op when not local).
- 50cc3b6 (speed): the correctness fix left the m-range full (S^2 triangle),
  making local bwd 2-7x slower than FA2. Added the windowed m-block range
  (mirror BlockInfo.get_m_block_min_max), gated `if self.is_local`. Local bwd
  FA4/FA2 now 0.87 (S2048) / 0.94 (S4096) / 1.01 (S8192) -- 1.9-8.5x faster,
  competitive-to-winning vs FA2. SM90/SM100 unaffected (separate files).

## 2026-06-02 — sibling-bug hunt after the local-backward fix

Probed backward feature combos the benchmark suite never exercised (vs SDPA):
- D256 local non-causal, D128 local causal, softcap causal, softcap+local:
  all CORRECT now (grad_rel ~3-4e-3) — the local-backward fix is robust and
  composes with softcap and D128.
- DISCOVERED (separate, PRE-EXISTING, uncommon): symmetric BIDIRECTIONAL window
  (window_size_left>0 AND window_size_right>0, non-causal) is wrong in the
  FA4-cute FORWARD (FA4 vs SDPA out_rel 0.34; FA2 correct at 0.0025), so the
  backward inherits it. The forward threads window_size_right through but
  mis-handles it. No target model uses symmetric bidirectional windows (gemma
  is causal-local), so low priority; needs a separate forward-kernel fix.
  Common cases (causal, causal-local/sliding-window, full, softcap) are correct.

## 2026-06-02 — FIXED: non-causal symmetric bidirectional window (forward+backward)

The bidirectional-window bug (window_left>0 AND window_right>0) is now FIXED
(commit 318eccb). Root cause: the SM80-base forward (flash_fwd.py, used by SM120
non-TMA / D256) re-processed the first n-block for query rows where
i+window_right >= seqlen — get_n_block_min_causal_local_mask returns >= n_block_max
there, so the unmasked Phase-3 loop re-ran the already-processed first block
(double-count -> wrong; out-of-range read -> NaN for large window_right). Only
those boundary rows were affected; causal/causal-local/full were correct. SM90
already caps n_block_max the same way. Fix: clamp unmasked_n_block_start to
n_block_max-1 (no-op for causal). Forward out_rel 0.44->~2e-3; backward
grad_rel ~3-5e-3 (the window_right wiring from the local-backward fix already
handled the backward). Regression tests added (local backward + bidirectional).
This is a general fix for all arches using flash_fwd.py, not SM120-specific.

## 2026-06-02 — investigated the "sm_120 feature gaps" (NOT real gaps)

The test_suite failures earlier attributed to SplitKV/combine, d!=dv (MLA), and
varlen gaps were re-investigated per-file/in-isolation:
- d!=dv (test_flash_attn_sm120_dgtdv.py): 12/12 PASS in isolation. Works.
- combine / SplitKV-combine kernel (test_flash_attn_combine): PASSES in isolation
  (12/12); its full-run failures were a CASCADE from varlen failures corrupting
  the CUDA context in the same process. The combine kernel works.
- varlen (test_flash_attn_varlen_unpad_output): EVERY param passes alone, but the
  full parametrization fails ~45/96. NOT the compile cache (cache-disabled: same
  ~43 fails), ORDER-DEPENDENT (failing set changes under `-n 8`), and
  VARLEN-SPECIFIC (dgtdv/combine each compile 12 distinct kernels with 0 fails).
  => pre-existing varlen-specific cross-test CUDA/process-state contamination
  (wrong numerical output ~1.37 for later cases), a TEST-HARNESS issue, not a
  kernel feature gap. Present at baseline dfb7a24. Does not affect real usage
  (which compiles few kernels and reuses them) or any campaign change.

CONCLUSION: no real sm_120 feature gaps among these — the kernels work. The
failures are a pre-existing test-batching artifact; run via the documented
two-pass workflow or smaller per-process batches. Not a kernel fix.

## 2026-06-02 — CORRECTION + FIX: the "feature gaps" were a real varlen+GQA bug

My earlier "test-batching contamination" diagnosis was WRONG. Re-investigation
(FA4 varlen vs FA4 dense batch=N, the gold comparison) found a REAL bug:
varlen + GQA (qpkv>1) forward returned GARBAGE for sequence index >= 1 (seq0 OK,
seq1+ rel ~1.0), for ALL head dims; MHA correct. Root cause: flash_fwd.py:918
offset the pack_gqa composite mode 0 (qpkv, seqlen_q) with a SCALAR token
offset_q -> crd2idx decomposes it colexicographically -> wrong packed-Q base for
qpkv>1, batch>0. Fix (d27ebd0): offset the seqlen sub-mode ((None, offset_q), 0)
when pack_gqa, matching the O/LSE offset_batch_Q epilogue.

Validation: varlen GQA/MQA (qpkv4/8, D64/128/256, causal+nc, 2/3/4 seqs) now
rel 0.0 vs dense (was ~1.0); backward already correct; MHA unchanged. The FULL
varlen test passes 96/96 in one batch run -> there was NO contamination; the
combine failures were cascade from the GQA garbage corrupting the process. This
is a HIGH-IMPACT fix: GQA + packed/varlen sequences is ubiquitous, and it was
silently wrong for all but the first sequence on sm_120. General fix (all arches
on flash_fwd.py). Only remaining cute-test failure: marginal pre-existing qpkv16
backward tolerance (0.0553 vs 0.05).

## varlen + GQA BACKWARD dK/dV — non-block-aligned cu_seqlens (FIXED, flash_bwd.py)
A second, distinct varlen+GQA bug (sibling to the forward Q-offset one): the
backward returned GARBAGE dK/dV (~100% rel err) whenever a sequence start offset
in cu_seqlens_k was not a multiple of n_block_size (64). dQ correct, fwd correct,
MHA correct, aligned-offset varlen correct. This is the real cause of the
test_flash_attn_varlen.py failures (the failing params were softmax_scale=0.1
GQA/MQA — generate_varlen_args uses randint seqlens => non-aligned offsets; the
0.1 was a red herring, scale is applied correctly: FA4 vs SDPA rel ~3e-3).

Root cause: varlen bwd uses pack_gqa=False (interface.py:3119) -> the GQA
atomic-add dK/dV path WROTE dk_accum/dv_accum at raw `seqlen.offset_k +
batch_idx*n_block_size`, but the dKV postprocess READER floors to a block
boundary (seqlen.padded_offset_k). They agree iff offset_k % 64 == 0. dQ was fine
because writer+reader both floor (padded_offset_q). Fix (1 line, flash_bwd.py
~1585): `padded_offset_k = seqlen.padded_offset_k`. seqlen created with
tile_n=n_block_size; cluster_size==1 on this path. General fix (all arches on the
SM80-base bwd). Validated: repro all-ok; exact failing test configs dk/dv
~5e-4..5e-3 (D64/128/256, causal+nc, fp16+bf16); pytest GQA/MQA scale=0.1 subset
288 green (72 + 216). PRE-EXISTING (baseline dfb7a24; non-local/non-causal, so
is_local-gated bwd + cu_seqlens_q-gated splits never touched it).

## 2026-06-02 — MAJOR: the sweep's "losses" are a clock-boost MEASUREMENT artifact

Re-investigated the top forward AND backward gaps (gemma4-e4b S1024 causal
0.715x; D128 backward S8192 noncausal qpkv4 0.856x; the whole bwd-d128 0.966x
geomean). With a controlled IN-PROCESS interleaved A/B on the exact shapes,
every one is parity-to-WIN, not a loss:

  Forward S1024 (in-process FA4/FA2 vs subprocess sweep ratio):
    gemma4-e4b   c=1 D256 qpkv4 : 1.150  (sweep 0.715)
    qwen3.5-27b  c=1 D256 qpkv6 : 0.968  (sweep 0.912)
    qwen3.5-9b   c=0 D256 qpkv4 : 0.993  (sweep 0.947)
    qwen3.5-27b  c=0 D256 qpkv6 : 0.975  (sweep 0.951)
    qwen3-14b    c=1 D128 qpkv5 : 1.010  (sweep 0.954)
    qwen3.5-122b c=0 D256 qpkv16: 1.007  (sweep 0.974)
  Backward (backward-only timing, in-process interleaved vs sweep):
    qwen3-vl-8b  S8192 c=0 qpkv4 : 1.002 (sweep 0.856)
    qwen3-embed  S8192 c=0 qpkv4 : 1.001 (sweep 0.866)
    qwen3-30b    S8192 c=0 qpkv8 : 1.008 (sweep 0.902)
    qwen3-vl-8b  S8192 c=1 qpkv4 : 0.999 (sweep 0.892)
    qwen3-30b    S8192 c=1 qpkv8 : 1.012 (sweep 0.943)

ROOT CAUSE (proven): model_shape_bench_runner.py used warmup=2, iters=5, and
model_variant_matrix.py runs `for impl in ["fa2","fa4"]` — fa2 ALWAYS first in
a fresh, cold/boosted-clock subprocess; fa4 second when the card has warmed and
throttled. On a 24 ms S8192 backward the GPU boosts at the start of a burst then
settles: a single run shows p10=19.4 ms but median=24 ms — a 24% swing WITHIN one
run. median-of-5 lands fa2 in the boosted regime (~20 ms) and fa4 in steady
state (~24 ms) -> spurious 0.84-0.90. Reproduced 3x at the old setting
(0.844/0.901/0.891); at warmup=10 iters=40 it is rock-stable 0.997/1.001/0.998.
Same mechanism at S1024 (0.1 ms kernels never ramp clocks in 2 warmup iters).

This means the campaign's reported standing (fwd 1.015x, bwd-d128 0.966x / 7/24
wins) is ARTIFICIALLY PESSIMISTIC — biased systematically against FA4 by the
fa2-first ordering + under-warmup. The real in-process picture is parity-to-win
across the board, including the "primary place to improve" (D128 backward).

FIX (agent_space, gitignored, no commit): model_shape_bench_runner.py now does a
WALL-CLOCK warmup soak (`--warmup-ms`, default 400) so clocks reach steady state
regardless of kernel duration, and the sweep passes --warmup 5 --iters 40
--warmup-ms 400. Re-running the full fwd + bwd-d128 sweeps with the corrected
harness to establish the TRUE baseline before chasing any further "gap".

IMPLICATION: gaps #2 (gemma4-e4b causal) and #3 (small-S wide-GQA D256) from the
campaign brief are PHANTOM. Gap #1 (D128 bwd large-S) is also phantom (parity
in-process). Do NOT spend kernel/dispatch effort on these — verify any future
"loss" in-process interleaved with adequate warmup BEFORE treating it as real.

## 2026-06-02 — WIN: qwen3-14b S4096 causal D128 qpkv5 fwd tile 64x128 -> 128x64

After correcting the harness, the ONE genuine forward loss confirmed in-process
is qwen3-14b (Hq40/Hkv8 D128 qpkv5, non-pack -> TMA path) S4096 causal: steady-
state ~0.906x vs FA2 (block0 cold reads 0.98 but sustained-load settles ~0.91).
Tile sweep via an in-process env probe (interleaved, all variants same inputs):
  cur(64,128,1)=0.906  128,64,1=0.932  128,48,1=0.907  64,96,1=0.892
  64,64,1=0.823  128,128,1=0.608  128,32,1=0.831  64,64,2=0.828  128,64,1,256=0.932
128x64 is best: +2.8% (0.906->0.932, fa4 1.6875->1.6414 ms), confirmed across two
runs (1.028x and 1.045x vs cur). Still a slight loss vs FA2 but the gap is
narrowed. Correctness: FA4 vs SDPA rel 9.3e-4 (bf16, PASS).

SCOPE: changed ONLY the (128,5,4096,1) lookup cell. S1024 (cur 1.004x, already a
win) and S8192 (cur 1.011x, already a win) REGRESS under 128x64 (0.916 / 0.972),
so they keep 64x128. Surgical one-cell edit, interface.py. (commit pending)

## 2026-06-02 — TRUE backward-d128 standing (in-process, contention-robust)

Built an in-process interleaved full bwd-d128 matrix (agent_space/
sm120_bwd_d128_inproc_matrix.py) to bypass the subprocess clock-boost artifact.
Headline geomean reads 0.95 / 6-of-24, BUT that is dragged entirely by the
S1024 cells, which are NOISE-dominated (1-1.6 ms multi-launch kernels; the same
cell read 0.84 in the matrix and 1.00 in an earlier verify). With adequate
warmup(4)/iters(20), the real-weight cells are at PARITY:

  S8192 (all 8 cells): 0.982-1.010   -> parity
  S4096 (focused reconfirm, 8 interleaved blocks):
    qwen3-vl  c=1 qpkv4 = 0.992   qwen3-30b c=1 qpkv8 = 1.016
    qwen3-vl  c=0 qpkv4 = 0.993
  S1024: 0.79-1.07, unstable -> not reliably measurable (esp. under the user's
    concurrent bench_nvfp4 GPU-cycling job; qwen3-30b S4096 spread was
    0.368-2.844 single-block, median still recovered 1.016).

CONCLUSION: backward D128 is at parity for all training-relevant seqlens
(S>=4096). The campaign brief's "backward D128 is a net LOSS (0.966, 7/24),
primary place to improve" was the measurement artifact + S1024 noise, NOT a real
kernel deficit. Gap #1 (D128 bwd large-S noncausal qpkv4) is confirmed PHANTOM
(S8192 nc qpkv4 = 0.993-1.001). S1024 backward MAY have a small real loss but is
unmeasurable under current contention; defer to a clean-GPU session (candidate
lever: nonpack M-split for the underfilled small-S grid, see
[[sm120-backward-split-underfill-principle]]). No kernel change warranted now.

## 2026-06-02 — S1024 backward RESOLVED: parity-to-WIN, not a loss

Settled the S1024 D128 backward question with 20 interleaved blocks,
warmup=5/iters=40 (agent_space/sm120_bwd_s1024_settle.py). Despite the user's
concurrent bench_nvfp4 contention, the median over 20 blocks is a clean WIN:
  qwen3-vl  c=0 qpkv4 = 1.027 IQR[1.014,1.061]
  qwen3-vl  c=1 qpkv4 = 1.044 IQR[1.024,1.106]
  qwen3-30b c=1 qpkv8 = 1.041 IQR[1.009,1.116]
  qwen3-embed c=1 qpkv4= 1.048 IQR[0.999,1.082]
The earlier 0.78-0.84 readings were undersampling (5 blocks, fewer iters).

DEFINITIVE: backward D128 is parity-to-WIN at EVERY seqlen (S1024/4096/8192).
There is NO real backward-D128 gap to close — the entire "0.966 / 7-of-24 net
loss, primary place to improve" premise was 100% the clock-boost + fa2-first
measurement artifact. No backward kernel/dispatch work is warranted on D128.
The nonpack-M-split lever (D256-only anyway) is NOT needed here.

## 2026-06-02 — BIG WIN: general D256 forward wide tile (128x64+Qregs+256t) at S>=4096

The D256 forward was forced to a 64x64 tile only because 128x64 won't fit
Q+K+V in the 99 KB SMEM cap. Staging Q through registers (smem = max(Q,V)+K)
makes 128x64 fit, and it is much faster. The existing qregs paths enabled this
only for a handful of narrow B=1 long-seq shapes; it generalizes to ALL D256.

Discovered by an autonomous FA4-vs-FA4 in-process tile explorer
(agent_space/sm120_fwd_tile_explore.py) + robust confirm
(sm120_d256_wide_confirm.py, 12 interleaved blocks). At S>=4096 (square),
128x64+Qregs+256t beats 64x64 by (wide/cur):
  qpkv4 (9b)   : S4096 c0 1.114 c1 1.058 | S8192 c0 1.139 c1 1.117
  qpkv8 (35b)  : S4096 c0 1.115 c1 1.065 | S8192 c0 1.139 c1 1.121
  qpkv16 (122b): S4096 c0 1.140 c1 1.113 | S8192 c0 1.137 c1 1.126
  qpkv4 (0.8b) : S4096 c0 1.134 c1 1.000 | S8192 c0 1.110 c1 1.088
vs FA2 these shapes go from ~1.01 to ~1.09-1.16. Output is BIT-IDENTICAL to the
64x64 path (probe_vs_cur rel 0.0; the per-key reduction order is unchanged) and
matches SDPA at rel 3e-4..3e-3.

GATING: S>=4096 only. S<=2048 is mixed (several causal shapes regress 0.91-0.97)
so it stays 64x64. Excludes shapes already on a specific qregs path (qpkv6,
B=1 qregs128/qpkv8/16-causal), local, paged, varlen, qv, sparse, mask/score_mod,
and learnable_sink. Env kill-switch FLASH_ATTENTION_SM120_D256_WIDE=0.

CORRECTNESS: pytest d=256 seqlen 4096 (gqa) = 20 pass / 20 fail, IDENTICAL set
with wide ON vs OFF -> the change adds ZERO regressions. (The 20 failures are a
PRE-EXISTING learnable_sink+D256-large-seqlen bug present on the 64x64 path too;
flagged separately below, excluded from the wide path.) This is the largest
forward win of the campaign: it lifts most of the 54 D256 cells by +6-14%.

PRE-EXISTING BUG FLAGGED (not from this campaign): has_learnable_sink=True at
D256 seqlen 4096 fails the reference check on BOTH 64x64 and 128x64 paths. No
target model uses learnable sinks; needs a separate investigation.

## 2026-06-02 — WIN: qpkv4 D128 S1024 causal fwd tile 64x64 -> 64x96 (+5-6%)

Autonomous explorer found the qpkv4 D128 (Hq32/Hkv8; qwen3-vl & qwen3-embedding)
S1024 causal cell was a slight loss (0.97-0.98 vs FA2) on the current (64,64,1)
tile. Among 7 candidates, 64x96 is the robust winner over 3 runs (15 interleaved
blocks each): +5-6% vs current, flipping it to 1.02-1.04 vs FA2. Correctness vs
SDPA rel 2.3e-3 (PASS). Surgical: only the (128,4,1024,1) lookup cell; the
non-causal cell keeps 128x64.

## 2026-06-02 — FIX: learnable_sink on the SM80-base forward kernel (D256 + pack-GQA)

The pre-existing learnable_sink+D256 failure flagged earlier is now FIXED. Root
cause: it was NOT a numerical bug — flash_fwd.py (FlashAttentionForwardSm80, the
SM80-base forward used by SM120 for D256 and all pack-GQA/varlen/paged shapes)
HARD-ASSERTED `learnable_sink is None`. The shared softmax.finalize(sink_val=...)
already implements the sink denominator term; only the SM80-base kernel never
wired it. interface.py already routes sink shapes here (use_tma_sm120 requires
learnable_sink is None) and already passes learnable_sink_tensor to __call__ and
keys the compile cache on `learnable_sink is not None` — so the kernel body was
the only gap.

Fix (flash_fwd.py): removed the assertion; threaded learnable_sink through
kernel() and _paged_kv_mainloop(); added compute_sink_val() (mirrors SM90:
scalar per head for non-pack, per-row fragment via the QK identity-tensor
row->q_head map for pack_gqa); passed sink_val to softmax.finalize at all three
finalize sites (main, block-sparse, paged-KV). Fully guarded by
`learnable_sink is None` -> returns None -> byte-identical no-op for the common
(no-sink) path.

VALIDATION:
- Direct vs fp32 SDPA-with-sink reference: D128 & D256, S 512-4096, causal+nc,
  pack_gqa qpkv4 -> rel ~2-3e-3 (all PASS; previously raised AssertionError).
- pytest d=256 seqlen 4096 FULL set (mha/mqa/gqa x softcap{0,15} x sink{T,F} x
  deterministic): 120 passed / 0 failed (was 60 pass / 60 FAIL — the 60 fails
  were exactly the sink=True configs). Zero regression on the 60 non-sink cases.
This also enables learnable_sink on Ampere (SM80) via the same base kernel.

## 2026-06-02 — post-fix forward sweep + in-process truth

Clean corrected forward sweep (GPU0, no contention, warmup-ms soak harness)
AFTER the qpkv5/qpkv4/D256-wide commits:
  geomean 1.044, median 1.037, wins 53/78. D256 1.072, D128 0.983, gemma 1.071.
D256 wide win is visible (D256 S>=4096 rows now 1.04-1.24; nc up to 1.24).

BUT the subprocess sweep STILL under-rates FA4 (residual fa2-first ordering bias
survives the warmup fix). In-process interleaved verification of the D128
"laggards" shows almost all are artifacts:
  qwen3-vl  S1024 nc qpkv4 : sweep 0.813 -> in-proc 1.000
  qwen3-30b S8192 c  qpkv8 : sweep 0.917 -> in-proc 1.004
  qwen3-vl  S8192 c  qpkv4 : sweep 0.931 -> in-proc 0.999
  qwen3-emb S8192 c  qpkv4 : sweep 0.959 -> in-proc 1.000
So the sweep's D128 0.983 is bias; in-process D128 is parity-to-win and the true
overall forward is well above 1.044 (~1.07+).

GENUINE forward laggards (confirmed in-process, NOT artifacts):
  1. qwen3-14b qpkv5 S4096 causal D128 = 0.938 (non-pack -> TMA path; tile
     already 128x64-optimized, tile sweep found nothing better).
  2. gemma LOCAL (sliding-window 512) D256: e4b qpkv4 S4096 c=1 = 0.933,
     e2b qpkv8 S8192 c=1 = 0.954, e4b qpkv4 S8192 = 0.967, e2b qpkv8 S4096 nc
     = 0.967. Real, and local is gemma's PRIMARY attention mode -> high value.
Next: attack the gemma-local D256 path (current dispatch 64x16 qpkv4 / 64x32
qpkv8 for local).

## 2026-06-02 — WIN: gemma LOCAL D256 forward wide tile (128x{32,64}+Qregs+256t) at S>=4096

The genuine gemma-local forward laggards (0.93-0.97 vs FA2) are now fixed with
the same Q-in-regs trick as the dense D256 wide path, applied to the local
(sliding-window) dispatch. The narrow local path used 64x16 (qpkv4) / 64x32
(qpkv8) / 64x64 (qpkv2 fell through); 128x{32,64}+Qregs+256t is far faster:
tile_n=32 for window<=512, tile_n=64 for window~1024.

FA4-vs-FA4 interleaved + SDPA-window validated (sm120_local_wide_confirm.py):
  gemma4-e4b qpkv4 w512: S4096 c1 0.93->0.99 (+7%), S8192 c1 0.96->1.05 (+10%),
                          S4096 nc 0.93->0.99 (+6%)
  gemma4-e2b qpkv8 w512: S8192 c1 0.99->1.06 (+7%), S4096 nc 0.96->1.00 (+4%)
  gemma4-31b qpkv2 w1024: S4096 c1 1.00->1.12 (+11%), S8192 c1 1.01->1.14 (+13%)
new-default in-process re-verify (no env): e4b S4096 c1 0.995, e2b S8192 1.011,
e4b S8192 1.080, gemma31 S4096 1.125 — all at/above parity (were 0.93-0.97).
Output rel vs SDPA-window ~2e-3 (bit-equivalent retiling).

GATING: local D256, S>=4096 (square), qpkv in {1,2,4,8}, no paged/qv/varlen/
sparse/mask/score/sink. Required relaxing the `not local` clause in
sm120_q_in_regs (local + Q-in-regs is correct here). Kill-switch
FLASH_ATTENTION_SM120_LOCAL_D256_WIDE=0. S<4096 keeps the narrow 64x16/64x32
tile (unchanged).

CORRECTNESS: dedicated SM120 local test (test_flash_attn_sm120_local.py, S=256
small-S path) 49 passed / 0 failed (unchanged path intact). Wide-path shapes
SDPA-window rel ~2e-3. High value: local is gemma's PRIMARY attention mode.

## 2026-06-02 — FINAL forward standing after the wide-tile + gemma-local wins

Clean forward sweep (GPU0, no contention) after all this session's commits:
  geomean 1.053 (campaign start 1.015), median 1.041, wins 59/78.
  D128 1.004, D256 1.075, gemma 1.087 (was 1.071).
All gemma-local rows now >=1.0 (1.00-1.08); the gemma-local laggard family is
fixed. (In-process truth is higher than 1.053 — the subprocess sweep still has
residual fa2-first bias at S1024; e.g. the sweep's worst row gemma4-e4b S8192 c1
D256 qpkv4 nc reads 0.888 but is 1.085 in-process; qwen3.6-35b S1024 nc 0.821 is
~1.0 in-process.)

Session forward-perf commits: qpkv5-S4096 tile (+2.8%), D256 general wide tile
(+6-14%), qpkv4-S1024 tile (+5-6%), gemma-local wide tile (+3-13%), plus the
learnable_sink correctness fix. The D256-wide + local-wide Q-in-regs
generalization is the dominant lever (lifts the bulk of the 54 D256 cells).

ONE genuine forward laggard remains: qwen3-14b qpkv5 S4096 causal D128 = 0.938
(in-process, consistent). Non-pack -> TMA path, tile already 128x64-optimized
(tile sweep found nothing better); near its architectural floor on sm_120.
Not worth further dispatch effort.

## 2026-06-02 — WIN: extend D256 wide tile to S2048 non-causal (num_head>=16)

Followup to the D256 wide tile: at S2048 non-causal the wide tile (128x64+Qregs
+256t) also wins for the larger-head dense models, but the small Hq8 model and
S2048 causal are mixed, so gate S2048 to (not causal and num_head>=16).
New-default in-process (was ~1.03 at 64x64):
  qwen3.5-9b   Hq16 S2048 nc: 1.109 (+8%)
  qwen3.6-35b  Hq16 S2048 nc: 1.122 (+9%)
  qwen3.5-122b Hq32 S2048 nc: 1.134 (+9%)
Excluded (unchanged 64x64, verified): qwen3.5-0.8b Hq8 S2048 nc = 0.998 (median
of 12), S2048 causal (9b 0.995, 122b 1.029). Correctness vs SDPA rel ~1-3e-3.
One-line gate change; not in the standard bench (which uses 1024/4096/8192) but
real perf for S2048 training/inference.

## 2026-06-02 — D256 BACKWARD is parity in-process (NOT an occupancy-walled laggard)

In-process interleaved D256 backward matrix (sm120_bwd_d256_inproc.py, 28 cells,
backward-only timing): geomean 0.9961, wins 13/28. The larger shapes WIN:
  qwen3.5-27b qpkv6 S2048 nc 1.056, S1024 c 1.075
  qwen3.5-122b qpkv16 S2048 nc 1.064, S1024 nc 1.043
  qwen3.6-35b qpkv8 S1024 nc 1.018; gemma4-31b qpkv2 S2048 nc 1.021
Only the tiny S1024 gemma cells dip (gemma4-e2b qpkv8 S1024 c 0.875, e4b qpkv4
S1024 nc 0.904) — ~0.4ms multi-launch kernels, noise-dominated.

So the D256 backward "occupancy wall 0.93-0.98 laggard" (campaign brief's
"biggest latent prize") is NOT a real FA4-vs-FA2 deficit — it was the biased
harness + S1024 noise. The 1-CTA/SM occupancy wall is a real HARDWARE limit
(255 reg, no tcgen05) but FA2 hits the same wall, so the RATIO is ~parity. There
is no D256-backward laggard to chase and no rewrite is warranted (consistent with
[[sm120-d256-backward-occupancy-wall]] / [[sm120-backward-kernel-changes-rejected]]
on the kernel being unfixable — but the ratio was never actually a loss).

## 2026-06-02 — WIN: D256 wide tile for S2048 causal qpkv16 (num_head>=32)

Edge followup: at S2048 CAUSAL the wide tile helps ONLY the widest head count.
qwen3.5-122b (qpkv16 Hq32) +6.5% (wide/cur 1.065), new-default fa4/fa2 1.094;
qwen3.5-9b / qwen3.6-35b (Hq16) regress (0.96) so gated out via num_head>=32.
gemma-local S1024/S2048 was also probed: wide REGRESSES at S1024 (narrow tile
better) and is mixed/laggard at S2048 (e4b/e2b ~0.8 with either tile — tiny
0.15ms kernels where FA4 launch overhead dominates, not tile-fixable), so the
local gate stays S>=4096. Only the dense S2048-causal-qpkv16 cell is added.
Correctness vs SDPA rel ~1e-3.

## 2026-06-02 — Backward deep-profiling pass (register/occupancy + config sweep + ext. inspiration)

Goal: find backward wins (it's at parity in-process; wanted to push above). Result:
the SM120 backward is already at its sm_120 optimum — no dispatch win found, and
the kernel-level low-hanging fruit is already implemented.

PROFILING (cuFuncGetAttribute via dump_kernel_attributes, no root needed):
  D128 bwd main dKV kernel: 255 reg/thread, local=168B (SPILLING), 256 thr -> 1 CTA/SM
  D128 dQ kernel: 209 reg -> 1 CTA/SM
  D256 bwd main kernel: 255 reg, local=72B (spilling), 256 thr -> 1 CTA/SM
=> BOTH D128 and D256 backward are register-bound at the 255 cap -> 1 CTA/SM
   (16.7% occ). 2 CTA/SM needs <=128 reg/thread; the acc_dK/dV/dQ accumulators
   make that infeasible (intrinsic, no tcgen05 on sm_120). FA2 hits the same
   wall -> ratio is parity (confirmed: D256 bwd geomean 0.996 in-process).

DISPATCH CONFIG SWEEP (isolated subprocess per config, FA4-vs-FA4 interleaved):
  num_stages_Q=2 at S4096: SLOWER (0.90-0.92) -> current ns=1 for S<8192 is right
  num_stages_dO=2: SMEM OVERFLOW (crash) on both D128 and D256
  V_in_regs=1: SLOWER (0.88-0.98)
  num_stages_Q=2,dO=2: overflow. D256 any extra stage: overflow (smem-capped).
  => no dispatch lever helps; the config (ns=1, ns_Q=2 only for D128 S>=8192,
     M-split for underfilled grids) is already optimal.

EXTERNAL INSPIRATION (gau-nernst/learn-cuda, forward-only FA2 on sm_120, studied
by subagent): its v1->v5 ladder wins came from (1) XOR smem swizzle (+18pts),
(2) cp.async multi-stage pipelining, (3) ldmatrix.x4.trans, (4) in-place bf16
P/dS packing + smem aliasing to cut spills. Assessment vs our backward:
  (1) ALREADY DONE — get_smem_layout_atom swizzles all tiles (Q/K/V/dO/P/dS,
      swizzle_bits=3); the flash_bwd.py "TODO swizzle=3?" is already satisfied.
  (2) ALREADY DONE/tuned — cp.async staging IS num_stages; more stages overflow
      smem or slow it (see sweep above).
  (3) likely already via CuTeDSL copy atoms.
  (4) the ONLY untried lever with potential upside: reduce the local-memory
      SPILLS (local=168/72) by in-place bf16 packing of P/dS. But: deep CuTeDSL
      surgery on Tri Dao's FA2 backward, can't confirm the spill is on the hot
      path WITHOUT ncu (root-gated here, ERR_NVGPUCTRPERM), high regression risk
      on a parity kernel. NOT attempted blind; recommend only with ncu access.

CONCLUSION: backward (D128 + D256) is at parity and well-optimized; the register
wall is intrinsic to sm_120. No safe backward win remains at the dispatch level.
The forward Q-in-regs wide-tile lever has no backward analogue (the wall is
registers, not smem). See [[sm120-d256-backward-occupancy-wall]].

## 2026-06-02 — Backward ncu profiling (sudo) — DEFINITIVE: at sm_120 floor, spill-surgery not worth it

ncu (sudo, hardware counters) on the backward main kernels:

D128 qpkv8 S4096 causal — COMPUTE-BOUND (spill-fix would NOT help):
  dominant kernel (209 reg): 2.25 ms, Compute(SM) 73.8%, warp-cyc/issued 13.2, occ 16.6%
  dKV kernel (255 reg, local=168B spill): 855 us, Compute(SM) 72.1%, warp-cyc/issued 5.24
  -> both compute(MMA)-bound; FA2 does the same MMAs -> parity is intrinsic.

D256 qpkv16 S2048 causal — LATENCY/OCCUPANCY-BOUND:
  dominant dKV kernel (255 reg): 1.82 ms, Compute(SM) 39%, Mem 56%, warp-cyc/issued 24.9,
  occ 16.6%. Top stall = Long Scoreboard 30.4%. Occupancy section "Est. Local Speedup
  83.3%" = the 1-CTA/SM wall (need ~6x occ; 255->~42 reg, infeasible).
  Memory traffic on this kernel:
    global_op_ld = 208.4M sectors (91%)  <- Q/K/V/dO reloads (structural FA2 pattern)
    local_op_ld  =  19.5M sectors (8.5%) <- register SPILL reads
    local_op_st  =   1.2M sectors (0.5%)
  => the spill is only ~8.5% of load traffic; the stall is dominated by GLOBAL
     reloads, which FA2 also pays. Eliminating the spill (in-place bf16 P/dS
     packing surgery) removes ~8.5% of traffic -> ~2% kernel speedup AT BEST, on
     a parity kernel, for high-risk CuTeDSL surgery. NOT worth it (ncu-confirmed).

CONCLUSION (hardware-backed): the SM120 backward is at its architectural floor.
- D128 backward: compute(MMA-throughput)-bound -> can't beat FA2's identical MMAs.
- D256 backward: global-reload latency + 1-CTA/SM occupancy wall (255 reg from
  acc_dK/dV/dQ); FA2 has the same algorithm -> parity. smem-capped, no room for
  more staging; no TMA for D256 bwd on sm_120.
The ONLY theoretical path to break the D256 occupancy wall is a kernel REDESIGN
splitting dK-only and dV-only into separate kernels (each holds ~half the
accumulators -> maybe <=128 reg -> 2 CTA/SM), trading 2x S=QK^T recompute +
extra Q/K reloads for 2x occupancy. Speculative (could net zero since it adds
the very global-reload traffic that dominates), multi-day, high regression risk
on a parity kernel. Not attempted without explicit go-ahead on that specific gamble.

## 2026-06-02 — Backward redesign DE-RISK (opus design agent + ncu) — predicts NET LOSS

Per user go-ahead to attempt the dK/dV-split redesign, de-risked BEFORE building:

OPUS REGISTER-BUDGET ANALYSIS (flash_bwd.py): acc_dK + acc_dV = exactly 128
reg/thread ((64,256) fp32 each = 64). Peak adds transient acc_dQ ((m=64,hd=256)
=64, reduced to gmem per m-tile) -> ~192 reg in accumulators alone -> 255 total.
- dK/dV KERNEL SPLIT (the requested approach): dK-only ~137, dV-only ~113 ->
  dK-only does NOT reach <=128, AND doubles the global Q/K/V/dO traffic that is
  already 91% of load cost. INFEASIBLE.
- The ONLY path to <=128 is head_dim-splitting acc_dK/dV (quarter-split ~105 reg)
  PLUS extracting dQ to its own pass PLUS dropping the dead acc_S_pre. But the
  quarter head_dim-split recomputes S=QK^T and dP=dO@V^T (the dominant GEMMs) 4x.

EMPIRICAL PROXY (n_block=32, halves acc_dK/dV): ncu shows the dKV kernel STAYS
at 255 reg / 1 CTA/SM (16.64% occ) anyway — because acc_dQ (m_block x head_dim,
independent of n_block) + working tiles still dominate — and it is 16.5% SLOWER
(0.79->0.92 ms, D256 MHA S2048 c) from the doubled n-block count. So even a 2x
overhead with NO occupancy gain already regresses +16%; the head_dim quarter-
split's 4x GEMM recompute would be worse, and would likely push the (currently
39%-compute, latency-bound) kernel compute-bound.

VERDICT: the backward occupancy wall is intrinsic. The dK/dV split is infeasible;
the head_dim quarter-split is the only path to 2 CTA/SM but the recompute cost is
predicted (by the n=32 proxy + 4x-GEMM accounting) to EXCEED the occupancy gain
-> net loss. De-risk says do NOT build the multi-day redesign. Reported to user
for a final call on whether to build the head_dim quarter-split prototype anyway.

## 2026-06-02 — REJECTED: L2-residency bwd scheduler (SingleTileLPTBwdScheduler) on SM120

Pursued the "more efficient front and back" redirect. An opus brainstorm flagged
the register-free L2-residency backward swizzle scheduler (already implemented +
SM90-wired, gated behind `deterministic`, unused on the SM80-base SM120 path) as
the highest-leverage, lowest-risk lever for the D256 backward's 91%-global-reload
bottleneck. Wired it into the SM120 backward (use_lpt_bwd flag, compile-keyed,
seqlen_k/element_size/lpt args) behind FLASH_ATTENTION_SM120_BWD_LPT and A/B'd:

  D256 qpkv16 S2048 c : lpt/off 1.487  (with spt reversal) -> 1.060 (swizzle only)
  D256 qpkv16 S4096 c : 1.825
  D256 qpkv8  S4096 c : 1.463 -> 1.037 (swizzle only)
  D256 qpkv8  S4096 nc: 0.997 (neutral)
  D256 qpkv4  S8192 c : 1.838
  D128 qpkv8  S4096 c : 1.947 ;  D128 qpkv4 S8192 c : 2.547
  grads correct (rel ~1e-3) — pure scheduling regression, not a correctness issue.

WHY IT LOSES: the DEFAULT SM120 backward grid (n-block fastest within head) already
runs consecutive n-block CTAs of the SAME head concurrently, and all of a head's
CTAs reload the same Q/dO — so the L2 already caches the reload working set. The
LPT head-grouping swizzle DISRUPTS this natural locality (and the spt block-reversal,
gated on causal, is catastrophic: up to 2.5x). Even the pure swizzle w/o reversal
is 4-6% slower. REVERTED in full.

CONCLUSION (now also empirically, not just by analysis): the SM120 backward is at
its floor. The register-free scheduling lever — the last plausible win — loses
because the default scheduler already exploits the available L2 locality. Combined
with the ncu profiling (D128 compute-bound; D256 occupancy-walled, spill only 8.5%)
and the occupancy-redesign de-risk (dK/dV split infeasible, head_dim-split net-loss),
there is no backward win available on sm_120. Forward remains the productive surface
(1.053 geomean, Q-in-regs wide tile). See [[sm120-d256-backward-occupancy-wall]].

## 2026-06-02 — WIN: extend D256 wide tile to VARLEN forward (+7-11%, packed-seq)

Autonomous forward win-hunt: dense-forward dispatch (tiles/stages/threads) is
exhausted (a stage/thread explorer found no >3% wins; qpkv5 S4096 causal is
MMA-issue-efficiency-bound per ncu: FA4 62% SM vs FA2 70%, not dispatch-fixable).
But the VARLEN forward path was needlessly excluded from the D256 wide tile
(cu_seqlens gate). varlen D256 forward (packed-sequence training, ubiquitous)
ran at 64x64; the 128x64+Qregs+256t wide tile helps it too:
  RTX6000 A/B (packed [4096,2048,1024,1024]):
    qwen3.5-122b qpkv16 c: wide/cur 1.105 (1.041->1.150 vs FA2)
    qwen3.5-9b   qpkv4  c: 1.083 (1.038->1.124)
    qwen3.6-35b  qpkv8  c: 1.075 (1.039->1.116)
  new-default vs SDPA-varlen: 122b 1.203, 35b nc 1.199, rel 1-5e-3 (PASS).
Fix: relaxed the cu_seqlens exclusion in sm120_d256_wide (kept seqused excluded —
untested). Same SM80-base kernel, bit-identical retiling. Validated:
test_flash_attn_varlen.py d=256 = 1296 passed / 0 failed.

Also benchmarked the rest of varlen forward (D128 qpkv4/8, D256 qpkv4/8/16): all
parity-to-win (1.01-1.10) on the existing tiles -> no other varlen-fwd gap.

## 2026-06-02 — WIN: extend gemma-LOCAL D256 wide tile to varlen forward

Same as the dense varlen extension, for the sliding-window (local) D256 path
(gemma packed-sequence training). Relaxed the cu_seqlens exclusion in
sm120_local_d256_wide. RTX6000 A/B (packed [4096,2048,1024,1024], window):
  gemma4-31b qpkv2 w1024: wide/cur 1.120 (0.978->1.095 vs FA2)
  gemma4-e4b  qpkv4 w512: 1.068 (0.885->0.946)
  gemma4-e2b  qpkv8 w512: 1.005 (marginal)
new-default vs SDPA-windowed-varlen: 31b 1.023, e4b 0.940 (improved from 0.885),
rel ~2.5e-3 (PASS). Net positive across all gemma-local varlen shapes. seqused
stays on the narrow path. (varlen test doesn't parametrize local -> validated via
SDPA-windowed-varlen on the exact gemma shapes; local-wide and varlen-wide are
each independently pytest-validated.)

## 2026-06-02 — BIG WIN: SplitKV (FlashDecoding) for SM120 forward — decode 2-4x

Investigated decode/SplitKV (seqlen_q=1, large KV). Found SM120 hard-disabled
SplitKV (interface.py `if arch//10==12 and num_splits>1: num_splits=1`), so
decode launched only ~batch*num_head_kv CTAs (B=1 -> ~8 CTAs on 188 SMs), each
streaming the whole KV cache: FA4 was 0.10-0.21x of FA2 at B=1 Sk>=16384.

Implemented SplitKV on the SM80-base forward (opus agent in a worktree, reviewed
+ re-validated by me): the split scheduler (SingleTileScheduler), split n-block
range (BlockInfo), combine kernel, partial buffers, and per-split
softmax.finalize ALL already existed and are arch-neutral — the kernel just
hard-coded split off. Changes (all const_expr(self.is_split_kv)-gated, so the
non-split/training path compiles byte-identical):
- flash_fwd.py (~194L): is_split_kv/num_splits ctor; split-aware O/LSE 5D/4D
  layout transpose (mirrors SM100); split_idx from work tile; BlockInfo split
  range; empty-split guard (has_work -> O=0/LSE=-inf, combine drops them);
  direct fp32 reg->gmem partial-O write (bypasses the bf16 smem O buffer); 3
  epilogue call sites. NO softmax math change.
- flash_fwd_combine.py (~15L): gate griddepcontrol.wait (PDL) to arch>=90 (it's
  illegal on the sm_80-compiled SM120 target; combine never ran on SM120 before
  since split was disabled).
- interface.py: remove the disable + assert; force non-TMA for split; pass
  is_split_kv/num_splits to the ctor; num_splits in compile_key (SM120-split
  only); DECODE AUTO-TRIGGER: for seqlen_q<=8 + non-varlen/paged/MLA, request
  num_splits=0 BEFORE the pack_gqa disable (the GQA+SplitKV combo is unsupported,
  so pack_gqa must be off; ordering matters). The num_splits heuristic
  self-protects (returns 1 for a filled grid, e.g. large batch), so prefill/
  training (seqlen_q>8) is never touched.

VALIDATION:
- Correctness vs SDPA (bottom-right causal for seqlen_q<seqlen_k): decode B in
  {1,4,64}, Sk in {4096..32768}, D128/256, qpkv 4/8/16, seqlen_q 1/2/4/8 -> rel
  2e-3..8e-3 (PASS). Explicit num_splits 2..128 all correct. (NOTE: the earlier
  "no-split rel 0.85" was a transient buggy auto-elif (set num_splits after the
  pack_gqa disable), since removed — the true no-split path is correct.)
- Speed: decode now 0.32-0.65x of FA2 (was 0.10-0.21x) -> ~2-4x faster. Split vs
  no-split ~3.2-3.7x (B1 Sk32768 D128: 0.99->0.27ms). FA4 still slower than FA2
  absolute (residual = SM80-base per-CTA decode inefficiency: no TMA, tile_m
  wasted on few query rows -> needs a Blackwell-native decode kernel, separate).
- No regression: training (seqlen_q>=1024) early-trigger does NOT fire, wide-tile
  wins intact (122b 1.139, vl 1.016); pytest test_flash_attn d256 seqlen4096
  120 passed / 0 failed.

GATED OUT (untested, left on safe path): varlen+split, paged+split, seqused,
MLA(qv). The structural code paths exist but are not validated — don't rely yet.

## 2026-06-03 — WIN: D128 decode tile 128x64 -> 16x64/1-warp (+50-68% on top of SplitKV)

ncu on the SplitKV decode kernel (B1 Sk32768 D128 qpkv8) showed it COMPUTE-bound
(SM 81.7%, DRAM only 19.6%) — decode should be memory-bound. Cause: the default
128x64 tile runs the MMA on ~120 empty query rows (seqlen_q=1, ~8 packed real
rows -> 6% M-utilization). A tiny 16x64 / 32-thread (1-warp) tile cuts the wasted
MMA. Decode dispatch now picks 16x64/32t for D128 seqlen_q<=8:
  D128 q8 B1 Sk32768: 0.406 -> 0.646 vs FA2 (+59%)
  D128 q4 B1 Sk16384: 0.535 -> 0.816 (+53%)
Cumulative decode (orig -> SplitKV -> +tile): D128 0.10-0.21 -> 0.40-0.54 ->
0.65-0.82x of FA2 (~4-6x faster than the original disabled-SplitKV baseline).
D256 decode does NOT benefit from the small tile (kept on the lookup path,
still ~0.38-0.76x) -> needs the kernel rewrite. Training (seqlen_q>8) unaffected.
Correctness vs SDPA rel 5-8e-3 (PASS).

REMAINING: tile-tuning plateaus (~0.6-0.8x D128) because the SM80-base kernel's
MMA is structurally tied to tile_m*warps. The memory floor for this decode is
~45us vs FA2 99us, so a memory-bound (GEMV-style, no wasted MMA) sm_120 decode
kernel could match/beat FA2. That's the kernel-rewrite effort (next).

## 2026-06-03 — Custom sm_120 GEMV decode kernel (FlashAttentionDecodeSm120, gated)

Built a from-scratch memory-bound decode kernel (opus agent + my re-validation).
flash_fwd_decode_sm120.py (~327L): one CTA per (split, kv_head, batch) processes
all qhead_per_kvhead query rows together (KV read ONCE, no GQA redundancy ->
grid 1504->188), Q.K^T and P.V as GEMV (FMA + warp shuffles, NO m16n8k16 MMA on
empty query rows), cp.async double-buffered K/V streaming, online softmax with
cross-thread-group smem reduction, writing fp32 partial O/LSE for the existing
combine. Gated behind FLASH_ATTENTION_SM120_DECODE_KERNEL (default OFF); dispatch
is additive + early-returns through the combine (flag off = byte-identical).

RE-VALIDATED (my env, RTX6000, seqlen_q=1, causal=False, vs SDPA + baseline + FA2):
  D256 q4  B4  Sk16384: new/base 1.19x, new/fa2 0.948 (nearly FA2!)
  D256 q16 B16 Sk32768: new/base 1.31x
  D256 q16 B1  Sk32768: new/base 0.96 (~tie)
  D128 q8/q4: new/base 1.00-1.07 (tie — D128 already tile-fixed memory-leaning)
  correctness rel 9e-4..5.9e-3 (PASS; GEMV accumulates fp32 -> often 10-100x
  tighter than baseline).
ncu proof: D256 Sk32768 qpkv8 main kernel baseline 132us/SM61%/DRAM30% (compute-
bound) -> new 90us/SM18%/DRAM44% (memory-leaning). The 7.7x early fix was
partitioning keys across thread-groups (a naive every-thread-all-keys version
was 369us/DRAM5%).

VERDICT: real D256 decode win (1.2-1.3x over baseline; one shape ~0.95x FA2),
tie on D128, does NOT beat FA2 overall. Same wall as everywhere on sm_120: still
register-bound (254 reg -> 1 CTA/SM, DRAM 44% not the ~1.5TB/s floor) — needs
register reduction / warp-specialization for >=2 CTA/SM to close the rest.
qpkv16 may fall back (reduction smem). Gated/causal=False/seqlen_q=1 only.
Merged gated+off as an opt-in D256-decode improvement and a foundation; lab
notes in agent_space/DECODE_KERNEL_NOTES.md. See [[sm120-d256-backward-occupancy-wall]].

## 2026-06-03 — Decode occupancy-crack: VALIDATED NEGATIVE (it's access-bound, not warp-starved)

Attacked the decode kernel's 1-CTA/SM (smem-limited) wall to reach 2 CTA/SM.
Result: the hypothesis was WRONG. The kernel is NOT concurrency-limited; it is
memory-ACCESS-inefficiency-bound. Every occupancy lever failed (authoritative
single-process CUDA-event timing, qpkv8 D256 Sk32768, decode-only):
  baseline 128t/NS=2/partition/tile_n32 (65.5KB): 318us  <- already fastest
  NS=1/allkeys/tile_n32 (32KB, fits 2-3 CTA/SM): 1225us (3.9x SLOWER)
  256t/NS=1/allkeys: 1343us; tile_n 16/64 variants: 1228-1373us
  num_splits 23->47->94->188 (grid 184->1504): 318->317->324->354 (more CTAs = no help/worse)
Reducing smem to allow 2 CTA/SM forces the "allkeys" GEMV (no row-partition),
whose redundant on-chip FMA + smem re-reads dominate -> 4x worse. More
CTAs/warps don't help -> NOT warp-starved.

ncu SOL (baseline default config, which ncu profiles correctly): DRAM 46%,
Compute 18%, L2 17% — NOTHING saturated; top stall MIO scoreboard; "only 4 of 32
bytes per sector utilized" -> the per-thread GEMV K/V loads are bandwidth-
INEFFICIENT (uncoalesced effective access). Kernel moves 268MB at ~843 GB/s vs
FA2's ~1.14 TB/s. Plus R-fold redundancy at high qpkv.

VERDICT: 2 CTA/SM is the wrong target for decode. To beat FA2 needs a MEMORY-
ACCESS redesign (coalesced K/V streaming, fewer redundant smem reads), not
occupancy — a substantial rewrite. The shipping gated decode kernel (318us, D256
1.2-1.3x over baseline, ~0.6-0.95x FA2) is already its best config.

METHODOLOGY CAVEAT (important for future ncu work): `sudo ncu` serves a STALE
kernel from the root-owned /tmp/root/cutlass_python_cache and is INSENSITIVE to
env-var kernel configs (always profiles the default). Trust single-process
CUDA-event timing (config-responsive) for A/B; use sudo-ncu only for the
default-config occupancy/SOL snapshot. (My earlier decode ncu numbers were the
default config, so directionally valid, but cross-config ncu A/B is unreliable.)
