# SM120 Current Performance Snapshot - 2026-05-26

This records the current `sm120-integrate` state after restoring the SM120
packed GQA forward path.

## Commits

- Current winner marker: `d2d0ec2 bench: mark SM120 forward current winner`
- Kernel change it marks: `1b7db10 cute: restore SM120 packed GQA forward path`
- Previous non-packed fallback baseline: `c43a7b4 cute: keep SM120 GQA forward on nonpacked path`

## Environment

- GPU: RTX 5090, SM120, UUID `GPU-f423fa54-c41a-719d-89ca-e09ae9c1826a`
- Torch: `2.11.0+cu130`
- FA2 baseline: cached wheel `flash_attn-2.8.3+torch2.11.0.cu130-cp312-cp312-linux_x86_64.whl`
- FA4: editable `flash-attn-4` from this worktree
- Harness: `/home/rgilbreth/Desktop/AI-Software/sm120-overnight-logs/phase13_bench/bench_runner.py`
- Matrix: 5 presets x 4 seqlens x 2 causal modes x forward/backward = 80 paired cells
- Timing: fresh subprocess per cell, B=2, bf16, CUDA events, 3 warmup + 10 measured iterations

## FA4 vs FA2 2.8.3

Output directory: `/tmp/sm120_fa2_fa4_d2d0ec2_20260526_0800`

| Slice | Cells | FA4 / FA2 geomean |
|---|---:|---:|
| Forward | 40 | 1.0248x |
| Backward | 40 | 1.0204x |
| Overall | 80 | 1.0226x |

Preset geomeans, forward and backward combined:

| Preset | FA4 / FA2 geomean |
|---|---:|
| `llama2-7b` | 0.9945x |
| `llama3-8b` | 1.0344x |
| `mistral-7b` | 1.0230x |
| `mixtral-8x7b` | 1.0083x |
| `qwen2.5-7b` | 1.0537x |

## Change vs Prior Non-Packed Fallback Matrix

Prior output directory: `/tmp/sm120_fa2_fa4_current_20260526_0218`

| Slice | Prior FA4 / FA2 | Current FA4 / FA2 | Ratio change |
|---|---:|---:|---:|
| Forward | 0.9594x | 1.0248x | 1.0681x |
| Backward | 1.0233x | 1.0204x | 0.9972x |
| Overall | 0.9908x | 1.0226x | 1.0320x |

## Largest Current Wins

| Direction | Preset | S | Causal | FA2 TFLOPS | FA4 TFLOPS | FA4 / FA2 |
|---|---|---:|---:|---:|---:|---:|
| bwd | `qwen2.5-7b` | 1024 | 0 | 91.3 | 115.0 | 1.2592x |
| fwd | `qwen2.5-7b` | 2048 | 1 | 122.2 | 147.5 | 1.2071x |
| fwd | `qwen2.5-7b` | 1024 | 1 | 82.2 | 96.5 | 1.1749x |
| bwd | `llama3-8b` | 2048 | 1 | 114.4 | 131.6 | 1.1501x |
| fwd | `mixtral-8x7b` | 1024 | 1 | 87.0 | 99.7 | 1.1463x |
| bwd | `qwen2.5-7b` | 2048 | 0 | 143.7 | 163.0 | 1.1346x |

## Largest Current Regressions

| Direction | Preset | S | Causal | FA2 TFLOPS | FA4 TFLOPS | FA4 / FA2 |
|---|---|---:|---:|---:|---:|---:|
| bwd | `mixtral-8x7b` | 8192 | 1 | 180.6 | 159.4 | 0.8827x |
| fwd | `qwen2.5-7b` | 4096 | 0 | 185.8 | 167.0 | 0.8986x |
| bwd | `mistral-7b` | 2048 | 1 | 131.5 | 118.8 | 0.9036x |
| fwd | `llama2-7b` | 8192 | 0 | 192.9 | 177.2 | 0.9187x |
| bwd | `llama3-8b` | 1024 | 0 | 121.1 | 112.1 | 0.9254x |
| bwd | `mixtral-8x7b` | 1024 | 1 | 91.6 | 85.1 | 0.9284x |

## Validation Notes

- All 80 paired cells completed with `ok=True`.
- FA2 and FA4 import together by using FA2 from site-packages and FA4 `flash_attn.cute` from the editable worktree.
- CodeRabbit's earlier SM120 backward tuning harness needle concern is fixed at this state:
  `measure_one_bwd.py --help` imports successfully and applies the monkey patch.
- Nsight Compute counters are still blocked for non-root by `ERR_NVGPUCTRPERM`
  until `NVreg_RestrictProfilingToAdminUsers=0` is set, or `ncu` is run under
  `sudo` with isolated writable cache directories.
