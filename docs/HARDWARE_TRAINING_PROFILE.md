# Hardware Training Profile

This document records the local training machine profile for PolyVision experiments and explains expected impact on training speed and configuration choices.

## Machine Summary (Sanitized)

- Report timestamp: 2026-05-04 12:14:31
- OS: Windows 11 Pro (10.0.22621)
- CPU: Intel Core i9-13900HX (13th Gen)
- RAM: 32 GB
- Storage: SSD (1.9 TB)
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- GPU VRAM: 8 GB GDDR6 (8188 MB dedicated)
- CUDA cores: 4608
- Max graphics power: 140 W
- Memory bandwidth: 256.032 GB/s
- Driver: NVIDIA Game Ready 560.94 (2024-08-20)

Notes:
- Sensitive identifiers (device ID, part number, IRQ, exact BIOS IDs) are intentionally omitted.
- Display details are excluded because they do not materially affect RL training throughput.

## Why This Matters For RL Training

- GPU memory (8 GB VRAM) is the main limit for large batch sizes and larger policy/value networks.
- Strong CPU (i9-13900HX) is useful for environment stepping and data preprocessing.
- Fast SSD helps with checkpointing/logging cadence and evaluation artifact storage.
- 32 GB system RAM is generally sufficient for concurrent training + logging + evaluation.

## Practical Impact On Current PolyVision Stack

Given current architecture (`Tribes` via Py4J bridge), environment stepping overhead is significant. This means:

- Increasing `num_envs` may improve throughput at first, then flatten due to Java/Py4J overhead.
- Very large `num_envs` can increase JVM/process overhead faster than policy update efficiency.
- Stable gains usually come from balanced rollout sizing (`num_envs` x `num_steps`) rather than aggressively increasing only one.

## Async Benchmark Results (2026-05-04)

Benchmark conditions:

- Vectorization: `AsyncVectorEnv(context="spawn")`
- Startup jitter: `0.1s` to `2.0s`
- Script: `py_rl/cleanrl/cleanrl/benchmark_async_vector_envs.py`
- Compared: `num_envs` in `[12, 16, 20]`

Measured throughput:

| num_envs | tail_avg_sps | all_avg_sps | peak_sps | duration_s |
|---|---:|---:|---:|---:|
| 12 | 1561 | 1377 | 1596 | 27.19 |
| 16 | 1172 | 1097 | 1194 | 34.43 |
| 20 | 1058 | 937 | 1117 | 38.11 |

Observed stability during benchmark:

- No memory warnings observed.
- No worker process crashes observed.
- No excessive JVM boot failures observed.

## Recommended Configuration (PPO, Phase 1)

Based on measured SPS and stability on this machine:

- `num_envs`: **12** (recommended default)
- `num_steps`: 128
- `total_timesteps`: start with 100k to 500k for profiling, then scale
- `learning_rate`: keep baseline unless instability appears
- `capture_video`: off during training runs (enable only for selected eval runs)

Rationale:

- `num_envs=12` delivered the best sustained and peak throughput in local testing.
- Higher counts (`16`, `20`) reduced throughput due JVM/process overhead in current architecture.

## Monitoring Checklist

During longer runs, monitor:

- GPU memory usage and OOM events
- CPU utilization and per-process JVM count
- Steps/sec (or samples/sec) over time
- Episode return stability and variance
- Checkpoint save frequency vs I/O overhead

## Reproducibility Notes

- Pin seeds for comparable runs.
- Record driver version and key PPO args for each benchmark.
- If driver or CUDA stack changes, rerun a short baseline comparison before trusting historical speed/perf numbers.
