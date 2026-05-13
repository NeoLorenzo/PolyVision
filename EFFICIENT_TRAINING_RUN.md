# Efficient PPO Training Run Guide

This guide explains how to run a full PolyVision PPO training run with maximum practical SPS while still preserving the metrics needed for analysis.

## Core Principle

Separate **training**, **profiling**, and **debugging**.

| Mode | Purpose | Diagnostics | W&B | Timesteps |
|---|---|---:|---:|---:|
| Full training | Learn policy efficiently | Off | Offline | Millions |
| Debug run | Inspect exact behavior | On | Optional | 5k–20k |
| SPS profiling | Find bottlenecks | Minimal profiler only | Disabled/offline | 10k–20k |

Do not run full training with step diagnostics unless actively debugging. Step diagnostics are useful, but they add env-side payload cost and are not required for normal W&B graphs.

---

## Recommended Full Training Command

Use this for serious high-throughput training:

```powershell
$env:POLYVISION_LEVEL_POOL_GLOB='levels/phase1_pool_bardur_solo/*.csv'
$env:POLYVISION_LEVEL_SELECTION_MODE='round_robin'
$env:POLYVISION_INFO_MODE='fast'
$env:POLYVISION_STRICT_COORD_ASSERT='0'
$env:POLYVISION_SOLO_NO_OPPONENT_MODE='1'

# Log locally during training, then sync later.
$env:WANDB_MODE='offline'
$env:WANDB_CONSOLE='off'
$env:WANDB_SILENT='true'

python C:\PolyVision\py_rl\cleanrl\cleanrl\ppo.py `
  --env-id Tribes-v0 `
  --actor-mode legal_features `
  --max-legal-actions 1024 `
  --num-envs 20 `
  --total-timesteps 5000000 `
  --track `
  --save-model `
  --save-frequency 500000 `
  --no-enable-step-diagnostics `
  --no-validate-action-interface
```

After the run finishes, sync W&B manually:

```powershell
wandb sync wandb/offline-run-*
```

This keeps W&B graphs while avoiding live network overhead during training.

---

## Why Step Diagnostics Should Usually Be Off

Step diagnostics are for detailed debugging, not normal training.

They help answer questions like:

- Did the agent ignore a legal capture?
- Did it end turn while a useful move existed?
- Which legal actions were available?
- Was an illegal/fallback action triggered?
- Did it miss a city-upgrade completion?

They are not needed for standard training graphs such as:

- SPS
- episodic return
- final SPT
- city count
- tech research rate
- harvest counts
- fog cleared
- loss metrics
- entropy / KL

If a metric is important for every training run, it should be logged as a cheap scalar episode metric, not hidden behind expensive step diagnostics.

---

## Debug Run Command

Use this only when inspecting behavior in detail:

```powershell
$env:POLYVISION_LEVEL_POOL_GLOB='levels/phase1_pool_bardur_solo/*.csv'
$env:POLYVISION_LEVEL_SELECTION_MODE='round_robin'
$env:POLYVISION_INFO_MODE='fast'
$env:POLYVISION_STRICT_COORD_ASSERT='0'
$env:POLYVISION_SOLO_NO_OPPONENT_MODE='1'
$env:WANDB_MODE='offline'

python C:\PolyVision\py_rl\cleanrl\cleanrl\ppo.py `
  --env-id Tribes-v0 `
  --actor-mode legal_features `
  --max-legal-actions 1024 `
  --num-envs 20 `
  --total-timesteps 10000 `
  --track `
  --enable-step-diagnostics
```

Use this to inspect behavior, not to train for millions of timesteps.

---

## SPS Profiling Command

Use this when investigating throughput bottlenecks:

```powershell
$env:POLYVISION_LEVEL_POOL_GLOB='levels/phase1_pool_bardur_solo/*.csv'
$env:POLYVISION_LEVEL_SELECTION_MODE='round_robin'
$env:POLYVISION_INFO_MODE='fast'
$env:POLYVISION_STRICT_COORD_ASSERT='0'
$env:POLYVISION_SOLO_NO_OPPONENT_MODE='1'
$env:WANDB_MODE='offline'
$env:POLYVISION_PROFILE_SPS='1'

python C:\PolyVision\py_rl\cleanrl\cleanrl\ppo.py `
  --env-id Tribes-v0 `
  --actor-mode legal_features `
  --max-legal-actions 1024 `
  --num-envs 20 `
  --total-timesteps 10240 `
  --track `
  --no-enable-step-diagnostics `
  --no-validate-action-interface
```

Profiling runs should be short. The goal is to identify bottlenecks, not train a good model.

---

## Current Bottleneck Diagnosis

Recent benchmarks showed that SPS is mostly capped by env-side Python step cost, not neural network inference and not primarily Java.

Approximate bottleneck ranking:

1. `trainer/envs_step_wall`: ~70% of wall-clock time
2. `env/info_dict_build`: ~22–23% wall-clock estimated
3. `env/legal_action_feature_build`: ~21–22% wall-clock estimated
4. `trainer/ppo_update`: ~19% wall-clock
5. `env/reward_calculation`: ~10% wall-clock estimated

W&B online also reduced throughput. Prefer `WANDB_MODE=offline` and sync after training.

---

## Efficient Training Checklist

Before starting a full run:

1. Confirm the environment is using the solo Bardur map pool.
2. Use `WANDB_MODE=offline`, not `online`.
3. Keep `--track` enabled so metrics are still recorded.
4. Disable step diagnostics.
5. Disable validation/interface checks unless actively testing them.
6. Use `actor-mode legal_features` if strategic action features are needed.
7. Use `num-envs 20` unless profiling shows another count is better.
8. Save checkpoints periodically.
9. Sync W&B only after the run completes.

---

## When to Use Each Mode

### Use full training mode when:

- running 500k–5M+ timesteps;
- comparing actual model performance;
- tracking final SPT, city count, tech rates, and reward curves;
- trying to improve average policy quality.

### Use debug mode when:

- investigating illegal actions;
- checking whether the policy ignores captures or level-ups;
- inspecting exact legal action sets;
- validating reward/action-mask behavior;
- analyzing one failed episode in detail.

### Use SPS profiling mode when:

- checking why SPS is low;
- comparing W&B online vs offline;
- comparing `legal_features` vs `legal_only`;
- testing whether info construction or action-feature construction is the bottleneck.

---

## Do Not Do This for Full Training

Avoid this in long runs:

```powershell
--enable-step-diagnostics
$env:WANDB_MODE='online'
```

This combination is useful for visibility, but inefficient for sustained training.

---

## Key Rule

Use expensive diagnostics only when answering a specific debugging question. For normal training, log cheap scalar metrics and keep the environment step path as light as possible.
