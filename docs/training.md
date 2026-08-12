# Training

The primary trainer is `py_rl/cleanrl/cleanrl/ppo.py`. It is based on CleanRL PPO but contains PolyVision-specific actor paths, asynchronous JVM orchestration, validation, telemetry, and checkpoint metadata.

## Representative run

From the repository root in an activated Python environment:

```powershell
$env:POLYVISION_LEVEL_POOL_GLOB = 'levels/phase1_pool_bardur_real/*.csv'
$env:POLYVISION_SOLO_NO_OPPONENT_MODE = '1'
$env:POLYVISION_INFO_MODE = 'fast'
python py_rl/cleanrl/cleanrl/ppo.py `
    --actor-mode legal_features `
    --total-timesteps 500000 `
    --num-envs 12 `
    --num-steps 128 `
    --save-model `
    --save-frequency 100000
```

The trainer defaults to `legal_only`, 500,000 timesteps, 12 environments, 128 rollout steps, 256 legal slots, and strict 10,000-state preflight validation. `legal_features` uses the current 42-feature representation. `dense_debug` is available for equivalence/debug work but materializes logits over the full global action space.

## Important arguments

| Argument | Current default | Meaning |
|---|---:|---|
| `--actor-mode` | `legal_only` | `legal_only`, `legal_features`, or `dense_debug`. |
| `--total-timesteps` | 500,000 | Scheduled environment transitions. |
| `--num-envs` | 12 | Parallel spawned processes/JVMs. |
| `--num-steps` | 128 | Rollout length per environment. |
| `--max-legal-actions` | 256 | Fixed legal-slot tensor capacity. |
| `--legal-action-feature-dim` | 42 | Must match the wrapper for `legal_features`. |
| `--validation-states` | 10,000 | Strict pre-training decision states. |
| `--save-model` | false | Enables periodic and final checkpoint writes. |
| `--track` | false | Enables W&B in addition to local TensorBoard logs. |

Run `python py_rl/cleanrl/cleanrl/ppo.py --help` for the complete generated CLI.

## Preflight validation

Validation is enabled by default. It checks canonicalization, collisions, mask/slot agreement, sampled legal-ID execution, fallback behavior, and coverage of representative action families. Successful results are cached under `.cache/action_validator/` using a fingerprint that includes interface code hashes, geometry, feature contract, actor settings, and exact pool file identities.

Use `--force-revalidate-action-interface` after suspicious environmental changes. Disabling validation with `--no-validate-action-interface` is suitable only for narrowly controlled debugging.

## Outputs and checkpoints

TensorBoard events and checkpoints are written to `runs/<run_name>/`. With `--save-model`, periodic files are named `model_checkpoint_<step>.cleanrl_model`; the final path is either `--model-path` or `<run_dir>/ppo.cleanrl_model`.

Every saved model has an adjacent `.action_interface.json` sidecar. Evaluators require it and reject mismatched geometry, observation/action dimensions, actor mode, catalog fingerprint/version, feature version/dimension, canonicalizer version, or legal-slot capacity.

The trainer does not implement checkpoint resume or optimizer-state restoration. Loading an existing policy is an evaluation workflow, not a continuation workflow.

## Tracking

Local TensorBoard logging is always enabled:

```powershell
tensorboard --logdir runs
```

For W&B, install/configure `wandb`, then add `--track --wandb-project-name <project>`. Training summary values are useful diagnostics but are not substitutes for a fixed evaluation protocol.

## Parallel JVMs and profiling

Each vector worker starts its own JVM. Startup jitter defaults to 0.1–2.0 seconds to avoid a process-creation burst. If workers fail or memory pressure is high, reduce `--num-envs` before changing rollout or optimization settings.

Set `POLYVISION_PROFILE_SPS=1` to collect timing breakdowns. `POLYVISION_PROFILE_EVERY_N_STEPS` controls reporting frequency and `POLYVISION_PROFILE_OUTPUT_DIR` controls JSON output (default `outputs/sps_profiles`). `--enable-step-diagnostics` adds richer metrics at a throughput cost.

See [Reproducibility](reproducibility.md) before comparing runs.
