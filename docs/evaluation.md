# Evaluation

PolyVision has checkpoint introspection, scripted baselines, contract validators, and historical comparison artifacts. A current-interface multi-seed held-out benchmark is still an active research target.

## Inspect one checkpoint

`evaluate_brain.py` is the current checkpoint-aware policy inspection tool. It requires the model's `.action_interface.json` sidecar and validates compatibility before loading weights.

```powershell
$env:POLYVISION_SOLO_NO_OPPONENT_MODE = '1'
python evaluate_brain.py `
    --model-path runs/<run>/ppo.cleanrl_model `
    --level-pool-glob 'levels/phase1_pool_bardur_real/*.csv' `
    --seed 42
```

Add `--render-java` for the Swing viewer or `--manual-step` to pause between decisions. This is a one-episode introspection tool, not a statistical evaluator.

## Current baselines and audits

- `py_rl/cleanrl/cleanrl/evaluate_visible_greedy_movement.py` evaluates a policy-visible greedy movement baseline over repeated episodes.
- `py_rl/cleanrl/cleanrl/evaluate_no_fog_runtime_village_greedy.py` is explicitly a no-fog diagnostic; it is privileged and must not be presented as a fair policy baseline.
- `py_rl/cleanrl/cleanrl/privileged_nearest_village_oracle.py` and `tools/eval_org_only_oracle_vs_ppo.py` are research/oracle infrastructure. The latter currently hard-codes a removed legacy pool and is retained only to reproduce its historical protocol; it is not a current-pool evaluator.
- `audit_*.py`, `validate_*features.py`, and `legal_features_diagnostics.py` under the active CleanRL directory target specific action and feature invariants.
- `tools/validate_environment_contract.py` checks every map in a pool without evaluating policy quality.

Inspect each tool's `--help` and its visibility assumptions before using its output as evidence.

## Recommended controlled protocol

For a defensible current comparison:

1. freeze a held-out genuine-map manifest that was not used to select checkpoints;
2. use identical ordered map IDs and episode seeds for every policy;
3. record the commit SHA, dirty status, dependency environment, checkpoint hash, and action-interface sidecar;
4. state whether actions are deterministic argmax or sampled;
5. run multiple training seeds and enough evaluation episodes for uncertainty estimates;
6. compare against visible-information random/scripted baselines and label privileged oracles separately;
7. retain per-episode results plus a machine-readable summary.

Report mean, median, standard deviation/confidence interval, and percentiles for final T10 SPT, along with city count, second-city timing, village capture, research, fog discovery, reward return, illegal/fallback rates, and runtime.

## Interpreting existing evidence

The latest committed W&B export includes a completed 500K genuine-map validation run on the current 11×11, 256-slot, 42-feature contract. Its retained values are final training summaries, not held-out multi-seed results.

The strongest committed repeated-episode comparison is a historical 500-episode PPO-versus-Organization-oracle evaluation. It used an older map/action contract and a map sequence that was not strictly paired between policies. It is useful methodology and milestone evidence but is not directly comparable to current checkpoints. Details live in the clearly marked [historical benchmark registry](history/model-run-benchmark-log.md).
