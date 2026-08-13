# Evaluation

PolyVision has checkpoint introspection, scripted baselines, contract validators, and historical comparison artifacts. A current-interface multi-seed held-out benchmark is still an active research target.

## Inspect one checkpoint

`evaluate_brain.py` is the current checkpoint-aware policy inspection tool. It requires the model's `.action_interface.json` sidecar and validates compatibility before loading weights.

```powershell
$env:POLYVISION_SOLO_NO_OPPONENT_MODE = '1'
python evaluate_brain.py `
    --model-path runs/<run>/ppo.cleanrl_model `
    --level-pool-glob 'levels/phase1_pool_bardur_real/validation/*.csv' `
    --seed 42
```

Add `--render-java` for the Swing viewer or `--manual-step` to pause between decisions. This is a one-episode introspection tool, not a statistical evaluator. Use validation for development inspection; do not repeatedly inspect test maps.

## Dataset roles

- Use `validation/*.csv` for repeated model/configuration comparison and checkpoint selection.
- Use `test/*.csv` only after selection for a final generalization claim. Test-informed changes contaminate that test result for the development cycle.
- Use `human_benchmark/*.csv` for human-versus-agent challenge results under the same wrapper contract. This challenge set may influence future development and is not a substitute for pristine test evidence.
- Never call a result held out merely because the same map bytes were copied or renamed elsewhere; separation is enforced by canonical/content identity.

## Human benchmark workflow

Run `python tools/human_benchmark.py` for a persistent first-attempt human challenge on the separate 17-map pool. Official runs use `TribesGymWrapper`, the flattened policy observation, the legal-slot tensors, stable global IDs, and `env.step(global_id)`; diagnostic renderers and raw action details are blocked. Replays remain separate from the canonical first completion. See [Human benchmark](human-benchmark.md) for commands, registry format, parity validation, and interpretation.

## Current baselines and audits

- `py_rl/cleanrl/cleanrl/evaluate_visible_greedy_movement.py` evaluates a policy-visible greedy movement baseline over repeated episodes.
- `py_rl/cleanrl/cleanrl/evaluate_no_fog_runtime_village_greedy.py` is explicitly a no-fog diagnostic; it is privileged and must not be presented as a fair policy baseline.
- `py_rl/cleanrl/cleanrl/privileged_nearest_village_oracle.py` and `tools/eval_org_only_oracle_vs_ppo.py` are research/oracle infrastructure. The latter defaults to the development validation pool but remains a privileged diagnostic rather than a fair policy baseline.
- `audit_*.py`, `validate_*features.py`, and `legal_features_diagnostics.py` under the active CleanRL directory target specific action and feature invariants.
- `tools/validate_environment_contract.py` checks every map in a pool without evaluating policy quality.

Inspect each tool's `--help` and its visibility assumptions before using its output as evidence.

## Recommended controlled protocol

For a defensible current comparison:

1. select checkpoints and configurations only on validation, then evaluate the selected result on the pristine test pool;
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
