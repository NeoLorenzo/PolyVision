# Evaluation

> **Seed-1 is shelved before pristine test.** Its validation remains valid for historical `v1_mixed_capital_regression`, but it is not a candidate for the corrected task. The test maps were reset only for environment-contract audits: no checkpoint, policy action, capability score, or tuning feedback was involved. They remain pristine with respect to model capability evaluation and development feedback.

PolyVision has checkpoint introspection, a canonical Phase 1 batch evaluator, fair policy-visible baselines, contract validators, and historical comparison artifacts. Current-interface multi-seed evidence remains an active research target.

## Canonical Phase 1 validation suite

`tools/evaluate_phase1.py` is the maintained batch evaluator. The current run is a **validation evaluation**: its results may be inspected and used for future model decisions. The pristine test pool must remain untouched until a candidate checkpoint and protocol are deliberately frozen. The CLI refuses `--pool test` unless `--confirm-test` is also supplied.

```powershell
python tools/evaluate_phase1.py `
    --model-path 'runs/<run>/ppo.cleanrl_model' `
    --pool validation `
    --suite full `
    --repeats-per-map 5 `
    --seed 42
```

| Policy | Runs per map | Validation maps | Total episodes |
|---|---:|---:|---:|
| PPO argmax | 1 | 250 | 250 |
| PPO sampled | 5 | 250 | 1,250 |
| Visible greedy | 1 | 250 | 250 |
| Random legal | 5 | 250 | 1,250 |
| **Total** |  |  | **3,000** |

PPO argmax selects the highest-logit valid legal slot. PPO sampled uses a recorded per-episode PyTorch generator. Random legal samples uniformly from the policy-visible valid slots. Visible greedy consumes only the flattened observation and the padded legal IDs, validity mask, and 42-dimensional legal features; neither baseline reads raw Java actions, hidden state, or no-fog information. Every choice executes through `env.step(global_id)`.

Final Turn-10 stars per turn is the primary capability metric; shaped return is diagnostic. The evaluator reports episode distributions, but stochastic headline means and confidence intervals aggregate the five replicates into one mean per map before inference. Policies share canonical map identities and environment seeds for the same replicate, and comparisons report paired map-level deltas, uncertainty, and win/tie/loss counts.

Outputs are written to `outputs/evaluations/<evaluation_id>/`: `config.json`, `episodes.jsonl`, `per_map.csv`, `summary.json`, `summary.csv`, and `comparison.csv`. Configuration records manifest/pool identity, ordered map hashes, schedule/RNG rules, checkpoint and sidecar hashes, interface metadata, Git/runtime provenance, and relevant `POLYVISION_*` settings. `--max-maps` and nonstandard repeats produce an explicitly partial, smoke, noncanonical result.

## First canonical validation result

The first complete 3,000-episode suite, `20260814T110912Z_validation_canonical`, evaluated the Seed-1 10M checkpoint across all 250 validation maps. PPO argmax achieved mean T10 SPT 14.576 (map-level bootstrap 95% CI [14.204, 14.948]), compared with 13.678 for PPO sampled, 7.128 for visible greedy, and 5.897 for random legal. PPO argmax beat visible greedy on all 250 paired maps.

Opening-audit interpretation: the result remains internally valid for the historical mixed-opening task. Training used 55.02% two-unit and 44.98% one-unit maps; validation used 56.40% and 43.60%. Do not reinterpret this result as evaluation under a universal two-unit opening.

Future canonical evaluations record `phase1_opening_version=v2_guaranteed_two_unit`. Historical sidecars lacking that field, or explicitly marked v1, fail corrected-environment compatibility by default. Retrain and revalidate under v2 before considering the pristine capability test.

Read the authoritative [Seed-1 mixed-opening reflection](results/Phase1_Seed1_Mixed_Opening_Validation_Reflection.md). This historical result is preserved but shelved; it is development evidence for v1, not the final Phase 1 candidate or a pristine-test result.

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

## Additional controlled-evaluation guidance

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

The latest W&B export includes the completed first 10M training run on the current 11×11, 256-slot, 42-feature contract. Its retained values are training snapshots; the canonical validation evidence is the separate 3,000-episode result linked above. Both describe the first Phase 1 model, not the final Phase 1 model.

The strongest committed repeated-episode comparison is a historical 500-episode PPO-versus-Organization-oracle evaluation. It used an older map/action contract and a map sequence that was not strictly paired between policies. It is useful methodology and milestone evidence but is not directly comparable to current checkpoints. Details live in the clearly marked [historical benchmark registry](history/model-run-benchmark-log.md).
