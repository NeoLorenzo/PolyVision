# Evaluation

> **Current Reference Benchmark:** The active reference baseline is the **Phase 1 v4 PARITY001 Seed3 16M Terminal-SPT frozen reference benchmark** (`model_checkpoint_16000000.cleanrl_model`, SHA-256 `bda59c9eb4603734c3ed599922174890c5b5a1d4a4fad3fd998c289dd3f53574`), evaluated under `phase1_environment_version=v4_exact_per_city_state` (586-d observation) and `phase1_opening_version=v2_guaranteed_two_unit` with Terminal-SPT reward shaping enabled. Phase 1 is **not** complete; active optimization continues.
>
> The previous [Phase 1 v3 Seed3 16M Terminal-SPT Reference Run](results/Phase1_V3_Seed3_16M_TerminalSPT_Reference_Run.md) (19.34 test argmax SPT, 505-d observation) and [Phase 1 v3 Seed3 16M Reference Run](results/Phase1_V3_Seed3_16M_Reference_Run.md) (16.88 test argmax SPT) are preserved as historical benchmarks. Historical Seed-1 remains shelved and valid only for historical `v1_mixed_capital_regression`.

PolyVision has checkpoint introspection, a canonical Phase 1 batch evaluator, fair policy-visible baselines, contract validators, and historical comparison artifacts. Current-interface multi-seed evidence remains an active research target.

## Canonical Phase 1 validation and test suite

`tools/evaluate_phase1.py` is the maintained batch evaluator. The validation pool (`levels/phase1_pool_bardur_real/validation/*.csv`) is used for model development and checkpoint selection. The fixed held-out test pool (`levels/phase1_pool_bardur_real/test/*.csv`) is evaluated only after a candidate checkpoint and protocol are deliberately frozen. The CLI refuses `--pool test` unless `--confirm-test` is also supplied.

```powershell
python tools/evaluate_phase1.py `
    --model-path 'runs/<run>/ppo.cleanrl_model' `
    --pool validation `
    --suite full `
    --repeats-per-map 5 `
    --seed 42
```

| Policy | Runs per map | Pool maps | Total episodes |
|---|---:|---:|---:|
| PPO argmax | 1 | 250 | 250 |
| PPO sampled | 5 | 250 | 1,250 |
| Visible greedy | 1 | 250 | 250 |
| Random legal | 5 | 250 | 1,250 |
| **Total** |  |  | **3,000** |

PPO argmax selects the highest-logit valid legal slot. PPO sampled uses a recorded per-episode PyTorch generator. Random legal samples uniformly from the policy-visible valid slots. Visible greedy consumes only the flattened observation and the padded legal IDs, validity mask, and 42-dimensional legal features; neither baseline reads raw Java actions, hidden state, or no-fog information. Every choice executes through `env.step(global_id)`.

Final Turn-10 stars per turn is the primary capability metric; shaped return is diagnostic. The evaluator reports episode distributions, but stochastic headline means and confidence intervals aggregate the five replicates into one mean per map before inference. Policies share canonical map identities and environment seeds for the same replicate, and comparisons report paired map-level deltas, uncertainty, and win/tie/loss counts.

Outputs are written to `outputs/evaluations/<evaluation_id>/`: `config.json`, `episodes.jsonl`, `per_map.csv`, `summary.json`, `summary.csv`, and `comparison.csv`. Configuration records manifest/pool identity, ordered map hashes, schedule/RNG rules, checkpoint and sidecar hashes, interface metadata, Git/runtime provenance, and relevant `POLYVISION_*` settings. `--max-maps` and nonstandard repeats produce an explicitly partial, smoke, noncanonical result.

## Active Reference Benchmark Results (Phase 1 v4 PARITY001 Seed3 16M Terminal-SPT)

The authoritative run card is documented in [Phase 1 v4 PARITY001 Seed3 16M Terminal-SPT Reference Run](results/Phase1_V4_PARITY001_Seed3_16M_TerminalSPT_Reference_Run.md).

### Validation evaluation (`outputs/evaluations/20260826_phase1_v4_parity001_seed3_16m_terminal_spt_validation_canonical`)

Evaluated on 250 held-out validation maps (3,000 episodes total):
- **PPO argmax:** Mean 19.78 Turn-10 SPT (95% CI [19.30, 20.26]), Median 20.00
- **PPO sampled:** Mean 19.47 Turn-10 SPT (95% CI [19.16, 19.79]), Median 19.40
- **Visible greedy:** Mean 7.93 Turn-10 SPT (95% CI [7.78, 8.09]), Median 8.00
- **Random legal:** Mean 6.67 Turn-10 SPT (95% CI [6.58, 6.75]), Median 6.60
- **Paired comparisons:**
  - PPO argmax vs visible greedy: 250 W / 0 T / 0 L (+11.85 SPT mean advantage, 95% CI [11.39, 12.30])
  - PPO sampled vs random legal: 250 W / 0 T / 0 L (+12.80 SPT mean advantage, 95% CI [12.50, 13.08])
  - PPO argmax vs random legal: 250 W / 0 T / 0 L (+13.12 SPT mean advantage, 95% CI [12.67, 13.57])

### Fixed held-out test evaluation (`outputs/evaluations/20260826_phase1_v4_parity001_seed3_16m_terminal_spt_pristine_test`)

Evaluated on 250 fixed held-out test maps after checkpoint freeze (3,000 episodes total):
- **PPO argmax:** Mean 19.78 Turn-10 SPT (95% CI [19.30, 20.25]), Median 20.00
- **PPO sampled:** Mean 19.26 Turn-10 SPT (95% CI [18.92, 19.58]), Median 19.40
- **Visible greedy:** Mean 7.96 Turn-10 SPT (95% CI [7.82, 8.10]), Median 8.00
- **Random legal:** Mean 6.70 Turn-10 SPT (95% CI [6.62, 6.79]), Median 6.60
- **Paired comparisons:**
  - PPO argmax vs visible greedy: 250 W / 0 T / 0 L (+11.82 SPT mean advantage, 95% CI [11.36, 12.26])
  - PPO sampled vs random legal: 250 W / 0 T / 0 L (+12.55 SPT mean advantage, 95% CI [12.23, 12.86])
  - PPO argmax vs random legal: 250 W / 0 T / 0 L (+13.07 SPT mean advantage, 95% CI [12.61, 13.54])
- **Validation $\rightarrow$ Test deltas:** argmax $-0.01$ SPT (flat), sampled $-0.22$ SPT, greedy $+0.02$ SPT, random $+0.03$ SPT.

### Comparison: Active v4 PARITY001 Reference vs. Superseded v3 Terminal-SPT Reference

| Split / Policy | Superseded v3 Reference (505-d) | Active v4 PARITY001 Reference (586-d) | Paired Mean Δ (v4 $-$ v3) | Paired 95% CI for Δ |
|---|---:|---:|---:|---|
| **Validation PPO argmax** | 19.87 SPT | **19.78 SPT** | **$-0.09$ SPT** | [$-0.54$, $+0.39$] |
| **Validation PPO sampled** | 19.53 SPT | **19.47 SPT** | **$-0.06$ SPT** | [$-0.30$, $+0.19$] |
| **Validation Visible greedy** | 7.93 SPT | **7.93 SPT** | 0.00 SPT | [$0.00$, $0.00$] |
| **Validation Random legal** | 6.67 SPT | **6.67 SPT** | 0.00 SPT | [$0.00$, $0.00$] |
| **Test PPO argmax** | 19.34 SPT | **19.78 SPT** | **$+0.43$ SPT** | **[$+0.004$, $+0.84$]** |
| **Test PPO sampled** | 18.96 SPT | **19.26 SPT** | **$+0.30$ SPT** | **[$+0.06$, $+0.53$]** |
| **Test Visible greedy** | 7.96 SPT | **7.96 SPT** | 0.00 SPT | [$0.00$, $0.00$] |
| **Test Random legal** | 6.70 SPT | **6.70 SPT** | 0.00 SPT | [$0.00$, $0.00$] |

On the fixed held-out test set, deterministic argmax scored +0.43 SPT higher than the v3 reference, with a paired confidence interval strictly above zero (95% CI [+0.004, +0.840]). Both models represent single training seeds (Seed 3), and this comparison does not establish causality.

---

## Historical Phase 1 v3 Seed3 16M Terminal-SPT Reference Results (Superseded)

The run card for this superseded reference is documented in [Phase 1 v3 Seed3 16M Terminal-SPT Reference Run](results/Phase1_V3_Seed3_16M_TerminalSPT_Reference_Run.md).

### Historical v3 Validation evaluation (`outputs/evaluations/20260825_phase1_v3_seed3_16m_terminal_spt_validation_canonical`)

Evaluated on 250 held-out validation maps (3,000 episodes total):
- **PPO argmax:** Mean 19.87 Turn-10 SPT (95% CI [19.46, 20.29]), Median 20.00
- **PPO sampled:** Mean 19.53 Turn-10 SPT (95% CI [19.25, 19.82]), Median 19.60
- **Visible greedy:** Mean 7.93 Turn-10 SPT (95% CI [7.78, 8.09]), Median 8.00
- **Random legal:** Mean 6.67 Turn-10 SPT (95% CI [6.58, 6.75]), Median 6.60
- **Paired comparisons:**
  - PPO argmax vs visible greedy: 250 W / 0 T / 0 L (+11.94 SPT mean advantage, 95% CI [11.52, 12.36])
  - PPO sampled vs random legal: 250 W / 0 T / 0 L (+12.86 SPT mean advantage, 95% CI [12.59, 13.14])
  - PPO argmax vs random legal: 250 W / 0 T / 0 L (+13.20 SPT mean advantage, 95% CI [12.78, 13.62])

### Historical v3 Test evaluation (`outputs/evaluations/20260825_phase1_v3_seed3_16m_terminal_spt_pristine_test`)

Evaluated on 250 held-out test maps after checkpoint freeze (3,000 episodes total):
- **PPO argmax:** Mean 19.34 Turn-10 SPT (95% CI [18.95, 19.75]), Median 19.00
- **PPO sampled:** Mean 18.96 Turn-10 SPT (95% CI [18.64, 19.27]), Median 19.00
- **Visible greedy:** Mean 7.96 Turn-10 SPT (95% CI [7.82, 8.10]), Median 8.00
- **Random legal:** Mean 6.70 Turn-10 SPT (95% CI [6.62, 6.79]), Median 6.60
- **Paired comparisons:**
  - PPO argmax vs visible greedy: 250 W / 0 T / 0 L (+11.39 SPT mean advantage, 95% CI [10.98, 11.79])
  - PPO sampled vs random legal: 250 W / 0 T / 0 L (+12.25 SPT mean advantage, 95% CI [11.95, 12.55])
  - PPO argmax vs random legal: 250 W / 0 T / 0 L (+12.64 SPT mean advantage, 95% CI [12.22, 13.05])

---

## Historical Phase 1 v3 Seed3 16M Reference Results (Superseded)

The original reference run card is documented in [Phase 1 v3 Seed3 16M Reference Run](results/Phase1_V3_Seed3_16M_Reference_Run.md).

### Historical Validation evaluation (`outputs/evaluations/20260824_phase1_v3_seed3_16m_validation_canonical`)

Evaluated on 250 held-out validation maps (3,000 episodes total):
- **PPO argmax:** Mean 17.18 Turn-10 SPT (95% CI [16.72, 17.66]), Median 17.00
- **PPO sampled:** Mean 17.70 Turn-10 SPT (95% CI [17.37, 18.03]), Median 17.60
- **Visible greedy:** Mean 7.93 Turn-10 SPT (95% CI [7.78, 8.09]), Median 8.00
- **Random legal:** Mean 6.67 Turn-10 SPT (95% CI [6.58, 6.75]), Median 6.60
- **Paired comparisons:** PPO argmax beat visible greedy on 249/250 maps (249 W / 0 T / 1 L, +9.25 SPT mean difference). PPO sampled beat random legal on 250/250 maps (+11.03 SPT). PPO argmax beat random legal on 250/250 maps (+10.52 SPT).

### Historical Pristine test evaluation (`outputs/evaluations/20260824_phase1_v3_seed3_16m_pristine_test`)

Evaluated on 250 pristine test maps after checkpoint freeze (3,000 episodes total):
- **PPO argmax:** Mean 16.88 Turn-10 SPT (95% CI [16.46, 17.31]), Median 17.00
- **PPO sampled:** Mean 17.55 Turn-10 SPT (95% CI [17.20, 17.89]), Median 17.50
- **Visible greedy:** Mean 7.96 Turn-10 SPT (95% CI [7.82, 8.10]), Median 8.00
- **Random legal:** Mean 6.70 Turn-10 SPT (95% CI [6.62, 6.79]), Median 6.60
- **Paired comparisons:** PPO argmax beat visible greedy on all 250 maps (250 W / 0 T / 0 L, +8.93 SPT mean difference). PPO sampled beat random legal on 250/250 maps (+10.84 SPT). PPO argmax beat random legal on 250/250 maps (+10.18 SPT).
- **Validation $\rightarrow$ Test deltas:** argmax $-0.30$ SPT, sampled $-0.15$ SPT, greedy $+0.03$ SPT, random $+0.03$ SPT.

### Historical Diagnostic 4M Checkpoint Evaluation (`outputs/evaluations/20260824_phase1_v3_seed3_4m_validation_argmax`)

A focused diagnostic evaluation was conducted on the frozen 4,000,000-step checkpoint (`model_checkpoint_4000000.cleanrl_model`) across all 250 validation maps (1 episode per map, deterministic argmax, seed 42) to test the specific hypothesis that PPO learned deterministic Forestry early in training and subsequently forgot it:
- **Purpose:** Hypothesis testing for behavioral analysis (not checkpoint selection or benchmark ranking).
- **Scope:** Confined strictly to the development validation pool (250 maps); zero test maps were accessed.
- **Results:** PPO argmax achieved mean 17.088 Turn-10 SPT (95% CI [16.64, 17.54], median 17.00), 0 / 250 explicit Forestry adoption (0.0%), and 0.048 mean lumber huts (from 1 ruin drop).
- **Finding:** The 4M deterministic policy behaviorally matched the 16M reference model (17.18 SPT, 0.0% Forestry), refuting the catastrophic forgetting hypothesis and showing that mid-training telemetry reflected stochastic rollout exploration rather than an established argmax strategy.


## Historical Phase 1 Seed-1 validation result

The first complete 3,000-episode suite, `20260814T110912Z_validation_canonical`, evaluated the Seed-1 10M checkpoint across all 250 validation maps. PPO argmax achieved mean T10 SPT 14.576 (map-level bootstrap 95% CI [14.204, 14.948]), compared with 13.678 for PPO sampled, 7.128 for visible greedy, and 5.897 for random legal. PPO argmax beat visible greedy on all 250 paired maps.

Opening-audit interpretation: the result remains internally valid for the historical mixed-opening task. Training used 55.02% two-unit and 44.98% one-unit maps; validation used 56.40% and 43.60%. Do not reinterpret this result as evaluation under a universal two-unit opening.

Read the authoritative [Seed-1 mixed-opening reflection](results/Phase1_Seed1_Mixed_Opening_Validation_Reflection.md). This historical result is preserved but shelved; it is development evidence for v1, not the current Phase 1 reference candidate or a pristine-test result.

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

The historical W&B export includes the completed 10M training run for Seed-1. Its retained values are training snapshots; canonical evaluation evidence is documented in dedicated validation and test suites. The active reference baseline is the Phase 1 v3 Seed3 16M Terminal-SPT frozen reference benchmark (`model_checkpoint_16000000.cleanrl_model`), while Phase 1 optimization continues.

The strongest committed repeated-episode comparison from early development is a historical 500-episode PPO-versus-Organization-oracle evaluation. It used an older map/action contract and a map sequence that was not strictly paired between policies. It is useful methodology and milestone evidence but is not directly comparable to current checkpoints. Details live in the clearly marked [historical benchmark registry](history/model-run-benchmark-log.md).
