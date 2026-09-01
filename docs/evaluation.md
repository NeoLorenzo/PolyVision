# Evaluation

> **Current Reference Benchmark:** The active reference baseline is the **Phase 1 v5 PARITY002 Seed3 16M Terminal-SPT frozen reference benchmark** (`model_checkpoint_16000000.cleanrl_model`, SHA-256 `924d4603fa5038b3ca11081cdfcb5c7a00063949dab44f50f05989e5c9dab061`), evaluated under `phase1_environment_version=v5_human_information_parity` (6,424-d observation) and `phase1_opening_version=v2_guaranteed_two_unit` with Terminal-SPT reward shaping enabled. This is single-training-seed evidence (Seed 3). Phase 1 is **not** complete; active optimization continues.
>
> The previous [Phase 1 v4 PARITY001 Seed3 16M Terminal-SPT Reference Run](results/Phase1_V4_PARITY001_Seed3_16M_TerminalSPT_Reference_Run.md) (19.78 test argmax SPT, 586-d observation), [Phase 1 v3 Seed3 16M Terminal-SPT Reference Run](results/Phase1_V3_Seed3_16M_TerminalSPT_Reference_Run.md) (19.34 test argmax SPT, 505-d observation), and [Phase 1 v3 Seed3 16M Reference Run](results/Phase1_V3_Seed3_16M_Reference_Run.md) (16.88 test argmax SPT) are preserved as historical benchmarks. Historical Seed-1 remains shelved and valid only for historical `v1_mixed_capital_regression`.

PolyVision has checkpoint introspection, a canonical Phase 1 batch evaluator, fair policy-visible baselines, contract validators, and historical comparison artifacts. For the constrained Phase 1 Bardur Turn-10 task, current-interface multi-seed evaluation and training are a deferred research stage, not yet authorized by the roadmap: GitHub issue #3 owns defining and evaluating the prerequisite human-relative gate through the canonical [human benchmark](human-benchmark.md) workflow. This documentation does not define an exact statistical threshold; multi-seed work remains deferred until issue #3 establishes the gate and the applicable model or candidate passes it.

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

PPO argmax selects the highest-logit valid legal slot. PPO sampled uses a recorded per-episode PyTorch generator. Random legal samples uniformly from the policy-visible valid slots. Visible greedy consumes only the flattened observation and the padded legal IDs, validity mask, and 47-dimensional legal features; neither baseline reads raw Java actions, hidden state, or no-fog information. Every choice executes through `env.step(global_id)`.

Final Turn-10 stars per turn is the primary capability metric; shaped return is diagnostic. The evaluator reports episode distributions, but stochastic headline means and confidence intervals aggregate the five replicates into one mean per map before inference. Policies share canonical map identities and environment seeds for the same replicate, and comparisons report paired map-level deltas, uncertainty, and win/tie/loss counts.

Outputs are written to `outputs/evaluations/<evaluation_id>/`: `config.json`, `episodes.jsonl`, `per_map.csv`, `summary.json`, `summary.csv`, and `comparison.csv`. Configuration records manifest/pool identity, ordered map hashes, schedule/RNG rules, checkpoint and sidecar hashes, interface metadata, Git/runtime provenance, and relevant `POLYVISION_*` settings. `--max-maps` and nonstandard repeats produce an explicitly partial, smoke, noncanonical result.

## Active Reference Benchmark Results (Phase 1 v5 PARITY002 Seed3 16M Terminal-SPT)

The authoritative run card is documented in [Phase 1 v5 PARITY002 Seed3 16M Terminal-SPT Reference Run](results/Phase1_V5_PARITY002_Seed3_16M_TerminalSPT_Reference_Run.md).

### Validation evaluation (`outputs/evaluations/20260827_phase1_v5_parity002_seed3_16m_terminal_spt_validation_canonical`)

Evaluated on 250 held-out validation maps (3,000 episodes total):
- **PPO argmax:** Mean 20.52 Turn-10 SPT (95% CI [19.98, 21.04]), Median 21.00
- **PPO sampled:** Mean 19.53 Turn-10 SPT (95% CI [19.19, 19.88]), Median 19.60
- **Visible greedy:** Mean 7.93 Turn-10 SPT (95% CI [7.78, 8.09]), Median 8.00
- **Random legal:** Mean 6.67 Turn-10 SPT (95% CI [6.58, 6.75]), Median 6.60
- **Paired comparisons:**
  - PPO argmax vs visible greedy: 250 W / 0 T / 0 L (+12.58 SPT mean advantage, 95% CI [12.09, 13.11])
  - PPO sampled vs random legal: 250 W / 0 T / 0 L (+12.86 SPT mean advantage, 95% CI [12.55, 13.18])
  - PPO argmax vs random legal: 250 W / 0 T / 0 L (+13.85 SPT mean advantage, 95% CI [13.33, 14.36])

### Fixed held-out test evaluation (`outputs/evaluations/20260827_phase1_v5_parity002_seed3_16m_terminal_spt_pristine_test`)

Evaluated on 250 fixed held-out test maps after checkpoint freeze (3,000 episodes total):
- **PPO argmax:** Mean 20.17 Turn-10 SPT (95% CI [19.67, 20.67]), Median 20.00
- **PPO sampled:** Mean 19.36 Turn-10 SPT (95% CI [19.04, 19.69]), Median 19.40
- **Visible greedy:** Mean 7.96 Turn-10 SPT (95% CI [7.82, 8.10]), Median 8.00
- **Random legal:** Mean 6.70 Turn-10 SPT (95% CI [6.62, 6.79]), Median 6.60
- **Paired comparisons:**
  - PPO argmax vs visible greedy: 250 W / 0 T / 0 L (+12.22 SPT mean advantage, 95% CI [11.73, 12.68])
  - PPO sampled vs random legal: 250 W / 0 T / 0 L (+12.66 SPT mean advantage, 95% CI [12.33, 12.97])
  - PPO argmax vs random legal: 250 W / 0 T / 0 L (+13.47 SPT mean advantage, 95% CI [13.01, 13.96])
- **Validation $\rightarrow$ Test deltas:** argmax $-0.35$ SPT, sampled $-0.17$ SPT, greedy $+0.03$ SPT, random $+0.03$ SPT.

### Comparison: Active v5 PARITY002 Reference vs. Superseded v4 PARITY001 Reference

| Split / Policy | Superseded v4 Reference (586-d) | Active v5 PARITY002 Reference (6,424-d) | Paired Mean Δ (v5 $-$ v4) | Paired 95% CI for Δ |
|---|---:|---:|---:|---|
| **Validation PPO argmax** | 19.78 SPT | **20.52 SPT** | **$+0.73$ SPT** | **[$+0.15$, $+1.34$]** |
| **Validation PPO sampled** | 19.47 SPT | **19.53 SPT** | **$+0.06$ SPT** | [$-0.22$, $+0.33$] |
| **Validation Visible greedy** | 7.93 SPT | **7.93 SPT** | 0.00 SPT | [$0.00$, $0.00$] |
| **Validation Random legal** | 6.67 SPT | **6.67 SPT** | 0.00 SPT | [$0.00$, $0.00$] |
| **Test PPO argmax** | 19.78 SPT | **20.17 SPT** | **$+0.40$ SPT** | [$-0.14$, $+0.94$] |
| **Test PPO sampled** | 19.26 SPT | **19.36 SPT** | **$+0.11$ SPT** | [$-0.16$, $+0.38$] |
| **Test Visible greedy** | 7.96 SPT | **7.96 SPT** | 0.00 SPT | [$0.00$, $0.00$] |
| **Test Random legal** | 6.70 SPT | **6.70 SPT** | 0.00 SPT | [$0.00$, $0.00$] |

On the fixed held-out test set, deterministic argmax scored +0.40 SPT higher than the v4 reference (+0.73 SPT on validation). Both models represent single training seeds (Seed 3), and because the 95% CI on the test split crosses zero and multiple parity modifications were introduced simultaneously, this comparison does not establish causality for individual interface features.

---

## Historical Phase 1 v4 PARITY001 Seed3 16M Terminal-SPT Reference Results (Superseded)

The run card for this superseded reference is documented in [Phase 1 v4 PARITY001 Seed3 16M Terminal-SPT Reference Run](results/Phase1_V4_PARITY001_Seed3_16M_TerminalSPT_Reference_Run.md).

### Historical v4 Validation evaluation (`outputs/evaluations/20260826_phase1_v4_parity001_seed3_16m_terminal_spt_validation_canonical`)

Evaluated on 250 held-out validation maps (3,000 episodes total):
- **PPO argmax:** Mean 19.78 Turn-10 SPT (95% CI [19.30, 20.26]), Median 20.00
- **PPO sampled:** Mean 19.47 Turn-10 SPT (95% CI [19.16, 19.79]), Median 19.40
- **Visible greedy:** Mean 7.93 Turn-10 SPT (95% CI [7.78, 8.09]), Median 8.00
- **Random legal:** Mean 6.67 Turn-10 SPT (95% CI [6.58, 6.75]), Median 6.60
- **Paired comparisons:**
  - PPO argmax vs visible greedy: 250 W / 0 T / 0 L (+11.85 SPT mean advantage, 95% CI [11.39, 12.30])
  - PPO sampled vs random legal: 250 W / 0 T / 0 L (+12.80 SPT mean advantage, 95% CI [12.50, 13.08])
  - PPO argmax vs random legal: 250 W / 0 T / 0 L (+13.12 SPT mean advantage, 95% CI [12.67, 13.57])

### Historical v4 Test evaluation (`outputs/evaluations/20260826_phase1_v4_parity001_seed3_16m_terminal_spt_pristine_test`)

Evaluated on 250 fixed held-out test maps after checkpoint freeze (3,000 episodes total):
- **PPO argmax:** Mean 19.78 Turn-10 SPT (95% CI [19.30, 20.25]), Median 20.00
- **PPO sampled:** Mean 19.26 Turn-10 SPT (95% CI [18.92, 19.58]), Median 19.40
- **Visible greedy:** Mean 7.96 Turn-10 SPT (95% CI [7.82, 8.10]), Median 8.00
- **Random legal:** Mean 6.70 Turn-10 SPT (95% CI [6.62, 6.79]), Median 6.60
- **Paired comparisons:**
  - PPO argmax vs visible greedy: 250 W / 0 T / 0 L (+11.82 SPT mean advantage, 95% CI [11.36, 12.26])
  - PPO sampled vs random legal: 250 W / 0 T / 0 L (+12.55 SPT mean advantage, 95% CI [12.23, 12.86])
  - PPO argmax vs random legal: 250 W / 0 T / 0 L (+13.07 SPT mean advantage, 95% CI [12.61, 13.54])

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
