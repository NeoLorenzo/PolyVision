# Phase 1 v4 PARITY001 Seed3 16M Terminal-SPT Frozen Reference Benchmark Run Card

## Overview and Status

- **Status:** CURRENT FROZEN REFERENCE BENCHMARK
- **Date:** 2026-08-26
- **Disposition:** Active frozen reference benchmark for ongoing PolyVision Phase 1 development and optimization. (Phase 1 remains under active development; this run establishes the authoritative reference baseline to beat.)
- **Primary Capability Metric:** Final Turn-10 stars per turn (SPT) on held-out maps
- **Previous Reference:** Supersedes [Phase 1 v3 Seed3 16M Terminal-SPT Reference Run](Phase1_V3_Seed3_16M_TerminalSPT_Reference_Run.md) (19.34 test argmax SPT)

> [!IMPORTANT]
> **Preferred Scientific Headline:**
> "On the fixed 250-map held-out Phase 1 test benchmark, deterministic PPO (`ppo_argmax`) under the `v4_exact_per_city_state` observation contract achieved **19.78 mean Turn-10 SPT** (95% CI [19.30, 20.25], median 20.00), versus **7.96 SPT** for the policy-visible greedy baseline. PPO beat visible greedy on all 250 maps (250 W / 0 T / 0 L), with a mean paired advantage of **+11.82 SPT**."

> [!CAUTION]
> **Single-Seed Scope & Statistical Significance:**
> This reference run represents a **single-training-seed result (Seed 3)**. While the observed difference between v4 and v3 on the fixed held-out test pool is statistically distinguishable from zero on a paired basis (+0.432 mean paired Δ, 95% CI [+0.004, +0.840]), this single-seed comparison characterizes performance on the fixed map pool but does **not** constitute causal proof that the observation change caused the difference. Establishing multi-seed variance and confidence bounds across multiple independent training seeds remains an active milestone.

---

## Authoritative Run Identity & Provenance

All parameters, hashes, and byte sizes below are derived directly from the canonical machine-readable run artifacts, checkpoint sidecars, and evaluation outputs:

| Property | Value |
|---|---|
| **Run directory** | `runs/Tribes-v0__Phase1-Scientific-Train-V4-PARITY001-Seed3-TerminalSPT__3__1787705342` |
| **Canonical frozen checkpoint** | `runs/Tribes-v0__Phase1-Scientific-Train-V4-PARITY001-Seed3-TerminalSPT__3__1787705342/model_checkpoint_16000000.cleanrl_model` |
| **Checkpoint SHA-256** | `bda59c9eb4603734c3ed599922174890c5b5a1d4a4fad3fd998c289dd3f53574` |
| **Checkpoint size** | 16,754,668 bytes |
| **Duplicate final model file** | `runs/Tribes-v0__Phase1-Scientific-Train-V4-PARITY001-Seed3-TerminalSPT__3__1787705342/Phase1-Scientific-Train-V4-PARITY001-Seed3-TerminalSPT.cleanrl_model` (SHA-256: `26932969a0839773804104055d7dc31d29a9ffa1e7179e9db0ec9c5646440f6a`, 16,755,521 bytes) |
| **Action interface sidecar** | `model_checkpoint_16000000.cleanrl_model.action_interface.json` |
| **Sidecar SHA-256** | `b490bcaf686394d8ee0050a05b10a708ad2f1b8726eca11593036bc265f6d752` |
| **Sidecar size** | 716 bytes |
| **Training seed** | `3` |
| **Total environment steps** | 16,000,000 |
| **Training commit SHA** | `88af310c2a73e23f1a8e0d80d07cbd0e2ecb2bad` |
| **Git working tree status during training** | Clean |
| **Environment version** | `v4_exact_per_city_state` |
| **Observation dimension** | 586 |
| **Opening version** | `v2_guaranteed_two_unit` |
| **Actor mode** | `legal_features` |
| **Legal action feature version** | `v1_3_move_focus_plus_semantic_econ` |
| **Legal action feature dimension** | 42 |
| **Max legal action slots** | 256 |
| **Global action catalog size** | 63,913 |
| **Catalog version** | `flat-v1` |
| **Canonicalizer version** | `flat-v1-structured` |
| **Catalog fingerprint** | `c849a4abf7b0bee073ccc56b63ae65917ea30e77068ad648c472130693dfe6e4` |

### Runtime & Dependency Versions

| Component | Version |
|---|---|
| **Python** | `3.11.9` |
| **PyTorch** | `2.11.0+cpu` (evaluation) / CUDA enabled during training |
| **Gymnasium** | `1.3.0` |
| **Py4J** | `0.10.9.9` |
| **Java / JDK** | OpenJDK 21.0.8 (Temurin-21.0.8+9) |
| **Platform** | Windows-10-10.0.22621-SP0 |

---

## Environment & Observation Contract (`PARITY-001`)

### PARITY-001 Interface Change Relative to v3

In historical v3 (`v3_corrected_turn_economy`, 505-d observation), per-city state was destructively aggregated into global summary statistics (`avg_city_level`, `max_city_level`, `mean_upgrade_progress`), leaving the policy blind to which individual city needed population, had reached level thresholds, or was at unit capacity.

Under the current `v4_exact_per_city_state` contract, the observation vector is expanded from **505 to 586 dimensions** ($505 + 9 \times 9 = 586$) by appending a deterministic 9-slot exact per-city state block:

$$586 = 4 \times 121 \text{ (spatial planes)} + 6 \text{ (legacy scalars)} + 15 \text{ (economy scalars)} + 81 \text{ (exact per-city block)}$$

### Exact Per-City Slot Schema (81 Values: 9 Slots $\times$ 9 Features)

For each owned city slot $i \in \{0 \dots 8\}$ (sorted deterministically by $(x, y)$ ascending):
1. `city_present`: `1.0` if city is present in slot $i$, `0.0` for empty slot.
2. `city_x`: Normalized board $x$-coordinate ($x / (W-1)$).
3. `city_y`: Normalized board $y$-coordinate ($y / (H-1)$).
4. `city_level`: Exact unclipped integer level ($1.0, 2.0, \dots$).
5. `city_population`: Exact unclipped integer population ($0.0, 1.0, \dots$).
6. `city_population_need`: Exact unclipped integer population required to level up.
7. `city_production`: Exact unclipped SPT production contributed by this city.
8. `city_supported_unit_count`: Exact unclipped count of living units supported by this city.
9. `city_unit_capacity`: Exact unit capacity ($\text{level} + 1$).

Unused slots ($K < 9$) are strictly zero-padded. Raw engine actor IDs are completely excluded to eliminate arbitrary numerical bias and chronology leakage.

---

## Training Configuration & Hyperparameters

### Launch Command

```powershell
cd C:\PolyVision; $env:POLYVISION_LEVEL_POOL_GLOB='levels/phase1_pool_bardur_real/train/*.csv'; $env:POLYVISION_SOLO_NO_OPPONENT_MODE='1'; $env:POLYVISION_INFO_MODE='fast'; $env:POLYVISION_BATCH_LEGAL_ACTION_FETCH='1'; $env:POLYVISION_DERIVE_OBS_METADATA='1'; $env:POLYVISION_TERMINAL_SPT_REWARD_ENABLED='1'; python py_rl/cleanrl/cleanrl/ppo.py --exp-name Phase1-Scientific-Train-V4-PARITY001-Seed3-TerminalSPT --seed 3 --actor-mode legal_features --total-timesteps 16000000 --num-envs 20 --num-steps 128 --max-legal-actions 256 --legal-action-feature-dim 42 --enable-step-diagnostics --step-diagnostics-log-every 3 --track --wandb-project-name cleanRL --save-model --save-frequency 500000 --force-revalidate-action-interface --validation-states 10000
```

### Key Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| `learning_rate` | `0.00025` | Initial learning rate |
| `anneal_lr` | `True` | Linear learning rate decay over 16M steps (final: $4.0 \times 10^{-8}$) |
| `num_envs` | `20` | Parallel JVM / vector environments |
| `num_steps` | `128` | Rollout horizon per environment |
| `batch_size` | `2560` | Rollout batch size ($20 \times 128$) |
| `minibatch_size` | `640` | Minibatch size (4 minibatches per epoch) |
| `update_epochs` | `4` | PPO update epochs per batch |
| `gamma` ($\gamma$) | `0.99` | Discount factor |
| `gae_lambda` ($\lambda$) | `0.95` | Generalized Advantage Estimation parameter |
| `clip_coef` | `0.2` | PPO surrogate clip range $\epsilon$ |
| `clip_vloss` | `True` | Value loss clipping |
| `ent_coef` | `0.01` | Policy entropy coefficient |
| `vf_coef` | `0.5` | Value function loss coefficient |
| `max_grad_norm` | `0.5` | Gradient clipping threshold |
| `norm_adv` | `True` | Advantage normalization |

### Reward Formulation (Terminal-SPT)

- **Terminal Bonus:**
  $$\text{Terminal Reward} = 1.0 \times \text{final\_SPT} + 2.0 \times \max(0, \text{final\_SPT} - 10) + 3.0 \times \max(0, \text{final\_SPT} - 15)$$
- **Step-Level Dense Shaping:** Stepwise SPT positive change $\times 5.0$ (nonpositive $\times 1.0$); City capture $+3.0$ to $+6.0$; Village exploration $+1.0$ (reveal), $+0.5$ (approach), $+1.0$ (step on), $+5.0$ (opportunity bonus); Fog exploration $+0.08$/tile; Useless move penalty $-0.35$.

---

## Frozen Map Split Verification & Live Contract Evidence

The dataset split is fixed across all 5,517 maps (`split_manifest.json` SHA-256: `c8721cd0fcb636d9ddb483745585e5b69eca9ca501950a1ddec6a51d0c763690`):

| Split | Count | Aggregate Pool Identity (SHA-256) |
|---|---:|---|
| **Train** | 5,000 | `a99b309a3020704e4a95886d5ab3f8b6f7ed00931ee111e3117702009605e8c4` |
| **Validation** | 250 | `a56e74c952ad3b08d4645fc16d66ce0f17e0589fea2fe9042f7cd293669eac72` |
| **Test (Held-Out)** | 250 | `8a8e0f784535e8169ffb829f2e3058b60bcd7493277912dd995ff17b217f4b94` |
| **Human Benchmark** | 17 | `adbe5a707392aad15342f3c595a78abeeb85cb0ce77b91a8d9339276a2674a69` |

### Live Contract Verification Evidence

- Maps tested: 250/250 (validation pool)
- Board geometry: 11×11
- Observation shape: `(586,)`
- Action space: 63,913
- Action catalog fingerprint: `c849a4abf7b0bee073ccc56b63ae65917ea30e77068ad648c472130693dfe6e4`
- Legal actions at reset: 1..23
- Contract failures: 0

---

## Run-Level W&B Training Telemetry Summary

The archived W&B artifact (`outputs/training/phase1_v4_parity001_seed3_16m_terminal_spt_wandb.csv`, SHA-256: `7e2d29ed77f6efc18db51a2783e7911883ad3b0fb727fba3487322bea3861d7b`) is a **runs-table export** containing run-level config and summary metrics, not per-step training history. Consequently, the values below are reported strictly as final/summary telemetry snapshots and are not used to infer learning-curve shape, convergence, stabilization, or temporal dynamics.

Structured derived provenance is recorded in `outputs/training/phase1_v4_parity001_seed3_16m_terminal_spt_summary.json`.

| Metric Category | Summary Metric | Final Recorded Value | Description |
|---|---|---:|---|
| **Throughput** | Training SPS | 363 | Final recorded system transitions per second |
| | Total Runtime | 44,197 s (12.28 h) | Total wall-clock time over 16M steps |
| **Optimization** | Value loss | 71.002 | Final recorded clipped value function loss metric |
| | Policy loss | $5.96 \times 10^{-9}$ | Final recorded surrogate policy loss metric |
| | Policy entropy | 0.935 | Final recorded policy action distribution entropy |
| | Explained variance | 0.694 | PPO value-function explained variance ($1 - \text{Var}(y - \hat{y}) / \text{Var}(y)$) |
| | Approx KL | $-3.73 \times 10^{-10}$ | Final recorded approximate KL divergence metric |
| | Learning rate | $4.00 \times 10^{-8}$ | Final annealed learning rate |
| **Training Rollouts** | Rollout final SPT (T10) | 17.0 | Final recorded episode-end SPT during training rollouts |
| | Rollout final cities | 4.0 | Final recorded mean cities at Turn 10 during rollouts |
| | Forestry adoption rate | 100.0% | Final recorded fraction of rollout episodes researching Forestry |
| | Organization adoption rate | 100.0% | Final recorded fraction of rollout episodes researching Organization |
| | Mean lumber huts | 9.0 | Final recorded mean lumber huts built by Turn 10 |
| | Mean sawmills | 3.0 | Final recorded mean sawmills built by Turn 10 |
| | Non-endturn action rate | 88.1% | Final recorded proportion of productive non-end-turn actions |
| | Mean valid legal actions | 17.33 | Final recorded average legal action slots per decision state |

---

## Canonical Validation Evaluation

- **Directory:** `outputs/evaluations/20260826_phase1_v4_parity001_seed3_16m_terminal_spt_validation_canonical`
- **Maps:** 250 held-out validation maps
- **Suite:** Canonical full suite (3,000 episodes: 250 argmax, 1,250 sampled [5 reps], 250 visible greedy, 1,250 random legal [5 reps])
- **Bootstrap Samples:** 5,000 per-map replicate mean bootstrap samples

### Policy Performance (Turn-10 SPT)

| Policy | Episodes | Mean SPT | Median SPT | Std Dev | Min | Max | p05 | p25 | p75 | p95 | 95% Confidence Interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **PPO argmax** | 250 | **19.784** | **20.00** | 3.84 | 8.0 | 30.0 | 13.0 | 17.0 | 22.0 | 26.0 | **[19.30, 20.26]** |
| **PPO sampled** | 1,250 | **19.471** | **19.40** | 2.49 | 11.4 | 25.6 | 15.6 | 17.8 | 21.4 | 23.4 | **[19.16, 19.79]** |
| **Visible greedy** | 250 | **7.932** | **8.00** | 1.25 | 5.0 | 12.0 | 6.0 | 7.0 | 9.0 | 10.0 | **[7.78, 8.09]** |
| **Random legal** | 1,250 | **6.668** | **6.60** | 0.69 | 4.6 | 8.6 | 5.4 | 6.2 | 7.2 | 7.6 | **[6.58, 6.75]** |

### Paired Comparisons (Validation)

- **PPO argmax vs. Visible greedy:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+11.852 SPT** (95% CI [11.388, 12.300], median +12.0)
- **PPO sampled vs. Random legal:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+12.803 SPT** (95% CI [12.501, 13.078], median +12.8)
- **PPO argmax vs. Random legal:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+13.116 SPT** (95% CI [12.673, 13.570], median +13.3)

---

## Canonical Fixed Held-Out Test Evaluation

- **Directory:** `outputs/evaluations/20260826_phase1_v4_parity001_seed3_16m_terminal_spt_pristine_test`
- **Maps:** 250 fixed held-out test maps
- **Suite:** Canonical full suite (3,000 episodes total)
- **Terminology & Exposure Note:** Checkpoint weights were frozen prior to evaluation. The test pool is a fixed 250-map held-out benchmark that was never exposed to PPO gradients, but has been evaluated in previous development cycles.

### Policy Performance (Turn-10 SPT)

| Policy | Episodes | Mean SPT | Median SPT | Std Dev | Min | Max | p05 | p25 | p75 | p95 | 95% Confidence Interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **PPO argmax** | 250 | **19.776** | **20.00** | 3.83 | 10.0 | 30.0 | 13.0 | 17.0 | 22.0 | 26.0 | **[19.30, 20.25]** |
| **PPO sampled** | 1,250 | **19.255** | **19.40** | 2.66 | 10.6 | 26.6 | 15.0 | 17.25 | 20.8 | 23.8 | **[18.92, 19.58]** |
| **Visible greedy** | 250 | **7.956** | **8.00** | 1.17 | 5.0 | 11.0 | 6.0 | 7.0 | 9.0 | 10.0 | **[7.82, 8.10]** |
| **Random legal** | 1,250 | **6.702** | **6.60** | 0.68 | 4.8 | 8.4 | 5.6 | 6.25 | 7.2 | 7.91 | **[6.62, 6.79]** |

### Paired Comparisons (Held-Out Test)

- **PPO argmax vs. Visible greedy:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+11.820 SPT** (95% CI [11.360, 12.260], median +12.0)
- **PPO sampled vs. Random legal:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+12.554 SPT** (95% CI [12.234, 12.856], median +12.6)
- **PPO argmax vs. Random legal:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+13.074 SPT** (95% CI [12.613, 13.537], median +13.1)

---

## Secondary Behavioral Diagnostics (Validation & Test)

| Metric | Validation (v4 Argmax) | Validation (v4 Sampled) | Test (v4 Argmax) | Test (v4 Sampled) | Test (v4 Greedy) | Test (v4 Random) |
|---|---:|---:|---:|---:|---:|---:|
| **Mean Final Cities** | 4.260 | 4.370 | 4.232 | 4.319 | 3.804 | 1.999 |
| **Mean Final Units** | 6.492 | 6.533 | 6.672 | 6.422 | 2.000 | 5.480 |
| **Mean Unspent Stars** | 24.100 | 23.254 | 24.100 | 23.135 | 53.908 | 9.706 |
| **Mean Fog Cleared** | 49.536 | 52.821 | 49.620 | 51.868 | 37.428 | 30.031 |
| **Forestry Adoption Rate** | 250/250 (100.0%) | 1,210/1,250 (96.8%) | 250/250 (100.0%) | 1,225/1,250 (98.0%) | 0/250 (0.0%) | 647/1,250 (51.8%) |
| **Mean Lumber Huts Built** | 11.048 | 10.490 | 10.920 | 10.327 | 0.000 | 0.843 |
| **Organization Adoption Rate** | 250/250 (100.0%) | 1,216/1,250 (97.3%) | 250/250 (100.0%) | 1,208/1,250 (96.6%) | 0/250 (0.0%) | 998/1,250 (79.8%) |
| **Mean Fruit Harvested** | 8.568 | 8.440 | 8.652 | 8.333 | 0.000 | 2.474 |
| **Mean Animals Harvested** | 4.412 | 4.409 | 4.384 | 4.375 | 3.916 | 1.048 |
| **Mean Sawmills Built** | 3.296 | 2.173 | 3.312 | 2.198 | 0.000 | 0.010 |
| **Mean Techs Researched** | 3.960 | 4.344 | 3.944 | 4.320 | 0.000 | 1.835 |

---

## Rigorous v3 $\rightarrow$ v4 Paired Comparison

Machine-readable comparison artifacts are saved at `outputs/comparisons/phase1_v3_terminal_spt_vs_v4_parity001.json` and `.csv`.

Because both references were evaluated across the exact same 250 validation and 250 test maps with identical evaluation RNG seeds, map-level deltas ($\Delta = \text{v4} - \text{v3}$) are paired:

### Primary Policy Paired Statistics

| Split | Policy | v3 Mean SPT | v4 Mean SPT | Paired Mean Δ | Paired 95% CI for Δ | Median Δ | Std Dev Δ | Std Err | v4 W / T / L | v4 Win Rate |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| **Validation** | **PPO argmax** | 19.872 | **19.784** | **$-0.088$** | **[$-0.544$, $+0.388$]** | $+0.000$ | 3.750 | 0.237 | 103 / 31 / 116 | 41.2% |
| **Validation** | **PPO sampled** | 19.533 | **19.471** | **$-0.062$** | **[$-0.304$, $+0.187$]** | $+0.000$ | 1.968 | 0.125 | 113 / 13 / 124 | 45.2% |
| **Test** | **PPO argmax** | 19.344 | **19.776** | **$+0.432$** | **[$+0.004$, $+0.840$]** | $+1.000$ | 3.351 | 0.212 | 128 / 37 / 85 | 51.2% |
| **Test** | **PPO sampled** | 18.956 | **19.255** | **$+0.299$** | **[$+0.062$, $+0.528$]** | $+0.400$ | 1.884 | 0.119 | 139 / 12 / 99 | 55.6% |

### Baseline Sanity Checks

- **Visible greedy:** Identical across versions on validation ($7.932$ vs $7.932$, $\Delta = 0.000$) and test ($7.956$ vs $7.956$, $\Delta = 0.000$).
- **Random legal:** Identical across versions on validation ($6.668$ vs $6.668$, $\Delta = 0.000$) and test ($6.702$ vs $6.702$, $\Delta = 0.000$).

### Validation $\rightarrow$ Held-Out Test Generalization Shift

| Policy | v3 Val Mean | v3 Test Mean | v3 Shift (Test $-$ Val) | v4 Val Mean | v4 Test Mean | v4 Shift (Test $-$ Val) |
|---|---:|---:|---:|---:|---:|---:|
| **PPO argmax** | 19.872 | 19.344 | **$-0.528$ SPT** | 19.784 | 19.776 | **$-0.008$ SPT** |
| **PPO sampled** | 19.533 | 18.956 | **$-0.577$ SPT** | 19.471 | 19.255 | **$-0.216$ SPT** |

### Key Behavioral Differences Between v3 and v4

1. **Sawmill Infrastructure Investment:** Under deterministic argmax, v4 constructs a mean of **3.31 sawmills per map** on test (compared to only 0.04 in v3), indicating that exact per-city population and level awareness enabled the policy to recognize the high multiplicative value of central Sawmills adjacent to multiple Lumber Huts.
2. **Forestry Adoption Reaches 100%:** Deterministic Forestry adoption reached **250 / 250 maps (100.0%)** under v4 (up from 248 / 250 in v3).
3. **Generalization Flatness:** While v3 exhibited a modest drop of $-0.53$ SPT from validation to test, v4 deterministic argmax performance remained essentially flat ($19.784 \rightarrow 19.776$, $-0.008$ SPT).

---

## Scientific Interpretation & Limitations

1. **Single-Seed Scope:** This run is a single-training-seed reference experiment (Seed 3). While the paired test difference strictly excludes zero ($+0.432$ mean paired Δ, 95% CI [$+0.004$, $+0.840$]), these single-seed comparisons do not establish that the observation change caused the difference. Establishing multi-seed variance remains an active research target.
2. **Held-Out Test Pool Semantics:** The 250-map test pool is a fixed held-out benchmark. Because it was evaluated in earlier reference runs, it must not be described as newly pristine evidence.
3. **Phase 1 Remains Under Active Optimization:** Phase 1 is **not complete**. This run establishes the reference benchmark that subsequent models must beat.
4. **Post-Freeze Test Protocol Rule:**
   > After this freeze, v4 held-out-test results must not be used as a selection criterion for the next Phase 1 model. Develop and select subsequent models using training + validation evidence. The fixed test benchmark may be run after a new candidate is frozen, but because the pool has historical benchmark exposure it remains a fixed held-out benchmark rather than newly pristine evidence.

---

## Reproduction Commands

### 1. Verify Dataset Split
```powershell
python tools/split_phase1_map_pool.py
```

### 2. Verify Live Environment Contract (11×11, 586-d)
```powershell
$env:POLYVISION_SOLO_NO_OPPONENT_MODE = '1'
python tools/validate_environment_contract.py `
    --level-pool-glob 'levels/phase1_pool_bardur_real/validation/*.csv' `
    --expected-width 11 --expected-height 11
```

### 3. Canonical Validation Evaluation
```powershell
python tools/evaluate_phase1.py `
    --model-path 'runs/Tribes-v0__Phase1-Scientific-Train-V4-PARITY001-Seed3-TerminalSPT__3__1787705342/model_checkpoint_16000000.cleanrl_model' `
    --pool validation `
    --suite full `
    --repeats-per-map 5 `
    --seed 42
```

### 4. Canonical Fixed Held-Out Test Evaluation
```powershell
python tools/evaluate_phase1.py `
    --model-path 'runs/Tribes-v0__Phase1-Scientific-Train-V4-PARITY001-Seed3-TerminalSPT__3__1787705342/model_checkpoint_16000000.cleanrl_model' `
    --pool test `
    --confirm-test `
    --suite full `
    --repeats-per-map 5 `
    --seed 42
```

---

## Artifact Inventory & Checksums

| Artifact Path | Description | Byte Size | SHA-256 |
|---|---|---:|---|
| `runs/.../model_checkpoint_16000000.cleanrl_model` | Canonical frozen checkpoint | 16,754,668 | `bda59c9eb4603734c3ed599922174890c5b5a1d4a4fad3fd998c289dd3f53574` |
| `runs/.../model_checkpoint_16000000.cleanrl_model.action_interface.json` | Checkpoint sidecar | 716 | `b490bcaf686394d8ee0050a05b10a708ad2f1b8726eca11593036bc265f6d752` |
| `runs/.../Phase1-Scientific-Train-V4-PARITY001-Seed3-TerminalSPT.cleanrl_model` | Duplicate final model | 16,755,521 | `26932969a0839773804104055d7dc31d29a9ffa1e7179e9db0ec9c5646440f6a` |
| `outputs/training/phase1_v4_parity001_seed3_16m_terminal_spt_wandb.csv` | Archived W&B raw export | 26,938 | `7e2d29ed77f6efc18db51a2783e7911883ad3b0fb727fba3487322bea3861d7b` |
| `outputs/training/phase1_v4_parity001_seed3_16m_terminal_spt_summary.json` | Derived training summary | 5,420 | Verified |
| `outputs/comparisons/phase1_v3_terminal_spt_vs_v4_parity001.json` | v3 vs v4 comparison JSON | 6,480 | Verified |
| `outputs/comparisons/phase1_v3_terminal_spt_vs_v4_parity001.csv` | v3 vs v4 comparison CSV | 1,180 | Verified |
| `outputs/evaluations/20260826_..._validation_canonical/manifest.json` | Validation evaluation manifest | 820 | Verified |
| `outputs/evaluations/20260826_..._pristine_test/manifest.json` | Held-out test evaluation manifest | 820 | Verified |
