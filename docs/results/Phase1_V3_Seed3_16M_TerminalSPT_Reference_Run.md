# Phase 1 v3 Seed3 16M Terminal-SPT Frozen Reference Benchmark Run Card

## Overview and Status

- **Status:** CURRENT FROZEN REFERENCE BENCHMARK
- **Date:** 2026-08-25
- **Disposition:** Active frozen reference benchmark for PolyVision Phase 1 optimization (Phase 1 remains under active development; this run establishes the current benchmark to beat)
- **Primary Capability Metric:** Final Turn-10 stars per turn (SPT) on held-out maps
- **Previous Reference:** Supersedes the original [Phase 1 v3 Seed3 16M Reference Run](Phase1_V3_Seed3_16M_Reference_Run.md) (16.88 test argmax SPT)

> [!IMPORTANT]
> **Preferred Scientific Headline:**
> "On 250 pristine held-out test maps, deterministic PPO (`ppo_argmax`) trained with Terminal-SPT reward shaping achieved **19.34 mean Turn-10 SPT** (95% CI [18.95, 19.75], median 19.00), compared with **7.96 SPT** for the policy-visible greedy baseline. PPO beat visible greedy on all 250 maps (250 W / 0 T / 0 L), with a mean paired advantage of **+11.39 SPT**."

---

## Authoritative Run Identity & Provenance

All parameters and hashes below are derived directly from the canonical machine-readable run artifacts and checkpoint sidecars:

| Property | Value |
|---|---|
| **Run directory** | `runs/Tribes-v0__Phase1-Scientific-Train-V3-Seed3-TerminalSPT__3__1787602198` |
| **Canonical frozen checkpoint** | `runs/Tribes-v0__Phase1-Scientific-Train-V3-Seed3-TerminalSPT__3__1787602198/model_checkpoint_16000000.cleanrl_model` |
| **Checkpoint SHA-256** | `853bc4b1d60cf114a715da0ae7a253af357cfded78f5ef9cc7422182b976889a` |
| **Checkpoint size** | 16,713,196 bytes |
| **Duplicate final model file** | `runs/Tribes-v0__Phase1-Scientific-Train-V3-Seed3-TerminalSPT__3__1787602198/Phase1-Scientific-Train-V3-Seed3-TerminalSPT.cleanrl_model` (SHA-256: `23e296cf4d9c0bcff34eadf42d48aa742f41a26bafe49aa3b68966935b3f3afb`, 16,713,671 bytes) |
| **Action interface sidecar** | `model_checkpoint_16000000.cleanrl_model.action_interface.json` |
| **Sidecar SHA-256** | `0ce2b01464a47506c19bf5c34b4e39f294c992aea0607ab3755f15d5e10db736` |
| **Sidecar size** | 718 bytes |
| **Training seed** | `3` |
| **Total environment steps** | 16,000,000 |
| **Git commit SHA** | `d14c14fb3ac31b4d8632ffe6610ba69e005a653f` |
| **Git working tree status** | Clean during training |
| **Actor mode** | `legal_features` |
| **Observation dimension** | 505 |
| **Legal action feature version** | `v1_3_move_focus_plus_semantic_econ` |
| **Legal action feature dimension** | 42 |
| **Max legal action slots** | 256 |
| **Global action catalog size** | 63,913 |
| **Catalog version** | `flat-v1` |
| **Canonicalizer version** | `flat-v1-structured` |
| **Catalog fingerprint** | `c849a4abf7b0bee073ccc56b63ae65917ea30e77068ad648c472130693dfe6e4` |
| **Environment version** | `v3_corrected_turn_economy` |
| **Opening version** | `v2_guaranteed_two_unit` |

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

## Training Configuration & Hyperparameters

### Launch Command

```powershell
cd C:\PolyVision; $env:POLYVISION_LEVEL_POOL_GLOB='levels/phase1_pool_bardur_real/train/*.csv'; $env:POLYVISION_SOLO_NO_OPPONENT_MODE='1'; $env:POLYVISION_INFO_MODE='fast'; $env:POLYVISION_BATCH_LEGAL_ACTION_FETCH='1'; $env:POLYVISION_DERIVE_OBS_METADATA='1'; $env:POLYVISION_TERMINAL_SPT_REWARD_ENABLED='1'; python py_rl/cleanrl/cleanrl/ppo.py --exp-name Phase1-Scientific-Train-V3-Seed3-TerminalSPT --seed 3 --actor-mode legal_features --total-timesteps 16000000 --num-envs 20 --num-steps 128 --max-legal-actions 256 --legal-action-feature-dim 42 --enable-step-diagnostics --step-diagnostics-log-every 3 --track --wandb-project-name cleanRL --save-model --save-frequency 500000 --force-revalidate-action-interface --validation-states 10000
```

### Key Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| `learning_rate` | `0.00025` | Initial learning rate |
| `anneal_lr` | `True` | Linear learning rate decay over 16M steps |
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

---

## Reward Configuration (Terminal-SPT Experiment)

This run explicitly enabled the **Terminal-SPT reward bonus** (`POLYVISION_TERMINAL_SPT_REWARD_ENABLED=1`) on top of the standard Phase 1 dense step-level shaping terms.

> [!NOTE]
> **Historical Configuration Note:**
> At training time, Terminal-SPT was explicitly enabled via environment configuration, and `actor_mode=legal_features` was explicitly passed via CLI flags. Following the success of this experiment, Terminal-SPT was promoted to the standard Phase-1 default reward configuration and `legal_features` was promoted to the default PPO actor mode. The recorded metadata in the checkpoint sidecars remains historical provenance from this run.

### Exact Terminal Bonus Formulation

Upon Turn 10 completion, a non-stepwise terminal reward bonus is added to the transition reward:

$$\text{Terminal Reward} = w_{\text{base}} \times \text{final\_SPT} + w_{>10} \times \max(0, \text{final\_SPT} - 10) + w_{>15} \times \max(0, \text{final\_SPT} - 15)$$

The exact coefficient values used during training:
- **Base weight ($w_{\text{base}}$):** `1.0`
- **Over-10 weight ($w_{>10}$):** `2.0`
- **Over-15 weight ($w_{>15}$):** `3.0`

### Step-Level Dense Shaping Terms

- **Stepwise SPT change:** Positive change $\times 5.0$; nonpositive change $\times 1.0$
- **City capture bonus:** $+3.0$ to $+6.0$ per captured city
- **Village exploration & approach:** $+1.0$ for revealing uncaptured village, $+0.5$ for moving closer, $+1.0$ for stepping onto village, $+0.5$ breadcrumb
- **Village opportunity bonus/penalty:** $+5.0$ for moving onto visible neutral village, $-2.0$ for missing visible capture opportunity
- **Fog exploration:** $+0.08$ per newly cleared tile (capped at 5 tiles/step)
- **Useless move penalty:** $-0.35$ for zero-fog, non-village moves

> [!NOTE]
> Reward terms exist exclusively during training rollouts to shape the gradient objective. The agent's observation at inference/evaluation time does not receive or depend on reward values. Final Turn-10 SPT on held-out maps remains the definitive capability metric.

---

## Environment & Task Contract

- **Tribe:** Bardur (tribe 0)
- **Map Distribution:** Genuine Drylands maps, 11×11 grid
- **Horizon:** Turn 10 truncation (handoff at Turn 2 following scripted opening)
- **Scripted Opening:** `v2_guaranteed_two_unit` (two animals, Workshop, two original-warrior moves on Turn 1 excluding capital, mandatory second-warrior spawn, handoff at Turn 2 with 7 stars, 4 SPT, 1 city, 2 units)
- **Economy:** `v3_corrected_turn_economy` (authentic accumulating star economy via `gs.incTick()` and `gs.initTurn()`)
- **Action Interface:** Global catalog of 63,913 discrete action IDs; max 256 legal slots per decision state; 42-d legal action features
- **Policy Space Restrictions:** Combat attacks excluded from policy action space; Drylands Fishing masked at policy layer

---

## Frozen Split Definition

The dataset split is fixed and hash-verified across all 5,517 maps (`split_manifest.json` SHA-256: `c8721cd0fcb636d9ddb483745585e5b69eca9ca501950a1ddec6a51d0c763690`):

| Split | Count | Aggregate Pool Identity (SHA-256) |
|---|---:|---|
| **Train** | 5,000 | `a99b309a3020704e4a95886d5ab3f8b6f7ed00931ee111e3117702009605e8c4` |
| **Validation** | 250 | `a56e74c952ad3b08d4645fc16d66ce0f17e0589fea2fe9042f7cd293669eac72` |
| **Test (Pristine)** | 250 | `8a8e0f784535e8169ffb829f2e3058b60bcd7493277912dd995ff17b217f4b94` |
| **Human Benchmark** | 17 | `adbe5a707392aad15342f3c595a78abeeb85cb0ce77b91a8d9339276a2674a69` |

---

## Canonical Validation Evaluation

- **Evaluation Directory:** `outputs/evaluations/20260825_phase1_v3_seed3_16m_terminal_spt_validation_canonical`
- **Maps:** 250 held-out validation maps
- **Suite:** Full canonical suite (3,000 episodes total: 250 argmax, 1,250 sampled [5 repeats], 250 visible greedy, 1,250 random legal [5 repeats])
- **Bootstrap Samples:** 5,000 per-map replicate mean bootstrap samples

### Policy Performance (Final Turn-10 SPT)

| Policy | Episodes | Mean SPT | Median SPT | Std Dev | Min | Max | p05 | p25 | p75 | p95 | 95% Confidence Interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **PPO argmax** | 250 | **19.872** | **20.00** | 3.47 | 13.0 | 29.0 | 14.0 | 18.0 | 22.0 | 25.55 | **[19.46, 20.29]** |
| **PPO sampled** | 1,250 | **19.533** | **19.60** | 2.31 | 11.4 | 24.6 | 15.6 | 18.0 | 21.2 | 22.91 | **[19.25, 19.82]** |
| **Visible greedy** | 250 | **7.932** | **8.00** | 1.25 | 5.0 | 12.0 | 6.0 | 7.0 | 9.0 | 10.0 | **[7.78, 8.09]** |
| **Random legal** | 1,250 | **6.668** | **6.60** | 0.69 | 4.6 | 8.6 | 5.4 | 6.2 | 7.2 | 7.6 | **[6.58, 6.75]** |

### Paired Comparisons (Validation)

- **PPO argmax vs. Visible greedy:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+11.94 SPT** (95% CI [11.52, 12.36], median +12.0)
- **PPO sampled vs. Random legal:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+12.86 SPT** (95% CI [12.59, 13.14], median +12.8)
- **PPO argmax vs. Random legal:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+13.20 SPT** (95% CI [12.78, 13.62], median +13.2)

### Secondary Diagnostic Metrics (Validation)

| Metric | PPO Argmax | PPO Sampled | Visible Greedy | Random Legal |
|---|---:|---:|---:|---:|
| **Mean Final Cities** | 4.300 | 4.418 | 3.820 | 2.001 |
| **Mean Final Units** | 7.220 | 6.474 | 2.000 | 5.441 |
| **Mean Unspent Stars** | 21.796 | 21.554 | 53.836 | 9.551 |
| **Mean Fog Tiles Cleared** | 55.132 | 56.046 | 38.804 | 31.129 |
| **Forestry Adoption Rate** | 248 / 250 (99.2%) | 1,157 / 1,250 (92.6%) | 0 / 250 (0.0%) | 660 / 1,250 (52.8%) |
| **Mean Lumber Huts Built** | 11.032 | 9.966 | 0.000 | 0.867 |
| **Organization Adoption Rate** | 250 / 250 (100.0%) | 1,156 / 1,250 (92.5%) | 0 / 250 (0.0%) | 1,012 / 1,250 (81.0%) |
| **Mean Fruit Harvested** | 8.568 | 7.678 | 0.000 | 2.305 |
| **Mean Animals Harvested** | 4.468 | 4.607 | 3.864 | 1.090 |

---

## Canonical Pristine Test Evaluation

- **Evaluation Directory:** `outputs/evaluations/20260825_phase1_v3_seed3_16m_terminal_spt_pristine_test`
- **Maps:** 250 pristine held-out test maps
- **Suite:** Full canonical suite (3,000 episodes total)
- **Protocol Note:** Checkpoint weights and evaluation protocol were frozen prior to running this test evaluation.

### Policy Performance (Final Turn-10 SPT)

| Policy | Episodes | Mean SPT | Median SPT | Std Dev | Min | Max | p05 | p25 | p75 | p95 | 95% Confidence Interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **PPO argmax** | 250 | **19.344** | **19.00** | 3.30 | 12.0 | 29.0 | 14.0 | 17.0 | 21.0 | 25.0 | **[18.95, 19.75]** |
| **PPO sampled** | 1,250 | **18.956** | **19.00** | 2.55 | 11.8 | 24.6 | 14.8 | 17.2 | 20.6 | 23.31 | **[18.64, 19.27]** |
| **Visible greedy** | 250 | **7.956** | **8.00** | 1.17 | 5.0 | 11.0 | 6.0 | 7.0 | 9.0 | 10.0 | **[7.82, 8.10]** |
| **Random legal** | 1,250 | **6.702** | **6.60** | 0.68 | 4.8 | 8.4 | 5.6 | 6.25 | 7.2 | 7.91 | **[6.62, 6.79]** |

### Paired Comparisons (Pristine Test)

- **PPO argmax vs. Visible greedy:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+11.39 SPT** (95% CI [10.98, 11.79], median +11.0)
- **PPO sampled vs. Random legal:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+12.25 SPT** (95% CI [11.95, 12.55], median +12.2)
- **PPO argmax vs. Random legal:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+12.64 SPT** (95% CI [12.22, 13.05], median +12.6)

### Secondary Diagnostic Metrics (Pristine Test)

| Metric | PPO Argmax | PPO Sampled | Visible Greedy | Random Legal |
|---|---:|---:|---:|---:|
| **Mean Final Cities** | 4.208 | 4.300 | 3.804 | 1.999 |
| **Mean Final Units** | 7.368 | 6.486 | 2.000 | 5.480 |
| **Mean Unspent Stars** | 21.640 | 21.316 | 53.908 | 9.706 |
| **Mean Fog Tiles Cleared** | 53.324 | 54.769 | 37.428 | 30.031 |
| **Forestry Adoption Rate** | 248 / 250 (99.2%) | 1,168 / 1,250 (93.4%) | 0 / 250 (0.0%) | 647 / 1,250 (51.8%) |
| **Mean Lumber Huts Built** | 10.772 | 9.520 | 0.000 | 0.843 |
| **Organization Adoption Rate** | 250 / 250 (100.0%) | 1,182 / 1,250 (94.6%) | 0 / 250 (0.0%) | 998 / 1,250 (79.8%) |
| **Mean Fruit Harvested** | 8.416 | 7.757 | 0.000 | 2.474 |
| **Mean Animals Harvested** | 4.460 | 4.390 | 3.916 | 1.048 |

---

## Generalization & Historical Comparison

### Validation $\rightarrow$ Test Generalization Deltas

| Policy | Validation Mean SPT | Test Mean SPT | Generalization Delta |
|---|---:|---:|---:|
| **PPO argmax** | 19.872 | 19.344 | **$-0.528$ SPT** ($-0.53$) |
| **PPO sampled** | 19.533 | 18.956 | **$-0.577$ SPT** ($-0.58$) |
| **Visible greedy** | 7.932 | 7.956 | **$+0.024$ SPT** ($+0.02$) |
| **Random legal** | 6.668 | 6.702 | **$+0.034$ SPT** ($+0.03$) |

The small drop of $-0.53$ SPT from validation to pristine test confirms strong generalization to unseen genuine Polytopia maps with negligible overfitting.

### Comparison to Previous Frozen Reference Run

Comparison against the superseded [Phase 1 v3 Seed3 16M Reference Run](Phase1_V3_Seed3_16M_Reference_Run.md) (`Phase1-Scientific-Train-V3-Seed3.cleanrl_model`):

| Evaluation Split / Metric | Superseded Reference | New Terminal-SPT Reference | Delta |
|---|---:|---:|---:|
| **Validation PPO argmax** | 17.184 | **19.872** | **+2.688 SPT (+2.69)** |
| **Validation PPO sampled** | 17.704 | **19.533** | **+1.829 SPT (+1.83)** |
| **Validation Visible greedy** | 7.932 | 7.932 | 0.000 |
| **Validation Random legal** | 6.668 | 6.668 | 0.000 |
| **Test PPO argmax** | 16.884 | **19.344** | **+2.460 SPT (+2.46)** |
| **Test PPO sampled** | 17.548 | **18.956** | **+1.408 SPT (+1.41)** |
| **Test Visible greedy** | 7.956 | 7.956 | 0.000 |
| **Test Random legal** | 6.702 | 6.702 | 0.000 |

### Key Behavioral Observations

1. **Policy Mode Hierarchy Inversion:**
   - In the previous reference run, stochastic `ppo_sampled` outperformed deterministic `ppo_argmax` (val: 17.70 vs 17.18; test: 17.55 vs 16.88).
   - Under the Terminal-SPT policy, deterministic `ppo_argmax` systematically outperforms `ppo_sampled` (validation: 19.87 vs 19.53, $+0.34$ SPT; test: 19.34 vs 18.96, $+0.38$ SPT).
2. **Deterministic Tier-2 Tech Adoption:**
   - While the old reference policy exhibited 0.0% explicit Forestry adoption under argmax, the Terminal-SPT model researched Forestry on **248 / 250 maps (99.2%)** under deterministic argmax, building a mean of 10.77 lumber huts per map on pristine test.
3. **Capital & Unit Production Efficiency:**
   - Unspent stars at Turn 10 dropped from 41.06 to 21.64, and final unit counts dropped from 13.80 to 7.37, indicating that surplus stars were channeled into economic development (tech and infrastructure) rather than late-game warrior stockpiling.

---

## Scientific Interpretation, Reproducibility & Limitations

1. **Single-Seed Evidence Caveat:**
   This result is a single-seed training experiment (Seed 3). While it provides very strong evidence consistent with Terminal-SPT reward shaping materially improving learned policy quality, it is **not** a multi-seed statistical proof of causality. Establishing multi-seed variance and confidence bounds across multiple independent training seeds remains an active milestone.
2. **Environment & Task Contract Stability:**
   The environment contract (`v3_corrected_turn_economy`), scripted opening (`v2_guaranteed_two_unit`), action space (63,913 global IDs), observation layout (505 dimensions), and evaluation map pools were held strictly invariant between this run and the previous reference run.
3. **Phase 1 Remains Under Development:**
   Phase 1 is **not** solved. This run serves as the new active frozen reference benchmark against which subsequent architectural, algorithmic, and representation enhancements will be measured.
4. **Historical Record Preservation:**
   The previous reference run and its associated behavioral analysis remain preserved as valid historical evidence for the baseline reward configuration.
