# Phase 1 v5 PARITY002 Seed3 16M Terminal-SPT Frozen Reference Benchmark Run Card

## Overview and Status

- **Status:** CURRENT FROZEN REFERENCE BENCHMARK
- **Date:** 2026-08-27
- **Disposition:** Active frozen reference benchmark for ongoing PolyVision Phase 1 development and optimization. (Phase 1 remains under active development; this run establishes the authoritative reference baseline to beat.)
- **Primary Capability Metric:** Final Turn-10 stars per turn (SPT) on held-out maps
- **Previous Reference:** Supersedes [Phase 1 v4 PARITY001 Seed3 16M Terminal-SPT Reference Run](Phase1_V4_PARITY001_Seed3_16M_TerminalSPT_Reference_Run.md) (19.78 test argmax SPT)

> [!IMPORTANT]
> **Preferred Scientific Headline:**
> "On the fixed 250-map held-out Phase 1 test benchmark, deterministic PPO (`ppo_argmax`) under the `v5_human_information_parity` observation contract achieved **20.17 mean Turn-10 SPT** (95% CI [19.67, 20.67], median 20.00), versus **7.96 SPT** for the policy-visible greedy baseline. PPO beat visible greedy on all 250 maps (250 W / 0 T / 0 L), with a mean paired advantage of **+12.22 SPT**."

> [!CAUTION]
> **Single-Seed Scope & Statistical Interpretation:**
> This reference run represents a **single-training-seed result (Seed 3)**. While the v5 reference achieves higher mean performance than the v4 reference on both validation (+0.732 paired Δ, 95% CI [+0.152, +1.344]) and held-out test (+0.396 paired Δ, 95% CI [$-0.136$, $+0.944$]), the paired 95% confidence interval on the held-out test split crosses zero. Furthermore, this single-seed comparison characterizes reference performance across the fixed map pools but does **not** constitute causal proof that individual PARITY-002 representation features caused the performance change, as no controlled single-feature ablations were trained. Establishing multi-seed variance and confidence bounds across multiple independent training seeds remains an active milestone.

---

## Authoritative Run Identity & Provenance

All parameters, hashes, and byte sizes below are derived directly from the canonical machine-readable run artifacts, checkpoint sidecars, and evaluation outputs:

| Property | Value |
|---|---|
| **Run directory** | `runs/Tribes-v0__Phase1-Scientific-Train-V5-PARITY002-Seed3-TerminalSPT__3__1787788415` |
| **Canonical frozen checkpoint** | `runs/Tribes-v0__Phase1-Scientific-Train-V5-PARITY002-Seed3-TerminalSPT__3__1787788415/model_checkpoint_16000000.cleanrl_model` |
| **Checkpoint SHA-256** | `924d4603fa5038b3ca11081cdfcb5c7a00063949dab44f50f05989e5c9dab061` |
| **Checkpoint size** | 19,744,556 bytes |
| **Separately saved final model file** | `runs/Tribes-v0__Phase1-Scientific-Train-V5-PARITY002-Seed3-TerminalSPT__3__1787788415/Phase1-Scientific-Train-V5-PARITY002-Seed3-TerminalSPT.cleanrl_model` (SHA-256: `022f45f6a648238f46126ac6a6a573b91164188a4b1a73622220a46a679b4d9a`, 19,745,409 bytes) |
| **Action interface sidecar** | `model_checkpoint_16000000.cleanrl_model.action_interface.json` |
| **Sidecar SHA-256** | `06f25252f960717b60c72178819d755fcd85d3bcc83d8290bcce0f7626517dc0` |
| **Sidecar size** | 715 bytes |
| **Training seed** | `3` |
| **Total environment steps** | 16,000,000 / 16,000,000 |
| **Final training SPS** | 577 |
| **Training commit SHA** | `977ad353ef4d8c95fea05eea905bd49e3c71db93` |
| **Git working tree status during training** | Clean |
| **Environment version** | `v5_human_information_parity` |
| **Observation dimension** | 6,424 (52 spatial channels $\times$ 121 cells = 6,292 spatial values + 132 scalar/structured values) |
| **Opening version** | `v2_guaranteed_two_unit` |
| **Actor mode** | `legal_features` |
| **Legal action feature version** | `v1_4_parity_spatial_and_cost` |
| **Legal action feature dimension** | 47 |
| **Max legal action slots** | 256 |
| **Global action catalog size** | 63,913 |
| **Catalog version** | `flat-v1` |
| **Canonicalizer version** | `flat-v1-structured` |
| **Catalog fingerprint** | `c849a4abf7b0bee073ccc56b63ae65917ea30e77068ad648c472130693dfe6e4` |

> [!NOTE]
> The canonical frozen model artifact is `model_checkpoint_16000000.cleanrl_model`. The separately saved `Phase1-Scientific-Train-V5-PARITY002-Seed3-TerminalSPT.cleanrl_model` file is a secondary final model artifact saved at run completion and is recorded for provenance verification only.

### Runtime & Dependency Versions

| Component | Version |
|---|---|
| **Python** | `3.11.9` |
| **PyTorch** | `2.11.0+cpu` (evaluation) / CUDA enabled during training (`CUDA 13.3`, Ada architecture) |
| **Gymnasium** | `1.3.0` |
| **Py4J** | `0.10.9.9` |
| **Java / JDK** | OpenJDK 21.0.8 (Temurin-21.0.8+9) |
| **Platform** | Windows-10-10.0.22621-SP0 |

---

## Environment & Observation Contract (`PARITY-002`)

### PARITY-002 Human Information Parity Interface

The `v5_human_information_parity` contract resolves all confirmed information disparities identified during the Human–AI Information Parity Audit. The observation vector is expanded from 586 to **6,424 total dimensions** ($52 \times 121 + 132 = 6,424$):

$$\text{Total Dimension} = 6,292 \text{ (spatial planes)} + 6 \text{ (legacy scalars)} + 12 \text{ (economy scalars)} + 24 \text{ (tech researched)} + 90 \text{ (exact per-city block)} = 6,424$$

### 1. Spatial Channel Layout (52 Channels $\times$ 121 Cells = 6,292 Values)
- **Channels 0..3 (Terrain & Fog):** Terrain categorical index (0..7), Fog binary mask, In-bounds mask, Territory ownership mask.
- **Channels 4..22 (19 Supported Building Types):** Binary presence planes in authoritative Java enum order (`PORT`, `MINE`, `FORGE`, `FARM`, `WINDMILL`, `CUSTOMS_HOUSE`, `LUMBER_HUT`, `SAWMILL`, `TEMPLE`, `WATER_TEMPLE`, `FOREST_TEMPLE`, `MOUNTAIN_TEMPLE`, `ALTAR_OF_PEACE`, `EMPERORS_TOMB`, `EYE_OF_GOD`, `GATE_OF_POWER`, `GRAND_BAZAR`, `PARK_OF_FORTUNE`, `TOWER_OF_WISDOM`).
- **Channel 23 (Road Grid):** Binary road network connectivity plane.
- **Channels 24..35 (12 Supported Unit Types):** Categorical unit presence planes (`WARRIOR`, `RIDER`, `DEFENDER`, `SWORDMAN`, `ARCHER`, `CATAPULT`, `KNIGHT`, `MIND_BENDER`, `BOAT`, `SHIP`, `BATTLESHIP`, `SUPERUNIT`).
- **Channels 36..42 (Unit Status & Resources):** Unit turn status (`FRESH` vs `MOVED`), Unit HP (normalized), Resource animal presence, Resource fruit presence, Forest presence, Neutral village presence, Ruin presence.
- **Channels 43..51 (9 Unit Home-City Slot Associations):** Binary planes indicating which deterministic city slot ($0 \dots 8$) supports the visible owned unit at each cell.

### 2. Scalar and Tech Vectors (42 Values)
- **Legacy Scalars (6):** Turn count, Current stars, Total SPT, Owned city count, Controlled unit count, Techs researched count.
- **Economy Scalars (12):** Harvested animals, Harvested fruit, Forests cleared, Lumber huts built, Sawmills built, Farms built, Windmills built, Mines built, Forges built, Ports built, Customs houses built, Monuments built.
- **Tech Tree Researched Vector (24):** Full 24-element binary vector for all technologies in `TECHNOLOGY_ORDER`.

### 3. Exact Per-City Block (90 Values: 9 Slots $\times$ 10 Features)
For each owned city slot $i \in \{0 \dots 8\}$ (sorted deterministically by $(x, y)$ ascending):
`city_present`, `city_x`, `city_y`, `city_level`, `city_population`, `city_population_need`, `city_production`, `city_supported_unit_count`, `city_unit_capacity`, `city_is_capital`. Unused slots are zero-padded. Raw actor IDs are completely excluded.

### 4. Legal Action Features (47 Dimensions)
- **Features 0..41:** Action family indicators, semantic economy tags, spatial delta features (`v1_3_move_focus_plus_semantic_econ`).
- **Feature 42:** `action_star_cost_norm` (`star_cost / 50.0`), reflecting dynamic technology scaling ($4 + \text{tier} \times N_{\text{cities}}$) and static unit/building costs.
- **Features 43..46:** Normalized spatial coordinates (`src_x/10`, `src_y/10`, `target_x/10`, `target_y/10`).

---

## Training Configuration & Hyperparameters

### Launch Command

```powershell
cd C:\PolyVision; $env:POLYVISION_LEVEL_POOL_GLOB='levels/phase1_pool_bardur_real/train/*.csv'; $env:POLYVISION_SOLO_NO_OPPONENT_MODE='1'; $env:POLYVISION_INFO_MODE='fast'; $env:POLYVISION_BATCH_LEGAL_ACTION_FETCH='1'; $env:POLYVISION_DERIVE_OBS_METADATA='1'; $env:POLYVISION_TERMINAL_SPT_REWARD_ENABLED='1'; python py_rl/cleanrl/cleanrl/ppo.py --exp-name Phase1-Scientific-Train-V5-PARITY002-Seed3-TerminalSPT --seed 3 --actor-mode legal_features --total-timesteps 16000000 --num-envs 20 --num-steps 128 --max-legal-actions 256 --legal-action-feature-dim 47 --enable-step-diagnostics --step-diagnostics-log-every 3 --track --wandb-project-name cleanRL --save-model --save-frequency 500000 --force-revalidate-action-interface --validation-states 10000
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

## Dataset / Map Split Verification & Fixed Benchmark Pools

The dataset split is fixed across all 5,517 maps (`split_manifest.json` SHA-256: `c8721cd0fcb636d9ddb483745585e5b69eca9ca501950a1ddec6a51d0c763690`):

| Split | Count | Aggregate Pool Identity (SHA-256) | Role in Workflow |
|---|---:|---|---|
| **Train** | 5,000 | `a99b309a3020704e4a95886d5ab3f8b6f7ed00931ee111e3117702009605e8c4` | Training rollouts and PPO optimization |
| **Validation** | 250 | `a56e74c952ad3b08d4645fc16d66ce0f17e0589fea2fe9042f7cd293669eac72` | Checkpoint selection, intermediate evaluations |
| **Test (Held-Out)** | 250 | `8a8e0f784535e8169ffb829f2e3058b60bcd7493277912dd995ff17b217f4b94` | Frozen post-training held-out benchmark |
| **Human Benchmark** | 17 | `adbe5a707392aad15342f3c595a78abeeb85cb0ce77b91a8d9339276a2674a69` | Controlled Human–AI comparison games |

---

## Run-Level W&B Training Telemetry Summary

The archived W&B artifact (`outputs/training/phase1_v5_parity002_seed3_16m_terminal_spt_wandb.csv`, SHA-256: `df75e8ef8c9d09cce6d2fbbf366ea20835f8c6ebae06c64188b839818816c7cb`) is a **runs-table export** containing run-level configuration and summary metrics. Structured derived provenance is recorded in `outputs/training/phase1_v5_parity002_seed3_16m_terminal_spt_summary.json`.

| Metric Category | Summary Metric | Final Recorded Value | Description |
|---|---|---:|---|
| **Throughput** | Training SPS | 577 | Final recorded system transitions per second |
| | Total Runtime | 27,814 s (7.73 h) | Total wall-clock time over 16M steps |
| **Optimization** | Value loss | 45.760 | Final recorded clipped value function loss metric |
| | Policy loss | $-3.73 \times 10^{-8}$ | Final recorded surrogate policy loss metric |
| | Policy entropy | 1.317 | Final recorded policy action distribution entropy |
| | Explained variance | 0.837 | PPO value-function explained variance ($1 - \text{Var}(y - \hat{y}) / \text{Var}(y)$) |
| | Approx KL | $-1.05 \times 10^{-9}$ | Final recorded approximate KL divergence metric |
| | Learning rate | $4.00 \times 10^{-8}$ | Final annealed learning rate |
| **Training Rollouts** | Rollout final SPT (T10) | 19.0 | Final recorded episode-end SPT during training rollouts |
| | Rollout final cities | 5.0 | Final recorded mean cities at Turn 10 during rollouts |
| | Forestry adoption rate | 100.0% | Final recorded fraction of rollout episodes researching Forestry |
| | Organization adoption rate | 100.0% | Final recorded fraction of rollout episodes researching Organization |
| | Mean lumber huts | 9.0 | Final recorded mean lumber huts built by Turn 10 |
| | Mean sawmills | 0.0 | Final recorded mean sawmills built by Turn 10 |
| | Non-endturn action rate | 90.4% | Final recorded proportion of productive non-end-turn actions |
| | Mean valid legal actions | 21.91 | Final recorded average legal action slots per decision state |

---

## Canonical Validation Evaluation

- **Directory:** `outputs/evaluations/20260827_phase1_v5_parity002_seed3_16m_terminal_spt_validation_canonical`
- **Maps:** 250 held-out validation maps
- **Suite:** Canonical full suite (3,000 episodes: 250 argmax, 1,250 sampled [5 reps], 250 visible greedy, 1,250 random legal [5 reps])
- **Bootstrap Samples:** 5,000 per-map replicate mean bootstrap samples (seed 42)

### Policy Performance (Turn-10 SPT)

| Policy | Episodes | Mean SPT | Median SPT | Std Dev | Min | Max | p05 | p25 | p75 | p95 | 95% Confidence Interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **PPO argmax** | 250 | **20.516** | **21.00** | 4.18 | 10.0 | 30.0 | 13.0 | 17.0 | 24.0 | 27.0 | **[19.98, 21.04]** |
| **PPO sampled** | 1,250 | **19.530** | **19.60** | 2.71 | 10.8 | 26.6 | 15.0 | 17.6 | 21.55 | 23.8 | **[19.19, 19.88]** |
| **Visible greedy** | 250 | **7.932** | **8.00** | 1.25 | 5.0 | 12.0 | 6.0 | 7.0 | 9.0 | 10.0 | **[7.78, 8.09]** |
| **Random legal** | 1,250 | **6.668** | **6.60** | 0.69 | 4.6 | 8.6 | 5.4 | 6.2 | 7.2 | 7.6 | **[6.58, 6.75]** |

### Paired Comparisons (Validation)

- **PPO argmax vs. Visible greedy:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+12.584 SPT** (95% CI [12.092, 13.108], median +13.0)
- **PPO sampled vs. Random legal:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+12.862 SPT** (95% CI [12.550, 13.184], median +13.0)
- **PPO argmax vs. Random legal:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+13.848 SPT** (95% CI [13.334, 14.364], median +13.8)

---

## Canonical Fixed Held-Out Test Evaluation

- **Directory:** `outputs/evaluations/20260827_phase1_v5_parity002_seed3_16m_terminal_spt_pristine_test`
- **Maps:** 250 fixed held-out test maps
- **Suite:** Canonical full suite (3,000 episodes total)
- **Protocol Note:** `--confirm-test` was explicitly supplied; weights were strictly frozen prior to evaluation.

### Policy Performance (Turn-10 SPT)

| Policy | Episodes | Mean SPT | Median SPT | Std Dev | Min | Max | p05 | p25 | p75 | p95 | 95% Confidence Interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **PPO argmax** | 250 | **20.172** | **20.00** | 3.96 | 9.0 | 33.0 | 13.0 | 17.0 | 23.0 | 25.55 | **[19.67, 20.67]** |
| **PPO sampled** | 1,250 | **19.361** | **19.40** | 2.68 | 11.0 | 27.4 | 15.2 | 17.6 | 21.0 | 23.91 | **[19.04, 19.69]** |
| **Visible greedy** | 250 | **7.956** | **8.00** | 1.17 | 5.0 | 11.0 | 6.0 | 7.0 | 9.0 | 10.0 | **[7.82, 8.10]** |
| **Random legal** | 1,250 | **6.702** | **6.60** | 0.68 | 4.8 | 8.4 | 5.6 | 6.25 | 7.2 | 7.91 | **[6.62, 6.79]** |

### Paired Comparisons (Held-Out Test)

- **PPO argmax vs. Visible greedy:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+12.216 SPT** (95% CI [11.728, 12.684], median +12.0)
- **PPO sampled vs. Random legal:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+12.659 SPT** (95% CI [12.328, 12.974], median +12.6)
- **PPO argmax vs. Random legal:** 250 W / 0 T / 0 L (100.0% win rate), mean advantage **+13.470 SPT** (95% CI [13.013, 13.959], median +13.4)

---

## Secondary Behavioral Diagnostics (Validation & Test)

| Metric | Validation (v5 Argmax) | Validation (v5 Sampled) | Test (v5 Argmax) | Test (v5 Sampled) | Test (v5 Greedy) | Test (v5 Random) |
|---|---:|---:|---:|---:|---:|---:|
| **Mean Final Cities** | 4.976 | 4.948 | 4.900 | 4.898 | 3.804 | 1.999 |
| **Mean Final Units** | 13.172 | 11.630 | 13.292 | 11.611 | 2.000 | 5.480 |
| **Mean Unspent Stars** | 32.504 | 27.030 | 31.428 | 27.635 | 53.908 | 9.706 |
| **Mean Fog Cleared** | 63.608 | 64.166 | 61.908 | 63.478 | 37.428 | 30.031 |
| **Forestry Adoption Rate** | 249/250 (99.6%) | 1,158/1,250 (92.6%) | 250/250 (100.0%) | 1,147/1,250 (91.8%) | 0/250 (0.0%) | 647/1,250 (51.8%) |
| **Mean Lumber Huts Built** | 12.948 | 10.763 | 12.788 | 10.569 | 0.000 | 0.843 |
| **Organization Adoption Rate** | 249/250 (99.6%) | 1,197/1,250 (95.8%) | 249/250 (99.6%) | 1,200/1,250 (96.0%) | 0/250 (0.0%) | 998/1,250 (79.8%) |
| **Mean Fruit Harvested** | 9.904 | 9.522 | 9.952 | 9.600 | 0.000 | 2.474 |
| **Mean Animals Harvested** | 5.336 | 4.850 | 5.384 | 4.804 | 3.916 | 1.048 |
| **Mean Sawmills Built** | 0.004 | 0.176 | 0.000 | 0.152 | 0.000 | 0.081 |
| **Mean Techs Researched** | 3.024 | 3.866 | 3.020 | 3.854 | 1.000 | 6.214 |

---

## Rigorous v4 $\rightarrow$ v5 Paired Comparison

Machine-readable comparison artifacts are saved at `outputs/comparisons/phase1_v4_terminal_spt_vs_v5_parity002.json` and `.csv`.

Because both references were evaluated across the exact same 250 validation and 250 test maps with identical evaluation RNG seeds, map-level deltas ($\Delta = \text{v5} - \text{v4}$) are paired:

### Primary Policy Paired Statistics

| Split | Policy | v4 Mean SPT | v5 Mean SPT | Paired Mean Δ | Paired 95% CI for Δ | Median Δ | Std Dev Δ | Std Err | v5 W / T / L | v5 Win Rate |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| **Validation** | **PPO argmax** | 19.784 | **20.516** | **$+0.732$** | **[$+0.152$, $+1.344$]** | $+0.000$ | 4.777 | 0.302 | 118 / 24 / 108 | 47.2% |
| **Validation** | **PPO sampled** | 19.471 | **19.530** | **$+0.059$** | **[$-0.223$, $+0.327$]** | $+0.000$ | 2.208 | 0.140 | 123 / 9 / 118 | 49.2% |
| **Test** | **PPO argmax** | 19.776 | **20.172** | **$+0.396$** | **[$-0.136$, $+0.944$]** | $+0.000$ | 4.293 | 0.272 | 120 / 30 / 100 | 48.0% |
| **Test** | **PPO sampled** | 19.255 | **19.361** | **$+0.106$** | **[$-0.158$, $+0.378$]** | $+0.200$ | 2.130 | 0.135 | 130 / 10 / 110 | 52.0% |

### Baseline Sanity Checks

- **Visible greedy:** Identical across versions on validation ($7.932$ vs $7.932$, $\Delta = 0.000$) and test ($7.956$ vs $7.956$, $\Delta = 0.000$).
- **Random legal:** Identical across versions on validation ($6.668$ vs $6.668$, $\Delta = 0.000$) and test ($6.702$ vs $6.702$, $\Delta = 0.000$).

### Validation $\rightarrow$ Held-Out Test Generalization Shift

| Policy | v4 Val Mean | v4 Test Mean | v4 Shift (Test $-$ Val) | v5 Val Mean | v5 Test Mean | v5 Shift (Test $-$ Val) |
|---|---:|---:|---:|---:|---:|---:|
| **PPO argmax** | 19.784 | 19.776 | **$-0.008$ SPT** | 20.516 | 20.172 | **$-0.344$ SPT** |
| **PPO sampled** | 19.471 | 19.255 | **$-0.216$ SPT** | 19.530 | 19.361 | **$-0.169$ SPT** |

### Key Behavioral Differences Between v4 and v5

1. **Territorial Expansion and Unit Production:** Under deterministic argmax, v5 test episodes show a higher mean city count (mean **4.90** vs 4.23 in v4), higher military unit counts (**13.29** vs 6.67), and increased fog exploration (**61.91** vs 49.62 tiles).
2. **Resource Harvesting Focus:** v5 recorded higher mean fruit harvesting (**9.95** vs 8.65) and animal harvesting (**5.38** vs 4.38), and constructed more lumber huts (**12.79** vs 10.92).
3. **Infrastructure Strategy Shift:** Whereas v4 invested in central Sawmills (3.31 per map), v5 opted for direct population growth across more cities without building sawmills (0.00 per map on test), demonstrating that different viable macroeconomic strategies exist within Phase 1.
4. **Generalization Stability:** v5 exhibits modest validation-to-test degradation ($-0.344$ SPT for argmax, $-0.169$ SPT for sampled), confirming robust generalization across unseen map seeds.

---

## Scientific Interpretation & Limitations

1. **Single-Seed Scope:** This run is a single-training-seed reference experiment (Seed 3). While the v5 reference sets new state-of-the-art headline performance on both splits, the paired 95% confidence interval on the held-out test split crosses zero ([$ -0.136$, $+0.944$]). Multi-seed replication across independent training seeds remains necessary to establish statistical significance across training initialization variance.
2. **No Causal Claim for Individual Parity Features:** The v5 run incorporated multiple interface modifications simultaneously (road plane, 19 building channels, 12 unit channels, 9 unit home-city channels, full 24-tech vector, dynamic action star cost, spatial action coordinates). Because no controlled single-feature ablation runs were executed, this result demonstrates that the comprehensive v5 interface trains effectively and outperforms v4, but does **not** prove which individual feature drove the performance delta.
3. **Phase 1 Benchmark Scope:** Phase 1 remains a Turn-10 solo economic opening benchmark on 11×11 maps. It does not represent full 30-turn games, multi-tribe combat, tech tree mastery beyond early tiers, or diplomacy.
4. **Post-Freeze Test Protocol Rule:**
   > After this freeze, v5 held-out-test results must not be used as a selection criterion for the next Phase 1 model. Develop and select subsequent models using training + validation evidence only. The fixed test benchmark may be run after a new candidate is frozen.

---

## Reproduction Commands

### 1. Verify Dataset Split
```powershell
python tools/split_phase1_map_pool.py
```

### 2. Verify Live Environment Contract (11×11, 6,424-d)
```powershell
$env:POLYVISION_SOLO_NO_OPPONENT_MODE = '1'
python tools/validate_environment_contract.py `
    --level-pool-glob 'levels/phase1_pool_bardur_real/validation/*.csv' `
    --expected-width 11 --expected-height 11
```

### 3. Canonical Validation Evaluation
```powershell
python tools/evaluate_phase1.py `
    --model-path 'runs/Tribes-v0__Phase1-Scientific-Train-V5-PARITY002-Seed3-TerminalSPT__3__1787788415/model_checkpoint_16000000.cleanrl_model' `
    --pool validation `
    --suite full `
    --repeats-per-map 5 `
    --seed 42
```

### 4. Canonical Fixed Held-Out Test Evaluation
```powershell
python tools/evaluate_phase1.py `
    --model-path 'runs/Tribes-v0__Phase1-Scientific-Train-V5-PARITY002-Seed3-TerminalSPT__3__1787788415/model_checkpoint_16000000.cleanrl_model' `
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
| `runs/.../model_checkpoint_16000000.cleanrl_model` | Canonical frozen checkpoint | 19,744,556 | `924d4603fa5038b3ca11081cdfcb5c7a00063949dab44f50f05989e5c9dab061` |
| `runs/.../model_checkpoint_16000000.cleanrl_model.action_interface.json` | Checkpoint sidecar | 715 | `06f25252f960717b60c72178819d755fcd85d3bcc83d8290bcce0f7626517dc0` |
| `runs/.../Phase1-Scientific-Train-V5-PARITY002-Seed3-TerminalSPT.cleanrl_model` | Separately saved final model | 19,745,409 | `022f45f6a648238f46126ac6a6a573b91164188a4b1a73622220a46a679b4d9a` |
| `outputs/training/phase1_v5_parity002_seed3_16m_terminal_spt_wandb.csv` | Archived W&B runs export | 27,046 | `df75e8ef8c9d09cce6d2fbbf366ea20835f8c6ebae06c64188b839818816c7cb` |
| `outputs/training/phase1_v5_parity002_seed3_16m_terminal_spt_summary.json` | Derived training summary | 5,420 | Verified |
| `outputs/comparisons/phase1_v4_terminal_spt_vs_v5_parity002.json` | v4 vs v5 comparison JSON | 17,500 | Verified |
| `outputs/comparisons/phase1_v4_terminal_spt_vs_v5_parity002.csv` | v4 vs v5 comparison CSV | 1,106 | Verified |
| `outputs/evaluations/20260827_..._validation_canonical/manifest.json` | Validation evaluation manifest | 1,404 | `ac94ddb6ee86ee7a5df74d9e57be37ff11a7741639ae5703f8a427b08499252c` |
| `outputs/evaluations/20260827_..._pristine_test/manifest.json` | Held-out test evaluation manifest | 1,397 | `a00b0f0a4f5fbc57df076943a579bf6bb9f0d14878a87b8d4474737d97b5f903` |
