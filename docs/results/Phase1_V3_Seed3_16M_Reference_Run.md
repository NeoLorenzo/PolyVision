# Phase 1 v3 Seed3 16M Frozen Reference Benchmark Run Card

## Overview and Status

- **Status:** Phase 1 v3 Seed3 16M frozen reference benchmark
- **Disposition:** Frozen reference benchmark while Phase 1 optimization continues (Phase 1 is **not** complete; this run establishes the baseline to beat)
- **Primary Metric:** Final Turn-10 stars per turn (SPT)

> [!IMPORTANT]
> **Preferred Scientific Headline:**
> "On 250 held-out test maps, deterministic PPO achieved 16.88 mean Turn-10 SPT (95% CI 16.46–17.31), compared with 7.96 for the policy-visible greedy baseline. PPO beat greedy on all 250 maps, with a mean advantage of +8.93 SPT."

## Authoritative Run Identity & Provenance

| Property | Value |
|---|---|
| **Run directory** | `runs/Tribes-v0__Phase1-Scientific-Train-V3-Seed3__3__1787526811` |
| **Final model file** | `Phase1-Scientific-Train-V3-Seed3.cleanrl_model` |
| **Model SHA-256** | `A6803CB00E08522819FDEB74BEDE8816CF131C1F28244C12D44F3EEFB9D2C234` |
| **Sidecar SHA-256** | `0CE2B01464A47506C19BF5C34B4E39F294C992AEA0607AB3755F15D5E10DB736` |
| **Training seed** | `3` |
| **Training timesteps** | 16,000,000 |
| **Training git commit** | `3bdabaa4f1de8236b5bc80f1f0fbadbd1905282e` |
| **Actor mode** | `legal_features` |
| **Environment version** | `v3_corrected_turn_economy` |
| **Opening version** | `v2_guaranteed_two_unit` |

## Training Command

```powershell
cd C:\PolyVision; $env:POLYVISION_LEVEL_POOL_GLOB='levels/phase1_pool_bardur_real/train/*.csv'; $env:POLYVISION_SOLO_NO_OPPONENT_MODE='1'; $env:POLYVISION_INFO_MODE='fast'; $env:POLYVISION_BATCH_LEGAL_ACTION_FETCH='1'; $env:POLYVISION_DERIVE_OBS_METADATA='1'; python py_rl/cleanrl/cleanrl/ppo.py --exp-name Phase1-Scientific-Train-V3-Seed3 --seed 3 --actor-mode legal_features --total-timesteps 16000000 --num-envs 20 --num-steps 128 --max-legal-actions 256 --legal-action-feature-dim 42 --enable-step-diagnostics --step-diagnostics-log-every 3 --track --wandb-project-name cleanRL --save-model --save-frequency 500000 --force-revalidate-action-interface --validation-states 10000
```

## Environment & Task Contract

- **Tribe:** Bardur (tribe 0)
- **Map distribution:** Genuine Drylands maps, 11×11 grid
- **Horizon:** Turn-10 truncation (handoff at Turn 2 following scripted opening)
- **Opening:** `v2_guaranteed_two_unit` (two animals, Workshop, two original-warrior moves excluding capital on Turn 1, mandatory second-warrior spawn on capital, handoff at Turn 2 with 7 stars, 4 SPT, 1 city, 2 units)
- **Economy:** `v3_corrected_turn_economy` (authentic accumulating star economy via `gs.incTick()` and `gs.initTurn()`)
- **Action space:** Global catalog of 63,913 stable discrete action IDs; max 256 legal slots per decision state
- **Action features:** 42-dimensional semantic/economic legal-action features
- **Observation:** 505-dimensional flattened vector
- **Policy restrictions:**
  - Combat attacks excluded from policy action space
  - Drylands Fishing research masked at the policy layer

## Frozen Split Definition

The dataset split is fixed and hash-verified across all 5,517 maps:

| Split | Count | Aggregate Pool Identity (SHA-256) |
|---|---:|---|
| **Train** | 5,000 | `a99b309a3020704e4a95886d5ab3f8b6f7ed00931ee111e3117702009605e8c4` |
| **Validation** | 250 | `a56e74c952ad3b08d4645fc16d66ce0f17e0589fea2fe9042f7cd293669eac72` |
| **Test (Pristine)** | 250 | `8a8e0f784535e8169ffb829f2e3058b60bcd7493277912dd995ff17b217f4b94` |
| **Human Benchmark** | 17 | `adbe5a707392aad15342f3c595a78abeeb85cb0ce77b91a8d9339276a2674a69` |

---

## Canonical Validation Evaluation

- **Evaluation Directory:** `outputs/evaluations/20260824_phase1_v3_seed3_16m_validation_canonical`
- **Maps:** 250 held-out validation maps
- **Suite:** Full canonical suite (3,000 episodes total)

### Policy Performance (Final Turn-10 SPT)

| Policy | Episodes | Mean SPT | Median SPT | 95% Confidence Interval |
|---|---:|---:|---:|---|
| **PPO argmax** | 250 | 17.18 | 17.00 | [16.72, 17.66] |
| **PPO sampled** | 1,250 | 17.70 | 17.60 | [17.37, 18.03] |
| **Visible greedy** | 250 | 7.93 | 8.00 | [7.78, 8.09] |
| **Random legal** | 1,250 | 6.67 | 6.60 | [6.58, 6.75] |

### Paired Comparisons (Validation)

- **PPO argmax vs. Visible greedy:** 249 W / 0 T / 1 L (+9.25 SPT mean advantage, 95% CI [8.84, 9.69])
- **PPO sampled vs. Random legal:** 250 W / 0 T / 0 L (+11.03 SPT mean advantage, 95% CI [10.72, 11.34])
- **PPO argmax vs. Random legal:** 250 W / 0 T / 0 L (+10.52 SPT mean advantage, 95% CI [10.08, 10.96])

---

## Canonical Pristine Test Evaluation

- **Evaluation Directory:** `outputs/evaluations/20260824_phase1_v3_seed3_16m_pristine_test`
- **Maps:** 250 pristine held-out test maps
- **Suite:** Full canonical suite (3,000 episodes total)
- **Protocol Note:** The model was frozen prior to running this test evaluation.

### Policy Performance (Final Turn-10 SPT)

| Policy | Episodes | Mean SPT | Median SPT | 95% Confidence Interval |
|---|---:|---:|---:|---|
| **PPO argmax** | 250 | 16.88 | 17.00 | [16.46, 17.31] |
| **PPO sampled** | 1,250 | 17.55 | 17.50 | [17.20, 17.89] |
| **Visible greedy** | 250 | 7.96 | 8.00 | [7.82, 8.10] |
| **Random legal** | 1,250 | 6.70 | 6.60 | [6.62, 6.79] |

### Paired Comparisons (Pristine Test)

- **PPO argmax vs. Visible greedy:** 250 W / 0 T / 0 L (+8.93 SPT mean advantage, 95% CI [8.54, 9.32])
- **PPO sampled vs. Random legal:** 250 W / 0 T / 0 L (+10.84 SPT mean advantage, 95% CI [10.51, 11.16])
- **PPO argmax vs. Random legal:** 250 W / 0 T / 0 L (+10.18 SPT mean advantage, 95% CI [9.78, 10.58])

### Validation $\rightarrow$ Test Generalization Deltas

- **PPO argmax:** $-0.30$ SPT
- **PPO sampled:** $-0.15$ SPT
- **Visible greedy:** $+0.03$ SPT
- **Random legal:** $+0.03$ SPT

The slight drop of $-0.30$ SPT from validation to test confirms strong out-of-distribution map generalization with minimal overfitting to the validation split.

---

## Human Benchmark Comparison

- **Agent Evaluation Directory:** `outputs/evaluations/20260824_phase1_v3_seed3_16m_human_benchmark_argmax`
- **PPO argmax on 17 Benchmark Maps:** Mean 16.71 SPT, Median 17.00 SPT, 95% CI [14.94, 18.47]
- **Human Progress:** 1 completed canonical first attempt out of 17 benchmark maps.
  - **Map:** `map_004393.csv`
  - **Human Turn-10 SPT:** 27
  - **PPO argmax on `map_004393.csv`:** 20
  - **Delta (Human $-$ PPO):** $+7$ SPT

> [!CAUTION]
> **Important Caveat on Human Comparison:**
> This human result represents an illustrative $n=1$ anecdotal comparison on a single map. It is **not** a statistically generalizable claim about human versus agent capability. The remaining 16 benchmark maps have not yet recorded completed human first attempts.

---

## Scientific Interpretation & Findings

1. **Held-Out Map Generalization:**
   The evaluation provides robust evidence that this single trained agent generalizes to previously unseen genuine Polytopia maps under the Phase 1 task contract, outperforming the visible greedy baseline on 250/250 test maps (+8.93 SPT advantage).
2. **Single-Seed Limitation:**
   This result reflects a single training seed (Seed 3). It does not provide evidence of variance or robustness across training seeds.
3. **Environment Comparability (V2 vs V3):**
   Performance differences between historical V1/V2 runs and this V3 run must **not** be interpreted as a controlled algorithmic improvement. The underlying environment mechanics changed materially (monotonic tick advancement, accumulating star economy, and Drylands Fishing masking) and the training budget was expanded to 16M steps.
4. **Historical Evidence:**
   Historical V1 and V2 results remain documented and valid exclusively within their respective historical task definitions.
5. **Active Status:**
   This run serves as the **frozen reference benchmark** for Phase 1. Phase 1 is **not** complete; active optimization continues to target higher Turn-10 SPT efficiency.
6. **Behavioral Failure Analysis:**
   A detailed behavioral analysis and hypothesis audit is documented in [Phase 1 v3 Seed3 16M Behavioral Failure Analysis](Phase1_V3_Seed3_16M_Behavioral_Failure_Analysis.md). Key findings establish that deterministic argmax exhibits 0.0% explicit Forestry adoption across all 250 validation maps, while stochastic sampled replicates with Forestry achieve a +2.533 mean paired SPT advantage across 237 within-map comparisons (descriptive association).
