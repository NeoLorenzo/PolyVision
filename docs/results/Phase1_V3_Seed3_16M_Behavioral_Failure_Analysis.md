# Phase 1 v3 Seed3 16M Behavioral Failure Analysis (Historical)

> [!NOTE]
> **Historical Policy Diagnostic Notice:**
> This behavioral failure analysis specifically analyzes the **previous reference checkpoint** (`runs/Tribes-v0__Phase1-Scientific-Train-V3-Seed3__3__1787526811/Phase1-Scientific-Train-V3-Seed3.cleanrl_model`, 16.88 test argmax SPT) trained under the baseline dense shaping reward without Terminal-SPT. It provides a detailed diagnostic audit of why that specific policy failed to adopt deterministic Forestry. It does not describe the subsequent [Phase 1 v3 Seed3 16M Terminal-SPT Reference Run](Phase1_V3_Seed3_16M_TerminalSPT_Reference_Run.md), which achieved 99.2% deterministic Forestry adoption and 19.34 test argmax SPT.

## 1. Executive Summary

This document presents a comprehensive behavioral failure analysis and empirical hypothesis audit of the previous frozen reference model for PolyVision Phase 1: **`runs/Tribes-v0__Phase1-Scientific-Train-V3-Seed3__3__1787526811/Phase1-Scientific-Train-V3-Seed3.cleanrl_model`** (16,000,000 environment steps, Seed 3).

### Key Verified Findings
* **Validation Performance:** Deterministic PPO (`ppo_argmax`) achieves **17.18 mean Turn-10 SPT** (median 17.00, 95% CI [16.72, 17.66]), substantially exceeding the policy-visible greedy baseline of **7.93 SPT** and random legal baseline of **6.67 SPT**.
* **Deterministic Forestry Absence:** Deterministic PPO researched explicit **Forestry in 0.0% of all 250 validation episodes** (`0 / 250`). The 0.076 mean lumber huts built under argmax resulted entirely from stochastic Ruin `EXAMINE` research bonuses on 2 maps, an instrumentation-tracking nuance rather than an environment contradiction.
* **Sampled Policy Advantage & Forestry Association:** Stochastic PPO (`ppo_sampled`) outperforms deterministic `ppo_argmax` by **+0.52 SPT** (**17.70 vs 17.18** overall). In a within-map paired analysis across 237 eligible validation maps, sampled replicates with Forestry achieved a **+2.533 mean paired SPT advantage** over non-Forestry replicates (184 wins, 3 ties, 50 losses). This pattern is **descriptive and associational, not causal**.
* **Refutation of "Catastrophic Forgetting":** Evaluation of the frozen **4,000,000-step checkpoint** on all 250 validation maps yielded **17.088 mean SPT** and **0.0% explicit Forestry adoption** (`0 / 250`), behaviorally matching the 16M model. The hypothesis that PPO learned a deterministic Forestry strategy at 4M and subsequently forgot it is **contradicted**. Mid-training telemetry adoption reflected stochastic rollout sampling over a ~15–25% policy logit tail, not an established argmax mode.
* **End-Game Capital Accumulation & Army Expansion:** Across all 250 validation maps, deterministic PPO accumulated a mean of **41.06 unspent stars at Turn 10** while reaching a mean final army size of **13.80 units** (including the 2 scripted opening units). Warrior production is an available star sink when Tier-1 technology is exhausted, though final unit count correlates positively with empire scale ($r = +0.766$) and is not established as an isolated causal drag on SPT.
* **Mechanistic Hypotheses Requiring Future Ablation:** The divergence between deterministic Tier-1 monoculture (Organization-only) and stochastic Forestry adoption is hypothesized to stem from (1) temporal discounting of delayed economic rewards under dense step-level shaping, (2) representation challenges in learning state-action interactions between the 505-d global spatial observation and discrete 42-d legal action rows, and (3) argmax suppression of sub-threshold exploratory actions.

---

## 2. Dataset and Methodology

### Task Contract & Provenance
* **Tribe:** Bardur (tribe 0)
* **Map Distribution:** Genuine 11×11 Drylands maps from the frozen pool manifest
* **Horizon:** Turn 10 truncation (handoff at Turn 2 following scripted `v2_guaranteed_two_unit` opening)
* **Economy:** Authentic accumulating economy `v3_corrected_turn_economy` (`gs.incTick()`, `gs.initTurn()`)
* **Policy Space:** 63,913 global action catalog, max 256 legal slots, 42-d semantic action features, 505-d flat observation, attacks excluded, Drylands Fishing masked
* **Reference Model:** `Phase1-Scientific-Train-V3-Seed3.cleanrl_model` (SHA-256: `A6803CB00E08522819FDEB74BEDE8816CF131C1F28244C12D44F3EEFB9D2C234`)

### Data Integrity & Safety Guarantee
* **Evaluation Scope:** Strictly confined to the **250 held-out validation maps** and historical training records. **Zero test-pool maps (0/250) were evaluated or inspected during this failure analysis.**
* **Reference Artifacts Inspected:**
  1. `outputs/evaluations/20260824_phase1_v3_seed3_16m_validation_canonical/` (per-map, comparison, summary, and 3,000 episode logs)
  2. `outputs/evaluations/20260824_phase1_v3_seed3_4m_validation_argmax/` (focused diagnostic evaluation of 4M checkpoint across 250 validation maps)
  3. `runs/Tribes-v0__Phase1-Scientific-Train-V3-Seed3__3__1787526811/events.out.tfevents...` (4,449,907 TensorBoard event records across 43 telemetry tags)
  4. Per-step trajectory replays with counterfactual logit and probability extraction across 19 representative validation episodes and 1 human benchmark episode (`map_004393.csv`).

---

## 3. Performance Distribution

### Validation Distribution Summary (250 Maps)

| Metric | PPO Argmax | PPO Sampled | Visible Greedy | Random Legal |
|---|---:|---:|---:|---:|
| **Mean Turn-10 SPT** | **17.18** | **17.70** | **7.93** | **6.67** |
| **Median Turn-10 SPT** | 17.00 | 17.60 | 8.00 | 6.60 |
| **Std Dev** | 3.72 | 2.67 | 1.25 | 0.69 |
| **Min / Max** | 6.0 / 30.0 | 9.8 / 24.4 | 5.0 / 12.0 | 4.6 / 8.6 |
| **5th / 95th Percentile** | 11.0 / 23.0 | 13.29 / 21.80 | 6.0 / 10.0 | 5.4 / 7.6 |
| **95% Confidence Interval** | [16.72, 17.66] | [17.37, 18.03] | [7.78, 8.09] | [6.58, 6.75] |

```
PPO Argmax Validation SPT Histogram (N=250):
  6 -  9 SPT:  [###                  ]  13 maps ( 5.2%)
 10 - 13 SPT:  [######               ]  34 maps (13.6%)
 14 - 17 SPT:  [#################    ]  88 maps (35.2%)
 18 - 21 SPT:  [################     ]  79 maps (31.6%)
 22 - 25 SPT:  [#####                ]  27 maps (10.8%)
 26 - 30 SPT:  [#                    ]   9 maps ( 3.6%)
```

### Strata Comparison: Bottom 25 vs Middle 200 vs Top 25

| Metric | Bottom 25 Maps | Middle 200 Maps | Top 25 Maps | Top-to-Bottom Delta |
|---|---:|---:|---:|---:|
| **Mean Turn-10 SPT** | **10.96** | **17.14** | **23.76** | **+12.80** |
| **Capturable Villages (Map Total)** | 3.72 | 5.59 | 6.96 | +3.24 |
| **Captured Villages by T10** | 2.24 (60.2%) | 4.25 (76.0%) | 5.96 (85.6%) | +3.72 (+25.4% capture rate) |
| **Turn 2nd City Captured** | **5.88** | **4.72** | **4.56** | **-1.32 turns earlier** |
| **Final Unit Count (Total Units)** | 9.88 | 13.92 | 16.80 | +6.92 |
| **Fruit Harvested** | 5.96 | 10.47 | 15.24 | +9.28 |
| **Animals Harvested** | 3.00 | 5.72 | 8.52 | +5.52 |
| **Fog Tiles Cleared** | 53.92 | 67.41 | 82.28 | +28.36 |
| **Unspent Stars at T10** | 39.76 | 40.94 | 43.36 | +3.60 |
| **Organization Researched Rate** | 100.0% (T2.04) | 100.0% (T2.14) | 100.0% (T2.08) | 0.0% diff |
| **Explicit Forestry Researched Rate** | **0.0%** | **0.0%** | **0.0%** | **0.0% across all strata** |
| **Lumber Huts Built** | 0.28 | 0.06 | 0.00 | -0.28 |

### Baseline Comparison Outliers
* **PPO Argmax vs Visible Greedy:** 249 Wins, 0 Ties, 1 Loss (Mean advantage $+9.25$ SPT).
* **The Single Map Where Greedy Beat PPO:** `map_001810.csv` (Greedy 10.0 SPT vs PPO 8.0 SPT, Delta $-2.0$).
  * *Diagnostic Cause:* On `map_001810.csv`, villages were clustered along narrow corridors. Visible greedy followed a hard heuristic prioritizing immediate village moves and instant captures. PPO argmax was drawn by high-reveal fog moves into empty border tiles, delaying 2nd city capture until Turn 6.

---

## 4. High-vs-Low Performance Behaviors

### Primary Drivers of Performance Variance
1. **Village Density & Spatial Accessibility:** High-SPT maps have an average of 6.96 capturable villages vs 3.72 on low-SPT maps. Because PPO relies primarily on city founding (+1 SPT base + +1 SPT Workshop) and fruit/animal harvesting to generate SPT, map village density forms a major structural driver of performance.
2. **Second-City Timing:** On top-performing maps, the 2nd city is captured by Turn 4.56 on average; on bottom maps, capture is delayed to Turn 5.88 (or later). Earlier capture allows compounding economic returns across subsequent turns.
3. **End-Game Star Accumulation:**
   * Across all validation episodes, PPO finishes Turn 10 with **41.06 unspent stars**.
   * In Turn 8–10, the policy frequently possesses 25–45 stars in bank, but having researched only Organization and Hunting, has limited productive infrastructure options.
4. **Army Expansion & Tile Utilization:**
   * Mean total unit count reaches **13.80** (including the 2 scripted opening units).
   * Spawning additional warriors costs 2 stars plus unit capacity. While useful for scouting perimeter fog and claiming distant villages, extensive spawning in crowded cities can cause local tile congestion without economic compounding once scouting is complete.
5. **Economic Monoculture (The Tier-1 Pattern):**
   * PPO executes a consistent opening across nearly all maps: Research Organization on Turn 2, harvest visible fruit and animals, and choose Workshop on city level-ups.
   * In deterministic argmax mode, PPO does not progress to Tier-2 economic technology (Forestry), relying exclusively on Tier-1 resource gathering.

---

## 5. Argmax-vs-Sampled Discrepancy & Within-Map Analysis

### The $+0.52$ SPT Aggregate Advantage of Stochastic Policy
Sampled PPO achieves **17.70 mean SPT vs 17.18 for argmax**. It beats argmax on 135/250 maps (54.0%), ties on 6 maps, and loses on 109 maps.

```
Sampled Advantage Distribution (Sampled Mean SPT - Argmax SPT):
  Delta > +6.0 SPT:   [###                 ]  10 maps ( 4.0%)
  Delta +2.0 to +6.0: [#######             ]  52 maps (20.8%)
  Delta  0.0 to +2.0: [############        ]  73 maps (29.2%)
  Ties (Delta = 0.0): [#                   ]   6 maps ( 2.4%)
  Delta -2.0 to  0.0: [#########           ]  58 maps (23.2%)
  Delta < -2.0 SPT:   [########            ]  51 maps (20.4%)
```

### Within-Map Paired Replicate Analysis (Descriptive)
To understand what behaviors distinguish high-performing stochastic rollouts from low-performing ones on the exact same maps, we analyze the 1,250 canonical `ppo_sampled` episodes across the 237 validation maps that contain both Forestry and non-Forestry sampled replicates:

* **Eligible Within-Map Comparisons:** 237 / 250 maps (94.8%)
* **Mean Paired Difference ($\text{Mean}_{\text{Forestry}} - \text{Mean}_{\text{No-Forestry}}$):** **+2.533 SPT**
* **Median Paired Difference:** **+2.500 SPT**
* **Record for Forestry Replicates:** **184 Wins (77.6%) / 3 Ties (1.3%) / 50 Losses (21.1%)**
* **Distribution of Paired Differences:**
  * $\Delta > +4.0$ SPT: 74 maps (31.2%)
  * $\Delta +2.0\text{ to }+4.0$ SPT: 50 maps (21.1%)
  * $\Delta 0.0\text{ to }+2.0$ SPT: 60 maps (25.3%)
  * $\Delta = 0.0$ (Ties): 3 maps (1.3%)
  * $\Delta -2.0\text{ to }0.0$ SPT: 34 maps (14.3%)
  * $\Delta -4.0\text{ to }-2.0$ SPT: 13 maps (5.5%)
  * $\Delta < -4.0$ SPT: 3 maps (1.3%)

> [!NOTE]
> **Methodological Status: Descriptive, Not Causal.**
> This within-map comparison demonstrates a strong positive association between stochastic Forestry adoption and higher final SPT. However, it is not an isolated causal proof: replicates that adopt Forestry may also have benefited from favorable early exploration moves, earlier village discovery, or higher initial star liquidity.

### Audit of Large Sampled-vs-Argmax Advantage Maps
On maps where sampled PPO achieved the largest gains over argmax, individual trajectory audits show that stochastic Forestry adoption strongly coincided with high SPT:
* `map_004817.csv` (Argmax 12.0 vs Sampled Mean 20.60, $\Delta = +8.60$): Forestry adopted in 3/5 replicates (replicates with Forestry averaged 22.0 SPT and 13.0 huts vs non-Forestry replicates at 18.5 SPT and 0 huts).
* `map_005425.csv` (Argmax 12.0 vs Sampled Mean 20.60, $\Delta = +8.60$): Forestry adopted in 4/5 replicates (replicates with Forestry averaged 21.25 SPT and 13.25 huts vs non-Forestry replicate at 18.0 SPT).
* `map_002016.csv` (Argmax 10.0 vs Sampled Mean 16.00, $\Delta = +6.00$): Forestry adopted in 2/5 replicates (replicates with Forestry averaged 19.5 SPT and 13.5 huts vs non-Forestry replicates at 13.67 SPT).

---

## 6. Representative Trajectory Audits

### Stratified Sample of Reconstructed Episodes

```
========================================================================================================================
Map             Strata   T10 SPT  Stars  Cities  Units  Decisions  Org Turn  Forestry Turn  2nd City Turn  Fruit  Animals
========================================================================================================================
map_001527.csv  LOW         5.0   29.0       2      6         50      T2.0           None           T6.0      1        2
map_001810.csv  LOW (L)     8.0   33.0       3      8         62      T2.0           None           T6.0      3        3
map_002988.csv  LOW        10.0   40.0       3      9         67      T2.0           None           T5.0      5        3
map_003638.csv  LOW        10.0   36.0       3      9         69      T2.0           None           T5.0      4        4
map_003209.csv  LOW        10.0   38.0       3      9         68      T2.0           None           T5.0      5        3
------------------------------------------------------------------------------------------------------------------------
map_003840.csv  MED        17.0   26.0       6      9         76      T2.0           None           T4.0     11        3
map_000542.csv  MED        17.0   41.0       5     13         91      T2.0           None           T4.0     10        5
map_005348.csv  MED        17.0   41.0       5     14         91      T2.0           None           T4.0     10        5
map_002828.csv  MED        17.0   38.0       5     14         88      T2.0           None           T4.0      9        6
map_001709.csv  MED        17.0   30.0       6     12         82      T2.0           None           T4.0     12        3
------------------------------------------------------------------------------------------------------------------------
map_004614.csv  HIGH       25.0   47.0       8     17        113      T2.0           None           T4.0     16        7
map_001749.csv  HIGH       26.0   37.0       7     14        110      T2.0           None           T4.0     17        9
map_001465.csv  HIGH       26.0   42.0       8     20        120      T2.0           None           T4.0     16        8
map_000190.csv  HIGH       28.0   55.0       8     18        126      T2.0           None           T4.0     18        9
map_001238.csv  HIGH       30.0   40.0       9     19        126      T2.0           None           T4.0     18       12
------------------------------------------------------------------------------------------------------------------------
map_002016.csv  GAP_SAMP   10.0   44.0       3      9         72      T2.0           None           T6.0      4        4
map_005425.csv  GAP_SAMP   12.0   46.0       3     11         79      T2.0           None           T6.0      5        4
map_004817.csv  GAP_SAMP   12.0   39.0       4     12         85      T2.0           None           T5.0      6        4
map_004393.csv  HUMAN_MAP  20.0   33.0       6     15        100      T2.0           None           T4.0     11        7
========================================================================================================================
```

### Micro-Audit: Low Map `map_001527.csv` (Final SPT = 5.0)
* **Turn 2 (Stars: 7, SPT: 4):** Researches Organization (cost 5), moves unit (5,4) $\rightarrow$ (4,4), trains warrior at (5,5), ends turn with 0 stars.
* **Turn 3–5:** Explores south into heavy mountains. Fails to locate a second village until Turn 5.
* **Turn 6:** Captures 2nd village at (7,6) (SPT becomes 5).
* **Turn 7–10:** No further villages found. Spawns additional warriors. Harvests 1 fruit, 2 animals. Finishes Turn 10 with **29 unspent stars**, 2 cities, 6 units, and **5 SPT**.

### Micro-Audit: High Map `map_001238.csv` (Final SPT = 30.0)
* **Turn 2–4:** Rapid 3-directional expansion. 2nd village captured on Turn 4, 3rd village on Turn 5, 4th on Turn 6, 5th on Turn 7, 6th on Turn 8, 7th & 8th on Turn 9, 9th on Turn 10.
* **Turn 10:** 9 cities, 19 units, 18 fruit harvested, 12 animals harvested, 40 unspent stars. 0 Forestry researched. Reached 30 SPT purely through high map village density and rapid multi-city expansion.

---

## 7. Counterfactual Action Analysis

To inspect decision-state logits and probabilities at critical inflection points:

### Case 1: Sub-Threshold Forestry Probability (`map_005425.csv`, Step 66, Turn 9)
* **State Context:** 3 cities, 11 units, 24 stars in bank, SPT = 12. Multiple forests available inside owned borders.
* **Chosen Action:** `End turn` ($p = 0.498$, logit $-0.70$)
* **Top Alternatives:**
  1. `End turn` ($p = 0.498$, logit $-0.70$) — **CHOSEN**
  2. `Research FORESTRY` ($p = 0.206$, logit $-1.58$)
  3. `Research CLIMBING` ($p = 0.070$, logit $-2.66$)
  4. `Research ARCHERY` ($p = 0.070$, logit $-2.66$)
  5. `Research FARMING` ($p = 0.062$, logit $-2.78$)
* **Assessment:** `Research FORESTRY` costs 7 stars. With 24 stars in bank and 12 SPT, researching Forestry would enable building multiple lumber huts across turns 9–10. The policy maintained **20.6% probability mass on Forestry**, but argmax deterministically selected `End turn`.

### Case 2: Fog Scouting vs Village Step (`map_001810.csv`, Step 18, Turn 4)
* **State Context:** Warrior positioned at (4,6) adjacent to neutral visible village at (5,6).
* **Chosen Action:** `Move unit (3,2) -> (2,2)` into unscouted fog ($p = 0.612$, logit $+0.45$)
* **Alternative Action:** `Move unit (4,6) -> (5,6)` onto neutral village ($p = 0.284$, logit $-0.32$)
* **Assessment:** The policy preferred clearing 3 fog tiles on the western flank over stepping onto the neutral village immediately, delaying capture. This illustrates how per-step fog clearance shaping can compete with village approach in local action selection.

---

## 8. Reward-Alignment Hypotheses

### Correlation of Telemetry Components with Final SPT

| Telemetry Metric / Component | Correlation with Final Turn-10 SPT ($r$) | Empirical Interpretation |
|---|---:|---|
| `final_city_count` | **+0.898** | Strongest positive correlate with final economy |
| `total_shaped_return` | **+0.975** | Tightly coupled with episode length and city captures |
| `policy_decisions` | **+0.872** | More cities $\rightarrow$ more unit moves $\rightarrow$ higher decisions |
| `final_unit_count` | **+0.766** | Collinear with city count (larger empires naturally host more units) |
| `fruit_harvested_t10` | **+0.765** | Primary engine of Tier-1 city level-ups |
| `animals_harvested_t10` | **+0.651** | Secondary engine of Tier-1 city level-ups |
| `fog_tiles_cleared_total` | **+0.678** | Necessary for village discovery, though step bounties can divert moves |
| `turn_second_city_captured` | **-0.361** | Earlier capture correlates with higher final economy |
| `final_stars` (unspent at T10) | **+0.152** | Weak positive correlation; capital remains unspent at horizon |
| `lumber_huts_built_t10` | **-0.033** | Degenerate under argmax (near-zero variance; 2 ruin maps only) |

### Mechanistic Hypotheses on Reward Structure
1. **Immediate vs Delayed Economic Returns:**
   * Spending 2 stars on fruit/animal harvesting provides immediate $+1$ city population $\rightarrow$ city level-up $\rightarrow$ immediate stepwise delta SPT reward.
   * In contrast, spending 7 stars on Forestry yields **0.0 immediate reward**, requiring subsequent multi-step actions (`Build LUMBER_HUT`) across multiple turns to yield economic return. Under standard discounting ($\gamma = 0.99$), this return delay may disadvantage tech research relative to immediate gathering.
2. **Fog Clearance vs Village Pathing Competition:**
   * Moves clearing multiple fog tiles receive immediate $+0.08$ to $+0.24$ shaped reward. While scouting is necessary to find villages ($r = +0.678$), high fog bounties can create local logit competition against direct village approach.
3. **Absence of Terminal Treasury Penalty:**
   * Unspent bank stars carry zero penalty at Turn 10 truncation, providing no pressure to convert excess treasury into late-game infrastructure.

---

## 9. Feature-Representation Considerations

### Architectural Context
The `legal_features` actor architecture provides the policy network with **both the 505-dimensional global observation vector** (encoding full grid terrain, ownership, units, and resources) **and each 42-dimensional legal-action feature row**.

Because terrain and city tiles are present in the global state, the policy has physical access to environmental information. The feature-representation challenge is therefore narrower:

1. **State-Action Interaction Learning Difficulty:**
   * When evaluating `Research FORESTRY` (action feature index 24), the action row does not carry the current count of owned forest tiles. To evaluate tech payoff, the network must learn cross-feature bilinear interactions between the global spatial grid and the discrete action feature row.
2. **Absence of Action Cost Margin in Action Rows:**
   * The legal-action features lack a normalized `action_star_cost` scalar. Evaluating affordability and cost margins relative to the treasury requires cross-referencing global scalar features with discrete action types.
3. **Local Spawning Context:**
   * Training a warrior receives identical action-type features (`is_train_or_spawn=1.0`, `unit_type_warrior=1.0`) regardless of whether the city is a spacious frontline base or a landlocked, crowded capital.

---

## 10. Training Dynamics & Telemetry Audit

### Inspection of the 16M Step Training Telemetry

```
========================================================================================================================
Training Step         100k       500k         1M         2M         4M         8M        12M        16M   Metric Min/Max
========================================================================================================================
Final T10 SPT          8.0       12.0       20.0       17.0       21.0       16.0       16.0       21.0        4.0 / 36.0
Episodic Reward     0.5726     1.4380     1.1350     1.2126     1.9420     1.9800     1.4821     1.0310   -0.715 / 4.885
Policy Entropy      2.2743     1.5270     1.3983     1.3254     1.4307     1.4589     1.5374     1.4221   1.2666 / 2.488
Explained Var       0.1956     0.4464     0.6953     0.8050     0.8431     0.8346     0.8519     0.8616   0.0051 / 0.903
Value Loss         27.3546   108.0358    78.8845    59.1546    86.7122    61.9329    80.7040    56.4139  19.069 / 167.72
Learning Rate       2.5e-4     2.4e-4     2.3e-4     2.2e-4     1.9e-4     1.3e-4     6.3e-5     0.0e-0   0.0000 / 2.5e-4
Forestry Rate         1.00       1.00       0.00       1.00       1.00       1.00       0.00       0.00      0.00 / 1.00
Lumber Huts Built      2.0        6.0        0.0       12.0       15.0        8.0        0.0        0.0       0.0 / 24.0
Fog Cleared T10       39.0       56.0       64.0       54.0       59.0       70.0       62.0       71.0       0.0 / 93.0
========================================================================================================================
```

### Telemetry vs Argmax Reality: The 4M Verification
* During training (between 2M and 8M steps), TensorBoard telemetry logged episodes with Forestry adoption (rate $= 1.0$) and 8–15 lumber huts.
* Direct validation evaluation of the frozen 4M checkpoint ([`Section 15.3`](#153-task-3-empirical-test-of-the-forgotten-forestry-hypothesis)) revealed that deterministic argmax adoption at 4M was **0.0%** (`0 / 250`), matching the 16M model.
* **Finding:** The mid-training telemetry recorded behavior during **stochastic rollouts** where policy entropy ($\sim 1.43$) sampled actions across the ~15–25% logit tail. The deterministic argmax policy never adopted Forestry at any checkpoint.

---

## 11. Human-Map Illustrative Comparison (`map_004393.csv`)

On benchmark map `map_004393.csv`, a human player achieved **27 Turn-10 SPT**, while PPO argmax achieved **20 Turn-10 SPT** (Delta: $+7$ SPT in favor of human).

```
========================================================================================================================
Strategic Dimension             Human First Attempt (27 SPT)     PPO Argmax (20 SPT)
========================================================================================================================
Final City Count                7 cities (100% of map)           6 cities (85.7% of map)
Final Unit Count (Total Units)  4 units                          15 units (+11 additional)
Techs Researched                Organization + FORESTRY          Organization ONLY
Turn Forestry Researched        Turn 4                           NEVER (0% adoption)
Lumber Huts Built               17 lumber huts                   0 lumber huts
Forests Cleared                 2 forests                        0 forests
Average City Level              Level 2.71                       Level 2.17
Fog Tiles Cleared               50 tiles                         67 tiles (+17 over-scouted)
Final Unspent Stars             51 stars                         33 stars
Decision Count                  95 decisions                     100 decisions
Core Strategic Concept          Infrastructure & Lumber Scaling  Expansion & Unit Spam
========================================================================================================================
```

> [!CAUTION]
> **Important Caveat:** This comparison represents an illustrative $n=1$ anecdotal benchmark on a single map. It is not a statistically generalizable claim about human versus agent capability.

---

## 12. Empirical Hypotheses for Future Investigation

The following candidate hypotheses represent plausible research directions to test via controlled experimental ablations:

1. **Tech-Delayed Economy Reward Bridging (Hypothesis):**
   * Providing milestone reward support for unlocking Tier-2 economic tech (`FORESTRY`) or lumber hut construction may reduce the temporal credit-assignment delay and allow tech actions to compete with immediate gathering in argmax logits.
2. **Fog Clearance Reward Attenuation (Hypothesis):**
   * Decaying or reducing per-step fog clearance bounties may reduce tactical distraction when warriors are positioned adjacent to capturable villages.
3. **Action-Local Context Features (Hypothesis):**
   * Augmenting the 42-d legal action vector with normalized action cost and owned forest density may improve state-action representation efficiency without requiring complex spatial cross-attention.
4. **Entropy Floor & LR Annealing (Hypothesis):**
   * Maintaining an exploration entropy floor may prevent the policy from prematurely concentrating probability mass exclusively on Tier-1 actions.

---

## 13. Unresolved Questions Requiring Controlled Ablation

The current evidence establishes clear behavioral associations but leaves the following causal questions open for future empirical testing:

1. **Reward Shaping Causality:** Does adding explicit tech-milestone rewards reliably induce deterministic argmax Forestry adoption, and does it translate into higher net SPT without distorting early expansion?
2. **Feature vs Optimization Limitations:** Can PPO learn to research Forestry purely through reward/entropy adjustments on the existing 42-d feature interface, or is action-feature enrichment required for reliable state-action credit assignment?
3. **Fog Reward Net Trade-off:** Does attenuating fog clearance shaping improve village capture tempo without impairing overall map exploration and village discovery rates?
4. **Multi-Seed Robustness:** Do the behavioral patterns observed in Seed 3 (Organization monoculture, star accumulation, sub-threshold Forestry probability) generalize across independent training seeds?

---

## 14. Synthesized Failure Analysis Summary

### The Diagnosed Bottlenecks & Revised Confidence

| Diagnosed Phenomenon | Observed Manifestation | Confidence | Status |
|---|---|:---:|:---:|
| **Deterministic Forestry Absence** | 0/250 argmax episodes researched Forestry at both 4M and 16M; 0.076 huts arose from ruin drops. | **Certain** | **VERIFIED** |
| **Sampled Policy Headroom** | Sampled rollouts with Forestry achieve +2.533 paired SPT over non-Forestry rollouts across 237 maps. | **High** | **SUPPORTED (Descriptive)** |
| **Reward Horizon Discounting** | Immediate harvest rewards dominate delayed 7-star tech investments under $\gamma=0.99$. | **High** | **SUPPORTED (Mechanistic)** |
| **State-Action Interaction Complexity** | Forest context exists in 505-d global state but requires learning cross-feature interactions with action rows. | **Moderate** | **SUPPORTED (Representation)** |
| **Fog vs Village Competition** | Step bounties can bias individual moves away from villages, though scouting remains necessary. | **Moderate** | **SUPPORTED (Qualitative)** |
| **Catastrophic Forgetting** | Claim that deterministic Forestry was learned at 4M and forgotten by 16M. | **Falsified** | **CONTRADICTED** |

---

## 15. Empirical Hypothesis Verification and Audit

To validate the diagnoses and claims before designing interventions, targeted empirical evaluations and metric audits were conducted against the canonical dataset and saved checkpoints.

### 15.1 Task 1: Resolution of Forestry vs Lumber-Hut Consistency
* **Reported Baseline:** PPO Argmax Forestry adoption $= 0/250$ ($0.0\%$), Mean Lumber Huts built $= 0.076$.
* **Audit of Canonical Episodes ($N=250$ Argmax):**
  * Exact count of argmax episodes with `forestry_researched == 1`: **0 / 250 (0.0%)**.
  * Exact count of argmax episodes with $\ge 1$ lumber hut built: **2 / 250 (0.80%)**.
  * Total lumber huts constructed across all 250 maps: **19 huts** ($19 / 250 = 0.0760$ mean).
  * **Exhibiting Maps:**
    1. `map_001171.csv`: 7 lumber huts built on Turn 7; `forestry_researched = 0`, `turn_forestry_researched = -1`.
    2. `map_001676.csv`: 12 lumber huts built across Turns 9–10; `forestry_researched = 0`, `turn_forestry_researched = -1`.
* **Root Mechanism (Environment Semantics & Instrumentation):**
  * In both episodes, a warrior executed the `EXAMINE` action on an unexamined Ruin tile (`Step 30` on map 1171, `Step 59` on map 1676).
  * The underlying Java engine (`ExamineCommand.java`) rolled the `RESEARCH` examine bonus, executing `technologyTree.researchAtRandom()`, which directly unlocked `FORESTRY` inside Java's `GameState`.
  * The Python gym wrapper's metric tracker (`_update_economy_counters_from_action` in `register_env.py`) only listens for explicit `RESEARCH_TECH` actions to update `_researched_techs_t10` and `_turn_forestry_researched`. Because the tech was granted via `EXAMINE` rather than `RESEARCH_TECH`, the instrumentation recorded `forestry_researched = 0`.
  * However, with Forestry unlocked in `GameState`, `BUILD LUMBER_HUT` actions immediately became legal in the policy-visible action catalog. PPO argmax selected them, constructing 7 and 12 lumber huts respectively.
  * **Conclusion:** There is zero logical contradiction. Deterministic PPO never chose `RESEARCH_TECH: FORESTRY` (0/250). The 0.076 mean lumber huts resulted entirely from stochastic ruin rewards granting Forestry for free.

---

### 15.2 Task 2: Sampled PPO Forestry Outcome Analysis (1,250 Canonical Episodes)
Using exclusively the 1,250 existing `ppo_sampled` episodes from `outputs/evaluations/20260824_phase1_v3_seed3_16m_validation_canonical`:

#### 1. Split-Episode Aggregate Summary

| Metric | Sampled WITH Forestry Researched ($N=583$) | Sampled WITHOUT Forestry Researched ($N=667$) | Delta (With vs Without) |
|---|---:|---:|---:|
| **Mean Turn-10 SPT** | **18.806** | **16.736** | **+2.070** |
| **Median Turn-10 SPT** | 19.000 | 17.000 | +2.000 |
| **Std Dev SPT** | 4.168 | 3.696 | +0.472 |
| **Mean Final Cities** | 4.774 | 5.147 | -0.373 |
| **Mean Final Units** | 12.878 | 12.649 | +0.229 |
| **Mean Final Stars (Bank)** | 26.762 | 28.142 | -1.380 |
| **Mean Lumber Huts Built** | **10.364** | **0.067** | **+10.297** |
| **Mean Forestry Turn** | **6.276** (Median 6.0, Min 2, Max 10) | N/A | — |

*(Note: Including 4 ruin-granted Forestry episodes yields identical figures: $N=587$, Mean SPT 18.821 vs $N=663$, Mean SPT 16.710).*

#### 2. Within-Map Paired Replicate Analysis
Evaluating all maps having $\ge 1$ sampled replicate with Forestry and $\ge 1$ replicate without Forestry:
* **Eligible Maps:** **237 out of 250 maps (94.8%)**
* **Mean Paired Difference ($\text{Mean}_{\text{Forestry}} - \text{Mean}_{\text{No-Forestry}}$):** **+2.533 SPT**
* **Median Paired Difference:** **+2.500 SPT**
* **Win / Tie / Loss Record for Forestry:** **184 Wins (77.6%) / 3 Ties (1.3%) / 50 Losses (21.1%)**
* **Standard Deviation of Difference:** 3.233 SPT (Min $-6.000$, Max $+12.750$)
* **Distribution of Paired Differences:**
  * $\Delta > +4.0$ SPT: **74 maps (31.2%)**
  * $\Delta +2.0\text{ to }+4.0$ SPT: **50 maps (21.1%)**
  * $\Delta 0.0\text{ to }+2.0$ SPT: **60 maps (25.3%)**
  * $\Delta = 0.0$ (Ties): **3 maps (1.3%)**
  * $\Delta -2.0\text{ to }0.0$ SPT: **34 maps (14.3%)**
  * $\Delta -4.0\text{ to }-2.0$ SPT: **13 maps (5.5%)**
  * $\Delta < -4.0$ SPT: **3 maps (1.3%)**

> [!NOTE]
> **Methodological Caveat:** This within-map analysis is strictly **descriptive, not causal**. Stochastic replicates that successfully adopted Forestry may have benefited from early favorable exploration rolls, early village sightings, or higher initial star liquidity that made Forestry affordable.

#### 3. Audit of Large Sampled-vs-Argmax Improvement Maps
Testing whether the largest sampled-advantage maps identified in Section 5 coincide with Forestry adoption:
* `map_004817.csv` (Argmax 12.0 vs Sampled Mean 20.60, $\Delta = +8.60$): Forestry adopted in 3/5 replicates (reps 0, 2, 4 averaging 22.0 SPT and 13.0 huts vs non-Forestry reps 1, 3 averaging 18.5 SPT and 0 huts).
* `map_005425.csv` (Argmax 12.0 vs Sampled Mean 20.60, $\Delta = +8.60$): Forestry adopted in 4/5 replicates (reps 0, 1, 3, 4 averaging 21.25 SPT and 13.25 huts vs non-Forestry rep 2 at 18.0 SPT).
* `map_002016.csv` (Argmax 10.0 vs Sampled Mean 16.00, $\Delta = +6.00$): Forestry adopted in 2/5 replicates (reps 2, 4 averaging 19.5 SPT and 13.5 huts vs non-Forestry reps 0, 1, 3 averaging 13.67 SPT).
* **Finding:** Across all top 20 sampled-advantage maps, the large performance gains strongly coincide with stochastic Forestry adoption and subsequent lumber hut construction.

---

### 15.3 Task 3: Empirical Test of the "Forestry Was Forgotten" Hypothesis
To test whether PPO possessed a deterministic Forestry policy mid-training that was subsequently forgotten, the frozen 4,000,000-step checkpoint (`model_checkpoint_4000000.cleanrl_model`) was evaluated across all **250 held-out validation maps** under deterministic argmax (`ppo_argmax`, 1 replicate per map, seed 42):

| Metric | 4,000,000-Step Checkpoint (Argmax) | 16,000,000-Step Reference (Argmax) | Delta (16M vs 4M) |
|---|---:|---:|---:|
| **Mean Turn-10 SPT** | **17.088** (95% CI [16.64, 17.54]) | **17.180** (95% CI [16.72, 17.66]) | **+0.092** |
| **Median Turn-10 SPT** | **17.000** | **17.000** | 0.000 |
| **Forestry Adoption Rate** | **0 / 250 (0.00%)** | **0 / 250 (0.00%)** | **0.00%** |
| **Mean Forestry Turn** | **N/A** (0 episodes researched) | **N/A** (0 episodes researched) | — |
| **Mean Lumber Huts Built** | **0.048** (12 huts on 1 ruin map) | **0.076** (19 huts on 2 ruin maps) | +0.028 |
| **Mean Final Stars (Bank)** | **40.796** | **41.060** | +0.264 |
| **Mean Final Units** | **13.808** | **13.800** | -0.008 |
| **Mean Final Cities** | **5.184** | **5.160** | -0.024 |

#### Audit Conclusion on "Catastrophic Forgetting"
* The hypothesis that PPO "learned and then catastrophically forgot Forestry" is **CONTRADICTED**.
* At 4M steps, deterministic argmax had an adoption rate of **0.0%** and achieved 17.088 SPT—statistically indistinguishable from the final 16M model (17.180 SPT).
* The mid-training adoption seen in TensorBoard telemetry (Forestry Rate $= 1.0$, Lumber Huts $= 15.0$) was entirely an artifact of **stochastic rollouts** sampling across a $\sim 15\text{--}25\%$ policy logit tail during training with active policy entropy ($\sim 1.43$).
* Forestry was never an established mode in the deterministic argmax policy at any point during training.

---

### 15.4 Task 4: Systematic Claim Classification and Evidence Audit

| Claim / Conclusion from Prior Analysis | Classification | Empirical Basis and Required Nuance |
|---|:---:|---|
| **Forestry Adoption Deficit (Argmax = 0%)** | **VERIFIED** | Exactly 0/250 argmax episodes researched Forestry at both 4M and 16M steps. The 0.076 lumber huts arose solely from stochastic Ruin `EXAMINE` rewards. |
| **Sampled Policy Superiority via Forestry** | **SUPPORTED BUT NOT CAUSAL** | Sampled replicates with Forestry average +2.070 SPT over non-Forestry replicates, winning 77.6% of within-map pairings (+2.533 mean paired SPT). However, this is descriptive correlation, as star liquidity and scouting luck may co-occur with Forestry adoption. |
| **"Catastrophic Forgetting" of Forestry** | **CONTRADICTED** | 4M argmax validation evaluation shows 0.0% Forestry adoption and 17.088 SPT, matching 16M (17.180 SPT). Forestry was never an argmax policy mode; it was only sampled stochastically during high-entropy rollouts. |
| **"Reward-Limited" Hypothesis** | **SUPPORTED (Mechanistic)** | Step-level harvesting and village rewards provide immediate delta SPT shaping, whereas Forestry costs 7 stars and delivers 0 immediate reward. Under $\gamma = 0.99$, delayed economic returns face steep temporal discounting. |
| **"Feature-Limited" Hypothesis** | **SUPPORTED (Representation)** | The actor receives both the 505-d global spatial grid and the 42-d action features. The issue is not absolute information absence (forest tiles are visible on the grid), but rather inductive bias / representation difficulty in learning cross-feature interactions between global forest density and discrete tech action rows. |
| **Fog Clearance Reward Distorts Village Pathing** | **SUPPORTED BUT NOT CAUSAL** | Trajectory logits demonstrate cases where warriors choose fog moves ($p \approx 0.61$) over adjacent village captures ($p \approx 0.28$). However, fog clearance is also necessary to find villages ($r = +0.678$); the distortion is mechanistic rather than an isolated causal ablation. |
| **Warrior Spam Directly Causes Low SPT** | **SUPPORTED BUT NOT CAUSAL** | High unit counts correlate positively with SPT overall ($r = +0.766$) due to city count collinearity. Warrior spawning is primarily a symptom of star accumulation with no other available tech sinks (Organization only), rather than the sole driver of low SPT. |
| **Expected +3.0 to +4.5 SPT Upside from Reward Fix** | **SPECULATIVE** | While sampled Forestry replicates reach 18.806 SPT and human play reaches 27 SPT, projecting an unconditional +3.0 to +4.5 SPT gain assumes that unlocking Forestry will not introduce new failure modes (e.g. over-investing in tech early and delaying city expansion). |

---

PHASE 1 BEHAVIORAL FAILURE ANALYSIS: COMPLETE
