# Rewards

Phase 1 optimizes a shaped economy-and-expansion reward, not raw game victory and not final stars per turn (SPT) alone.

At each policy step the current reward is:

```text
scaled change in Bardur SPT
+ city-capture bonus
+ exploration and village-progress shaping
+ optional terminal SPT bonus
```

## SPT component

SPT is computed as the sum of production for Bardur-owned cities. A positive stepwise SPT change is multiplied by 5; a zero or negative change uses multiplier 1. This asymmetry makes productive economic actions more salient during the short horizon.

## Active shaping

The wrapper currently adds signals for:

- capturing new cities (3 to 6, capped by the number captured on the step);
- revealing uncaptured villages (+1 each);
- moving closer to a visible uncaptured village (+0.5);
- moving onto a village (+1);
- movement breadcrumbs toward new village progress (+0.5);
- clearing fog (+0.08 per newly visible tile, capped at five tiles per step);
- moving onto a visible neutral village (+5), or missing that opportunity with the selected move (-2);
- a movement that reveals no fog and makes no useful village progress (-0.35).

Two formerly considered hold/move-off-village shaping terms are present as diagnostics but currently have zero weight. `POLYVISION_RESOURCE_GATHER_UPGRADE_FILTER_ENABLED` controls an action filter, not a reward term, and defaults off.

## Terminal reward

Terminal-SPT reward is part of the standard Phase-1 reward configuration (enabled by default). Completion of Turn 10 adds a terminal return bonus:

```text
base_weight × final_SPT
+ over_10_weight × max(0, final_SPT - 10)
+ over_15_weight × max(0, final_SPT - 15)
```

Default weights are `1.0`, `2.0`, and `3.0` respectively and can be overridden with `POLYVISION_TERMINAL_SPT_BASE_WEIGHT`, `POLYVISION_TERMINAL_SPT_OVER_10_WEIGHT`, and `POLYVISION_TERMINAL_SPT_OVER_15_WEIGHT`.

The terminal bonus can be explicitly disabled for controlled ablations or backward comparisons by setting the environment variable `POLYVISION_TERMINAL_SPT_REWARD_ENABLED=0`.

### Terminal-SPT Reference Experiment (Phase 1 v3 Seed3 16M)

The active current reference model, [Phase 1 v3 Seed3 16M Terminal-SPT Reference Run](results/Phase1_V3_Seed3_16M_TerminalSPT_Reference_Run.md), was trained with `POLYVISION_TERMINAL_SPT_REWARD_ENABLED=1` and default weights ($w_{\text{base}}=1.0$, $w_{>10}=2.0$, $w_{>15}=3.0$).

In this 16M single-seed training experiment, adding the non-stepwise terminal SPT incentive was associated with major behavioral and capability improvements over the same-seed baseline without terminal SPT:
- **Pristine test argmax SPT:** Improved from **16.88** to **19.34** (+2.46 SPT)
- **Validation argmax SPT:** Improved from **17.18** to **19.87** (+2.69 SPT)
- **Deterministic Forestry adoption:** Rose from **0.0%** (`0 / 250` validation maps) to **99.2%** (`248 / 250` validation and test maps)
- **Mode hierarchy inversion:** Deterministic argmax outperformed stochastic sampling on test (19.34 vs 18.96), whereas the non-terminal baseline had sampled outperforming argmax (17.55 vs 16.88)

> [!CAUTION]
> **Single-Seed Scope:**
> This comparison reflects a single-seed experiment (Seed 3). While the result is consistent with Terminal-SPT reward shaping materially resolving the delayed economic investment hurdle, it does not constitute multi-seed statistical proof of causality.

## Training Reward vs. Evaluation Metric

It is critical to distinguish between:
1. **Training Reward Objective:** The dense step-level shaping bonuses and terminal SPT multiplier are used exclusively during training rollouts to compute PPO surrogate loss gradients.
2. **Policy Observations at Evaluation Time:** Reward signals and return accumulations are **never** part of the policy's observation vector ($505$-dimensional spatial/economic state) during inference or evaluation.
3. **Primary Evaluation Metric:** Final Turn-10 stars per turn (SPT) on held-out maps is the primary capability metric. Total shaped training return is a diagnostic scalar and must not be treated as a measure of policy capability.

Because Phase 1 is shaped and combat-restricted, its return is not a measure of full-game strength. Report final SPT, city count, expansion timing, research, legality/fallback rates, and raw shaped return separately.

## Research Status on Shaping and Delayed Economic Returns

The interaction between dense step-level shaping rewards (such as immediate fruit/animal gathering deltas and fog clearance bounties) and multi-step delayed economic investments (such as Tier-2 Forestry research) was historically problematic under pure step shaping, where deterministic argmax exhibited an Organization monoculture (analyzed in detail in the historical [Phase 1 v3 Seed3 16M Behavioral Failure Analysis](results/Phase1_V3_Seed3_16M_Behavioral_Failure_Analysis.md)).

Enabling the non-stepwise Terminal-SPT bonus established a strong terminal credit-assignment signal that successfully unlocked deterministic Forestry adoption (99.2% on pristine test) and set the new current benchmark of **19.34 test argmax SPT**. Future work will investigate multi-seed confirmation and whether simplified or normalized reward formulations achieve similar or superior credit assignment.
