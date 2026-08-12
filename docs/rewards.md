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

The terminal SPT bonus is disabled by default. When `POLYVISION_TERMINAL_SPT_REWARD_ENABLED=1`, completion of Turn 10 adds:

```text
base_weight × final_SPT
+ over_10_weight × max(0, final_SPT - 10)
+ over_15_weight × max(0, final_SPT - 15)
```

Default weights are 1, 2, and 3 respectively and can be changed with the matching `POLYVISION_TERMINAL_SPT_*_WEIGHT` variables. Record these values with experiments; enabling this option changes the optimization objective.

## Metrics are not all rewards

The environment reports many economy, research, exploration, action-quality, and tactical-mistake metrics. These are diagnostics unless explicitly included above. In particular, final T10 SPT and training-summary charts should not be mistaken for the per-step reward or for controlled held-out evaluation.

Because Phase 1 is shaped and combat-restricted, its return is not a measure of full-game strength. Report final SPT, city count, expansion timing, research, legality/fallback rates, and raw shaped return separately.
