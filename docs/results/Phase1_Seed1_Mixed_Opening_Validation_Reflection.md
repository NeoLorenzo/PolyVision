# Phase 1 Seed-1 mixed-opening validation reflection

## Experiment identity

- Model/run: `Phase1-Scientific-Train-Seed1`
- Checkpoint: historical Seed-1 checkpoint and its unchanged sidecar
- Validation evaluation: `outputs/evaluations/20260814T110912Z_validation_canonical`
- Date: 2026-08-14
- Training pool: 5,000 genuine maps
- Validation pool: 250 held-out genuine maps
- Primary metric: Turn-10 SPT
- Historical opening contract: `v1_mixed_capital_regression`
- Final disposition: **historical / shelved**

This run was intended to be the first proper Phase 1 scientific model: PPO trained on 5,000 genuine maps, followed by a canonical evaluation on 250 held-out validation maps. Its metrics are real and its artifacts are preserved. The later environment-contract audit changed the scientific interpretation, not the recorded observations.

## Validation result

Deterministic PPO argmax achieved mean Turn-10 SPT **14.576**, median **14**, range **6–22**, p5 **10**, p25 **13**, p75 **17**, and p95 **19**. Its approximately 95% confidence interval for the mean was **[14.20, 14.95]**. The PPO sampled map-level mean was **13.678**; visible greedy scored **7.128** and random legal **5.897**.

PPO argmax beat visible greedy on every validation map: **250 W / 0 T / 0 L**, mean delta **+7.448**. It also beat random legal **250 W / 0 T / 0 L**, mean delta **+8.679**. PPO therefore learned useful behavior that generalized strongly to held-out maps under the historical task.

## Discovery during the human benchmark

The initial interpretation was that Seed-1 was strong enough to proceed to pristine test. That conclusion was reasonable from the validation evidence available at the time, but it is now superseded.

At the beginning of the human benchmark, the interface displayed `Turn 2 | stars=5 | SPT=4 | cities=1 | visible units=1`. The intended opening required two warriors. The attempt on `map_003696.csv` was stopped; no canonical human score was recorded, and completed first attempts remained zero.

## Historical opening audit

The behavior-preserving audit reset every manifest map using the same wrapper as training, validation, and human play. It did not load a model, take a policy-controlled action, or compute test capability scores.

| Pool | One unit | Two units | Two-unit rate |
|---|---:|---:|---:|
| Train | 2,249 / 5,000 | 2,751 / 5,000 | 55.02% |
| Validation | 109 / 250 | 141 / 250 | 56.40% |
| Test contract | 112 / 250 | 138 / 250 | 55.20% |
| Human benchmark | 9 / 17 | 8 / 17 | 47.06% |
| Total | 2,479 / 5,517 | 3,038 / 5,517 | 55.07% |

Both scripted moves succeeded on 100% of maps. No swallowed exception caused the bug. On 44.93% of maps, the existing Turn-1 scorer deliberately selected a move returning the original warrior to its capital. With the capital occupied, Java correctly omitted the warrior-spawn action. The opening treated that missing action as optional and handed control over with one unit.

On `map_003696.csv`, the original warrior moved `(3,5) → (2,4)` on Turn 0, then regressed `(2,4) → (3,5)` on Turn 1. Spawn was unavailable, producing the observed one-unit handoff at 5 stars, 4 SPT, and one city.

## Revised scientific interpretation

Training and validation used the same wrapper behavior and closely matched mixed-opening distributions: 55.02% versus 56.40% two-unit starts. The canonical validation therefore remains internally valid for `v1_mixed_capital_regression`. It demonstrates strong held-out learning for that historical task.

It does not answer the intended question for a guaranteed-two-unit Phase 1 environment. Correcting the opening materially changes approximately 45% of training initial states. Seed-1 will consequently receive no pristine model test, no canonical human comparison, and no reuse as though the task contract were unchanged. It is preserved as one of the substantial developmental experiments leading to the eventual rigorous Phase 1 tests.

The 250 test maps were reset only for scripted-opening contract verification. They remain pristine with respect to model capability evaluation and development feedback: no checkpoint was loaded, no post-opening policy action was selected, and no model-quality or SPT capability score was produced.

## What this run taught us

- PPO can learn strong behavior through the Phase 1 observation and action interfaces.
- The training pipeline produced a policy that generalized strongly to held-out maps.
- The evaluation suite demonstrated decisive baseline superiority.
- The human benchmark exposed an environment inconsistency before pristine test.
- Test discipline prevented a flawed-contract checkpoint from consuming the final capability benchmark.
- Initial-state invariants require the same scientific rigor as rewards and action semantics.

## Corrective action

The corrected contract is `v2_guaranteed_two_unit`. The Turn-1 scripted candidate mask excludes the Bardur capital only while moving the original opening warrior; all remaining candidates retain the historical scorer and normal policy gameplay is unchanged. The second-warrior spawn is mandatory, required phases raise diagnostic errors, and reset validates the complete Turn-2 handoff: turn counter 2, 5 stars, 4 SPT, one city, the original warrior away from capital, and a new warrior occupying capital.

Post-fix validation is recorded in the [scripted-opening audit](Phase1_Scripted_Opening_Audit.md). The historical audit remains at `outputs/opening_audit/20260814_phase1_opening_full_seed42`; corrected artifacts are separate.

## Final status

`Phase1-Scientific-Train-Seed1` is **shelved, not discarded**. Its canonical validation remains preserved and meaningful for the historical mixed-opening environment, but the prior “proceed to test” conclusion is superseded. A new model must be trained from scratch under `v2_guaranteed_two_unit`, validated again, and only then considered for pristine model-capability test.
