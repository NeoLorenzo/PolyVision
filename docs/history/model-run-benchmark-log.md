> **Historical document**
>
> This registry preserves evidence from multiple older PolyVision interfaces. It is not current usage documentation, and its commands, paths, dimensions, and results must not be applied to the current environment without restoring the matching historical revision.

# PolyVision model run benchmark log

Last updated: 2026-08-14

This document is the canonical index of historical PolyVision model runs, their changelog names, recorded training configuration, and the performance evidence currently available in the repository.

## Critical Historical Compatibility Warning

> **Historical runs are not assumed to be comparable with one another or runnable in the current environment.** PolyVision's observation layout, legal-action interface, feature dimensions, action filtering, reward shaping, map pools, episode telemetry, Java bridge, and training code changed repeatedly across these experiments. A checkpoint can load successfully and still be evaluated under the wrong task contract.

The only defensible way to re-test an old run is to find the commit that produced it, restore that historical code and dependency environment in an isolated checkout or worktree, restore the matching checkpoint and action-interface metadata, and run the evaluation there. Results produced by loading an old checkpoint into the current environment must be treated as compatibility experiments, not reproductions of the original run.

The tables below therefore preserve evidence; they do not assert that every row is directly comparable.

## Data Sources and Interpretation

- W&B source export: [`wandb_export_2026-08-14T11_37_19.454+01_00.csv`](../../wandb_export_2026-08-14T11_37_19.454+01_00.csv), exported on 2026-08-14. It contains exactly the 29 registered runs below.
- Changelog identity mapping: the plain-English label to W&B/run-folder mapping has been preserved verbatim from the previous version of this file and the changelog history.
- Dedicated evaluation artifacts: [`outputs/evaluations/20260814T110912Z_validation_canonical/`](../../outputs/evaluations/20260814T110912Z_validation_canonical/) contains the current canonical 3,000-episode validation result, while [`outputs/org_only_oracle_vs_ppo/`](../../outputs/org_only_oracle_vs_ppo/) preserves an older-interface diagnostic comparison.
- Local `runs/` and `wandb/` directories are gitignored. Their event files and checkpoints are not durable repository evidence unless separately archived.

### Evidence classes

1. **Dedicated evaluation:** repeated episodes under an explicitly recorded evaluation protocol. This is the preferred evidence for model quality.
2. **W&B training summary snapshot:** the final value retained by W&B for a logged scalar. Depending on the metric and trainer version, an episode metric can represent the last completed episode or last logging window—not the run-wide mean.
3. **Historical metadata only:** run identity or configuration was retained, but the corresponding performance metric was never logged.

Do not rank models from the W&B snapshot table alone. Missing values mean “not present in the export,” not zero. Historical model labels such as `(2.5M)` are preserved identifiers and may differ from the exported `global_step` or configured target.

## Canonical Changelog Name ↔ Run Name Mapping

This mapping is deliberately kept separate and explicit because the run-folder identifier is the join key across the changelog, W&B, local checkpoints, TensorBoard events, and any future recovered evaluation.

| Changelog / plain-English model name | W&B and run-folder name |
|---|---|
| `Phase1-MVP-001 (2.5M)` | `Tribes-v0__ppo__1__1777901752` |
| `Phase1-Stability-002 (1M)` | `Tribes-v0__ppo__1__1777909271` |
| `Phase1-Learning-003 (1.5M)` | `Tribes-v0__ppo__1__1777911407` |
| `Phase1-Learning-004 (1.25M)` | `Tribes-v0__ppo__1__1777914322` |
| `Phase1-Learning-005 (1.25M)` | `Tribes-v0__ppo__1__1777917066` |
| `Phase1-Learning-006 (1M)` | `Tribes-v0__ppo__1__1777919819` |
| `Phase1-Learning-006 (43.25M)` | `Tribes-v0__ppo__1__1777921614` |
| `Phase1-Learning-007 (1M)` | `Tribes-v0__ppo__1__1777986222` |
| `Phase1-Learning-008 (1M)` | `Tribes-v0__ppo__1__1777990600` |
| `Phase1-Learning-009 (1M)` | `Tribes-v0__ppo__1__1777995722` |
| `Phase1-Learning-010 (1M)` | `Tribes-v0__ppo__1__1778000233` |
| `Phase1-Learning-010 (2.75M)` | `Tribes-v0__ppo__1__1778008176` |
| `Phase1-Generalizing-011 (33M)` | `Tribes-v0__ppo__1__1778027687` |
| `Phase1-Generation-012 (4M)` | `Tribes-v0__ppo__1__1778077298` |
| `Phase1-Learning-013 (4.5M)` | `Tribes-v0__ppo__1__1778088607` |
| `Phase1-Data-014 (3M)` | `Tribes-v0__ppo__1__1778104307` |
| `Phase1-Data-015 (1M)` | `Tribes-v0__ppo__1__1778147403` |
| `Phase1-Learning-016 (4M)` | `Tribes-v0__ppo__1__1778158665` |
| `Phase1-Learning-017 (1M)` | `Tribes-v0__ppo__1__1778175810` |
| `Phase1-Data-018 (250K)` | `Tribes-v0__ppo__1__1778180156` |
| `Phase1-Data-018 (6M)` | `Tribes-v0__ppo__1__1778183254` |
| `Phase1-Data-019 (600K)` | `Tribes-v0__ppo__1__1778260387` |
| `Phase1-Data-019 (5M)` | `Tribes-v0__ppo__1__1778266653` |
| `Phase1-Learning-020 (1.5M)` | `Tribes-v0__ppo__1__1778324123` |
| `Phase1-Data-021 (2M)` | `Tribes-v0__ppo__1__1778358997` |
| `Phase1-Data-022 (1M)` | `Tribes-v0__ppo__1__1778673883` |
| `Phase1-Data-023 (6M)` | `Tribes-v0__ppo__1__1778695665` |
| `Phase1-Map_Gen-027 (500K genuine-map validation)` | `Tribes-v0__ppo__1__1786565606` |
| `Phase1-First_Eval-030 (10M, seed 1)` | `Tribes-v0__Phase1-Scientific-Train-Seed1__1__1786668037` |

## Exported Run Configuration and Completion

`Steps` is exported `global_step / total_timesteps`; it is more precise than the historical name. `Interface` is `max_legal_actions / legal_action_feature_dim`. An em dash means the field was not logged. `Final SPS` is the last W&B summary value, not a controlled throughput benchmark.

| Model name | W&B state | Created (UTC) | Steps: logged / planned | Actor | Interface | Envs × rollout steps | Final SPS |
|---|---|---:|---:|---|---:|---:|---:|
| `Phase1-MVP-001 (2.5M)` | killed | 2026-05-04 | 2,964,492 / 5,000,000 (59.3%) | — | — | 12×128 | 1,221 |
| `Phase1-Stability-002 (1M)` | killed | 2026-05-04 | 1,209,960 / 5,000,000 (24.2%) | — | — | 12×128 | 953 |
| `Phase1-Learning-003 (1.5M)` | killed | 2026-05-04 | 1,614,912 / 5,000,000 (32.3%) | — | — | 12×128 | 968 |
| `Phase1-Learning-004 (1.25M)` | finished | 2026-05-04 | 1,499,136 / 1,500,000 (99.9%) | — | — | 12×128 | 1,045 |
| `Phase1-Learning-005 (1.25M)` | finished | 2026-05-04 | 1,499,136 / 1,500,000 (99.9%) | — | — | 12×128 | 708 |
| `Phase1-Learning-006 (1M)` | killed | 2026-05-04 | 1,007,316 / 1,500,000 (67.2%) | — | — | 12×128 | 664 |
| `Phase1-Learning-006 (43.25M)` | killed | 2026-05-04 | 43,376,640 / 45,000,000 (96.4%) | — | — | 12×128 | 839 |
| `Phase1-Learning-007 (1M)` | killed | 2026-05-05 | 1,108,272 / 1,500,000 (73.9%) | — | — | 12×128 | 718 |
| `Phase1-Learning-008 (1M)` | killed | 2026-05-05 | 1,156,764 / 1,500,000 (77.1%) | — | — | 12×128 | 683 |
| `Phase1-Learning-009 (1M)` | finished | 2026-05-05 | 999,936 / 1,000,000 (100.0%) | — | — | 12×128 | 677 |
| `Phase1-Learning-010 (1M)` | finished | 2026-05-05 | 999,936 / 1,000,000 (100.0%) | — | — | 12×128 | 624 |
| `Phase1-Learning-010 (2.75M)` | killed | 2026-05-05 | 2,776,668 / 4,000,000 (69.4%) | — | — | 12×128 | 626 |
| `Phase1-Generalizing-011 (33M)` | killed | 2026-05-06 | 33,130,040 / 35,000,000 (94.7%) | — | — | 20×128 | 869 |
| `Phase1-Generation-012 (4M)` | finished | 2026-05-06 | 4,497,920 / 4,500,000 (100.0%) | — | — | 20×128 | 791 |
| `Phase1-Learning-013 (4.5M)` | finished | 2026-05-06 | 4,748,800 / 4,750,000 (100.0%) | — | — | 20×128 | 811 |
| `Phase1-Data-014 (3M)` | killed | 2026-05-06 | 3,982,740 / 40,000,000 (10.0%) | — | — | 20×128 | 99 |
| `Phase1-Data-015 (1M)` | finished | 2026-05-07 | 998,400 / 1,000,000 (99.8%) | `legal_only` | 1024/— | 20×128 | 378 |
| `Phase1-Learning-016 (4M)` | finished | 2026-05-07 | 3,998,720 / 4,000,000 (100.0%) | `legal_only` | 1024/— | 20×128 | 403 |
| `Phase1-Learning-017 (1M)` | finished | 2026-05-07 | 998,400 / 1,000,000 (99.8%) | `legal_only` | 1024/— | 20×128 | 383 |
| `Phase1-Data-018 (250K)` | killed | 2026-05-07 | 481,700 / 1,000,000 (48.2%) | `legal_features` | 1024/22 | 20×128 | 163 |
| `Phase1-Data-018 (6M)` | finished | 2026-05-07 | 5,998,080 / 6,000,000 (100.0%) | `legal_features` | 1024/22 | 20×128 | 169 |
| `Phase1-Data-019 (600K)` | finished | 2026-05-08 | 599,040 / 600,000 (99.8%) | `legal_features` | 1024/22 | 20×128 | 102 |
| `Phase1-Data-019 (5M)` | finished | 2026-05-08 | 4,999,680 / 5,000,000 (100.0%) | `legal_features` | 1024/22 | 20×128 | 111 |
| `Phase1-Learning-020 (1.5M)` | killed | 2026-05-09 | 1,537,020 / 2,000,000 (76.9%) | `legal_features` | 1024/22 | 20×128 | 82 |
| `Phase1-Data-021 (2M)` | finished | 2026-05-09 | 1,999,360 / 2,000,000 (100.0%) | `legal_features` | 1024/22 | 20×128 | 92 |
| `Phase1-Data-022 (1M)` | finished | 2026-05-13 | 998,400 / 1,000,000 (99.8%) | `legal_features` | 1024/22 | 20×128 | 156 |
| `Phase1-Data-023 (6M)` | finished | 2026-05-14 | 5,998,080 / 6,000,000 (100.0%) | `legal_features` | 1024/42 | 20×128 | 164 |
| `Phase1-Map_Gen-027 (500K genuine-map validation)` | finished | 2026-08-12 | 499,200 / 500,000 (99.8%) | `legal_features` | 256/42 | 20×128 | 291 |
| `Phase1-First_Eval-030 (10M, seed 1)` | finished | 2026-08-14 | 9,999,360 / 10,000,000 (100.0%) | `legal_features` | 256/42 | 20×128 | 327 |

## Final W&B Training Summary Snapshots

These are evidence-class 2 snapshots, **not standardized evaluation results**. `Last T10 SPT` uses the last available value from `charts/final_spt_t10`, falling back to the older `charts/episode_end_spt` schema. The other episode fields likewise reflect W&B's final retained summary value. `Explained variance` is a training critic diagnostic, not a gameplay score.

| Model name | Last T10 SPT | Custom SPT return | Villages | Avg city level | Fog cleared | Organization rate | Techs | Explained variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Phase1-MVP-001 (2.5M)` | — | 2.0 | — | — | — | — | — | 0.358 |
| `Phase1-Stability-002 (1M)` | 4.0 | 4.0 | — | — | — | — | — | 0.250 |
| `Phase1-Learning-003 (1.5M)` | 4.0 | 4.0 | — | — | — | — | — | 0.237 |
| `Phase1-Learning-004 (1.25M)` | 5.0 | 5.0 | — | — | — | — | — | 0.626 |
| `Phase1-Learning-005 (1.25M)` | 4.0 | 4.0 | — | — | — | — | — | 0.654 |
| `Phase1-Learning-006 (1M)` | 5.0 | 5.0 | — | — | — | — | — | 0.441 |
| `Phase1-Learning-006 (43.25M)` | 8.0 | 8.0 | — | — | — | — | — | 1.000 |
| `Phase1-Learning-007 (1M)` | 5.0 | 5.0 | — | — | — | — | — | 0.614 |
| `Phase1-Learning-008 (1M)` | 7.0 | 7.0 | — | — | — | — | — | 0.548 |
| `Phase1-Learning-009 (1M)` | 4.0 | 4.0 | — | — | — | — | — | 0.328 |
| `Phase1-Learning-010 (1M)` | 7.0 | 7.0 | — | — | — | — | — | 0.664 |
| `Phase1-Learning-010 (2.75M)` | 7.0 | 7.0 | — | — | — | — | — | 0.650 |
| `Phase1-Generalizing-011 (33M)` | 7.0 | 7.0 | — | — | — | — | — | 0.805 |
| `Phase1-Generation-012 (4M)` | 4.0 | 4.0 | 1.0 | — | 7.0 | — | — | 0.534 |
| `Phase1-Learning-013 (4.5M)` | 7.0 | 7.0 | 2.0 | — | 4.0 | — | — | 0.646 |
| `Phase1-Data-014 (3M)` | 5.0 | 5.0 | 2.0 | — | 15.0 | — | — | 0.523 |
| `Phase1-Data-015 (1M)` | 7.0 | 7.0 | 2.0 | — | 23.0 | — | — | 0.637 |
| `Phase1-Learning-016 (4M)` | 7.0 | 7.0 | 2.0 | — | 37.0 | — | — | 0.688 |
| `Phase1-Learning-017 (1M)` | 7.0 | 7.0 | 2.0 | 2.00 | 30.0 | 0.0% | 0.0 | 0.521 |
| `Phase1-Data-018 (250K)` | 4.0 | 4.0 | 1.0 | 2.00 | 32.0 | 0.0% | 0.0 | 0.377 |
| `Phase1-Data-018 (6M)` | 4.0 | 4.0 | 1.0 | 2.00 | 49.0 | 0.0% | 0.0 | 0.710 |
| `Phase1-Data-019 (600K)` | 9.0 | 9.0 | 3.0 | 2.00 | 32.0 | 0.0% | 1.0 | 0.553 |
| `Phase1-Data-019 (5M)` | 4.0 | 4.0 | 1.0 | 2.00 | 43.0 | 0.0% | 0.0 | 0.666 |
| `Phase1-Learning-020 (1.5M)` | 9.0 | 9.0 | 3.0 | 2.00 | 48.0 | 50.0% | 1.5 | 0.697 |
| `Phase1-Data-021 (2M)` | 6.0 | 6.0 | 2.0 | 2.00 | 30.0 | 50.0% | 1.5 | 0.733 |
| `Phase1-Data-022 (1M)` | 10.0 | 10.0 | 3.0 | 2.33 | 62.0 | 100.0% | 1.0 | 0.614 |
| `Phase1-Data-023 (6M)` | 14.0 | 14.0 | 4.0 | 2.25 | 51.0 | 100.0% | 2.0 | 0.817 |
| `Phase1-Map_Gen-027 (500K genuine-map validation)` | 14.0 | 14.0 | 4.0 | 2.25 | 50.0 | 100.0% | 3.5 | 0.446 |
| `Phase1-First_Eval-030 (10M, seed 1)` | 10.0 | 10.0 | 3.0 | 2.00 | 35.0 | 100.0% | 2.5 | 0.848 |

### Phase1-First_Eval-030 first evaluation candidate

The new export records `Tribes-v0__Phase1-Scientific-Train-Seed1__1__1786668037` as the first model to be evaluated under `Phase1-First_Eval-030`. The seed-1 run finished after 30,604 seconds and logged global step 9,999,360 of a planned 10,000,000. It used the `legal_features` actor with 256 legal-action slots and 42 features per slot, 20 environments, 128 rollout steps, batch size 2,560, four 640-sample minibatches, and four update epochs. PPO used learning rate 0.00025 with linear annealing, gamma 0.99, GAE lambda 0.95, clip coefficient 0.2 with clipped value loss, entropy coefficient 0.01, value coefficient 0.5, maximum gradient norm 0.5, and normalized advantages. CUDA and deterministic PyTorch mode were enabled.

The action interface was force-revalidated against 10,000 validation states using validation seed 12,345 with caching enabled. Step diagnostics were logged every three updates; the configured maximum fallback-end-turn and illegal-sample rates were both 0.0001. Models were configured to save every 500,000 steps.

The final W&B summary retained 327 SPS, T10 SPT 10, three villages, average city level 2.0, 35 fog tiles cleared, 73.33% village capture, 100% Organization research, 2.5 researched technologies, average second-city capture turn 5.5, and critic explained variance 0.848. **These values are a single retained training-summary snapshot and may not accurately represent typical or held-out model performance.** They must not be treated as a controlled comparison or as the result of `Phase1-First_Eval-030`. The training and evaluation work performed in the current session, with its recorded protocol and repeated outcomes, is the more important evidence and should take precedence over this snapshot.

### Phase1-Map_Gen-027 genuine-map training validation

The latest export records `Tribes-v0__ppo__1__1786565606` as a finished 500K validation run for the genuine-map update. It reached global step 499,200 of 500,000 in 1,811 seconds with 20 environments, 128 rollout steps, the `legal_features` actor, 256 legal-action slots, and 42 legal-action features. The final W&B summary retained 291 SPS, T10 SPT 14, four villages, average city level 2.25, 50 fog tiles cleared, 100% Organization research, 3.5 researched technologies, average second-city capture turn 6.5, and 62.5% village capture.

These values establish that the new 11×11 genuine-map contract completed a substantial PPO validation run with the 256-slot interface. They remain final training-summary snapshots rather than a held-out, multi-episode evaluation and should not be interpreted as a controlled improvement over historical models. The failed earlier 128-slot attempt is intentionally not registered as a benchmark row for this update.

## Dedicated Multi-Episode Evaluation Evidence

### Phase1-First_Eval-030 canonical validation

The first properly trained Phase 1 model, `Phase1-First_Eval-030 (10M, seed 1)`, completed the full canonical validation suite under evaluation ID `20260814T110912Z_validation_canonical`: 250 PPO-argmax episodes, 1,250 PPO-sampled episodes, 250 visible-greedy episodes, and 1,250 random-legal episodes across all 250 manifest-verified validation maps.

| Policy | Maps | Episodes | Mean T10 SPT | Median | Map-level 95% CI |
|---|---:|---:|---:|---:|---:|
| PPO argmax | 250 | 250 | 14.576 | 14.0 | [14.204, 14.948] |
| PPO sampled | 250 | 1,250 | 13.678 | 13.4 | [13.416, 13.936] |
| Visible greedy | 250 | 250 | 7.128 | 7.0 | [6.944, 7.320] |
| Random legal | 250 | 1,250 | 5.897 | 5.8 | [5.811, 5.986] |

PPO argmax beat visible greedy on all 250 paired maps with mean difference +7.448 SPT (95% CI [7.132, 7.780]). Sources: [`summary.json`](../../outputs/evaluations/20260814T110912Z_validation_canonical/summary.json), [`per_map.csv`](../../outputs/evaluations/20260814T110912Z_validation_canonical/per_map.csv), [`episodes.jsonl`](../../outputs/evaluations/20260814T110912Z_validation_canonical/episodes.jsonl), and the [detailed validation reflection](../results/PolyVision_Phase1_Validation_Results.md).

This was the first long Phase 1 scientific model, **not the final Phase 1 model**. Its final disposition is **historical / shelved**: it will not proceed to pristine model-capability test or canonical human comparison.

The later [scripted-opening audit](../results/Phase1_Scripted_Opening_Audit.md) established that this checkpoint was trained and evaluated under the historical mixed-opening contract: training maps were 55.02% two-unit / 44.98% one-unit handoffs, while validation maps were 56.40% / 43.60%. The validation comparison remains internally valid for that shared historical task, but it must not be described as a universal two-unit-opening result.

The environment was subsequently corrected as `v2_guaranteed_two_unit` and passed all 5,517 maps. Because this changes approximately 45% of training initial states, the old checkpoint is preserved as a meaningful developmental experiment but a new model must be trained from scratch. Its historical sidecar lacks the v2 opening identity and therefore fails current compatibility by default.

### Historical Organization-only oracle comparison

The earlier 500-episode Organization-only oracle comparison evaluated `Phase1-Data-019 (5M)` / `Tribes-v0__ppo__1__1778266653`:

| Metric | PPO | Organization-only oracle |
|---|---:|---:|
| Mean final T10 SPT | 10.518 | 8.790 |
| Median final T10 SPT | 10.000 | 10.000 |
| P25 / P75 / P90 final T10 SPT | 8 / 13 / 14 | 7 / 10 / 13 |
| Maximum final T10 SPT | 16 | 13 |
| Mean city count | 3.316 | 2.654 |
| Mean turn second city captured | 5.366 | 5.138 |
| Organization research rate | 0.0% | 76.8% |
| Mean fruit harvested | 1.666 | 2.120 |
| Mean fog tiles cleared | 53.274 | 37.058 |

Sources: [`report.md`](../../outputs/org_only_oracle_vs_ppo/run_20260509_104703/report.md), [`summary.json`](../../outputs/org_only_oracle_vs_ppo/run_20260509_104703/summary.json), and [`per_episode_results.csv`](../../outputs/org_only_oracle_vs_ppo/run_20260509_104703/per_episode_results.csv).

This evaluation is useful but not a universal leaderboard result:

- it evaluates only one registered checkpoint;
- it used the historical 1024-slot / 22-feature interface and its historical environment;
- the oracle uses a different policy contract from PPO;
- the recorded round-robin sequence gave PPO and oracle different map files at the same `episode_index`, so the output is not a strictly paired-map comparison;
- it reports one training seed and does not provide confidence intervals or across-seed variation.

An earlier eight-episode smoke comparison was intentionally removed from the current tree because its sample was too small to retain as useful benchmark evidence.

## What Can and Cannot Be Compared

Within this document, compare values only when all of the following are known to match:

- source commit and restored dependency/runtime environment;
- observation schema and action-interface fingerprint;
- actor mode, legal slot count, and legal-action feature dimension;
- map pool, selection mode, exact map IDs, and episode seeds;
- scripted opening, horizon, action filters, reward shaping, and terminal reward settings;
- deterministic versus stochastic action selection;
- checkpoint step and checkpoint file hash;
- metric definition and aggregation method.

If any item is unknown, record the result as historical evidence rather than claiming a model-to-model improvement.

## Required Format for Future Benchmark Entries

Future runs should retain the canonical mapping above and add a dedicated evaluation record containing:

- changelog model name and exact W&B/run-folder name;
- source commit, dirty-worktree status, checkpoint path, checkpoint SHA-256, and action-interface metadata;
- environment/dependency restoration instructions;
- complete training configuration and actual final global step;
- evaluation protocol version, held-out map manifest, episode seeds, and episode count;
- mean, median, standard deviation, confidence interval, and percentiles for final T10 SPT;
- economy, expansion, research, exploration, legality/fallback, and runtime metrics;
- raw per-episode output plus a machine-readable summary;
- clearly identified random, scripted, previous-best, and privileged/oracle baselines;
- an explicit comparability statement naming which earlier results, if any, share the same protocol.

Until those fields exist, this log should be read as a carefully indexed historical record, not as proof that the row with the largest final W&B scalar is the best PolyVision model.
