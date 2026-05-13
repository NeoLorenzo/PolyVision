# Changelog

All notable changes to this project are documented in this file.

## [Phase1-Data-022] - 2026-05-13

### Scope
- Improve end-to-end Phase-1 SPS instrumentation across Java bridge, Python wrapper, and PPO trainer timing/reporting.
- Add fast-path legal-action batching and optional equivalence checks to validate behavior parity with legacy per-action JSON flow.
- Add solo/no-opponent execution mode controls to stabilize single-tribe throughput experiments.
- Add generated profiling outputs, validator snapshots, and a new 12x12 Bardur phase-1 level pool for data-driven iteration.

### Implemented
- Updated Java bridge wrapper behavior and profiling toggles in `pol_env/Tribes/py/gym_env.py`:
  - added runtime env switches:
    - `POLYVISION_SOLO_NO_OPPONENT_MODE`,
    - `POLYVISION_PROFILE_SPS`,
    - `POLYVISION_BATCH_LEGAL_ACTION_FETCH`,
    - `POLYVISION_BATCH_LEGAL_FETCH_EQUIV_CHECK`,
    - `POLYVISION_BATCH_LEGAL_FETCH_EQUIV_CHECK_EVERY_N_STEPS`.
  - added per-step/per-legal-call profiling state:
    - `_list_actions_call_count`,
    - `_last_step_profile`,
    - `_last_list_actions_profile`.
  - instrumented `step(...)` with detailed timing buckets for Java apply, parsing, legal regeneration, reward assembly, and info construction.
  - split legal action fetch paths into:
    - `_list_actions_legacy(...)` (existing per-action JSON path),
    - `_list_actions_batch(...)` (new `listActionsJsonBatch()` array payload path).
  - added optional batch-vs-legacy equivalence assertion path to catch raw action mismatches during rollout.
- Expanded wrapper-level instrumentation and legality diagnostics in `pol_env/Tribes/py/register_env.py`:
  - expanded `info_mode` handling to include `"train"` in addition to `"fast"` and `"debug"`.
  - added profiling controls:
    - `POLYVISION_PROFILE_SPS`,
    - `POLYVISION_PROFILE_EVERY_N_STEPS`.
  - added feature/legal-equivalence validation controls:
    - `POLYVISION_FEATURE_EQUIV_CHECK`,
    - `POLYVISION_LEGAL_SUMMARY_EQUIV_CHECK`,
    - `POLYVISION_LEGAL_SUMMARY_EQUIV_CHECK_EVERY_N_STEPS`,
    - `POLYVISION_FILTER_EQUIV_CHECK`,
    - `POLYVISION_FILTER_EQUIV_CHECK_EVERY_N_STEPS`,
    - `POLYVISION_BATCH_LEGAL_FETCH_EQUIV_CHECK`,
    - `POLYVISION_BATCH_LEGAL_FETCH_EQUIV_CHECK_EVERY_N_STEPS`.
  - added and wired legal-summary/equivalence helpers:
    - `_build_step_legal_action_summary(...)`,
    - `_assert_legal_summary_equivalence(...)`.
  - added village-lookup optimization helpers used by tactical/legal scans:
    - `_build_village_lookup_mask(...)`,
    - `_owned_units_on_visible_uncaptured_village_without_capture_from_sets(...)`,
    - updated `_is_move_to_visible_uncaptured_village(...)` call paths to support precomputed lookup data.
  - added high-granularity `profile_env_reset_*` and `profile_env_step_*` metrics into `info` payload for trainer-side aggregation, including:
    - pre/post fast-forward timing,
    - Java apply sub-stage timing,
    - post-legal filtering/canonicalization/mask-build timing,
    - reward subcomponent timing,
    - info/diagnostic payload build timing,
    - legal-action count and raw-character volume counters.
- Updated Java game-state invalidation support in `pol_env/Tribes/src/core/game/GameState.java`:
  - added `invalidateComputedActions()` to force recomputation of cached legal-action sets after manual turn-state rewrites.
- Updated Java Python bridge API and solo-turn behavior in `pol_env/Tribes/src/core/game/PythonEnv.java`:
  - added solo mode state and accessors:
    - `soloNoOpponentMode`,
    - `setSoloNoOpponentMode(boolean enabled)`,
    - `getSoloNoOpponentMode()`.
  - refactored action serialization with `buildActionJsonObject(...)` to centralize per-action JSON formatting.
  - added `listActionsJsonBatch()` returning one JSON array payload for all legal actions.
  - updated `stepByIndex(...)` end-turn behavior for solo/single-tribe runs:
    - when solo mode is enabled and selected action is `END_TURN`, prevents unintended game-over termination,
    - invalidates cached actions and reinitializes turn/action state to continue training progression.
- Added trainer-side SPS profiling pipeline in `py_rl/cleanrl/cleanrl/ppo.py`:
  - added env-flag helpers:
    - `_env_flag_bool(...)`,
    - `_env_flag_int(...)`.
  - added profiling/stat primitives:
    - `_TimerStats`,
    - `_ScalarStats`,
    - `SPSProfiler` with timer/scalar key maps for environment/trainer stages.
  - added run-time profiler controls:
    - `POLYVISION_PROFILE_SPS`,
    - `POLYVISION_PROFILE_EVERY_N_STEPS`,
    - `POLYVISION_PROFILE_WRITE_JSON`,
    - `POLYVISION_PROFILE_OUTPUT_DIR` (default `outputs/sps_profiles`).
  - integrated profiler measurements across:
    - env reset/step,
    - tensor conversion and rollout extraction,
    - policy/value forward timing,
    - PPO update timing,
    - logging/checkpoint/final-save timing.
  - added scalar handling updates:
    - blocklisted `reward/terminal_spt_bonus` from default scalar passthrough in `log_scalar`,
    - added explicit `economy/episode_end_village_capture_pct_t10` logging path from episode-end infos.
  - added end-of-run profile summary and JSON emission (file names: `sps_profile_<run_name>.json`).
- Added runbook `EFFICIENT_TRAINING_RUN.md`:
  - documents three operating modes (full training, debug, SPS profiling),
  - provides concrete PowerShell command templates with environment flags,
  - records SPS bottleneck hierarchy and a checklist for high-throughput runs,
  - includes guidance on offline W&B sync workflow.
- Added validator cache artifacts under `.cache/action_validator`:
  - `20d1149fe7357ccaf8f0f877999fc2b8ea529d848237c5cfef22bde05e885041.json`:
    - `checked_states=100`, `passed=true`.
  - `2d9aead549c836fe65ff614ff08147cce3b1a5983fe95dccb3b75cd81b35dfc0.json`:
    - `checked_states=10000`, `passed=true`.
  - `da6b5e6cb44fbccf7df9c27d85a8afeddc974d00686cf2da7ede7b6fcf74d40a.json`:
    - `checked_states=100`, `passed=true`.
  - `e43fbbdee14a6623e837c98e1d44188b08d3cfe94e2b4b190a13e131362a51b2.json`:
    - `checked_states=10000`, `passed=true`.
  - each cache snapshot stores action-interface fingerprint payloads/hashes for replay-safe validation reuse against current wrapper/bridge states at capture time.
- Added Phase-1 level pool data in `pol_env/Tribes/levels/phase1_pool_bardur_solo`:
  - new files `phase1_12x12_pool_000.csv` through `phase1_12x12_pool_255.csv` (`256` total CSV maps),
  - each file provides a 12x12 tile-grid map layout encoded in the existing terrain/resource token format used by the Java/Python environment loaders.
- Updated benchmark registry in `model_run_benchmark_log.md`:
  - added `Tribes-v0__ppo__1__1778673883 -> Phase1-Data-022 (1M)`.
- Updated `CHANGELOG.md` with this detailed `[Phase1-Data-022]` entry to keep source-control documentation synchronized with the full pending change set.

## [Phase1-Data-021] - 2026-05-12

### Scope
- Refine Phase-1 wrapper control over resource-gather legality filtering and truncation timing.
- Add human-in-the-loop wrapper play tooling for legal-action inspection and manual stepping.
- Record new validator cache snapshots and human play session output artifacts.

### Implemented
- Updated wrapper configuration and step semantics in `pol_env/Tribes/py/register_env.py`:
  - added `RESOURCE_GATHER_UPGRADE_FILTER_ENABLED_DEFAULT = False`.
  - added runtime env switch:
    - `POLYVISION_RESOURCE_GATHER_UPGRADE_FILTER_ENABLED` parsed into `_resource_gather_upgrade_filter_enabled`.
  - made resource-gather legality gating conditional:
    - `RESOURCE_GATHERING` actions are filtered by `_is_resource_gather_legal_for_upgrade(...)` only when the new runtime toggle is enabled.
    - when toggle is disabled, resource-gather actions are no longer auto-pruned by upgrade-feasibility logic.
  - adjusted Python-side T10 truncation boundary:
    - changed internal truncation check from `turn_count >= MAX_TURNS` to `turn_count > MAX_TURNS`,
    - documented turn-index behavior after the scripted Bardur opening (reset starts at Turn 2),
    - effect: truncation now occurs after completing Bardur Turn 10.
  - exposed runtime filter flag in episode info:
    - added `resource_gather_upgrade_filter_enabled` to `info` payload.
- Added human wrapper runner utility `tools/play_human_t10_wrapper.py`:
  - provides interactive legal-action driving over `TribesGymWrapper` using global action IDs.
  - prints per-step state summary:
    - turn, SPT, stars, city count, unit count, map/seed when available.
  - supports paged legal-action browsing with compact action summaries and detail inspection.
  - supports action selection via:
    - list index,
    - global action ID (`g <gid>`),
    - random pick (`r` / `--auto-random`).
  - includes optional map/visual output:
    - ANSI map display in terminal (`--show-ansi-map`),
    - Java renderer window updates (`--render-java`).
  - includes optional run log persistence (`--save-json`) to `outputs/human_wrapper_runs`.
  - captures step history and final info snapshot in saved session payload.
- Added validator cache artifacts under `.cache/action_validator`:
  - `b30499c4eb7fb352fc36be312ec5342700f0416eed1b0a560d4339522eca71ef.json`,
  - `09b0d252751b4d5cf286ba890b1750afbd63e96fb4c94e34b42e83421801e884.json`.
  - both records:
    - indicate `passed: true`,
    - include full fingerprint payloads for `actor_mode=legal_features`, `states=10000`, `max_legal_actions=1024`, `legal_action_feature_dim=22`,
    - capture updated `register_env.py` hash states and Java action-tree hashes for validator cache reuse.
- Added human wrapper run outputs in `outputs/human_wrapper_runs`:
  - `human_run_20260509_200129.json`:
    - short/early session with `history_count=0` (no executed wrapper steps),
    - includes config/final-info scaffold for run traceability.
  - `human_run_20260509_201725.json`:
    - interactive session with `history_count=33` executed steps,
    - final snapshot includes:
      - `turn_count=10`,
      - `spt=10.0`,
      - `city_count=3`,
      - `stars=5`.
  - both files include run timestamps, config block, final info payload, and per-step history list (when actions were executed).
- Updated benchmark registry in `model_run_benchmark_log.md`:
  - added `Tribes-v0__ppo__1__1778358997 -> Phase1-Data-021 (2M)`.
- Updated `CHANGELOG.md` with this detailed `[Phase1-Data-021]` entry to keep source-control documentation synchronized with the update.

## [Phase1-Learning-020] - 2026-05-09

### Scope
- Add tactical-mistake telemetry and optional terminal SPT bonus shaping for Phase-1 learning runs.
- Expand PPO logging to track tactical error rates and terminal-SST decomposition signals.
- Add tooling to compare an Organization-only oracle against the latest PPO checkpoint and validate terminal SPT bonus behavior.
- Record generated benchmark/evaluation artifacts for reproducibility.

### Implemented
- Updated reward shaping and telemetry in `pol_env/Tribes/py/register_env.py`:
  - added terminal reward configuration defaults:
    - `TERMINAL_SPT_REWARD_ENABLED_DEFAULT = False`,
    - `TERMINAL_SPT_BASE_WEIGHT_DEFAULT = 1.0`,
    - `TERMINAL_SPT_OVER_10_WEIGHT_DEFAULT = 2.0`,
    - `TERMINAL_SPT_OVER_15_WEIGHT_DEFAULT = 3.0`.
  - added new tactical move shaping constants:
    - `MOVE_ONTO_VISIBLE_NEUTRAL_VILLAGE_REWARD = 5.0`,
    - `MOVE_MISS_VISIBLE_NEUTRAL_VILLAGE_PENALTY = 2.0`.
  - added terminal reward runtime state and env-var parsing:
    - `_terminal_spt_reward_enabled`,
    - `_terminal_spt_base_weight`,
    - `_terminal_spt_over_10_weight`,
    - `_terminal_spt_over_15_weight`,
    - `_terminal_spt_bonus_applied_this_episode`.
  - added generic env parsing helpers:
    - `_parse_float_env(...)`,
    - `_parse_bool_env(...)`.
  - introduced optional terminal SPT bonus at T10 truncation:
    - applies once per episode when `POLYVISION_TERMINAL_SPT_REWARD_ENABLED` is enabled,
    - computes components:
      - `terminal_spt_base_component = base_weight * final_spt`,
      - `terminal_spt_over_10_component = over10_weight * max(final_spt - 10, 0)`,
      - `terminal_spt_over_15_component = over15_weight * max(final_spt - 15, 0)`,
    - adds `terminal_spt_bonus` to final reward.
  - added tactical-window bookkeeping before action application:
    - tactical window condition: combat-disabled, turn <= `MAX_TURNS`, city_count < 4,
    - detects legal opportunities:
      - capture available,
      - level-up available,
      - resource gather that completes city upgrade available,
      - move onto visible neutral village available,
      - any useful move available (village-targeting, fog-revealing, or village-distance reducing).
  - added tactical mistake numerator/denominator counters emitted each step:
    - `tm_missed_move_onto_visible_village_num`,
    - `tm_move_onto_visible_village_available_den`,
    - `tm_ignored_capture_num`,
    - `tm_capture_available_den`,
    - `tm_end_turn_with_level_up_num`,
    - `tm_level_up_available_den`,
    - `tm_missed_city_upgrade_completion_num`,
    - `tm_completion_gather_available_den`,
    - `tm_move_off_neutral_village_before_capture_num`,
    - `tm_unit_on_neutral_village_capture_illegal_den`,
    - `tm_end_turn_with_useful_move_num`,
    - `tm_useful_move_available_den`.
  - added explicit tactical move reward/penalty branch:
    - reward when selected move enters visible neutral village during tactical window,
    - penalty when selected move misses an available move-onto-visible-village option.
  - added step info fields for terminal bonus decomposition:
    - `terminal_spt_reward_enabled`,
    - `terminal_spt_bonus`,
    - `terminal_final_spt`,
    - `terminal_spt_base_component`,
    - `terminal_spt_over_10_component`,
    - `terminal_spt_over_15_component`.
  - added debug reward field:
    - `reward_move_onto_visible_neutral_village_tactical`.
  - added helper methods for tactical opportunity detection:
    - `_resource_gather_action_completes_city_upgrade(...)`,
    - `_move_action_reveals_any_fog(...)`.
- Expanded PPO metric logging in `py_rl/cleanrl/cleanrl/ppo.py`:
  - added per-iteration accumulators for all tactical mistake numerator/denominator counters.
  - added always-on tactical counter ingestion each env step via `_extract_vector_field(...)`.
  - extended episode-end telemetry collection with terminal bonus fields:
    - collects `terminal_spt_bonus` from done/final info,
    - tracks derived `final_spt_over_10` and `final_spt_over_15`.
  - added scalar logs:
    - `reward/terminal_spt_bonus`,
    - `charts/final_spt_t10`,
    - `charts/final_spt_over_10`,
    - `charts/final_spt_over_15`.
  - added tactical-mistake rate logs at iteration end:
    - `tactical_mistakes/missed_move_onto_visible_village_rate`,
    - `tactical_mistakes/ignored_capture_rate`,
    - `tactical_mistakes/end_turn_with_level_up_available_rate`,
    - `tactical_mistakes/missed_city_upgrade_completion_rate`,
    - `tactical_mistakes/move_off_neutral_village_before_capture_rate`,
    - `tactical_mistakes/end_turn_with_useful_move_available_rate`.
- Added evaluation harness `tools/eval_org_only_oracle_vs_ppo.py`:
  - loads latest or explicit `.cleanrl_model` and action-interface metadata,
  - infers actor mode from sidecar/state-dict fallback,
  - evaluates paired episodes for:
    - `ppo_latest`,
    - `oracle_org_only` (rule policy prioritizing Organization research, captures, village-progress moves, and constrained economy actions),
  - emits per-episode rows with map/seed/turn and economy/research metrics,
  - writes comparison outputs:
    - `per_episode_results.csv`,
    - `summary.json`,
    - `report.md`,
  - prints PPO-vs-oracle comparison table in terminal.
- Added smoke test `tools/smoke_test_terminal_spt_reward.py`:
  - runs matched seeded episodes with terminal bonus disabled vs enabled,
  - validates:
    - non-terminal reward parity,
    - exactly one enabled bonus event on terminal step,
    - disabled run has zero terminal bonus,
    - observed bonus equals configured closed-form expectation,
    - terminal reward delta equals terminal bonus,
    - component sum (`base + over10 + over15`) matches observed bonus.
- Added validator cache artifact:
  - `.cache/action_validator/1a4fad466b164a5386eb6dbd670f3659312bd23b53832ab05209a94416fb18a9.json`
  - captures strict action-interface validation pass over `10000` states with updated file fingerprint payload.
- Added oracle-vs-ppo evaluation outputs under `outputs/org_only_oracle_vs_ppo`:
  - run `run_20260509_101915`:
    - `summary.json` + `report.md` + `per_episode_results.csv` (`17` lines: header + `16` rows for `8` paired episodes),
    - reported verdict: tie on mean final SPT (`9.000` vs `9.000`).
  - run `run_20260509_104703`:
    - `summary.json` + `report.md` + `per_episode_results.csv` (`1001` lines: header + `1000` rows for `500` paired episodes),
    - reported verdict: PPO beats oracle on mean final SPT (`10.518` vs `8.790`),
    - includes comparative city-count/research/harvest/fog metrics.
- Updated benchmark registry in `model_run_benchmark_log.md`:
  - added `Tribes-v0__ppo__1__1778260387 -> Phase1-Data-019 (600K)`,
  - added `Tribes-v0__ppo__1__1778266653 -> Phase1-Data-019 (5M)`,
  - added `Tribes-v0__ppo__1__1778324123 -> Phase1-Learning-020 (1.5M)`.
- Updated `CHANGELOG.md` with this detailed `[Phase1-Learning-020]` entry to keep source-control documentation synchronized with the update.

## [Phase1-Data-019] - 2026-05-09

### Scope
- Audit and harden coordinate-frame handling between Python wrappers and Java runtime observations.
- Add full-visibility observation plumbing for diagnostics and action-legality forensics.
- Expand Phase-1 telemetry and trainer logging around research/economy timing.
- Add focused audit/evaluation scripts plus validator cache artifacts for reproducible investigation.

### Implemented
- Updated observation bridge in `pol_env/Tribes/py/gym_env.py`:
  - added `get_observation(full_visibility: bool = False)`:
    - default path returns fog-of-war constrained `observationJson()`,
    - diagnostic path returns full-visibility `observationJsonFull()`.
- Refactored Java observation serialization in `pol_env/Tribes/src/core/game/PythonEnv.java`:
  - extracted shared serializer `observationJsonFromState(GameState state)`,
  - preserved existing fogged path via `observationJson()` using `gs.copy(activeTribeID)`,
  - added diagnostic full-state path `observationJsonFull()` using live `gs`,
  - clarified comments that full visibility is diagnostic-only and not training-safe.
- Extended and corrected wrapper diagnostics/telemetry in `pol_env/Tribes/py/register_env.py`:
  - set village-hold shaping terms to zero for diagnostics-only runs:
    - `HOLD_NEUTRAL_VILLAGE_END_TURN_REWARD = 0.0`,
    - `MOVE_OFF_NEUTRAL_VILLAGE_WHEN_CAPTURE_ILLEGAL_PENALTY = 0.0`.
  - added turn/economy counters (reset on episode init/reset):
    - `_turn_forestry_researched`, `_turn_organization_researched`,
    - `_researched_techs_t10`,
    - `_animals_harvested_t10`, `_fruit_harvested_t10`,
    - `_lumber_huts_built_t10`, `_sawmills_built_t10`,
    - `_forests_cleared_t10`.
  - added `_update_economy_counters_from_action(...)` and wired it into step action execution:
    - records research timing by action stream,
    - counts resource gathering/build/clear-forest events through T10.
  - added parsing helpers for action-repr fallback:
    - `_parse_tech_type_from_action_repr(...)`,
    - `_parse_building_type_from_action_repr(...)`.
  - expanded `info` payload on reset/step with new economy counters:
    - `animals_harvested_t10`,
    - `fruit_harvested_t10`,
    - `lumber_huts_built_t10`,
    - `sawmills_built_t10`,
    - `forests_cleared_t10`.
  - shifted research telemetry source from obs tech arrays to tracked executed actions:
    - `techs_researched = len(_researched_techs_t10)`,
    - `forestry_researched = "FORESTRY" in _researched_techs_t10`,
    - `organization_researched = "ORGANIZATION" in _researched_techs_t10`,
    - added `turn_forestry_researched`, `turn_organization_researched`.
  - added neutral-village hold diagnostics in debug info:
    - `reward_hold_neutral_village_end_turn`,
    - `reward_move_off_neutral_village_when_capture_illegal`,
    - `unit_on_neutral_village_capture_illegal`,
    - `unit_on_neutral_village_capture_illegal_count`,
    - `end_turn_while_unit_on_neutral_village`,
    - `moved_off_neutral_village_before_capture`.
  - normalized village-target coordinate logic to Java/runtime `(x,y)` only:
    - removed dual `(x,y)/(y,x)` matching from village move/capture checks,
    - removed swapped-distance fallback in `_min_manhattan_distance(...)`.
  - added explicit board-coordinate helper layer:
    - `_board_get_by_java_coord(...)`, `_board_get_int_by_java_coord(...)`,
    - `_board_get_by_py_coord(...)`,
    - `java_to_py_coord(...)`, `py_to_java_coord(...)`.
  - updated `_board_dimensions_from_obs(...)` semantics and docs:
    - treats board arrays as `board[x][y]`,
    - returns `(width_x, height_y)` in Java/runtime frame.
  - moved fog/reveal helpers to Java-coordinate indexing:
    - `_count_adjacent_fog_tiles(...)`,
    - `_estimate_newly_revealed_tiles_if_move(...)`,
    - `_tile_has_adjacent_fog(...)`.
  - hardened visible-uncaptured-village detection:
    - `_get_visible_uncaptured_village_positions(...)` now excludes city-actor occupied coords,
    - added `_validate_visible_uncaptured_villages(...)`,
    - added `_city_actor_at_java_coord(...)`,
    - added `_unit_has_any_legal_move_or_capture(...)`,
    - optional strict assertion gate via `POLYVISION_STRICT_COORD_ASSERT`.
  - added capture/hold helper methods:
    - `_legal_capture_unit_ids(...)`,
    - `_owned_units_on_visible_uncaptured_village_without_capture(...)`.
- Extended trainer diagnostics and validation caching in `py_rl/cleanrl/cleanrl/ppo.py`:
  - added args:
    - `step_diagnostics_log_every`,
    - `validation_cache_enabled`,
    - `force_revalidate_action_interface`.
  - added validator fingerprint/cache pipeline:
    - `_hash_file_sha256(...)`,
    - `_build_action_validator_fingerprint(...)`,
    - extended `_validate_action_interface(...)` to:
      - include actor/config dimensions in cache key,
      - hash `ppo.py`, `register_env.py`, `PythonEnv.java`, and all Java action files,
      - read successful cache hits from `.cache/action_validator/<fingerprint>.json`,
      - write pass records with coverage flags and fingerprint payload.
  - added runtime guard: `step_diagnostics_log_every > 0`.
  - added always-on episode-end research logging even when step diagnostics are disabled:
    - `research/episode_end_techs_researched_t10`,
    - `research/episode_end_forestry_researched_t10_rate`,
    - `research/episode_end_organization_researched_t10_rate`,
    - `research/avg_turn_forestry_researched`,
    - `research/avg_turn_organization_researched`.
  - throttled high-frequency diagnostic scalar logging by `step_diagnostics_log_every`.
  - expanded step-diagnostics episode-end aggregations with new economy metrics:
    - `economy/episode_end_animals_harvested_t10`,
    - `economy/episode_end_fruit_harvested_t10`,
    - `economy/episode_end_lumber_huts_built_t10`,
    - `economy/episode_end_sawmills_built_t10`,
    - `economy/episode_end_forests_cleared_t10`.
- Added targeted audit script `py_rl/cleanrl/cleanrl/audit_capture_legality_pipeline.py`:
  - replays wrapper filter stages to locate where `CAPTURE` disappears,
  - inspects Java runtime internals via reflection (`gs`, unit/tile fields),
  - prints six-part forensic case traces,
  - aggregates causal buckets/rates and emits verdict bucket (`A`-`E`).
- Added targeted audit script `py_rl/cleanrl/cleanrl/audit_distance_zero_no_capture.py`:
  - audits "distance-zero to visible village but no capture" cases,
  - tracks same-unit next-turn outcomes and capture availability transitions,
  - attributes rates to swapped-coordinate false positives, active-tribe timing, and freshness/tile conditions.
- Added targeted audit script `py_rl/cleanrl/cleanrl/audit_target_contains_visible_village.py`:
  - audits `target_contains_visible_uncaptured_village` feature truth cases,
  - logs direct/swapped/non-candidate target matches,
  - prints per-case mismatch diagnostics and summary cause histogram.
- Added targeted audit script `py_rl/cleanrl/cleanrl/audit_visible_village_targets.py`:
  - audits candidate village classification against enemy cities/units/capital occupancy,
  - tracks targetable-vs-capturable mismatch rates,
  - reports distance-zero no-capture and feature-target drift statistics.
- Added diagnostic evaluator `py_rl/cleanrl/cleanrl/evaluate_no_fog_runtime_village_greedy.py`:
  - implements a no-fog runtime-visible greedy selector with capture/move/economy priorities,
  - includes wait-event before/after traces for "hold on neutral village" scenarios,
  - emits extended T10 SPT/city/fog/research/hold behavior summary metrics.
- Added baseline evaluator `py_rl/cleanrl/cleanrl/evaluate_visible_greedy_movement.py`:
  - implements legal-slot feature-scored visible-info greedy policy,
  - reports T10 performance and selected-move feature statistics.
- Added privileged oracle evaluator `py_rl/cleanrl/cleanrl/privileged_nearest_village_oracle.py`:
  - parses hidden villages from level CSVs,
  - runs oracle move selection against hidden targets,
  - performs coordinate-transform evidence validation (`identity` vs `swapped`),
  - prints before/after metric comparisons and optional failed-episode traces.
- Added validator cache artifacts under `.cache/action_validator`:
  - `468820aab7df46cf8e7c7ce0fc42c3ed7ec3261a4f8c92b20b716914ee71c4d5.json`,
  - `e148dd1a241b0dba32dff8cbbb8f7fb967db9aeb68241c23b2f9f20c9b19de70.json`,
  - `e2b23c19e26c5172b849f12ef5a3c158c150136897c73ca32c8635e8d0bf0151.json`.
  - each cache record stores validator fingerprint payload, coverage flags, checked-state count, and pass status.
- Updated `CHANGELOG.md` with this detailed `[Phase1-Data-019]` entry to keep source-control documentation synchronized with the update.

## [Phase1-Data-018] - 2026-05-08

### Scope
- Add per-legal-action engineered feature tensors to the Tribes wrapper action interface.
- Extend PPO with a new legal-slot policy mode that consumes those action features.
- Add dedicated diagnostics tooling for feature-shape, wiring, influence, and action-quality checks.

### Implemented
- Updated legal-action interface metadata and feature extraction in `pol_env/Tribes/py/register_env.py`:
  - added legal-feature schema constants:
    - `LEGAL_ACTION_FEATURE_VERSION = "v1_1_move_focus"`,
    - `REVEAL_CLIP = 12.0`,
    - `ADJ_FOG_MAX = 8.0`,
    - `LEGAL_ACTION_FEATURE_NAMES` (22-feature ordered tuple),
    - `ACTION_FEATURE_DIM = len(LEGAL_ACTION_FEATURE_NAMES)`.
  - reset and step `info` payloads now include:
    - `legal_action_features_padded` (shape `[max_legal_actions, ACTION_FEATURE_DIM]`, `float32`),
    - `legal_action_feature_dim`,
    - `legal_action_feature_version`.
  - added padded feature builder:
    - `_build_legal_action_features_padded(...)`:
      - aligns slot-wise with `legal_global_ids_padded` and `legal_action_valid_mask`,
      - uses legal-ID to raw-action mapping,
      - leaves invalid/padded slots as zero vectors.
  - added per-action featurizer:
    - `_compute_legal_action_feature_vector(action, obs)`:
      - move features:
        - move indicator,
        - normalized newly revealed fog tiles,
        - normalized adjacent fog count at destination,
        - normalized adjacent fog delta (`dst - src`),
        - zero-reveal move flag,
        - target hits visible uncaptured village flag,
        - has-visible-uncaptured-village flag,
        - normalized distance delta to nearest visible uncaptured village,
        - immediate backtrack flag,
        - target-in-owned-city-bounds flag,
        - normalized distance-from-capital delta,
        - warrior-unit flag.
      - non-move/action-class flags:
        - `END_TURN`, `CAPTURE`, `SPAWN/TRAIN`, `RESEARCH_TECH`,
        - `RESOURCE_GATHERING`, `LEVEL_UP`, `BUILD`,
        - `CLEAR_FOREST`, `GROW_FOREST`, and `is_other`.
  - added helper methods used by featurization:
    - `_count_adjacent_fog_tiles(...)`,
    - `_estimate_newly_revealed_tiles_if_move(...)` (includes ranger/mountain sight-range handling),
    - `_get_capital_position(...)`,
    - `_is_inside_owned_city_bounds(...)`.
- Extended PPO actor pipeline with `legal_features` mode in `py_rl/cleanrl/cleanrl/ppo.py`:
  - CLI/config updates:
    - `--actor-mode` now supports `legal_only`, `legal_features`, `dense_debug`,
    - added `--legal-action-feature-dim` (default `22`) and positive-value validation.
  - `Agent` constructor now accepts `legal_action_feature_dim`.
  - in `legal_features` mode, agent now initializes:
    - `action_feature_encoder` MLP (`feature_dim -> 32 -> 32`),
    - `action_scorer` MLP over concatenated `[state_embed, action_id_embed, action_feature_embed]`.
  - `get_action_and_value(...)` now accepts `legal_action_features` and enforces:
    - rank-3 tensor requirement,
    - batch/slot dimension alignment with legal IDs,
    - last-dimension match with configured feature width.
  - added vector-info extraction utility:
    - `_extract_vector_legal_feature_tensors(...)`:
      - validates presence, shape, finiteness, and env-row availability,
      - returns device tensor `[num_envs, max_legal_actions, feature_dim]`.
  - rollout/training integration:
    - added `legal_action_features_buf` storage for legal-feature mode,
    - extracts feature tensors at reset and every step,
    - passes feature tensors through action sampling, old-logprob recompute checks, and PPO minibatch updates,
    - extends legal-action-mode invariant checks to cover `legal_features`.
  - action-interface metadata sidecar now includes:
    - `legal_action_feature_dim`,
    - `legal_action_feature_version` (read from env reset infos).
  - startup compatibility checks now validate:
    - model/env feature-dimension agreement for legal-feature mode.
- Added diagnostics utility `py_rl/cleanrl/cleanrl/legal_features_diagnostics.py`:
  - imports PPO legal-slot extractors and `Agent`, then runs four diagnostics:
    - `test_shape_alignment(...)`:
      - validates feature tensor shape/dtype and slot-level recomputation parity,
      - verifies invalid slots are zero,
      - prints slot dump rows with slot/global-id/action summary/features.
    - `test_actor_uses_features(...)`:
      - verifies logits change when legal features are zeroed or perturbed.
    - `test_selected_vs_average_move_features(...)`:
      - rollout-based selected-vs-average move-quality statistics (reveal/adjoining-fog/village-distance/backtrack metrics).
    - `test_feature_scale_distribution(...)`:
      - samples states and prints feature min/max/mean and legal/move count distribution summaries.
  - provides CLI flags for model path, seed, rollout steps, sampled states, and slot-dump verbosity.
- Updated benchmark registry in `model_run_benchmark_log.md`:
  - added `Tribes-v0__ppo__1__1778180156 -> Phase1-Data-018 (250K)`,
  - added `Tribes-v0__ppo__1__1778183254 -> Phase1-Data-018 (6M)`.
- Updated `CHANGELOG.md` with this detailed `[Phase1-Data-018]` entry to keep source-control documentation synchronized with the update.

## [Phase1-Learning-017] - 2026-05-07

### Scope
- Strengthen early-game village acquisition behavior with stricter move-progress constraints and targeted anti-stall penalties.
- Expand end-of-episode learning telemetry for economy/research timing milestones.
- Improve procedural capital spacing and regenerate a targeted subset of Phase1 map-pool CSVs to reflect updated generator behavior.

### Implemented
- Updated reward shaping, move filtering, and telemetry in `pol_env/Tribes/py/register_env.py`:
  - removed visible-village neglect penalty constants and their runtime application:
    - removed `VISIBLE_VILLAGE_NEGLECT_PENALTY`,
    - removed `VISIBLE_VILLAGE_NEGLECT_GRACE_TURNS`,
    - removed `reward_visible_village_neglect_penalty` info field.
  - added `USELESS_MOVE_FOG_MISS_PENALTY = 0.35` and new step-time penalty branch for low-value moves:
    - only considered when combat actions are disabled (`ATTACK` absent from allowed action types),
    - active while still below 2 cities and within the episode horizon,
    - applies when a chosen move reveals no new tiles despite at least one legal move for that unit being able to reveal adjacent fog.
  - added new turn trackers:
    - `_turn_first_uncaptured_village_visible`,
    - `_turn_second_city_captured`,
    - reset both on episode reset and populate as milestones occur.
  - added `_tech_name_to_obs_idx` vocabulary map initialization from `Types.java` technology ordering to support robust per-tech observation lookups.
  - expanded reward/decision telemetry:
    - `avg_city_level`,
    - `techs_researched`,
    - `forestry_researched`,
    - `organization_researched`,
    - `turn_first_uncaptured_village_visible` (or `-1` when unseen),
    - `turn_second_city_captured` (or `-1` when uncaptured),
    - debug reward fields:
      - `reward_useless_move_fog_miss_penalty`,
      - `visible_uncaptured_villages_before_move`,
      - `newly_revealed_tiles`,
      - `unit_had_any_legal_fog_revealing_move`.
  - strengthened sub-2-city move forcing logic when visible uncaptured villages exist:
    - after village-capture and direct village-entering move checks, wrapper now identifies the closest owned unit to visible villages and forces only moves that strictly reduce that unit’s Manhattan distance to visible village targets.
  - added move-targeting helpers:
    - `_is_move_reducing_distance_to_targets(...)`,
    - `_closest_owned_unit_to_targets(...)`.
  - added fog-opportunity helpers used by anti-stall shaping:
    - `_tile_has_adjacent_fog(...)`,
    - `_unit_had_any_legal_fog_revealing_move(...)`.
  - added city/research telemetry helpers:
    - `_get_avg_city_level(...)`,
    - `_get_researched_tech_count(...)`,
    - `_has_researched_tech(...)`.
- Expanded terminal rollout telemetry logging in `py_rl/cleanrl/cleanrl/ppo.py`:
  - added per-episode-end extraction and logging for new wrapper metrics:
    - `charts/episode_end_avg_city_level_t10`,
    - `research/episode_end_techs_researched_t10`,
    - research completion rates:
      - `research/episode_end_forestry_researched_t10_rate`,
      - `research/episode_end_organization_researched_t10_rate`,
    - timing means:
      - `charts/avg_turn_first_uncaptured_village_visible`,
      - `charts/avg_turn_second_city_captured`.
  - supports both vector info arrays and `final_info` fallback extraction paths for these metrics.
- Improved capital-spacing constraints in `pol_env/Tribes/src/core/levelgen/LevelGenerator.java`:
  - updated quadrant-capital picker to enforce minimum pairwise capital distance with graceful relaxation:
    - starts from `initialMinCapitalDistance = max(3, mapSize / 4)`,
    - filters both local-domain and fallback candidates by `isFarEnoughFromCapitals(...)`,
    - progressively relaxes distance threshold only when no candidates exist at current threshold.
  - added helper:
    - `isFarEnoughFromCapitals(int cell, ArrayList<Integer> capitals, int minDistance)`.
  - effect: better-separated starting capitals by default without dead-ending map generation when strict spacing is infeasible.
- Updated Phase1 pool generation tooling in `pol_env/Tribes/py/generate_phase1_map_pool.py`:
  - added CLI arg `--start-index` (default `0`) to support append/mix generation without overwriting lower-index pool files.
  - output filename index now uses `start_index + i` while seed progression remains based on loop index and seed-step.
- Regenerated targeted subset of phase1 pool CSV maps in `pol_env/Tribes/levels/phase1_pool` to reflect generator updates:
  - modified 78 map files:
    - `phase1_12x12_pool_006.csv`,
    - `phase1_12x12_pool_008.csv`,
    - `phase1_12x12_pool_014.csv`-`phase1_12x12_pool_015.csv`,
    - `phase1_12x12_pool_019.csv`,
    - `phase1_12x12_pool_022.csv`,
    - `phase1_12x12_pool_026.csv`-`phase1_12x12_pool_027.csv`,
    - `phase1_12x12_pool_039.csv`,
    - `phase1_12x12_pool_041.csv`,
    - `phase1_12x12_pool_045.csv`,
    - `phase1_12x12_pool_050.csv`,
    - `phase1_12x12_pool_055.csv`-`phase1_12x12_pool_056.csv`,
    - `phase1_12x12_pool_064.csv`-`phase1_12x12_pool_127.csv`.
  - these updates are full-content map rewrites (terrain/resources/city/village layouts) produced by the revised generation path.
- Updated benchmark registry in `model_run_benchmark_log.md`:
  - added run mapping `Tribes-v0__ppo__1__1778175810` -> `Phase1-Learning-017 (1M)`.
- Updated `CHANGELOG.md` with this detailed `[Phase1-Learning-017]` entry to keep source-control documentation synchronized with the update.

## [Phase1-Learning-016] - 2026-05-07

### Scope
- Learning-focused reward retuning to emphasize productive SPT gains while damping city-capture bonus magnitude.
- Add a narrow early-turn movement anti-backtrack constraint for a specific opening unit to stabilize early trajectory quality.
- Keep evaluator/trainer diagnostics aligned with the new reward decomposition and reduce dashboard noise in fast info mode.

### Implemented
- Updated reward shaping and early-turn movement constraints in `pol_env/Tribes/py/register_env.py`:
  - tuned city-capture shaping constants:
    - `CAPTURE_CITY_BONUS_MIN`: `4.0 -> 3.0`,
    - `CAPTURE_CITY_BONUS_MAX`: `8.0 -> 6.0`.
  - added SPT reward scaling constants:
    - `SPT_INCREASE_REWARD_MULTIPLIER = 5.0`,
    - `SPT_NONPOSITIVE_REWARD_MULTIPLIER = 1.0`.
  - reward computation now distinguishes:
    - raw `base_delta_spt`,
    - shaped `delta_spt_reward` (positive deltas amplified 5x, non-positive unchanged).
  - final reward now uses:
    - `reward = delta_spt_reward + reward_adjustment`
    - (instead of `base_delta_spt + reward_adjustment`).
  - added `info["delta_spt_reward"]` telemetry for explicit shaped-vs-raw SPT attribution.
  - added unit previous-tile tracking state:
    - wrapper field `_unit_previous_tiles`,
    - reset clearing in episode reset path,
    - move-source updates in selected action path and opening autoplayer path.
  - replaced ad-hoc move parse path with structured extractor:
    - added `_extract_move_components(...)` to recover `unit_id/src/dst` from structured fields first, with repr/obs fallback.
  - added targeted early-turn anti-backtrack filter:
    - `_apply_t1_t2_unit2_backtrack_mask(...)` runs during allowed-index filtering,
    - active only on turns 1-2,
    - blocks `MOVE` actions where unit `2` immediately returns to its previous tile,
    - preserves safety by reverting to original allowed set if filtering would produce zero legal actions.
  - integrated backtrack filter into action filtering pipeline immediately before mask/mapping construction.
- Updated reward-debug reconstruction in `evaluate_brain.py`:
  - `print_reward_breakdown(...)` now reads `delta_spt_reward` from info payload (fallback to `delta_spt`).
  - added explicit `delta_spt_reward` line in printed breakdown.
  - reconstructed total now uses `delta_spt_reward + shaping_sum` for parity with wrapper reward output.
- Updated trainer diagnostics behavior in `py_rl/cleanrl/cleanrl/ppo.py`:
  - set W&B init to `sync_tensorboard=False` (explicit manual metric logging path retained).
  - detects wrapper `info_mode` from reset infos and derives `debug_chart_mode`.
  - startup action-interface banner now prints detected `info_mode`.
  - core diagnostic series is now mode-aware:
    - always logs high-signal metrics: `unit_count`, `stars`, `reward`,
    - logs lower-signal internals (`turn`, `selected_global_id`, `selected_raw_java_index`) only in debug chart mode.
  - removed duplicate per-episode-end `custom_spt_return` logging branch to reduce redundant chart noise.
- Updated benchmark registry in `model_run_benchmark_log.md`:
  - added run mapping `Tribes-v0__ppo__1__1778158665` -> `Phase1-Learning-016 (4M)`.
- Updated `CHANGELOG.md` with this detailed `[Phase1-Learning-016]` entry to keep source-control documentation synchronized with the update.

## [Phase1-Data-015] - 2026-05-07

### Scope
- Complete the trainer/evaluator transition onto legal-slot action tensors backed by the global action-ID catalog.
- Remove remaining fallback behavior in the wrapper and enforce strict fail-fast semantics for illegal IDs and out-of-bounds move selections.
- Align run metadata, logging, and model-sidecar interface fields with the new actor-mode contract.

### Implemented
- Updated legal-slot interface and fail-fast behavior in `pol_env/Tribes/py/register_env.py`:
  - added `MAX_LEGAL_ACTIONS_DEFAULT = 1024` and runtime override via `POLYVISION_MAX_LEGAL_ACTIONS`.
  - added wrapper field `_max_legal_actions` with validated positive bound.
  - reset/step info now includes:
    - `max_legal_actions`,
    - `legal_global_ids_padded`,
    - `legal_action_valid_mask`,
    - `legal_action_count`.
  - removed fast-mode `legal_global_ids` emission path in favor of fixed-width padded legal-slot tensors.
  - added `_build_legal_slot_tensors(action_mask)`:
    - converts legal global IDs into fixed-width padded tensors,
    - emits matching boolean valid-mask,
    - raises if legal count exceeds `max_legal_actions`,
    - raises on duplicate legal IDs.
  - hardened action selection path:
    - illegal sampled global IDs now raise `RuntimeError` immediately (instead of fallback-to-`END_TURN`).
    - selected `MOVE` actions with out-of-bounds destination now raise `RuntimeError` immediately (instead of fallback-to-`END_TURN`).
  - retained action-interface diagnostics while tightening enforcement semantics around illegal sampling and legal-slot integrity.
- Updated PPO actor/training pipeline for legal-slot mode in `py_rl/cleanrl/cleanrl/ppo.py`:
  - added args:
    - `actor_mode` (`legal_only` default, `dense_debug` optional),
    - `max_legal_actions`,
    - `old_logprob_recompute_tol`.
  - extended `Agent`:
    - supports dual actor modes:
      - `dense_debug`: legacy dense logits head over full action space,
      - `legal_only`: state encoder + global action embedding scored only over legal-slot candidates.
    - `get_action_and_value(...)` now supports:
      - dense masked mode,
      - legal-slot mode with `legal_global_ids` + `legal_action_valid_mask` + optional `selected_slot`.
  - added padded legal-tensor extraction helpers:
    - vector path: `_extract_vector_legal_tensors(...)`,
    - single-env path: `_extract_action_mask_from_info_dict(...)` now reads padded legal tensors.
    - vector dense-mask fallback path now also supports reconstructing mask from `legal_global_ids_padded` + `legal_action_valid_mask`.
  - added startup guards:
    - validates `actor_mode`,
    - validates `max_legal_actions > 0`,
    - exports `POLYVISION_MAX_LEGAL_ACTIONS`,
    - checks actor/action-embedding dimensions against env action space.
  - added rollout buffers for legal-slot mode:
    - `selected_slots`,
    - `legal_global_ids_buf`,
    - `legal_action_valid_mask_buf`,
    - and dense `action_masks` buffer for `dense_debug`.
  - added strict invariants:
    - selected slot must map back to sampled global action ID during rollout,
    - env-reported `selected_global_id` must match sampled action,
    - pre-update recompute of old logprobs must match stored values within tolerance,
    - minibatch legal-slot mapping consistency checks during PPO updates.
  - expanded validator checks:
    - verifies padded legal ID/mask length consistency,
    - checks `legal_action_count == valid_mask.sum`,
    - checks duplicates absent in valid padded IDs.
  - added action-interface metadata fields:
    - now includes `actor_mode` and `max_legal_actions`.
  - added action-interface startup print and tensorboard metadata dump (`meta/action_interface`).
  - introduced `log_scalar(...)` helper:
    - writes to TensorBoard and, when tracking is enabled, mirrors metrics to W&B with `global_step`.
  - switched scalar logging calls to `log_scalar(...)` for charts/losses/SPS/end-of-episode metrics.
- Updated evaluator compatibility with legal-slot actor mode in `evaluate_brain.py`:
  - added model-sidecar action-interface metadata loading (`.action_interface.json`) via `load_action_interface_meta(...)`.
  - added actor-mode inference fallback from state dict (`infer_actor_mode_from_state_dict(...)`).
  - sets `POLYVISION_MAX_LEGAL_ACTIONS` from model metadata (default fallback retained).
  - constructs `Agent(..., actor_mode=..., max_legal_actions=...)` to match training-side architecture.
  - prints resolved actor mode and max legal actions at startup.
  - replaced direct use of `_build_action_mask_and_indices(...)` with `_build_action_mask_and_mapping(...)`.
  - in legal-only mode:
    - builds legal-slot tensors with `_build_legal_slot_tensors(...)`,
    - samples through `agent.get_action_and_value(...)` legal-slot interface,
    - derives per-legal-action probabilities from slot logits for debug output.
  - in dense-debug mode:
    - preserves dense masked sampling behavior.
  - action-to-raw mapping now resolves through canonical `legal_id_to_raw_index` instead of modulo position remapping.
- Updated benchmark registry in `model_run_benchmark_log.md`:
  - added run mapping `Tribes-v0__ppo__1__1778147403` -> `Phase1-Data-015 (1M)`.
- Updated `CHANGELOG.md` with this detailed `[Phase1-Data-015]` entry to keep source-control documentation fully in sync with the update.

## [Phase1-Data-014] - 2026-05-06

### Scope
- Convert the Phase 1 Python action interface from positional/variable indexing to a deterministic global action-ID contract.
- Add structured Java action metadata and strict pre-training/runtime validation so action-mask correctness is enforced end-to-end.
- Add explicit validator tooling and run-registry bookkeeping for the data-interface migration.

### Implemented
- Reworked action-space/canonicalization pipeline in `pol_env/Tribes/py/register_env.py`:
  - added `GlobalActionCatalog` class to build a deterministic flat global action-ID space for fixed board dimensions.
  - catalog now owns typed/global offsets and ID mapping for:
    - `END_TURN`,
    - `MOVE`,
    - `CAPTURE`,
    - `TRAIN`/`SPAWN`,
    - `RESOURCE_GATHERING`,
    - `CLEAR_FOREST`,
    - `GROW_FOREST`,
    - `BUILD`,
    - `RESEARCH_TECH`,
    - `LEVEL_UP`,
    - `EXAMINE`.
  - added action-catalog fingerprinting via deterministic JSON + SHA-256 hash to track offset-table integrity.
  - wrapper action space is now driven by catalog size (`Discrete(catalog.total_size)`) instead of fixed placeholder `Discrete(200)` when bootstrap succeeds.
  - added `CATALOG_VERSION = "flat-v1"` and `CANONICALIZER_VERSION = "flat-v1-structured"` telemetry/version guards.
  - expanded allowed action types to include `SPAWN`.
  - added Java enum vocabulary loading from `src/core/Types.java`:
    - `_load_action_vocab(...)`,
    - `_extract_enum_names_from_types_java(...)`,
    - and typed lookup tables for technologies/units/resources/buildings/city-level-up options.
  - replaced positional mask builder with deterministic global-ID mapping flow:
    - `_filter_allowed_raw_indices(...)` for legality/guardrail filtering,
    - `_build_action_mask_and_mapping(...)` for global-ID canonicalization + dense mask generation.
  - added canonicalization for structured action payloads:
    - `_canonicalize_action_to_global_id(...)` now maps per-action typed fields + coordinate payloads into global IDs.
  - added fail-fast canonicalization guards:
    - raises on any uncanonicalized allowed legal action,
    - raises on duplicate global-ID collisions.
  - switched sampling semantics from modulo remap to direct global-ID selection:
    - sampled global ID must exist in current legal ID mapping,
    - illegal sampled IDs increment counters and force `END_TURN` fallback.
  - added runtime counters/telemetry for safety monitoring:
    - `illegal_sample_count`, `fallback_end_turn_count`, `total_action_decisions`,
    - `illegal_sample_rate`, `fallback_end_turn_rate`,
    - `selected_global_id`, `selected_raw_java_index`,
    - canonicalization diagnostics and by-type legal-action counts.
  - added info-mode split with environment knob `POLYVISION_INFO_MODE`:
    - `fast` mode emits sparse `legal_global_ids` (instead of full dense mask),
    - `debug` mode retains richer payloads (including dense `action_mask` and detailed diagnostics).
  - added action validation mode knob `POLYVISION_ACTION_VALIDATION_MODE` and wrapper state for per-step canonicalization diagnostics.
  - added helper `_diag_for_info(...)` to trim/shape diagnostic payloads by info mode.
  - retained/extended debug-only move verification and game-state telemetry path under debug info mode.
  - added utility helpers:
    - `_action_int(...)`, `_action_str(...)`,
    - `_unit_position_by_id(...)`, `_city_position_by_id(...)`,
    - `_parse_target_xy_from_action_repr(...)`,
    - `_get_owned_unit_count(...)`, `_get_tribe_stars(...)`.
- Added structured legal-action schema export in `pol_env/Tribes/src/core/game/PythonEnv.java`:
  - `listActionsJson()` now adds `schema_version = 1` and enriches every action with typed fields through `addStructuredActionFields(...)`.
  - added typed field extraction per action class:
    - `Move`: `unit_id`, `src_x/src_y`, `dst_x/dst_y`,
    - `Capture`: `unit_id`, `capture_type`, `target_city_id`, source/target coordinates/tiles,
    - `Spawn`: `city_id` + city coordinates/tile + `unit_type`,
    - `ResourceGathering`: city fields + `resource_type` + target fields,
    - `Build`: city fields + `building_type` + target fields,
    - `ClearForest` / `GrowForest`: city fields + target fields,
    - `LevelUp`: city fields + `levelup_choice` + target fields,
    - `ResearchTech`: `tribe_id` + `tech_type`,
    - `Examine`: `unit_id` + source coordinates.
  - added shared helpers:
    - `addCityFields(...)`,
    - `putTargetFieldsFromCityAction(...)`,
    - `safeGetUnit(...)`, `safeGetCity(...)`.
  - this provides structured canonicalization inputs to Python without relying solely on `repr` parsing.
- Hardened PPO training against action-interface drift in `py_rl/cleanrl/cleanrl/ppo.py`:
  - added CLI/runtime args:
    - `validate_action_interface` (default `True`),
    - `validation_states` (default `10000`),
    - `validation_seed` (default `12345`),
    - `max_illegal_sample_rate` (default `0.0001`),
    - `max_fallback_end_turn_rate` (default `0.0001`).
  - extended vector mask extraction to support sparse fast-mode payload:
    - reconstructs dense masks from `legal_global_ids` + optional `_legal_global_ids` validity mask.
  - added `_extract_action_mask_from_info_dict(...)` for single-env validator compatibility with dense/sparse info formats.
  - added strict `_validate_action_interface(...)` pre-training validator:
    - fails on canonicalization gaps/collisions/mask inconsistencies,
    - verifies no modulo-mapping remnant gate in wrapper source,
    - checks sampled legal IDs are never reclassified illegal/fallbacked,
    - performs scenario coverage checks over legal action types (spawn/resource/capture/research/clear-forest).
  - validator is executed before vectorized training when `validate_action_interface=True`.
  - added startup action-interface contract checks:
    - `action_mask` width must match `env.action_space.n`,
    - actor output dimension must match `env.action_space.n`,
    - reported wrapper `global_action_space_n` must match trainer action space.
  - added runtime fail-fast guards during rollout:
    - abort on any non-zero `duplicate_global_id_collisions`,
    - abort on any non-zero `uncanonicalized_legal_actions`,
    - abort if illegal/fallback rates exceed configured thresholds.
  - added action-interface metadata capture (`catalog/canonicalizer/map dims/hash/action-space`) and sidecar persistence to:
    - checkpoint files (`.action_interface.json`),
    - final saved model (`.action_interface.json`).
  - added additional diagnostic scalar logging for core action telemetry (`turn`, `unit_count`, `stars`, `reward`, selected IDs).
- Added standalone strict validator tool `pol_env/Tribes/py/validate_action_interface.py`:
  - CLI utility to run deterministic action-interface validation outside training loop.
  - validates:
    - zero collisions,
    - zero uncanonicalized legal actions,
    - `mask_ones == unique_legal_global_ids`,
    - legal sampled IDs remain legal and do not trigger fallback.
  - includes situation-coverage tracking for spawn/resource/capture/research/clear-forest legal opportunities.
  - supports configurable `--states` and `--seed`.
- Updated benchmark registry in `model_run_benchmark_log.md`:
  - added run mapping `Tribes-v0__ppo__1__1778104307` -> `Phase1-Data-014 (3M)`.

## [Phase1-Learning-013] - 2026-05-06

### Scope
- Learning-phase reward-shaping iteration focused on stronger village discovery/capture incentives and removal of delayed/terminal penalty pressure.
- Keep run-tracking metadata aligned with the latest training checkpoint.

### Implemented
- Updated village/city reward shaping in `pol_env/Tribes/py/register_env.py`:
  - removed constants:
    - `SECOND_VILLAGE_BY_T10_PENALTY`,
    - `SECOND_VILLAGE_DELAY_PENALTY`,
    - `SECOND_VILLAGE_DELAY_START_TURN`,
    - fixed `CAPTURE_CITY_BONUS = 2.0`.
  - added new positive-shaping constants:
    - `REVEAL_UNCAPTURED_VILLAGE_REWARD = 1.0`,
    - `MOVE_CLOSER_TO_VISIBLE_VILLAGE_REWARD = 0.5`,
    - `MOVE_ONTO_VILLAGE_REWARD = 1.0`,
    - `CAPTURE_CITY_BONUS_MIN = 4.0`,
    - `CAPTURE_CITY_BONUS_MAX = 8.0`.
  - changed city-capture bonus computation:
    - now scales with new cities captured in-step (`+4` per city) with hard cap at `+8` total per step.
  - added newly-revealed-village reward logic:
    - computes visible uncaptured villages before and after selected action,
    - adds reward proportional to newly revealed village count.
  - added move-progress reward toward visible villages:
    - computes moved unit position pre/post selected move,
    - uses new Manhattan-distance helper to award when distance to nearest visible uncaptured village decreases.
  - added explicit reward for stepping onto a visible uncaptured village tile.
  - removed second-village delay penalty application on `END_TURN`.
  - removed terminal Turn-10 no-expansion penalty branch (`SECOND_VILLAGE_BY_T10_PENALTY` on truncation).
  - extended emitted info telemetry:
    - `reward_reveal_uncaptured_village`,
    - `reward_move_closer_to_visible_village`,
    - `reward_move_onto_village`,
    - `newly_revealed_uncaptured_villages`.
  - kept compatibility info key `reward_second_village_delay_penalty`, now always set to `0.0`.
  - added helper `_min_manhattan_distance(origin, targets)` with XY/YX fallback minimization for coordinate-order robustness.
- Updated benchmark registry in `model_run_benchmark_log.md`:
  - added run mapping `Tribes-v0__ppo__1__1778088607` -> `Phase1-Learning-013 (4.5M)`.

## [Phase1-Generation-012] - 2026-05-06

### Scope
- Extend Phase 1 procedural generation from a single drylands-style path into explicit map-type-driven generation profiles.
- Harden movement legality and diagnostics around board bounds across Java gameplay, Python wrapper filtering, and evaluator tooling.
- Regenerate and expand the fixed Phase 1 map pool for broader deterministic curriculum coverage.

### Implemented
- Added map-type profile configuration in `pol_env/Tribes/src/core/TribesConfig.java`:
  - introduced `MAP_TYPE` enum with per-profile `initialLand`, `smoothing`, and `relief` presets:
    - `DRYLANDS`, `LAKES`, `CONTINENTS`, `PANGEA`, `ARCHIPELAGO`, `WATERWORLD`.
  - added `DEFAULT_MAP_TYPE = MAP_TYPE.CONTINENTS`.
- Extended game initialization APIs to carry map-type selection:
  - `pol_env/Tribes/src/core/game/Game.java`:
    - added overloaded `init(...)` accepting `TribesConfig.MAP_TYPE`.
    - existing `init(...)` now delegates to overload with `DEFAULT_MAP_TYPE`.
  - `pol_env/Tribes/src/core/game/GameState.java`:
    - added overloaded `init(...)` accepting `TribesConfig.MAP_TYPE`.
    - level generator initialization now uses map-type-specific smoothing/relief/initial-land values.
- Extended level-generation CLI argument model in `pol_env/Tribes/src/core/levelgen/GenerateLevelCli.java`:
  - usage now supports optional `[mapType]` after optional `[initialLand]`.
  - added map-type parsing/validation helpers (`isMapType`, `parseMapType`).
  - generator init now consumes selected map type profile values.
- Refactored procedural generation pipeline in `pol_env/Tribes/src/core/levelgen/LevelGenerator.java`:
  - added map-type state and support in `init(...)` overload (`TribesConfig.MAP_TYPE`).
  - replaced single land-pass logic with profile-aware base-land generation:
    - generic smoothed generation,
    - dedicated Pangea clustered-center growth,
    - dedicated Continents multi-cluster growth.
  - reworked capital-placement strategy selection:
    - quadrant-style placement for Drylands/Lakes/Archipelago/Waterworld,
    - village-conversion placement for Continents/Pangea with Pangea coastal bias,
    - distance-based and quadrant fallbacks.
  - added/expanded phased village generation:
    - suburb phase (Lakes/Archipelago),
    - pre-terrain village density phase (Lakes/Archipelago/Waterworld),
    - post-terrain village saturation pass,
    - tiny-island village pass for Continents/Pangea by map-size target.
  - replaced per-cell random terrain/resource assignment with quota allocation:
    - tribe-normalized terrain quotas for mountain/forest/plain residual.
    - per-tribe, per-band resource quotas for fruit/crops/animals/ore/fish/whales.
    - deterministic weighted count allocator with fractional remainder handling.
  - preserved capital write ordering by sorted index so level loading remains stable with tribe order.
- Added board-bounds helper in `pol_env/Tribes/src/core/game/Board.java`:
  - new `isInBounds(int x, int y)` utility.
- Hardened move generation and execution to prevent off-board destinations:
  - `pol_env/Tribes/src/core/actions/unitactions/factory/MoveFactory.java`:
    - skips out-of-bounds path nodes.
  - `pol_env/Tribes/src/core/actions/unitactions/Move.java`:
    - feasibility now fails when destination is null or out of bounds.
  - `pol_env/Tribes/src/core/actions/unitactions/command/MoveCommand.java`:
    - early-return false for null/out-of-bounds destination before execution.
- Added debug rendering overlays and dynamic board-size sync in `pol_env/Tribes/src/gui/GameView.java`:
  - new debug overlays for map border, per-tile coordinate labels, and corner legend.
  - `update(...)` now refreshes `gridSize` from live board size to keep renderer bounds aligned.
- Updated Python map-generation scripts to expose map-type selection:
  - `pol_env/Tribes/py/generate_phase1_map_pool.py`:
    - removed duplicate `argparse` import.
    - added `--map-type` (default `DRYLANDS`) and forwards it to Java CLI.
  - `pol_env/Tribes/py/generate_fixed_map.py`:
    - removed duplicate `argparse` import.
    - added `--map-type` (default `DRYLANDS`) and forwards it to Java CLI.
- Added bounds and fog-coverage diagnostics in `pol_env/Tribes/py/register_env.py`:
  - tracks `initial_visible_tiles` and cumulative `fog_tiles_cleared_total`.
  - filters `MOVE` actions whose parsed destination is outside board bounds.
  - when sampled move destination is out-of-bounds, falls back to `END_TURN`.
  - adds step-time move verification info:
    - requested vs actual move position,
    - destination match flag,
    - actual in-bounds flag.
  - asserts unit positions are in bounds after selected action and after forced non-Bardur turn advancement.
  - extends info payload with:
    - `fog_tiles_cleared_step`, `fog_tiles_cleared_total`,
    - `visible_tiles`, `initial_visible_tiles`,
    - move verification fields (`move_verify_*`).
  - upgraded move `repr` parsing to prefer explicit `by unit ... to x:y` pattern before numeric fallback.
- Expanded evaluator-side move/bounds diagnostics in `evaluate_brain.py`:
  - upgraded move parsing with explicit `by unit ... to x:y` regex-first path.
  - simplified relative-delta inference to direct coordinate subtraction.
  - added board-dimension helper and map-bounds banner output.
  - enriched action labels with `from -> to`, relative deltas, and `in_bounds`.
  - added pre-selection legal-action sanity check logging for off-board move options.
  - added post-step `MOVE_VERIFY` log comparing requested destination with actual unit position and board bounds.
  - updated policy move-grid off-board handling to use width/height instead of single map-size assumption.
- Extended PPO terminal metrics logging in `py_rl/cleanrl/cleanrl/ppo.py`:
  - added generic per-metric extraction helper for vectorized `infos`.
  - now logs end-of-episode:
    - `charts/episode_end_village_count_t10` from `city_count`,
    - `charts/episode_end_fog_tiles_cleared_t10` from `fog_tiles_cleared_total`,
    - alongside existing end-episode `spt` metrics.
- Updated benchmark registry in `model_run_benchmark_log.md`:
  - added run mapping `Tribes-v0__ppo__1__1778077298` -> `Phase1-Generation-012 (4M)`.
- Added constraint check coverage in `test_phase1_constraints.py`:
  - added move-destination parser helper.
  - added assertion pass ensuring every allowed `MOVE` action destination is within current board bounds before selecting `END_TURN`.
- Regenerated and expanded phase1 pool maps in `pol_env/Tribes/levels/phase1_pool`:
  - regenerated existing pool files:
    - `phase1_12x12_pool_000.csv` through `phase1_12x12_pool_031.csv`.
  - added new pool files:
    - `phase1_12x12_pool_032.csv` through `phase1_12x12_pool_127.csv`.
  - effective pool size is now 128 maps (`000`-`127`).
- Added new map-generation parity reference document `MapGen.md`:
  - captures cleaned Polytopia-generation reference notes,
  - documents current Tribes parity status by subsystem,
  - includes map-type, village, resource, and modifier reference tables.

## [Phase1-Generalizing-011] - 2026-05-05 - 2026-05-06

### Scope
- Shift Phase 1 from fixed deterministic-map training toward controlled map variation to reduce overfitting and break local optima.
- Keep training reproducible at the run level while ensuring each episode is not the exact same map state.

### Implemented
- Updated `pol_env/Tribes/py/register_env.py` reset seeding to use a deterministic per-run seed stream instead of `seed or 42`:
  - explicit reset seed re-initializes deterministic streams,
  - subsequent resets (without explicit seed) draw deterministic per-episode seeds from the run stream.
- Added deterministic map-pool selection support in `pol_env/Tribes/py/register_env.py`:
  - new env knob `POLYVISION_LEVEL_POOL_GLOB` (glob for candidate CSV maps),
  - default fallback glob `levels/phase1_pool/*.csv`,
  - level selection mode via `POLYVISION_LEVEL_SELECTION_MODE`:
    - `round_robin` (default),
    - `seeded_random` (deterministic RNG stream).
- Added base-seed override knob `POLYVISION_BASE_SEED` for deterministic runs when Gym does not provide reset seeds.
- Added per-episode map/seed telemetry in reset and step info:
  - `map_path`, `map_id`, `map_pool_index`, `map_pool_size`, `episode_seed`, `level_selection_mode`.
- Added map-pool generator script `pol_env/Tribes/py/generate_phase1_map_pool.py`:
  - generates a deterministic pool of 12x12 CSV maps from a base seed and stride,
  - intended output directory: `levels/phase1_pool`.
  - generated and added pool files:
    - `levels/phase1_pool/phase1_12x12_pool_000.csv` through `levels/phase1_pool/phase1_12x12_pool_031.csv`.
- Fixed Tribes Java level-generation parity bug in `pol_env/Tribes/src/core/levelgen/LevelGenerator.java`:
  - corrected capital-owner comparisons in starting-resource post-generation:
    - `owner == IMPERIUS.getKey()`
    - `owner == BARDUR.getKey()`
  - this restores intended guaranteed starting resources around capitals (including Bardur animals for Hunting parity).
- Fixed player-order consistency for generated maps in `pol_env/Tribes/src/core/levelgen/LevelGenerator.java`:
  - capital cells are now sorted before writing city tokens, so row-major level loading reconstructs players in the provided tribe order.
  - this prevents random generated maps from assigning non-Bardur tribe to player index 0, which previously broke Bardur-specific opening assumptions.
- Updated map-generator Python scripts to ensure parity fixes are compiled into map generation:
  - `pol_env/Tribes/py/generate_phase1_map_pool.py`
  - `pol_env/Tribes/py/generate_fixed_map.py`
  - both now compile `LevelGenerator.java` alongside `GenerateLevelCli.java`.
- Switched Phase1 generation to drylands-style all-land starts for lower-noise training:
  - updated `pol_env/Tribes/src/core/levelgen/GenerateLevelCli.java` to use `initialLand=1.0` by default.
  - `GenerateLevelCli` now accepts optional explicit `[initialLand]` argument.
  - updated generator scripts to pass `--initial-land` explicitly (default `1.0`) when producing map pools/fixed maps.
  - rationale: simplify training distribution and avoid island/water start variance that introduces unnecessary early-game noise relative to the intended opening curriculum.
- Updated evaluator map selection parity with trainer in `evaluate_brain.py`:
  - `--show-opening` no longer hard-resets `env.level_file`; it now uses wrapper seed/map pool selection logic.
  - added optional evaluator flags:
    - `--level-pool-glob`
    - `--level-selection-mode`
    - `--base-seed`
  - this makes debug rollouts follow the same map distribution as training by default when pool settings are shared.
  - `--show-opening` replay now also prints selected map basename, pool index, and episode seed for run-trace clarity.
- Added multiprocessing-serialization hardening in `pol_env/Tribes/py/register_env.py`:
  - reset/step info payloads are now sanitized to pickle-safe primitives/containers.
  - added defensive fallback that strips or stringifies any residual non-picklable objects.
  - purpose: prevent AsyncVectorEnv worker crashes from `TypeError: cannot pickle '_thread.RLock' object` when unexpected runtime objects leak into info channels.
- Training performance optimizations with no policy-logic changes:
  - `pol_env/Tribes/py/register_env.py`:
    - opening move-score grid printouts are now gated behind `POLYVISION_OPENING_GRID_DEBUG` (default off).
  - `py_rl/cleanrl/cleanrl/ppo.py`:
    - added `enable_step_diagnostics` (default `False`) to disable expensive per-step diagnostic extraction/logging in the hot loop unless explicitly requested.
    - core PPO rollout/update behavior is unchanged; this only trims instrumentation overhead.
- Updated parallel environment count target for training to `num_envs=20` (recommended/default run configuration):
  - reason: a direct throughput sweep (`50,000` timesteps each) over `num_envs=8..20` showed best sustained SPS at `20` in this setup.
  - stability follow-up at `num_envs=24` (short repeated runs) was faster in some outlier runs but materially noisier overall, so `20` was selected as the safer high-throughput baseline for long runs.
- Short benchmark results (local single-process surrogate, same map pool/settings before vs after):
  - opening-grid print gating:
    - before: `1500 steps in 6.450s` (`232.6 SPS`)
    - after: `1500 steps in 6.156s` (`243.6 SPS`)
    - delta: `+4.7%` SPS
  - step-diagnostics overhead (diagnostics on vs off):
    - diagnostics on: `260.3 SPS`
    - diagnostics off: `274.1 SPS`
    - delta: `+5.3%` SPS
- Updated benchmark registry in `model_run_benchmark_log.md`:
  - added `Tribes-v0__ppo__1__1778008176` -> `Phase1-Learning-010 (2.75M)`.
  - added `Tribes-v0__ppo__1__1778027687` -> `Phase1-Generalizing-011 (33M)`.

## [Phase1-Learning-010] - 2026-05-05

### Scope
- Add a hard early-expansion guardrail so village capture pathing is mandatory when available.

### Implemented
- Updated action masking in `pol_env/Tribes/py/register_env.py` with a strict village-priority override:
  - in `_build_action_mask_and_indices(...)`, after normal whitelist and resource-feasibility filtering, the wrapper now checks for village-entering `MOVE` actions.
  - condition:
    - `city_count < 2`
    - at least one legal `MOVE` action steps onto a visible uncaptured village tile.
  - behavior:
    - when condition is met, allowed action indices are replaced with only those village-entering move indices.
    - this guarantees the policy must choose a village-entering move when one is available.
    - updated behavior: when `city_count < 2` and a legal village `CAPTURE` exists, that unit is frozen (its `MOVE`/`CAPTURE` actions are masked out), and village capture is deferred/auto-executed immediately before `END_TURN`.
    - this allows non-unit decisions (for example `RESEARCH_TECH`) to happen first in the same turn while still guaranteeing village capture before turn handoff.
  - queue lifecycle details:
    - added wrapper state `_queued_village_capture_unit_ids` initialized in constructor and reset on episode reset.
    - queued set is cleared automatically once `city_count >= 2`.
- Added helper parsing/validation for village-entering move detection in `pol_env/Tribes/py/register_env.py`:
  - `_parse_move_unit_and_dest_from_action_repr(...)`
  - `_is_move_to_visible_uncaptured_village(...)`
  - validation includes:
    - unit ownership check (`tribeId == 0`),
    - destination tile check against visible uncaptured village coordinates.
- Added `_is_capture_of_village(...)` in `pol_env/Tribes/py/register_env.py` to detect and force Bardur-owned village capture actions while still below 2 cities.
  - includes coordinate-order fallback when matching unit tile to visible village tile, so forced capture remains reliable across Java/Python axis-order quirks.
- Added `_parse_unit_id_from_action_repr(...)` in `pol_env/Tribes/py/register_env.py` for robust unit-freeze and deferred-capture routing.
- Added deferred-capture telemetry in `pol_env/Tribes/py/register_env.py`:
  - `deferred_village_captures_before_end_turn`
  - `queued_village_capture_unit_ids`
- Hardened village detection consistency in `pol_env/Tribes/py/register_env.py`:
  - added `_get_visible_uncaptured_village_positions(...)` and reused it in:
    - `_is_move_to_visible_uncaptured_village(...)`
    - `_has_visible_uncaptured_village(...)`
    - `_has_unit_on_visible_uncaptured_village(...)`
  - ensures the forced village mask and reward-shaping checks share the same village source of truth.
- Fixed debug output ambiguity in `evaluate_brain.py`:
  - action-relative `dX/dY` now normalizes coordinate-order variants before printing.
  - move-grid orientation now uses direct `(rel_x, rel_y)` world mapping for consistent visualization.
  - executed action label is now resolved before stepping the env, so it no longer incorrectly prints `[rel dX=+0, dY=+0]` after moves.
- Updated benchmark registry in `model_run_benchmark_log.md`:
  - added run mapping `Tribes-v0__ppo__1__1778000233` -> `Phase1-Learning-010 (1M)`.

## [Phase1-Learning-009] - 2026-05-05

### Scope
- Learning-only planning and analysis phase focused on improving post-opening policy decisions.
- Objective is to improve model performance by reviewing the latest model runs slowly and deriving candidate soft and hard rules for the next training iteration.

### Planned Review Method
- Replay latest model runs step-by-step to inspect decision quality and opportunity cost at each turn.
- Record repeated policy mistakes, weak priorities, and missed high-value actions under true fog-of-war conditions.
- Separate candidate interventions into:
  - soft shaping rules (reward/punishment nudges),
  - hard constraints/guardrails (strict behavior limits or action gating).

### Planned Output
- Produce a shortlist of actionable behavior-shaping rules tied directly to observed run evidence.
- Prioritize rules that are low-risk, testable, and likely to improve early economy tempo, village capture consistency, and overall growth reliability.
- Carry selected rules into the next model update after this review pass.

### Implemented
- Added anti-sunken-cost action gating for early economy investments in `pol_env/Tribes/py/register_env.py`:
  - updated `_build_action_mask_and_indices(...)` to apply legality checks for `RESOURCE_GATHERING` actions before adding them to the allowed mask.
  - introduced `_is_resource_gather_legal_for_upgrade(...)` to block gather actions that cannot realistically complete the current city upgrade with available stars and legal follow-up gathers.
  - this prevents partial/incomplete investment patterns (sunken-cost starts) where stars are spent but no near-term city level-up can be achieved.
- Implemented resource-investment feasibility helpers in `pol_env/Tribes/py/register_env.py`:
  - `_get_bardur_stars(obs)` to read current Bardur stars from observation payload.
  - `_parse_city_id_from_action_repr(...)` and `_parse_resource_type_from_action_repr(...)` to recover city/resource metadata from legal-action `repr` strings.
  - `_resource_cost_and_population_bonus(...)` to map resource gathers to star cost and population gain assumptions.
  - feasibility solver uses a bounded 0/1 knapsack-style DP to compute minimum star cost needed to reach missing population for upgrade.
- Added fog-of-war-clearance reward shaping in `pol_env/Tribes/py/register_env.py`:
  - new reward constants:
    - `FOG_CLEAR_REWARD_PER_TILE = 0.08`
    - `FOG_CLEAR_REWARD_MAX_TILES = 5`
  - for `MOVE` actions, reward now includes a fog-clearance component based on visible-tile delta between pre-step and immediate post-selected-action observations.
  - fog-clearance contribution is capped per step using max cleared tiles constant.
  - added `_count_visible_tiles(obs)` helper using board terrain visibility (`terrain != 7` treated as visible).
  - added telemetry field `reward_fog_clearance` and included this component in `reward_adjustment`.
- Updated introspection output in `evaluate_brain.py` to include new reward component:
  - `print_reward_breakdown(...)` now reads and prints `reward_fog_clearance`.
  - shaping reconstruction now includes fog-clearance reward in `shaping_sum` and reconstructed-total reporting.
- Updated benchmark registry in `model_run_benchmark_log.md`:
  - added run mapping `Tribes-v0__ppo__1__1777995722` -> `Phase1-Learning-009 (1M)`.

## [Phase1-Learning-008] - 2026-05-05

### Scope
- Learning-only planning and analysis phase focused on post-fog-of-war behavior shaping.
- Objective is to improve model performance by reviewing the latest model runs in detail and identifying candidate soft and hard rules for the next training iteration.

### Planned Review Method
- Replay the latest model runs slowly to inspect decision quality turn-by-turn.
- Incorporate the confirmed finding from the previous model review that the hardcoded opening sequence is already performing perfectly and does not require change in this phase.
- Record repeated policy mistakes, weak priorities, and missed high-value opportunities under true fog-of-war conditions.
- Separate candidate interventions into:
  - soft shaping rules (reward/punishment nudges),
  - hard constraints/guardrails (strict behavior limits or action gating).

### Planned Output
- Produce a shortlist of actionable behavior-shaping rules with clear rationale tied to observed run behavior.
- Prioritize rules that are low-risk, easy to validate, and likely to improve early economy tempo, village capture consistency, and overall growth reliability.
- Carry selected rules into the next model update after this review pass.

### Implemented
- Added richer reward-attribution diagnostics in `evaluate_brain.py`:
  - added `print_reward_breakdown(info, total_reward, truncated=False)` to print per-step reward components:
    - `delta_spt` base reward,
    - `reward_capture_city_bonus`,
    - `reward_second_village_delay_penalty`,
    - `reward_visible_village_neglect_penalty`,
    - `reward_village_breadcrumb`,
    - conditional Turn-10 failure penalty reconstruction (`-3.0`) on truncation.
  - prints shaping sum, `reward_adjustment` from `info`, reconstructed total, and environment-returned total for consistency checks.
  - now calls reward-breakdown logging after each executed step in policy introspection output.
- Refined opening replay logging behavior in `evaluate_brain.py`:
  - in `--show-opening` traced stepping, filters out Tribe 1 opening actions from print/manual-step/render pacing.
  - keeps focus on Bardur-controlled opening actions during slow replay inspection.
  - during replay rendering, now checks step-result `activeTribeID` and only renders when Bardur is active to avoid Tribe 1 visual flicker.
- Updated turn-accounting and reward baseline behavior in `pol_env/Tribes/py/register_env.py`:
  - `step(...)` now captures `start_obs` from `_last_obs` and computes reward baseline from Bardur SPT delta (`base_delta_spt`) instead of relying on Java-returned scalar reward.
  - added forced non-Bardur turn advancement before and after selected action execution via `_force_non_bardur_turns_to_end(...)`.
  - reward is now computed as:
    - `base_delta_spt + reward_adjustment`.
- Added Bardur-perspective helpers in `pol_env/Tribes/py/register_env.py`:
  - `_get_active_tribe_id(obs)` to robustly read `activeTribeID`.
  - `_compute_bardur_spt(obs)` to sum production only from Bardur-owned cities (`tribeID == 0`).
  - `_force_non_bardur_turns_to_end(obs, max_loops=16)` to auto-advance non-Bardur turns using legal `END_TURN` actions.
- Expanded wrapper telemetry in `pol_env/Tribes/py/register_env.py`:
  - added `forced_pre_end_turns`, `forced_post_end_turns`,
  - added Bardur-centric `delta_spt` and `spt`,
  - added current `activeTribeID`,
  - updated `java_done` to reflect combined done state after forced post-action turn advancement.
- Updated benchmark registry in `model_run_benchmark_log.md`:
  - added run mapping `Tribes-v0__ppo__1__1777990600` -> `Phase1-Learning-008 (1M)`.

## [Phase1-Learning-007] - 2026-05-05

### Scope
- Learning-only correction phase dedicated to restoring true fog-of-war training conditions.
- Focus on aligning the environment, observations, and debugging workflow with partial observability so this model trains on the intended information limits.

### Root Cause Correction
- Prior models were effectively trained with fog of war disabled because agent-side full observation was enabled.
- This created an unintended reward/punishment landscape and made prior learning behavior less representative of real partial-observable play.
- This phase corrects that configuration gap before further reward/heuristic iteration.

### Scope Guardrail
- Rule/heuristic retuning items are intentionally deferred to the next model iteration and are not treated as the objective of this phase entry.

### Implemented
- Enabled fog-of-war-constrained agent observation in `pol_env/Tribes/src/core/Constants.java`:
  - set `PLAY_WITH_FULL_OBS = false` for agent-facing gameplay state.
  - retained `GUI_FORCE_FULL_OBS = true` so display behavior remains independently configurable from agent observation mode.
- Updated Python bridge observation generation in `pol_env/Tribes/src/core/game/PythonEnv.java`:
  - `observationJson()` now builds from the active tribe POV copy (`gs.copy(activeTribeID)`) instead of omniscient state.
  - board tile payloads (`terrain`, `resource`, `unit`, `city`, `building`, `network`) now come from the POV-constrained game state.
  - tribe payloads now come from `pov.getTribes()` so hidden enemy data is no longer serialized into Python observations.
  - top-level metadata (`tick`, `gameIsOver`, `activeTribeID`, `gameMode`) is emitted from the same POV copy for consistency.
- Updated rendering path in `pol_env/Tribes/src/core/game/PythonEnv.java`:
  - `renderGui()` now calls `gui.update(gs.copy(gs.getActiveTribeID()), null)` so visual debugging reflects the same fog-constrained state the policy receives.
- Extended debugging and inspection tooling for validating fog-constrained behavior:
  - `evaluate_brain.py`:
    - added robust `MOVE` action `repr` parsing (`parse_move_action_repr`) using regex extraction of unit/destination coordinates.
    - added live unit position lookup from `_last_obs` (`get_unit_pos_from_env_obs`) with id-key and key-scan fallback paths.
    - added `format_action_for_debug(...)` so printed chosen/legal `MOVE` actions show relative deltas (`dX`, `dY`) from current unit tile.
    - added `print_policy_move_grid(...)` to render a relative move-probability grid (`POLICY_MOVE_GRID`) centered on the acting unit.
    - added `--show-opening` CLI flag to replay the hardcoded opening action-by-action before policy control starts.
    - when `--show-opening` is enabled, wraps `tribes_env.step` with tracing to print each opening action and optional manual stepping.
    - preserves render/manual-step integration during opening replay and restores the original `step` function afterward.
    - rebuilds action mask/info after opening replay so downstream introspection uses valid post-opening state.
  - `pol_env/Tribes/py/register_env.py`:
    - expanded `score_move(...)` signature to accept `current_override` and explicit `map_size`.
    - applies recovered current coordinates when missing in action `repr`.
    - adds edge-distance fallback pressure when current coordinates are unknown.
    - resolves `map_size` from local observation once and passes through scoring calls.
    - logs per-candidate move metadata (`unit_id`, origin, destination, deltas, score validity) in `scored_moves`.
    - adds `OPENING_MOVE_GRID` score visualization around the selected unit with board-boundary checks and orientation mapping to match Java view.
    - emits explicit fallback message when unit-origin resolution is unavailable.
- Updated benchmark and run-tracking metadata:
  - `model_run_benchmark_log.md`:
    - added run mapping `Tribes-v0__ppo__1__1777986222` -> `Phase1-Learning-007 (1M)`.
- Added Java source manifest file:
  - `pol_env/Tribes/sources.txt`:
    - introduced a repository-local source-file manifest listing Java source paths under `pol_env/Tribes/src`.
    - intended to support repeatable compile/build invocation workflows that rely on explicit source lists.
- Updated changelog run label precision in historical entry:
  - corrected `Phase1-Learning-006 (43M)` to `Phase1-Learning-006 (43.25M)`.

## [Phase1-Learning-006] - 2026-05-04 - 2026-05-05

### Scope
- Learning-only reward-system simplification pass to reduce variance and tighten capture incentives in `pol_env/Tribes/py/register_env.py`.
- Focused on flatter, safer shaping terms and clearer per-turn village-pressure signals.

### Scope Correction
- Realization recorded: recent training runs were not on a drylands-only map configuration.
- Drylands-only terrain is no longer a project requirement.
- Effective Phase 1 map scope is now:
  - fixed `12x12` map
  - generated/managed using the Tribes framework tooling and level pipeline
  - current default wrapper target remains `levels/phase1_12x12_2bardur.csv`
- Prior drylands references remain in older entries as historical planning context.

### Implemented
- Flattened reward/penalty mechanics by removing exponential (`math.exp`) scaling from active shaping logic.
- Updated city-capture reward:
  - replaced decayed capture bonus with flat `+2.0` per city-count increase (`CAPTURE_CITY_BONUS`).
- Updated delayed second-village penalty:
  - now flat `-0.2` per turn (`SECOND_VILLAGE_DELAY_PENALTY`) starting at Turn 4 while still on one city.
- Updated visible-village neglect penalty:
  - now flat `-0.5` per turn (`VISIBLE_VILLAGE_NEGLECT_PENALTY`) after a 2-turn grace window (`VISIBLE_VILLAGE_NEGLECT_GRACE_TURNS`) when a visible uncaptured village exists and city count is still one.
- Added village "breadcrumb" reward:
  - grants `+0.5` per turn (`VILLAGE_BREADCRUMB_REWARD`) when an owned unit is standing on a visible uncaptured village tile.
  - implemented via `_has_unit_on_visible_uncaptured_village(...)`.
- Removed exploration-based shaping reward path from active logic:
  - deleted exploration reward constants/state and `_compute_exploration_reward(...)` usage.
- Verified capture action mask coverage:
  - `ALLOWED_ACTION_TYPES` explicitly includes both `CAPTURE` and `EXAMINE`.
- Updated reward telemetry keys:
  - replaced decayed capture/exploration telemetry with flat-mechanic fields:
    - `reward_capture_city_bonus`
    - `reward_second_village_delay_penalty`
    - `reward_visible_village_neglect_penalty`
    - `reward_village_breadcrumb`
    - `unit_on_visible_uncaptured_village`
- Added standalone policy-inspection script at repository root:
  - `evaluate_brain.py`
  - loads a `.cleanrl_model` checkpoint and runs a single episode with per-step policy introspection
  - prints turn/SPT, critic value estimate, and legal-action probability distribution before each action
  - supports live Java rendering (`--render-java`) and manual step-through controls (`--manual-step`) for one-action-at-a-time inspection
- Initialized and populated benchmark registry file:
  - `model_run_benchmark_log.md`
  - added run-folder-to-plain-label mapping for the 7 tracked models, including:
    - `Phase1-Learning-006 (1M)`
    - `Phase1-Learning-006 (43.25M)`

## [Phase1-Learning-005] - 2026-05-04

### Scope
- Learning-only iteration focused on expanding the hard-coded opening routine to improve early-game efficiency before policy control takes over.
- No gameplay-rule expansion; this phase is limited to deterministic opening optimization, opening validation, and related telemetry.

### Planned Focus Areas
- Expand the deterministic Bardur opening beyond the current baseline sequence to better position early exploration and village capture tempo.
- Prioritize opening actions that increase early economy and map-control readiness before Turn 2 actions.
- Add stricter fail-fast validation for each required opening action so invalid opening states surface immediately during reset.
- Add opening diagnostics to track which scripted opening steps succeeded, which failed, and at what action index.
- Keep the opening logic modular so alternative opening variants can be tested and benchmarked in future learning iterations.

### Implemented
- Updated `pol_env/Tribes/py/register_env.py` `_apply_bardur_opening(...)` to run a scripted opening through the start of Turn 2 before RL takes over.
- Added robust move parsing for Java `repr` strings using regex (`re.findall`) to extract unit/current/destination coordinates from `MOVE` actions.
- Added nested move-scoring helper `score_move(...)` with weighted incentives for:
  - diagonal movement,
  - center-directed movement (distance-to-center reduction),
  - unit dispersion (distance from a second unit when available).
- Implemented explicit Turn 0 / Turn 1 state-machine sequence:
  - Turn 0: `ANIMAL` harvest x2 -> `LEVEL_UP WORKSHOP` -> best-scored `MOVE` -> `END_TURN`
  - Turn 1: ensure Bardur active -> best-scored warrior `MOVE` -> `WARRIOR` spawn/train action -> `END_TURN`
  - End: fast-forward to Bardur active again and return obs at start of Turn 2.
- Added defensive helper logic to improve reset stability:
  - `ensure_bardur_turn(...)` to safely skip non-Bardur turns via `END_TURN`,
  - try/except fallbacks around scored move execution so blocked/malformed actions are skipped instead of crashing.
- Added explicit opener horizon bookkeeping:
  - sets wrapper `_turn_count = 2` after scripted opening completes so RL starts from Turn 2.
- Removed Turn-0 no-move penalty from active reward shaping in `pol_env/Tribes/py/register_env.py`:
  - deleted `T0_NO_MOVE_PENALTY` usage and telemetry field emission
  - rationale: Turn 0 through Turn 2 are now scripted, so this penalty no longer reflects policy behavior.

## [Phase1-Learning-004] - 2026-05-04

### Scope
- Learning-only iteration (v2) for Phase 1 reward design and exploration pressure.
- No gameplay-rule expansion; changes stay focused on reward shaping, diagnostics, and policy learning behavior.

### Planned Reward Updates
- Add early-turn exploration incentive:
  - small positive reward for newly revealed tiles, with strict per-turn cap
  - reward applies only through Turn 10 horizon and is intended to bootstrap movement/exploration decisions
- Add time-decayed village-capture bonus:
  - village capture reward is strongest early and decays each turn toward Turn 10
  - keeps capture objective strong while prioritizing earlier tempo gains
- Add time-escalating penalty for delayed second village:
  - no penalty before Turn 3
  - penalty starts at Turn 3 and scales up each turn through Turn 10 if second village has not been secured

### Planned Conditional Constraints
- Add visible-village neglect penalty:
  - when at least one capturable/visible village exists and city count is still below 2,
    apply a delayed penalty if village remains uncaptured for multiple turns
  - include counter reset behavior when village ownership changes or visibility is lost
- Add opening-tempo guardrail:
  - apply a strong penalty if the initial unit does not move on Turn 0 when a legal move exists
  - treat blocked/no-legal-move cases as exempt to avoid false penalties

### Planned Telemetry
- Add diagnostic info fields to support reward attribution and tuning:
  - explored tile delta (per step and cumulative)
  - first/second village capture turn
  - turns since first visible uncaptured village
  - initial-unit moved on T0 (boolean)
  - per-component reward breakdown (exploration, capture decay bonus, delay penalty, neglect penalty, opening penalty)

### Implemented
- Updated `pol_env/Tribes/py/register_env.py` with Phase 1 learning-v2 reward shaping:
  - exploration reward for newly visited own-unit tiles:
    - `EXPLORATION_REWARD_PER_NEW_TILE = 0.02`
    - per-turn cap `EXPLORATION_REWARD_CAP_PER_TURN = 0.20`
  - time-decayed village capture bonus:
    - `CAPTURE_REWARD_BASE = 1.5`
    - exponential decay `exp(-0.35 * turn_count)`
    - applied when city count increases
  - time-escalating second-village delay penalty:
    - starts at Turn 3 when still on baseline city count
    - scales by `SECOND_VILLAGE_DELAY_BASE * exp(0.35 * (turn_count - 3))`
  - visible-village neglect penalty:
    - tracks consecutive end-turns with visible uncaptured village while still below second village
    - grace window `VISIBLE_VILLAGE_NEGLECT_GRACE_TURNS = 3`
    - penalty escalates exponentially after grace window
  - Turn-0 opening guardrail:
    - applies `T0_NO_MOVE_PENALTY = -3.0` if no `MOVE` is taken before first end-turn and a legal move existed
  - preserved prior hard-failure horizon penalty:
    - `SECOND_VILLAGE_BY_T10_PENALTY = -3.0` if episode reaches Turn 10 with only baseline city count
- Added wrapper helper/state for shaping logic:
  - `_get_owned_unit_tiles(...)`
  - `_compute_exploration_reward(...)`
  - `_has_visible_uncaptured_village(...)`
  - `_has_legal_move_action(...)`
  - per-episode shaping state reset on `reset(...)`
- Updated observation parsing in wrapper to match live schema:
  - city count and scalar features now read from `obs["tribes"]["0"]`
  - uses `star` key (not `stars`) from tribe payload
- Added reward-attribution telemetry keys in `info`:
  - `reward_exploration`
  - `reward_capture_decay_bonus`
  - `reward_second_village_delay_penalty`
  - `reward_visible_village_neglect_penalty`
  - `reward_t0_no_move_penalty`
  - plus shaping state telemetry:
    - `visible_uncaptured_village`
    - `visible_village_streak_turns`
    - `moved_on_t0`

## [Phase1-Learning-003] - 2026-05-04

### Scope
- This update series is reserved exclusively for training/learning improvements to help the bot learn faster and more reliably.
- No gameplay-rule expansion is planned in this section; changes should focus on optimization, exploration, reward-learning behavior, and policy stability.

### Planned Focus Areas
- PPO learning-curve tuning (entropy behavior, policy/value balance, and stability controls).
- Action-selection learning quality under masked discrete actions.
- Reward-shaping and telemetry-guided learning diagnostics to reduce SPT plateaus.
- Hyperparameter iteration for better economy-growth behavior within the fixed Turn-10 horizon.

### Implemented
- Updated reward progression signal in `pol_env/Tribes/py/gym_env.py`:
  - changed per-step reward from absolute `SPT` to `delta_spt` (`current_spt - previous_spt`)
  - keeps `info["spt"]` and `info["delta_spt"]` telemetry intact
- Added village-expansion milestone rewards in `pol_env/Tribes/py/register_env.py`:
  - `FIRST_EXPANSION_BONUS = +1.0` when city count reaches `starting_city_count + 1`
  - `SECOND_EXPANSION_BONUS = +2.0` when city count reaches `starting_city_count + 2`
  - milestone bonuses are one-time per episode and tracked with:
    - `_first_expansion_rewarded`
    - `_second_expansion_rewarded`
- Added Turn-10 hard-failure penalty in `pol_env/Tribes/py/register_env.py`:
  - `SECOND_VILLAGE_BY_T10_PENALTY = -3.0`
  - applied when wrapper horizon truncates at Turn 10 and city count is still at baseline (`current_city_count <= starting_city_count`)
- Added reward diagnostics in wrapper `info` for monitoring and debugging:
  - `info["city_count"]`
  - `info["starting_city_count"]`
  - `info["reward_adjustment"]`
- Added helper method in wrapper:
  - `_get_city_count(obs)` derives city count from `obs["tribe"]["citiesID"]`
- Added wrapper episode-state initialization for milestone logic:
  - sets `_starting_city_count`, `_last_city_count`, and milestone flags after deterministic Bardur opening on reset

## [Phase1-Stability-002] - 2026-05-04

### Added
- Added masked-action support to PPO policy sampling in `py_rl/cleanrl/cleanrl/ppo.py`:
  - `Agent.get_action_and_value(..., action_mask=...)` now accepts a mask tensor
  - invalid action logits are set to `-1e8` before `Categorical(...)`
  - prevents policy entropy/gradient pollution from illegal actions
- Added vector-info action-mask extraction utility in PPO:
  - reads `infos["action_mask"]` and optional `infos["_action_mask"]`
  - falls back to all-valid mask when absent
- Added Phase 1 diagnostic telemetry in PPO for dashboard debugging:
  - `charts/mean_valid_actions`
  - `charts/non_endturn_rate`
  - `charts/mean_delta_spt`
  - `charts/episode_end_spt`
- Added Java level generation helper tooling for fixed-map workflows:
  - `pol_env/Tribes/src/core/levelgen/GenerateLevelCli.java`
  - `pol_env/Tribes/py/generate_fixed_map.py`
  - enables seed-based map generation and freezing to CSV for reproducible training

### Changed
- Updated PPO rollout loop to feed action masks into action sampling:
  - initializes mask from `envs.reset(...)->infos`
  - refreshes mask every step from `envs.step(...)->infos`
- Added deterministic Bardur Turn-0 heuristic opening in `pol_env/Tribes/py/register_env.py`:
  - on reset, executes:
    - `RESOURCE_GATHERING` (ANIMAL) x2
    - `LEVEL_UP` with `WORKSHOP`
  - agent starts from post-opening state (economy head start)
  - reset now fails fast with explicit errors if expected opening actions are unavailable
- Consolidated training map set to a single fixed CSV:
  - retained only `pol_env/Tribes/levels/phase1_12x12_2bardur.csv`
  - removed all other map CSVs under `pol_env/Tribes/levels/`
- Updated Phase 1 wrapper default map:
  - `PHASE1_LEVEL_FILE` now points to `levels/phase1_12x12_2bardur.csv`
- Updated economic action whitelist in `pol_env/Tribes/py/register_env.py`:
  - added `CAPTURE` so village settlement/city capture can be chosen by policy

### Current Training Observation
- Current long-run behavior still plateaus near `SPT = 4` by Turn 10:
  - `charts/episode_end_spt` flat around 4
  - `charts/mean_delta_spt` flat around 0
  - `charts/non_endturn_rate` decreases over training despite legal action count rising
- Interpretation:
  - the policy is not yet converting exploration/movement into economy growth within the 10-turn horizon
  - additional iteration on map layout, horizon pressure, and policy exploration is still needed

## [Phase1-MVP-001] - 2026-05-04

### Added
- Added strict Phase 1 level file:
  - `pol_env/Tribes/levels/phase1_bardur_drylands.csv`
  - Bardur capital (`c:2`) plus a distant dummy enemy capital (`c:1`) to keep Java game state alive
  - drylands-style terrain only (no shallow/deep water tiles)
- Added quick verification script:
  - `test_phase1_constraints.py`
  - validates turn-10 truncation and whitelist-constrained action execution
- Added custom PPO telemetry in `py_rl/cleanrl/cleanrl/ppo.py`:
  - logs `charts/custom_spt_return` from `infos["spt"]` whenever an env episode ends (`truncations` or `terminations`)
  - fallback extraction from `infos["final_info"][env_idx]["spt"]` when direct vector key is unavailable
- Added explicit model checkpoint options to PPO CLI:
  - `--save-model`
  - `--model-path`
  - saves `agent.state_dict()` at end of training
- Added periodic checkpointing controls to PPO CLI:
  - `--save-frequency` (default: `500000`)
  - when `--save-model` is enabled, checkpoint files are written during training as:
    - `runs/<run_name>/model_checkpoint_<global_step>.cleanrl_model`
  - final model save remains:
    - `runs/<run_name>/ppo.cleanrl_model` (or custom `--model-path`)

### Changed
- Updated `pol_env/Tribes/py/register_env.py` to enforce strict Phase 1 wrapper constraints:
  - default level file now `levels/phase1_bardur_drylands.csv`
  - added whitelist filter for legal action types:
    - `END_TURN`, `MOVE`, `EXAMINE`, `RESOURCE_GATHERING`, `CLEAR_FOREST`,
      `GROW_FOREST`, `LEVEL_UP`, `RESEARCH_TECH`, `BUILD`
  - action remapping now uses whitelist-index -> raw Java action index mapping
  - added per-step action mask (`info["action_mask"]`) and action debug metadata
  - added wrapper turn counter based on executed `END_TURN` actions
  - `step()` now returns `truncated=True` when turn counter reaches 10 (unless already terminated)
  - added Phase 1 terminal override in `step()`:
    - Java `done` is ignored (`terminated=False` always in wrapper output)
    - horizon control is Python-only via `truncated=(turn_count >= 10)`
    - debug metadata includes `info["java_done"]` and `info["terminated_overridden"]`
- Updated `pol_env/Tribes/py/gym_env.py` reward logic:
  - removed prior score/relative/terminal shaping
  - reward is now dense absolute Stars-Per-Turn (SPT), computed from observation city production
  - added `info["spt"]` and `info["delta_spt"]`
- Updated strict Phase 1 level to prevent immediate Java single-player game-over:
  - `phase1_bardur_drylands.csv` now includes a distant dummy enemy capital (`c:1`)
  - keeps Java state alive while Python controls episode horizon
- Updated `pol_env/Tribes/py/register_env.py` fail-fast behavior:
  - removed synthetic `NO_OP` fallback path
  - wrapper now raises `RuntimeError` when Java returns zero legal actions

### Validation
- Ran smoke test:
  - `python py_rl/cleanrl/cleanrl/ppo.py --env-id Tribes-v0 --num-envs 4 --total-timesteps 4096 --no-track --save-model`
  - output included `model_saved=runs\Tribes-v0__ppo__1__1777901078\ppo.cleanrl_model`
- Verified TensorBoard event tags in smoke run:
  - `charts/custom_spt_return` present with 149 scalar points
  - confirms custom SPT telemetry is being emitted even when default episodic stats are absent

### Documentation
- Updated `docs/TRAINING.md` with:
  - checkpointing usage for long runs
  - tracked 5,000,000-step launch command with online W&B sync

## [Preflight-002] - 2026-05-04

### Added (Async + Benchmark)
- Added dependency lock file at repository root:
  - `requirements-lock.txt` (captured via `python -m pip freeze`)
  - records exact package versions used in the validated local run environment
- Updated `docs/TRAINING.md` with:
  - preflight sanity command for async PPO smoke run
  - recommended `requirements-lock.txt` workflow for reproducibility

### Validation
- Completed short end-to-end async PPO sanity run:
  - command: `python py_rl/cleanrl/cleanrl/ppo.py --total-timesteps 6144 --num-steps 64 --no-track --no-capture-video --startup-jitter-min-s 0.1 --startup-jitter-max-s 2.0`
  - result: exit code `0`
  - observed SPS progression: `766 -> 1192`
- Completed JVM orphan-process cleanup check around training run:
  - captured Java PIDs before and after run
  - result: `JAVA_NEW=` (no new lingering Java process IDs after `envs.close()`)
- Confirmed warnings cleanup from Tyro typing updates:
  - no `wandb_entity` / `target_kl` None-type parser warnings in sanity run after `Optional[...]` fix

### Added
- Added `py_rl/cleanrl/cleanrl/benchmark_async_vector_envs.py`:
  - runs a quick throughput matrix for `num_envs` in `[12, 16, 20]`
  - executes `ppo.py` with Async settings and startup jitter
  - parses `SPS` output and reports `tail_avg_sps`, `all_avg_sps`, and `peak_sps`
  - auto-sets `POLYVISION_VERBOSE_RESETS=0` for cleaner benchmark logs

### Changed
- Updated `py_rl/cleanrl/cleanrl/ppo.py` to use true parallel stepping:
  - replaced `gym.vector.SyncVectorEnv(...)` with `gym.vector.AsyncVectorEnv(...)`
  - configured multiprocessing context as `context="spawn"` for Windows-safe worker startup
- Added randomized startup jitter in `ppo.py` env factory to reduce JVM launch storms:
  - each worker sleeps a random delay before creating the env
  - default jitter range: `0.1s` to `2.0s`
  - exposed as CLI args:
    - `--startup-jitter-min-s`
    - `--startup-jitter-max-s`
- Updated `pol_env/Tribes/py/register_env.py` reset logging behavior:
  - `Reset: Available actions = ...` now prints only when `POLYVISION_VERBOSE_RESETS=1`
  - default behavior is quiet (`POLYVISION_VERBOSE_RESETS=0`)
- Fixed Windows JVM classpath joining in `pol_env/Tribes/py/gym_env.py`:
  - replaced hardcoded `":"` separator with `os.pathsep`
  - resolves `JavaPackage object is not callable` startup failure when loading `core.game.PythonEnv` on Windows
- Updated `py_rl/cleanrl/cleanrl/ppo.py` defaults and typing:
  - changed `Args.num_envs` default from `4` to `12` for Phase 1 local throughput
  - changed `Args.wandb_entity` to `Optional[str]`
  - changed `Args.target_kl` to `Optional[float]`
  - removes Tyro warnings caused by `None` defaults on non-optional typed fields

### Benchmark
- Executed async throughput benchmark (`py_rl/cleanrl/cleanrl/benchmark_async_vector_envs.py`) on 2026-05-04 with:
  - `AsyncVectorEnv(context="spawn")`
  - startup jitter `0.1s` to `2.0s`
  - `num_envs` in `[12, 16, 20]`
- Results:
  - `num_envs=12`: `tail_avg_sps=1561`, `all_avg_sps=1377`, `peak_sps=1596`
  - `num_envs=16`: `tail_avg_sps=1172`, `all_avg_sps=1097`, `peak_sps=1194`
  - `num_envs=20`: `tail_avg_sps=1058`, `all_avg_sps=937`, `peak_sps=1117`
- Justification for defaulting to `num_envs=12`:
  - best sustained throughput (`tail_avg_sps`) and best peak SPS in benchmark
  - higher env counts degraded throughput due JVM/process overhead in this architecture

## [Preflight-001] - 2026-05-03

### Added
- Created `docs/PLAIN_ENGLISH_GUIDE.md`, a non-technical overview covering:
  - how the codebase works in plain English
  - project goals and phased direction
  - practical steps for designing and training a custom model
- Created `docs/ENVIRONMENT_API.md` with documented contracts for:
  - `TribesGymEnv` reset/step/render/list_actions/close behavior
  - `TribesGymWrapper` observation flattening, action handling, and Gymnasium tuple shape
  - current reward composition and runtime notes
- Created `docs/MVP_PHASE1_SPEC.md` defining Phase 1 constraints and acceptance criteria.
- Created `docs/TESTING.md` with ordered smoke-test workflow and troubleshooting.
- Created `docs/TRAINING.md` with baseline/custom training commands and evaluation entrypoints.
- Added upstream attribution sections to:
  - `README.md`
  - `pol_env/Tribes/README.md`

### Changed
- Reworked root `README.md` into the canonical project entrypoint with:
  - docs navigation map
  - quickstart (compile, venv, install, smoke run)
  - explicit headless runtime expectation note
  - current scope statement for Phase 1 status
- Consolidated high-level project vision into `README.md` as canonical source, including:
  - project vision statement
  - Phase 1 MVP framing
  - architecture/tech stack summary
  - multi-phase roadmap
  - immediate next steps
- Updated `pol_env/Tribes/README.md` to explicitly defer canonical project docs to root `README.md` + `docs/`.
- Expanded root `.gitignore` to cover local runtime artifacts:
  - `pol_env/Tribes/img_step_*.png`
  - `pol_env/Tribes/game_recording_*.mp4`
  - `pol_env/Tribes/py/polytopia_simulator.log`
  - `pol_env/Tribes/game_moves_*.json`
  - `.DS_Store`
  - `*.py[cod]`

### Fixed
- Removed unused `torch` import from `pol_env/Tribes/py/register_env.py`, resolving fresh-setup failure in `test_simple.py` (`ModuleNotFoundError: No module named 'torch'`).

### Removed
- Removed generated/accidental artifacts from tracked workspace files:
  - `pol_env/.DS_Store`
  - `pol_env/Tribes/game_recording_20250914_041435.mp4`
  - `pol_env/Tribes/img_step_0.png`
  - `pol_env/Tribes/img_step_1.png`
  - `pol_env/Tribes/img_step_2.png`
  - `pol_env/Tribes/img_step_3.png`
  - `pol_env/Tribes/img_step_4.png`
  - `pol_env/Tribes/py/polytopia_simulator.log`
  - `py_rl/cleanrl/=0.10.9.7`
- Removed `polyvision_plan.md` after consolidating high-level vision/roadmap ownership into `README.md`.

### Validation
- Verified runner execution:
  - `python pol_env/Tribes/py/run_gym.py` -> exit code `0`
  - `cd pol_env/Tribes/py && python run_gym.py` -> exit code `0`
- Verified smoke test after fixes:
  - `python test_simple.py` -> successful run
- Verified expected headless behavior:
  - Java GUI warning (`HeadlessException` / missing `DISPLAY`) may appear while simulation continues.
