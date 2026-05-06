# Changelog

All notable changes to this project are documented in this file.

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
