# Changelog

All notable changes to this project are documented in this file.

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
