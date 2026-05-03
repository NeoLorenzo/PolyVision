# Environment API

This project exposes Polytopia (`Tribes`) to Python at two layers:

1. `TribesGymEnv` (low-level bridge): `pol_env/Tribes/py/gym_env.py`
2. `TribesGymWrapper` (Gymnasium-compatible): `pol_env/Tribes/py/register_env.py`

## 1) Low-Level Bridge: `TribesGymEnv`

### Construction
`make_default_env()` builds `TribesGymEnv` with:
- classpath classes: `pol_env/Tribes/out`
- JSON library: `pol_env/Tribes/lib/json.jar`

If no Py4J port is provided, the wrapper launches a JVM process automatically.

### `reset(level_file, seed=42, mode="SCORE") -> dict`
Initializes a game and returns raw JSON observation (decoded into Python dict).

Important parameters:
- `level_file`: path to level CSV
- `seed`: deterministic seed
- `mode`: Java `GAME_MODE` enum name (`"SCORE"` currently used in scripts)

### `step(action_index) -> (obs, reward, done, info)`
Executes a single indexed action.

Returned fields:
- `obs` (`dict`): raw JSON observation
- `reward` (`float`): computed in Python wrapper (see formula below)
- `done` (`bool`): terminal from Java env
- `info` (`dict`): metadata including tick, active tribe, score vector, and reward components

Reward formula (current implementation):
- `progress_reward = (tribe0_score_after - tribe0_score_before) / 100.0`
- `relative_reward = (tribe0_score_after - mean(other_scores)) / 1000.0`
- `step_penalty = -0.01`
- `reward = progress_reward + relative_reward + step_penalty`
- Terminal bonus/penalty:
  - `+100` if tribe 0 wins
  - `-100` otherwise

### `action_space_n -> int`
Returns the current number of legal actions from Java for the active state.

### `list_actions() -> list[dict]`
Returns legal actions as structured objects (parsed from JSON), typically including:
- `type`
- `repr`
- action-specific fields

### `render(mode)`
Supported modes:
- `"ansi"`: returns textual board + metadata
- `"human"`: prints the textual board
- `"rgb_image"`: returns `PIL.Image`
- `"rgb_array"`: returns pixel data
- `"java"`: opens/updates Java Swing GUI (requires display server)

### `close()`
Shuts down the Py4J gateway.

## 2) Gymnasium Wrapper: `TribesGymWrapper`

Registered env id:
- `Tribes-v0`

### Observation space
- Flattens selected observation fields into a float32 vector:
  - `board.terrain` (flattened)
  - `board.unitID` (flattened)
  - `board.cityID` (flattened)
  - tribe features: stars, score, city count, nKills
  - game features: tick, activeTribeID

### Action space
- Uses fixed `gym.spaces.Discrete(200)` to accommodate dynamic legal action counts.
- During `step(action)`, if sampled action is out of legal range, it is mapped with modulo:
  - `action = action % current_valid_action_count`

### `reset(seed=None, options=None)`
Returns:
- flattened observation (`np.ndarray`)
- info dict containing `valid_actions`

### `step(action)`
Returns Gymnasium 5-tuple:
- `(obs_array, reward, done, truncated, info)`

Notes:
- `truncated` is currently always `False`.
- `info` includes `valid_actions` and `original_action` (mapped action index).

## 3) Operational Notes

- Java classes must be compiled to `pol_env/Tribes/out` before running.
- In headless environments, `render("java")` may raise/display `HeadlessException`; simulation can still run.
- `run_gym.py` writes demo frame images (`img_step_*.png`) under `pol_env/Tribes/`.

## 4) Known Gaps Relevant to Phase 1 MVP

Current environment behavior is general-purpose and still includes non-economic actions.
Phase 1 constraints (Bardur + Drylands + 10-turn + no-combat + SPT objective) are not fully enforced yet.
See `docs/MVP_PHASE1_SPEC.md` for implementation targets.
