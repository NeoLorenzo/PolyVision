# PolyVision

PolyVision is a reinforcement-learning research environment for training, evaluating, and diagnosing agents that play an economy-first version of *The Battle of Polytopia*. It combines a headless Java game engine (`Tribes`) with Python through Py4J, exposes the game through Gymnasium, and includes a CleanRL-derived PPO trainer with a legality-aware action interface.

The current implementation is centered on a reproducible Phase 1 task: control Bardur through Turn 10, explore a pool of 12×12 Drylands maps, capture neutral villages, grow cities, research useful technologies, and maximize Stars Per Turn (SPT). Combat actions are excluded from the Phase 1 policy interface.

> **Project status:** Phase 1 is implemented and actively being iterated. The repository can compile and run the Java engine, launch Gymnasium environments, train PPO policies across parallel JVM-backed environments, save and evaluate checkpoints, validate the action interface, profile throughput, and run targeted tactical/economic diagnostics. It is still a research codebase rather than a packaged library or production game-playing application.

## What works today

| Capability | Current implementation |
|---|---|
| Java game simulation | Legacy `Tribes` engine runs headlessly and is launched automatically by the Python bridge. |
| Python bridge | Py4J wrapper supports reset, step, legal-action enumeration, observation retrieval, text/image rendering, and JVM cleanup. |
| Gymnasium environment | `Tribes-v0` provides fixed observation and action spaces, Gymnasium reset/step semantics, episode statistics, and multiprocessing-safe info payloads. |
| Phase 1 horizon | The wrapper owns episode termination and truncates after Bardur completes Turn 10. |
| Phase 1 maps | Bundled 12×12 map pools support deterministic round-robin or seeded-random selection. The repository includes a 128-map Phase 1 pool and a 256-map Bardur-solo pool. |
| Bardur opening | Reset executes a deterministic scripted Bardur opening before policy control begins. Policy-visible play starts after the opening, with turn accounting aligned to the Turn 10 objective. |
| Solo play | `POLYVISION_SOLO_NO_OPPONENT_MODE=1` removes opponent participation for the recommended Phase 1 training setup. When it is disabled, non-Bardur turns are fast-forwarded with `END_TURN`. |
| Economy-first action set | Movement, village capture, spawning, resource gathering, forestry actions, buildings, technology research, city level-ups, examine, and end-turn actions are supported. `ATTACK` is not exposed to the Phase 1 policy. |
| Stable action IDs | Dynamic Java legal actions are canonicalized into a deterministic global action catalog. There is no modulo remapping of arbitrary policy outputs. |
| Legal-action policies | PPO can score only the current legal slots (`legal_only`) or combine legal IDs with 42 semantic/economic action features (`legal_features`). A dense masked actor remains available for debugging. |
| Reward shaping | Dense SPT rewards are combined with exploration, neutral-village, capture, and tactical shaping. An additional terminal SPT bonus is configurable and disabled by default. |
| Validation | Training can run a strict 10,000-state preflight validator, cache successful validation fingerprints, reject action collisions/canonicalization gaps, and abort on excessive illegal/fallback rates. |
| Training | Vectorized PPO supports CUDA, deterministic seeds, TensorBoard, optional Weights & Biases, periodic checkpoints, final model export, and action-interface metadata sidecars. |
| Evaluation | The repository includes checkpoint introspection, greedy baselines, an Organization-focused oracle comparison, hidden-village oracle analysis, and human terminal play. |
| Diagnostics | Targeted scripts cover capture legality, visible-village targeting, coordinate consistency, legal feature tensors, reward behavior, and environment throughput. |
| Profiling | Optional end-to-end SPS profiling breaks time down across JVM calls, observation parsing, legal-action generation, feature construction, reward calculation, info construction, rollout collection, and PPO updates. |

## Phase 1 task

The implemented Phase 1 task is designed to isolate early-game economy and exploration before introducing adversarial combat.

### Objective

- Controlled tribe: Bardur (tribe ID `0` in the current task)
- Map family: 12×12 Drylands Phase 1 maps
- Horizon: through the completion of Bardur Turn 10
- Primary outcome: final Stars Per Turn
- Supporting outcomes: village capture, city growth, useful research, resource conversion, exploration, and efficient movement
- Combat: `ATTACK` is filtered from the policy-visible action set

SPT is calculated from the production of cities owned by Bardur. The wrapper records final SPT and a much larger set of supporting metrics so two agents with similar reward can still be compared behaviorally.

### Default maps versus recommended solo maps

The wrapper's code-level fallback is `levels/phase1_12x12_2bardur.csv`. If no pool override is supplied and the bundled default pool exists, it uses:

```text
levels/phase1_pool/*.csv                 # 128 maps
```

The recommended no-opponent training configuration uses:

```text
levels/phase1_pool_bardur_solo/*.csv     # 256 maps
```

Set the pool and solo behavior explicitly for comparable experiments:

```powershell
$env:POLYVISION_LEVEL_POOL_GLOB='levels/phase1_pool_bardur_solo/*.csv'
$env:POLYVISION_LEVEL_SELECTION_MODE='round_robin'
$env:POLYVISION_SOLO_NO_OPPONENT_MODE='1'
```

`round_robin` is deterministic and cycles through the pool. `seeded_random` uses a separate seeded random stream for map choice. `POLYVISION_BASE_SEED` controls the wrapper's default episode-seed stream when callers do not pass a seed to `reset()`.

### Map geometry and checkpoint compatibility

One `TribesGymWrapper` instance requires a dimension-homogeneous level pool. The wrapper fixes its observation space and global action catalog from its bootstrap map, validates every later reset against that geometry, and rejects rectangular or mixed-dimension pools before returning an incompatible observation.

The synthetic Phase-1 pools are currently 12x12. Genuine maps under `levels/phase1_pool_bardur_real/` are 11x11 and remain inactive by default. Their interface sizes differ:

| Pool geometry | Observation dimension | Global action-space size |
|---|---:|---:|
| 12x12 | 597 | 89,305 |
| 11x11 | 505 | 63,913 |

PPO checkpoints are geometry- and action-interface-specific. Metadata validation checks map dimensions, observation dimension, action-space size, action-catalog fingerprint, actor mode, and legal-action feature version/dimension before loading model tensors. A 12x12 checkpoint cannot be loaded into an 11x11 environment, and checkpoints without sufficient compatibility metadata are rejected. Action-validator caches are likewise keyed by geometry, interface metadata, and a content hash of the resolved map pool.

Run the reset-only genuine-pool contract check with:

```powershell
python tools/validate_environment_contract.py `
    --level-pool-glob 'levels/phase1_pool_bardur_real/*.csv' `
    --expected-width 11 `
    --expected-height 11
```

### Scripted opening and turn handling

`TribesGymWrapper.reset()` loads the selected map and seed, executes the scripted Bardur opening, initializes economy/research counters, builds the legal action set, and then returns policy control. The opening advances the underlying game before the learned policy acts; wrapper turn accounting therefore deliberately differs from a raw engine reset.

The Gym wrapper ignores the Java engine's general terminal condition for this task. It reports:

- `terminated = False` for the Phase 1 horizon;
- `truncated = True` after Bardur finishes Turn 10.

When the opponent is present, the wrapper automatically ends non-Bardur turns so the policy is only asked to make Bardur decisions. In recommended solo mode, the Java bridge disables opponent participation directly.

### Allowed and filtered actions

The Phase 1 wrapper admits these raw action families:

- `END_TURN`
- `MOVE`
- `CAPTURE` (primarily neutral-village/city acquisition in Phase 1)
- `EXAMINE`
- `SPAWN` / canonical `TRAIN`
- `RESOURCE_GATHERING`
- `CLEAR_FOREST`
- `GROW_FOREST`
- `LEVEL_UP`
- `RESEARCH_TECH`
- `BUILD`

It rejects action families outside that set, including combat attacks. It also applies task-specific guardrails:

- out-of-bounds movement is rejected;
- `CITY_WALL` level-up choices are hard-masked;
- an optional resource-gathering filter can restrict gathers to city-upgrade-relevant choices;
- before Bardur owns two cities, legal village captures and visible-village progress can be prioritized;
- early immediate backtracking for the opening unit is constrained;
- canonicalization failures and global-ID collisions fail fast.

These guardrails mean Phase 1 is not an untouched full-game Polytopia environment. It is a deliberately shaped curriculum for early expansion and economic learning.

## Architecture

```text
PPO / evaluation / human controller
                │
                ▼
Gymnasium TribesGymWrapper (`Tribes-v0`)
  - Phase 1 horizon and scripted opening
  - observation flattening
  - global action catalog + legal slots
  - action filtering and reward shaping
  - metrics, diagnostics, and profiling
                │
                ▼
Low-level TribesGymEnv
  - Py4J process lifecycle
  - JSON observation/action transport
  - raw reset, step, render, and action listing
                │
                ▼
Java `Tribes` engine / PythonEnv bridge
```

### Main components

| Path | Responsibility |
|---|---|
| `pol_env/Tribes/src/` | Java game engine and the `core.game.PythonEnv` bridge class. |
| `pol_env/Tribes/py/gym_env.py` | Low-level Py4J client and JVM lifecycle. |
| `pol_env/Tribes/py/register_env.py` | Gymnasium wrapper, Phase 1 behavior, action catalog, features, rewards, and telemetry. |
| `pol_env/Tribes/levels/` | Fixed maps and generated Phase 1 map pools. |
| `py_rl/cleanrl/cleanrl/ppo.py` | Primary PPO implementation used for current training. |
| `evaluate_brain.py` | Single-checkpoint policy introspection and optional live Java rendering. |
| `tools/` | Human play, reward smoke tests, and policy/oracle comparison utilities. |
| `runs/` | TensorBoard logs, checkpoints, final models, and action-interface metadata. |
| `outputs/` | Evaluation reports, human-run traces, and SPS profiles. |

## Environment interfaces

PolyVision exposes two related APIs.

### Low-level Py4J bridge

`pol_env.Tribes.py.gym_env.TribesGymEnv` is useful when inspecting the engine directly.

```python
from pol_env.Tribes.py.gym_env import make_default_env

env = make_default_env()
obs = env.reset("levels/SampleLevel.csv", seed=42, mode="SCORE")
actions = env.list_actions()
obs, reward, done, info = env.step(0)
print(env.render(mode="ansi"))
env.close()
```

Core operations:

- `reset(level_file, seed=42, mode="SCORE") -> dict`
- `list_actions() -> list[dict]`
- `action_space_n -> int`
- `step(raw_java_action_index) -> (dict, float, bool, dict)`
- `render("ansi" | "human" | "rgb_image" | "rgb_array" | "java")`
- `close()`

At this layer, reward is the raw step-to-step SPT delta. Phase 1 filtering, shaped reward, global action IDs, and Turn 10 truncation belong to the Gymnasium wrapper.

### Gymnasium wrapper

Importing `pol_env.Tribes.py.register_env` registers `Tribes-v0`:

```python
import gymnasium as gym
import pol_env.Tribes.py.register_env  # registers Tribes-v0

env = gym.make("Tribes-v0")
obs, info = env.reset(seed=42)

# Choose from current legal global IDs, not an arbitrary integer.
valid = info["legal_action_valid_mask"]
legal_ids = info["legal_global_ids_padded"]
action = int(legal_ids[valid][0])

obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

Always call `close()` (or use a `try/finally`) when constructing environments directly. Each low-level environment owns a JVM gateway.

## Observation model

The Gymnasium observation is a flat `float32` vector. It retains the original observation prefix for checkpoint compatibility and appends newer visible-information economy features.

The vector includes:

- flattened terrain, unit-ID, and city-ID boards;
- Bardur stars, score, city count, and kill count;
- engine tick and active tribe;
- a fog-masked visible resource board;
- normalized current stars and current SPT;
- current turn and two turns-remaining representations;
- Organization and Forestry research flags;
- normalized researched-technology count;
- city count, average/max city level, and city-upgrade progress summaries;
- fraction of cities ready to level up and whether any level-up is available.

Fogged resources are masked before being appended. Normal training observations are intended to remain fog-of-war constrained; scripts explicitly labeled as privileged or no-fog diagnostics must not be treated as fair policy baselines.

## Action model

The Java engine exposes a dynamic list of legal actions whose raw indices can change every step. PolyVision converts those actions into a deterministic, dimension-dependent global catalog.

### Global action catalog

The catalog assigns stable ranges to:

- end turn;
- source-to-destination moves;
- source-to-target captures with capture-mode variants;
- unit training/spawning by unit type and city tile;
- resource gathering by resource type and tile;
- clear/grow forest by tile;
- building type and tile;
- technology research;
- city level-up choice and tile;
- examine by tile.

Every legal raw Java action must canonicalize to exactly one global ID. The wrapper builds a binary action mask and a global-ID-to-raw-index mapping on reset and after every step. An illegal sampled global ID is not silently wrapped with modulo; the wrapper records the event and uses a controlled `END_TURN` fallback. PPO can abort if illegal samples or fallbacks exceed configured thresholds.

### Sparse legal-slot tensors

The wrapper also emits fixed-width sparse tensors so PPO does not need to score the entire global catalog:

- `legal_global_ids_padded`: global IDs for current legal choices;
- `legal_action_valid_mask`: distinguishes real slots from padding;
- `legal_action_count`: number of usable slots;
- `legal_action_features_padded`: per-slot semantic/economic features.

The current default capacity is 256 legal slots. The wrapper and trainer fail fast if the legal set exceeds the configured capacity; use the same `--max-legal-actions` value for training and evaluation metadata. A previous 128-slot trial failed at 1,832,960 training steps when a state exposed 133 legal actions, so 128 must not be used for long runs.

### Legal-action features

`legal_features` mode augments each legal action with 42 features. They cover:

- movement and fog reveal estimates;
- adjacent-fog counts and reveal deltas;
- visible neutral-village targeting and distance progress;
- immediate backtracking and capital-distance movement;
- warrior movement and action-type one-hot indicators;
- research identity, including Organization and Forestry;
- resource identity, including animals, fruit, fish, crops, and metal;
- building identity, including lumber huts and sawmills;
- level-up identity, including Workshop;
- expected population and immediate SPT deltas;
- whether an action makes a city level-up available;
- city-upgrade readiness and progress before the action.

The feature schema is versioned (`v1_3_move_focus_plus_semantic_econ`) and saved with checkpoints so incompatible interfaces can be detected.

## Reward and episode metrics

The wrapper's base reward is Bardur's step-to-step SPT delta:

- positive SPT deltas are multiplied by `5.0`;
- zero or negative deltas use multiplier `1.0`.

The current shaped reward can additionally include:

- revealing an uncaptured neutral village;
- moving closer to or onto a visible neutral village;
- breadcrumb progress toward village acquisition;
- a bounded reward for clearing fog;
- a penalty for a movement action that reveals no fog when revealing moves existed;
- a city/village capture bonus scaled within configured bounds;
- a tactical reward or penalty around visible-village movement choices.

Some historical shaping hooks remain present but are intentionally zeroed for measurement, including hold-on-village and move-off-village terms.

An optional terminal SPT bonus can be enabled with `POLYVISION_TERMINAL_SPT_REWARD_ENABLED=1`. It adds weighted components for final SPT, SPT above 10, and SPT above 15. The defaults are weights `1.0`, `2.0`, and `3.0`; the feature itself is **disabled by default**.

Episode and diagnostic info includes, among other fields:

- final/current SPT, reward, stars, city count, city levels, and unit count;
- captured villages, capturable villages, and capture percentage;
- first visible-village turn and second-city capture turn;
- researched technologies and Forestry/Organization timing;
- animals/fruit harvested, lumber huts/sawmills built, and forests cleared;
- fog tiles cleared;
- expected-versus-actual population and SPT deltas for selected economy actions;
- missed capture, missed level-up, missed city-upgrade completion, and useful-move counters;
- action canonicalization, collision, illegal-sample, and fallback statistics;
- map ID, pool index, episode seed, and selection mode in debug information;
- detailed timing fields when SPS profiling is active.

`POLYVISION_INFO_MODE=fast` minimizes expensive payloads for training. `debug` exposes large action and diagnostic structures. `train` is also accepted for trainer-oriented payload behavior.

## Setup

### Prerequisites

- Python 3.9 or newer
- A Java JDK (Java 8 or newer) providing both `java` and `javac`
- `pip`
- Optional: CUDA-capable PyTorch for GPU PPO updates
- Optional: Weights & Biases for experiment tracking

Verify the required runtimes:

```powershell
python --version
java -version
javac -version
```

### 1. Compile the Java engine

PowerShell, from the repository root:

```powershell
Set-Location pol_env/Tribes
New-Item -ItemType Directory -Force out | Out-Null
$sources = Get-ChildItem -Path src -Recurse -Filter *.java | ForEach-Object FullName
javac -cp "lib/json.jar" -d out -sourcepath src $sources
Set-Location ../..
```

Bash, from the repository root:

```bash
cd pol_env/Tribes
mkdir -p out
find src -name "*.java" -exec javac -cp "lib/json.jar" -d out -sourcepath src {} +
cd ../..
```

Successful compilation creates `pol_env/Tribes/out/core/game/PythonEnv.class`.

### 2. Create a Python environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip wheel
python -m pip install -r pol_env/Tribes/py/requirements.txt
python -m pip install -r py_rl/requirements.txt
```

For exactly reproduced local experiments, `requirements-lock.txt` records a fuller environment snapshot. The two smaller requirements files remain the clearest statement of direct bridge/training dependencies.

## Smoke tests

Run these from the repository root after compilation and dependency installation:

```powershell
# Ensures the checkmark/cross characters used by the scripts render on legacy Windows consoles.
$env:PYTHONUTF8='1'

python test_simple.py
python test_gym.py
python test_env.py
```

Direct bridge/render smoke test:

```powershell
Set-Location pol_env/Tribes/py
python run_gym.py
Set-Location ../../..
```

Expected behavior:

- the JVM starts and the environment resets;
- `Tribes-v0` can be created;
- observations and legal actions are returned;
- steps execute and the JVM closes afterward;
- `run_gym.py` may save `img_step_*.png` frames under `pol_env/Tribes/`;
- Java GUI rendering can fail with `HeadlessException` or a display warning on terminal-only machines while headless simulation continues normally.

If a Windows console reports a `UnicodeEncodeError` while printing a checkmark, set `PYTHONUTF8=1` as shown above. That is a console-encoding issue rather than an environment failure.

The root scripts are practical integration/smoke checks, not a comprehensive CI suite. Much of `py_rl/cleanrl/tests/` belongs to upstream CleanRL and is not PolyVision-specific coverage.

## PPO training

The primary trainer is:

```text
py_rl/cleanrl/cleanrl/ppo.py
```

Current defaults include `Tribes-v0`, 12 parallel environments, 128 rollout steps, 500,000 total timesteps, CUDA when available, deterministic PyTorch behavior, and strict action-interface validation.

### Actor modes

| Mode | Behavior | Intended use |
|---|---|---|
| `legal_only` | Encodes state and current legal global IDs, then scores legal candidates with state/action embeddings. | Current default and efficient baseline. |
| `legal_features` | Adds the 42-feature semantic/economic vector for every legal slot before scoring. | Experiments that need explicit action meaning and predicted economic effects. |
| `dense_debug` | Produces logits for the entire global action space and applies a dense mask. | Debugging and compatibility checks; less memory-efficient. |

### Short training smoke run

PowerShell:

```powershell
$env:POLYVISION_LEVEL_POOL_GLOB='levels/phase1_pool_bardur_solo/*.csv'
$env:POLYVISION_LEVEL_SELECTION_MODE='round_robin'
$env:POLYVISION_SOLO_NO_OPPONENT_MODE='1'
$env:POLYVISION_INFO_MODE='fast'

python py_rl/cleanrl/cleanrl/ppo.py `
  --env-id Tribes-v0 `
  --actor-mode legal_only `
  --max-legal-actions 256 `
  --num-envs 12 `
  --num-steps 64 `
  --total-timesteps 6144 `
  --no-track `
  --no-capture-video `
  --no-validate-action-interface
```

This skips the 10,000-state validator so the smoke run starts quickly. Do not interpret a short smoke run as a policy-quality result.

### Full tracked and checkpointed run

```powershell
$env:POLYVISION_LEVEL_POOL_GLOB='levels/phase1_pool_bardur_solo/*.csv'
$env:POLYVISION_LEVEL_SELECTION_MODE='round_robin'
$env:POLYVISION_SOLO_NO_OPPONENT_MODE='1'
$env:POLYVISION_INFO_MODE='fast'
$env:POLYVISION_STRICT_COORD_ASSERT='0'

# Record locally during training and sync later to avoid live-network overhead.
$env:WANDB_MODE='offline'
$env:WANDB_CONSOLE='off'
$env:WANDB_SILENT='true'

python py_rl/cleanrl/cleanrl/ppo.py `
  --env-id Tribes-v0 `
  --actor-mode legal_features `
  --max-legal-actions 256 `
  --num-envs 20 `
  --total-timesteps 5000000 `
  --track `
  --save-model `
  --save-frequency 500000 `
  --enable-step-diagnostics `
  --step-diagnostics-log-every 3
```

The strict action-interface validator is enabled by default here. A successful fingerprint is cached under `.cache/action_validator/` and reused while the relevant interface code and configuration remain unchanged. Use `--force-revalidate-action-interface` when you explicitly want a fresh validation pass.

> **W&B chart requirement:** Keep `--enable-step-diagnostics` enabled for training runs that will be compared by gameplay metrics. In the current trainer, this flag controls not only detailed debugging data but also the logging path for final SPT, custom SPT return, unit count, stars, reward, city count, fog cleared, economy summaries, mean valid actions, non-end-turn rate, and mean delta SPT. Using `--no-enable-step-diagnostics` leaves optimization, research, tactical-rate, and SPS charts available, but the gameplay charts above will be absent. The flag cannot be changed after a process starts, so verify the recorded `enable_step_diagnostics` hyperparameter in W&B before leaving a long run unattended. A cadence of `--step-diagnostics-log-every 3` matches the previous Phase1-Data-023 comparison run and limits high-frequency scalar logging.

At the end of every PPO iteration, the console reports average SPS, completed/scheduled steps, percentage progress, estimated time remaining, and estimated local finish time. The ETA uses average throughput since rollout collection began, so it is approximate and normally becomes more stable after the first several iterations.

Hardware and code changes can alter the best `num_envs`. Historical repository measurements have favored both 12 and 20 under different code paths; benchmark on the target machine instead of treating one count as universally optimal.

### Outputs

Each run is named approximately:

```text
Tribes-v0__ppo__<seed>__<unix_timestamp>
```

The trainer writes:

- TensorBoard events and configuration under `runs/<run_name>/`;
- periodic `model_checkpoint_<step>.cleanrl_model` files when checkpointing is enabled;
- a final `ppo.cleanrl_model` unless `--model-path` overrides it;
- `<model>.action_interface.json` beside each saved model;
- optional videos under `videos/<run_name>/`;
- optional SPS JSON reports under `outputs/sps_profiles/`.

View TensorBoard logs with:

```powershell
tensorboard --logdir runs
```

Offline W&B runs can be uploaded later with:

```powershell
wandb sync wandb/offline-run-*
```

## Profiling and debug modes

Separate full training, behavioral debugging, and throughput profiling. Expensive step diagnostics are valuable when investigating a specific failure but reduce sustained training throughput.

For a model-quality or benchmark run, diagnostics must currently remain enabled because several required W&B gameplay charts share that logging gate. Reserve `--no-enable-step-diagnostics` for short SPS/throughput measurements where missing gameplay charts are acceptable.

### Behavioral debug run

Add:

```powershell
--enable-step-diagnostics --step-diagnostics-log-every 1
```

Use a short run (typically thousands rather than millions of steps). Debug mode records exact legal choices, selected actions, reward components, tactical misses, movement verification, and additional state summaries.

### SPS profile

```powershell
$env:POLYVISION_PROFILE_SPS='1'
$env:POLYVISION_PROFILE_WRITE_JSON='1'
$env:POLYVISION_PROFILE_EVERY_N_STEPS='1000'

python py_rl/cleanrl/cleanrl/ppo.py `
  --env-id Tribes-v0 `
  --actor-mode legal_features `
  --max-legal-actions 256 `
  --num-envs 20 `
  --total-timesteps 10240 `
  --no-track `
  --no-enable-step-diagnostics `
  --no-validate-action-interface
```

The profiler reports wall-clock contributions from environment reset/step work, Java transport, legal-action processing, feature building, reward calculation, tensor conversion, rollout collection, logging, checkpointing, and PPO optimization.

## Evaluation and interactive tools

### Inspect a trained policy

`evaluate_brain.py` loads an explicit checkpoint or the newest `.cleanrl_model` under `runs/**`:

```powershell
python evaluate_brain.py --model-path runs/<run_name>/ppo.cleanrl_model --seed 42
```

Optional interactive rendering:

```powershell
python evaluate_brain.py `
  --model-path runs/<run_name>/ppo.cleanrl_model `
  --render-java `
  --show-opening `
  --step-delay-s 0.25
```

Use `--level-pool-glob`, `--level-selection-mode`, and `--base-seed` to mirror a training map configuration.

### Play the Turn 10 wrapper manually

```powershell
python tools/play_human_t10_wrapper.py `
  --level-pool-glob 'levels/phase1_pool_bardur_solo/*.csv' `
  --max-legal-actions 256 `
  --show-ansi-map
```

Add `--render-java` for the Swing view or `--auto-random` for a non-interactive legal-action smoke run. Human traces can be saved as JSON under `outputs/human_wrapper_runs/`.

### Baselines and model comparisons

Useful entry points include:

```text
tools/eval_org_only_oracle_vs_ppo.py
py_rl/cleanrl/cleanrl/evaluate_visible_greedy_movement.py
py_rl/cleanrl/cleanrl/evaluate_no_fog_runtime_village_greedy.py
py_rl/cleanrl/cleanrl/privileged_nearest_village_oracle.py
```

The hidden-village oracle and no-fog evaluator deliberately use information unavailable to a normal agent. They are diagnostic ceilings and causal probes, not fair competitors.

### Targeted validators and audits

```text
tools/smoke_test_terminal_spt_reward.py
py_rl/cleanrl/cleanrl/legal_features_diagnostics.py
py_rl/cleanrl/cleanrl/validate_bardur_features.py
py_rl/cleanrl/cleanrl/validate_capture_features.py
py_rl/cleanrl/cleanrl/audit_capture_legality_pipeline.py
py_rl/cleanrl/cleanrl/audit_distance_zero_no_capture.py
py_rl/cleanrl/cleanrl/audit_target_contains_visible_village.py
py_rl/cleanrl/cleanrl/audit_visible_village_targets.py
```

Most scripts expose their exact arguments through `--help`.

## Configuration reference

Frequently used environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `POLYVISION_LEVEL_POOL_GLOB` | `levels/phase1_pool/*.csv` when present | Select map files. Relative globs resolve under `pol_env/Tribes/`. |
| `POLYVISION_LEVEL_SELECTION_MODE` | `round_robin` | `round_robin` or `seeded_random`. |
| `POLYVISION_BASE_SEED` | `42` | Base for wrapper-managed episode and map-selection streams. |
| `POLYVISION_SOLO_NO_OPPONENT_MODE` | `0` | Disable opponent participation in the Java bridge. Recommended as `1` for solo Phase 1 training. |
| `POLYVISION_MAX_LEGAL_ACTIONS` | `256` | Fixed sparse legal-slot capacity; normally set by PPO's matching CLI option. |
| `POLYVISION_INFO_MODE` | `fast` | Info payload mode: `fast`, `train`, or `debug`. |
| `POLYVISION_TERMINAL_SPT_REWARD_ENABLED` | `0` | Enable the Turn 10 terminal SPT bonus. |
| `POLYVISION_TERMINAL_SPT_BASE_WEIGHT` | `1.0` | Weight for final SPT. |
| `POLYVISION_TERMINAL_SPT_OVER_10_WEIGHT` | `2.0` | Weight for final SPT above 10. |
| `POLYVISION_TERMINAL_SPT_OVER_15_WEIGHT` | `3.0` | Weight for final SPT above 15. |
| `POLYVISION_RESOURCE_GATHER_UPGRADE_FILTER_ENABLED` | `0` | Restrict resource gathering using the city-upgrade relevance guard. |
| `POLYVISION_PROFILE_SPS` | `0` | Enable detailed timing instrumentation. |
| `POLYVISION_PROFILE_EVERY_N_STEPS` | `1000` | Profiling report cadence. |
| `POLYVISION_PROFILE_WRITE_JSON` | enabled by trainer default | Write the final SPS profile JSON. |
| `POLYVISION_PROFILE_OUTPUT_DIR` | `outputs/sps_profiles` | SPS profile output directory. |
| `POLYVISION_BATCH_LEGAL_ACTION_FETCH` | `0` | Use the batched Java legal-action transport path. |
| `POLYVISION_STRICT_COORD_ASSERT` | `0` | Enable strict coordinate consistency assertions. |
| `POLYVISION_VERBOSE_RESETS` | `0` | Print reset/map/legal-action summaries. |
| `POLYVISION_OPENING_GRID_DEBUG` | `0` | Print detailed opening-grid diagnostics. |
| `POLYVISION_ACTION_VALIDATION_MODE` | `0` | Enable wrapper validation behavior used by action-interface checks. |

Equivalence-check flags also exist for optimized feature, filter, legal-summary, and batched-fetch paths. They are intended for development verification and can add overhead:

```text
POLYVISION_FEATURE_EQUIV_CHECK
POLYVISION_FILTER_EQUIV_CHECK
POLYVISION_FILTER_EQUIV_CHECK_EVERY_N_STEPS
POLYVISION_LEGAL_SUMMARY_EQUIV_CHECK
POLYVISION_LEGAL_SUMMARY_EQUIV_CHECK_EVERY_N_STEPS
POLYVISION_BATCH_LEGAL_FETCH_EQUIV_CHECK
POLYVISION_BATCH_LEGAL_FETCH_EQUIV_CHECK_EVERY_N_STEPS
```

## Current limitations

- Phase 1 is a shaped early-game curriculum, not the full competitive game.
- Combat attacks and learned opponent play are outside the active task.
- The scripted opening constrains the states from which the policy begins learning.
- The wrapper contains tactical guardrails that sometimes narrow otherwise legal engine choices, especially before the second city is acquired.
- The Gym observation is a hand-built flat vector, not a learned visual representation of the commercial game.
- Each parallel environment launches a JVM; environment stepping and Python-side legal/action-feature processing remain major throughput costs.
- The fixed 256-slot legal tensor is intentionally fail-fast. It is a pragmatic increase after a 128-slot run encountered 133 legal actions, not proof that every future task will remain below 256; broader action sets may require a larger capacity and new checkpoint metadata.
- Map-generation parity with the commercial game is partial. See `MapGen.md` for the current parity inventory.
- Root smoke scripts provide integration confidence but do not constitute comprehensive automated regression coverage.
- The Java Swing renderer requires a graphical display. Headless training and image/text rendering do not.
- `docs/ENVIRONMENT_API.md`, `docs/MVP_PHASE1_SPEC.md`, and parts of `docs/TRAINING.md` describe earlier interfaces or implementation targets and should be treated as historical until refreshed. This README and the current source code describe the active interface.
- No Computer Vision or device-control path for the live commercial game is implemented yet.
- No public submission service or community leaderboard is implemented yet.

## Roadmap

### Continue Phase 1 hardening

- improve policy quality and map-to-map generalization;
- reduce environment/action-feature overhead;
- expand regression tests around legality, coordinates, observations, and rewards;
- keep model/interface metadata and evaluation tooling synchronized;
- calibrate map generation and reward shaping from measured outcomes rather than single-map behavior.

### Phase 2: full game and combat

- restore the broader military action space;
- extend episode horizons;
- introduce active opponents and multi-agent/self-play training;
- shift evaluation toward wins, score, captures, and robustness against varied policies.

### Phase 3: live-game PolyVision agent

- add Computer Vision for board, unit, city, resource, and UI-state recognition;
- map visual state into a compatible policy representation;
- add safe device/screen control;
- validate decisions against native bots before any human competition.

### Phase 4: community evaluation

- define a stable submission and model-metadata contract;
- create repeatable benchmark suites and held-out maps/seeds;
- accept model submissions;
- publish comparable results and a leaderboard.

## Documentation map

| Document | Purpose | Status |
|---|---|---|
| `CHANGELOG.md` | Detailed Phase 1 implementation and experiment history. | Most complete historical record. |
| `model_run_benchmark_log.md` | Maps run-folder names to plain-English experiment labels. | Current through Phase1-Data-023. |
| `EFFICIENT_TRAINING_RUN.md` | Training/debug/profiling operating guidance. | Useful, but older examples may use the previous 1024 legal-slot default. |
| `docs/TROUBLESHOOTING.md` | Resolutions for observed operational issues. | Current log. |
| `docs/HARDWARE_TRAINING_PROFILE.md` | Local hardware and historical throughput measurements. | Hardware-specific snapshot. |
| `MapGen.md` | Map-generation behavior and Tribes parity inventory. | Snapshot/reference. |
| `docs/PLAIN_ENGLISH_GUIDE.md` | Non-technical project orientation. | High-level background. |
| `docs/ENVIRONMENT_API.md` | Earlier bridge/wrapper API description. | Partly stale; source and this README take precedence. |
| `docs/MVP_PHASE1_SPEC.md` | Original Phase 1 implementation target. | Historical planning document. |
| `docs/TRAINING.md` | Earlier baseline commands. | Partly stale; use commands above. |
| `docs/TESTING.md` | Earlier smoke-test sequence. | Broadly useful, with older status language. |
| `pol_env/Tribes/README.md` | Tribes-specific compilation and bridge setup. | Setup reference. |

## Attribution and lineage

This repository is a fork and continuation of substantial upstream work.

- Upstream: <https://github.com/ClaireBookworm/polytopia_rl>
- PolyVision fork: <https://github.com/NeoLorenzo/PolyVision>
- The core game engine and bridge foundations come from the upstream `polytopia_rl` project and its contributors.
- RL code under `py_rl/cleanrl` includes CleanRL-derived and upstream components.

Preserve upstream licenses, attribution, and notices when modifying or redistributing the engine or RL components. PolyVision extends that foundation with the Phase 1 curriculum, map pools, fixed action catalog, legality-aware PPO actors, reward shaping, evaluation tools, diagnostics, profiling, and project roadmap described above.
