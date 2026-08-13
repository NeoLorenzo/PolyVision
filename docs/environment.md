# Environment

The registered Gymnasium environment is `Tribes-v0`, implemented by `TribesGymWrapper` in `pol_env/Tribes/py/register_env.py`.

## Gymnasium API

```python
import os
import gymnasium as gym
import pol_env.Tribes.py.register_env  # registers Tribes-v0

os.environ["POLYVISION_LEVEL_POOL_GLOB"] = "levels/phase1_pool_bardur_real/train/*.csv"
os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"

env = gym.make("Tribes-v0")
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(0)
env.close()
```

The example's action `0` is the stable global `END_TURN` ID. In general, select only IDs represented by `info["legal_global_ids_padded"]` where `info["legal_action_valid_mask"]` is true; see [Actions](actions.md).

## Phase 1 episode

- The policy controls tribe ID 0, Bardur.
- Reset executes a deterministic opening through the start of Turn 2: two animal harvests, a Workshop level-up, scripted exploratory movement, and the Turn 0/Turn 1 transitions.
- Policy control covers the remaining economy-first horizon through Bardur Turn 10.
- Java terminal state is ignored for this curriculum. `terminated` is always false; `truncated` becomes true after completing Bardur Turn 10, when the wrapper's counter advances to 11.
- Combat `ATTACK` actions are excluded. Capture, movement, research, resource gathering, building, training/spawning, level-up, forest actions, examine, and end-turn actions can be exposed subject to legality and Phase 1 filters.

`POLYVISION_SOLO_NO_OPPONENT_MODE=1` asks the Java bridge to omit opponent play. The current genuine corpus has a single in-bounds Bardur starter and is normally used with this setting. Without it, the wrapper can fast-forward non-Bardur turns when the loaded game has other active tribes.

## Map selection and seeds

Set `POLYVISION_LEVEL_POOL_GLOB` relative to `pol_env/Tribes/` or as an absolute glob. The default training pool is:

```text
levels/phase1_pool_bardur_real/train/*.csv
```

This is the only pool allowed for gradients. Explicit overrides select `validation`, `test`, or `human_benchmark` for their documented evaluation roles. An explicit glob that matches no maps raises an error rather than silently falling back to training, preventing an evaluation typo from producing training-pool results.

`POLYVISION_LEVEL_SELECTION_MODE` accepts `round_robin` (default) or `seeded_random`. `POLYVISION_BASE_SEED` defaults to 42 for the internal episode-seed stream. Passing a seed to `reset()` reinitializes both the episode and map-selection streams.

All maps in one environment must be square and the same size. The first loaded map fixes observation and action-catalog dimensions; a later mismatch raises `MapGeometryError`.

## Information payloads

The default `POLYVISION_INFO_MODE=fast` includes the tensors and metadata required by PPO. `train` is also accepted. `debug` adds larger diagnostic payloads such as the dense action mask, map path/ID, pool index, seed, selection mode, and detailed reward/action telemetry. Use debug mode for audits, not routine throughput runs.

Important always-available fields include legal slot IDs and mask, legal feature tensors, legal count, map geometry, observation dimension, action-space size, catalog fingerprints, interface versions, and episode-end economy metrics.

The official [human benchmark](human-benchmark.md) consumes this same fast-mode policy interface. Its readable map is reconstructed from the flattened observation and its menu is exactly `legal_global_ids_padded[legal_action_valid_mask]`. It does not consume debug info, raw Java observation state, or a renderer.

## Validation guarantees

The wrapper checks map geometry and observation shape, one-to-one action canonicalization, legal-ID uniqueness, legal-slot capacity, chosen-ID legality, and selected raw-action correspondence. The PPO trainer additionally runs a strict state-sampling validator by default and aborts if illegal-sample or fallback rates exceed configured thresholds.

## Lifecycle

Each environment owns a Py4J gateway and JVM. Call `env.close()` in `finally` blocks. PPO uses `AsyncVectorEnv` with the `spawn` context and staggers JVM startup; reduce `--num-envs` if process or memory pressure is high. A crashed Python worker can leave Java processes briefly alive; see [Troubleshooting](troubleshooting.md).
