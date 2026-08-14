# PolyVision

PolyVision is a reinforcement-learning research environment for *The Battle of Polytopia*. It connects a Java implementation of the game to Python through Py4J, exposes a constrained task through Gymnasium, and trains legality-aware PPO policies with a heavily adapted CleanRL implementation.

The project does **not** currently solve the full game. Its active curriculum is an economy-first Bardur task ending after Turn 10, with combat attacks excluded from the policy interface. This narrower setting is used to study reliable action representation, exploration, city growth, research, and evaluation before expanding toward full-game play.

## Overview

PolyVision runs on genuine Polytopia initial-map data. Compressed save states are converted into canonical, schema-versioned JSON and then into validated Tribes CSV maps. Phase 1 contains 5,517 genuine 11×11 Bardur Drylands maps in a frozen identity-based split: 5,000 train, 250 validation, 250 test, and 17 human benchmark maps.

The policy interacts through stable global action IDs rather than state-local Java indices. Each state also carries the current legal IDs, a fixed-capacity validity mask, and optional semantic/economic action features. Strict validation checks that every policy-visible legal action has one collision-free global identity and that selected IDs execute the intended Java action.

## Current capabilities

- Java Tribes engine controlled from Python through Py4J
- Gymnasium `Tribes-v0` environment with deterministic reset and cleanup behavior
- 11×11 genuine Bardur corpus with a reproducible, hash-verified experimental split
- geometry-derived observations and global action catalog
- legality-aware PPO actor modes: `legal_only`, `legal_features`, and `dense_debug`
- 42-dimensional semantic/economic legal-action features
- shaped Phase 1 rewards for economy, expansion, and exploration
- strict pre-training action-interface validation with content-aware caching
- checkpoint sidecars that enforce environment/interface compatibility
- asynchronous multi-JVM training, TensorBoard/W&B telemetry, diagnostics, and SPS profiling
- canonical deterministic/sampled validation evaluation with paired policy-visible baselines and machine-readable reports
- persistent first-attempt human benchmarking with model-interface and information parity

## Architecture

```mermaid
flowchart LR
    A["Genuine Polytopia .state data"] --> B["Canonical JSON and validated CSV"]
    B --> C["Java Tribes engine"]
    C <--> D["Py4J bridge"]
    D <--> E["Gymnasium environment"]
    E --> F["Stable IDs and legal-action interface"]
    F <--> G["Adapted CleanRL PPO"]
    G --> H["Evaluation and diagnostics"]
```

The Java engine remains authoritative for game state and raw legal actions. The Python wrapper defines the research curriculum, observation, stable action interface, reward, horizon, and validation contract. See [Architecture](docs/architecture.md) for the full control flow.

## Current Phase 1 task

The policy controls Bardur after the fail-closed `v2_guaranteed_two_unit` opening: two animals, Workshop, two original-warrior moves, a required second-warrior spawn, and handoff at Turn 2. Turn-1 opening selection alone excludes the capital; ordinary policy gameplay is unchanged. The corrected audit passed 5,517/5,517 maps. Historical Seed-1 used `v1_mixed_capital_regression` (55.07% two-unit) and is preserved but shelved before pristine capability test. See the [scripted-opening audit](docs/results/Phase1_Scripted_Opening_Audit.md) and [reflection](docs/results/Phase1_Seed1_Mixed_Opening_Validation_Reflection.md).

`ATTACK` is deliberately unavailable to the Phase 1 policy. The environment prioritizes early village-expansion lines through active action filters and trains on shaped SPT, city capture, village discovery/progress, and fog-clearing signals. Final T10 SPT and other economy metrics are reported separately from shaped return.

On the active maps, the observation has 505 values and the global action space has 63,913 IDs. Only a small legal subset is selectable in a state; by default it is represented in 256 slots. See [Actions](docs/actions.md), [Observations](docs/observations.md), and [Rewards](docs/rewards.md) for the exact current contract.

## Quick start

Requirements are Python 3.10+, a JDK with `java` and `javac`, and PowerShell for the commands below.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt

Set-Location pol_env/Tribes
$sources = Get-ChildItem -Path src -Recurse -Filter *.java | ForEach-Object FullName
javac -cp "lib/json.jar" -d out -sourcepath src $sources
Set-Location ../..
```

Verify the frozen split and live training-pool contract:

```powershell
python tools/split_phase1_map_pool.py
$env:POLYVISION_LEVEL_POOL_GLOB = 'levels/phase1_pool_bardur_real/train/*.csv'
$env:POLYVISION_SOLO_NO_OPPONENT_MODE = '1'
python tools/validate_environment_contract.py --expected-width 11 --expected-height 11
```

The wrapper defaults to the training pool only; the explicit setting above makes the experimental role visible in commands and run records. Static split verification checks all 5,517 hashes and assignments, while the live validator resets the selected pool and checks geometry, observation, catalog, and interface consistency. For detailed installation notes and a smaller smoke test, use [Getting started](docs/getting-started.md).

## Training

The primary PPO entrypoint is:

```text
py_rl/cleanrl/cleanrl/ppo.py
```

A representative tracked-interface run is:

```powershell
$env:POLYVISION_LEVEL_POOL_GLOB = 'levels/phase1_pool_bardur_real/train/*.csv'
$env:POLYVISION_SOLO_NO_OPPONENT_MODE = '1'
$env:POLYVISION_INFO_MODE = 'fast'
python py_rl/cleanrl/cleanrl/ppo.py `
    --actor-mode legal_features `
    --total-timesteps 500000 `
    --num-envs 12 `
    --num-steps 128 `
    --save-model `
    --save-frequency 100000
```

Strict 10,000-state interface validation runs before training by default and is cached by code, configuration, and map-pool identity. TensorBoard output and models are written under `runs/`; W&B is optional with `--track`. Each model is accompanied by an action-interface JSON sidecar required by current evaluators. See [Training](docs/training.md) before launching long runs.

## Evaluation and evidence

Use `evaluate_brain.py` for compatible one-episode policy inspection:

```powershell
python evaluate_brain.py `
    --model-path runs/<run>/ppo.cleanrl_model `
    --level-pool-glob 'levels/phase1_pool_bardur_real/validation/*.csv' `
    --seed 42
```

The repository also includes visible-information scripted baselines, privileged diagnostics, action/feature audits, and a pool contract validator. These tools have different visibility and protocol assumptions; [Evaluation](docs/evaluation.md) identifies their intended use.

Run the canonical 3,000-episode Phase 1 validation suite with:

```powershell
python tools/evaluate_phase1.py `
    --model-path runs/<run>/ppo.cleanrl_model `
    --pool validation `
    --suite full `
    --repeats-per-map 5 `
    --seed 42
```

This compares deterministic PPO, sampled PPO, visible greedy, and random legal on the same 250 validation maps. Final T10 SPT is primary; stochastic uncertainty is computed across per-map replicate means. Results and provenance are retained under `outputs/evaluations/`. Test evaluation is separately guarded and is not part of ordinary development evaluation.

Run a permanent human challenge attempt with:

```powershell
python tools/human_benchmark.py
```

The command selects an unplayed human-benchmark map and presents exactly the wrapper-filtered stable global IDs available to PPO. Results remain separate from pristine test evidence; see [Human benchmark](docs/human-benchmark.md).

The first complete canonical validation suite has now evaluated the first properly trained Phase 1 model over 3,000 episodes and all 250 validation maps. PPO argmax reached mean T10 SPT 14.576 and beat the visible-greedy baseline on every paired map. See [PolyVision Phase 1 Validation Results](docs/results/PolyVision_Phase1_Validation_Results.md) for the full reflection, uncertainty, failure cases, and limitations.

This was the first long Phase 1 scientific model, but it is now **historical / shelved**, not the final candidate. Its validation informs development under the historical mixed-opening task; a fresh v2 model and new validation are required before pristine capability test.

## Repository structure

| Path | Purpose |
|---|---|
| `pol_env/Tribes/src/` | Java Tribes engine and Py4J-facing environment |
| `pol_env/Tribes/py/` | Bridge, Gymnasium wrapper, validators, and utilities |
| `pol_env/Tribes/levels/phase1_pool_bardur_real/` | Frozen train/validation/test/human Phase 1 split and manifest |
| `py_rl/cleanrl/cleanrl/ppo.py` | Primary PolyVision PPO trainer |
| `tools/polytopia_state_converter/` | Compressed save to canonical JSON |
| `tools/polytopia_map_converter/` | Canonical JSON to validated Tribes CSV |
| `tools/evaluate_phase1.py` | Canonical Phase 1 validation evaluator |
| `tools/` | Additional evaluation and environment-contract utilities |
| `docs/` | Current project documentation |

## Documentation

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Environment](docs/environment.md)
- [Actions](docs/actions.md)
- [Observations](docs/observations.md)
- [Rewards](docs/rewards.md)
- [Training](docs/training.md)
- [Evaluation](docs/evaluation.md)
- [First Phase 1 validation results](docs/results/PolyVision_Phase1_Validation_Results.md)
- [Phase 1 scripted-opening audit](docs/results/Phase1_Scripted_Opening_Audit.md)
- [Human benchmark](docs/human-benchmark.md)
- [Maps](docs/maps.md)
- [Reproducibility](docs/reproducibility.md)
- [Troubleshooting](docs/troubleshooting.md)

Focused manuals remain beside the [state converter](tools/polytopia_state_converter/README.md), [map converter](tools/polytopia_map_converter/README.md), and optional [map harvester](tools/polytopia_harvester/README.md). `CHANGELOG.md` records implementation history; [historical benchmark evidence](docs/history/model-run-benchmark-log.md) is explicitly non-normative.

## Current limitations

- The task ends after Turn 10 and does not represent full-game strategy.
- Combat attacks are excluded from policy control.
- The active corpus covers one tribe and one 11×11 Drylands distribution.
- The observation is a compact engineered vector, not the complete game state.
- Reward and action filtering encode a Phase 1 curriculum and can bias learned behavior.
- Checkpoints are tightly coupled to geometry and action-interface metadata.
- The trainer saves weights and interface metadata, not full optimizer/RNG state for exact resume.
- Current-interface multi-training-seed validation/test benchmark results are not yet established.

## Roadmap

Near-term work is to run reproducible multi-seed validation and pristine-test baselines on the frozen split. Later phases can broaden map distributions and tribes, relax curriculum filters, add combat and opponents, extend observations and action semantics, and evaluate full-game policies. Those items are future work, not current capabilities.

## Attribution and licenses

PolyVision is a fork and continuation of [ClaireBookworm/polytopia_rl](https://github.com/ClaireBookworm/polytopia_rl) and retains substantial upstream Tribes engine and bridge work. RL code under `py_rl/cleanrl` includes CleanRL-derived components; its license is in [`py_rl/cleanrl/LICENSE`](py_rl/cleanrl/LICENSE). The state converter vendors separately licensed dependencies with their notices. Preserve upstream attribution and applicable licenses when redistributing the project.

*The Battle of Polytopia* is the intellectual property of its respective owner. PolyVision is a research project and does not claim affiliation or endorsement.
