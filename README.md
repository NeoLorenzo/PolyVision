# PolyVision

PolyVision is a reinforcement-learning research environment for *The Battle of Polytopia*. It connects a Java implementation of the game to Python through Py4J, exposes a constrained task through Gymnasium, and trains legality-aware PPO policies with a heavily adapted CleanRL implementation.

The project does **not** currently solve the full game. Its active curriculum is an economy-first Bardur task ending after Turn 10, with combat attacks excluded from the policy interface. This narrower setting is used to study reliable action representation, exploration, city growth, research, and evaluation before expanding toward full-game play.

## Overview

PolyVision runs on genuine Polytopia initial-map data. Compressed save states are converted into canonical, schema-versioned JSON and then into validated Tribes CSV maps. The current committed pool contains 256 genuine 11×11 Bardur Drylands maps.

The policy interacts through stable global action IDs rather than state-local Java indices. Each state also carries the current legal IDs, a fixed-capacity validity mask, and optional semantic/economic action features. Strict validation checks that every policy-visible legal action has one collision-free global identity and that selected IDs execute the intended Java action.

## Current capabilities

- Java Tribes engine controlled from Python through Py4J
- Gymnasium `Tribes-v0` environment with deterministic reset and cleanup behavior
- 11×11 genuine Bardur map corpus and reproducible map conversion pipeline
- geometry-derived observations and global action catalog
- legality-aware PPO actor modes: `legal_only`, `legal_features`, and `dense_debug`
- 42-dimensional semantic/economic legal-action features
- shaped Phase 1 rewards for economy, expansion, and exploration
- strict pre-training action-interface validation with content-aware caching
- checkpoint sidecars that enforce environment/interface compatibility
- asynchronous multi-JVM training, TensorBoard/W&B telemetry, diagnostics, and SPS profiling
- checkpoint introspection, scripted baselines, feature audits, and controlled-evaluation infrastructure

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

The policy controls Bardur after a deterministic opening that harvests the starting animals, takes the Workshop upgrade, and advances through the start of Turn 2. It then chooses economy, research, movement, capture, development, and turn-ending actions through Bardur Turn 10.

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

Select and validate the genuine pool:

```powershell
$env:POLYVISION_LEVEL_POOL_GLOB = 'levels/phase1_pool_bardur_real/*.csv'
$env:POLYVISION_SOLO_NO_OPPONENT_MODE = '1'
python tools/validate_environment_contract.py --expected-width 11 --expected-height 11
```

The explicit pool setting is currently required because the wrapper's internal fallback still names removed legacy levels. The validator resets all 256 maps and checks geometry, observation, catalog, and pool-wide interface consistency. For detailed installation notes and a smaller smoke test, use [Getting started](docs/getting-started.md).

## Training

The primary PPO entrypoint is:

```text
py_rl/cleanrl/cleanrl/ppo.py
```

A representative tracked-interface run is:

```powershell
$env:POLYVISION_LEVEL_POOL_GLOB = 'levels/phase1_pool_bardur_real/*.csv'
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
    --level-pool-glob 'levels/phase1_pool_bardur_real/*.csv' `
    --seed 42
```

The repository also includes visible-information scripted baselines, privileged diagnostics, action/feature audits, and a pool contract validator. These tools have different visibility and protocol assumptions; [Evaluation](docs/evaluation.md) identifies their intended use.

The latest committed experiment evidence includes a completed 500K training validation on the current genuine-map, 256-slot, 42-feature interface. Its recorded metrics are training-summary snapshots, not a held-out multi-seed benchmark. The strongest committed repeated-episode comparison used an older interface and is preserved as historical evidence only. Definitive current-interface, held-out, multi-seed results remain an active research target.

## Repository structure

| Path | Purpose |
|---|---|
| `pol_env/Tribes/src/` | Java Tribes engine and Py4J-facing environment |
| `pol_env/Tribes/py/` | Bridge, Gymnasium wrapper, validators, and utilities |
| `pol_env/Tribes/levels/phase1_pool_bardur_real/` | Active genuine 11×11 Bardur maps |
| `py_rl/cleanrl/cleanrl/ppo.py` | Primary PolyVision PPO trainer |
| `tools/polytopia_state_converter/` | Compressed save to canonical JSON |
| `tools/polytopia_map_converter/` | Canonical JSON to validated Tribes CSV |
| `tools/` | Evaluation and environment-contract utilities |
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
- Current-interface held-out multi-seed benchmark results are not yet established.

## Roadmap

Near-term work is to harden the genuine-map interface, freeze train/held-out manifests, and publish reproducible multi-seed baselines. Later phases can broaden map distributions and tribes, relax curriculum filters, add combat and opponents, extend observations and action semantics, and evaluate full-game policies. Those items are future work, not current capabilities.

## Attribution and licenses

PolyVision is a fork and continuation of [ClaireBookworm/polytopia_rl](https://github.com/ClaireBookworm/polytopia_rl) and retains substantial upstream Tribes engine and bridge work. RL code under `py_rl/cleanrl` includes CleanRL-derived components; its license is in [`py_rl/cleanrl/LICENSE`](py_rl/cleanrl/LICENSE). The state converter vendors separately licensed dependencies with their notices. Preserve upstream attribution and applicable licenses when redistributing the project.

*The Battle of Polytopia* is the intellectual property of its respective owner. PolyVision is a research project and does not claim affiliation or endorsement.
