# Getting started

This guide gets the current 11×11 genuine-map environment running from the repository root. PolyVision launches one Java Virtual Machine (JVM) per environment process through Py4J; no separately managed gateway is required.

## Prerequisites

- Python 3.10 or newer
- A JDK with `java` and `javac` on `PATH`
- PowerShell for the commands below; Bash equivalents are straightforward
- A CUDA-capable PyTorch installation only if GPU training is required

The committed `pol_env/Tribes/lib/json.jar` is part of the Java classpath. The Python environment needs Gymnasium, Py4J, NumPy, PyTorch, Tyro, and TensorBoard. W&B is optional.

## Install and compile

Create and activate a virtual environment, then install the pinned project environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Compile the Tribes engine:

```powershell
Set-Location pol_env/Tribes
$sources = Get-ChildItem -Path src -Recurse -Filter *.java | ForEach-Object FullName
javac -cp "lib/json.jar" -d out -sourcepath src $sources
Set-Location ../..
```

Successful compilation creates `pol_env/Tribes/out/core/game/PythonEnv.class`.

## Validate the current map contract

The runtime wrapper still contains a legacy fallback path that names removed levels. Until that implementation default is corrected, select the genuine pool explicitly in every shell that creates `Tribes-v0`:

```powershell
$env:POLYVISION_LEVEL_POOL_GLOB = 'levels/phase1_pool_bardur_real/*.csv'
$env:POLYVISION_SOLO_NO_OPPONENT_MODE = '1'
python tools/validate_environment_contract.py --expected-width 11 --expected-height 11
```

This resets every map and verifies the pool-wide geometry, observation shape, action-space size, and catalog fingerprint. The current committed corpus contains 256 maps.

For a smaller bridge smoke test:

```powershell
python -c "from pol_env.Tribes.py.register_env import TribesGymWrapper; e=TribesGymWrapper(); o,i=e.reset(seed=42); print(o.shape, e.action_space.n, i['legal_action_count']); e.close()"
```

Always call `close()` when directly constructing an environment. Training and validation tools do this automatically.

## Start a short PPO run

The trainer performs strict action-interface validation by default. A full first validation covers 10,000 decision states; for a quick installation check, use a small explicit validation budget:

```powershell
$env:POLYVISION_LEVEL_POOL_GLOB = 'levels/phase1_pool_bardur_real/*.csv'
$env:POLYVISION_SOLO_NO_OPPONENT_MODE = '1'
python py_rl/cleanrl/cleanrl/ppo.py --total-timesteps 512 --num-envs 1 --num-steps 128 --validation-states 100 --actor-mode legal-only --no-cuda
```

Use the default validation budget for research runs. See [Training](training.md) for actor modes, checkpoints, tracking, and profiling.

## Next steps

- [Architecture](architecture.md) explains the Java-to-PPO control flow.
- [Environment](environment.md), [Actions](actions.md), and [Observations](observations.md) define the active RL contract.
- [Maps](maps.md) describes genuine map ingestion and validation.
- [Troubleshooting](troubleshooting.md) covers JVM, Windows, geometry, and compatibility failures.
