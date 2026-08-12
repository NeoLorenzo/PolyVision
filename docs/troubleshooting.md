# Troubleshooting

## The environment cannot find a level file

The current runtime wrapper still has legacy fallback constants for removed levels. Set the active pool before constructing `Tribes-v0`:

```powershell
$env:POLYVISION_LEVEL_POOL_GLOB = 'levels/phase1_pool_bardur_real/*.csv'
$env:POLYVISION_SOLO_NO_OPPONENT_MODE = '1'
```

The glob is resolved relative to `pol_env/Tribes/` unless absolute.

## Py4J reports missing Java classes

Confirm `java -version` and `javac -version` both work; a JRE without `javac` is insufficient. Recompile from `pol_env/Tribes`:

```powershell
$sources = Get-ChildItem -Path src -Recurse -Filter *.java | ForEach-Object FullName
javac -cp "lib/json.jar" -d out -sourcepath src $sources
Test-Path out/core/game/PythonEnv.class
```

The final command should print `True`. The Python bridge constructs its classpath from `out` and `lib/json.jar`; no manual `CLASSPATH` is normally needed.

Cleanup commit `6814978` briefly removed `core.levelgen.LevelGenerator` while `GameState.java` still depended on it. `Phase1-CI_Implimentation-029` restored the generator and CLI exactly. If a checkout at the cleanup commit fails with missing `LevelGenerator` symbols, update to this revision or later rather than relying on a pre-existing `out/core/game/PythonEnv.class`.

## JVMs linger or parallel startup fails

Always call `env.close()`, ideally in `finally`. PPO creates one JVM per vector worker and staggers startup. Reduce `--num-envs` if Windows process creation, memory, or handle limits are stressed. After an abnormal trainer exit, inspect Java processes before starting a large replacement run.

## PowerShell will not activate the environment

If execution policy blocks `.\.venv\Scripts\Activate.ps1`, use the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe tools/validate_environment_contract.py --expected-width 11 --expected-height 11
```

PowerShell uses the backtick for line continuation. Commands copied into `cmd.exe` or Bash need different continuation and environment-variable syntax.

## Map geometry mismatch

All files selected by one pool glob must be square and have identical dimensions. The current corpus is 11×11. A malformed rectangle fails before Java loading; a later map that differs from the first map fails the fixed-geometry contract. Check that a manifest or unrelated CSV was not placed inside the level directory.

## Observation or action dimensions differ

On the current corpus, expected values are observation shape `(505,)`, action space `63913`, legal capacity `256`, and action-feature dimension `42`. A fallback warning followed by placeholder dimensions means initialization failed earlier; fix the preceding JVM/map error rather than training with the placeholders.

## Legal-action capacity exceeded

The wrapper raises if the current legal set exceeds `POLYVISION_MAX_LEGAL_ACTIONS`. Do not increase only the evaluator or trainer value: the environment variable, trainer `--max-legal-actions`, and checkpoint metadata must agree. Treat an overflow as an interface audit event and validate the chosen capacity over representative states.

## Strict action validation fails

Read the first reported invariant: uncanonicalized actions, duplicate IDs, mask disagreement, fallback, and sampled-ID mismatch indicate different causes. Re-run with `--force-revalidate-action-interface`; use `POLYVISION_INFO_MODE=debug` for richer payloads. Do not disable validation for a research run to bypass an unexplained failure.

## Checkpoint is incompatible

Keep `<model>.action_interface.json` beside the model. Current loaders intentionally reject absent metadata and differences in geometry, dimensions, actor mode, catalog fingerprint/version, feature contract, or legal capacity. Use a checkpoint produced by the same environment contract; tensor shapes alone are not sufficient evidence of compatibility.

## W&B charts or throughput look wrong

TensorBoard output is always written under `runs/`. W&B is optional and uses `global_step` as its metric axis. `--enable-step-diagnostics`, `POLYVISION_INFO_MODE=debug`, action-feature construction, and large numbers of JVMs can materially affect SPS. Compare throughput only with the same interface and diagnostic settings.
