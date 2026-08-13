# Tribes engine integration

This directory contains the Java Tribes engine, assets, levels, and Python/Py4J bridge used by PolyVision. Current project setup and environment instructions live in the root [README](../../README.md) and [getting-started guide](../../docs/getting-started.md).

## Compile

From this directory in PowerShell:

```powershell
$sources = Get-ChildItem -Path src -Recurse -Filter *.java | ForEach-Object FullName
javac -cp "lib/json.jar" -d out -sourcepath src $sources
```

Compilation should create `out/core/game/PythonEnv.class`. Python launches the JVM with `out` and `lib/json.jar`; a manual `CLASSPATH` is not normally needed.

The frozen runtime dataset is under `levels/phase1_pool_bardur_real/`; the wrapper defaults to its `train/` pool only. See [Maps](../../docs/maps.md) for split semantics, provenance, and validation.

## Attribution

This integration builds on major upstream work from [ClaireBookworm/polytopia_rl](https://github.com/ClaireBookworm/polytopia_rl). Preserve upstream credit and applicable notices when modifying or redistributing it.
