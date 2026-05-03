# Testing Guide

This guide defines the practical test flow for the current codebase.

## Prerequisites

1. Compile Java classes:
```bash
cd pol_env/Tribes
mkdir -p out
find src -name "*.java" -exec javac -cp "lib/json.jar" -d out -sourcepath src {} +
```

2. Activate Python environment and install dependencies:
```bash
cd /workspaces/PolyVision
source .venv/bin/activate
pip install -r pol_env/Tribes/py/requirements.txt
pip install -r py_rl/requirements.txt
```

## Smoke Tests (Run in this order)

### 1) Bridge + environment sanity
```bash
cd /workspaces/PolyVision
python test_simple.py
```
Expected:
- wrapper constructs
- reset succeeds
- several random steps execute

### 2) Gymnasium registration sanity
```bash
cd /workspaces/PolyVision
python test_gym.py
```
Expected:
- `gym.make("Tribes-v0")` succeeds
- one step succeeds

### 3) End-to-end runner sanity
```bash
cd /workspaces/PolyVision/pol_env/Tribes/py
python run_gym.py
```
Expected:
- prints `reset_ok` and `n_actions`
- runs multiple steps
- in headless terminals, Java GUI warning may appear while run continues

## Additional Local Checks

```bash
cd /workspaces/PolyVision
python test_env.py
```
This is a broader wrapper behavior check with richer printed diagnostics.

## Common Failure Modes

- `No module named py4j` or `gymnasium`:
  - install `pol_env/Tribes/py/requirements.txt`
- `No module named torch` in training/eval scripts:
  - install `py_rl/requirements.txt`
- Java class/path failures (`PythonEnv` not found):
  - Java `out/` classes were not compiled
- `HeadlessException` for GUI:
  - expected on terminal-only systems without DISPLAY

## Current Test Positioning

- The scripts above are practical smoke tests, not a strict CI-grade unit suite.
- `py_rl/cleanrl/tests` belongs to upstream CleanRL-style coverage and is not the first-line check for Polyvision integration.
