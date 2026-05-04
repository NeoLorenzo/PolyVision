# Training Guide

This guide documents the current baseline training/evaluation paths that exist in this repository.

## Prerequisites

1. Java compiled (`pol_env/Tribes/out` exists)
2. Python env active
3. Installed dependencies:
```bash
pip install -r pol_env/Tribes/py/requirements.txt
pip install -r py_rl/requirements.txt
```

Optional but recommended after setup:
```bash
python -m pip freeze > requirements-lock.txt
```

## Baseline PPO Run (Current Default)

Run from repo root:
```bash
cd /workspaces/PolyVision
python py_rl/cleanrl/cleanrl/ppo.py --total-timesteps 5000 --num-steps 32 --track False
```

Notes:
- Uses `Tribes-v0` env id by default in this forked script.
- Writes TensorBoard logs under `runs/`.

## Preflight Sanity Check (Recommended Before MVP Runs)

Run a short async smoke train to confirm environment, bridge, and multiprocessing health:

```bash
python py_rl/cleanrl/cleanrl/ppo.py --total-timesteps 6144 --num-steps 64 --no-track --no-capture-video --startup-jitter-min-s 0.1 --startup-jitter-max-s 2.0
```

Expected result:
- completes with exit code `0`
- prints increasing `SPS` values
- no lingering Java worker processes after completion

## Custom Polyvision Variants

### Semantic PPO
```bash
cd /workspaces/PolyVision
python py_rl/cleanrl/our_cleanrl/ppo_semantic.py --total-timesteps 5000 --num-steps 32 --track False
```

### Action-Quality PPO
```bash
cd /workspaces/PolyVision
python py_rl/cleanrl/our_cleanrl/ppo_action_quality.py --total-timesteps 5000 --num-steps 32 --track False
```

## Evaluation Scripts

### Quick comparison
```bash
cd /workspaces/PolyVision
python quick_eval.py
```

### Extended comparison harness
```bash
cd /workspaces/PolyVision
python evaluate_models.py
```

## Important Caveats

- `quick_eval.py` references semantic/action-quality script paths under `py_rl/cleanrl/cleanrl/`.
- In this repo, those two scripts currently live under `py_rl/cleanrl/our_cleanrl/`.
- If quick_eval fails for those models, run the direct commands above.

## Reproducibility Recommendations

For experiments you want to compare:
- set explicit `--seed`
- keep `--num-envs`, `--num-steps`, and `--total-timesteps` fixed
- record command + commit hash + config in experiment notes
- keep `requirements-lock.txt` updated when dependency versions change

## MVP Alignment Reminder

Current training scripts are general-purpose and not yet locked to Phase 1 economic-only constraints.
Before large runs, align environment behavior with `docs/MVP_PHASE1_SPEC.md`.
