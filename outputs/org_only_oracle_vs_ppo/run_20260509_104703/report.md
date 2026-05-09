# Org-only Oracle vs Latest PPO

- Timestamp: 2026-05-09T10:47:03.063273Z
- Episodes: 500
- PPO checkpoint: `C:\PolyVision\runs\Tribes-v0__ppo__1__1778266653\ppo.cleanrl_model`
- PPO deterministic: `True`

## Verdict
- PPO beats oracle baseline on mean final SPT

## Summary

| Metric | PPO | Oracle (Org-only) |
|---|---:|---:|
| Mean final SPT | 10.518 | 8.790 |
| Mean city count | 3.316 | 2.654 |
| Mean turn second city captured | 5.366 | 5.138 |
| Organization research rate | 0.00% | 76.80% |
| Forestry research rate | 0.00% | 0.00% |
| Mean fruit harvested | 1.666 | 2.120 |
| Mean fog tiles cleared | 53.274 | 37.058 |
