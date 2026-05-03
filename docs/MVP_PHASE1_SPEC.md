# Phase 1 MVP Specification

This document translates `polyvision_plan.md` into implementation constraints tied to actual code locations.

## Target MVP

Goal: train an economy-first agent that maximizes Stars Per Turn (SPT) by turn 10.

Required constraints:
- Tribe: Bardur
- Map type: Drylands
- Turn limit: 10 turns
- Single-player economic mode (no combat opponents)
- Reward focused on economy/SPT outcome

## Current State vs Target

| Area | Target | Current State | Primary Files |
|---|---|---|---|
| Tribe selection | Force Bardur | Not enforced globally | `pol_env/Tribes/py/gym_env.py`, Java level/init paths |
| Map type | Drylands | Sample level/default flow | `pol_env/Tribes/levels/*`, reset/init logic |
| Turn horizon | 10 turns | General episode termination from Java env | `pol_env/Tribes/py/register_env.py`, `pol_env/Tribes/py/gym_env.py` |
| Opponents/combat | Disabled for MVP | Full action set currently available | `pol_env/Tribes/py/register_env.py`, action filtering point |
| Reward | SPT-centric | Mixed score delta + relative score + step penalty | `pol_env/Tribes/py/gym_env.py` |

## Implementation Hooks

### 1) Episode boundary (10 turns)
- Add explicit truncation logic in Gym wrapper `step()` based on `info['tick']`.
- Return `truncated=True` at turn limit and end episode consistently.

### 2) Economic-only action filtering
- Filter `list_actions()` in Python before exposing valid actions to agent.
- Keep only economic action classes for MVP (move/explore, harvest, upgrade, selected research).
- Ensure action index remapping stays stable after filtering.

### 3) SPT reward redesign
- Replace/override current reward shaping in `TribesGymEnv.step()` or wrapper-level reward adapter.
- Recommended MVP reward:
  - Dense proxy each step: delta in stars income capability
  - Terminal score: SPT at turn 10 with strong terminal weighting

### 4) Deterministic benchmark configuration
- Lock seed strategy and level configuration for reproducible early experiments.
- Keep one baseline seed set for CI/smoke and one rotating set for training robustness.

## Acceptance Criteria for MVP Environment

Environment build is MVP-ready when:
1. Every episode ends at exactly 10 turns.
2. Action set exposed to policy is economic-only.
3. Reward reflects economic objective (SPT), not combat ranking.
4. At least one smoke run executes 100 episodes without runtime errors.
5. Documentation and commands in `docs/TESTING.md` and `docs/TRAINING.md` run as written.

## Non-Goals in Phase 1

- No combat optimization.
- No multi-agent adversarial training.
- No CV/ADB integration.
- No leaderboard automation.
