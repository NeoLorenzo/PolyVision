#!/usr/bin/env python3
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.abspath("."))

from pol_env.Tribes.py.register_env import TribesGymWrapper


def parse_move_destination(action_repr):
    if not isinstance(action_repr, str):
        return None
    nums = re.findall(r"-?\d+", action_repr)
    if len(nums) < 3:
        return None
    try:
        return int(nums[-2]), int(nums[-1])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level-file", type=str, default=None, help="Optional override level file path used by TribesGymWrapper.")
    args = parser.parse_args()

    env = TribesGymWrapper(level_file=args.level_file)
    obs, info = env.reset(seed=42)
    print(f"reset: valid_actions={info.get('valid_actions')} turn_count={info.get('turn_count')}")

    steps = 0
    terminated = False
    truncated = False
    action_types_seen = []

    while not (terminated or truncated):
        # Prefer END_TURN when available to advance horizon quickly.
        step_action = 0
        legal_actions = env.tribes_env.list_actions()
        _, allowed_indices = env._build_action_mask_and_indices(legal_actions)
        raw_obs = getattr(env.tribes_env, "_last_obs", {})
        terrain = raw_obs.get("board", {}).get("terrain", []) if isinstance(raw_obs, dict) else []
        width = len(terrain[0]) if terrain and isinstance(terrain[0], (list, tuple)) else 0
        height = len(terrain)

        for raw_idx in allowed_indices:
            act = legal_actions[raw_idx]
            if act.get("type") != "MOVE":
                continue
            dest = parse_move_destination(str(act.get("repr", "")))
            if dest is None:
                continue
            dx, dy = dest
            if not (0 <= dx < width and 0 <= dy < height):
                raise AssertionError(
                    f"Found off-map legal MOVE action: repr={act.get('repr')} bounds=({width}x{height})"
                )

        for allowed_pos, raw_idx in enumerate(allowed_indices):
            if legal_actions[raw_idx].get("type") == "END_TURN":
                step_action = allowed_pos
                break

        obs, reward, terminated, truncated, info = env.step(step_action)
        steps += 1
        atype = info.get("selected_action_type")
        action_types_seen.append(atype)
        print(
            f"step={steps} turn_count={info.get('turn_count')} "
            f"atype={atype} reward={reward:.2f} terminated={terminated} truncated={truncated}"
        )

        if steps > 2000:
            print("ERROR: safety break hit")
            break

    env.close()

    print("\nsummary:")
    print(f"  total_steps={steps}")
    print(f"  level_file={env.level_file}")
    print(f"  final_turn_count={info.get('turn_count')}")
    print(f"  terminated={terminated}")
    print(f"  truncated={truncated}")
    print(f"  unique_action_types={sorted(set(action_types_seen))}")


if __name__ == "__main__":
    main()
