import argparse
import os
import random
import sys

import gymnasium as gym
import numpy as np

# Ensure repo root is importable.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Ensure env registration side-effect runs.
from pol_env.Tribes.py import register_env  # noqa: F401


def run_validator(states: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make("Tribes-v0")
    try:
        obs, info = env.reset(seed=seed)
        checked = 0
        saw_spawn_warrior = False
        saw_resource_animal = False
        saw_capture_village = False
        saw_research_forestry = False
        saw_clear_forest = False
        while checked < int(states):
            if int(info.get("duplicate_global_id_collisions", 0)) != 0:
                raise RuntimeError(f"Collision detected at checked_state={checked}")
            if int(info.get("uncanonicalized_legal_actions", 0)) != 0:
                raise RuntimeError(
                    f"Uncanonicalized legal actions at checked_state={checked}: "
                    f"{info.get('uncanonicalized_repr_examples', [])}"
                )
            if int(info.get("mask_ones", -1)) != int(info.get("unique_legal_global_ids", -2)):
                raise RuntimeError(
                    f"mask_ones mismatch unique_legal_global_ids at checked_state={checked}: "
                    f"{info.get('mask_ones')} vs {info.get('unique_legal_global_ids')}"
                )

            action_mask = np.asarray(info.get("action_mask", None))
            if action_mask is None or action_mask.ndim != 1:
                raise RuntimeError(f"Invalid action mask shape at checked_state={checked}: {getattr(action_mask, 'shape', None)}")

            valid_ids = np.where(action_mask > 0)[0]
            if len(valid_ids) == 0:
                raise RuntimeError(f"No legal masked actions at checked_state={checked}")

            # Situation-specific coverage checks.
            uw = env.unwrapped
            legal_actions = uw.tribes_env.list_actions()
            for a in legal_actions:
                a_type = str(a.get("type", "")).upper()
                gid, _reason = uw._canonicalize_action_to_global_id(a, uw.tribes_env._last_obs)
                if gid is None:
                    continue
                if action_mask[int(gid)] <= 0:
                    continue
                repr_s = str(a.get("repr", "")).upper()
                if a_type in ("SPAWN", "TRAIN") and "WARRIOR" in repr_s:
                    saw_spawn_warrior = True
                if a_type == "RESOURCE_GATHERING" and "ANIMAL" in repr_s:
                    saw_resource_animal = True
                if a_type == "CAPTURE" and "VILLAGE" in repr_s:
                    saw_capture_village = True
                if a_type == "RESEARCH_TECH" and "FORESTRY" in repr_s:
                    saw_research_forestry = True
                if a_type == "CLEAR_FOREST":
                    saw_clear_forest = True

            sampled = int(np.random.choice(valid_ids))
            obs, reward, terminated, truncated, info = env.step(sampled)
            if bool(info.get("illegal_sampled_global_id", False)):
                raise RuntimeError(f"Sampled legal ID became illegal at checked_state={checked}: sampled={sampled}")
            if bool(info.get("fallback_to_end_turn", False)):
                raise RuntimeError(f"Fallback to END_TURN after legal sample at checked_state={checked}: sampled={sampled}")
            if int(info.get("selected_global_id", -1)) != sampled:
                raise RuntimeError(
                    f"Selected global ID mismatch at checked_state={checked}: "
                    f"selected={info.get('selected_global_id')} sampled={sampled}"
                )
            checked += 1

            if terminated or truncated:
                obs, info = env.reset()

        print(f"[ACTION_VALIDATOR] PASS: validated {checked} decision states")
        print(
            "[ACTION_VALIDATOR] PASS: situation coverage seen:"
            f" spawn_warrior={saw_spawn_warrior},"
            f" resource_animal={saw_resource_animal},"
            f" capture_village={saw_capture_village},"
            f" research_forestry={saw_research_forestry},"
            f" clear_forest={saw_clear_forest}"
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Strict global action-interface validator for Tribes-v0")
    parser.add_argument("--states", type=int, default=10000, help="Decision states to validate")
    parser.add_argument("--seed", type=int, default=12345, help="Validation seed")
    args = parser.parse_args()
    run_validator(args.states, args.seed)


if __name__ == "__main__":
    main()
