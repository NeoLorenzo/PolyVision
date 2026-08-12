#!/usr/bin/env python3
"""Reset-only contract smoke test for a dimension-homogeneous Tribes map pool."""

import argparse
import os
import sys
from collections import Counter


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pol_env.Tribes.py.register_env import TribesGymWrapper


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level-pool-glob",
        default="levels/phase1_pool_bardur_real/*.csv",
        help="Pool glob relative to pol_env/Tribes, or an absolute glob.",
    )
    parser.add_argument("--expected-width", type=int, required=True)
    parser.add_argument("--expected-height", type=int, required=True)
    args = parser.parse_args()

    os.environ["POLYVISION_LEVEL_POOL_GLOB"] = args.level_pool_glob
    os.environ["POLYVISION_LEVEL_SELECTION_MODE"] = "round_robin"
    os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"
    os.environ["POLYVISION_INFO_MODE"] = "fast"

    env = TribesGymWrapper()
    pool_size = len(env._level_pool)
    expected_obs_dim = 4 * args.expected_width * args.expected_height + 21
    fingerprints = Counter()
    action_sizes = Counter()
    legal_counts = []
    try:
        for index in range(pool_size):
            obs, info = env.reset(seed=0 if index == 0 else None)
            actual = env._board_dimensions_from_obs(env.tribes_env._last_obs)
            if actual != (args.expected_width, args.expected_height):
                raise RuntimeError(f"map {index}: actual geometry {actual}")
            if tuple(obs.shape) != (expected_obs_dim,):
                raise RuntimeError(f"map {index}: observation shape {obs.shape}")
            if not env.observation_space.contains(obs):
                raise RuntimeError(f"map {index}: observation is outside declared space")
            if (env._catalog.width, env._catalog.height) != actual:
                raise RuntimeError(f"map {index}: catalog geometry disagrees with board")
            reported = (int(info["map_width"]), int(info["map_height"]))
            if reported != actual:
                raise RuntimeError(f"map {index}: reset metadata geometry {reported} disagrees with board {actual}")
            if int(info["observation_dim"]) != expected_obs_dim:
                raise RuntimeError(f"map {index}: reset metadata observation_dim={info['observation_dim']}")
            if int(info["action_space_n"]) != int(env.action_space.n):
                raise RuntimeError(f"map {index}: reset metadata action_space_n={info['action_space_n']}")
            if str(info["action_catalog_fingerprint"]) != str(env._catalog_fingerprint):
                raise RuntimeError(f"map {index}: reset metadata catalog fingerprint mismatch")
            fingerprints[str(env._catalog_fingerprint)] += 1
            action_sizes[int(env.action_space.n)] += 1
            legal_counts.append(int(info["legal_action_count"]))
    finally:
        close = getattr(env.tribes_env, "close", None)
        if callable(close):
            close()

    if len(fingerprints) != 1 or len(action_sizes) != 1:
        raise RuntimeError(
            f"Pool contract varied: fingerprints={dict(fingerprints)}, action_sizes={dict(action_sizes)}"
        )
    print(f"Maps tested:          {pool_size}")
    print(f"Geometry:             {args.expected_width}x{args.expected_height}")
    print(f"Observation shape:    ({expected_obs_dim},)")
    print(f"Action space:         {next(iter(action_sizes))}")
    print(f"Catalog fingerprint:  {next(iter(fingerprints))}")
    print(f"Legal actions/reset:  {min(legal_counts)}..{max(legal_counts)}")
    print("Contract failures:    0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
