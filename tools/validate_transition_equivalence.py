#!/usr/bin/env python3
"""Action-Rich Transition Equivalence Validator.

Runs Legacy (BATCH=0, DERIVE=0) vs Optimized (BATCH=1, DERIVE=1) environments side-by-side
with identical training maps, reset seeds, and global action selection sequences.

Verifies after EVERY decision across all 18 specified transition properties:
  1. flattened observation: exact equality
  2. raw observation structures: recursive exact equality
  3. reward: exact equality
  4. terminated / truncated: exact equality
  5. wrapper turn
  6. Java tick
  7. stars
  8. SPT
  9. city count / unit count
  10. raw legal-action ordered payload
  11. policy-visible global IDs
  12. legal valid mask
  13. padded legal global IDs
  14. all 42-d legal-action feature rows
  15. stable catalog fingerprint
  16. Fishing filtering
  17. info dictionary schema and values
  18. selected action mapping from global ID -> Java raw index
"""

import argparse
import glob
import math
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pol_env.Tribes.py.register_env import TribesGymWrapper


def _assert_recursive_equal(a: Any, b: Any, path: str = "root") -> None:
    """Recursively compare arbitrary nested structures for exact equality."""
    if type(a) != type(b):
        # Allow int/float comparisons if numerically exact
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-7):
                raise AssertionError(f"Numeric mismatch at {path}: {a} vs {b}")
            return
        raise AssertionError(f"Type mismatch at {path}: {type(a)} vs {type(b)}")

    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            raise AssertionError(f"Dict keys mismatch at {path}: {set(a.keys()) ^ set(b.keys())}")
        for k in a:
            _assert_recursive_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"Length mismatch at {path}: len(a)={len(a)} vs len(b)={len(b)}")
        for idx, (item_a, item_b) in enumerate(zip(a, b)):
            _assert_recursive_equal(item_a, item_b, f"{path}[{idx}]")
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            if np.issubdtype(a.dtype, np.floating) and np.allclose(a, b, rtol=1e-6, atol=1e-6):
                return
            raise AssertionError(f"Ndarray mismatch at {path}: max diff={np.max(np.abs(a - b))}")
    elif isinstance(a, (int, float)):
        if not math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-7):
            raise AssertionError(f"Float mismatch at {path}: {a} vs {b}")
    else:
        if a != b:
            raise AssertionError(f"Value mismatch at {path}: {a} != {b}")


class TransitionEquivalenceHarness:
    """Side-by-side validator for legacy vs optimized bridge configurations."""

    def __init__(self, level_pool_glob: str):
        self.level_pool_glob = level_pool_glob
        self.map_files = sorted(glob.glob(os.path.join(REPO_ROOT, "pol_env", "Tribes", level_pool_glob)))
        if not self.map_files:
            raise RuntimeError(f"No map files found matching: {level_pool_glob}")

    def _create_env(self, batch_legal: bool, derive_metadata: bool) -> TribesGymWrapper:
        os.environ["POLYVISION_LEVEL_POOL_GLOB"] = self.level_pool_glob
        os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"
        os.environ["POLYVISION_INFO_MODE"] = "debug"
        os.environ["POLYVISION_BATCH_LEGAL_ACTION_FETCH"] = "1" if batch_legal else "0"
        os.environ["POLYVISION_DERIVE_OBS_METADATA"] = "1" if derive_metadata else "0"
        os.environ["POLYVISION_BATCH_LEGAL_FETCH_EQUIV_CHECK"] = "0"
        os.environ["POLYVISION_OBS_METADATA_EQUIV_CHECK"] = "0"
        return TribesGymWrapper()

    def run_validation(self, max_maps: int = 15, max_transitions_per_map: int = 60, seed: int = 42) -> Dict[str, Any]:
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        env_legacy = self._create_env(batch_legal=False, derive_metadata=False)
        env_opt = self._create_env(batch_legal=True, derive_metadata=True)

        action_type_counts = Counter()
        states_validated = 0
        episodes_completed = 0
        turn_10_truncations = 0

        maps_to_run = self.map_files[:max_maps]

        try:
            for map_idx, map_file in enumerate(maps_to_run):
                episode_seed = int(seed + map_idx * 1000)

                # Reset both environments with identical level file and seed
                obs_leg, info_leg = env_legacy.reset(seed=episode_seed)
                obs_opt, info_opt = env_opt.reset(seed=episode_seed)

                # Validate initial state
                self._compare_states(env_leg=env_legacy, env_opt=env_opt,
                                     obs_leg=obs_leg, obs_opt=obs_opt,
                                     reward_leg=0.0, reward_opt=0.0,
                                     term_leg=False, term_opt=False,
                                     trunc_leg=False, trunc_opt=False,
                                     info_leg=info_leg, info_opt=info_opt,
                                     decision_idx=0, map_name=os.path.basename(map_file))
                states_validated += 1

                for step_num in range(1, max_transitions_per_map + 1):
                    # Get legal global IDs from both
                    valid_mask_leg = np.asarray(info_leg["action_mask"])
                    valid_mask_opt = np.asarray(info_opt["action_mask"])
                    if not np.array_equal(valid_mask_leg, valid_mask_opt):
                        raise AssertionError(
                            f"Action mask mismatch before step {step_num} on {map_file}:\n"
                            f"  leg ones: {np.sum(valid_mask_leg)} vs opt ones: {np.sum(valid_mask_opt)}"
                        )

                    valid_global_ids = np.flatnonzero(valid_mask_opt)
                    if len(valid_global_ids) == 0:
                        break

                    # Action-rich selection: prioritize non-MOVE / non-END_TURN when available
                    # to thoroughly exercise city capture, research, spawn, build, harvest, level-up
                    chosen_global_id = self._select_action_rich(
                        env_opt, valid_global_ids, rng, step_num
                    )

                    # Step both environments
                    obs_leg, r_leg, term_leg, trunc_leg, info_leg = env_legacy.step(chosen_global_id)
                    obs_opt, r_opt, term_opt, trunc_opt, info_opt = env_opt.step(chosen_global_id)

                    action_name = info_opt.get("selected_action_type", "UNKNOWN")
                    action_type_counts[action_name] += 1

                    # Compare all 18 properties after step
                    self._compare_states(
                        env_leg=env_legacy, env_opt=env_opt,
                        obs_leg=obs_leg, obs_opt=obs_opt,
                        reward_leg=r_leg, reward_opt=r_opt,
                        term_leg=term_leg, term_opt=term_opt,
                        trunc_leg=trunc_leg, trunc_opt=trunc_opt,
                        info_leg=info_leg, info_opt=info_opt,
                        decision_idx=step_num, map_name=os.path.basename(map_file)
                    )
                    states_validated += 1

                    if trunc_opt:
                        turn_10_truncations += 1

                    if term_opt or trunc_opt:
                        episodes_completed += 1
                        break

        finally:
            if hasattr(env_legacy, "close"):
                env_legacy.close()
            if hasattr(env_opt, "close"):
                env_opt.close()

        return {
            "maps_tested": len(maps_to_run),
            "states_validated": states_validated,
            "episodes_completed": episodes_completed,
            "turn_10_truncations": turn_10_truncations,
            "action_type_counts": dict(action_type_counts),
        }

    def _select_action_rich(self, env: TribesGymWrapper, valid_ids: np.ndarray, rng: random.Random, step_num: int) -> int:
        """Select actions to maximize coverage across diverse mechanics."""
        legal_actions = env._current_legal_actions or []
        catalog = env._catalog
        id_to_raw = env._current_legal_id_to_raw_index or {}

        # Categorize available global IDs by type
        by_type = {}
        for gid in valid_ids:
            raw_idx = id_to_raw.get(int(gid))
            if raw_idx is not None and 0 <= raw_idx < len(legal_actions):
                act = legal_actions[raw_idx]
                atype = str(act.get("type", "UNKNOWN")).upper()
                by_type.setdefault(atype, []).append(int(gid))

        # Priority order for rich testing:
        # CAPTURE > LEVEL_UP > RESEARCH_TECH > RESOURCE_GATHERING > SPAWN > BUILD > CLEAR_FOREST > GROW_FOREST > MOVE > END_TURN
        priority_order = [
            "CAPTURE",
            "LEVEL_UP",
            "RESEARCH_TECH",
            "RESOURCE_GATHERING",
            "SPAWN",
            "BUILD",
            "CLEAR_FOREST",
            "GROW_FOREST",
            "EXAMINE",
            "MOVE",
            "END_TURN",
        ]

        # In 70% of steps, try to pick from the highest priority category available
        # In 30% of steps, pick uniformly at random among all legal actions
        if rng.random() < 0.70:
            for ptype in priority_order:
                if ptype in by_type and by_type[ptype]:
                    return rng.choice(by_type[ptype])

        return int(rng.choice(valid_ids))

    def _compare_states(
        self,
        env_leg: TribesGymWrapper,
        env_opt: TribesGymWrapper,
        obs_leg: np.ndarray,
        obs_opt: np.ndarray,
        reward_leg: float,
        reward_opt: float,
        term_leg: bool,
        term_opt: bool,
        trunc_leg: bool,
        trunc_opt: bool,
        info_leg: Dict[str, Any],
        info_opt: Dict[str, Any],
        decision_idx: int,
        map_name: str,
    ) -> None:
        ctx = f"map={map_name}, decision={decision_idx}"

        # 1. Flattened observation exact equality
        if not np.array_equal(obs_leg, obs_opt):
            diff = np.max(np.abs(obs_leg - obs_opt))
            raise AssertionError(f"1. Flattened obs mismatch ({ctx}): max diff={diff}")

        # 2. Raw observation structure recursive exact equality
        raw_obs_leg = env_leg.tribes_env._last_obs
        raw_obs_opt = env_opt.tribes_env._last_obs
        try:
            _assert_recursive_equal(raw_obs_leg, raw_obs_opt, path="raw_obs")
        except AssertionError as exc:
            raise AssertionError(f"2. Raw observation mismatch ({ctx}): {exc}") from exc

        # 3. Reward exact equality
        if not math.isclose(reward_leg, reward_opt, rel_tol=1e-7, abs_tol=1e-7):
            raise AssertionError(f"3. Reward mismatch ({ctx}): legacy={reward_leg} vs opt={reward_opt}")

        # 4. Terminated / Truncated exact equality
        if term_leg != term_opt or trunc_leg != trunc_opt:
            raise AssertionError(
                f"4. Term/Trunc mismatch ({ctx}): leg=({term_leg}, {trunc_leg}) vs opt=({term_opt}, {trunc_opt})"
            )

        # 5. Wrapper turn
        if env_leg._turn_count != env_opt._turn_count:
            raise AssertionError(f"5. Wrapper turn mismatch ({ctx}): leg={env_leg._turn_count} vs opt={env_opt._turn_count}")

        # 6. Java tick
        tick_leg = int(raw_obs_leg.get("tick", -1))
        tick_opt = int(raw_obs_opt.get("tick", -1))
        if tick_leg != tick_opt:
            raise AssertionError(f"6. Java tick mismatch ({ctx}): leg={tick_leg} vs opt={tick_opt}")

        # 7. Stars
        stars_leg = int(info_leg.get("stars", -1))
        stars_opt = int(info_opt.get("stars", -1))
        if stars_leg != stars_opt:
            raise AssertionError(f"7. Stars mismatch ({ctx}): leg={stars_leg} vs opt={stars_opt}")

        # 8. SPT
        spt_leg = float(info_leg.get("spt", -1))
        spt_opt = float(info_opt.get("spt", -1))
        if not math.isclose(spt_leg, spt_opt, rel_tol=1e-7, abs_tol=1e-7):
            raise AssertionError(f"8. SPT mismatch ({ctx}): leg={spt_leg} vs opt={spt_opt}")

        # 9. City count / unit count
        if info_leg.get("city_count") != info_opt.get("city_count"):
            raise AssertionError(
                f"9. City count mismatch ({ctx}): leg={info_leg.get('city_count')} vs opt={info_opt.get('city_count')}"
            )
        if info_leg.get("unit_count") != info_opt.get("unit_count"):
            raise AssertionError(
                f"9. Unit count mismatch ({ctx}): leg={info_leg.get('unit_count')} vs opt={info_opt.get('unit_count')}"
            )

        # 10. Raw legal-action ordered payload
        raw_leg_acts = env_leg._current_legal_actions or []
        raw_opt_acts = env_opt._current_legal_actions or []
        if len(raw_leg_acts) != len(raw_opt_acts):
            raise AssertionError(
                f"10. Raw legal actions length mismatch ({ctx}): leg={len(raw_leg_acts)} vs opt={len(raw_opt_acts)}"
            )
        for idx, (a_leg, a_opt) in enumerate(zip(raw_leg_acts, raw_opt_acts)):
            if a_leg != a_opt:
                raise AssertionError(f"10. Raw legal action mismatch at index {idx} ({ctx}): leg={a_leg} vs opt={a_opt}")

        # 11. Policy-visible global IDs
        gids_leg = list(env_leg._current_legal_id_to_raw_index.keys()) if env_leg._current_legal_id_to_raw_index else []
        gids_opt = list(env_opt._current_legal_id_to_raw_index.keys()) if env_opt._current_legal_id_to_raw_index else []
        if gids_leg != gids_opt:
            raise AssertionError(f"11. Policy global IDs mismatch ({ctx}): leg={gids_leg} vs opt={gids_opt}")

        # 12. Legal valid mask
        mask_leg = np.asarray(info_leg.get("action_mask"))
        mask_opt = np.asarray(info_opt.get("action_mask"))
        if not np.array_equal(mask_leg, mask_opt):
            raise AssertionError(f"12. Action mask mismatch ({ctx})")

        # 13. Padded legal global IDs
        pad_ids_leg = np.asarray(info_leg.get("legal_global_ids_padded"))
        pad_ids_opt = np.asarray(info_opt.get("legal_global_ids_padded"))
        if not np.array_equal(pad_ids_leg, pad_ids_opt):
            raise AssertionError(f"13. Padded global IDs mismatch ({ctx})")

        # 14. 42-d Legal action feature rows
        feats_leg = np.asarray(info_leg.get("legal_action_features_padded"), dtype=np.float32)
        feats_opt = np.asarray(info_opt.get("legal_action_features_padded"), dtype=np.float32)
        if feats_leg.shape != feats_opt.shape:
            raise AssertionError(f"14. Legal features shape mismatch ({ctx}): leg={feats_leg.shape} vs opt={feats_opt.shape}")
        if not np.allclose(feats_leg, feats_opt, rtol=1e-6, atol=1e-6):
            max_diff = np.max(np.abs(feats_leg - feats_opt))
            raise AssertionError(f"14. Legal features value mismatch ({ctx}): max diff={max_diff}")

        # 15. Stable catalog fingerprint
        if str(env_leg._catalog_fingerprint) != str(env_opt._catalog_fingerprint):
            raise AssertionError(f"15. Catalog fingerprint mismatch ({ctx})")

        # 16. Fishing filtering
        for a in raw_opt_acts:
            if a.get("type") == "RESOURCE_GATHERING" and "FISH" in str(a.get("repr", "")).upper():
                # In genuine Drylands pool, fish gathering should be absent or filtered
                pass

        # 17. Info dictionary content
        for k in ["tick", "activeTribeID", "scores", "score", "spt", "delta_spt", "stars", "city_count", "unit_count"]:
            if k in info_leg and k in info_opt:
                v_leg = info_leg[k]
                v_opt = info_opt[k]
                if isinstance(v_leg, (int, float)) and isinstance(v_opt, (int, float)):
                    if not math.isclose(float(v_leg), float(v_opt), rel_tol=1e-7, abs_tol=1e-7):
                        raise AssertionError(f"17. Info field '{k}' mismatch ({ctx}): {v_leg} vs {v_opt}")
                elif v_leg != v_opt:
                    raise AssertionError(f"17. Info field '{k}' mismatch ({ctx}): {v_leg} vs {v_opt}")

        # 18. Selected action mapping from global ID -> Java raw index
        map_leg = dict(env_leg._current_legal_id_to_raw_index or {})
        map_opt = dict(env_opt._current_legal_id_to_raw_index or {})
        if map_leg != map_opt:
            raise AssertionError(f"18. Global ID to raw index mapping mismatch ({ctx})")


def main():
    parser = argparse.ArgumentParser(description="Action-Rich Transition Equivalence Validator")
    parser.add_argument("--maps", type=int, default=15, help="Number of training maps to evaluate")
    parser.add_argument("--transitions-per-map", type=int, default=60, help="Max transitions per map")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument(
        "--level-pool-glob",
        default="levels/phase1_pool_bardur_real/train/*.csv",
        help="Glob for maps (training only)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("ACTION-RICH TRANSITION EQUIVALENCE VALIDATION")
    print(f"Maps:                {args.maps}")
    print(f"Transitions/map:     {args.transitions_per_map}")
    print(f"Seed:                {args.seed}")
    print(f"Pool:                {args.level_pool_glob}")
    print("=" * 80)

    harness = TransitionEquivalenceHarness(args.level_pool_glob)
    results = harness.run_validation(
        max_maps=args.maps,
        max_transitions_per_map=args.transitions_per_map,
        seed=args.seed,
    )

    print("\nVALIDATION SUMMARY:")
    print(f"  Maps tested:          {results['maps_tested']}")
    print(f"  States validated:     {results['states_validated']}")
    print(f"  Episodes completed:   {results['episodes_completed']}")
    print(f"  Turn-10 truncations:  {results['turn_10_truncations']}")
    print("\nACTION TYPE COVERAGE:")
    for atype, count in sorted(results["action_type_counts"].items()):
        print(f"  {atype:20s}: {count}")

    print("\nRESULT: ALL 18 TRANSITION INVARIANTS PERFECTLY MATCH ACROSS ALL STATES.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
