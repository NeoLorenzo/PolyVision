import argparse
import os
import sys
from collections import Counter
from typing import Dict, Optional, Set, Tuple

import gymnasium as gym
import numpy as np


_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import pol_env.Tribes.py.register_env as register_env  # noqa: F401
except Exception:
    pass

import py_rl.cleanrl.cleanrl.evaluate_no_fog_runtime_village_greedy as nf


def _tile_info(obs: Dict, xy: Tuple[int, int]) -> Dict:
    x, y = int(xy[0]), int(xy[1])
    board = obs.get("board", {})
    terrain = board.get("terrain", []) if isinstance(board, dict) else []
    city_id = board.get("cityID", []) if isinstance(board, dict) else []
    t_val = None
    c_val = None
    try:
        t_val = int(terrain[x][y])
    except Exception:
        t_val = None
    try:
        c_val = int(city_id[x][y])
    except Exception:
        c_val = None
    return {"xy": (x, y), "terrain": t_val, "city_id": c_val}


def _city_actor_at(obs: Dict, xy: Tuple[int, int]) -> Optional[Dict]:
    x, y = int(xy[0]), int(xy[1])
    cities = obs.get("city", {})
    if not isinstance(cities, dict):
        return None
    for cid, c in cities.items():
        if not isinstance(c, dict):
            continue
        try:
            cx = int(c.get("x", -1))
            cy = int(c.get("y", -1))
        except Exception:
            continue
        if cx == x and cy == y:
            out = dict(c)
            out["city_id"] = cid
            return out
    return None


def _capture_unit_positions(env, obs: Dict) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    uw = env.unwrapped
    legal_actions = getattr(uw, "_current_legal_actions", [])
    for a in legal_actions:
        if str(a.get("type", "")).upper() != "CAPTURE":
            continue
        uid = a.get("unit_id", None)
        if uid is None:
            uid = uw._parse_unit_id_from_action_repr(str(a.get("repr", "")))
        if uid is None:
            continue
        pos = uw._unit_position_by_id(obs, int(uid))
        if pos is not None:
            out.add((int(pos[0]), int(pos[1])))
    return out


def _audit(args):
    os.environ.setdefault("POLYVISION_LEVEL_POOL_GLOB", "levels/phase1_pool/*.csv")
    os.environ.setdefault("POLYVISION_LEVEL_SELECTION_MODE", "round_robin")

    env = gym.make("Tribes-v0")
    uw = env.unwrapped

    total_move_actions_audited = 0
    feature_true_count = 0
    target_direct_match = 0
    target_swapped_only_match = 0
    target_neither_match = 0
    mismatch_causes = Counter()

    obs_vec, info = env.reset(seed=int(args.seed))
    for ep in range(int(args.episodes)):
        if ep > 0:
            obs_vec, info = env.reset(seed=(int(args.seed) + ep) if args.seed_per_episode else None)

        done = False
        safety = 0
        while not done:
            full_obs = uw.tribes_env.get_observation(full_visibility=True)
            # Use wrapper's fixed detector exactly.
            candidates = uw._get_visible_uncaptured_village_positions(full_obs)
            capture_positions = _capture_unit_positions(env, full_obs)

            ids, valid, feats = nf._extract_legal_tensors(info)
            valid_slots = np.where(valid)[0]

            for slot in valid_slots:
                gid = int(ids[int(slot)])
                rec = nf._build_move_record(env, gid, full_obs)
                if rec is None:
                    continue
                total_move_actions_audited += 1

                feat_target = float(feats[int(slot), 5]) > 0.5
                if not feat_target:
                    continue

                feature_true_count += 1
                src = (int(rec["src"][0]), int(rec["src"][1]))
                dst = (int(rec["dst"][0]), int(rec["dst"][1]))
                dst_swapped = (int(dst[1]), int(dst[0]))
                direct = dst in candidates
                swapped = dst_swapped in candidates

                if direct:
                    target_direct_match += 1
                elif swapped:
                    target_swapped_only_match += 1
                    mismatch_causes["swapped_coordinate_match_only"] += 1
                else:
                    target_neither_match += 1
                    tile_dst = _tile_info(full_obs, dst)
                    tile_sw = _tile_info(full_obs, dst_swapped)
                    city_dst = _city_actor_at(full_obs, dst)
                    city_sw = _city_actor_at(full_obs, dst_swapped)
                    if tile_dst.get("terrain", None) == 7:
                        mismatch_causes["target_is_fog_tile"] += 1
                    elif city_dst is not None:
                        mismatch_causes["target_has_city_actor"] += 1
                    elif tile_dst.get("terrain", None) != 4:
                        mismatch_causes["target_not_village_terrain"] += 1
                    else:
                        mismatch_causes["other_non_candidate"] += 1

                    print(
                        f"mismatch ep={ep+1} turn={int(nf._as_scalar(info.get('turn_count', 0), 0))} "
                        f"slot={int(slot)} gid={gid} repr={str(rec['action'].get('repr', ''))} "
                        f"src={src} dst={dst} dst_swapped={dst_swapped} direct={direct} swapped={swapped} "
                        f"tile_dst={tile_dst} tile_swapped={tile_sw} city_dst={city_dst} city_swapped={city_sw} "
                        f"capture_on_dst={dst in capture_positions} capture_on_swapped={dst_swapped in capture_positions}"
                    )

                # Requested per-feature-true logging.
                tile_dst = _tile_info(full_obs, dst)
                tile_sw = _tile_info(full_obs, dst_swapped)
                city_dst = _city_actor_at(full_obs, dst)
                city_sw = _city_actor_at(full_obs, dst_swapped)
                print(
                    f"target_true ep={ep+1} turn={int(nf._as_scalar(info.get('turn_count', 0), 0))} "
                    f"slot={int(slot)} gid={gid} repr={str(rec['action'].get('repr', ''))} "
                    f"src={src} dst={dst} dst_swapped={dst_swapped} "
                    f"in_candidates_direct={direct} in_candidates_swapped={swapped} "
                    f"tile_dst={tile_dst} tile_swapped={tile_sw} city_dst={city_dst} city_swapped={city_sw} "
                    f"capture_on_dst={dst in capture_positions} capture_on_swapped={dst_swapped in capture_positions}"
                )

            chosen_slot, _diag = nf._choose_slot(env, info, ids, valid, feats, full_obs)
            chosen_gid = int(ids[int(chosen_slot)])
            obs_vec, _reward, terminated, truncated, info = env.step(chosen_gid)
            done = bool(terminated or truncated)
            safety += 1
            if safety > int(args.max_steps_per_episode):
                raise RuntimeError("Episode safety cap exceeded.")

        if args.progress_every > 0 and ((ep + 1) % int(args.progress_every) == 0):
            print(f"progress episodes={ep + 1}/{args.episodes}")

    env.close()

    most_common_cause = "none"
    if mismatch_causes:
        most_common_cause = mismatch_causes.most_common(1)[0][0]

    print("\n=== target_contains_visible_uncaptured_village audit summary ===")
    print(f"episodes: {int(args.episodes)}")
    print(f"total_move_actions_audited: {total_move_actions_audited}")
    print(f"feature_target_village_true_count: {feature_true_count}")
    print(f"target_matches_candidate_directly: {target_direct_match}")
    print(f"target_matches_candidate_swapped_only: {target_swapped_only_match}")
    print(f"target_matches_neither: {target_neither_match}")
    print(
        "feature_target_contains_visible_uncaptured_village_true_on_non_candidate_count: "
        f"{target_neither_match}"
    )
    print(f"most_common_mismatch_cause: {most_common_cause}")
    print(f"mismatch_cause_breakdown: {dict(mismatch_causes)}")


def main():
    parser = argparse.ArgumentParser(description="Audit target_contains_visible_uncaptured_village mismatches.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed-per-episode", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--max-steps-per-episode", type=int, default=256)
    args = parser.parse_args()
    _audit(args)


if __name__ == "__main__":
    main()
