import argparse
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

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


def _safe_int(v, default=-1):
    try:
        return int(v)
    except Exception:
        return default


def _enemy_capital_coords(obs: Dict) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    tribes = obs.get("tribes", {})
    cities = obs.get("city", {})
    if not isinstance(tribes, dict) or not isinstance(cities, dict):
        return out
    for tid_s, t in tribes.items():
        if not isinstance(t, dict):
            continue
        tid = _safe_int(tid_s, default=-1)
        if tid < 0 or tid == 0:
            continue
        cap_id = _safe_int(t.get("capitalID", -1), default=-1)
        if cap_id < 0:
            continue
        c = cities.get(str(cap_id), None)
        if not isinstance(c, dict):
            continue
        cx = _safe_int(c.get("x", -1), default=-1)
        cy = _safe_int(c.get("y", -1), default=-1)
        if cx >= 0 and cy >= 0:
            out.add((cx, cy))
    return out


def _city_owner_by_coord(obs: Dict) -> Dict[Tuple[int, int], int]:
    out: Dict[Tuple[int, int], int] = {}
    cities = obs.get("city", {})
    if not isinstance(cities, dict):
        return out
    for city in cities.values():
        if not isinstance(city, dict):
            continue
        cx = _safe_int(city.get("x", -1), default=-1)
        cy = _safe_int(city.get("y", -1), default=-1)
        owner = _safe_int(city.get("tribeID", -1), default=-1)
        if cx >= 0 and cy >= 0:
            out[(cx, cy)] = owner
    return out


def _enemy_unit_coords(obs: Dict) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    units = obs.get("unit", {})
    if not isinstance(units, dict):
        return out
    for u in units.values():
        if not isinstance(u, dict):
            continue
        tribe_id = _safe_int(u.get("tribeId", -1), default=-1)
        if tribe_id < 0 or tribe_id == 0:
            continue
        x = _safe_int(u.get("x", -1), default=-1)
        y = _safe_int(u.get("y", -1), default=-1)
        if x >= 0 and y >= 0:
            out.add((x, y))
    return out


def _owned_unit_coords(obs: Dict) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for _uid, x, y in nf._owned_units(obs):
        out.add((int(x), int(y)))
    return out


def _capture_unit_coords(env, obs: Dict) -> Set[Tuple[int, int]]:
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


def _legal_move_targets(env, obs: Dict, legal_slots: np.ndarray, ids: np.ndarray) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for s in legal_slots:
        gid = int(ids[int(s)])
        rec = nf._build_move_record(env, gid, obs)
        if rec is None:
            continue
        out.add((int(rec["dst"][0]), int(rec["dst"][1])))
    return out


def _terrain_at(obs: Dict, x: int, y: int) -> int:
    board = obs.get("board", {})
    terrain = board.get("terrain", []) if isinstance(board, dict) else []
    try:
        return int(terrain[int(x)][int(y)])
    except Exception:
        return -999


def _city_id_at(obs: Dict, x: int, y: int) -> int:
    board = obs.get("board", {})
    city_ids = board.get("cityID", []) if isinstance(board, dict) else []
    try:
        return int(city_ids[int(x)][int(y)])
    except Exception:
        return -999


def run_audit(args):
    os.environ.setdefault("POLYVISION_LEVEL_POOL_GLOB", "levels/phase1_pool/*.csv")
    os.environ.setdefault("POLYVISION_LEVEL_SELECTION_MODE", "round_robin")

    env = gym.make("Tribes-v0")
    uw = env.unwrapped

    total_candidates = 0
    true_neutral_village = 0
    enemy_owned_city = 0
    enemy_units_on_candidate = 0
    targetable_not_capturable = 0

    distance_zero_visible_no_capture = 0
    distance_zero_caused_enemy_city = 0
    distance_zero_caused_enemy_unit = 0
    distance_zero_caused_non_neutral = 0

    feat_target_true = 0
    feat_target_true_on_enemy_city = 0
    feat_target_true_on_non_candidate = 0
    feat_dist_delta_nonzero = 0

    reward_new_revealed_non_neutral = 0
    reward_new_revealed_enemy_city = 0

    obs_vec, info = env.reset(seed=int(args.seed))
    for ep in range(int(args.episodes)):
        if ep > 0:
            obs_vec, info = env.reset(seed=(int(args.seed) + ep) if args.seed_per_episode else None)

        done = False
        safety = 0
        while not done:
            full_before = uw.tribes_env.get_observation(full_visibility=True)
            villages_before = nf._visible_uncaptured_village_positions(full_before)
            owned_units_before = _owned_unit_coords(full_before)
            enemy_units = _enemy_unit_coords(full_before)
            enemy_caps = _enemy_capital_coords(full_before)
            city_owner = _city_owner_by_coord(full_before)

            ids, valid, feats = nf._extract_legal_tensors(info)
            valid_slots = np.where(valid)[0]
            move_slots = np.array([s for s in valid_slots if nf._build_move_record(env, int(ids[int(s)]), full_before) is not None], dtype=np.int64)
            capture_coords = _capture_unit_coords(env, full_before)
            move_targets = _legal_move_targets(env, full_before, move_slots, ids)

            for s in move_slots:
                feat_target = float(feats[int(s), 5]) > 0.5
                if feat_target:
                    feat_target_true += 1
                    gid = int(ids[int(s)])
                    rec = nf._build_move_record(env, gid, full_before)
                    if rec is not None:
                        dst = (int(rec["dst"][0]), int(rec["dst"][1]))
                        owner = city_owner.get(dst, None)
                        if owner is not None and owner != 0:
                            feat_target_true_on_enemy_city += 1
                        if dst not in villages_before:
                            feat_target_true_on_non_candidate += 1
                if abs(float(feats[int(s), 7])) > 1e-9:
                    feat_dist_delta_nonzero += 1

            for (x, y) in sorted(villages_before):
                total_candidates += 1
                t_val = _terrain_at(full_before, x, y)
                c_id = _city_id_at(full_before, x, y)
                owner = city_owner.get((x, y), None)
                is_neutral = (t_val == 4 and c_id == -1 and owner is None)
                is_enemy_city = owner is not None and owner != 0
                has_enemy_unit = (x, y) in enemy_units
                capture_on_tile = (x, y) in capture_coords
                targetable = (x, y) in move_targets
                enemy_capital = (x, y) in enemy_caps
                tile_kind = "neutral_village" if is_neutral else ("city" if c_id != -1 or owner is not None else "other")

                if is_neutral:
                    true_neutral_village += 1
                if is_enemy_city:
                    enemy_owned_city += 1
                if has_enemy_unit:
                    enemy_units_on_candidate += 1
                if targetable and not capture_on_tile:
                    targetable_not_capturable += 1

                print(
                    f"candidate ep={ep+1} turn={int(nf._as_scalar(info.get('turn_count', 0), 0))} "
                    f"coord=({x},{y}) tile_type={t_val} kind={tile_kind} owner={owner} city_id={c_id} "
                    f"capture_exists_on_tile={capture_on_tile} enemy_city_present={is_enemy_city} "
                    f"enemy_unit_present={has_enemy_unit} enemy_capital_tile={enemy_capital} targetable_by_move={targetable}"
                )

            if villages_before:
                zero_coords = sorted(villages_before.intersection(owned_units_before))
                capture_exists_any = len(capture_coords) > 0
                if zero_coords and not capture_exists_any:
                    distance_zero_visible_no_capture += 1
                    caused_enemy_city = False
                    caused_enemy_unit = False
                    caused_non_neutral = False
                    for (x, y) in zero_coords:
                        t_val = _terrain_at(full_before, x, y)
                        c_id = _city_id_at(full_before, x, y)
                        owner = city_owner.get((x, y), None)
                        if owner is not None and owner != 0:
                            caused_enemy_city = True
                        if (x, y) in enemy_units:
                            caused_enemy_unit = True
                        if not (t_val == 4 and c_id == -1 and owner is None):
                            caused_non_neutral = True
                    if caused_enemy_city:
                        distance_zero_caused_enemy_city += 1
                    if caused_enemy_unit:
                        distance_zero_caused_enemy_unit += 1
                    if caused_non_neutral:
                        distance_zero_caused_non_neutral += 1

            chosen_slot, _diag = nf._choose_slot(env, info, ids, valid, feats, full_before)
            chosen_gid = int(ids[int(chosen_slot)])
            obs_vec, _reward, terminated, truncated, info = env.step(chosen_gid)
            full_after = uw.tribes_env.get_observation(full_visibility=True)
            villages_after = nf._visible_uncaptured_village_positions(full_after)
            new_villages = villages_after - villages_before
            for (x, y) in new_villages:
                t_val = _terrain_at(full_after, x, y)
                c_id = _city_id_at(full_after, x, y)
                owner = _city_owner_by_coord(full_after).get((x, y), None)
                if not (t_val == 4 and c_id == -1 and owner is None):
                    reward_new_revealed_non_neutral += 1
                if owner is not None and owner != 0:
                    reward_new_revealed_enemy_city += 1

            done = bool(terminated or truncated)
            safety += 1
            if safety > int(args.max_steps_per_episode):
                raise RuntimeError("Episode safety cap exceeded.")

        if args.progress_every > 0 and ((ep + 1) % int(args.progress_every) == 0):
            print(f"progress episodes={ep + 1}/{args.episodes}")

    env.close()

    print("\n=== audit summary ===")
    print(f"episodes: {int(args.episodes)}")
    print(f"total_visible_uncaptured_village_candidates: {total_candidates}")
    print(f"true_neutral_villages: {true_neutral_village}")
    print(f"enemy_owned_cities_counted_as_candidates: {enemy_owned_city}")
    print(f"candidates_with_enemy_units_on_them: {enemy_units_on_candidate}")
    print(f"targetable_but_not_capturable_candidates: {targetable_not_capturable}")
    print(f"distance_zero_to_visible_village_but_no_capture_turns: {distance_zero_visible_no_capture}")
    print(f"distance_zero_caused_by_enemy_city: {distance_zero_caused_enemy_city}")
    print(f"distance_zero_caused_by_enemy_unit: {distance_zero_caused_enemy_unit}")
    print(f"distance_zero_caused_by_non_neutral_ownership: {distance_zero_caused_non_neutral}")
    print(f"feature_target_contains_visible_uncaptured_village_true_count: {feat_target_true}")
    print(f"feature_target_true_on_enemy_city_count: {feat_target_true_on_enemy_city}")
    print(f"feature_target_true_on_non_candidate_count: {feat_target_true_on_non_candidate}")
    print(f"feature_distance_delta_nonzero_count: {feat_dist_delta_nonzero}")
    print(f"newly_revealed_uncaptured_village_non_neutral_count: {reward_new_revealed_non_neutral}")
    print(f"newly_revealed_uncaptured_village_enemy_city_count: {reward_new_revealed_enemy_city}")

    return {
        "total": total_candidates,
        "neutral": true_neutral_village,
        "enemy_city": enemy_owned_city,
        "enemy_unit": enemy_units_on_candidate,
        "targetable_not_capturable": targetable_not_capturable,
        "dz": distance_zero_visible_no_capture,
        "dz_enemy_city": distance_zero_caused_enemy_city,
        "dz_enemy_unit": distance_zero_caused_enemy_unit,
        "dz_non_neutral": distance_zero_caused_non_neutral,
        "feat_target_true": feat_target_true,
        "feat_target_enemy_city": feat_target_true_on_enemy_city,
        "feat_target_non_candidate": feat_target_true_on_non_candidate,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit visible-uncaptured-village classification and targeting.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed-per-episode", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--max-steps-per-episode", type=int, default=256)
    run_audit(parser.parse_args())


if __name__ == "__main__":
    main()
