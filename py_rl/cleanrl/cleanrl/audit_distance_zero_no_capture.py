import argparse
import os
import re
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


def _tile_info(obs: Dict, xy: Tuple[int, int]) -> Dict:
    x, y = int(xy[0]), int(xy[1])
    board = obs.get("board", {})
    terrain = board.get("terrain", []) if isinstance(board, dict) else []
    city_id = board.get("cityID", []) if isinstance(board, dict) else []
    t_val = None
    c_val = None
    try:
        t_val = int(terrain[y][x])
    except Exception:
        t_val = None
    try:
        c_val = int(city_id[y][x])
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
        cx = _safe_int(c.get("x", -1), -1)
        cy = _safe_int(c.get("y", -1), -1)
        if cx == x and cy == y:
            out = dict(c)
            out["city_id"] = str(cid)
            return out
    return None


def _owned_units(obs: Dict) -> List[Tuple[int, int, int, int]]:
    out = []
    units = obs.get("unit", {})
    if not isinstance(units, dict):
        return out
    for uid_s, u in units.items():
        if not isinstance(u, dict):
            continue
        tribe = _safe_int(u.get("tribeId", -1), -1)
        if tribe != 0:
            continue
        uid = _safe_int(uid_s, -1)
        x = _safe_int(u.get("x", -1), -1)
        y = _safe_int(u.get("y", -1), -1)
        if uid >= 0 and x >= 0 and y >= 0:
            out.append((uid, tribe, x, y))
    return out


def _parse_action_dest(action: Dict, uw, obs: Dict) -> Optional[Tuple[int, int]]:
    dx = action.get("dst_x", None)
    dy = action.get("dst_y", None)
    try:
        if dx is not None and dy is not None:
            return int(dx), int(dy)
    except Exception:
        pass
    parsed = uw._parse_move_unit_and_dest_from_action_repr(str(action.get("repr", "")))
    if parsed is not None:
        return int(parsed[1]), int(parsed[2])
    return None


def _parse_action_src(action: Dict, uw, obs: Dict) -> Optional[Tuple[int, int]]:
    sx = action.get("src_x", None)
    sy = action.get("src_y", None)
    try:
        if sx is not None and sy is not None:
            return int(sx), int(sy)
    except Exception:
        pass
    uid = action.get("unit_id", None)
    if uid is None:
        uid = uw._parse_unit_id_from_action_repr(str(action.get("repr", "")))
    if uid is None:
        return None
    pos = uw._unit_position_by_id(obs, int(uid))
    if pos is None:
        return None
    return int(pos[0]), int(pos[1])


def _capture_actions(uw) -> List[Dict]:
    legal_actions = getattr(uw, "_current_legal_actions", [])
    return [a for a in legal_actions if str(a.get("type", "")).upper() == "CAPTURE"]


def _capture_available_for_unit_tile(uw, obs: Dict, unit_id: int, tile: Tuple[int, int]) -> bool:
    caps = _capture_actions(uw)
    tx, ty = int(tile[0]), int(tile[1])
    for a in caps:
        uid = a.get("unit_id", None)
        if uid is None:
            uid = uw._parse_unit_id_from_action_repr(str(a.get("repr", "")))
        if uid is None or int(uid) != int(unit_id):
            continue
        pos = uw._unit_position_by_id(obs, int(uid))
        if pos is None:
            continue
        if (int(pos[0]), int(pos[1])) == (tx, ty):
            return True
    return False


def _actions_reference_coord(uw, obs: Dict, actions: List[Dict], coord: Tuple[int, int]) -> bool:
    cx, cy = int(coord[0]), int(coord[1])
    pattern = re.compile(rf"\bto\s+{cx}\s*:\s*{cy}\b")
    for a in actions:
        src = _parse_action_src(a, uw, obs)
        dst = _parse_action_dest(a, uw, obs)
        tx = a.get("target_x", None)
        ty = a.get("target_y", None)
        if src == (cx, cy) or dst == (cx, cy):
            return True
        try:
            if tx is not None and ty is not None and int(tx) == cx and int(ty) == cy:
                return True
        except Exception:
            pass
        if pattern.search(str(a.get("repr", ""))):
            return True
    return False


def _run(args):
    os.environ.setdefault("POLYVISION_LEVEL_POOL_GLOB", "levels/phase1_pool/*.csv")
    os.environ.setdefault("POLYVISION_LEVEL_SELECTION_MODE", "round_robin")
    os.environ["POLYVISION_INFO_MODE"] = "debug"

    env = gym.make("Tribes-v0")
    uw = env.unwrapped

    total_cases = 0
    cause_direct_no_capture = 0
    cause_swapped_false_positive = 0
    cause_coord_mismatch = 0
    cause_not_bardur = 0
    cause_unit_not_on_village_java_coords = 0
    cause_capture_next_step_turn = 0
    cause_unexplained = 0

    printed_cases = 0
    case_id_ctr = 0
    pending: Dict[int, Dict] = {}
    finalized_cases: List[Dict] = []

    obs_vec, info = env.reset(seed=int(args.seed))
    decision_idx = 0
    last_move = None

    for ep in range(int(args.episodes)):
        if ep > 0:
            for cid, c in list(pending.items()):
                if not c.get("resolved", False):
                    c["resolved"] = True
                    c["capture_next"] = False
                    if c.get("print_me", False) and not c.get("printed", False):
                        _print_case(c)
                        c["printed"] = True
                finalized_cases.append(c)
            pending.clear()
            obs_vec, info = env.reset(seed=(int(args.seed) + ep) if args.seed_per_episode else None)
            last_move = None

        done = False
        safety = 0
        while not done:
            decision_idx += 1
            full_obs = uw.tribes_env.get_observation(full_visibility=True)
            active_tribe = _safe_int(full_obs.get("activeTribeID", -1), -1)
            bardur_turn = int(nf._as_scalar(info.get("turn_count", 0), 0))
            candidates = uw._get_visible_uncaptured_village_positions(full_obs)
            cap_actions = _capture_actions(uw)
            cap_reprs = [str(a.get("repr", "")) for a in cap_actions]

            # Update pending cases with capture availability on later decisions/turns.
            for cid, c in list(pending.items()):
                if c.get("resolved", False):
                    continue
                if _capture_available_for_unit_tile(uw, full_obs, c["unit_id"], c["village"]):
                    c["resolved"] = True
                    c["capture_next"] = True
                    if decision_idx == c["decision_idx"] + 1:
                        c["capture_next_decision"] = True
                    if bardur_turn > c["turn"]:
                        c["capture_next_turn"] = True
                    if c.get("print_me", False) and not c.get("printed", False):
                        _print_case(c)
                        c["printed"] = True
                    finalized_cases.append(c)
                    pending.pop(cid, None)
                elif bardur_turn > c["turn"] + 1:
                    c["resolved"] = True
                    c["capture_next"] = False
                    if c.get("print_me", False) and not c.get("printed", False):
                        _print_case(c)
                        c["printed"] = True
                    finalized_cases.append(c)
                    pending.pop(cid, None)

            zero_units = []
            if len(cap_actions) == 0 and len(candidates) > 0:
                for uid, owner, ux, uy in _owned_units(full_obs):
                    if (ux, uy) in candidates:
                        zero_units.append((uid, owner, ux, uy))

            if len(zero_units) > 0:
                total_cases += 1
                zero_units.sort(key=lambda t: t[0])
                uid, owner, ux, uy = zero_units[0]
                village = (int(ux), int(uy))
                village_swapped = (int(village[1]), int(village[0]))
                unit_swapped = (int(uy), int(ux))
                dist_direct = 0
                dist_swap_u = 0 if unit_swapped in candidates else min(
                    (abs(unit_swapped[0] - vx) + abs(unit_swapped[1] - vy) for vx, vy in candidates),
                    default=999999,
                )
                swapped_candidates = {(int(vy), int(vx)) for vx, vy in candidates}
                dist_swap_v = 0 if village_swapped in candidates else min(
                    (abs(ux - sx) + abs(uy - sy) for sx, sy in swapped_candidates),
                    default=999999,
                )
                dist_both = 0 if unit_swapped in swapped_candidates else min(
                    (abs(unit_swapped[0] - sx) + abs(unit_swapped[1] - sy) for sx, sy in swapped_candidates),
                    default=999999,
                )

                legal_actions = getattr(uw, "_current_legal_actions", [])
                unit_moves = []
                src_direct_match = False
                src_swapped_match = False
                for a in legal_actions:
                    if str(a.get("type", "")).upper() != "MOVE":
                        continue
                    auid = a.get("unit_id", None)
                    if auid is None:
                        auid = uw._parse_unit_id_from_action_repr(str(a.get("repr", "")))
                    if auid is None or int(auid) != int(uid):
                        continue
                    unit_moves.append(str(a.get("repr", "")))
                    src = _parse_action_src(a, uw, full_obs)
                    if src == (ux, uy):
                        src_direct_match = True
                    if src == (uy, ux):
                        src_swapped_match = True

                ref_direct = _actions_reference_coord(uw, full_obs, legal_actions, village)
                ref_swapped = _actions_reference_coord(uw, full_obs, legal_actions, village_swapped)
                tile_v = _tile_info(full_obs, village)
                tile_vs = _tile_info(full_obs, village_swapped)
                city_v = _city_actor_at(full_obs, village)
                city_vs = _city_actor_at(full_obs, village_swapped)

                just_moved_onto = False
                if isinstance(last_move, dict):
                    if int(last_move.get("unit_id", -1)) == int(uid) and tuple(last_move.get("dst", (-1, -1))) == village:
                        just_moved_onto = True

                forced_pre = _safe_int(info.get("forced_pre_end_turns", -1), -1)
                forced_post = _safe_int(info.get("forced_post_end_turns", -1), -1)

                # Cause flags
                direct_flag = True
                swapped_false = dist_direct > 0 and (dist_swap_u == 0 or dist_swap_v == 0 or dist_both == 0)
                coord_mismatch = bool(src_swapped_match and not src_direct_match)
                not_bardur = active_tribe != 0
                unit_not_on_java = not src_direct_match and len(unit_moves) > 0

                case_id_ctr += 1
                c = {
                    "case_id": case_id_ctr,
                    "episode": ep + 1,
                    "turn": bardur_turn,
                    "decision_idx": decision_idx,
                    "unit_id": int(uid),
                    "unit_owner": int(owner),
                    "unit_pos": (int(ux), int(uy)),
                    "unit_pos_swapped": unit_swapped,
                    "village": village,
                    "village_swapped": village_swapped,
                    "tile_village": tile_v,
                    "tile_village_swapped": tile_vs,
                    "city_actor_village": city_v,
                    "city_actor_village_swapped": city_vs,
                    "dist_direct": dist_direct,
                    "dist_swap_unit": int(dist_swap_u),
                    "dist_swap_village": int(dist_swap_v),
                    "dist_both_swapped": int(dist_both),
                    "capture_reprs": cap_reprs,
                    "unit_move_reprs": unit_moves,
                    "ref_direct": bool(ref_direct),
                    "ref_swapped": bool(ref_swapped),
                    "just_moved_onto": bool(just_moved_onto),
                    "active_tribe": int(active_tribe),
                    "forced_pre_end_turns": int(forced_pre),
                    "forced_post_end_turns": int(forced_post),
                    "capture_next": False,
                    "capture_next_decision": False,
                    "capture_next_turn": False,
                    "resolved": False,
                    "printed": False,
                    "print_me": printed_cases < int(args.print_first_cases),
                    "cause_direct": direct_flag,
                    "cause_swapped_false": swapped_false,
                    "cause_coord_mismatch": coord_mismatch,
                    "cause_not_bardur": not_bardur,
                    "cause_unit_not_on_java": unit_not_on_java,
                }
                pending[c["case_id"]] = c
                if c["print_me"]:
                    printed_cases += 1

            ids, valid, feats = nf._extract_legal_tensors(info)
            chosen_slot, _diag = nf._choose_slot(env, info, ids, valid, feats, full_obs)
            chosen_gid = int(ids[int(chosen_slot)])
            chosen_move = nf._build_move_record(env, chosen_gid, full_obs)
            last_move = None
            if chosen_move is not None:
                last_move = {"unit_id": int(chosen_move["unit_id"]), "dst": tuple(chosen_move["dst"])}

            obs_vec, _reward, terminated, truncated, info = env.step(chosen_gid)
            done = bool(terminated or truncated)
            safety += 1
            if safety > int(args.max_steps_per_episode):
                raise RuntimeError("Episode safety cap exceeded.")

        if args.progress_every > 0 and ((ep + 1) % int(args.progress_every) == 0):
            print(f"progress episodes={ep + 1}/{args.episodes}")

    # Finalize pending cases.
    for cid, c in list(pending.items()):
        if not c.get("resolved", False):
            c["resolved"] = True
            c["capture_next"] = False
        if c.get("print_me", False) and not c.get("printed", False):
            _print_case(c)
            c["printed"] = True
        finalized_cases.append(c)
    pending.clear()

    # Aggregate final causes.
    all_cases = list(finalized_cases)
    for c in all_cases:
        if c.get("cause_direct", False):
            cause_direct_no_capture += 1
        if c.get("cause_swapped_false", False):
            cause_swapped_false_positive += 1
        if c.get("cause_coord_mismatch", False):
            cause_coord_mismatch += 1
        if c.get("cause_not_bardur", False):
            cause_not_bardur += 1
        if c.get("cause_unit_not_on_java", False):
            cause_unit_not_on_village_java_coords += 1
        if c.get("capture_next", False):
            cause_capture_next_step_turn += 1

        explained = (
            c.get("cause_swapped_false", False)
            or c.get("cause_coord_mismatch", False)
            or c.get("cause_not_bardur", False)
            or c.get("cause_unit_not_on_java", False)
            or c.get("capture_next", False)
        )
        if not explained:
            cause_unexplained += 1

    env.close()

    denom = max(1, int(total_cases))
    print("\n=== distance_zero_no_capture audit summary ===")
    print(f"episodes: {int(args.episodes)}")
    print(f"total_distance_zero_no_capture_cases: {int(total_cases)}")
    print(f"percent_caused_by_direct_coordinate_match_but_no_capture: {100.0 * cause_direct_no_capture / denom:.6f}")
    print(f"percent_caused_by_swapped_coordinate_distance_false_positive: {100.0 * cause_swapped_false_positive / denom:.6f}")
    print(f"percent_caused_by_unit_village_coordinate_convention_mismatch: {100.0 * cause_coord_mismatch / denom:.6f}")
    print(f"percent_caused_by_not_bardur_active_turn: {100.0 * cause_not_bardur / denom:.6f}")
    print(f"percent_caused_by_unit_not_on_village_tile_under_java_action_coordinates: {100.0 * cause_unit_not_on_village_java_coords / denom:.6f}")
    print(f"percent_caused_by_capture_only_becoming_available_next_step_turn: {100.0 * cause_capture_next_step_turn / denom:.6f}")
    print(f"percent_unexplained: {100.0 * cause_unexplained / denom:.6f}")


def _print_case(c: Dict):
    print(
        f"\ncase={c['case_id']} ep={c['episode']} turn={c['turn']}"
        f"\n1) episode_turn: ep={c['episode']} turn={c['turn']}"
        f"\n2) unit: id={c['unit_id']} owner={c['unit_owner']} pos={c['unit_pos']} pos_swapped={c['unit_pos_swapped']}"
        f"\n3) village: coord={c['village']} tile={c['tile_village']} city_actor={c['city_actor_village']}"
        f" swapped_coord={c['village_swapped']} swapped_tile={c['tile_village_swapped']} swapped_city_actor={c['city_actor_village_swapped']}"
        f"\n4) distances: direct={c['dist_direct']} swapped_unit={c['dist_swap_unit']} swapped_village={c['dist_swap_village']} both_swapped={c['dist_both_swapped']}"
        f"\n5) legal_actions: capture_reprs={c['capture_reprs']} unit_move_reprs={c['unit_move_reprs']}"
        f" refs_direct={c['ref_direct']} refs_swapped={c['ref_swapped']}"
        f"\n6) timing: just_moved_onto_tile={c['just_moved_onto']} capture_next_decision={c['capture_next_decision']} capture_next_turn={c['capture_next_turn']}"
        f" capture_only_next_step_turn={c['capture_next']} active_tribe={c['active_tribe']}"
        f" forced_pre_end_turns={c['forced_pre_end_turns']} forced_post_end_turns={c['forced_post_end_turns']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Audit distance_zero_to_visible_village_but_no_capture cases.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed-per-episode", action="store_true")
    parser.add_argument("--print-first-cases", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-steps-per-episode", type=int, default=256)
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
