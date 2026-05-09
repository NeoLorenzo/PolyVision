import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np


_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import pol_env.Tribes.py.register_env as register_env  # noqa: F401
except Exception:
    pass


IDX_NEW_REVEAL_NORM = 1
IDX_IS_ZERO_REVEAL_MOVE = 4
IDX_IS_END_TURN = 12
IDX_IS_CAPTURE = 13
IDX_IS_TRAIN_OR_SPAWN = 14
IDX_IS_RESOURCE_GATHERING = 16
IDX_IS_LEVEL_UP = 17
IDX_IS_IMMEDIATE_BACKTRACK = 8


def _as_scalar(value, default=None):
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        return _as_scalar(value[0], default=default)
    return value


def _extract_legal_tensors(info: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "legal_global_ids_padded" not in info or "legal_action_valid_mask" not in info:
        raise RuntimeError("Missing legal_global_ids_padded/legal_action_valid_mask in info.")
    if "legal_action_features_padded" not in info:
        raise RuntimeError("Missing legal_action_features_padded in info.")

    ids = np.asarray(info["legal_global_ids_padded"], dtype=np.int64)
    valid = np.asarray(info["legal_action_valid_mask"], dtype=bool)
    feats = np.asarray(info["legal_action_features_padded"], dtype=np.float32)
    if ids.ndim > 1:
        ids = ids.reshape(-1)
    if valid.ndim > 1:
        valid = valid.reshape(-1)
    if feats.ndim == 3:
        feats = feats[0]
    elif feats.ndim != 2:
        feats = feats.reshape((-1, feats.shape[-1]))
    n = min(ids.shape[0], valid.shape[0], feats.shape[0])
    return ids[:n], valid[:n], feats[:n]


def _visible_uncaptured_village_positions(obs: Dict) -> Set[Tuple[int, int]]:
    # Keep this aligned with the wrapper's neutral-visible-village definition.
    # Use wrapper helper if present to avoid classifier drift.
    return _visible_uncaptured_village_positions_fallback(obs)


def _visible_uncaptured_village_positions_from_env(env, obs: Dict) -> Set[Tuple[int, int]]:
    uw = env.unwrapped
    fn = getattr(uw, "_get_visible_uncaptured_village_positions", None)
    if callable(fn):
        try:
            out = fn(obs)
            return {(int(x), int(y)) for (x, y) in out}
        except Exception:
            pass
    return _visible_uncaptured_village_positions_fallback(obs)


def _visible_uncaptured_village_positions_fallback(obs: Dict) -> Set[Tuple[int, int]]:
    # Java/runtime frame: board arrays are indexed as [x][y].
    board = obs.get("board", {})
    terrain = board.get("terrain", []) if isinstance(board, dict) else []
    city_ids = board.get("cityID", []) if isinstance(board, dict) else []
    out: Set[Tuple[int, int]] = set()
    if not isinstance(terrain, list) or not isinstance(city_ids, list):
        return out

    occupied_city_coords: Set[Tuple[int, int]] = set()
    city_map = obs.get("city", {})
    if isinstance(city_map, dict):
        for c in city_map.values():
            if not isinstance(c, dict):
                continue
            try:
                cx = int(c.get("x", -1))
                cy = int(c.get("y", -1))
            except Exception:
                continue
            if cx >= 0 and cy >= 0:
                occupied_city_coords.add((cx, cy))

    width = len(terrain)
    height = max((len(col) for col in terrain if isinstance(col, (list, tuple))), default=0)
    for x in range(width):
        col_t = terrain[x] if x < len(terrain) else []
        col_c = city_ids[x] if x < len(city_ids) else []
        if not isinstance(col_t, (list, tuple)):
            continue
        for y in range(height):
            try:
                t_val = int(col_t[y]) if y < len(col_t) else -1
                c_val = int(col_c[y]) if y < len(col_c) else -1
            except Exception:
                continue
            if t_val == 4 and c_val == -1 and (int(x), int(y)) not in occupied_city_coords:
                out.add((int(x), int(y)))
    return out


def _unit_by_id(obs: Dict, unit_id: int) -> Optional[Dict[str, Any]]:
    units = obs.get("unit", {})
    if not isinstance(units, dict):
        return None
    return units.get(str(int(unit_id)), None)


def _city_actor_by_coord(obs: Dict, coord: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    cx, cy = int(coord[0]), int(coord[1])
    city_map = obs.get("city", {})
    if not isinstance(city_map, dict):
        return None
    for c in city_map.values():
        if not isinstance(c, dict):
            continue
        try:
            if int(c.get("x", -1)) == cx and int(c.get("y", -1)) == cy:
                return c
        except Exception:
            continue
    return None


def _units_on_coord(obs: Dict, coord: Tuple[int, int]) -> List[Dict[str, Any]]:
    x, y = int(coord[0]), int(coord[1])
    out: List[Dict[str, Any]] = []
    units = obs.get("unit", {})
    if not isinstance(units, dict):
        return out
    for uid_s, u in units.items():
        if not isinstance(u, dict):
            continue
        try:
            ux = int(u.get("x", -1))
            uy = int(u.get("y", -1))
        except Exception:
            continue
        if ux == x and uy == y:
            uu = dict(u)
            uu["_id"] = int(uid_s)
            out.append(uu)
    return out


def _tile_info(obs: Dict, coord: Tuple[int, int]) -> Dict[str, Any]:
    x, y = int(coord[0]), int(coord[1])
    board = obs.get("board", {})
    if not isinstance(board, dict):
        return {"coord": (x, y), "in_bounds": False}

    terrain = board.get("terrain", [])
    resource = board.get("resource", [])
    building = board.get("building", [])
    city_ids = board.get("cityID", [])
    unit_ids = board.get("unitID", [])

    w = len(terrain) if isinstance(terrain, list) else 0
    h = max((len(col) for col in terrain if isinstance(col, (list, tuple))), default=0)
    in_bounds = 0 <= x < w and 0 <= y < h

    def _at(grid, default=-1):
        try:
            if not isinstance(grid, list):
                return default
            col = grid[x]
            if not isinstance(col, (list, tuple)):
                return default
            return int(col[y]) if y < len(col) else default
        except Exception:
            return default

    return {
        "coord": (x, y),
        "in_bounds": bool(in_bounds),
        "terrain": _at(terrain, -1),
        "resource": _at(resource, -1),
        "building": _at(building, -1),
        "cityID": _at(city_ids, -1),
        "unitID": _at(unit_ids, -1),
    }


def _raw_action_for_gid(env, gid: int):
    uw = env.unwrapped
    legal_id_to_raw = getattr(uw, "_current_legal_id_to_raw_index", {})
    legal_actions = getattr(uw, "_current_legal_actions", [])
    raw_idx = legal_id_to_raw.get(int(gid), None)
    if raw_idx is None or raw_idx < 0 or raw_idx >= len(legal_actions):
        return None, None
    return int(raw_idx), legal_actions[int(raw_idx)]


def _unit_action_legalities(env, unit_id: int) -> Dict[str, Any]:
    uw = env.unwrapped
    legal_actions = getattr(uw, "_current_legal_actions", [])
    move_actions = []
    capture_actions = []
    other_actions = []
    for a in legal_actions:
        uid = a.get("unit_id", None)
        if uid is None:
            uid = uw._parse_unit_id_from_action_repr(str(a.get("repr", "")))
        try:
            if uid is None or int(uid) != int(unit_id):
                continue
        except Exception:
            continue
        a_type = str(a.get("type", "")).upper()
        if a_type == "MOVE":
            move_actions.append(a)
        elif a_type == "CAPTURE":
            capture_actions.append(a)
        else:
            other_actions.append(a)
    return {
        "move_actions": move_actions,
        "capture_actions": capture_actions,
        "other_actions": other_actions,
        "can_move_proxy": bool(len(move_actions) > 0),
        "can_capture_proxy": bool(len(capture_actions) > 0),
        "can_act_proxy": bool(len(move_actions) > 0 or len(capture_actions) > 0 or len(other_actions) > 0),
        # Freshness isn't exposed directly; use a proxy from legal action availability.
        "is_fresh_proxy": bool(len(move_actions) > 0 or len(capture_actions) > 0),
    }


def _owned_units(obs: Dict) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    units = obs.get("unit", {})
    if not isinstance(units, dict):
        return out
    for uid_s, unit in units.items():
        if not isinstance(unit, dict):
            continue
        try:
            if int(unit.get("tribeId", -1)) != 0:
                continue
            uid = int(uid_s)
            ux = int(unit.get("x", -1))
            uy = int(unit.get("y", -1))
        except Exception:
            continue
        if ux >= 0 and uy >= 0:
            out.append((uid, ux, uy))
    return out


def _capture_legal_unit_ids(env) -> Set[int]:
    uw = env.unwrapped
    legal_actions = getattr(uw, "_current_legal_actions", [])
    out: Set[int] = set()
    for a in legal_actions:
        if str(a.get("type", "")).upper() != "CAPTURE":
            continue
        uid = a.get("unit_id", None)
        if uid is None:
            uid = uw._parse_unit_id_from_action_repr(str(a.get("repr", "")))
        try:
            if uid is not None:
                out.add(int(uid))
        except Exception:
            continue
    return out


def _units_on_visible_neutral_village_capture_illegal(env, full_obs: Dict) -> Set[int]:
    villages = _visible_uncaptured_village_positions_from_env(env, full_obs)
    if not villages:
        return set()
    capture_units = _capture_legal_unit_ids(env)
    out: Set[int] = set()
    for uid, ux, uy in _owned_units(full_obs):
        if (int(ux), int(uy)) in villages and int(uid) not in capture_units:
            out.add(int(uid))
    return out


def _min_manhattan(a: Tuple[int, int], targets: Set[Tuple[int, int]]) -> Optional[int]:
    if a is None or not targets:
        return None
    ax, ay = int(a[0]), int(a[1])
    return min(abs(ax - tx) + abs(ay - ty) for tx, ty in targets)


def _pct(arr: np.ndarray, thresh: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(100.0 * np.mean(arr >= float(thresh)))


def _mean_nonneg(values: List[float]) -> float:
    filtered = [float(v) for v in values if v is not None and float(v) >= 0.0]
    if not filtered:
        return float("nan")
    return float(np.mean(filtered))


def _build_move_record(env, gid: int, full_obs: Dict):
    uw = env.unwrapped
    legal_id_to_raw = getattr(uw, "_current_legal_id_to_raw_index", {})
    legal_actions = getattr(uw, "_current_legal_actions", [])
    raw_idx = legal_id_to_raw.get(int(gid), None)
    if raw_idx is None or raw_idx < 0 or raw_idx >= len(legal_actions):
        return None
    action = legal_actions[int(raw_idx)]
    unit_id, src_x, src_y, dst_x, dst_y = uw._extract_move_components(action, full_obs)
    if unit_id is None or src_x is None or src_y is None or dst_x is None or dst_y is None:
        return None
    return {
        "raw_idx": int(raw_idx),
        "action": action,
        "unit_id": int(unit_id),
        "src": (int(src_x), int(src_y)),
        "dst": (int(dst_x), int(dst_y)),
    }


def _choose_slot(env, info: Dict, ids: np.ndarray, valid: np.ndarray, feats: np.ndarray, full_obs: Dict):
    valid_slots = np.where(valid)[0]
    if valid_slots.size == 0:
        raise RuntimeError("No valid legal action slots.")

    villages = _visible_uncaptured_village_positions_from_env(env, full_obs)
    board = full_obs.get("board", {})
    terrain = board.get("terrain", []) if isinstance(board, dict) else []
    w = len(terrain) if isinstance(terrain, list) else 0
    h = max((len(col) for col in terrain if isinstance(col, (list, tuple))), default=0)
    dist_norm_den = float(max(1, (w + h - 2)))

    capture_slots = valid_slots[feats[valid_slots, IDX_IS_CAPTURE] > 0.5]
    if capture_slots.size > 0:
        return int(capture_slots[0]), {"reason": "capture_priority", "move": None}

    units_hold = _units_on_visible_neutral_village_capture_illegal(env, full_obs)

    move_slots = []
    for s in valid_slots:
        gid = int(ids[int(s)])
        rec = _build_move_record(env, gid, full_obs)
        if rec is not None and int(rec["unit_id"]) not in units_hold:
            move_slots.append((int(s), rec))

    if villages and move_slots:
        best_slot = None
        best_score = None
        best_rec = None
        for s, rec in move_slots:
            feat = feats[int(s)]
            src = rec["src"]
            dst = rec["dst"]
            d_before = _min_manhattan(src, villages)
            d_after = _min_manhattan(dst, villages)
            if d_before is None or d_after is None:
                dist_delta_norm = 0.0
            else:
                dist_delta_norm = float(d_before - d_after) / dist_norm_den
            target_contains = 1.0 if dst in villages else 0.0
            score = (
                10.0 * target_contains
                + 5.0 * dist_delta_norm
                + 1.0 * float(feat[IDX_NEW_REVEAL_NORM])
                - 2.0 * float(feat[IDX_IS_IMMEDIATE_BACKTRACK])
                - 1.0 * float(feat[IDX_IS_ZERO_REVEAL_MOVE])
            )
            if best_score is None or score > best_score:
                best_score = float(score)
                best_slot = int(s)
                best_rec = rec
        if best_slot is not None:
            return best_slot, {"reason": "village_target_move", "move": best_rec}

    if move_slots:
        best_slot = max(move_slots, key=lambda t: float(feats[int(t[0]), IDX_NEW_REVEAL_NORM]))[0]
        best_rec = next((rec for s, rec in move_slots if int(s) == int(best_slot)), None)
        return int(best_slot), {"reason": "reveal_fallback_move", "move": best_rec}

    stars = int(_as_scalar(info.get("stars", 0), default=0))
    level_up_slots = valid_slots[feats[valid_slots, IDX_IS_LEVEL_UP] > 0.5]
    if level_up_slots.size > 0:
        return int(level_up_slots[0]), {"reason": "level_up", "move": None}

    resource_slots = valid_slots[feats[valid_slots, IDX_IS_RESOURCE_GATHERING] > 0.5]
    if resource_slots.size > 0:
        return int(resource_slots[0]), {"reason": "resource", "move": None}

    train_slots = valid_slots[feats[valid_slots, IDX_IS_TRAIN_OR_SPAWN] > 0.5]
    if train_slots.size > 0 and stars > 0:
        uw = env.unwrapped
        legal_actions = getattr(uw, "_current_legal_actions", [])
        legal_id_to_raw = getattr(uw, "_current_legal_id_to_raw_index", {})
        for slot in train_slots:
            gid = int(ids[int(slot)])
            raw_idx = legal_id_to_raw.get(gid, None)
            if raw_idx is None or raw_idx < 0 or raw_idx >= len(legal_actions):
                continue
            action = legal_actions[int(raw_idx)]
            unit_type = str(action.get("unit_type", "")).upper()
            repr_s = str(action.get("repr", "")).upper()
            if unit_type == "WARRIOR" or "WARRIOR" in repr_s:
                return int(slot), {"reason": "warrior_train", "move": None}

    end_turn_slots = valid_slots[feats[valid_slots, IDX_IS_END_TURN] > 0.5]
    if end_turn_slots.size > 0:
        return int(end_turn_slots[0]), {"reason": "end_turn", "move": None, "hold_units": sorted(units_hold)}
    return int(valid_slots[0]), {"reason": "first_valid", "move": None, "hold_units": sorted(units_hold)}


def _safe_int(v, default=-1):
    try:
        return int(v)
    except Exception:
        return int(default)


def _capture_action_strs(env) -> List[str]:
    legal_actions = getattr(env.unwrapped, "_current_legal_actions", [])
    out: List[str] = []
    for a in legal_actions:
        if str(a.get("type", "")).upper() == "CAPTURE":
            out.append(str(a.get("repr", a)))
    return out


def _print_wait_trace_before(case_idx: int, event: Dict[str, Any]) -> None:
    print(f"\n--- WAIT_CASE_BEFORE #{case_idx} ---")
    print(f"episode={event['episode']} turn={event['turn']} activeTribeID={event['active_tribe_id']}")
    print(f"unit_id={event['unit_id']} unit_owner={event['unit_owner']} unit_pos={event['unit_pos']} village_coord={event['village_coord']}")
    print(
        "unit_fresh_proxy={fresh} canMove_proxy={move} canAct_proxy={act}".format(
            fresh=event["unit_fresh_proxy"],
            move=event["unit_can_move_proxy"],
            act=event["unit_can_act_proxy"],
        )
    )
    print(f"tile_at_unit={event['tile_at_unit']}")
    print(f"city_actor_at_unit={event['city_actor_at_unit']}")
    print(f"legal_capture_actions={event['legal_capture_actions']}")
    print(f"selected_end_turn_raw_action={event['selected_end_turn_raw_action']}")


def _print_wait_trace_after(case_idx: int, event: Dict[str, Any], outcome: Dict[str, Any]) -> None:
    print(f"--- WAIT_CASE_AFTER  #{case_idx} ---")
    print(
        f"next_turn={outcome['turn']} activeTribeID={outcome['active_tribe_id']} "
        f"same_unit_exists={outcome['same_unit_exists']} same_unit_pos={outcome['unit_pos']}"
    )
    print(
        f"same_unit_same_tile={outcome['same_unit_same_tile']} village_still_neutral={outcome['village_still_neutral']} "
        f"capture_exists_for_unit={outcome['capture_exists_for_unit']}"
    )
    print(
        "unit_fresh_proxy={fresh} canMove_proxy={move} canAct_proxy={act}".format(
            fresh=outcome["unit_fresh_proxy"],
            move=outcome["unit_can_move_proxy"],
            act=outcome["unit_can_act_proxy"],
        )
    )
    print(f"tile_at_village_next={outcome['tile_at_village']}")
    print(f"city_actor_at_village_next={outcome['city_actor_at_village']}")
    print(f"units_at_village_next={outcome['units_at_village']}")
    print(f"legal_capture_actions_next={outcome['legal_capture_actions']}")
    print(f"capture_missing_reason={outcome['capture_missing_reason']}")
    print(
        f"state_timing: just_moved_onto_this_tile_this_step=False "
        f"non_bardur_force_end_processed={outcome['active_tribe_id']==0}"
    )


def evaluate(args):
    os.environ.setdefault("POLYVISION_LEVEL_POOL_GLOB", "levels/phase1_pool/*.csv")
    os.environ.setdefault("POLYVISION_LEVEL_SELECTION_MODE", "round_robin")

    env = gym.make("Tribes-v0")
    uw = env.unwrapped

    terminal_spt: List[float] = []
    terminal_city_count: List[float] = []
    terminal_fog_cleared: List[float] = []
    terminal_unit_count: List[float] = []
    first_visible_turns: List[float] = []
    second_city_turns: List[float] = []
    second_city_not_captured = 0

    pre_second_move_count = 0
    pre_second_move_toward = 0
    pre_second_move_away = 0
    capture_existed_turns = 0
    capture_existed_not_selected = 0
    pre_second_turns_with_move = 0
    pre_second_end_turn_while_move_exists = 0
    pre_second_turns_total = 0
    distance_zero_visible_no_capture = 0
    hold_condition_turns = 0
    end_turn_while_holding = 0
    moved_off_before_capture = 0
    waiting_events = 0
    waiting_capture_next_turn = 0
    wait_same_unit_same_tile = 0
    wait_village_still_neutral = 0
    wait_unit_fresh_next_turn = 0
    wait_missing_capture_due_not_fresh = 0
    wait_missing_capture_due_unit_moved_or_missing = 0
    wait_missing_capture_due_tile_not_neutral = 0
    wait_missing_capture_despite_fresh_neutral = 0
    wait_missing_capture_unexplained = 0
    first_stand_second_village_turns: List[float] = []
    pending_wait_events: List[Dict[str, Any]] = []
    wait_trace_printed = 0

    obs_vec, info = env.reset(seed=int(args.seed))
    for ep in range(int(args.episodes)):
        if ep > 0:
            obs_vec, info = env.reset(seed=(int(args.seed) + ep) if args.seed_per_episode else None)

        first_visible_turn = None
        second_city_turn = None
        first_stand_second_village_turn = None
        done = False
        safety_steps = 0
        while not done:
            full_obs_before = uw.tribes_env.get_observation(full_visibility=True)
            villages_before = _visible_uncaptured_village_positions_from_env(env, full_obs_before)
            pre_turn = int(_as_scalar(info.get("turn_count", 0), default=0))
            pre_city_count = int(_as_scalar(info.get("city_count", 1), default=1))
            hold_units_before = _units_on_visible_neutral_village_capture_illegal(env, full_obs_before)
            active_tribe_before = _safe_int(full_obs_before.get("activeTribeID", -1), default=-1)

            if first_visible_turn is None and len(villages_before) > 0:
                first_visible_turn = int(pre_turn)
            if (
                first_stand_second_village_turn is None
                and pre_city_count < 2
                and len(hold_units_before) > 0
            ):
                first_stand_second_village_turn = int(pre_turn)

            ids, valid, feats = _extract_legal_tensors(info)
            valid_slots = np.where(valid)[0]
            capture_slots = valid_slots[feats[valid_slots, IDX_IS_CAPTURE] > 0.5]
            move_slots = []
            for s in valid_slots:
                gid = int(ids[int(s)])
                if _build_move_record(env, gid, full_obs_before) is not None:
                    move_slots.append(int(s))

            if pre_city_count < 2:
                pre_second_turns_total += 1
                if capture_slots.size > 0:
                    capture_existed_turns += 1
                if len(move_slots) > 0:
                    pre_second_turns_with_move += 1

                if villages_before:
                    units = _owned_units(full_obs_before)
                    any_zero = False
                    for _uid, ux, uy in units:
                        d = _min_manhattan((ux, uy), villages_before)
                        if d is not None and d == 0:
                            any_zero = True
                            break
                    if any_zero and capture_slots.size == 0:
                        distance_zero_visible_no_capture += 1
                if len(hold_units_before) > 0:
                    hold_condition_turns += 1

            chosen_slot, choice_diag = _choose_slot(env, info, ids, valid, feats, full_obs_before)
            chosen_gid = int(ids[int(chosen_slot)])
            chosen_feat = feats[int(chosen_slot)]
            chosen_is_capture = float(chosen_feat[IDX_IS_CAPTURE]) > 0.5
            chosen_is_end_turn = float(chosen_feat[IDX_IS_END_TURN]) > 0.5

            if pre_city_count < 2 and capture_slots.size > 0 and not chosen_is_capture:
                capture_existed_not_selected += 1
            if pre_city_count < 2 and chosen_is_end_turn and len(move_slots) > 0:
                pre_second_end_turn_while_move_exists += 1
            if len(hold_units_before) > 0 and chosen_is_end_turn:
                if pre_city_count < 2:
                    end_turn_while_holding += 1
                waiting_events += 1
                tracked_unit_id = int(sorted(list(hold_units_before))[0])
                tracked_unit = _unit_by_id(full_obs_before, tracked_unit_id) or {}
                tracked_pos = (
                    _safe_int(tracked_unit.get("x", -1), -1),
                    _safe_int(tracked_unit.get("y", -1), -1),
                )
                village_coord = tracked_pos
                selected_raw_idx, selected_raw_action = _raw_action_for_gid(env, chosen_gid)
                before_legality = _unit_action_legalities(env, tracked_unit_id)
                event = {
                    "episode": int(ep + 1),
                    "turn": int(pre_turn),
                    "target_turn": int(pre_turn + 1),
                    "unit_id": int(tracked_unit_id),
                    "unit_owner": _safe_int(tracked_unit.get("tribeId", -1), -1),
                    "unit_pos": tracked_pos,
                    "village_coord": village_coord,
                    "active_tribe_id": int(active_tribe_before),
                    "tile_at_unit": _tile_info(full_obs_before, tracked_pos),
                    "city_actor_at_unit": _city_actor_by_coord(full_obs_before, tracked_pos),
                    "legal_capture_actions": _capture_action_strs(env),
                    "selected_end_turn_raw_action": selected_raw_action if selected_raw_idx is not None else None,
                    "unit_fresh_proxy": bool(before_legality["is_fresh_proxy"]),
                    "unit_can_move_proxy": bool(before_legality["can_move_proxy"]),
                    "unit_can_act_proxy": bool(before_legality["can_act_proxy"]),
                }
                if wait_trace_printed < int(args.audit_wait_cases):
                    _print_wait_trace_before(wait_trace_printed + 1, event)
                pending_wait_events.append(event)

            move_rec = choice_diag.get("move", None)
            if pre_city_count < 2 and move_rec is not None:
                pre_second_move_count += 1
                src = move_rec["src"]
                dst = move_rec["dst"]
                d_before = _min_manhattan(src, villages_before) if villages_before else None
                d_after = _min_manhattan(dst, villages_before) if villages_before else None
                if d_before is not None and d_after is not None:
                    if d_after < d_before:
                        pre_second_move_toward += 1
                    elif d_after > d_before:
                        pre_second_move_away += 1
                if int(move_rec["unit_id"]) in hold_units_before:
                    moved_off_before_capture += 1

            obs_vec, _reward, terminated, truncated, info = env.step(chosen_gid)
            selected_global_id = _as_scalar(info.get("selected_global_id", None), default=None)
            if selected_global_id is not None and int(selected_global_id) != int(chosen_gid):
                raise RuntimeError(f"selected_global_id mismatch env={selected_global_id} chosen={chosen_gid}")

            post_city_count = int(_as_scalar(info.get("city_count", pre_city_count), default=pre_city_count))
            post_turn = int(_as_scalar(info.get("turn_count", pre_turn), default=pre_turn))
            if second_city_turn is None and post_city_count >= 2:
                second_city_turn = int(post_turn)
            if pending_wait_events:
                full_obs_after = uw.tribes_env.get_observation(full_visibility=True)
                villages_after = _visible_uncaptured_village_positions_from_env(env, full_obs_after)
                active_tribe_after = _safe_int(full_obs_after.get("activeTribeID", -1), default=-1)
                unresolved = []
                for event in pending_wait_events:
                    target_turn = int(event["target_turn"])
                    if int(post_turn) == target_turn:
                        unit_id = int(event["unit_id"])
                        unit_now = _unit_by_id(full_obs_after, unit_id)
                        same_unit_exists = unit_now is not None
                        if same_unit_exists:
                            unit_pos_now = (
                                _safe_int(unit_now.get("x", -1), -1),
                                _safe_int(unit_now.get("y", -1), -1),
                            )
                        else:
                            unit_pos_now = (-1, -1)
                        same_unit_same_tile = bool(same_unit_exists and unit_pos_now == tuple(event["village_coord"]))
                        village_still_neutral = tuple(event["village_coord"]) in villages_after
                        legal_now = _unit_action_legalities(env, unit_id)
                        capture_exists_for_unit = bool(legal_now["can_capture_proxy"])
                        unit_fresh_proxy = bool(legal_now["is_fresh_proxy"])

                        if capture_exists_for_unit:
                            waiting_capture_next_turn += 1
                        if same_unit_same_tile:
                            wait_same_unit_same_tile += 1
                        if village_still_neutral:
                            wait_village_still_neutral += 1
                        if unit_fresh_proxy:
                            wait_unit_fresh_next_turn += 1

                        if not capture_exists_for_unit:
                            if not same_unit_exists or not same_unit_same_tile:
                                wait_missing_capture_due_unit_moved_or_missing += 1
                                missing_reason = "unit_missing_or_not_on_same_tile"
                            elif not village_still_neutral:
                                wait_missing_capture_due_tile_not_neutral += 1
                                missing_reason = "tile_not_neutral_village_anymore"
                            elif not unit_fresh_proxy:
                                wait_missing_capture_due_not_fresh += 1
                                missing_reason = "unit_not_fresh_proxy"
                            elif unit_fresh_proxy and village_still_neutral and same_unit_same_tile:
                                wait_missing_capture_despite_fresh_neutral += 1
                                missing_reason = "capture_missing_despite_fresh_on_neutral_village"
                            else:
                                wait_missing_capture_unexplained += 1
                                missing_reason = "unexplained"
                        else:
                            missing_reason = "capture_available"

                        outcome = {
                            "turn": int(post_turn),
                            "active_tribe_id": int(active_tribe_after),
                            "same_unit_exists": bool(same_unit_exists),
                            "unit_pos": unit_pos_now,
                            "same_unit_same_tile": bool(same_unit_same_tile),
                            "village_still_neutral": bool(village_still_neutral),
                            "unit_fresh_proxy": bool(unit_fresh_proxy),
                            "unit_can_move_proxy": bool(legal_now["can_move_proxy"]),
                            "unit_can_act_proxy": bool(legal_now["can_act_proxy"]),
                            "legal_capture_actions": _capture_action_strs(env),
                            "capture_exists_for_unit": bool(capture_exists_for_unit),
                            "capture_missing_reason": missing_reason,
                            "tile_at_village": _tile_info(full_obs_after, tuple(event["village_coord"])),
                            "city_actor_at_village": _city_actor_by_coord(full_obs_after, tuple(event["village_coord"])),
                            "units_at_village": _units_on_coord(full_obs_after, tuple(event["village_coord"])),
                        }
                        if wait_trace_printed < int(args.audit_wait_cases):
                            _print_wait_trace_after(wait_trace_printed + 1, event, outcome)
                            wait_trace_printed += 1
                    elif int(post_turn) < target_turn:
                        unresolved.append(event)
                pending_wait_events = unresolved

            done = bool(terminated or truncated)
            safety_steps += 1
            if safety_steps > int(args.max_steps_per_episode):
                raise RuntimeError("Episode safety cap exceeded.")

        spt = float(_as_scalar(info.get("spt", np.nan), default=np.nan))
        city_count_t10 = float(_as_scalar(info.get("city_count", np.nan), default=np.nan))
        fog_t10 = float(_as_scalar(info.get("fog_tiles_cleared_total", np.nan), default=np.nan))
        unit_t10 = float(_as_scalar(info.get("unit_count", np.nan), default=np.nan))

        if second_city_turn is None:
            second_city_not_captured += 1

        terminal_spt.append(spt)
        terminal_city_count.append(city_count_t10)
        terminal_fog_cleared.append(fog_t10)
        terminal_unit_count.append(unit_t10)
        first_visible_turns.append(float(first_visible_turn if first_visible_turn is not None else -1))
        second_city_turns.append(float(second_city_turn if second_city_turn is not None else -1))
        first_stand_second_village_turns.append(
            float(first_stand_second_village_turn if first_stand_second_village_turn is not None else -1)
        )

        if args.progress_every > 0 and ((ep + 1) % int(args.progress_every) == 0):
            print(f"progress episodes={ep + 1}/{args.episodes}")

    env.close()

    spt_arr = np.asarray(terminal_spt, dtype=np.float64)
    village_arr = np.asarray(terminal_city_count, dtype=np.float64)
    fog_arr = np.asarray(terminal_fog_cleared, dtype=np.float64)
    unit_arr = np.asarray(terminal_unit_count, dtype=np.float64)

    summary = {
        "episodes": int(args.episodes),
        "mean_terminal_spt_t10": float(np.mean(spt_arr)),
        "median_terminal_spt_t10": float(np.median(spt_arr)),
        "p75_terminal_spt_t10": float(np.percentile(spt_arr, 75)),
        "p90_terminal_spt_t10": float(np.percentile(spt_arr, 90)),
        "percent_spt_ge_10": _pct(spt_arr, 10.0),
        "percent_spt_ge_15": _pct(spt_arr, 15.0),
        "percent_spt_ge_20": _pct(spt_arr, 20.0),
        "mean_village_count_t10": float(np.mean(village_arr)),
        "median_village_count_t10": float(np.median(village_arr)),
        "fail_rate_second_city_not_captured_by_t10": float(100.0 * second_city_not_captured / max(1, int(args.episodes))),
        "avg_turn_first_uncaptured_village_visible": _mean_nonneg(first_visible_turns),
        "avg_turn_second_city_captured": _mean_nonneg(second_city_turns),
        "mean_fog_tiles_cleared_t10": float(np.mean(fog_arr)),
        "mean_unit_count_t10": float(np.mean(unit_arr)),
        "percent_pre_second_city_moves_toward_visible_village": float(
            100.0 * pre_second_move_toward / max(1, pre_second_move_count)
        ),
        "percent_pre_second_city_moves_away_from_visible_village": float(
            100.0 * pre_second_move_away / max(1, pre_second_move_count)
        ),
        "percent_turns_capture_existed_but_not_selected": float(
            100.0 * capture_existed_not_selected / max(1, capture_existed_turns)
        ),
        "percent_end_turn_before_second_city_while_legal_move_existed": float(
            100.0 * pre_second_end_turn_while_move_exists / max(1, pre_second_turns_with_move)
        ),
        "percent_distance_zero_to_visible_village_but_no_capture_available": float(
            100.0 * distance_zero_visible_no_capture / max(1, pre_second_turns_total)
        ),
        "unit_on_neutral_village_capture_illegal_rate": float(
            100.0 * hold_condition_turns / max(1, pre_second_turns_total)
        ),
        "end_turn_while_unit_on_neutral_village_rate": float(
            100.0 * end_turn_while_holding / max(1, hold_condition_turns)
        ),
        "moved_off_neutral_village_before_capture_rate": float(
            100.0 * moved_off_before_capture / max(1, hold_condition_turns)
        ),
        "capture_next_turn_after_waiting_on_village_rate": float(
            100.0 * waiting_capture_next_turn / max(1, waiting_events)
        ),
        "wait_events_total": int(waiting_events),
        "capture_available_next_bardur_turn_rate": float(
            100.0 * waiting_capture_next_turn / max(1, waiting_events)
        ),
        "same_unit_same_tile_next_turn_rate": float(
            100.0 * wait_same_unit_same_tile / max(1, waiting_events)
        ),
        "village_still_neutral_next_turn_rate": float(
            100.0 * wait_village_still_neutral / max(1, waiting_events)
        ),
        "unit_fresh_next_turn_rate": float(
            100.0 * wait_unit_fresh_next_turn / max(1, waiting_events)
        ),
        "capture_missing_despite_unit_fresh_on_neutral_village_rate": float(
            100.0 * wait_missing_capture_despite_fresh_neutral / max(1, waiting_events)
        ),
        "capture_missing_because_unit_not_fresh_rate": float(
            100.0 * wait_missing_capture_due_not_fresh / max(1, waiting_events)
        ),
        "capture_missing_because_unit_moved_or_missing_rate": float(
            100.0 * wait_missing_capture_due_unit_moved_or_missing / max(1, waiting_events)
        ),
        "capture_missing_because_tile_not_neutral_village_rate": float(
            100.0 * wait_missing_capture_due_tile_not_neutral / max(1, waiting_events)
        ),
        "unexplained_missing_capture_rate": float(
            100.0 * wait_missing_capture_unexplained / max(1, waiting_events)
        ),
        "avg_turn_unit_first_stands_on_second_village": _mean_nonneg(first_stand_second_village_turns),
    }
    return summary


def _print_table(summary: Dict[str, float]) -> None:
    rows = [
        "episodes",
        "mean_terminal_spt_t10",
        "median_terminal_spt_t10",
        "p75_terminal_spt_t10",
        "p90_terminal_spt_t10",
        "percent_spt_ge_10",
        "percent_spt_ge_15",
        "percent_spt_ge_20",
        "mean_village_count_t10",
        "median_village_count_t10",
        "fail_rate_second_city_not_captured_by_t10",
        "avg_turn_first_uncaptured_village_visible",
        "avg_turn_second_city_captured",
        "mean_fog_tiles_cleared_t10",
        "mean_unit_count_t10",
        "percent_pre_second_city_moves_toward_visible_village",
        "percent_pre_second_city_moves_away_from_visible_village",
        "percent_turns_capture_existed_but_not_selected",
        "percent_end_turn_before_second_city_while_legal_move_existed",
        "percent_distance_zero_to_visible_village_but_no_capture_available",
        "unit_on_neutral_village_capture_illegal_rate",
        "end_turn_while_unit_on_neutral_village_rate",
        "moved_off_neutral_village_before_capture_rate",
        "capture_next_turn_after_waiting_on_village_rate",
        "wait_events_total",
        "capture_available_next_bardur_turn_rate",
        "same_unit_same_tile_next_turn_rate",
        "village_still_neutral_next_turn_rate",
        "unit_fresh_next_turn_rate",
        "capture_missing_despite_unit_fresh_on_neutral_village_rate",
        "capture_missing_because_unit_not_fresh_rate",
        "capture_missing_because_unit_moved_or_missing_rate",
        "capture_missing_because_tile_not_neutral_village_rate",
        "unexplained_missing_capture_rate",
        "avg_turn_unit_first_stands_on_second_village",
    ]
    print("\n=== no-fog runtime-village greedy evaluation ===")
    for key in rows:
        val = summary.get(key, None)
        if isinstance(val, float):
            shown = "nan" if np.isnan(val) else f"{val:.6f}"
        else:
            shown = str(val)
        print(f"{key:65s} {shown}")


def main():
    parser = argparse.ArgumentParser(description="No-fog runtime-visible village greedy diagnostic for Tribes-v0.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to evaluate.")
    parser.add_argument("--seed", type=int, default=1, help="Base seed.")
    parser.add_argument("--seed-per-episode", action="store_true", help="Use seed+episode for each reset.")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N episodes.")
    parser.add_argument("--max-steps-per-episode", type=int, default=256, help="Safety cap.")
    parser.add_argument("--audit-wait-cases", type=int, default=0, help="Print first N wait-event before/after traces.")
    args = parser.parse_args()

    summary = evaluate(args)
    _print_table(summary)


if __name__ == "__main__":
    main()
