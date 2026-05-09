import argparse
import json
import os
import sys
from collections import Counter
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

import py_rl.cleanrl.cleanrl.evaluate_no_fog_runtime_village_greedy as nofog


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


def _safe_int(v, default=-1):
    try:
        return int(v)
    except Exception:
        return int(default)


def _get_reflected_gs(env) -> Optional[Any]:
    try:
        jenv = env.unwrapped.tribes_env._env
        cls = jenv.getClass()
        fld = cls.getDeclaredField("gs")
        fld.setAccessible(True)
        return fld.get(jenv)
    except Exception:
        return None


def _java_get_int_field(obj, field_name: str) -> Optional[int]:
    if obj is None:
        return None
    try:
        cls = obj.getClass()
        while cls is not None:
            try:
                fld = cls.getDeclaredField(field_name)
                fld.setAccessible(True)
                return int(fld.getInt(obj))
            except Exception:
                cls = cls.getSuperclass()
    except Exception:
        return None
    return None


def _java_unit_info(gs, unit_id: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "exists": False,
        "unit_id": int(unit_id),
        "tribe_id": None,
        "x": None,
        "y": None,
        "is_fresh": None,
        "can_move": None,
        "can_attack": None,
        "is_finished": None,
        "status_raw": None,
    }
    if gs is None:
        return out
    try:
        u = gs.getActor(int(unit_id))
    except Exception:
        return out
    if u is None:
        return out
    out["exists"] = True
    try:
        out["tribe_id"] = int(u.getTribeId())
    except Exception:
        pass
    try:
        pos = u.getPosition()
        out["x"] = _java_get_int_field(pos, "x")
        out["y"] = _java_get_int_field(pos, "y")
    except Exception:
        pass
    try:
        out["is_fresh"] = bool(u.isFresh())
    except Exception:
        pass
    try:
        out["can_move"] = bool(u.canMove())
    except Exception:
        pass
    try:
        out["can_attack"] = bool(u.canAttack())
    except Exception:
        pass
    try:
        out["is_finished"] = bool(u.isFinished())
    except Exception:
        pass
    try:
        status = u.getStatus()
        out["status_raw"] = str(status)
    except Exception:
        pass
    return out


def _java_tile_info(gs, x: int, y: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "x": int(x),
        "y": int(y),
        "terrain_key": None,
        "terrain_name": None,
        "resource_key": None,
        "resource_name": None,
        "city_id_board": None,
        "unit_id_board": None,
        "city_in_borders_id": None,
        "city_in_borders_tribe": None,
        "unit_at_tile_id": None,
        "unit_at_tile_tribe": None,
    }
    if gs is None:
        return out
    try:
        b = gs.getBoard()
        t = b.getTerrainAt(int(x), int(y))
        out["terrain_key"] = int(t.getKey()) if t is not None else None
        out["terrain_name"] = str(t) if t is not None else None
        r = b.getResourceAt(int(x), int(y))
        out["resource_key"] = int(r.getKey()) if r is not None else -1
        out["resource_name"] = str(r) if r is not None else None
        out["city_id_board"] = int(b.getCityIdAt(int(x), int(y)))
        out["unit_id_board"] = int(b.getUnitIDAt(int(x), int(y)))
        c = b.getCityInBorders(int(x), int(y))
        if c is not None:
            out["city_in_borders_id"] = int(c.getActorId())
            out["city_in_borders_tribe"] = int(c.getTribeId())
        u = b.getUnitAt(int(x), int(y))
        if u is not None:
            out["unit_at_tile_id"] = int(u.getActorId())
            out["unit_at_tile_tribe"] = int(u.getTribeId())
    except Exception:
        pass
    return out


def _capture_actions_for_unit(raw_actions: List[Dict[str, Any]], unit_id: int) -> List[Dict[str, Any]]:
    out = []
    for a in raw_actions:
        if str(a.get("type", "")).upper() != "CAPTURE":
            continue
        uid = a.get("unit_id", None)
        if uid is None:
            m = nofog.re
        try:
            if uid is not None and int(uid) == int(unit_id):
                out.append(a)
        except Exception:
            continue
    return out


def _raw_mentions_unit_or_coord(raw_actions: List[Dict[str, Any]], unit_id: int, xy: Tuple[int, int]) -> List[Dict[str, Any]]:
    x, y = int(xy[0]), int(xy[1])
    out = []
    for a in raw_actions:
        repr_s = str(a.get("repr", ""))
        uid = a.get("unit_id", None)
        mention_uid = False
        try:
            mention_uid = uid is not None and int(uid) == int(unit_id)
        except Exception:
            mention_uid = False
        mention_xy = False
        for key_x, key_y in (("src_x", "src_y"), ("dst_x", "dst_y"), ("target_x", "target_y"), ("city_x", "city_y")):
            try:
                if key_x in a and key_y in a and int(a.get(key_x)) == x and int(a.get(key_y)) == y:
                    mention_xy = True
                    break
            except Exception:
                continue
        if mention_uid or mention_xy or (f"{x} : {y}" in repr_s):
            out.append(a)
    return out


def _detector_classification_details(uw, obs: Dict, coord: Tuple[int, int]) -> Dict[str, Any]:
    x, y = int(coord[0]), int(coord[1])
    villages = uw._get_visible_uncaptured_village_positions(obs)
    board = obs.get("board", {}) if isinstance(obs, dict) else {}
    terrain = board.get("terrain", []) if isinstance(board, dict) else []
    city_ids = board.get("cityID", []) if isinstance(board, dict) else []
    t_val = None
    c_val = None
    try:
        t_val = int(terrain[x][y])
    except Exception:
        pass
    try:
        c_val = int(city_ids[x][y])
    except Exception:
        pass
    city_actor = nofog._city_actor_by_coord(obs, (x, y))
    return {
        "coord": (x, y),
        "in_detector_set": (x, y) in villages,
        "terrain_raw": t_val,
        "cityID_raw": c_val,
        "city_actor_present": city_actor is not None,
        "city_actor": city_actor,
        "detector_reason": (
            f"terrain==4({t_val==4}), cityID==-1({c_val==-1 if c_val is not None else False}), "
            f"coord_not_in_city_actor_coords({city_actor is None})"
        ),
    }


def _replay_filter_pipeline(uw, raw_actions: List[Dict[str, Any]], obs: Dict) -> Dict[str, Any]:
    removal_reasons: Dict[int, List[str]] = {}

    def _remove(idx: int, reason: str):
        removal_reasons.setdefault(int(idx), []).append(str(reason))

    stage0 = []
    for idx, a in enumerate(raw_actions):
        a_type = str(a.get("type", "")).upper()
        if a_type not in uw.ALLOWED_ACTION_TYPES:
            _remove(idx, "disallowed_action_type")
            continue
        if a_type == "MOVE" and not uw._is_move_destination_within_board(a, obs):
            _remove(idx, "move_destination_out_of_board")
            continue
        if a_type == "RESOURCE_GATHERING":
            if not uw._is_resource_gather_legal_for_upgrade(a, raw_actions, obs):
                _remove(idx, "resource_gather_not_legal_for_upgrade")
                continue
        stage0.append(idx)

    stage1 = list(stage0)
    forced_village_captures = []
    forced_village_moves = []
    forced_progress_moves = []
    if uw._get_city_count(obs) < 2 and stage1:
        for idx in stage1:
            a = raw_actions[idx]
            if a.get("type") != "CAPTURE":
                continue
            if uw._is_capture_of_village(a, obs):
                forced_village_captures.append(idx)

        if forced_village_captures:
            frozen_units = set()
            for idx in forced_village_captures:
                uid = uw._parse_unit_id_from_action_repr(str(raw_actions[idx].get("repr", "")))
                if uid is not None:
                    frozen_units.add(int(uid))
            new_stage = []
            for idx in stage1:
                a = raw_actions[idx]
                a_type = a.get("type")
                if a_type not in ("MOVE", "CAPTURE"):
                    new_stage.append(idx)
                    continue
                uid = uw._parse_unit_id_from_action_repr(str(a.get("repr", "")))
                if uid is None or int(uid) not in frozen_units:
                    new_stage.append(idx)
                else:
                    _remove(idx, "forced_village_capture_freeze_other_actions_for_unit")
            if new_stage:
                stage1 = new_stage
            else:
                end_idx = next((i for i, a in enumerate(raw_actions) if a.get("type") == "END_TURN"), None)
                stage1 = [end_idx] if end_idx is not None else []
                for idx in stage0:
                    if idx not in stage1:
                        _remove(idx, "forced_village_capture_fallback_to_end_turn")
        else:
            visible_village_positions = uw._get_visible_uncaptured_village_positions(obs)
            for idx in stage1:
                a = raw_actions[idx]
                if a.get("type") != "MOVE":
                    continue
                if uw._is_move_to_visible_uncaptured_village(a, obs):
                    forced_village_moves.append(idx)
            if forced_village_moves:
                new_set = set(forced_village_moves)
                for idx in stage1:
                    if idx not in new_set:
                        _remove(idx, "forced_village_moves_only")
                stage1 = list(forced_village_moves)
            elif visible_village_positions:
                closest_unit_id, _closest_pos, _closest_dist = uw._closest_owned_unit_to_targets(obs, visible_village_positions)
                if closest_unit_id is not None:
                    for idx in stage1:
                        a = raw_actions[idx]
                        if str(a.get("type", "")).upper() != "MOVE":
                            continue
                        if uw._is_move_reducing_distance_to_targets(a, obs, closest_unit_id, visible_village_positions):
                            forced_progress_moves.append(idx)
                if forced_progress_moves:
                    new_set = set(forced_progress_moves)
                    for idx in stage1:
                        if idx not in new_set:
                            _remove(idx, "forced_progress_moves_only")
                    stage1 = list(forced_progress_moves)

    stage2 = uw._apply_t1_t2_unit2_backtrack_mask(list(stage1), raw_actions, obs)
    if len(stage2) < len(stage1):
        s2 = set(stage2)
        for idx in stage1:
            if idx not in s2:
                _remove(idx, "t1_t2_unit2_backtrack_mask")

    return {
        "stage0_allowed": stage0,
        "stage1_after_guardrail": stage1,
        "stage2_final": stage2,
        "forced_village_captures": forced_village_captures,
        "forced_village_moves": forced_village_moves,
        "forced_progress_moves": forced_progress_moves,
        "removal_reasons": removal_reasons,
    }


def _print_case_trace(case_i: int, payload: Dict[str, Any]) -> None:
    print(f"\n========== CASE #{case_i} ==========")
    print("1) Runtime state")
    print(json.dumps(payload["runtime_state"], indent=2, sort_keys=True, default=str))
    print("2) Village detector")
    print(json.dumps(payload["village_detector"], indent=2, sort_keys=True, default=str))
    print("3) Java raw legal actions BEFORE wrapper filtering")
    print(json.dumps(payload["raw_actions_block"], indent=2, sort_keys=True, default=str))
    print("4) Wrapper filtered legal actions AFTER _filter_allowed_raw_indices")
    print(json.dumps(payload["filtered_actions_block"], indent=2, sort_keys=True, default=str))
    print("5) Action mask/mapping")
    print(json.dumps(payload["mask_mapping_block"], indent=2, sort_keys=True, default=str))
    print("6) Java capture condition audit")
    print(json.dumps(payload["java_conditions"], indent=2, sort_keys=True, default=str))
    print(f"exact_cause={payload['exact_cause']}")


def run(args):
    os.environ.setdefault("POLYVISION_LEVEL_POOL_GLOB", "levels/phase1_pool/*.csv")
    os.environ.setdefault("POLYVISION_LEVEL_SELECTION_MODE", "round_robin")

    env = gym.make("Tribes-v0")
    uw = env.unwrapped

    cases_total = 0
    printed_cases = 0
    raw_capture_exists = 0
    raw_capture_missing = 0
    raw_capture_exists_but_filtered = 0
    filtered_capture_exists_but_missing_from_mask = 0
    missing_due_to_python_detector_disagree = 0
    exact_causes = Counter()
    missing_by_java_condition = Counter()

    pending_wait_events: List[Dict[str, Any]] = []

    obs_vec, info = env.reset(seed=int(args.seed))
    for ep in range(int(args.episodes)):
        if ep > 0:
            obs_vec, info = env.reset(seed=(int(args.seed) + ep) if args.seed_per_episode else None)

        done = False
        safety_steps = 0
        while not done:
            full_obs_before = uw.tribes_env.get_observation(full_visibility=True)
            ids, valid, feats = nofog._extract_legal_tensors(info)
            chosen_slot, _choice_diag = nofog._choose_slot(env, info, ids, valid, feats, full_obs_before)
            chosen_gid = int(ids[int(chosen_slot)])
            chosen_feat = feats[int(chosen_slot)]
            chosen_is_end_turn = float(chosen_feat[nofog.IDX_IS_END_TURN]) > 0.5
            pre_turn = int(_as_scalar(info.get("turn_count", 0), default=0))
            hold_units_before = nofog._units_on_visible_neutral_village_capture_illegal(env, full_obs_before)

            if len(hold_units_before) > 0 and chosen_is_end_turn:
                tracked_unit_id = int(sorted(list(hold_units_before))[0])
                tracked_unit = nofog._unit_by_id(full_obs_before, tracked_unit_id) or {}
                tracked_pos = (
                    _safe_int(tracked_unit.get("x", -1), -1),
                    _safe_int(tracked_unit.get("y", -1), -1),
                )
                pending_wait_events.append(
                    {
                        "episode": int(ep + 1),
                        "pre_turn": int(pre_turn),
                        "target_turn": int(pre_turn + 1),
                        "unit_id": int(tracked_unit_id),
                        "unit_pos": tracked_pos,
                    }
                )

            obs_vec, _reward, terminated, truncated, info = env.step(chosen_gid)
            post_turn = int(_as_scalar(info.get("turn_count", pre_turn), default=pre_turn))

            if pending_wait_events:
                full_obs_after = uw.tribes_env.get_observation(full_visibility=True)
                villages_after = nofog._visible_uncaptured_village_positions_from_env(env, full_obs_after)
                active_tribe_after = _safe_int(full_obs_after.get("activeTribeID", -1), -1)
                raw_actions = uw.tribes_env.list_actions()
                filter_replay = _replay_filter_pipeline(uw, raw_actions, full_obs_after)
                filtered_idxs = list(filter_replay["stage2_final"])
                mapping = getattr(uw, "_current_legal_id_to_raw_index", {}) or {}
                mapping_raw_idxs = set(int(v) for v in mapping.values())

                unresolved = []
                for ev in pending_wait_events:
                    if int(post_turn) != int(ev["target_turn"]):
                        if int(post_turn) < int(ev["target_turn"]):
                            unresolved.append(ev)
                        continue

                    unit_id = int(ev["unit_id"])
                    unit_now = nofog._unit_by_id(full_obs_after, unit_id)
                    same_unit_exists = unit_now is not None
                    if same_unit_exists:
                        unit_pos_now = (
                            _safe_int(unit_now.get("x", -1), -1),
                            _safe_int(unit_now.get("y", -1), -1),
                        )
                    else:
                        unit_pos_now = (-1, -1)
                    same_unit_same_tile = bool(same_unit_exists and unit_pos_now == tuple(ev["unit_pos"]))
                    village_still_neutral = tuple(ev["unit_pos"]) in villages_after
                    legal_now = nofog._unit_action_legalities(env, unit_id)
                    unit_fresh_proxy = bool(legal_now["is_fresh_proxy"])
                    no_capture_for_unit = not bool(legal_now["can_capture_proxy"])

                    if not (same_unit_same_tile and village_still_neutral and unit_fresh_proxy and no_capture_for_unit):
                        continue

                    cases_total += 1
                    gs = _get_reflected_gs(env)
                    j_u = _java_unit_info(gs, unit_id)
                    j_t = _java_tile_info(gs, unit_pos_now[0], unit_pos_now[1])
                    detector = _detector_classification_details(uw, full_obs_after, unit_pos_now)

                    raw_caps_for_unit = []
                    raw_caps_any = []
                    for a in raw_actions:
                        if str(a.get("type", "")).upper() == "CAPTURE":
                            raw_caps_any.append(a)
                            try:
                                if int(a.get("unit_id", -99999)) == int(unit_id):
                                    raw_caps_for_unit.append(a)
                            except Exception:
                                pass
                    raw_has_capture = len(raw_caps_for_unit) > 0
                    if raw_has_capture:
                        raw_capture_exists += 1
                    else:
                        raw_capture_missing += 1

                    filtered_caps_for_unit = []
                    filtered_caps_any = []
                    for idx in filtered_idxs:
                        a = raw_actions[int(idx)]
                        if str(a.get("type", "")).upper() == "CAPTURE":
                            filtered_caps_any.append({"raw_idx": int(idx), "action": a})
                            try:
                                if int(a.get("unit_id", -99999)) == int(unit_id):
                                    filtered_caps_for_unit.append({"raw_idx": int(idx), "action": a})
                            except Exception:
                                pass

                    filtered_has_capture = len(filtered_caps_for_unit) > 0
                    if raw_has_capture and not filtered_has_capture:
                        raw_capture_exists_but_filtered += 1

                    capture_gids = []
                    for gid, ridx in mapping.items():
                        try:
                            a = raw_actions[int(ridx)]
                        except Exception:
                            continue
                        if str(a.get("type", "")).upper() == "CAPTURE":
                            capture_gids.append(int(gid))

                    mask_capture_for_unit = False
                    for gid, ridx in mapping.items():
                        try:
                            a = raw_actions[int(ridx)]
                        except Exception:
                            continue
                        if str(a.get("type", "")).upper() != "CAPTURE":
                            continue
                        try:
                            if int(a.get("unit_id", -99999)) == int(unit_id):
                                mask_capture_for_unit = True
                                break
                        except Exception:
                            continue
                    if filtered_has_capture and not mask_capture_for_unit:
                        filtered_capture_exists_but_missing_from_mask += 1

                    raw_capture_ridxs_for_unit = set()
                    for i, a in enumerate(raw_actions):
                        if str(a.get("type", "")).upper() == "CAPTURE":
                            try:
                                if int(a.get("unit_id", -99999)) == int(unit_id):
                                    raw_capture_ridxs_for_unit.add(int(i))
                            except Exception:
                                pass
                    removed_capture_reasons = []
                    for ridx in sorted(raw_capture_ridxs_for_unit):
                        removed_capture_reasons.extend(filter_replay["removal_reasons"].get(int(ridx), []))

                    # Java capture conditions for VILLAGE path.
                    cond_unit_exists = bool(j_u.get("exists"))
                    cond_unit_active_tribe = bool(
                        cond_unit_exists
                        and (j_u.get("tribe_id") is not None)
                        and int(j_u.get("tribe_id")) == int(active_tribe_after)
                    )
                    cond_unit_fresh = bool(j_u.get("is_fresh") is True)
                    cond_tile_is_village = bool(j_t.get("terrain_name") == "VILLAGE" or j_t.get("terrain_key") == 4)
                    cond_unit_at_tile = bool(j_t.get("unit_at_tile_id") is not None and int(j_t.get("unit_at_tile_id")) == int(unit_id))
                    cond_tile_not_city = bool((j_t.get("city_in_borders_id") is None) and int(j_t.get("city_id_board", -1)) == -1)

                    java_pass = bool(
                        cond_unit_exists
                        and cond_unit_active_tribe
                        and cond_unit_fresh
                        and cond_tile_is_village
                        and cond_unit_at_tile
                    )

                    python_disagrees = bool(detector["in_detector_set"] and (not cond_tile_is_village or not cond_tile_not_city))
                    if python_disagrees:
                        missing_due_to_python_detector_disagree += 1

                    exact_cause = "unknown"
                    if raw_has_capture and not filtered_has_capture:
                        cause = "raw_capture_removed_by_filter"
                        if removed_capture_reasons:
                            cause += ":" + "|".join(sorted(set(removed_capture_reasons)))
                        exact_cause = cause
                    elif raw_has_capture and filtered_has_capture and not mask_capture_for_unit:
                        exact_cause = "capture_dropped_in_mask_or_mapping"
                    elif not raw_has_capture:
                        if not cond_unit_exists:
                            exact_cause = "java_missing_unit_actor"
                            missing_by_java_condition["unit_missing"] += 1
                        elif not cond_unit_active_tribe:
                            exact_cause = "java_unit_not_active_tribe"
                            missing_by_java_condition["unit_not_active_tribe"] += 1
                        elif not cond_unit_fresh:
                            exact_cause = "java_unit_not_fresh"
                            missing_by_java_condition["unit_not_fresh"] += 1
                        elif not cond_tile_is_village:
                            exact_cause = "java_tile_not_village"
                            missing_by_java_condition["tile_not_village"] += 1
                        elif not cond_unit_at_tile:
                            exact_cause = "java_unit_not_at_tile"
                            missing_by_java_condition["unit_not_at_tile"] += 1
                        elif not java_pass:
                            exact_cause = "java_hidden_condition_failed"
                            missing_by_java_condition["hidden_condition"] += 1
                        elif python_disagrees:
                            exact_cause = "python_detector_disagrees_with_java"
                            missing_by_java_condition["python_java_disagree"] += 1
                        else:
                            exact_cause = "raw_missing_unknown_despite_java_conditions_pass"
                            missing_by_java_condition["unknown"] += 1

                    exact_causes[exact_cause] += 1

                    if printed_cases < int(args.debug_cases):
                        runtime_state = {
                            "episode": int(ev["episode"]),
                            "turn": int(post_turn),
                            "active_tribe": int(active_tribe_after),
                            "unit_id": int(unit_id),
                            "unit_owner_python": _safe_int(unit_now.get("tribeId", -1), -1) if isinstance(unit_now, dict) else None,
                            "unit_position_python": tuple(unit_pos_now),
                            "unit_position_java": (j_u.get("x"), j_u.get("y")),
                            "unit_freshness_java": {
                                "isFresh": j_u.get("is_fresh"),
                                "canMove": j_u.get("can_move"),
                                "canAttack": j_u.get("can_attack"),
                                "isFinished": j_u.get("is_finished"),
                                "status": j_u.get("status_raw"),
                            },
                            "unit_freshness_proxy": {
                                "is_fresh_proxy": bool(unit_fresh_proxy),
                                "canMove_proxy": bool(legal_now["can_move_proxy"]),
                                "canAct_proxy": bool(legal_now["can_act_proxy"]),
                            },
                            "tile_at_unit_python": nofog._tile_info(full_obs_after, unit_pos_now),
                            "tile_at_unit_java": j_t,
                            "city_actor_at_tile_python": nofog._city_actor_by_coord(full_obs_after, unit_pos_now),
                        }
                        raw_actions_block = {
                            "raw_actions_total": int(len(raw_actions)),
                            "raw_capture_actions_count": int(len(raw_caps_any)),
                            "raw_capture_actions_repr": [str(a.get("repr", a)) for a in raw_caps_any],
                            "raw_actions_mentioning_unit_or_coord": _raw_mentions_unit_or_coord(raw_actions, unit_id, unit_pos_now),
                            "raw_actions_all": raw_actions,
                        }
                        filtered_actions_block = {
                            "filtered_actions_total": int(len(filtered_idxs)),
                            "filtered_capture_actions_count": int(len(filtered_caps_any)),
                            "filtered_capture_actions_repr": [str(x["action"].get("repr", x["action"])) for x in filtered_caps_any],
                            "raw_capture_for_unit_exists": bool(raw_has_capture),
                            "filtered_capture_for_unit_exists": bool(filtered_has_capture),
                            "raw_capture_removed_reasons": sorted(set(removed_capture_reasons)),
                            "filter_replay": {
                                "forced_village_captures": filter_replay["forced_village_captures"],
                                "forced_village_moves": filter_replay["forced_village_moves"],
                                "forced_progress_moves": filter_replay["forced_progress_moves"],
                            },
                        }
                        ids = np.asarray(info.get("legal_global_ids_padded", []), dtype=np.int64).reshape(-1)
                        vmask = np.asarray(info.get("legal_action_valid_mask", []), dtype=bool).reshape(-1)
                        legal_slot_capture_gids = []
                        feats = np.asarray(info.get("legal_action_features_padded", []), dtype=np.float32)
                        if feats.ndim == 3:
                            feats = feats[0]
                        if feats.ndim == 2 and ids.size == vmask.size:
                            for s in np.where(vmask)[0]:
                                if int(s) < feats.shape[0] and float(feats[int(s), nofog.IDX_IS_CAPTURE]) > 0.5:
                                    legal_slot_capture_gids.append(int(ids[int(s)]))
                        mask_mapping_block = {
                            "capture_global_ids_from_mapping": sorted(capture_gids),
                            "capture_global_ids_in_legal_slots": sorted(set(legal_slot_capture_gids)),
                            "mapping_has_capture_for_unit": bool(mask_capture_for_unit),
                            "mapping_size": int(len(mapping)),
                            "mapping_capture_count_total": int(len(capture_gids)),
                            "diagnostic_failure_stage": (
                                "raw_generation"
                                if not raw_has_capture
                                else "wrapper_filtering"
                                if raw_has_capture and not filtered_has_capture
                                else "mask_or_slot_construction"
                                if filtered_has_capture and not mask_capture_for_unit
                                else "unknown"
                            ),
                        }
                        java_conditions = {
                            "unit_exists": cond_unit_exists,
                            "unit_owner_is_active_tribe": cond_unit_active_tribe,
                            "unit_is_fresh": cond_unit_fresh,
                            "tile_is_village": cond_tile_is_village,
                            "unit_at_tile_matches": cond_unit_at_tile,
                            "tile_not_city_actor_proxy": cond_tile_not_city,
                            "java_village_capture_conditions_passed": java_pass,
                            "python_detector_disagrees_with_java": python_disagrees,
                        }

                        payload = {
                            "runtime_state": runtime_state,
                            "village_detector": detector,
                            "raw_actions_block": raw_actions_block,
                            "filtered_actions_block": filtered_actions_block,
                            "mask_mapping_block": mask_mapping_block,
                            "java_conditions": java_conditions,
                            "exact_cause": exact_cause,
                        }
                        _print_case_trace(printed_cases + 1, payload)
                        printed_cases += 1

                pending_wait_events = unresolved

            done = bool(terminated or truncated)
            safety_steps += 1
            if safety_steps > int(args.max_steps_per_episode):
                raise RuntimeError("Episode safety cap exceeded.")

        if args.progress_every > 0 and ((ep + 1) % int(args.progress_every) == 0):
            print(f"progress episodes={ep + 1}/{args.episodes}")

    env.close()

    summary = {
        "cases_total": int(cases_total),
        "raw_capture_exists_rate": float(100.0 * raw_capture_exists / max(1, cases_total)),
        "raw_capture_missing_rate": float(100.0 * raw_capture_missing / max(1, cases_total)),
        "raw_capture_exists_but_filtered_rate": float(100.0 * raw_capture_exists_but_filtered / max(1, cases_total)),
        "filtered_capture_exists_but_missing_from_mask_rate": float(
            100.0 * filtered_capture_exists_but_missing_from_mask / max(1, cases_total)
        ),
        "missing_due_to_python_detector_disagreeing_with_java_rate": float(
            100.0 * missing_due_to_python_detector_disagree / max(1, cases_total)
        ),
        "missing_due_to_java_condition_rate_by_condition": {
            k: float(100.0 * v / max(1, cases_total)) for k, v in sorted(missing_by_java_condition.items())
        },
        "most_common_exact_cause": exact_causes.most_common(1)[0][0] if len(exact_causes) > 0 else "none",
        "exact_cause_histogram": dict(exact_causes),
    }

    print("\n=== capture legality pipeline summary ===")
    for k in (
        "cases_total",
        "raw_capture_exists_rate",
        "raw_capture_missing_rate",
        "raw_capture_exists_but_filtered_rate",
        "filtered_capture_exists_but_missing_from_mask_rate",
        "missing_due_to_python_detector_disagreeing_with_java_rate",
        "most_common_exact_cause",
    ):
        print(f"{k:58s} {summary[k]}")
    print("missing_due_to_java_condition_rate_by_condition")
    print(json.dumps(summary["missing_due_to_java_condition_rate_by_condition"], indent=2, sort_keys=True))
    print("exact_cause_histogram")
    print(json.dumps(summary["exact_cause_histogram"], indent=2, sort_keys=True))

    # Top-level interpretation bucket A-E.
    dominant = summary["most_common_exact_cause"]
    if summary["missing_due_to_python_detector_disagreeing_with_java_rate"] >= 50.0:
        verdict = "C"
    elif "raw_capture_removed_by_filter" in dominant:
        verdict = "B"
    elif "mask_or" in dominant or "dropped_in_mask" in dominant:
        verdict = "D"
    elif "disagrees" in dominant:
        verdict = "E"
    elif "tile_not_village" in dominant or "unit_not_fresh" in dominant or "java_" in dominant or "raw_missing" in dominant:
        verdict = "A"
    else:
        verdict = "C"
    print(f"verdict_bucket={verdict}")


def main():
    parser = argparse.ArgumentParser(description="Audit where CAPTURE disappears for unit-on-neutral-village cases.")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes to run.")
    parser.add_argument("--seed", type=int, default=1, help="Base seed.")
    parser.add_argument("--seed-per-episode", action="store_true", help="Use seed+episode for each reset.")
    parser.add_argument("--debug-cases", type=int, default=100, help="Print first N full traces.")
    parser.add_argument("--progress-every", type=int, default=50, help="Progress print cadence.")
    parser.add_argument("--max-steps-per-episode", type=int, default=256, help="Safety cap.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
