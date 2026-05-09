import argparse
import os
import re
import sys
from dataclasses import dataclass
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


IDX_IS_MOVE = 0
IDX_NEW_REVEAL_NORM = 1
IDX_IS_END_TURN = 12
IDX_IS_CAPTURE = 13
IDX_IS_TRAIN_OR_SPAWN = 14
IDX_IS_RESOURCE_GATHERING = 16
IDX_IS_LEVEL_UP = 17

TRANSFORM_IDENTITY = "identity"
TRANSFORM_SWAPPED = "swapped"


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


def _parse_hidden_villages_from_level_csv(level_path: str) -> Set[Tuple[int, int]]:
    villages: Set[Tuple[int, int]] = set()
    with open(level_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for y, line in enumerate(lines):
        cells = [c.strip() for c in line.split(",")]
        for x, token in enumerate(cells):
            terrain_char = token.split(":", 1)[0].strip().lower() if token else ""
            if terrain_char == "v":
                villages.add((int(x), int(y)))
    return villages


def _transform_coord(xy: Tuple[int, int], mode: str) -> Tuple[int, int]:
    if mode == TRANSFORM_IDENTITY:
        return (int(xy[0]), int(xy[1]))
    if mode == TRANSFORM_SWAPPED:
        return (int(xy[1]), int(xy[0]))
    raise ValueError(f"Unsupported transform mode: {mode}")


def _inverse_transform_coord(xy: Tuple[int, int], mode: str) -> Tuple[int, int]:
    # identity and swapped are self-inverse.
    return _transform_coord(xy, mode)


def _apply_transform(villages_raw: Set[Tuple[int, int]], mode: str) -> Set[Tuple[int, int]]:
    return {_transform_coord(v, mode) for v in villages_raw}


def _owned_units(obs: Dict) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
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
            utype = int(unit.get("type", -1))
        except Exception:
            continue
        if ux < 0 or uy < 0:
            continue
        out.append((uid, ux, uy, utype))
    return out


def _owned_city_positions(obs: Dict) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    cities = obs.get("city", {})
    if not isinstance(cities, dict):
        return out
    for city in cities.values():
        if not isinstance(city, dict):
            continue
        try:
            if int(city.get("tribeID", -1)) != 0:
                continue
            cx = int(city.get("x", -1))
            cy = int(city.get("y", -1))
        except Exception:
            continue
        if cx >= 0 and cy >= 0:
            out.add((cx, cy))
    return out


def _visible_uncaptured_villages(obs: Dict) -> Set[Tuple[int, int]]:
    board = obs.get("board", {})
    terrain = board.get("terrain", []) if isinstance(board, dict) else []
    city_ids = board.get("cityID", []) if isinstance(board, dict) else []
    out: Set[Tuple[int, int]] = set()
    if not terrain or not city_ids:
        return out
    for y in range(len(terrain)):
        row_t = terrain[y]
        row_c = city_ids[y] if y < len(city_ids) else []
        for x in range(len(row_t)):
            try:
                t_val = int(row_t[x])
                c_val = int(row_c[x]) if x < len(row_c) else -1
            except Exception:
                continue
            if t_val == 4 and c_val == -1:
                out.add((int(x), int(y)))
    return out


def _chebyshev(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _nearest_unit_to_any_village(
    units: List[Tuple[int, int, int, int]],
    villages: Set[Tuple[int, int]],
) -> Tuple[Optional[int], Optional[Tuple[int, int]], Optional[int], Set[Tuple[int, int]]]:
    best_uid = None
    best_pos = None
    best_dist = None
    best_targets: Set[Tuple[int, int]] = set()
    if not units or not villages:
        return best_uid, best_pos, best_dist, best_targets
    for uid, ux, uy, _ in units:
        for v in villages:
            d = _chebyshev((ux, uy), v)
            if best_dist is None or d < best_dist:
                best_dist = int(d)
                best_uid = int(uid)
                best_pos = (int(ux), int(uy))
                best_targets = {v}
            elif d == best_dist:
                best_targets.add(v)
    return best_uid, best_pos, best_dist, best_targets


def _estimate_clear_range(obs: Dict, unit_id: int, dst_x: int, dst_y: int) -> int:
    clear_range = 1
    units = obs.get("unit", {})
    unit = units.get(str(int(unit_id)), {}) if isinstance(units, dict) else {}
    unit_type = None
    try:
        unit_type = int(unit.get("type", -1))
    except Exception:
        unit_type = None
    if unit_type == 10:
        clear_range += 1
    else:
        board = obs.get("board", {})
        terrain = board.get("terrain", []) if isinstance(board, dict) else []
        try:
            if int(terrain[int(dst_y)][int(dst_x)]) == 3:
                clear_range += 1
        except Exception:
            pass
    return int(clear_range)


def _visible_from_hidden(obs: Dict, villages: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    vis: Set[Tuple[int, int]] = set()
    board = obs.get("board", {})
    terrain = board.get("terrain", []) if isinstance(board, dict) else []
    if not isinstance(terrain, list) or not terrain:
        return vis
    h = len(terrain)
    w = max((len(r) for r in terrain if isinstance(r, (list, tuple))), default=0)
    for vx, vy in villages:
        if vx < 0 or vy < 0 or vx >= w or vy >= h:
            continue
        try:
            if int(terrain[vy][vx]) != 7:
                vis.add((vx, vy))
        except Exception:
            continue
    return vis


def _tile_info(obs: Dict, xy: Optional[Tuple[int, int]]) -> Dict:
    if xy is None:
        return {"xy": None, "terrain": None, "city_id": None}
    x, y = int(xy[0]), int(xy[1])
    board = obs.get("board", {})
    terrain = board.get("terrain", []) if isinstance(board, dict) else []
    city_id = board.get("cityID", []) if isinstance(board, dict) else []
    t = None
    c = None
    try:
        t = int(terrain[y][x])
    except Exception:
        t = None
    try:
        c = int(city_id[y][x])
    except Exception:
        c = None
    return {"xy": (x, y), "terrain": t, "city_id": c}


def _parse_unit_id_from_repr(action_repr: str) -> Optional[int]:
    if not isinstance(action_repr, str):
        return None
    m = re.search(r"by unit\s+(-?\d+)", action_repr, flags=re.IGNORECASE)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _unit_pos_by_id(obs: Dict, unit_id: int) -> Optional[Tuple[int, int]]:
    units = obs.get("unit", {})
    if not isinstance(units, dict):
        return None
    u = units.get(str(int(unit_id)), None)
    if not isinstance(u, dict):
        return None
    try:
        return int(u.get("x", -1)), int(u.get("y", -1))
    except Exception:
        return None


def _extract_capture_target_coord(action: Dict, obs: Dict) -> Optional[Tuple[int, int]]:
    tx = action.get("target_x", None)
    ty = action.get("target_y", None)
    try:
        if tx is not None and ty is not None:
            return int(tx), int(ty)
    except Exception:
        pass
    sx = action.get("src_x", None)
    sy = action.get("src_y", None)
    try:
        if sx is not None and sy is not None:
            return int(sx), int(sy)
    except Exception:
        pass
    uid = action.get("unit_id", None)
    try:
        if uid is not None:
            pos = _unit_pos_by_id(obs, int(uid))
            if pos is not None:
                return pos
    except Exception:
        pass
    uid_from_repr = _parse_unit_id_from_repr(str(action.get("repr", "")))
    if uid_from_repr is not None:
        pos = _unit_pos_by_id(obs, uid_from_repr)
        if pos is not None:
            return pos
    return None


@dataclass
class CoordEvidence:
    visible_total: int = 0
    visible_identity_matches: int = 0
    visible_swapped_matches: int = 0
    visible_neither: int = 0
    capture_total: int = 0
    capture_identity_matches: int = 0
    capture_swapped_matches: int = 0


def _update_coord_evidence(
    evidence: CoordEvidence,
    visible_runtime: Set[Tuple[int, int]],
    hidden_raw: Set[Tuple[int, int]],
    capture_targets: List[Tuple[int, int]],
) -> None:
    hidden_id = _apply_transform(hidden_raw, TRANSFORM_IDENTITY)
    hidden_sw = _apply_transform(hidden_raw, TRANSFORM_SWAPPED)

    for v in visible_runtime:
        evidence.visible_total += 1
        m_id = v in hidden_id
        m_sw = v in hidden_sw
        if m_id:
            evidence.visible_identity_matches += 1
        if m_sw:
            evidence.visible_swapped_matches += 1
        if not m_id and not m_sw:
            evidence.visible_neither += 1

    for c in capture_targets:
        evidence.capture_total += 1
        if c in hidden_id:
            evidence.capture_identity_matches += 1
        if c in hidden_sw:
            evidence.capture_swapped_matches += 1


def _choose_hidden_move_record(
    obs: Dict,
    move_records: List[Dict],
    hidden_uncaptured_villages: Set[Tuple[int, int]],
    preferred_unit_id: Optional[int],
) -> Optional[Dict]:
    if not hidden_uncaptured_villages or not move_records:
        return None
    visible_now = _visible_from_hidden(obs, hidden_uncaptured_villages)
    candidates = []
    for rec in move_records:
        uid = rec["unit_id"]
        dst = rec["dst"]
        if uid is None or dst is None:
            continue
        if preferred_unit_id is not None and int(uid) != int(preferred_unit_id):
            continue
        min_dist_after = min(_chebyshev(dst, vxy) for vxy in hidden_uncaptured_villages)
        onto = 1 if dst in hidden_uncaptured_villages else 0
        clear_range = _estimate_clear_range(obs, int(uid), int(dst[0]), int(dst[1]))
        reveal = 0
        for v in hidden_uncaptured_villages:
            if v in visible_now:
                continue
            if _chebyshev(dst, v) <= int(clear_range):
                reveal = 1
                break
        candidates.append((int(rec["slot"]), int(min_dist_after), int(reveal), int(onto), float(rec["feat"][IDX_NEW_REVEAL_NORM]), rec))
    if not candidates and preferred_unit_id is not None:
        return _choose_hidden_move_record(obs, move_records, hidden_uncaptured_villages, preferred_unit_id=None)
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[1], -t[2], -t[3], -t[4], t[0]))
    return candidates[0][5]


def _build_legal_records(env, obs: Dict, ids: np.ndarray, valid: np.ndarray, feats: np.ndarray) -> List[Dict]:
    valid_slots = np.where(valid)[0]
    uw = env.unwrapped
    legal_actions = getattr(uw, "_current_legal_actions", [])
    legal_id_to_raw = getattr(uw, "_current_legal_id_to_raw_index", {})
    out: List[Dict] = []
    for slot in valid_slots:
        s = int(slot)
        gid = int(ids[s])
        raw_idx = legal_id_to_raw.get(gid, None)
        action = legal_actions[int(raw_idx)] if raw_idx is not None and 0 <= int(raw_idx) < len(legal_actions) else {}
        a_type = str(action.get("type", "UNKNOWN")).upper()
        rec = {
            "slot": s,
            "gid": gid,
            "raw_idx": raw_idx,
            "action": action,
            "type": a_type,
            "repr": str(action.get("repr", "")),
            "feat": feats[s],
            "unit_id": None,
            "src": None,
            "dst": None,
        }
        if a_type == "MOVE":
            uid, sx, sy, dx, dy = uw._extract_move_components(action, obs)
            rec["unit_id"] = int(uid) if uid is not None else None
            rec["src"] = (int(sx), int(sy)) if sx is not None and sy is not None else None
            rec["dst"] = (int(dx), int(dy)) if dx is not None and dy is not None else None
        out.append(rec)
    return out


def _select_action_record(obs: Dict, info: Dict, legal_records: List[Dict], hidden_uncaptured_villages: Set[Tuple[int, int]]) -> Tuple[Dict, str]:
    city_count = int(_as_scalar(info.get("city_count", 1), default=1))
    stars = int(_as_scalar(info.get("stars", 0), default=0))
    captures = [r for r in legal_records if r["type"] == "CAPTURE"]
    if captures:
        return captures[0], "capture_priority"

    units = _owned_units(obs)
    preferred_uid, _pos, _dist, _targets = _nearest_unit_to_any_village(units, hidden_uncaptured_villages)
    moves = [r for r in legal_records if r["type"] == "MOVE"]
    chosen_move = _choose_hidden_move_record(obs, moves, hidden_uncaptured_villages, preferred_uid)
    if chosen_move is not None:
        return chosen_move, "hidden_village_move"

    if city_count >= 2:
        levelups = [r for r in legal_records if float(r["feat"][IDX_IS_LEVEL_UP]) > 0.5]
        if levelups:
            return levelups[0], "economy_level_up"
        resources = [r for r in legal_records if float(r["feat"][IDX_IS_RESOURCE_GATHERING]) > 0.5]
        if resources:
            return resources[0], "economy_resource"
        trains = [r for r in legal_records if float(r["feat"][IDX_IS_TRAIN_OR_SPAWN]) > 0.5]
        for r in trains:
            unit_type = str(r["action"].get("unit_type", "")).upper()
            repr_s = str(r["repr"]).upper()
            if stars > 0 and (unit_type == "WARRIOR" or "WARRIOR" in repr_s):
                return r, "economy_warrior_spawn"

    end_turns = [r for r in legal_records if float(r["feat"][IDX_IS_END_TURN]) > 0.5]
    if end_turns:
        return end_turns[0], "fallback_end_turn"
    return legal_records[0], "fallback_first_legal"


def _pct(arr: np.ndarray, thresh: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(100.0 * np.mean(arr >= float(thresh)))


def _mean_nonneg(values: List[float]) -> float:
    filtered = [float(v) for v in values if v is not None and float(v) >= 0.0]
    if not filtered:
        return float("nan")
    return float(np.mean(filtered))


def run_oracle(args, transform_mode: str, debug_failures: int = 0, run_label: str = "") -> Tuple[Dict[str, float], CoordEvidence]:
    env = gym.make("Tribes-v0")
    uw = env.unwrapped
    map_village_cache: Dict[str, Set[Tuple[int, int]]] = {}
    coord_evidence = CoordEvidence()

    terminal_spt: List[float] = []
    terminal_city_count: List[float] = []
    terminal_fog_cleared: List[float] = []
    terminal_unit_count: List[float] = []
    first_visible_turns: List[float] = []
    second_city_turns: List[float] = []
    second_city_not_captured = 0

    t2_nearest_distances: List[float] = []
    nearest_reached_by_t4 = 0
    nearest_reached_by_t5 = 0
    nearest_reached_by_t6 = 0

    pre_second_city_move_reduce = 0
    pre_second_city_move_increase = 0
    pre_second_city_move_total = 0
    pre_second_city_endturn_total = 0
    pre_second_city_endturn_with_move = 0
    capture_existed_turn_total = 0
    capture_existed_not_selected = 0
    episode_target_switched_before_second_city = 0

    debug_fail_budget = max(0, int(debug_failures))
    debug_fail_printed = 0

    for ep in range(int(args.episodes)):
        obs_vec, info = env.reset(seed=int(args.seed) + ep if args.seed_per_episode else None)
        obs = getattr(uw.tribes_env, "_last_obs", None)
        if not isinstance(obs, dict):
            raise RuntimeError("Failed to access wrapper observation dict.")

        level_path = str(getattr(uw, "_current_level_file", ""))
        if not level_path:
            raise RuntimeError("Missing _current_level_file from wrapper.")
        if level_path not in map_village_cache:
            map_village_cache[level_path] = _parse_hidden_villages_from_level_csv(level_path)
        hidden_raw_all = set(map_village_cache[level_path])
        hidden_transformed_all = _apply_transform(hidden_raw_all, transform_mode)

        hidden_uncaptured = hidden_transformed_all.difference(_owned_city_positions(obs))
        units_t2 = _owned_units(obs)
        _uid0, _pos0, nearest_dist_t2, nearest_targets_t2 = _nearest_unit_to_any_village(units_t2, hidden_uncaptured)
        if nearest_dist_t2 is None:
            nearest_dist_t2 = -1
            nearest_targets_t2 = set()
        t2_nearest_distances.append(float(nearest_dist_t2))
        nearest_target_reached_turn: Optional[int] = None

        trace_lines: List[str] = []
        prev_unit_pos: Dict[int, Tuple[int, int]] = {}
        target_switched = False
        last_target_before_second_city: Optional[Tuple[int, int]] = None
        second_city_already_captured = False

        done = False
        safety_steps = 0
        while not done:
            ids, valid, feats = _extract_legal_tensors(info)
            obs = getattr(uw.tribes_env, "_last_obs", obs)
            turn_now = int(_as_scalar(info.get("turn", -1), default=-1))
            stars_now = int(_as_scalar(info.get("stars", 0), default=0))
            city_count_now = int(_as_scalar(info.get("city_count", 1), default=1))
            if city_count_now >= 2:
                second_city_already_captured = True

            hidden_uncaptured = hidden_transformed_all.difference(_owned_city_positions(obs))
            units_now = _owned_units(obs)
            preferred_uid, preferred_pos, nearest_dist, nearest_targets = _nearest_unit_to_any_village(units_now, hidden_uncaptured)
            hidden_target = sorted(list(nearest_targets))[0] if nearest_targets else None

            if city_count_now < 2 and hidden_target is not None:
                if last_target_before_second_city is not None and hidden_target != last_target_before_second_city:
                    target_switched = True
                    trace_lines.append(
                        f"WARNING target village switched before second city capture: old={last_target_before_second_city} new={hidden_target} turn={turn_now}"
                    )
                last_target_before_second_city = hidden_target

            legal_records = _build_legal_records(env, obs, ids, valid, feats)
            capture_records = [r for r in legal_records if r["type"] == "CAPTURE"]
            move_records = [r for r in legal_records if r["type"] == "MOVE"]

            visible_runtime = _visible_uncaptured_villages(obs)
            capture_targets = [c for c in [_extract_capture_target_coord(r["action"], obs) for r in capture_records] if c is not None]
            _update_coord_evidence(coord_evidence, visible_runtime, hidden_raw_all, capture_targets)

            # Per-visible-village coordinate agreement print (A/B/C) in debug traces.
            if visible_runtime:
                hidden_id = _apply_transform(hidden_raw_all, TRANSFORM_IDENTITY)
                hidden_sw = _apply_transform(hidden_raw_all, TRANSFORM_SWAPPED)
                for v in sorted(list(visible_runtime)):
                    m_id = v in hidden_id
                    m_sw = v in hidden_sw
                    if m_id and not m_sw:
                        match_s = "A(identity)"
                    elif m_sw and not m_id:
                        match_s = "B(swapped)"
                    elif m_id and m_sw:
                        match_s = "A+B(both)"
                    else:
                        match_s = "C(neither)"
                    trace_lines.append(f"COORD_CHECK visible_village={v} match={match_s}")

            chosen, chosen_reason = _select_action_record(obs, info, legal_records, hidden_uncaptured)

            if capture_records:
                capture_existed_turn_total += 1
                if chosen["type"] != "CAPTURE":
                    capture_existed_not_selected += 1
                    if city_count_now < 2:
                        trace_lines.append(
                            f"WARNING city_count<2 legal CAPTURE exists but selected {chosen['type']} turn={turn_now}"
                        )

            if city_count_now < 2:
                if chosen["type"] == "END_TURN":
                    pre_second_city_endturn_total += 1
                    if move_records:
                        pre_second_city_endturn_with_move += 1
                        trace_lines.append(
                            f"WARNING city_count<2 chose END_TURN while legal MOVE exists turn={turn_now}"
                        )

                if chosen["type"] == "MOVE" and hidden_target is not None:
                    src = chosen["src"]
                    dst = chosen["dst"]
                    if src is not None and dst is not None:
                        d_before = _chebyshev(src, hidden_target)
                        d_after = _chebyshev(dst, hidden_target)
                        pre_second_city_move_total += 1
                        if d_after < d_before:
                            pre_second_city_move_reduce += 1
                        elif d_after > d_before:
                            pre_second_city_move_increase += 1

                        reducing_exists = False
                        for mr in move_records:
                            if mr["src"] is None or mr["dst"] is None:
                                continue
                            if _chebyshev(mr["dst"], hidden_target) < _chebyshev(mr["src"], hidden_target):
                                reducing_exists = True
                                break
                        if reducing_exists and d_after >= d_before:
                            trace_lines.append(
                                f"WARNING city_count<2 selected MOVE not reducing hidden-target distance ({d_before}->{d_after}) turn={turn_now}"
                            )

                        d_after_swapped = _chebyshev((int(dst[1]), int(dst[0])), hidden_target)
                        if d_after > d_before and d_after_swapped < d_before:
                            trace_lines.append(
                                f"WARNING possible x/y mismatch dst={dst} target={hidden_target} "
                                f"distance={d_before}->{d_after} swapped_after={d_after_swapped} turn={turn_now}"
                            )

            if city_count_now < 2 and hidden_target is not None and not capture_records:
                zero_units = []
                for uid, ux, uy, _ in units_now:
                    if _chebyshev((ux, uy), hidden_target) == 0:
                        zero_units.append((uid, ux, uy))
                if zero_units:
                    raw_target = _inverse_transform_coord(hidden_target, transform_mode)
                    swapped_of_raw = (int(raw_target[1]), int(raw_target[0]))
                    nearest_visible = None
                    vis = sorted(list(visible_runtime))
                    if vis:
                        best = None
                        for v in vis:
                            d = min(_chebyshev((u[1], u[2]), v) for u in zero_units)
                            if best is None or d < best[0]:
                                best = (d, v)
                        nearest_visible = best[1] if best is not None else None
                    trace_lines.append(
                        f"WARNING distance_to_hidden_target==0 but no legal CAPTURE turn={turn_now} "
                        f"units={zero_units} hidden_target={hidden_target} raw_target={raw_target} swapped_target={swapped_of_raw}"
                    )
                    trace_lines.append(
                        f"  tile_hidden={_tile_info(obs, hidden_target)} tile_swapped={_tile_info(obs, swapped_of_raw)} "
                        f"nearest_visible_village={nearest_visible}"
                    )
                    around_ids = {int(u[0]) for u in zero_units}
                    move_reprs = [r["repr"] for r in move_records if r["unit_id"] is not None and int(r["unit_id"]) in around_ids]
                    cap_reprs = [r["repr"] for r in capture_records if _parse_unit_id_from_repr(r["repr"]) in around_ids or r["action"].get("unit_id", None) in around_ids]
                    trace_lines.append(f"  local_legal_moves={move_reprs[:20]}")
                    trace_lines.append(f"  local_legal_captures={cap_reprs[:20]}")

            if 2 <= turn_now <= 10:
                chosen_uid = chosen["unit_id"] if chosen["type"] == "MOVE" else preferred_uid
                trace = [
                    f"[TRACE {run_label} ep={ep} turn={turn_now}] stars={stars_now} city_count={city_count_now}",
                    f"  visible_uncaptured_villages={sorted(list(visible_runtime))}",
                    f"  hidden_nearest_village_target={hidden_target}",
                    "  owned_units:",
                ]
                for uid, ux, uy, _ in sorted(units_now, key=lambda t: t[0]):
                    d = _chebyshev((ux, uy), hidden_target) if hidden_target is not None else None
                    trace.append(
                        f"    unit_id={uid} position=({ux},{uy}) distance_to_target={d} previous_position={prev_unit_pos.get(uid, None)}"
                    )
                trace.append(f"  legal_capture_actions={len(capture_records)}")
                trace.append(
                    f"  legal_move_actions_available_for_chosen_unit="
                    f"{len([r for r in move_records if chosen_uid is not None and r['unit_id'] is not None and int(r['unit_id']) == int(chosen_uid)])}"
                )
                trace.append(
                    f"  selected_action type={chosen['type']} slot={chosen['slot']} global_id={chosen['gid']} repr={chosen['repr']!r} reason={chosen_reason}"
                )
                if chosen["type"] == "MOVE":
                    src = chosen["src"]
                    dst = chosen["dst"]
                    db = _chebyshev(src, hidden_target) if src is not None and hidden_target is not None else None
                    da = _chebyshev(dst, hidden_target) if dst is not None and hidden_target is not None else None
                    did_dec = (da < db) if db is not None and da is not None else None
                    trace.append(
                        f"  move_detail source={src} target={dst} distance_before_to_hidden_target={db} "
                        f"distance_after_to_hidden_target={da} did_distance_decrease={did_dec} "
                        f"newly_revealed_tiles_if_move_norm={float(chosen['feat'][IDX_NEW_REVEAL_NORM]):.4f}"
                    )
                elif chosen["type"] == "END_TURN":
                    why = []
                    if capture_records:
                        why.append("capture_available")
                    if move_records:
                        why.append("move_available")
                    if not why:
                        why.append("no_move_or_capture")
                    trace.append(f"  end_turn_explanation={','.join(why)}")
                trace_lines.extend(trace)

            gid = int(chosen["gid"])
            obs_vec, _reward, terminated, truncated, info = env.step(gid)
            selected_global_id = _as_scalar(info.get("selected_global_id", None), default=None)
            if selected_global_id is not None and int(selected_global_id) != int(gid):
                raise RuntimeError(f"selected_global_id mismatch env={selected_global_id} chosen={gid}")

            obs_after = getattr(uw.tribes_env, "_last_obs", obs)
            for uid, ux, uy, _ in _owned_units(obs_after):
                prev_unit_pos[int(uid)] = (int(ux), int(uy))
            obs = obs_after

            turn_post = int(_as_scalar(info.get("turn", -1), default=-1))
            if nearest_target_reached_turn is None and nearest_targets_t2:
                for _uid, ux, uy, _ in _owned_units(obs):
                    if (ux, uy) in nearest_targets_t2:
                        nearest_target_reached_turn = int(turn_post)
                        break

            done = bool(terminated or truncated)
            safety_steps += 1
            if safety_steps > int(args.max_steps_per_episode):
                raise RuntimeError("Episode safety cap exceeded.")

        spt = float(_as_scalar(info.get("spt", np.nan), default=np.nan))
        city_count_t10 = float(_as_scalar(info.get("city_count", np.nan), default=np.nan))
        fog_t10 = float(_as_scalar(info.get("fog_tiles_cleared_total", np.nan), default=np.nan))
        unit_t10 = float(_as_scalar(info.get("unit_count", np.nan), default=np.nan))
        first_visible = float(_as_scalar(info.get("turn_first_uncaptured_village_visible", -1), default=-1))
        second_city = float(_as_scalar(info.get("turn_second_city_captured", -1), default=-1))

        terminal_spt.append(spt)
        terminal_city_count.append(city_count_t10)
        terminal_fog_cleared.append(fog_t10)
        terminal_unit_count.append(unit_t10)
        first_visible_turns.append(first_visible)
        second_city_turns.append(second_city)

        if target_switched and not second_city_already_captured:
            episode_target_switched_before_second_city += 1

        if second_city < 0:
            second_city_not_captured += 1
            if debug_fail_printed < debug_fail_budget:
                debug_fail_printed += 1
                print(
                    f"\n=== FAILED EPISODE TRACE {debug_fail_printed}/{debug_fail_budget} "
                    f"({run_label}, episode={ep}, map={os.path.basename(level_path)}, transform={transform_mode}) ==="
                )
                for ln in trace_lines:
                    print(ln)

        if nearest_target_reached_turn is not None:
            if nearest_target_reached_turn <= 4:
                nearest_reached_by_t4 += 1
            if nearest_target_reached_turn <= 5:
                nearest_reached_by_t5 += 1
            if nearest_target_reached_turn <= 6:
                nearest_reached_by_t6 += 1

        if args.progress_every > 0 and ((ep + 1) % int(args.progress_every) == 0):
            print(f"progress {run_label} episodes={ep + 1}/{args.episodes}")

    env.close()

    spt_arr = np.asarray(terminal_spt, dtype=np.float64)
    village_arr = np.asarray(terminal_city_count, dtype=np.float64)
    fog_arr = np.asarray(terminal_fog_cleared, dtype=np.float64)
    unit_arr = np.asarray(terminal_unit_count, dtype=np.float64)
    t2_dist_arr = np.asarray(t2_nearest_distances, dtype=np.float64)
    valid_t2 = t2_dist_arr[t2_dist_arr >= 0]
    episodes_f = float(max(1, int(args.episodes)))

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
        "fail_rate_second_city_not_captured_by_t10": float(100.0 * second_city_not_captured / episodes_f),
        "avg_turn_first_uncaptured_village_visible": _mean_nonneg(first_visible_turns),
        "avg_turn_second_city_captured": _mean_nonneg(second_city_turns),
        "mean_fog_tiles_cleared_t10": float(np.mean(fog_arr)),
        "mean_unit_count_t10": float(np.mean(unit_arr)),
        "avg_distance_T2_units_to_nearest_hidden_village": float(np.mean(valid_t2)) if valid_t2.size > 0 else float("nan"),
        "percent_maps_where_nearest_village_can_be_reached_by_T4": float(100.0 * nearest_reached_by_t4 / episodes_f),
        "percent_maps_where_nearest_village_can_be_reached_by_T5": float(100.0 * nearest_reached_by_t5 / episodes_f),
        "percent_maps_where_nearest_village_can_be_reached_by_T6": float(100.0 * nearest_reached_by_t6 / episodes_f),
        "percent_pre_second_city_moves_that_reduce_hidden_village_distance": (
            float(100.0 * pre_second_city_move_reduce / pre_second_city_move_total) if pre_second_city_move_total > 0 else float("nan")
        ),
        "percent_pre_second_city_moves_that_increase_hidden_village_distance": (
            float(100.0 * pre_second_city_move_increase / pre_second_city_move_total) if pre_second_city_move_total > 0 else float("nan")
        ),
        "percent_pre_second_city_end_turns_where_legal_move_existed": (
            float(100.0 * pre_second_city_endturn_with_move / pre_second_city_endturn_total) if pre_second_city_endturn_total > 0 else float("nan")
        ),
        "percent_turns_where_capture_existed_but_not_selected": (
            float(100.0 * capture_existed_not_selected / capture_existed_turn_total) if capture_existed_turn_total > 0 else float("nan")
        ),
        "percent_episodes_where_target_village_switched_before_second_city": float(
            100.0 * episode_target_switched_before_second_city / episodes_f
        ),
    }
    return summary, coord_evidence


def _infer_transform(evidence: CoordEvidence) -> Tuple[str, Dict[str, float]]:
    id_score = float(evidence.visible_identity_matches + evidence.capture_identity_matches)
    sw_score = float(evidence.visible_swapped_matches + evidence.capture_swapped_matches)
    selected = TRANSFORM_IDENTITY if id_score >= sw_score else TRANSFORM_SWAPPED
    visible_id_rate = (
        float(evidence.visible_identity_matches) / float(evidence.visible_total) if evidence.visible_total > 0 else float("nan")
    )
    visible_sw_rate = (
        float(evidence.visible_swapped_matches) / float(evidence.visible_total) if evidence.visible_total > 0 else float("nan")
    )
    capture_id_rate = (
        float(evidence.capture_identity_matches) / float(evidence.capture_total) if evidence.capture_total > 0 else float("nan")
    )
    capture_sw_rate = (
        float(evidence.capture_swapped_matches) / float(evidence.capture_total) if evidence.capture_total > 0 else float("nan")
    )
    stats = {
        "visible_match_rate_identity": visible_id_rate,
        "visible_match_rate_swapped": visible_sw_rate,
        "capture_match_rate_identity": capture_id_rate,
        "capture_match_rate_swapped": capture_sw_rate,
        "evidence_score_identity": id_score,
        "evidence_score_swapped": sw_score,
    }
    return selected, stats


def _print_comparison(before: Dict[str, float], after: Dict[str, float], infer_stats: Dict[str, float], selected: str) -> None:
    print("\n=== Coordinate Validation ===")
    print(f"coordinate match rate identity (visible) : {infer_stats['visible_match_rate_identity']:.6f}")
    print(f"coordinate match rate swapped  (visible) : {infer_stats['visible_match_rate_swapped']:.6f}")
    print(f"coordinate match rate identity (capture) : {infer_stats['capture_match_rate_identity']:.6f}")
    print(f"coordinate match rate swapped  (capture) : {infer_stats['capture_match_rate_swapped']:.6f}")
    print(f"selected transform                         : {selected}")

    keys = [
        "percent_pre_second_city_moves_that_reduce_hidden_village_distance",
        "percent_pre_second_city_moves_that_increase_hidden_village_distance",
        "fail_rate_second_city_not_captured_by_t10",
        "avg_turn_second_city_captured",
        "mean_village_count_t10",
        "mean_terminal_spt_t10",
    ]
    print("\n=== Before/After (500 episodes) ===")
    print(f"{'metric':70s} {'before(identity)':>18s} {'after(selected)':>18s}")
    for k in keys:
        b = before.get(k, float("nan"))
        a = after.get(k, float("nan"))
        bs = "nan" if (isinstance(b, float) and np.isnan(b)) else f"{float(b):.6f}"
        a_s = "nan" if (isinstance(a, float) and np.isnan(a)) else f"{float(a):.6f}"
        print(f"{k:70s} {bs:>18s} {a_s:>18s}")


def main():
    parser = argparse.ArgumentParser(description="Privileged hidden-village oracle with coordinate transform validation.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed-per-episode", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-steps-per-episode", type=int, default=256)
    parser.add_argument("--debug-failures", type=int, default=0, help="Print traces for first N failed episodes in BEFORE(identity) run.")
    args = parser.parse_args()

    before_summary, evidence = run_oracle(
        args=args,
        transform_mode=TRANSFORM_IDENTITY,
        debug_failures=int(args.debug_failures),
        run_label="before_identity",
    )
    selected, infer_stats = _infer_transform(evidence)

    if selected == TRANSFORM_IDENTITY:
        after_summary = dict(before_summary)
    else:
        after_summary, _ = run_oracle(
            args=args,
            transform_mode=selected,
            debug_failures=0,
            run_label="after_selected",
        )

    _print_comparison(before_summary, after_summary, infer_stats, selected)


if __name__ == "__main__":
    main()
