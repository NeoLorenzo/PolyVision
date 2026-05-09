import argparse
import csv
import glob
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch


_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import pol_env.Tribes.py.register_env as register_env  # noqa: F401
except Exception:
    pass

from py_rl.cleanrl.cleanrl.ppo import Agent


# Legal-action feature indices (from TribesGymWrapper.LEGAL_ACTION_FEATURE_NAMES)
IDX_IS_MOVE = 0
IDX_NEW_REVEAL_NORM = 1
IDX_ADJ_FOG_AFTER_NORM = 2
IDX_ADJ_FOG_DELTA_NORM = 3
IDX_IS_ZERO_REVEAL_MOVE = 4
IDX_TARGET_VISIBLE_UNCAPTURED_VILLAGE = 5
IDX_HAS_VISIBLE_UNCAPTURED_VILLAGE = 6
IDX_DIST_DELTA_VISIBLE_VILLAGE_NORM = 7
IDX_IS_IMMEDIATE_BACKTRACK = 8
IDX_TARGET_INSIDE_OWNED_CITY_BOUNDS = 9
IDX_DIST_FROM_CAPITAL_DELTA_NORM = 10
IDX_IS_END_TURN = 12
IDX_IS_CAPTURE = 13
IDX_IS_TRAIN_OR_SPAWN = 14
IDX_IS_RESEARCH = 15
IDX_IS_RESOURCE_GATHERING = 16
IDX_IS_LEVEL_UP = 17


def _as_scalar(value: Any, default: Any = None) -> Any:
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


def _find_latest_model(explicit_path: Optional[str]) -> str:
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(f"Model path does not exist: {explicit_path}")
        return explicit_path
    candidates = glob.glob(os.path.join(_repo_root, "runs", "**", "*.cleanrl_model"), recursive=True)
    if not candidates:
        raise FileNotFoundError("No .cleanrl_model files found under runs/**")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _load_action_interface_meta(model_path: str) -> Dict[str, Any]:
    meta_path = model_path + ".action_interface.json"
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _infer_actor_mode_from_state_dict(state_dict: Dict[str, Any]) -> str:
    keys = [str(k) for k in state_dict.keys()]
    has_dense_actor = any(k.startswith("actor.") for k in keys)
    has_feature_path = any(k.startswith("action_feature_encoder.") or k.startswith("action_scorer.") for k in keys)
    if has_dense_actor:
        return "dense_debug"
    if has_feature_path:
        return "legal_features"
    return "legal_only"


def _extract_single_legal_tensors(info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(info.get("legal_global_ids_padded", []), dtype=np.int64).reshape(-1)
    valid = np.asarray(info.get("legal_action_valid_mask", []), dtype=bool).reshape(-1)
    feats_raw = np.asarray(info.get("legal_action_features_padded", []), dtype=np.float32)
    if feats_raw.ndim == 3:
        feats = feats_raw[0]
    elif feats_raw.ndim == 2:
        feats = feats_raw
    else:
        feats = feats_raw.reshape((-1, int(feats_raw.shape[-1]) if feats_raw.size else 0))
    n = min(ids.shape[0], valid.shape[0], feats.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=bool), np.zeros((0, 0), dtype=np.float32)
    return ids[:n], valid[:n], feats[:n]


def _extract_move_components(env, action: Dict[str, Any], obs: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    uw = env.unwrapped
    try:
        return uw._extract_move_components(action, obs)
    except Exception:
        return None, None, None, None, None


def _get_visible_uncaptured_villages(env, obs: Dict[str, Any]) -> List[Tuple[int, int]]:
    uw = env.unwrapped
    fn = getattr(uw, "_get_visible_uncaptured_village_positions", None)
    if callable(fn):
        try:
            out = fn(obs)
            return [(int(x), int(y)) for (x, y) in out]
        except Exception:
            return []
    return []


def _parse_tech_type(env, action: Dict[str, Any]) -> str:
    uw = env.unwrapped
    t = None
    try:
        t = uw._action_str(action, "tech_type", None)
    except Exception:
        t = None
    if t is None:
        try:
            t = uw._parse_tech_type_from_action_repr(str(action.get("repr", "")))
        except Exception:
            t = None
    return str(t or "").upper()


def _parse_resource_type(env, action: Dict[str, Any]) -> str:
    uw = env.unwrapped
    r = None
    try:
        r = uw._action_str(action, "resource_type", None)
    except Exception:
        r = None
    if r is None:
        try:
            r = uw._parse_resource_type_from_action_repr(str(action.get("repr", "")))
        except Exception:
            r = None
    return str(r or "").upper()


def _parse_levelup_choice(env, action: Dict[str, Any]) -> str:
    uw = env.unwrapped
    c = None
    try:
        c = uw._action_str(action, "levelup_choice", None)
    except Exception:
        c = None
    if c is None:
        repr_s = str(action.get("repr", "")).upper()
        if "WORKSHOP" in repr_s:
            c = "WORKSHOP"
    return str(c or "").upper()


def _min_manhattan(src: Tuple[int, int], targets: Sequence[Tuple[int, int]]) -> Optional[int]:
    if src is None or not targets:
        return None
    sx, sy = int(src[0]), int(src[1])
    return min(abs(sx - int(tx)) + abs(sy - int(ty)) for tx, ty in targets)


def _compute_slot_logits_legal(agent: Agent, obs_t: torch.Tensor, legal_ids_t: torch.Tensor, valid_t: torch.Tensor, feats_t: Optional[torch.Tensor]) -> torch.Tensor:
    h = agent.state_encoder(obs_t)
    legal_emb = agent.action_embedding(legal_ids_t.long())
    if agent.actor_mode == "legal_features":
        if feats_t is None:
            raise RuntimeError("legal_features mode requires legal action features.")
        feat_emb = agent.action_feature_encoder(feats_t.float())
        repeated_state = h.unsqueeze(1).expand(-1, legal_ids_t.shape[1], -1)
        logits = agent.action_scorer(torch.cat([repeated_state, legal_emb, feat_emb], dim=-1)).squeeze(-1)
    else:
        logits = torch.einsum("bd,bkd->bk", h, legal_emb)
    logits = logits.masked_fill(~valid_t.bool(), -1e8)
    return logits


@dataclass
class LegalRecord:
    slot: int
    gid: int
    feats: np.ndarray
    raw_idx: Optional[int]
    action: Optional[Dict[str, Any]]
    action_type: str


def _collect_legal_records(env, info: Dict[str, Any]) -> List[LegalRecord]:
    ids, valid, feats = _extract_single_legal_tensors(info)
    uw = env.unwrapped
    legal_id_to_raw = getattr(uw, "_current_legal_id_to_raw_index", {})
    legal_actions = getattr(uw, "_current_legal_actions", [])
    out: List[LegalRecord] = []
    for slot in np.where(valid)[0]:
        gid = int(ids[int(slot)])
        raw_idx = legal_id_to_raw.get(gid, None)
        action = None
        action_type = "UNKNOWN"
        if raw_idx is not None and 0 <= int(raw_idx) < len(legal_actions):
            action = legal_actions[int(raw_idx)]
            action_type = str(action.get("type", "UNKNOWN")).upper()
        out.append(
            LegalRecord(
                slot=int(slot),
                gid=gid,
                feats=np.asarray(feats[int(slot)], dtype=np.float32),
                raw_idx=int(raw_idx) if raw_idx is not None else None,
                action=action,
                action_type=action_type,
            )
        )
    return out


def _argmax_slot(records: List[LegalRecord], score_fn) -> Optional[LegalRecord]:
    if not records:
        return None
    best = None
    best_score = None
    for rec in records:
        s = float(score_fn(rec))
        if best is None or s > best_score:
            best = rec
            best_score = s
    return best


def _choose_oracle_gid(env, info: Dict[str, Any], obs_vec: np.ndarray) -> int:
    records = _collect_legal_records(env, info)
    if not records:
        raise RuntimeError("No legal actions in oracle selector.")

    obs_full = getattr(env.unwrapped.tribes_env, "_last_obs", {})
    city_count = int(_as_scalar(info.get("city_count", 1), default=1))
    unit_count = int(_as_scalar(info.get("unit_count", 0), default=0))
    org_researched = int(_as_scalar(info.get("organization_researched", 0), default=0)) == 1
    visible_villages = _get_visible_uncaptured_villages(env, obs_full)

    # A. Research Organization immediately when legal, if not yet researched.
    if not org_researched:
        org_research = []
        for rec in records:
            if rec.action_type != "RESEARCH_TECH" or rec.action is None:
                continue
            tech = _parse_tech_type(env, rec.action)
            if tech == "ORGANIZATION":
                org_research.append(rec)
        if org_research:
            return int(org_research[0].gid)

    # B. Capture immediately when legal.
    captures = [r for r in records if r.action_type == "CAPTURE"]
    if captures:
        return int(captures[0].gid)

    # Candidate move records with decoded coords.
    move_data: List[Tuple[LegalRecord, Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]] = []
    for rec in records:
        if rec.action_type != "MOVE" or rec.action is None:
            continue
        uid, sx, sy, dx, dy = _extract_move_components(env, rec.action, obs_full)
        move_data.append((rec, uid, sx, sy, dx, dy))

    # C. If city_count < 2 and move onto visible neutral village exists, take best such move.
    if city_count < 2 and move_data:
        onto = [m for m in move_data if float(m[0].feats[IDX_TARGET_VISIBLE_UNCAPTURED_VILLAGE]) > 0.5]
        if onto:
            best = _argmax_slot(
                [m[0] for m in onto],
                lambda r: (
                    3.0 * float(r.feats[IDX_NEW_REVEAL_NORM])
                    + 1.5 * float(r.feats[IDX_ADJ_FOG_AFTER_NORM])
                    - 1.0 * float(r.feats[IDX_IS_IMMEDIATE_BACKTRACK])
                    - 0.5 * float(r.feats[IDX_TARGET_INSIDE_OWNED_CITY_BOUNDS])
                ),
            )
            if best is not None:
                return int(best.gid)

    # D. If visible village exists, move closest available unit toward it (largest distance reduction).
    if visible_villages and move_data:
        toward = []
        for rec, uid, sx, sy, dx, dy in move_data:
            if sx is None or sy is None or dx is None or dy is None:
                continue
            db = _min_manhattan((int(sx), int(sy)), visible_villages)
            da = _min_manhattan((int(dx), int(dy)), visible_villages)
            if db is None or da is None or da >= db:
                continue
            toward.append((rec, int(db), int(db - da)))
        if toward:
            toward.sort(
                key=lambda x: (
                    x[1],  # closest unit first
                    -x[2],  # biggest reduction
                    -float(x[0].feats[IDX_NEW_REVEAL_NORM]),
                    float(x[0].feats[IDX_IS_IMMEDIATE_BACKTRACK]),
                )
            )
            return int(toward[0][0].gid)

    # E. No visible village: explore fog with reveal-first, then center/unexplored, then avoid backtrack.
    if move_data:
        best = _argmax_slot(
            [m[0] for m in move_data],
            lambda r: (
                10.0 * float(r.feats[IDX_NEW_REVEAL_NORM])
                + 3.0 * float(r.feats[IDX_ADJ_FOG_AFTER_NORM])
                + 2.0 * float(r.feats[IDX_ADJ_FOG_DELTA_NORM])
                + 0.8 * float(r.feats[IDX_DIST_FROM_CAPITAL_DELTA_NORM])
                - 1.5 * float(r.feats[IDX_IS_IMMEDIATE_BACKTRACK])
                - 0.5 * float(r.feats[IDX_IS_ZERO_REVEAL_MOVE])
            ),
        )
        if best is not None and (
            float(best.feats[IDX_NEW_REVEAL_NORM]) > 0
            or float(best.feats[IDX_ADJ_FOG_AFTER_NORM]) > 0
            or float(best.feats[IDX_HAS_VISIBLE_UNCAPTURED_VILLAGE]) > 0
        ):
            return int(best.gid)

    # F. Economic actions (resource gathering), prioritizing immediate progress.
    resources = [r for r in records if r.action_type == "RESOURCE_GATHERING" and r.action is not None]
    if resources:
        uw = env.unwrapped

        def resource_score(rec: LegalRecord) -> float:
            r_type = _parse_resource_type(env, rec.action or {})
            completes_upgrade = False
            try:
                completes_upgrade = bool(uw._resource_gather_action_completes_city_upgrade(rec.action, obs_full))
            except Exception:
                completes_upgrade = False
            score = 0.0
            if completes_upgrade:
                score += 100.0
            if org_researched and r_type == "FRUIT":
                score += 25.0
            if r_type == "ANIMAL":
                score += 10.0
            if r_type in ("FISH", "CROPS"):
                score += 4.0
            return score

        best = _argmax_slot(resources, resource_score)
        if best is not None and resource_score(best) > 0:
            return int(best.gid)

    # G. City level-up choice: Workshop first if legal.
    levelups = [r for r in records if r.action_type == "LEVEL_UP" and r.action is not None]
    if levelups:
        workshops = [r for r in levelups if _parse_levelup_choice(env, r.action or {}) == "WORKSHOP"]
        if workshops:
            return int(workshops[0].gid)
        return int(levelups[0].gid)

    # H. Spawn/train warrior only when useful for exploration/capture.
    trains = [r for r in records if r.action_type in ("SPAWN", "TRAIN") and r.action is not None]
    if trains:
        warrior = []
        for rec in trains:
            unit_type = str(rec.action.get("unit_type", "")).upper()
            repr_s = str(rec.action.get("repr", "")).upper()
            if unit_type == "WARRIOR" or "WARRIOR" in repr_s:
                warrior.append(rec)
        useful_spawn = city_count < 2 or bool(visible_villages) or int(unit_count) < max(2, city_count + 1)
        if useful_spawn and warrior:
            return int(warrior[0].gid)

    # I. End turn only when no better useful action remains.
    end_turn = [r for r in records if r.action_type == "END_TURN"]
    if end_turn:
        return int(end_turn[0].gid)

    return int(records[0].gid)


def _numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _summarize(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "p25": None, "p75": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _mean(values: List[Optional[float]]) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _mean_nonneg(values: List[Optional[float]]) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v) and float(v) >= 0], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _rate_as_mean(values: List[Optional[float]]) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _spt_by_turn_tracker_update(turn_to_spt: Dict[int, float], info: Dict[str, Any]) -> None:
    turn = _as_scalar(info.get("turn_count", None), default=None)
    spt = _as_scalar(info.get("spt", None), default=None)
    if turn is None or spt is None:
        return
    try:
        turn_i = int(turn)
        spt_f = float(spt)
    except Exception:
        return
    turn_to_spt[turn_i] = spt_f


def _to_turn_list(turn_to_spt: Dict[int, float], max_turn: int = 10) -> List[Optional[float]]:
    return [float(turn_to_spt[t]) if t in turn_to_spt else None for t in range(0, int(max_turn) + 1)]


def _run_single_episode_policy(
    env,
    policy_name: str,
    seed: int,
    ppo_agent: Optional[Agent],
    ppo_actor_mode: Optional[str],
    ppo_device: torch.device,
    ppo_stochastic: bool,
) -> Dict[str, Any]:
    obs, info = env.reset(seed=int(seed))
    done = False
    episode_reward = 0.0
    step_count = 0
    turn_to_spt: Dict[int, float] = {}
    _spt_by_turn_tracker_update(turn_to_spt, info if isinstance(info, dict) else {})

    uw = env.unwrapped
    start_city_count = int(getattr(uw, "_starting_city_count", 1))
    level_file = str(getattr(uw, "_current_level_file", "") or "")
    level_id = os.path.basename(level_file) if level_file else None
    level_pool_index = _numeric(getattr(uw, "_current_level_index", None))
    episode_seed = _numeric(getattr(uw, "_last_reset_seed", seed))

    while not done:
        if not isinstance(info, dict):
            raise RuntimeError(f"{policy_name}: missing dict info for legal-action inference.")

        if policy_name == "oracle_org_only":
            action_gid = _choose_oracle_gid(env, info, obs)
        else:
            if ppo_agent is None:
                raise RuntimeError("PPO agent is not initialized.")
            obs_t = torch.tensor(obs, dtype=torch.float32, device=ppo_device).unsqueeze(0)
            if str(ppo_actor_mode) == "dense_debug":
                action_mask = np.zeros((int(env.action_space.n),), dtype=np.float32)
                ids, valid, _feats = _extract_single_legal_tensors(info)
                if ids.size > 0 and valid.size > 0:
                    valid_ids = ids[valid]
                    for gid in valid_ids:
                        g = int(gid)
                        if 0 <= g < int(env.action_space.n):
                            action_mask[g] = 1.0
                mask_t = torch.tensor(action_mask, dtype=torch.float32, device=ppo_device).unsqueeze(0)
                with torch.no_grad():
                    logits = ppo_agent.actor(obs_t)
                    masked_logits = logits.masked_fill(mask_t <= 0, -1e8)
                    if ppo_stochastic:
                        dist = torch.distributions.Categorical(logits=masked_logits)
                        action_gid = int(dist.sample().item())
                    else:
                        action_gid = int(torch.argmax(masked_logits, dim=-1).item())
            else:
                ids, valid, feats = _extract_single_legal_tensors(info)
                legal_ids_t = torch.tensor(ids, dtype=torch.long, device=ppo_device).unsqueeze(0)
                valid_t = torch.tensor(valid, dtype=torch.bool, device=ppo_device).unsqueeze(0)
                feats_t = None
                if str(ppo_actor_mode) == "legal_features":
                    feats_t = torch.tensor(feats, dtype=torch.float32, device=ppo_device).unsqueeze(0)
                with torch.no_grad():
                    logits = _compute_slot_logits_legal(ppo_agent, obs_t, legal_ids_t, valid_t, feats_t)
                    if ppo_stochastic:
                        dist = torch.distributions.Categorical(logits=logits)
                        slot = int(dist.sample().item())
                    else:
                        slot = int(torch.argmax(logits, dim=-1).item())
                    action_gid = int(legal_ids_t[0, slot].item())

        obs, reward, terminated, truncated, info = env.step(int(action_gid))
        episode_reward += float(reward)
        step_count += 1
        done = bool(terminated or truncated)
        if isinstance(info, dict):
            _spt_by_turn_tracker_update(turn_to_spt, info)
        if step_count > 5000:
            raise RuntimeError(f"{policy_name}: safety stop hit (too many steps).")

    if not isinstance(info, dict):
        info = {}

    final_spt = _numeric(_as_scalar(info.get("spt", None), default=None))
    final_city_count = _numeric(_as_scalar(info.get("city_count", None), default=None))
    villages_captured = None
    if final_city_count is not None:
        villages_captured = float(max(0.0, float(final_city_count) - float(start_city_count)))

    result = {
        "policy": policy_name,
        "episode_index": None,  # filled by caller
        "seed": int(seed),
        "episode_seed": int(episode_seed) if episode_seed is not None else None,
        "level_id": level_id,
        "map_file": level_file if level_file else None,
        "level_pool_index": int(level_pool_index) if level_pool_index is not None else None,
        "final_spt_t10": final_spt,
        "spt_by_turn": _to_turn_list(turn_to_spt, max_turn=10),
        "final_stars": _numeric(_as_scalar(info.get("stars", None), default=None)),
        "final_city_count": final_city_count,
        "final_avg_city_level": _numeric(_as_scalar(info.get("avg_city_level", None), default=None)),
        "turn_first_uncaptured_village_visible": _numeric(_as_scalar(info.get("turn_first_uncaptured_village_visible", None), default=None)),
        "turn_second_city_captured": _numeric(_as_scalar(info.get("turn_second_city_captured", None), default=None)),
        "final_unit_count": _numeric(_as_scalar(info.get("unit_count", None), default=None)),
        "organization_researched": _numeric(_as_scalar(info.get("organization_researched", None), default=None)),
        "turn_organization_researched": _numeric(_as_scalar(info.get("turn_organization_researched", None), default=None)),
        "forestry_researched": _numeric(_as_scalar(info.get("forestry_researched", None), default=None)),
        "turn_forestry_researched": _numeric(_as_scalar(info.get("turn_forestry_researched", None), default=None)),
        "techs_researched": _numeric(_as_scalar(info.get("techs_researched", None), default=None)),
        "fruit_harvested_t10": _numeric(_as_scalar(info.get("fruit_harvested_t10", None), default=None)),
        "animals_harvested_t10": _numeric(_as_scalar(info.get("animals_harvested_t10", None), default=None)),
        "villages_captured_t10": villages_captured,
        "fog_tiles_cleared_total": _numeric(_as_scalar(info.get("fog_tiles_cleared_total", None), default=None)),
        "total_reward": float(episode_reward),
        # In training logs, custom_spt_return corresponds to terminal SPT.
        "custom_spt_return": final_spt,
        "truncated": bool(_as_scalar(info.get("turn_count", 0), default=0) >= 10),
        "step_count": int(step_count),
    }
    return result


def _build_summary(per_episode_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in per_episode_rows:
        grouped.setdefault(str(row["policy"]), []).append(row)

    summary: Dict[str, Any] = {"by_policy": {}}
    for policy, rows in grouped.items():
        final_spt_vals = [_numeric(r.get("final_spt_t10")) for r in rows]
        city_count_vals = [_numeric(r.get("final_city_count")) for r in rows]
        second_city_turn_vals = [_numeric(r.get("turn_second_city_captured")) for r in rows]
        org_vals = [_numeric(r.get("organization_researched")) for r in rows]
        forestry_vals = [_numeric(r.get("forestry_researched")) for r in rows]
        fruit_vals = [_numeric(r.get("fruit_harvested_t10")) for r in rows]
        animal_vals = [_numeric(r.get("animals_harvested_t10")) for r in rows]
        fog_vals = [_numeric(r.get("fog_tiles_cleared_total")) for r in rows]

        spt_stats = _summarize(final_spt_vals)
        summary["by_policy"][policy] = {
            "episodes": int(len(rows)),
            "mean_final_spt": spt_stats["mean"],
            "median_final_spt": spt_stats["median"],
            "p25_final_spt": spt_stats["p25"],
            "p75_final_spt": spt_stats["p75"],
            "p90_final_spt": spt_stats["p90"],
            "max_final_spt": spt_stats["max"],
            "mean_city_count": _mean(city_count_vals),
            "mean_turn_second_city_captured": _mean_nonneg(second_city_turn_vals),
            "organization_research_rate": _rate_as_mean(org_vals),
            "forestry_research_rate": _rate_as_mean(forestry_vals),
            "mean_fruit_harvested": _mean(fruit_vals),
            "mean_animals_harvested": _mean(animal_vals),
            "mean_fog_tiles_cleared": _mean(fog_vals),
        }
    return summary


def _fmt(v: Optional[float], pct: bool = False) -> str:
    if v is None:
        return "null"
    if pct:
        return f"{100.0 * float(v):.2f}%"
    return f"{float(v):.3f}"


def _print_comparison_table(summary: Dict[str, Any]) -> None:
    base = summary.get("by_policy", {}).get("oracle_org_only", {})
    ppo = summary.get("by_policy", {}).get("ppo_latest", {})
    headers = [
        "metric",
        "ppo_latest",
        "oracle_org_only",
    ]
    rows = [
        ("mean final SPT", _fmt(ppo.get("mean_final_spt")), _fmt(base.get("mean_final_spt"))),
        ("median final SPT", _fmt(ppo.get("median_final_spt")), _fmt(base.get("median_final_spt"))),
        ("p25 final SPT", _fmt(ppo.get("p25_final_spt")), _fmt(base.get("p25_final_spt"))),
        ("p75 final SPT", _fmt(ppo.get("p75_final_spt")), _fmt(base.get("p75_final_spt"))),
        ("p90 final SPT", _fmt(ppo.get("p90_final_spt")), _fmt(base.get("p90_final_spt"))),
        ("max final SPT", _fmt(ppo.get("max_final_spt")), _fmt(base.get("max_final_spt"))),
        ("mean city count", _fmt(ppo.get("mean_city_count")), _fmt(base.get("mean_city_count"))),
        ("mean turn 2nd city", _fmt(ppo.get("mean_turn_second_city_captured")), _fmt(base.get("mean_turn_second_city_captured"))),
        ("Organization research rate", _fmt(ppo.get("organization_research_rate"), pct=True), _fmt(base.get("organization_research_rate"), pct=True)),
        ("Forestry research rate", _fmt(ppo.get("forestry_research_rate"), pct=True), _fmt(base.get("forestry_research_rate"), pct=True)),
        ("mean fruit harvested", _fmt(ppo.get("mean_fruit_harvested")), _fmt(base.get("mean_fruit_harvested"))),
        ("mean animals harvested", _fmt(ppo.get("mean_animals_harvested")), _fmt(base.get("mean_animals_harvested"))),
        ("mean fog tiles cleared", _fmt(ppo.get("mean_fog_tiles_cleared")), _fmt(base.get("mean_fog_tiles_cleared"))),
    ]
    w0 = max(len(headers[0]), max(len(r[0]) for r in rows))
    w1 = max(len(headers[1]), max(len(r[1]) for r in rows))
    w2 = max(len(headers[2]), max(len(r[2]) for r in rows))
    line = f"{headers[0]:<{w0}} | {headers[1]:>{w1}} | {headers[2]:>{w2}}"
    sep = f"{'-' * w0}-+-{'-' * w1}-+-{'-' * w2}"
    print(line)
    print(sep)
    for r in rows:
        print(f"{r[0]:<{w0}} | {r[1]:>{w1}} | {r[2]:>{w2}}")


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "policy",
        "episode_index",
        "seed",
        "episode_seed",
        "level_id",
        "map_file",
        "level_pool_index",
        "final_spt_t10",
        "spt_by_turn",
        "final_stars",
        "final_city_count",
        "final_avg_city_level",
        "turn_first_uncaptured_village_visible",
        "turn_second_city_captured",
        "final_unit_count",
        "organization_researched",
        "turn_organization_researched",
        "forestry_researched",
        "turn_forestry_researched",
        "techs_researched",
        "fruit_harvested_t10",
        "animals_harvested_t10",
        "villages_captured_t10",
        "fog_tiles_cleared_total",
        "total_reward",
        "custom_spt_return",
        "truncated",
        "step_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["spt_by_turn"] = json.dumps(out.get("spt_by_turn", []))
            writer.writerow(out)


def _write_markdown_report(path: str, summary: Dict[str, Any], model_path: str, args) -> None:
    base = summary.get("by_policy", {}).get("oracle_org_only", {})
    ppo = summary.get("by_policy", {}).get("ppo_latest", {})
    ppo_mean = ppo.get("mean_final_spt")
    base_mean = base.get("mean_final_spt")
    verdict = "inconclusive"
    if ppo_mean is not None and base_mean is not None:
        if base_mean > ppo_mean:
            verdict = "oracle baseline beats PPO on mean final SPT"
        elif base_mean < ppo_mean:
            verdict = "PPO beats oracle baseline on mean final SPT"
        else:
            verdict = "tie on mean final SPT"

    lines = [
        "# Org-only Oracle vs Latest PPO",
        "",
        f"- Timestamp: {datetime.utcnow().isoformat()}Z",
        f"- Episodes: {int(args.episodes)}",
        f"- PPO checkpoint: `{model_path}`",
        f"- PPO deterministic: `{not bool(args.ppo_stochastic)}`",
        "",
        f"## Verdict",
        f"- {verdict}",
        "",
        "## Summary",
        "",
        "| Metric | PPO | Oracle (Org-only) |",
        "|---|---:|---:|",
        f"| Mean final SPT | {_fmt(ppo.get('mean_final_spt'))} | {_fmt(base.get('mean_final_spt'))} |",
        f"| Mean city count | {_fmt(ppo.get('mean_city_count'))} | {_fmt(base.get('mean_city_count'))} |",
        f"| Mean turn second city captured | {_fmt(ppo.get('mean_turn_second_city_captured'))} | {_fmt(base.get('mean_turn_second_city_captured'))} |",
        f"| Organization research rate | {_fmt(ppo.get('organization_research_rate'), pct=True)} | {_fmt(base.get('organization_research_rate'), pct=True)} |",
        f"| Forestry research rate | {_fmt(ppo.get('forestry_research_rate'), pct=True)} | {_fmt(base.get('forestry_research_rate'), pct=True)} |",
        f"| Mean fruit harvested | {_fmt(ppo.get('mean_fruit_harvested'))} | {_fmt(base.get('mean_fruit_harvested'))} |",
        f"| Mean fog tiles cleared | {_fmt(ppo.get('mean_fog_tiles_cleared'))} | {_fmt(base.get('mean_fog_tiles_cleared'))} |",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Organization-only greedy oracle vs latest PPO on Tribes-v0.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42, help="Base seed used for per-episode seeds.")
    parser.add_argument("--model-path", type=str, default=None, help="Optional explicit PPO checkpoint.")
    parser.add_argument("--output-dir", type=str, default=os.path.join("outputs", "org_only_oracle_vs_ppo"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--ppo-stochastic", action="store_true", help="Sample PPO actions instead of argmax.")
    args = parser.parse_args()

    os.environ["POLYVISION_LEVEL_POOL_GLOB"] = "levels/phase1_pool/*.csv"
    os.environ["POLYVISION_LEVEL_SELECTION_MODE"] = "round_robin"
    os.environ["POLYVISION_INFO_MODE"] = "fast"
    os.environ["POLYVISION_STRICT_COORD_ASSERT"] = "0"
    os.environ["POLYVISION_MAX_LEGAL_ACTIONS"] = "1024"

    model_path = _find_latest_model(args.model_path)
    model_mtime = datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
    print(f"Using latest PPO checkpoint: {model_path}")
    print(f"Checkpoint mtime: {model_mtime}")

    meta = _load_action_interface_meta(model_path)
    state_dict = torch.load(model_path, map_location="cpu")
    actor_mode = str(meta.get("actor_mode", "")).strip().lower()
    if actor_mode not in ("legal_only", "legal_features", "dense_debug"):
        actor_mode = _infer_actor_mode_from_state_dict(state_dict)
    max_legal_actions = int(meta.get("max_legal_actions", 1024))
    legal_action_feature_dim = int(meta.get("legal_action_feature_dim", 22))
    os.environ["POLYVISION_MAX_LEGAL_ACTIONS"] = str(max_legal_actions)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    env = gym.make("Tribes-v0")
    env_adapter = SimpleNamespace(
        single_observation_space=env.observation_space,
        single_action_space=env.action_space,
    )
    ppo_agent = Agent(
        env_adapter,
        actor_mode=actor_mode,
        max_legal_actions=max_legal_actions,
        legal_action_feature_dim=legal_action_feature_dim,
    ).to(device)
    ppo_agent.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    ppo_agent.eval()

    print(
        "PPO load config: "
        f"actor_mode={actor_mode} max_legal_actions={max_legal_actions} "
        f"legal_action_feature_dim={legal_action_feature_dim} ppo_stochastic={bool(args.ppo_stochastic)}"
    )

    per_episode: List[Dict[str, Any]] = []
    for ep in range(int(args.episodes)):
        ep_seed = int(args.seed) + int(ep)
        ppo_row = _run_single_episode_policy(
            env=env,
            policy_name="ppo_latest",
            seed=ep_seed,
            ppo_agent=ppo_agent,
            ppo_actor_mode=actor_mode,
            ppo_device=device,
            ppo_stochastic=bool(args.ppo_stochastic),
        )
        ppo_row["episode_index"] = int(ep)
        per_episode.append(ppo_row)

        oracle_row = _run_single_episode_policy(
            env=env,
            policy_name="oracle_org_only",
            seed=ep_seed,
            ppo_agent=None,
            ppo_actor_mode=None,
            ppo_device=device,
            ppo_stochastic=False,
        )
        oracle_row["episode_index"] = int(ep)
        per_episode.append(oracle_row)

        if int(args.progress_every) > 0 and (ep + 1) % int(args.progress_every) == 0:
            print(f"Progress: episodes {ep + 1}/{args.episodes}")

    env.close()

    summary = _build_summary(per_episode)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "per_episode_results.csv")
    json_path = os.path.join(run_dir, "summary.json")
    md_path = os.path.join(run_dir, "report.md")

    _write_csv(csv_path, per_episode)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "episodes": int(args.episodes),
                    "seed": int(args.seed),
                    "model_path": model_path,
                    "model_mtime": model_mtime,
                    "actor_mode": actor_mode,
                    "max_legal_actions": int(max_legal_actions),
                    "legal_action_feature_dim": int(legal_action_feature_dim),
                    "ppo_stochastic": bool(args.ppo_stochastic),
                    "env_id": "Tribes-v0",
                    "actor_mode_requested": "legal_features",
                    "max_legal_actions_requested": 1024,
                    "env_overrides": {
                        "POLYVISION_LEVEL_POOL_GLOB": "levels/phase1_pool/*.csv",
                        "POLYVISION_LEVEL_SELECTION_MODE": "round_robin",
                        "POLYVISION_INFO_MODE": "fast",
                        "POLYVISION_STRICT_COORD_ASSERT": "0",
                    },
                },
                "summary": summary,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    _write_markdown_report(md_path, summary, model_path, args)

    _print_comparison_table(summary)
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote report: {md_path}")


if __name__ == "__main__":
    main()
