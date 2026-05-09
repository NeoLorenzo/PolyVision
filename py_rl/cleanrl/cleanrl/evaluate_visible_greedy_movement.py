import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


# Match existing scripts: ensure repo root is importable.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import pol_env.Tribes.py.register_env as register_env  # noqa: F401
except Exception:
    pass


IDX_IS_MOVE = 0
IDX_NEW_REVEAL_NORM = 1
IDX_ADJ_FOG_AFTER_NORM = 2
IDX_ADJ_FOG_DELTA_NORM = 3
IDX_IS_ZERO_REVEAL_MOVE = 4
IDX_TARGET_VISIBLE_UNCAPTURED_VILLAGE = 5
IDX_DIST_DELTA_VISIBLE_VILLAGE_NORM = 7
IDX_IS_IMMEDIATE_BACKTRACK = 8
IDX_TARGET_INSIDE_OWNED_CITY_BOUNDS = 9
IDX_IS_END_TURN = 12
IDX_IS_CAPTURE = 13
IDX_IS_TRAIN_OR_SPAWN = 14
IDX_IS_RESOURCE_GATHERING = 16
IDX_IS_LEVEL_UP = 17


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


def _slot_scores_city_lt2(move_feats: np.ndarray) -> np.ndarray:
    return (
        5.0 * move_feats[:, IDX_TARGET_VISIBLE_UNCAPTURED_VILLAGE]
        + 3.0 * move_feats[:, IDX_NEW_REVEAL_NORM]
        + 2.0 * move_feats[:, IDX_ADJ_FOG_AFTER_NORM]
        + 1.0 * move_feats[:, IDX_ADJ_FOG_DELTA_NORM]
        + 2.0 * move_feats[:, IDX_DIST_DELTA_VISIBLE_VILLAGE_NORM]
        - 2.0 * move_feats[:, IDX_IS_ZERO_REVEAL_MOVE]
        - 1.0 * move_feats[:, IDX_IS_IMMEDIATE_BACKTRACK]
        - 0.5 * move_feats[:, IDX_TARGET_INSIDE_OWNED_CITY_BOUNDS]
    )


def _slot_scores_city_ge2(move_feats: np.ndarray) -> np.ndarray:
    return (
        2.0 * move_feats[:, IDX_NEW_REVEAL_NORM]
        + 1.0 * move_feats[:, IDX_ADJ_FOG_AFTER_NORM]
        + 1.0 * move_feats[:, IDX_TARGET_VISIBLE_UNCAPTURED_VILLAGE]
        - 1.0 * move_feats[:, IDX_IS_ZERO_REVEAL_MOVE]
    )


def _choose_slot(
    env,
    info: Dict,
    ids: np.ndarray,
    valid: np.ndarray,
    feats: np.ndarray,
    city_count: int,
    stars: int,
) -> int:
    valid_slots = np.where(valid)[0]
    if valid_slots.size == 0:
        raise RuntimeError("No valid legal action slots available.")

    # 1) CAPTURE first.
    capture_slots = valid_slots[feats[valid_slots, IDX_IS_CAPTURE] > 0.5]
    if capture_slots.size > 0:
        return int(capture_slots[0])

    # 2/3) MOVE scoring (depends on city_count).
    move_slots = valid_slots[feats[valid_slots, IDX_IS_MOVE] > 0.5]
    if move_slots.size > 0:
        move_feats = feats[move_slots]
        scores = _slot_scores_city_lt2(move_feats) if int(city_count) < 2 else _slot_scores_city_ge2(move_feats)
        return int(move_slots[int(np.argmax(scores))])

    # 4) Minimal economy fallback.
    # Prefer warrior spawn/train when legal (stars gate is already encoded by legality).
    uw = env.unwrapped
    legal_actions = getattr(uw, "_current_legal_actions", [])
    legal_id_to_raw = getattr(uw, "_current_legal_id_to_raw_index", {})
    train_slots = valid_slots[feats[valid_slots, IDX_IS_TRAIN_OR_SPAWN] > 0.5]
    if train_slots.size > 0 and int(stars) > 0:
        for slot in train_slots:
            gid = int(ids[int(slot)])
            raw_idx = legal_id_to_raw.get(gid, None)
            if raw_idx is None or raw_idx < 0 or raw_idx >= len(legal_actions):
                continue
            action = legal_actions[int(raw_idx)]
            unit_type = str(action.get("unit_type", "")).upper()
            repr_s = str(action.get("repr", "")).upper()
            if unit_type == "WARRIOR" or "WARRIOR" in repr_s:
                return int(slot)

    resource_slots = valid_slots[feats[valid_slots, IDX_IS_RESOURCE_GATHERING] > 0.5]
    if resource_slots.size > 0:
        return int(resource_slots[0])

    level_up_slots = valid_slots[feats[valid_slots, IDX_IS_LEVEL_UP] > 0.5]
    if level_up_slots.size > 0:
        return int(level_up_slots[0])

    end_turn_slots = valid_slots[feats[valid_slots, IDX_IS_END_TURN] > 0.5]
    if end_turn_slots.size > 0:
        return int(end_turn_slots[0])

    return int(valid_slots[0])


def _pct(arr: np.ndarray, thresh: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(100.0 * np.mean(arr >= float(thresh)))


def _mean_nonneg(values: List[float]) -> float:
    filtered = [float(v) for v in values if v is not None and float(v) >= 0.0]
    if not filtered:
        return float("nan")
    return float(np.mean(filtered))


def evaluate(args):
    env = gym.make("Tribes-v0")

    terminal_spt: List[float] = []
    terminal_city_count: List[float] = []
    terminal_fog_cleared: List[float] = []
    terminal_unit_count: List[float] = []
    first_visible_turns: List[float] = []
    second_city_turns: List[float] = []
    second_city_not_captured: int = 0

    selected_move_reveal_norm: List[float] = []
    selected_move_zero_reveal: List[float] = []
    selected_move_adj_fog_delta: List[float] = []
    selected_move_visible_village_target: List[float] = []
    n_moves_taken = 0

    obs, info = env.reset(seed=int(args.seed))
    for ep in range(int(args.episodes)):
        if ep > 0:
            obs, info = env.reset()

        done = False
        safety_steps = 0
        while not done:
            ids, valid, feats = _extract_legal_tensors(info)
            city_count = int(_as_scalar(info.get("city_count", 1), default=1))
            stars = int(_as_scalar(info.get("stars", 0), default=0))

            chosen_slot = _choose_slot(env, info, ids, valid, feats, city_count=city_count, stars=stars)
            chosen_gid = int(ids[chosen_slot])
            chosen_feat = feats[chosen_slot]

            if float(chosen_feat[IDX_IS_MOVE]) > 0.5:
                n_moves_taken += 1
                selected_move_reveal_norm.append(float(chosen_feat[IDX_NEW_REVEAL_NORM]))
                selected_move_zero_reveal.append(float(chosen_feat[IDX_IS_ZERO_REVEAL_MOVE]))
                selected_move_adj_fog_delta.append(float(chosen_feat[IDX_ADJ_FOG_DELTA_NORM]))
                selected_move_visible_village_target.append(float(chosen_feat[IDX_TARGET_VISIBLE_UNCAPTURED_VILLAGE]))

            obs, _reward, terminated, truncated, info = env.step(chosen_gid)

            selected_global_id = _as_scalar(info.get("selected_global_id", None), default=None)
            if selected_global_id is not None and int(selected_global_id) != int(chosen_gid):
                raise RuntimeError(
                    f"selected_global_id mismatch: env={selected_global_id}, chosen={chosen_gid}"
                )

            done = bool(terminated or truncated)
            safety_steps += 1
            if safety_steps > int(args.max_steps_per_episode):
                raise RuntimeError("Episode safety cap exceeded; aborting.")

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

        if second_city < 0:
            second_city_not_captured += 1

        if args.progress_every > 0 and ((ep + 1) % int(args.progress_every) == 0):
            print(f"progress episodes={ep + 1}/{args.episodes}")

    env.close()

    spt_arr = np.asarray(terminal_spt, dtype=np.float64)
    village_arr = np.asarray(terminal_city_count, dtype=np.float64)
    fog_arr = np.asarray(terminal_fog_cleared, dtype=np.float64)
    unit_arr = np.asarray(terminal_unit_count, dtype=np.float64)

    move_reveal_arr = np.asarray(selected_move_reveal_norm, dtype=np.float64)
    move_zero_arr = np.asarray(selected_move_zero_reveal, dtype=np.float64)
    move_adj_delta_arr = np.asarray(selected_move_adj_fog_delta, dtype=np.float64)
    move_visible_target_arr = np.asarray(selected_move_visible_village_target, dtype=np.float64)

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
        "selected_move_reveal_norm_mean": float(np.mean(move_reveal_arr)) if move_reveal_arr.size > 0 else float("nan"),
        "selected_zero_reveal_move_rate": float(np.mean(move_zero_arr)) if move_zero_arr.size > 0 else float("nan"),
        "selected_adjacent_fog_delta_mean": float(np.mean(move_adj_delta_arr)) if move_adj_delta_arr.size > 0 else float("nan"),
        "selected_visible_village_target_rate": float(np.mean(move_visible_target_arr)) if move_visible_target_arr.size > 0 else float("nan"),
        "number_of_moves_taken": int(n_moves_taken),
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
        "selected_move_reveal_norm_mean",
        "selected_zero_reveal_move_rate",
        "selected_adjacent_fog_delta_mean",
        "selected_visible_village_target_rate",
        "number_of_moves_taken",
    ]
    print("\n=== visible-info greedy movement evaluation ===")
    for key in rows:
        val = summary.get(key, None)
        if isinstance(val, float):
            if np.isnan(val):
                shown = "nan"
            else:
                shown = f"{val:.6f}"
        else:
            shown = str(val)
        print(f"{key:50s} {shown}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a visible-info greedy movement baseline on Tribes-v0.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to evaluate.")
    parser.add_argument("--seed", type=int, default=1, help="Initial reset seed (subsequent resets are unseeded).")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N episodes.")
    parser.add_argument("--max-steps-per-episode", type=int, default=256, help="Safety cap for per-episode decisions.")
    args = parser.parse_args()

    summary = evaluate(args)
    _print_table(summary)


if __name__ == "__main__":
    main()
