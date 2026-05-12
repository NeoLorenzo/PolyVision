import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from pol_env.Tribes.py.register_env import TribesGymWrapper


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


def _valid_legal_records(env: TribesGymWrapper, info: Dict[str, Any]) -> List[Dict[str, Any]]:
    ids = np.asarray(info.get("legal_global_ids_padded", []), dtype=np.int64).reshape(-1)
    valid = np.asarray(info.get("legal_action_valid_mask", []), dtype=bool).reshape(-1)
    n = min(ids.shape[0], valid.shape[0])
    if n <= 0:
        return []

    uw = env.unwrapped
    legal_id_to_raw = getattr(uw, "_current_legal_id_to_raw_index", {})
    legal_actions = getattr(uw, "_current_legal_actions", [])

    records: List[Dict[str, Any]] = []
    for slot in np.where(valid[:n])[0]:
        gid = int(ids[int(slot)])
        raw_idx = legal_id_to_raw.get(gid, None)
        action = None
        action_type = "UNKNOWN"
        action_repr = "UNKNOWN"
        if raw_idx is not None and 0 <= int(raw_idx) < len(legal_actions):
            action = legal_actions[int(raw_idx)]
            action_type = str(action.get("type", "UNKNOWN"))
            action_repr = str(action.get("repr", action_type))
        records.append(
            {
                "slot": int(slot),
                "gid": gid,
                "raw_idx": int(raw_idx) if raw_idx is not None else None,
                "type": action_type,
                "repr": action_repr,
                "action": action,
            }
        )
    return records


def _print_city_production_breakdown(env: TribesGymWrapper) -> None:
    obs = getattr(env.tribes_env, "_last_obs", {})
    if not isinstance(obs, dict):
        return
    city_map = obs.get("city", {})
    if not isinstance(city_map, dict):
        return
    rows = []
    for city_id, city in city_map.items():
        if not isinstance(city, dict):
            continue
        try:
            if int(city.get("tribeID", -1)) != 0:
                continue
            level = int(city.get("level", 0))
            prod = int(city.get("production", 0))
            is_capital = bool(city.get("isCapital", False))
            cap_bonus = 1 if is_capital else 0
            extra = int(prod) - int(level) - int(cap_bonus)
            rows.append(
                f"city={city_id} prod={prod} level={level} capital={int(is_capital)} "
                f"extra_over_level={extra}"
            )
        except Exception:
            continue
    if rows:
        print("[City production] " + " | ".join(rows))


def _print_state(env: TribesGymWrapper, step_idx: int, info: Dict[str, Any]) -> None:
    turn = _as_scalar(info.get("turn_count", None), default=None)
    spt = _as_scalar(info.get("spt", None), default=None)
    stars = _as_scalar(info.get("stars", None), default=None)
    cities = _as_scalar(info.get("city_count", None), default=None)
    units = _as_scalar(info.get("unit_count", None), default=None)
    map_id = info.get("map_id", None)
    seed = info.get("episode_seed", None)
    print("\n" + "=" * 110)
    print(
        f"step={step_idx} turn={turn} spt={spt} stars={stars} city_count={cities} unit_count={units}"
        + (f" map={map_id}" if map_id is not None else "")
        + (f" seed={seed}" if seed is not None else "")
    )
    print("=" * 110)
    _print_city_production_breakdown(env)


def _print_actions(records: List[Dict[str, Any]], page: int, page_size: int) -> None:
    total = len(records)
    if total <= 0:
        print("No legal actions.")
        return
    pages = (total + page_size - 1) // page_size
    page = max(0, min(page, pages - 1))
    start = page * page_size
    end = min(total, start + page_size)
    print(f"Legal actions {start}-{end - 1} / {total - 1} (page {page + 1}/{pages})")
    for i in range(start, end):
        r = records[i]
        text = str(r["repr"]).replace("\n", " ")
        if len(text) > 100:
            text = text[:97] + "..."
        print(f"[{i:03d}] gid={r['gid']:6d} type={str(r['type']):18s} {text}")


def _print_ansi_map(env: TribesGymWrapper, max_lines: int) -> None:
    try:
        ansi = env.tribes_env.render(mode="ansi")
    except Exception as e:
        print(f"(map unavailable: {e})")
        return
    if not isinstance(ansi, str) or len(ansi.strip()) == 0:
        print("(map unavailable: empty render)")
        return
    lines = ansi.splitlines()
    cap = max(1, int(max_lines))
    if len(lines) > cap:
        lines = lines[:cap]
        lines.append("... (map truncated)")
    print("\n[Map]")
    for ln in lines:
        print(ln)


def _set_env_defaults(args) -> None:
    os.environ["POLYVISION_LEVEL_POOL_GLOB"] = str(args.level_pool_glob)
    os.environ["POLYVISION_LEVEL_SELECTION_MODE"] = str(args.level_selection_mode)
    os.environ["POLYVISION_INFO_MODE"] = str(args.info_mode)
    os.environ["POLYVISION_STRICT_COORD_ASSERT"] = str(args.strict_coord_assert)
    os.environ["POLYVISION_MAX_LEGAL_ACTIONS"] = str(int(args.max_legal_actions))


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the PPO training wrapper env as a human via legal global actions.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level-pool-glob", type=str, default="levels/phase1_pool/*.csv")
    parser.add_argument("--level-selection-mode", type=str, default="round_robin", choices=["round_robin", "seeded_random"])
    parser.add_argument("--info-mode", type=str, default="fast", choices=["fast", "debug"])
    parser.add_argument("--strict-coord-assert", type=str, default="0")
    parser.add_argument("--max-legal-actions", type=int, default=1024)
    parser.add_argument("--page-size", type=int, default=30)
    parser.add_argument("--render-java", action="store_true")
    parser.add_argument("--step-delay-s", type=float, default=0.1)
    parser.add_argument("--show-ansi-map", action="store_true", help="Show text map render in terminal each step.")
    parser.add_argument("--ansi-map-max-lines", type=int, default=40)
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--auto-random", action="store_true", help="Non-interactive random legal play (for smoke tests).")
    args = parser.parse_args()

    _set_env_defaults(args)

    env = TribesGymWrapper()
    history: List[Dict[str, Any]] = []
    page = 0
    step_idx = 0
    start_ts = datetime.utcnow().isoformat() + "Z"

    try:
        obs, info = env.reset(seed=int(args.seed))

        if args.render_java:
            try:
                env.tribes_env.render(mode="java")
                time.sleep(max(0.0, float(args.step_delay_s)))
            except Exception as e:
                print(f"warning: failed to open Java render window: {e}")

        done = False
        while not done:
            records = _valid_legal_records(env, info)
            if not records:
                raise RuntimeError("No legal actions available in wrapper state.")

            _print_state(env, step_idx, info)
            if args.show_ansi_map:
                _print_ansi_map(env, max_lines=int(args.ansi_map_max_lines))
            _print_actions(records, page=page, page_size=int(args.page_size))

            chosen_idx: Optional[int] = None
            if args.auto_random:
                chosen_idx = int(np.random.randint(0, len(records)))
                print(f"auto-random selected action index {chosen_idx}")
            else:
                while True:
                    user = input(
                        "choose action index, or command [n/p page, i <idx> details, g <gid>, r random, q quit]: "
                    ).strip()
                    if user.lower() in ("q", "quit", "exit"):
                        print("Exiting by user request.")
                        done = True
                        break
                    if user.lower() == "n":
                        page += 1
                        _print_actions(records, page=page, page_size=int(args.page_size))
                        continue
                    if user.lower() == "p":
                        page -= 1
                        _print_actions(records, page=page, page_size=int(args.page_size))
                        continue
                    if user.lower() == "r":
                        chosen_idx = int(np.random.randint(0, len(records)))
                        break
                    if user.startswith("i "):
                        try:
                            idx = int(user.split(" ", 1)[1].strip())
                            if 0 <= idx < len(records):
                                print(json.dumps(records[idx]["action"], indent=2, default=str))
                            else:
                                print("index out of range")
                        except Exception:
                            print("invalid detail command")
                        continue
                    if user.startswith("g "):
                        try:
                            gid = int(user.split(" ", 1)[1].strip())
                            idx = next((i for i, r in enumerate(records) if int(r["gid"]) == gid), None)
                            if idx is None:
                                print("gid not currently legal")
                                continue
                            chosen_idx = int(idx)
                            break
                        except Exception:
                            print("invalid gid command")
                            continue
                    try:
                        idx = int(user)
                        if 0 <= idx < len(records):
                            chosen_idx = idx
                            break
                        print("index out of range")
                    except Exception:
                        print("invalid input")

            if done:
                break
            if chosen_idx is None:
                raise RuntimeError("No action selected.")

            chosen = records[int(chosen_idx)]
            gid = int(chosen["gid"])
            action_type = str(chosen["type"])
            action_repr = str(chosen["repr"])

            obs, reward, terminated, truncated, next_info = env.step(gid)
            done = bool(terminated or truncated)
            info = next_info if isinstance(next_info, dict) else {}

            history.append(
                {
                    "step": int(step_idx),
                    "selected_idx": int(chosen_idx),
                    "gid": int(gid),
                    "type": action_type,
                    "repr": action_repr,
                    "reward": float(reward),
                    "turn_count": _as_scalar(info.get("turn_count", None), default=None),
                    "spt": _as_scalar(info.get("spt", None), default=None),
                    "city_count": _as_scalar(info.get("city_count", None), default=None),
                    "stars": _as_scalar(info.get("stars", None), default=None),
                    "unit_count": _as_scalar(info.get("unit_count", None), default=None),
                }
            )

            print(f"executed gid={gid} type={action_type} reward={float(reward):+.3f}")
            if args.render_java:
                try:
                    env.tribes_env.render(mode="java")
                    time.sleep(max(0.0, float(args.step_delay_s)))
                except Exception as e:
                    print(f"warning: Java render update failed: {e}")

            step_idx += 1

        print("\nEpisode ended.")
        print(
            f"final_turn={_as_scalar(info.get('turn_count', None), default=None)} "
            f"final_spt={_as_scalar(info.get('spt', None), default=None)} "
            f"final_city_count={_as_scalar(info.get('city_count', None), default=None)} "
            f"final_stars={_as_scalar(info.get('stars', None), default=None)} "
            f"final_unit_count={_as_scalar(info.get('unit_count', None), default=None)}"
        )

        if args.save_json:
            out_dir = os.path.join(_repo_root, "outputs", "human_wrapper_runs")
            os.makedirs(out_dir, exist_ok=True)
            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"human_run_{stamp}.json")
            payload = {
                "started_at_utc": start_ts,
                "ended_at_utc": datetime.utcnow().isoformat() + "Z",
                "config": {
                    "seed": int(args.seed),
                    "level_pool_glob": str(args.level_pool_glob),
                    "level_selection_mode": str(args.level_selection_mode),
                    "info_mode": str(args.info_mode),
                    "strict_coord_assert": str(args.strict_coord_assert),
                    "max_legal_actions": int(args.max_legal_actions),
                },
                "final_info": info,
                "history": history,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            print(f"saved run log to {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
