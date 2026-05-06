import argparse
import glob
import os
import re
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.distributions.categorical import Categorical

from pol_env.Tribes.py.register_env import TribesGymWrapper
from py_rl.cleanrl.cleanrl.ppo import Agent


def find_latest_model(explicit_path: str | None) -> str:
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(f"Model path does not exist: {explicit_path}")
        return explicit_path

    candidates = glob.glob(os.path.join("runs", "**", "*.cleanrl_model"), recursive=True)
    if not candidates:
        raise FileNotFoundError("No .cleanrl_model files found under runs/**")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def safe_array_mask(mask, action_n: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32).reshape(-1)
    if arr.shape[0] == action_n:
        return arr
    out = np.zeros(action_n, dtype=np.float32)
    n = min(action_n, arr.shape[0])
    out[:n] = arr[:n]
    return out


def compute_current_spt(env: TribesGymWrapper) -> float | None:
    raw_obs = getattr(env.tribes_env, "_last_obs", None)
    if raw_obs is None:
        return None
    try:
        return float(env.tribes_env._compute_spt_from_obs(raw_obs, tribe_id=0))
    except Exception:
        return None


def format_pct(p: float) -> str:
    return f"{(100.0 * p):6.2f}%"


def print_reward_breakdown(info, total_reward: float, truncated: bool = False):
    if not isinstance(info, dict):
        print(f"Reward Breakdown: total={float(total_reward):+.4f} (info unavailable)")
        return

    delta_spt = float(info.get("delta_spt", 0.0))
    capture_bonus = float(info.get("reward_capture_city_bonus", 0.0))
    delay_penalty = float(info.get("reward_second_village_delay_penalty", 0.0))
    neglect_penalty = float(info.get("reward_visible_village_neglect_penalty", 0.0))
    breadcrumb = float(info.get("reward_village_breadcrumb", 0.0))
    fog_clearance = float(info.get("reward_fog_clearance", 0.0))
    t10_penalty = 0.0
    if bool(truncated):
        turn_count = int(info.get("turn_count", -1))
        city_count = int(info.get("city_count", 0))
        starting_city_count = int(info.get("starting_city_count", 0))
        if turn_count >= 10 and city_count <= starting_city_count:
            t10_penalty = -3.0

    shaping_sum = capture_bonus + delay_penalty + neglect_penalty + breadcrumb + fog_clearance + t10_penalty
    reconstructed_total = delta_spt + shaping_sum
    reward_adjustment = float(info.get("reward_adjustment", shaping_sum))
    selected_action_type = str(info.get("selected_action_type", "UNKNOWN"))

    print("Reward Breakdown:")
    print(f"  selected_action_type:            {selected_action_type}")
    print(f"  base_delta_spt:                  {delta_spt:+.4f}")
    print(f"  capture_city_bonus:              {capture_bonus:+.4f}")
    print(f"  second_village_delay_penalty:    {delay_penalty:+.4f}")
    print(f"  visible_village_neglect_penalty: {neglect_penalty:+.4f}")
    print(f"  village_breadcrumb:              {breadcrumb:+.4f}")
    print(f"  fog_clearance_reward:            {fog_clearance:+.4f}")
    if t10_penalty != 0.0:
        print(f"  second_village_by_t10_penalty:   {t10_penalty:+.4f}")
    print(f"  shaping_sum:                     {shaping_sum:+.4f}")
    print(f"  reward_adjustment(info):         {reward_adjustment:+.4f}")
    print(f"  reconstructed_total:             {reconstructed_total:+.4f}")
    print(f"  env_returned_total:              {float(total_reward):+.4f}")


def parse_move_action_repr(action_repr: str):
    if not isinstance(action_repr, str):
        return None
    m = re.search(
        r"by unit\s+(-?\d+).*?\bto\s+(-?\d+)\s*:\s*(-?\d+)",
        action_repr,
        flags=re.IGNORECASE,
    )
    if m is not None:
        try:
            unit_id = int(m.group(1))
            dest_x = int(m.group(2))
            dest_y = int(m.group(3))
            return unit_id, dest_x, dest_y
        except Exception:
            return None
    nums = re.findall(r"-?\d+", action_repr)
    if len(nums) < 3:
        return None
    try:
        unit_id = int(nums[0])
        dest_x = int(nums[-2])
        dest_y = int(nums[-1])
        return unit_id, dest_x, dest_y
    except Exception:
        return None


def infer_relative_delta(cur_x: int, cur_y: int, dest_x: int, dest_y: int):
    return int(dest_x) - int(cur_x), int(dest_y) - int(cur_y)


def get_unit_pos_from_env_obs(env: TribesGymWrapper, unit_id: int):
    raw_obs = getattr(env.tribes_env, "_last_obs", None)
    if not isinstance(raw_obs, dict):
        return None
    units = raw_obs.get("unit", {})
    if not isinstance(units, dict):
        return None

    # Prefer direct id key lookup first.
    candidate = units.get(str(unit_id), None)
    if isinstance(candidate, dict):
        try:
            return int(candidate.get("x", -1)), int(candidate.get("y", -1))
        except Exception:
            pass

    # Fallback scan in case keys are not unit ids.
    for key, unit in units.items():
        if not isinstance(unit, dict):
            continue
        try:
            if int(key) == unit_id:
                return int(unit.get("x", -1)), int(unit.get("y", -1))
        except Exception:
            continue
    return None


def get_board_dims_from_env_obs(env: TribesGymWrapper):
    raw_obs = getattr(env.tribes_env, "_last_obs", None)
    terrain = raw_obs.get("board", {}).get("terrain", []) if isinstance(raw_obs, dict) else []
    height = len(terrain)
    width = max((len(r) for r in terrain), default=0) if terrain else 0
    return width, height


def format_action_for_debug(action_repr: str, action_type: str, env: TribesGymWrapper) -> str:
    if action_type != "MOVE":
        return action_repr
    parsed = parse_move_action_repr(action_repr)
    if parsed is None:
        return action_repr
    unit_id, dest_x, dest_y = parsed
    cur = get_unit_pos_from_env_obs(env, unit_id)
    if cur is None:
        return f"MOVE by unit {unit_id} [to {dest_x}:{dest_y} | rel dX=?, dY=?]"
    cur_x, cur_y = cur
    dx, dy = infer_relative_delta(cur_x, cur_y, dest_x, dest_y)
    raw_obs = getattr(env.tribes_env, "_last_obs", None)
    terrain = raw_obs.get("board", {}).get("terrain", []) if isinstance(raw_obs, dict) else []
    width = max((len(r) for r in terrain), default=0)
    height = len(terrain)
    in_bounds = width > 0 and height > 0 and (0 <= int(dest_x) < width and 0 <= int(dest_y) < height)
    return (
        f"MOVE by unit {unit_id} "
        f"[from {cur_x}:{cur_y} -> {dest_x}:{dest_y} | rel dX={dx:+d}, dY={dy:+d} | in_bounds={in_bounds}]"
    )


def print_policy_move_grid(env: TribesGymWrapper, legal_actions, allowed_indices, action_mask, probs_np, chosen_pos: int):
    # Build move candidates from currently legal, allowed actions.
    move_rows = []
    for pos, raw_idx in enumerate(allowed_indices):
        if pos >= len(probs_np) or action_mask[pos] <= 0:
            continue
        if not (0 <= raw_idx < len(legal_actions)):
            continue
        act = legal_actions[raw_idx]
        if str(act.get("type", "")) != "MOVE":
            continue
        parsed = parse_move_action_repr(str(act.get("repr", "")))
        if parsed is None:
            continue
        unit_id, dest_x, dest_y = parsed
        cur = get_unit_pos_from_env_obs(env, unit_id)
        if cur is None:
            continue
        cur_x, cur_y = cur
        dx, dy = infer_relative_delta(cur_x, cur_y, dest_x, dest_y)
        move_rows.append(
            {
                "pos": pos,
                "unit_id": unit_id,
                "cur": (int(cur_x), int(cur_y)),
                "dest": (int(dest_x), int(dest_y)),
                "dx": int(dx),
                "dy": int(dy),
                "p": float(probs_np[pos]),
            }
        )

    if not move_rows:
        return

    # Anchor on chosen move unit if chosen action is MOVE; otherwise highest-prob move.
    anchor = None
    for m in move_rows:
        if m["pos"] == chosen_pos:
            anchor = m
            break
    if anchor is None:
        anchor = max(move_rows, key=lambda m: m["p"])

    anchor_unit = anchor["unit_id"]
    anchor_cur = anchor["cur"]
    unit_rows = [m for m in move_rows if m["unit_id"] == anchor_unit and m["cur"] == anchor_cur]
    if not unit_rows:
        return

    raw_obs = getattr(env.tribes_env, "_last_obs", None)
    terrain = raw_obs.get("board", {}).get("terrain", []) if isinstance(raw_obs, dict) else []
    map_h = len(terrain)
    map_w = max((len(r) for r in terrain), default=0) if terrain else 0
    if map_w <= 0 or map_h <= 0:
        map_w = 15
        map_h = 15

    rel_prob = {}
    max_delta = 1
    for m in unit_rows:
        rel_prob[(m["dx"], m["dy"])] = m["p"]
        max_delta = max(max_delta, abs(m["dx"]), abs(m["dy"]))

    radius = max(2, max_delta)
    print(f"POLICY_MOVE_GRID: unit={anchor_unit} centered at current tile")
    print("  (numbers = move probability %, X = invalid/unavailable, U = unit)")
    for rel_y in range(-radius, radius + 1):
        row = []
        for rel_x in range(-radius, radius + 1):
            if rel_x == 0 and rel_y == 0:
                row.append("  U  ")
                continue

            world_x = int(anchor_cur[0]) + rel_x
            world_y = int(anchor_cur[1]) + rel_y
            off_board = world_x < 0 or world_y < 0 or world_x >= map_w or world_y >= map_h
            if off_board:
                row.append("  X  ")
                continue

            key = (rel_x, rel_y)
            if key in rel_prob:
                pct = 100.0 * rel_prob[key]
                row.append(f"{pct:5.1f}")
            else:
                row.append("  X  ")
        print(" ".join(row))


def print_move_bounds_sanity(env: TribesGymWrapper, legal_actions, allowed_indices):
    raw_obs = getattr(env.tribes_env, "_last_obs", None)
    terrain = raw_obs.get("board", {}).get("terrain", []) if isinstance(raw_obs, dict) else []
    height = len(terrain)
    width = max((len(r) for r in terrain), default=0) if terrain else 0
    if width <= 0 or height <= 0:
        return

    for pos, raw_idx in enumerate(allowed_indices):
        if not (0 <= raw_idx < len(legal_actions)):
            continue
        act = legal_actions[raw_idx]
        if str(act.get("type", "")) != "MOVE":
            continue
        parsed = parse_move_action_repr(str(act.get("repr", "")))
        if parsed is None:
            continue
        unit_id, dest_x, dest_y = parsed
        if not (0 <= int(dest_x) < width and 0 <= int(dest_y) < height):
            print(
                f"WARNING_OFFBOARD_MOVE: allowed_pos={pos} raw_idx={raw_idx} unit={unit_id} "
                f"dest={dest_x}:{dest_y} board={width}x{height} repr={act.get('repr')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-episode policy introspection for Tribes PPO.")
    parser.add_argument("--model-path", type=str, default=None, help="Optional explicit .cleanrl_model path.")
    parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
    parser.add_argument("--device", type=str, default="cpu", help="torch device, e.g. cpu or cuda.")
    parser.add_argument(
        "--render-java",
        action="store_true",
        help="Render live gameplay in the Java GUI while evaluating one episode.",
    )
    parser.add_argument(
        "--step-delay-s",
        type=float,
        default=0.25,
        help="Delay between rendered steps when --render-java is enabled.",
    )
    parser.add_argument(
        "--manual-step",
        action="store_true",
        help="Pause before each action. Press Enter to continue, or 'q' then Enter to quit.",
    )
    parser.add_argument(
        "--show-opening",
        action="store_true",
        help="Replay and render the hardcoded opening sequence step-by-step before policy control starts.",
    )
    parser.add_argument(
        "--level-pool-glob",
        type=str,
        default=None,
        help="Optional map pool glob, e.g. levels/phase1_pool/*.csv (matches training env setting).",
    )
    parser.add_argument(
        "--level-selection-mode",
        type=str,
        default=None,
        choices=["round_robin", "seeded_random"],
        help="Optional map selection mode override for wrapper.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Optional POLYVISION_BASE_SEED override for deterministic seed stream.",
    )
    args = parser.parse_args()

    # Optional env overrides so evaluate_brain can mirror trainer map settings.
    if args.level_pool_glob:
        os.environ["POLYVISION_LEVEL_POOL_GLOB"] = str(args.level_pool_glob)
    if args.level_selection_mode:
        os.environ["POLYVISION_LEVEL_SELECTION_MODE"] = str(args.level_selection_mode)
    if args.base_seed is not None:
        os.environ["POLYVISION_BASE_SEED"] = str(int(args.base_seed))

    model_path = find_latest_model(args.model_path)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    env = TribesGymWrapper()
    try:
        env_adapter = SimpleNamespace(
            single_observation_space=env.observation_space,
            single_action_space=env.action_space,
        )
        agent = Agent(env_adapter).to(device)

        state_dict = torch.load(model_path, map_location=device)
        agent.load_state_dict(state_dict, strict=False)
        agent.eval()

        print("=" * 100)
        print(f"Loaded model: {model_path}")
        print("=" * 100)

        if args.show_opening:
            # Reproduce reset initialization manually so we can visualize each
            # hardcoded opening action instead of skipping straight to Turn 2.
            # Important: use the same wrapper seed/map selection path as training,
            # rather than forcing env.level_file directly.
            episode_seed = env._resolve_episode_seed(seed=args.seed)
            level_file, level_index = env._select_level_for_reset(episode_seed)
            env._current_level_file = level_file
            env._current_level_index = int(level_index)
            env._last_reset_seed = int(episode_seed)
            env._episode_index += 1
            obs = env.tribes_env.reset(level_file, episode_seed)
            env._turn_count = 0
            print(f"Map: {os.path.basename(level_file)} | pool_index={level_index} | episode_seed={episode_seed}")

            if args.render_java:
                try:
                    env.tribes_env.render(mode="java")
                    time.sleep(max(0.0, args.step_delay_s))
                except Exception as e:
                    print(f"Warning: could not open Java render window: {e}")

            original_step = env.tribes_env.step

            def traced_step(action_index):
                legal = env.tribes_env.list_actions()
                action_desc = "UNKNOWN"
                action_type = "UNKNOWN"
                if 0 <= int(action_index) < len(legal):
                    act = legal[int(action_index)]
                    action_type = str(act.get("type", "UNKNOWN"))
                    action_desc = str(act.get("repr", act.get("type", "UNKNOWN")))
                action_desc_l = action_desc.lower()
                is_tribe1_action = ("by tribe 1" in action_desc_l) or ("tribe 1" in action_desc_l)
                should_log = not is_tribe1_action

                if should_log:
                    print(f"[Opening] Executing: {action_desc}")

                if args.manual_step and should_log:
                    user_in = input("Press Enter for next opening action ('q' + Enter to quit): ").strip().lower()
                    if user_in in ("q", "quit", "exit"):
                        raise KeyboardInterrupt("Opening replay interrupted by user.")

                out = original_step(action_index)

                if args.render_java and should_log:
                    try:
                        step_obs = out[0] if isinstance(out, tuple) and len(out) > 0 else None
                        active_tribe = -1
                        if isinstance(step_obs, dict):
                            try:
                                active_tribe = int(step_obs.get("activeTribeID", -1))
                            except Exception:
                                active_tribe = -1
                        # Avoid visual tribe-1 flicker during opening replay.
                        if active_tribe == 0:
                            env.tribes_env.render(mode="java")
                            time.sleep(max(0.0, args.step_delay_s))
                    except Exception as e:
                        print(f"Warning: Java render update failed during opening: {e}")
                return out

            try:
                env.tribes_env.step = traced_step
                obs = env._apply_bardur_opening(obs)
            finally:
                env.tribes_env.step = original_step

            env._starting_city_count = env._get_city_count(obs)
            env._last_city_count = env._starting_city_count
            env._moved_on_t0 = False
            env._visible_village_streak_turns = 0
            action_mask, allowed_indices = env._build_action_mask_and_indices()
            obs = env._dict_to_array(obs)
            info = {
                "valid_actions": len(allowed_indices),
                "raw_valid_actions": env.tribes_env.action_space_n,
                "turn_count": env._turn_count,
                "action_mask": action_mask,
            }
        else:
            obs, info = env.reset(seed=args.seed)
            if args.render_java:
                try:
                    env.tribes_env.render(mode="java")
                    time.sleep(max(0.0, args.step_delay_s))
                except Exception as e:
                    print(f"Warning: could not open Java render window: {e}")

        board_w, board_h = get_board_dims_from_env_obs(env)
        if board_w > 0 and board_h > 0:
            print(
                f"Map Bounds: {board_w}x{board_h} "
                f"(valid x: 0..{board_w - 1}, valid y: 0..{board_h - 1})"
            )

        done = False
        step_idx = 0

        while not done:
            legal_actions = env.tribes_env.list_actions()
            action_mask, allowed_indices = env._build_action_mask_and_indices(legal_actions)

            action_mask = safe_array_mask(action_mask, env.action_space.n)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                value_t = agent.get_value(obs_t)
                logits = agent.actor(obs_t)
                masked_logits = logits.masked_fill(mask_t <= 0, -1e8)
                probs = torch.softmax(masked_logits, dim=-1)
                dist = Categorical(logits=masked_logits)
                action_t = dist.sample()

            action = int(action_t.item())
            value = float(value_t.squeeze().detach().cpu().item())
            probs_np = probs.squeeze(0).detach().cpu().numpy()

            turn = info.get("turn_count", getattr(env, "_turn_count", "NA")) if isinstance(info, dict) else "NA"
            spt = compute_current_spt(env)
            spt_text = f"{spt:.2f}" if spt is not None else "NA"

            print("\n" + "-" * 100)
            print(f"Step {step_idx} | Turn {turn} | SPT {spt_text} | Critic Value {value:.4f}")
            print("Legal Action Probabilities:")

            if not allowed_indices:
                print("  (No allowed actions after whitelist filtering.)")
            else:
                print_move_bounds_sanity(env, legal_actions, allowed_indices)
                for pos, raw_idx in enumerate(allowed_indices):
                    if pos >= len(probs_np):
                        continue
                    if action_mask[pos] <= 0:
                        continue
                    act = legal_actions[raw_idx] if raw_idx < len(legal_actions) else {}
                    act_type = str(act.get("type", "UNKNOWN"))
                    raw_repr = str(act.get("repr", act_type))
                    act_repr = format_action_for_debug(raw_repr, act_type, env)
                    chosen = "  <-- chosen" if pos == action else ""
                    print(f"  [{pos:03d}] {format_pct(float(probs_np[pos]))} | {act_type:16s} | {act_repr}{chosen}")
                print_policy_move_grid(env, legal_actions, allowed_indices, action_mask, probs_np, action)

            if args.manual_step:
                user_in = input("Press Enter for next action ('q' + Enter to quit): ").strip().lower()
                if user_in in ("q", "quit", "exit"):
                    print("Stopped by user.")
                    break

            chosen_raw_idx = None
            chosen_action_label = "UNKNOWN"
            chosen_move_unit = None
            chosen_move_dest = None
            if allowed_indices:
                selected_allowed_pos = action % len(allowed_indices)
                chosen_raw_idx = allowed_indices[selected_allowed_pos]
                if 0 <= chosen_raw_idx < len(legal_actions):
                    chosen_act = legal_actions[chosen_raw_idx]
                    chosen_type = str(chosen_act.get("type", "UNKNOWN"))
                    chosen_raw_repr = str(chosen_act.get("repr", chosen_type))
                    chosen_action_label = format_action_for_debug(chosen_raw_repr, chosen_type, env)
                    if chosen_type == "MOVE":
                        parsed = parse_move_action_repr(chosen_raw_repr)
                        if parsed is not None:
                            chosen_move_unit, dest_x, dest_y = parsed
                            chosen_move_dest = (int(dest_x), int(dest_y))

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)

            print(f"Executed Action: idx={action} raw_idx={chosen_raw_idx} | {chosen_action_label}")
            if chosen_move_unit is not None and chosen_move_dest is not None:
                post_pos = get_unit_pos_from_env_obs(env, int(chosen_move_unit))
                board_w, board_h = get_board_dims_from_env_obs(env)
                in_bounds_post = False
                if post_pos is not None and board_w > 0 and board_h > 0:
                    in_bounds_post = 0 <= int(post_pos[0]) < board_w and 0 <= int(post_pos[1]) < board_h
                dest_match = post_pos == chosen_move_dest if post_pos is not None else False
                print(
                    "MOVE_VERIFY: "
                    f"unit={chosen_move_unit} "
                    f"requested={chosen_move_dest[0]}:{chosen_move_dest[1]} "
                    f"actual={post_pos[0]}:{post_pos[1] if post_pos is not None else '?'} "
                    f"dest_match={dest_match} "
                    f"actual_in_bounds={in_bounds_post} "
                    f"board={board_w}x{board_h}"
                    if post_pos is not None
                    else (
                        "MOVE_VERIFY: "
                        f"unit={chosen_move_unit} requested={chosen_move_dest[0]}:{chosen_move_dest[1]} "
                        f"actual=missing dest_match=False actual_in_bounds=False board={board_w}x{board_h}"
                    )
                )
            print(f"Reward: {float(reward):.4f} | terminated={terminated} truncated={truncated}")
            print_reward_breakdown(next_info, float(reward), truncated=bool(truncated))

            if args.render_java:
                try:
                    env.tribes_env.render(mode="java")
                    time.sleep(max(0.0, args.step_delay_s))
                except Exception as e:
                    print(f"Warning: Java render update failed at step {step_idx}: {e}")

            obs = next_obs
            info = next_info if isinstance(next_info, dict) else {}
            step_idx += 1

        print("\n" + "=" * 100)
        print("Episode finished.")
        final_turn = info.get("turn_count", getattr(env, "_turn_count", "NA")) if isinstance(info, dict) else "NA"
        final_city_count = info.get("city_count", "NA") if isinstance(info, dict) else "NA"
        final_spt = info.get("spt", compute_current_spt(env)) if isinstance(info, dict) else compute_current_spt(env)
        final_spt_text = f"{float(final_spt):.2f}" if final_spt is not None else "NA"
        print(f"Final Turn: {final_turn} | Final City Count: {final_city_count} | Final SPT: {final_spt_text}")
        print("=" * 100)

    finally:
        env.close()


if __name__ == "__main__":
    main()
