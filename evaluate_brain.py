import argparse
import glob
import os
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
    args = parser.parse_args()

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

        obs, info = env.reset(seed=args.seed)
        if args.render_java:
            try:
                env.tribes_env.render(mode="java")
                time.sleep(max(0.0, args.step_delay_s))
            except Exception as e:
                print(f"Warning: could not open Java render window: {e}")

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
                for pos, raw_idx in enumerate(allowed_indices):
                    if pos >= len(probs_np):
                        continue
                    if action_mask[pos] <= 0:
                        continue
                    act = legal_actions[raw_idx] if raw_idx < len(legal_actions) else {}
                    act_type = str(act.get("type", "UNKNOWN"))
                    act_repr = str(act.get("repr", act_type))
                    chosen = "  <-- chosen" if pos == action else ""
                    print(f"  [{pos:03d}] {format_pct(float(probs_np[pos]))} | {act_type:16s} | {act_repr}{chosen}")

            if args.manual_step:
                user_in = input("Press Enter for next action ('q' + Enter to quit): ").strip().lower()
                if user_in in ("q", "quit", "exit"):
                    print("Stopped by user.")
                    break

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)

            chosen_raw_idx = None
            chosen_action_label = "UNKNOWN"
            if allowed_indices:
                selected_allowed_pos = action % len(allowed_indices)
                chosen_raw_idx = allowed_indices[selected_allowed_pos]
                if 0 <= chosen_raw_idx < len(legal_actions):
                    chosen_action_label = str(
                        legal_actions[chosen_raw_idx].get("repr", legal_actions[chosen_raw_idx].get("type", "UNKNOWN"))
                    )

            print(f"Executed Action: idx={action} raw_idx={chosen_raw_idx} | {chosen_action_label}")
            print(f"Reward: {float(reward):.4f} | terminated={terminated} truncated={truncated}")

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
