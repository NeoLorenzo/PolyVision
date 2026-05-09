import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np


_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import pol_env.Tribes.py.register_env as register_env  # noqa: F401
except Exception:
    pass


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


def _first_valid_gid(info: Dict[str, Any]) -> int:
    ids = np.asarray(info.get("legal_global_ids_padded", []), dtype=np.int64).reshape(-1)
    valid = np.asarray(info.get("legal_action_valid_mask", []), dtype=bool).reshape(-1)
    n = min(ids.shape[0], valid.shape[0])
    if n <= 0:
        raise RuntimeError("No legal action tensors available.")
    idxs = np.where(valid[:n])[0]
    if idxs.size <= 0:
        raise RuntimeError("No valid legal actions in mask.")
    return int(ids[int(idxs[0])])


@dataclass
class EpisodeTrace:
    seed: int
    rewards: List[float]
    terminal_bonus_series: List[float]
    terminal_spt_series: List[Any]
    terminal_base_series: List[float]
    terminal_over10_series: List[float]
    terminal_over15_series: List[float]
    final_info: Dict[str, Any]
    final_step_reward: float


def _run_episode(env, seed: int) -> EpisodeTrace:
    _obs, info = env.reset(seed=int(seed))
    done = False
    rewards: List[float] = []
    terminal_bonus_series: List[float] = []
    terminal_spt_series: List[Any] = []
    terminal_base_series: List[float] = []
    terminal_over10_series: List[float] = []
    terminal_over15_series: List[float] = []
    final_info: Dict[str, Any] = {}
    final_step_reward = 0.0

    step_guard = 0
    while not done:
        if not isinstance(info, dict):
            raise RuntimeError("Expected dict info payload.")
        gid = _first_valid_gid(info)
        _obs, reward, terminated, truncated, info = env.step(gid)
        done = bool(terminated or truncated)
        rewards.append(float(reward))
        if isinstance(info, dict):
            terminal_bonus_series.append(float(_as_scalar(info.get("terminal_spt_bonus", 0.0), default=0.0)))
            terminal_spt_series.append(info.get("terminal_final_spt", None))
            terminal_base_series.append(float(_as_scalar(info.get("terminal_spt_base_component", 0.0), default=0.0)))
            terminal_over10_series.append(float(_as_scalar(info.get("terminal_spt_over_10_component", 0.0), default=0.0)))
            terminal_over15_series.append(float(_as_scalar(info.get("terminal_spt_over_15_component", 0.0), default=0.0)))
            final_info = info
        final_step_reward = float(reward)
        step_guard += 1
        if step_guard > 5000:
            raise RuntimeError("Episode safety cap exceeded.")

    return EpisodeTrace(
        seed=int(seed),
        rewards=rewards,
        terminal_bonus_series=terminal_bonus_series,
        terminal_spt_series=terminal_spt_series,
        terminal_base_series=terminal_base_series,
        terminal_over10_series=terminal_over10_series,
        terminal_over15_series=terminal_over15_series,
        final_info=final_info,
        final_step_reward=float(final_step_reward),
    )


def _set_common_env_vars() -> None:
    os.environ["POLYVISION_LEVEL_POOL_GLOB"] = "levels/phase1_pool/*.csv"
    os.environ["POLYVISION_LEVEL_SELECTION_MODE"] = "round_robin"
    os.environ["POLYVISION_INFO_MODE"] = "fast"
    os.environ["POLYVISION_STRICT_COORD_ASSERT"] = "0"
    os.environ["POLYVISION_MAX_LEGAL_ACTIONS"] = "1024"


def _make_env(terminal_enabled: bool, w_base: float, w10: float, w15: float):
    _set_common_env_vars()
    os.environ["POLYVISION_TERMINAL_SPT_REWARD_ENABLED"] = "1" if terminal_enabled else "0"
    os.environ["POLYVISION_TERMINAL_SPT_BASE_WEIGHT"] = str(float(w_base))
    os.environ["POLYVISION_TERMINAL_SPT_OVER_10_WEIGHT"] = str(float(w10))
    os.environ["POLYVISION_TERMINAL_SPT_OVER_15_WEIGHT"] = str(float(w15))
    return gym.make("Tribes-v0")


def _expected_bonus(final_spt: float, w_base: float, w10: float, w15: float) -> float:
    return (
        float(w_base) * float(final_spt)
        + float(w10) * max(0.0, float(final_spt) - 10.0)
        + float(w15) * max(0.0, float(final_spt) - 15.0)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test terminal SPT reward (disabled vs enabled).")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--over10-weight", type=float, default=2.0)
    parser.add_argument("--over15-weight", type=float, default=3.0)
    args = parser.parse_args()

    env_off = _make_env(False, args.base_weight, args.over10_weight, args.over15_weight)
    env_on = _make_env(True, args.base_weight, args.over10_weight, args.over15_weight)

    try:
        for ep in range(int(args.episodes)):
            seed = int(args.seed) + int(ep)
            off_trace = _run_episode(env_off, seed=seed)
            on_trace = _run_episode(env_on, seed=seed)

            if len(off_trace.rewards) != len(on_trace.rewards):
                raise RuntimeError(
                    f"Episode length mismatch at seed={seed}: off={len(off_trace.rewards)} on={len(on_trace.rewards)}"
                )
            if len(on_trace.rewards) < 1:
                raise RuntimeError(f"Empty episode at seed={seed}")

            # Non-terminal reward parity.
            non_terminal_same = True
            for i in range(len(on_trace.rewards) - 1):
                if abs(float(off_trace.rewards[i]) - float(on_trace.rewards[i])) > 1e-7:
                    non_terminal_same = False
                    raise RuntimeError(
                        f"Non-terminal reward mismatch seed={seed} step={i} "
                        f"off={off_trace.rewards[i]:.6f} on={on_trace.rewards[i]:.6f}"
                    )

            # Disabled run: never gets terminal bonus.
            off_bonus_count = sum(1 for v in off_trace.terminal_bonus_series if abs(float(v)) > 1e-9)
            if off_bonus_count != 0:
                raise RuntimeError(f"Disabled mode has non-zero terminal bonus count={off_bonus_count} seed={seed}")

            # Enabled run: bonus appears exactly once, on terminal step.
            on_bonus_nonzero_idx = [i for i, v in enumerate(on_trace.terminal_bonus_series) if abs(float(v)) > 1e-9]
            if len(on_bonus_nonzero_idx) != 1:
                raise RuntimeError(
                    f"Enabled mode terminal bonus count invalid seed={seed}: {len(on_bonus_nonzero_idx)}"
                )
            if on_bonus_nonzero_idx[0] != len(on_trace.rewards) - 1:
                raise RuntimeError(
                    f"Enabled mode terminal bonus not on final step seed={seed} idx={on_bonus_nonzero_idx[0]}"
                )

            final_spt = float(_as_scalar(on_trace.final_info.get("spt", 0.0), default=0.0))
            expected = _expected_bonus(final_spt, args.base_weight, args.over10_weight, args.over15_weight)
            observed = float(on_trace.terminal_bonus_series[-1])
            if abs(observed - expected) > 1e-6:
                raise RuntimeError(
                    f"Terminal bonus mismatch seed={seed}: observed={observed:.6f} expected={expected:.6f}"
                )

            # Terminal step decomposition checks and total-reward consistency.
            c_base = float(on_trace.terminal_base_series[-1])
            c10 = float(on_trace.terminal_over10_series[-1])
            c15 = float(on_trace.terminal_over15_series[-1])
            if abs((c_base + c10 + c15) - observed) > 1e-6:
                raise RuntimeError(
                    f"Component sum mismatch seed={seed}: sum={c_base+c10+c15:.6f} observed={observed:.6f}"
                )
            off_final = float(off_trace.rewards[-1])
            on_final = float(on_trace.rewards[-1])
            if abs((on_final - off_final) - observed) > 1e-6:
                raise RuntimeError(
                    f"Terminal reward delta mismatch seed={seed}: on-off={on_final-off_final:.6f} bonus={observed:.6f}"
                )

            base_terminal_reward = float(on_final - observed)
            print(
                f"seed={seed} "
                f"final_spt={final_spt:.3f} "
                f"base_terminal_reward={base_terminal_reward:.3f} "
                f"terminal_bonus={observed:.3f} "
                f"terminal_total_reward={on_final:.3f} "
                f"non_terminal_unchanged={non_terminal_same}"
            )

        print("SMOKE_TEST_PASS")
    finally:
        env_off.close()
        env_on.close()


if __name__ == "__main__":
    main()
