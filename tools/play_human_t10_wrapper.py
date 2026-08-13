#!/usr/bin/env python3
"""Ad hoc terminal play through the PPO-facing Phase 1 policy interface.

Use ``tools/human_benchmark.py`` for permanent official benchmark evidence.
This utility shares the same global-ID menu and ``env.step(global_id)`` path,
but permits explicitly acknowledged diagnostic renderers for ad hoc debugging.
Those renderers are prohibited in official benchmark mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol_env.Tribes.py.register_env import TribesGymWrapper
from tools.human_policy_interface import run_policy_visible_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--level-pool-glob",
        default="levels/phase1_pool_bardur_real/human_benchmark/*.csv",
        help="Pool relative to pol_env/Tribes or an absolute glob",
    )
    parser.add_argument("--level-selection-mode", default="round_robin", choices=["round_robin", "seeded_random"])
    parser.add_argument("--page-size", type=int, default=30)
    parser.add_argument("--save-json", action="store_true", help="Save an ad hoc run under outputs/human_wrapper_runs")
    parser.add_argument("--auto-random", action="store_true", help="Synthetic ad hoc smoke; never benchmark evidence")
    parser.add_argument("--unsafe-debug-ui", action="store_true", help="Acknowledge that diagnostic renderers are not benchmark-safe")
    parser.add_argument("--show-ansi-map", action="store_true", help="Unsafe ad hoc Java-observation ANSI renderer")
    parser.add_argument("--render-java", action="store_true", help="Unsafe ad hoc full Java Swing renderer")
    parser.add_argument("--step-delay-s", type=float, default=0.1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (args.show_ansi_map or args.render_java) and not args.unsafe_debug_ui:
        raise SystemExit("--show-ansi-map/--render-java require --unsafe-debug-ui and are forbidden for official benchmarks")

    os.environ["POLYVISION_LEVEL_POOL_GLOB"] = str(args.level_pool_glob)
    os.environ["POLYVISION_LEVEL_SELECTION_MODE"] = str(args.level_selection_mode)
    os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"
    os.environ["POLYVISION_INFO_MODE"] = "fast"
    os.environ["POLYVISION_MAX_LEGAL_ACTIONS"] = str(TribesGymWrapper.MAX_LEGAL_ACTIONS_DEFAULT)

    rng = np.random.default_rng(int(args.seed))

    def random_selector(actions, _observation, _info, _step):
        return int(actions[int(rng.integers(0, len(actions)))]["global_id"])

    def unsafe_callback(env, _observation, _info):
        if args.show_ansi_map:
            print(env.tribes_env.render(mode="ansi"))
        if args.render_java:
            env.tribes_env.render(mode="java")
            time.sleep(max(0.0, float(args.step_delay_s)))

    env = TribesGymWrapper()
    try:
        result = run_policy_visible_episode(
            env,
            episode_seed=int(args.seed),
            official=not args.unsafe_debug_ui,
            page_size=int(args.page_size),
            selector=random_selector if args.auto_random else None,
            state_callback=unsafe_callback if args.unsafe_debug_ui else None,
        )
    finally:
        env.close()

    print(f"Episode status: {result.status}")
    if result.status == "completed":
        print(f"Final T10 SPT: {result.final_info_metrics.get('terminal_final_spt', result.final_info_metrics.get('spt'))}")
    if args.save_json:
        output = REPO_ROOT / "outputs" / "human_wrapper_runs"
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"adhoc_{result.started_at_utc.replace(':', '').replace('-', '')}.json"
        path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
        print(f"Saved ad hoc run: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
