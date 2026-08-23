#!/usr/bin/env python3
"""Deterministic SPS / Bridge Throughput Benchmark.

Measures environment transitions/second (environment SPS) across three bridge configurations
on genuine Phase 1 training maps only:
  A. Baseline: Legacy legal action fetch + legacy Py4J metadata calls
  B. Batch only: Batched legal action fetch + legacy Py4J metadata calls
  C. Batch + Derived Metadata: Batched legal action fetch + observation-derived metadata

Uses identical warmup, seeds, maps, and deterministic action selection policy.
"""

import argparse
import glob
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pol_env.Tribes.py.register_env import TribesGymWrapper


def run_benchmark_run(
    batch_legal: bool,
    derive_metadata: bool,
    warmup_steps: int,
    measure_steps: int,
    seed: int,
    level_pool_glob: str,
    info_mode: str = "fast",
    enable_profiling: bool = False,
) -> Dict[str, Any]:
    # Configure environment
    os.environ["POLYVISION_LEVEL_POOL_GLOB"] = level_pool_glob
    os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"
    os.environ["POLYVISION_INFO_MODE"] = info_mode
    os.environ["POLYVISION_BATCH_LEGAL_ACTION_FETCH"] = "1" if batch_legal else "0"
    os.environ["POLYVISION_DERIVE_OBS_METADATA"] = "1" if derive_metadata else "0"
    os.environ["POLYVISION_BATCH_LEGAL_FETCH_EQUIV_CHECK"] = "0"
    os.environ["POLYVISION_OBS_METADATA_EQUIV_CHECK"] = "0"
    os.environ["POLYVISION_PROFILE_SPS"] = "1" if enable_profiling else "0"

    env = TribesGymWrapper()
    rng = random.Random(seed)

    # Warmup
    obs, info = env.reset(seed=seed)
    for _ in range(warmup_steps):
        valid_ids = np.flatnonzero(info["action_mask"]) if "action_mask" in info else None
        if valid_ids is None or len(valid_ids) == 0:
            legal_ids = info.get("legal_global_ids_padded", [])
            valid_mask = info.get("legal_action_valid_mask", [])
            if len(valid_mask) > 0:
                valid_ids = [legal_ids[i] for i, v in enumerate(valid_mask) if v]
            else:
                valid_ids = [0]
        action = int(rng.choice(valid_ids))
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            obs, info = env.reset()

    # Timed Measurement
    t_start = time.perf_counter()
    transitions = 0
    reset_count = 0

    accumulated_profile = {
        "java_step_call_s": 0.0,
        "java_observation_fetch_s": 0.0,
        "java_response_parse_s": 0.0,
        "java_legal_fetch_s": 0.0,
        "python_legal_parse_s": 0.0,
        "java_metadata_calls_s": 0.0,
    }

    while transitions < measure_steps:
        # Determine valid action
        valid_ids = np.flatnonzero(info["action_mask"]) if "action_mask" in info else None
        if valid_ids is None or len(valid_ids) == 0:
            legal_ids = info.get("legal_global_ids_padded", [])
            valid_mask = info.get("legal_action_valid_mask", [])
            if len(valid_mask) > 0:
                valid_ids = [legal_ids[i] for i, v in enumerate(valid_mask) if v]
            else:
                valid_ids = [0]

        action = int(rng.choice(valid_ids))
        obs, reward, term, trunc, info = env.step(action)
        transitions += 1

        if enable_profiling:
            p_step = getattr(env.tribes_env, "_last_step_profile", {}) or {}
            p_la = getattr(env.tribes_env, "_last_list_actions_profile", {}) or {}

            accumulated_profile["java_step_call_s"] += float(p_step.get("java_step_call_s", 0.0))
            accumulated_profile["java_observation_fetch_s"] += float(p_step.get("java_observation_fetch_s", 0.0))
            accumulated_profile["java_response_parse_s"] += float(p_step.get("java_response_parse_s", 0.0))
            accumulated_profile["java_legal_fetch_s"] += float(p_la.get("java_compute_bridge_s", 0.0))
            accumulated_profile["python_legal_parse_s"] += float(
                p_la.get("python_list_materialize_s", 0.0) + p_la.get("python_json_parse_s", 0.0)
            )
            accumulated_profile["java_metadata_calls_s"] += float(
                p_step.get("java_done_fetch_s", 0.0)
                + p_step.get("java_scores_fetch_s", 0.0)
                + p_step.get("java_tick_fetch_s", 0.0)
                + p_step.get("java_active_tribe_fetch_s", 0.0)
            )

        if term or trunc:
            obs, info = env.reset()
            reset_count += 1

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    sps = float(transitions) / elapsed if elapsed > 0 else 0.0

    if hasattr(env, "close"):
        env.close()

    return {
        "transitions": transitions,
        "elapsed_s": elapsed,
        "sps": sps,
        "resets": reset_count,
        "profile": accumulated_profile,
    }


def main():
    parser = argparse.ArgumentParser(description="Deterministic Bridge Throughput Benchmark")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup transitions per configuration")
    parser.add_argument("--transitions", type=int, default=1000, help="Measured transitions per repetition")
    parser.add_argument("--reps", type=int, default=3, help="Number of repetitions per configuration")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument(
        "--level-pool-glob",
        default="levels/phase1_pool_bardur_real/train/*.csv",
        help="Glob for training maps",
    )
    parser.add_argument("--info-mode", default="fast", help="Environment info mode (fast/train/debug)")
    parser.add_argument("--profile-timings", action="store_true", default=True, help="Record timing breakdowns")
    args = parser.parse_args()

    print("=" * 80)
    print("POLYVISION PRE-TRAINING THROUGHPUT BENCHMARK (TRAINING POOL ONLY)")
    print(f"Level Pool:          {args.level_pool_glob}")
    print(f"Info Mode:           {args.info_mode}")
    print(f"Warmup Transitions:  {args.warmup}")
    print(f"Measured Steps/Rep:  {args.transitions}")
    print(f"Repetitions:         {args.reps}")
    print(f"Base Seed:           {args.seed}")
    print("=" * 80)

    configs = [
        ("A. Legacy (Batch=0, Derive=0)", False, False),
        ("B. Batch Only (Batch=1, Derive=0)", True, False),
        ("C. Batch + Derived Meta (Batch=1, Derive=1)", True, True),
    ]

    all_results = {}

    for name, batch_flag, derive_flag in configs:
        print(f"\nEvaluating Configuration: {name} ...")
        rep_sps = []
        rep_times = []
        profiles = []

        for rep in range(args.reps):
            rep_seed = int(args.seed + rep * 1000)
            res = run_benchmark_run(
                batch_legal=batch_flag,
                derive_metadata=derive_flag,
                warmup_steps=args.warmup,
                measure_steps=args.transitions,
                seed=rep_seed,
                level_pool_glob=args.level_pool_glob,
                info_mode=args.info_mode,
                enable_profiling=args.profile_timings,
            )
            rep_sps.append(res["sps"])
            rep_times.append(res["elapsed_s"])
            profiles.append(res["profile"])
            print(f"  Rep {rep+1}/{args.reps}: {res['transitions']} transitions in {res['elapsed_s']:.3f}s -> {res['sps']:.2f} env transitions/sec")

        mean_sps = float(np.mean(rep_sps))
        std_sps = float(np.std(rep_sps))
        mean_time = float(np.mean(rep_times))

        # Average profile
        avg_profile = {}
        if profiles:
            for k in profiles[0]:
                avg_profile[k] = float(np.mean([p[k] for p in profiles]))

        all_results[name] = {
            "mean_sps": mean_sps,
            "std_sps": std_sps,
            "mean_time": mean_time,
            "profile": avg_profile,
            "rep_sps": rep_sps,
        }

    baseline_sps = all_results["A. Legacy (Batch=0, Derive=0)"]["mean_sps"]
    batch_only_sps = all_results["B. Batch Only (Batch=1, Derive=0)"]["mean_sps"]
    optimized_sps = all_results["C. Batch + Derived Meta (Batch=1, Derive=1)"]["mean_sps"]

    speedup_b_vs_a = (batch_only_sps / baseline_sps - 1.0) * 100.0 if baseline_sps > 0 else 0.0
    speedup_c_vs_a = (optimized_sps / baseline_sps - 1.0) * 100.0 if baseline_sps > 0 else 0.0
    speedup_c_vs_b = (optimized_sps / batch_only_sps - 1.0) * 100.0 if batch_only_sps > 0 else 0.0

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY (ENVIRONMENT TRANSITIONS / SEC)")
    print("=" * 80)
    print(f"A. Baseline (Legacy Fetch + Legacy Metadata):       {baseline_sps:7.2f} ± {all_results['A. Legacy (Batch=0, Derive=0)']['std_sps']:.2f} env transitions/sec")
    print(f"B. Batch Only (Batch Fetch + Legacy Metadata):       {batch_only_sps:7.2f} ± {all_results['B. Batch Only (Batch=1, Derive=0)']['std_sps']:.2f} env transitions/sec  (+{speedup_b_vs_a:+.2f}%)")
    print(f"C. Fully Optimized (Batch Fetch + Derived Metadata): {optimized_sps:7.2f} ± {all_results['C. Batch + Derived Meta (Batch=1, Derive=1)']['std_sps']:.2f} env transitions/sec  (+{speedup_c_vs_a:+.2f}% vs A, +{speedup_c_vs_b:+.2f}% vs B)")
    print("-" * 80)

    if args.profile_timings:
        print("\nAVERAGE TIME BREAKDOWN PER CONFIGURATION (over measured steps):")
        header = f"{'Component':<35s} | {'A. Baseline':<12s} | {'B. Batch Only':<12s} | {'C. Optimized':<12s}"
        print(header)
        print("-" * len(header))
        keys = [
            ("Java stepByIndex", "java_step_call_s"),
            ("Observation fetch", "java_observation_fetch_s"),
            ("Observation JSON parse", "java_response_parse_s"),
            ("Legal action fetch", "java_legal_fetch_s"),
            ("Legal action parse/materialize", "python_legal_parse_s"),
            ("Metadata bridge calls (4x)", "java_metadata_calls_s"),
        ]
        for label, k in keys:
            t_a = all_results["A. Legacy (Batch=0, Derive=0)"]["profile"].get(k, 0.0)
            t_b = all_results["B. Batch Only (Batch=1, Derive=0)"]["profile"].get(k, 0.0)
            t_c = all_results["C. Batch + Derived Meta (Batch=1, Derive=1)"]["profile"].get(k, 0.0)
            print(f"{label:<35s} | {t_a:10.4f}s  | {t_b:10.4f}s  | {t_c:10.4f}s ")
        print("=" * 80)


if __name__ == "__main__":
    main()
