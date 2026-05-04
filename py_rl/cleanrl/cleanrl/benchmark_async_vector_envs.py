#!/usr/bin/env python3
"""
Quick benchmark matrix for AsyncVectorEnv-backed Tribes PPO runs.

Runs ppo.py with num_envs in [12, 16, 20], parses printed SPS lines, and
reports average/peak throughput to help pick a sweet spot on local hardware.
"""

import os
import re
import subprocess
import sys
import time
from statistics import mean


SPS_RE = re.compile(r"SPS:\s+(\d+)")


def run_case(repo_root, num_envs):
    ppo_path = os.path.join(repo_root, "py_rl", "cleanrl", "cleanrl", "ppo.py")
    cmd = [
        sys.executable,
        ppo_path,
        "--num-envs",
        str(num_envs),
        "--total-timesteps",
        "30720",
        "--num-steps",
        "128",
        "--startup-jitter-min-s",
        "0.1",
        "--startup-jitter-max-s",
        "2.0",
    ]

    env = os.environ.copy()
    env["POLYVISION_VERBOSE_RESETS"] = "0"

    print(f"\n=== Benchmark case: num_envs={num_envs} ===")
    print(" ".join(cmd))
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    sps_values = []
    for line in proc.stdout:
        sys.stdout.write(line)
        match = SPS_RE.search(line)
        if match:
            sps_values.append(int(match.group(1)))

    ret = proc.wait()
    duration_s = time.time() - start

    if ret != 0:
        raise RuntimeError(f"num_envs={num_envs} failed with exit code {ret}")

    if not sps_values:
        raise RuntimeError(f"num_envs={num_envs} produced no SPS lines; cannot score throughput")

    tail = sps_values[-5:] if len(sps_values) >= 5 else sps_values
    return {
        "num_envs": num_envs,
        "peak_sps": max(sps_values),
        "tail_avg_sps": int(mean(tail)),
        "all_avg_sps": int(mean(sps_values)),
        "duration_s": round(duration_s, 2),
        "samples": len(sps_values),
    }


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidates = [12, 16, 20]
    results = []

    for n in candidates:
        try:
            results.append(run_case(repo_root, n))
        except Exception as e:
            print(f"\nCase num_envs={n} failed: {e}")

    if not results:
        print("\nNo successful benchmark runs.")
        raise SystemExit(1)

    print("\n=== AsyncVectorEnv throughput summary ===")
    print("num_envs | tail_avg_sps | all_avg_sps | peak_sps | duration_s | sps_samples")
    print("-------- | ------------ | ----------- | -------- | ---------- | -----------")
    for r in results:
        print(
            f"{r['num_envs']:>8} | {r['tail_avg_sps']:>12} | {r['all_avg_sps']:>11} | "
            f"{r['peak_sps']:>8} | {r['duration_s']:>10} | {r['samples']:>11}"
        )

    best = max(results, key=lambda x: x["tail_avg_sps"])
    print(f"\nRecommended num_envs based on tail_avg_sps: {best['num_envs']}")


if __name__ == "__main__":
    main()
