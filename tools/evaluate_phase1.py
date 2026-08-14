#!/usr/bin/env python
"""Canonical, policy-visible PolyVision Phase 1 batch evaluator."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import gymnasium
import numpy as np
import py4j
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol_env.Tribes.py.environment_contract import (  # noqa: E402
    environment_compatibility_metadata, read_checkpoint_metadata, validate_checkpoint_compatibility,
)
from pol_env.Tribes.py.register_env import TribesGymWrapper  # noqa: E402
from py_rl.cleanrl.cleanrl.ppo import Agent  # noqa: E402
from tools.phase1_eval_core import (  # noqa: E402
    ALL_POLICIES, SCHEMA_VERSION, PPOPolicy, RandomLegalPolicy, VisibleGreedyPolicy,
    aggregate_results, build_schedule, comparison_rows, json_safe, load_verified_pool,
    paired_stats, sha256_file, stable_seed, validate_schedule, write_csv,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def command_output(args):
    try:
        result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=10)
        return (result.stdout.strip() or result.stderr.strip() or None)
    except Exception:
        return None


def runtime_provenance() -> dict:
    status = command_output(["git", "status", "--porcelain"])
    return {
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "gymnasium": gymnasium.__version__, "py4j": getattr(py4j, "__version__", None),
        "java": command_output(["java", "-version"]),
    }


def make_policy(name, agent, device, policy_seed):
    if name == "ppo_argmax": return PPOPolicy(agent, device, False, None)
    if name == "ppo_sampled": return PPOPolicy(agent, device, True, policy_seed)
    if name == "random_legal": return RandomLegalPolicy(policy_seed)
    if name == "visible_greedy": return VisibleGreedyPolicy()
    raise ValueError(name)


OPTIONAL_METRICS = (
    "turn_second_city_captured", "fog_tiles_cleared_total", "turn_first_uncaptured_village_visible",
    "captured_villages_t10", "capturable_villages_total", "village_capture_pct_t10",
    "techs_researched", "organization_researched", "forestry_researched",
    "turn_organization_researched", "turn_forestry_researched", "animals_harvested_t10",
    "fruit_harvested_t10", "lumber_huts_built_t10", "sawmills_built_t10", "forests_cleared_t10",
    "illegal_sample_count", "fallback_end_turn_count", "illegal_sample_rate", "fallback_end_turn_rate",
)
REQUIRED_EPISODE_FIELDS = {
    "policy", "map_filename", "map_identity", "replicate_index", "episode_seed", "policy_seed",
    "final_spt_t10", "final_stars", "final_city_count", "final_unit_count", "total_shaped_return",
    "policy_decisions", "episode_duration_seconds", "contract_valid",
}


def run_episode(env, policy, record, max_steps: int) -> dict:
    # Scheduling is runner-only. Policies never receive or inspect these private fields.
    env._level_pool = [record.map_path]
    env._level_pool_size = 1
    started = time.perf_counter()
    initial_illegal_count = int(getattr(env, "_illegal_sample_count", 0))
    initial_fallback_count = int(getattr(env, "_fallback_end_turn_count", 0))
    obs, info = env.reset(seed=record.episode_seed)
    actual_path = str(Path(env._current_level_file).resolve())
    if actual_path != str(Path(record.map_path).resolve()):
        raise RuntimeError(f"Scheduled map mismatch: expected={record.map_path} actual={actual_path}")
    if sha256_file(Path(actual_path)) != record.csv_sha256:
        raise RuntimeError(f"Runtime map hash mismatch: {actual_path}")

    total_return, decisions = 0.0, 0
    terminated = truncated = False
    fallback_seen = illegal_seen = False
    while not (terminated or truncated):
        ids = np.asarray(info["legal_global_ids_padded"], dtype=np.int64).reshape(-1)
        valid = np.asarray(info["legal_action_valid_mask"], dtype=bool).reshape(-1)
        chosen_gid, chosen_slot = policy.choose_action(obs, info)
        if chosen_slot < 0 or chosen_slot >= ids.size or not valid[chosen_slot]:
            raise RuntimeError("Policy selected invalid/padded slot")
        if int(ids[chosen_slot]) != int(chosen_gid):
            raise RuntimeError("Chosen slot/global-ID mapping mismatch")
        if int(chosen_gid) not in set(ids[valid].tolist()):
            raise RuntimeError("Chosen ID absent from policy-visible valid legal set")
        obs, reward, terminated, truncated, info = env.step(int(chosen_gid))
        total_return += float(reward); decisions += 1
        selected = info.get("selected_global_id")
        if selected is not None and int(selected) != int(chosen_gid):
            raise RuntimeError(f"Environment selected_global_id mismatch: {selected} != {chosen_gid}")
        fallback_seen |= bool(info.get("fallback_to_end_turn", False))
        illegal_seen |= bool(info.get("illegal_sampled_global_id", False))
        if fallback_seen or illegal_seen:
            raise RuntimeError("Unexpected illegal-action/fallback path during official evaluation")
        if decisions > max_steps:
            raise RuntimeError(f"Episode exceeded safety cap of {max_steps} decisions")

    turn_count = int(info.get("turn_count", info.get("turn", -1)))
    contract_valid = bool(truncated and not terminated and turn_count == TribesGymWrapper.MAX_TURNS + 1)
    if not contract_valid:
        raise RuntimeError(f"Episode did not satisfy Turn-10 contract: terminated={terminated} truncated={truncated} turn={turn_count}")
    result = {
        "schema_version": SCHEMA_VERSION, "policy": record.policy,
        "map_filename": record.map_filename, "map_path": record.map_path,
        "map_identity": record.map_identity, "map_csv_sha256": record.csv_sha256,
        "replicate_index": record.replicate_index, "episode_seed": record.episode_seed,
        "policy_seed": record.policy_seed, "final_spt_t10": float(info["spt"]),
        "final_stars": float(info["stars"]) if info.get("stars") is not None else None,
        "final_city_count": int(info["city_count"]) if info.get("city_count") is not None else None,
        "final_unit_count": int(info["unit_count"]) if info.get("unit_count") is not None else None,
        "total_shaped_return": float(total_return), "policy_decisions": decisions,
        "terminated": bool(terminated), "truncated": bool(truncated), "final_turn_count": turn_count,
        "second_city_captured_by_t10": (int(info.get("turn_second_city_captured", -1)) >= 0),
        "episode_duration_seconds": float(time.perf_counter()-started),
        "selected_actions_valid": True, "selected_global_ids_matched": True,
        "no_policy_visible_fallback": not fallback_seen, "turn10_contract_valid": contract_valid,
        "map_schedule_valid": True, "contract_valid": True,
    }
    for key in OPTIONAL_METRICS:
        result[key] = info.get(key, None)
    result["illegal_sample_count"] = int(info.get("illegal_sample_count", initial_illegal_count)) - initial_illegal_count
    result["fallback_end_turn_count"] = int(info.get("fallback_end_turn_count", initial_fallback_count)) - initial_fallback_count
    result["illegal_sample_rate"] = float(result["illegal_sample_count"] / max(1, decisions))
    result["fallback_end_turn_rate"] = float(result["fallback_end_turn_count"] / max(1, decisions))
    return json_safe(result)


def write_outputs(out_dir, config, episodes, summary, per_map, comparisons):
    for index, row in enumerate(episodes):
        missing = sorted(REQUIRED_EPISODE_FIELDS - set(row))
        if missing:
            raise RuntimeError(f"Episode {index} missing required serialized fields: {missing}")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir/"config.json").write_text(json.dumps(json_safe(config),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    with (out_dir/"episodes.jsonl").open("w",encoding="utf-8") as f:
        for row in episodes: f.write(json.dumps(json_safe(row),sort_keys=True)+"\n")
    write_csv(out_dir/"per_map.csv",per_map)
    (out_dir/"summary.json").write_text(json.dumps(json_safe(summary),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    flat=[]
    for p,v in summary["policies"].items():
        h=v[v["headline_level"]]
        flat.append({"policy":p,"maps":v["maps"],"episodes":v["episodes"],"headline_level":v["headline_level"],
                     "mean_spt":h["mean"],"median_spt":h["median"],"stddev_spt":h["stddev"],
                     "min_spt":h["min"],"max_spt":h["max"],"p05_spt":h["p05"],"p25_spt":h["p25"],
                     "p75_spt":h["p75"],"p95_spt":h["p95"],"ci95_low":h["mean_ci95"][0],"ci95_high":h["mean_ci95"][1]})
    write_csv(out_dir/"summary.csv",flat); write_csv(out_dir/"comparison.csv",comparisons)


def print_summary(pool, map_count, summary, out_dir):
    print("\n"+"="*60+f"\nPOLYVISION PHASE 1 — {pool.upper()} EVALUATION\n"+"="*60)
    print(f"Pool: {pool}\nMaps: {map_count}\n")
    print(f"{'Policy':22s} {'Episodes':>8s} {'Mean SPT':>10s} {'Median':>8s} {'95% CI':>20s}")
    print("-"*72)
    for p,v in summary["policies"].items():
        h=v[v["headline_level"]]; ci=h["mean_ci95"]
        print(f"{p:22s} {v['episodes']:8d} {h['mean']:10.2f} {h['median']:8.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")
    for key,v in summary["paired"].items():
        if v: print(f"\n{key}: {v['wins']} W / {v['ties']} T / {v['losses']} L; mean delta {v['mean_difference']:+.2f} SPT")
    print(f"\nResults: {out_dir}")


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path",type=Path,help="Checkpoint required for PPO policies")
    p.add_argument("--pool",choices=["validation","test","human_benchmark"],default="validation")
    p.add_argument("--confirm-test",action="store_true",help="Required explicit opt-in for pristine test evaluation")
    p.add_argument("--suite",choices=["full"],default="full")
    p.add_argument("--policy",action="append",choices=ALL_POLICIES,help="Repeat for a narrower policy set")
    p.add_argument("--repeats-per-map",type=int,default=5)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--max-maps",type=int)
    p.add_argument("--device",default="cpu"); p.add_argument("--max-steps-per-episode",type=int,default=256)
    p.add_argument("--progress-every",type=int,default=25); p.add_argument("--output-root",type=Path,default=REPO_ROOT/"outputs/evaluations")
    p.add_argument("--evaluation-id"); return p.parse_args()


def main():
    args=parse_args()
    if args.pool=="test" and not args.confirm_test: raise SystemExit("Refusing pristine test evaluation without --pool test --confirm-test")
    if args.repeats_per_map<1 or (args.max_maps is not None and args.max_maps<1): raise SystemExit("Repeat/map counts must be positive")
    policies=tuple(dict.fromkeys(args.policy or ALL_POLICIES))
    if any(p.startswith("ppo_") for p in policies) and not args.model_path: raise SystemExit("--model-path is required for PPO policies")
    manifest,maps,manifest_path=load_verified_pool(REPO_ROOT,args.pool,args.max_maps)
    canonical=(args.pool=="validation" and len(maps)==250 and args.max_maps is None and args.repeats_per_map==5 and set(policies)==set(ALL_POLICIES))
    label="canonical" if canonical else "partial-smoke-noncanonical"
    evaluation_id=args.evaluation_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{args.pool}_{label}"
    out_dir=args.output_root/evaluation_id
    if out_dir.exists(): raise SystemExit(f"Output directory already exists: {out_dir}")
    schedule=build_schedule(maps,policies,args.repeats_per_map,args.seed)

    model_path=args.model_path.resolve() if args.model_path else None
    checkpoint_meta=read_checkpoint_metadata(str(model_path)) if model_path else None
    sidecar=Path(str(model_path)+".action_interface.json") if model_path else None
    if checkpoint_meta: os.environ["POLYVISION_MAX_LEGAL_ACTIONS"]=str(int(checkpoint_meta["max_legal_actions"]))
    os.environ["POLYVISION_LEVEL_POOL_GLOB"]=f"levels/phase1_pool_bardur_real/{args.pool}/*.csv"
    os.environ["POLYVISION_LEVEL_SELECTION_MODE"]="round_robin"
    os.environ["POLYVISION_INFO_MODE"]="fast"
    os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"]="1"
    device=torch.device(args.device if args.device=="cpu" or torch.cuda.is_available() else "cpu")
    env=TribesGymWrapper(); agent=None
    try:
        env_meta=None
        if checkpoint_meta:
            env_meta=environment_compatibility_metadata(env,actor_mode=checkpoint_meta["actor_mode"],max_legal_actions=checkpoint_meta["max_legal_actions"])
            validate_checkpoint_compatibility(checkpoint_meta,env_meta)
            adapter=SimpleNamespace(single_observation_space=env.observation_space,single_action_space=env.action_space)
            agent=Agent(adapter,actor_mode=checkpoint_meta["actor_mode"],max_legal_actions=checkpoint_meta["max_legal_actions"],legal_action_feature_dim=checkpoint_meta["legal_action_feature_dim"]).to(device)
            agent.load_state_dict(torch.load(model_path,map_location=device)); agent.eval()
        started=utc_now(); episodes=[]
        for i,record in enumerate(schedule,1):
            policy=make_policy(record.policy,agent,device,record.policy_seed)
            episodes.append(run_episode(env,policy,record,args.max_steps_per_episode))
            if args.progress_every and (i%args.progress_every==0 or i==len(schedule)): print(f"{record.policy}: total progress {i}/{len(schedule)}")
        validate_schedule(schedule,maps,policies,args.repeats_per_map)
        policy_summary,per_map=aggregate_results(episodes,args.seed)
        paired={
            "ppo_argmax_vs_visible_greedy":paired_stats(per_map,"ppo_argmax","visible_greedy",args.seed),
            "ppo_sampled_vs_random_legal":paired_stats(per_map,"ppo_sampled","random_legal",args.seed),
            "ppo_argmax_vs_random_legal":paired_stats(per_map,"ppo_argmax","random_legal",args.seed),
        }
        config={"evaluation_schema_version":SCHEMA_VERSION,"evaluation_id":evaluation_id,"status":"complete","canonical":canonical,
                "classification":label,"started_utc":started,"ended_utc":utc_now(),"pool":args.pool,
                "pool_glob":os.environ["POLYVISION_LEVEL_POOL_GLOB"],"pool_identity":manifest["pool_identities"][args.pool],
                "split_manifest_path":str(manifest_path),"split_manifest_sha256":sha256_file(manifest_path),
                "expected_manifest_map_count":manifest["pool_counts"][args.pool],"evaluated_map_count":len(maps),
                "ordered_maps":[m.__dict__ for m in maps],"evaluation_seed":args.seed,
                "schedule_rules":"episode seed is shared by map+replicate across policies; policy seed is policy-specific; exact manifest map is runner-selected and verified",
                "repeats_per_stochastic_map":args.repeats_per_map,"policies":list(policies),
                "policy_semantics":{"ppo_argmax":"highest-logit valid slot","ppo_sampled":"torch.multinomial over authoritative valid-slot probabilities with per-episode generator","random_legal":"uniform policy-visible valid slot","visible_greedy":"deterministic observation/legal-feature heuristic"},
                "model_path":str(model_path) if model_path else None,"checkpoint_sha256":sha256_file(model_path) if model_path else None,
                "sidecar_path":str(sidecar) if sidecar else None,"sidecar_sha256":sha256_file(sidecar) if sidecar else None,
                "checkpoint_metadata":checkpoint_meta,"environment_interface_metadata":env_meta,
                "runtime":runtime_provenance(),"polyvision_environment":{k:v for k,v in os.environ.items() if k.startswith("POLYVISION_")}}
        summary={"evaluation_schema_version":SCHEMA_VERSION,"evaluation_id":evaluation_id,"canonical":canonical,"pool":args.pool,
                 "primary_metric":"final Turn-10 stars per turn","bootstrap":{"samples":5000,"method":"fixed-seed nonparametric percentile bootstrap; stochastic headline bootstraps per-map replicate means"},
                 "policies":policy_summary,"paired":paired}
        write_outputs(out_dir,config,episodes,summary,per_map,comparison_rows(per_map)); print_summary(args.pool,len(maps),summary,out_dir)
    finally: env.close()


if __name__=="__main__": main()
