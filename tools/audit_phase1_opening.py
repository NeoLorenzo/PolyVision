#!/usr/bin/env python
"""Audit the historical Phase 1 scripted opening; never runs a policy or scores capability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol_env.Tribes.py.register_env import TribesGymWrapper  # noqa: E402

SCHEMA_VERSION = "polyvision-phase1-opening-audit-v1"
POOLS = ("train", "validation", "test", "human_benchmark")
PHASES = (
    "animal_harvest_1", "animal_harvest_2", "workshop_levelup",
    "turn0_starting_warrior_move", "turn0_end_turn",
    "turn1_starting_warrior_move", "turn1_second_warrior_spawn", "turn1_end_turn",
)


class OpeningTraceRecorder:
    def __init__(self): self.rows = []
    def record(self, row): self.rows.append(row)


def utc_now(): return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def load_manifest_maps(pool: str, map_filter: str | None = None, max_maps: int | None = None):
    manifest_path=REPO_ROOT/"pol_env/Tribes/levels/phase1_pool_bardur_real/split_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    selected=set(POOLS if pool=="all" else (pool,))
    rows=[]
    for row in manifest["maps"]:
        if row["pool"] not in selected: continue
        if map_filter and map_filter not in {row["filename"],Path(row["filename"]).stem,row["canonical_map_sha256"]} and not row["canonical_map_sha256"].startswith(map_filter): continue
        path=(manifest_path.parent/row["relative_path"]).resolve()
        if not path.is_file() or sha256_file(path)!=row["csv_sha256"]: raise RuntimeError(f"Manifest/hash mismatch: {path}")
        rows.append({**row,"path":str(path)})
    rows.sort(key=lambda r:(POOLS.index(r["pool"]),r["canonical_map_sha256"]))
    if map_filter and not rows: raise ValueError(f"Map not found in selected pool: {map_filter}")
    if not map_filter and max_maps is None:
        for p in selected:
            actual=sum(r["pool"]==p for r in rows); expected=int(manifest["pool_counts"][p])
            if actual!=expected: raise RuntimeError(f"Pool count mismatch {p}: {actual} != {expected}")
    return manifest, rows[:max_maps] if max_maps else rows, manifest_path


def step_map(trace: Sequence[Mapping[str,Any]]) -> dict:
    return {r["step_name"]:r for r in trace}


def unit_rows(raw_obs: Mapping[str,Any], capital):
    out=[]
    for key,u in (raw_obs.get("unit",{}) or {}).items():
        if not isinstance(u,dict) or int(u.get("tribeId",-1))!=0: continue
        xy=(int(u.get("x",-1)),int(u.get("y",-1)))
        out.append({"unit_id":int(key),"unit_type":str(u.get("type","UNKNOWN")),"x":xy[0],"y":xy[1],
                    "on_capital":capital is not None and xy==tuple(capital)})
    out.sort(key=lambda x:x["unit_id"])
    if out: out[0]["believed_original_starting_unit"]=True
    for x in out[1:]: x["believed_original_starting_unit"]=False
    return out


def final_snapshot(env, raw_obs):
    tribes=raw_obs.get("tribes",{}) or {}; cities=raw_obs.get("city",{}) or {}; tribe0=tribes.get("0",{}) or {}
    capital_id=tribe0.get("capitalID"); city=cities.get(str(capital_id),{}) if capital_id is not None else {}
    capital=[int(city.get("x",-1)),int(city.get("y",-1))] if isinstance(city,dict) and city else None
    units=unit_rows(raw_obs,capital); occupying=next((u for u in units if u["on_capital"]),None)
    return {"wrapper_turn_count":int(env._turn_count),"java_active_tribe_id":int(raw_obs.get("activeTribeID",-1)),
            "stars":int(env._get_tribe_stars(raw_obs,0)),"spt":float(env.tribes_env._compute_spt_from_obs(raw_obs,0)),
            "city_count":int(env._get_city_count(raw_obs)),"owned_unit_count":len(units),"visible_owned_unit_count":len(units),
            "owned_units":units,"capital_coordinate":capital,"capital_occupied":occupying is not None,
            "capital_occupying_unit_id":occupying["unit_id"] if occupying else None,
            "capital_occupying_unit_type":occupying["unit_type"] if occupying else None}


def classify(trace, handoff, reset_exception=None):
    steps=step_map(trace)
    flags=[]
    if reset_exception: return "OTHER_OPENING_FAILURE",["RESET_EXCEPTION"]
    for phase in ("animal_harvest_1","animal_harvest_2"):
        if not steps.get(phase,{}).get("success",False): flags.append("HARVEST_FAILURE")
    if not steps.get("workshop_levelup",{}).get("success",False): flags.append("WORKSHOP_FAILURE")
    for phase in ("turn0_end_turn","turn1_end_turn"):
        if not steps.get(phase,{}).get("success",False): flags.append("TURN_TRANSITION_FAILURE")
    moves=[steps.get("turn0_starting_warrior_move",{}),steps.get("turn1_starting_warrior_move",{})]
    if any(m.get("exception_type") for m in moves): flags.append("MOVE_EXCEPTION_SWALLOWED")
    if any(not m.get("success",False) for m in moves): flags.append("ONE_UNIT_MOVE_FAILED")
    spawn=steps.get("turn1_second_warrior_spawn",{})
    if spawn.get("exception_type"): flags.append("SPAWN_EXCEPTION")
    if not spawn.get("success",False):
        pre=spawn.get("pre") or {}; cap=pre.get("capital"); occupied=any([u.get("x"),u.get("y")]==cap for u in pre.get("units",[])) if cap else False
        flags.append("ONE_UNIT_SPAWN_UNAVAILABLE" if occupied or spawn.get("note")=="action_unavailable_or_unmatched" else "ONE_UNIT_SPAWN_SKIPPED")
    if handoff.get("owned_unit_count")==2 and not flags: return "EXPECTED_TWO_UNIT_OPENING",[]
    priority=("HARVEST_FAILURE","WORKSHOP_FAILURE","TURN_TRANSITION_FAILURE","MOVE_EXCEPTION_SWALLOWED","ONE_UNIT_MOVE_FAILED","SPAWN_EXCEPTION","ONE_UNIT_SPAWN_UNAVAILABLE","ONE_UNIT_SPAWN_SKIPPED")
    return next((p for p in priority if p in flags),"OTHER_OPENING_FAILURE"),sorted(set(flags))


def expectations(trace,handoff):
    s=step_map(trace); spawn=s.get("turn1_second_warrior_spawn",{}); pre=spawn.get("pre") or {}; post=spawn.get("post") or {}; cap=pre.get("capital")
    pre_occ=any([u.get("x"),u.get("y")]==cap for u in pre.get("units",[])) if cap else None
    post_cap=(post.get("capital") if post else handoff.get("capital_coordinate")); post_units=post.get("units",[]) if post else handoff.get("owned_units",[])
    post_occ=any([u.get("x"),u.get("y")]==post_cap for u in post_units) if post_cap else None
    return {"two_animals_harvested":all(s.get(x,{}).get("success",False) for x in ("animal_harvest_1","animal_harvest_2")),
            "workshop_claimed":s.get("workshop_levelup",{}).get("success",False),
            "turn0_move_executed":s.get("turn0_starting_warrior_move",{}).get("success",False),
            "turn1_move_executed":s.get("turn1_starting_warrior_move",{}).get("success",False),
            "second_warrior_spawn_attempted":spawn.get("attempted",False),"second_warrior_spawn_executed":spawn.get("success",False),
            "turn0_end_turn_executed":s.get("turn0_end_turn",{}).get("success",False),"turn1_end_turn_executed":s.get("turn1_end_turn",{}).get("success",False),
            "owned_units_at_handoff_eq_2":handoff.get("owned_unit_count")==2,"capital_free_before_spawn":None if pre_occ is None else not pre_occ,
            "capital_occupied_after_spawn":post_occ}


def audit_one(env,row,seed):
    rec=OpeningTraceRecorder(); env._opening_audit_recorder=rec; env._level_pool=[row["path"]]; env._level_pool_size=1
    started=time.perf_counter(); exc=None
    try: env.reset(seed=int(seed))
    except Exception as e: exc=e
    raw=env.tribes_env._last_obs if isinstance(env.tribes_env._last_obs,dict) else {}
    handoff=final_snapshot(env,raw) if raw else {}
    outcome,flags=classify(rec.rows,handoff,exc)
    return {"audit_schema_version":SCHEMA_VERSION,"pool":row["pool"],"map_filename":row["filename"],"map_path":row["path"],
            "canonical_map_sha256":row["canonical_map_sha256"],"csv_sha256":row["csv_sha256"],"episode_seed":int(seed),
            "opening_outcome":outcome,"outcome_flags":flags,"reset_success":exc is None,"reset_exception_type":type(exc).__name__ if exc else None,
            "reset_exception_message":str(exc) if exc else None,"trace":rec.rows,"handoff":handoff,"expectations":expectations(rec.rows,handoff),
            "audit_duration_seconds":time.perf_counter()-started}


def aggregate(rows):
    def metrics(group):
        n=len(group); units=Counter(str(r.get("handoff",{}).get("owned_unit_count","reset_failure")) for r in group); outcomes=Counter(r["opening_outcome"] for r in group)
        def rate(pred): return sum(bool(pred(r)) for r in group)/n if n else 0.0
        return {"maps":n,"unit_count_distribution":dict(units),"outcome_distribution":dict(outcomes),
                "expected_two_unit_rate":rate(lambda r:r["opening_outcome"]=="EXPECTED_TWO_UNIT_OPENING"),
                "turn0_move_success_rate":rate(lambda r:r["expectations"]["turn0_move_executed"]),
                "turn1_move_success_rate":rate(lambda r:r["expectations"]["turn1_move_executed"]),
                "spawn_success_rate":rate(lambda r:r["expectations"]["second_warrior_spawn_executed"]),
                "swallowed_move_exception_rate":rate(lambda r:"MOVE_EXCEPTION_SWALLOWED" in r["outcome_flags"]),
                "starting_warrior_on_capital_handoff_rate":rate(lambda r:any(u.get("believed_original_starting_unit") and u.get("on_capital") for u in r.get("handoff",{}).get("owned_units",[]))),
                "capital_free_before_spawn_but_spawn_failed_rate":rate(lambda r:r["expectations"]["capital_free_before_spawn"] is True and not r["expectations"]["second_warrior_spawn_executed"]),
                "turn0_move_worked_turn1_failed_rate":rate(lambda r:r["expectations"]["turn0_move_executed"] and not r["expectations"]["turn1_move_executed"]),
                "both_moves_worked_spawn_failed_rate":rate(lambda r:r["expectations"]["turn0_move_executed"] and r["expectations"]["turn1_move_executed"] and not r["expectations"]["second_warrior_spawn_executed"])}
    by_pool={p:metrics([r for r in rows if r["pool"]==p]) for p in POOLS if any(r["pool"]==p for r in rows)}
    exceptions=Counter()
    for r in rows:
        if r.get("reset_exception_type"): exceptions[r["reset_exception_type"]]+=1
        for s in r["trace"]:
            if s.get("exception_type"): exceptions[s["exception_type"]]+=1
    return {"audit_schema_version":SCHEMA_VERSION,"total_maps":len(rows),"overall":metrics(rows),"by_pool":by_pool,
            "exception_distribution":dict(exceptions),"train_vs_validation":{"train":by_pool.get("train"),"validation":by_pool.get("validation")}}


def write_csv(path,rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--pool",choices=[*POOLS,"all"],default="all")
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--map"); ap.add_argument("--max-maps",type=int)
    ap.add_argument("--print-trace",action="store_true"); ap.add_argument("--progress-every",type=int,default=100)
    ap.add_argument("--output-root",type=Path,default=REPO_ROOT/"outputs/opening_audit"); ap.add_argument("--audit-id")
    args=ap.parse_args(); manifest,maps,manifest_path=load_manifest_maps(args.pool,args.map,args.max_maps)
    audit_id=args.audit_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{args.pool}_seed{args.seed}"
    out=args.output_root/audit_id
    if out.exists(): raise SystemExit(f"Output exists: {out}")
    out.mkdir(parents=True); (out/"representative_traces").mkdir()
    os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"]="1"; os.environ["POLYVISION_INFO_MODE"]="fast"
    # Bootstrap only; every audited reset below is forced to its exact manifest path.
    os.environ["POLYVISION_LEVEL_POOL_GLOB"]="levels/phase1_pool_bardur_real/train/*.csv"
    config={"audit_schema_version":SCHEMA_VERSION,"audit_id":audit_id,"started_utc":utc_now(),"classification":"environment_contract_audit_not_model_evaluation",
            "phase1_opening_version":TribesGymWrapper.PHASE1_OPENING_VERSION,
            "pool":args.pool,"seed":args.seed,"map_filter":args.map,"requested_maps":len(maps),"manifest_path":str(manifest_path),
            "manifest_sha256":sha256_file(manifest_path),"pool_counts":manifest["pool_counts"],"runs_policy":False,"writes_model_scores":False}
    (out/"config.json").write_text(json.dumps(config,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    env=TribesGymWrapper(); rows=[]
    try:
        with (out/"maps.jsonl").open("w",encoding="utf-8") as f:
            for i,m in enumerate(maps,1):
                row=audit_one(env,m,args.seed); rows.append(row); f.write(json.dumps(row,sort_keys=True)+"\n"); f.flush()
                if args.print_trace: print(json.dumps(row,indent=2))
                if args.progress_every and (i%args.progress_every==0 or i==len(maps)): print(f"opening audit: {i}/{len(maps)}")
    finally: env.close()
    summary=aggregate(rows); summary["audit_id"]=audit_id
    summary["phase1_opening_version"]=TribesGymWrapper.PHASE1_OPENING_VERSION
    summary["completed_utc"]=utc_now()
    (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    summary_rows=[]
    for p,v in [("all",summary["overall"]),*summary["by_pool"].items()]: summary_rows.append({"pool":p,**{k:v[k] for k in v if not isinstance(v[k],dict)},"unit_count_distribution":json.dumps(v["unit_count_distribution"],sort_keys=True),"outcome_distribution":json.dumps(v["outcome_distribution"],sort_keys=True)})
    write_csv(out/"summary.csv",summary_rows)
    failures=[]
    for r in rows:
        if r["opening_outcome"]=="EXPECTED_TWO_UNIT_OPENING": continue
        failures.append({"pool":r["pool"],"map_filename":r["map_filename"],"canonical_map_sha256":r["canonical_map_sha256"],"episode_seed":r["episode_seed"],
                         "opening_outcome":r["opening_outcome"],"flags":";".join(r["outcome_flags"]),"owned_units":r.get("handoff",{}).get("owned_unit_count"),
                         "capital_occupied":r.get("handoff",{}).get("capital_occupied"),"reset_exception":r.get("reset_exception_message")})
    write_csv(out/"failures.csv",failures)
    success=[r for r in rows if r["opening_outcome"]=="EXPECTED_TWO_UNIT_OPENING"][:3]; failed=[r for r in rows if r["opening_outcome"]!="EXPECTED_TWO_UNIT_OPENING"][:3]
    suspected=[r for r in rows if r["map_filename"]=="map_003696.csv"]
    for r in [*suspected,*success,*failed]:
        name=f"{r['pool']}__{Path(r['map_filename']).stem}__{r['opening_outcome']}.json"
        (out/"representative_traces"/name).write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2)); print(f"Results: {out}")


if __name__=="__main__": main()
