"""Policy-visible policies, schedules, statistics, and serialization for Phase 1 evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

SCHEMA_VERSION = "polyvision-phase1-evaluation-v1"
STOCHASTIC_POLICIES = {"ppo_sampled", "random_legal"}
DETERMINISTIC_POLICIES = {"ppo_argmax", "visible_greedy"}
ALL_POLICIES = ("ppo_argmax", "ppo_sampled", "visible_greedy", "random_legal")

FEATURE = {
    "is_move": 0,
    "new_reveal": 1,
    "adj_fog_after": 2,
    "adj_fog_delta": 3,
    "zero_reveal": 4,
    "visible_village_target": 5,
    "dist_delta_visible_village": 7,
    "immediate_backtrack": 8,
    "inside_owned_city": 9,
    "unit_type_warrior": 11,
    "is_end_turn": 12,
    "is_capture": 13,
    "is_train": 14,
    "is_resource": 16,
    "is_level_up": 17,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(top_seed: int, *parts: Any) -> int:
    payload = ":".join([str(int(top_seed)), *(str(p) for p in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**31 - 1)


@dataclass(frozen=True)
class MapRecord:
    filename: str
    path: str
    canonical_map_sha256: str
    csv_sha256: str
    pool: str


@dataclass(frozen=True)
class ScheduleRecord:
    policy: str
    map_filename: str
    map_path: str
    map_identity: str
    csv_sha256: str
    replicate_index: int
    episode_seed: int
    policy_seed: int | None


def load_verified_pool(repo_root: Path, pool: str, max_maps: int | None = None) -> tuple[dict, list[MapRecord], Path]:
    if pool not in {"validation", "test", "human_benchmark"}:
        raise ValueError(f"Unsupported evaluation pool: {pool}")
    manifest_path = repo_root / "pol_env/Tribes/levels/phase1_pool_bardur_real/split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [r for r in manifest["maps"] if r["pool"] == pool]
    expected = int(manifest["pool_counts"][pool])
    if len(rows) != expected:
        raise RuntimeError(f"Manifest pool count mismatch for {pool}: {len(rows)} != {expected}")
    train_ids = {r["canonical_map_sha256"] for r in manifest["maps"] if r["pool"] == "train"}
    records = []
    root = manifest_path.parent
    for row in sorted(rows, key=lambda r: (r["canonical_map_sha256"], r["filename"])):
        path = (root / row["relative_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Manifest map missing: {path}")
        actual = sha256_file(path)
        if actual != row["csv_sha256"]:
            raise RuntimeError(f"Map hash mismatch: {path}")
        if row["canonical_map_sha256"] in train_ids:
            raise RuntimeError(f"Training-map identity leaked into {pool}: {row['filename']}")
        records.append(MapRecord(row["filename"], str(path), row["canonical_map_sha256"], actual, pool))
    identities = [m.canonical_map_sha256 for m in records]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"Duplicate canonical map identity in {pool}")
    return manifest, records[:max_maps] if max_maps else records, manifest_path


def build_schedule(maps: Sequence[MapRecord], policies: Sequence[str], repeats_per_map: int, seed: int) -> list[ScheduleRecord]:
    schedule = []
    for policy in policies:
        repeats = repeats_per_map if policy in STOCHASTIC_POLICIES else 1
        for m in maps:
            for rep in range(repeats):
                episode_seed = stable_seed(seed, "episode", m.canonical_map_sha256, rep)
                policy_seed = stable_seed(seed, "policy", policy, m.canonical_map_sha256, rep) if policy in STOCHASTIC_POLICIES else None
                schedule.append(ScheduleRecord(policy, m.filename, m.path, m.canonical_map_sha256, m.csv_sha256, rep, episode_seed, policy_seed))
    validate_schedule(schedule, maps, policies, repeats_per_map)
    return schedule


def validate_schedule(schedule: Sequence[ScheduleRecord], maps: Sequence[MapRecord], policies: Sequence[str], repeats: int) -> None:
    expected_ids = {m.canonical_map_sha256 for m in maps}
    for policy in policies:
        rows = [r for r in schedule if r.policy == policy]
        wanted = repeats if policy in STOCHASTIC_POLICIES else 1
        counts = {mid: 0 for mid in expected_ids}
        for row in rows:
            if row.map_identity not in counts:
                raise RuntimeError(f"Unexpected map in {policy} schedule")
            counts[row.map_identity] += 1
        bad = {k: v for k, v in counts.items() if v != wanted}
        if bad:
            raise RuntimeError(f"Schedule repetition mismatch for {policy}: {bad}")


def legal_tensors(info: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(info["legal_global_ids_padded"], dtype=np.int64).reshape(-1)
    valid = np.asarray(info["legal_action_valid_mask"], dtype=bool).reshape(-1)
    feats = np.asarray(info["legal_action_features_padded"], dtype=np.float32)
    if feats.ndim != 2:
        feats = feats.reshape(ids.size, -1)
    if ids.size != valid.size or feats.shape[0] != ids.size:
        raise RuntimeError("Policy-visible legal tensor shape mismatch")
    slots = np.flatnonzero(valid)
    if slots.size == 0:
        raise RuntimeError("No policy-visible valid legal action")
    if len(set(ids[slots].tolist())) != slots.size:
        raise RuntimeError("Duplicate valid policy-visible global IDs")
    return ids, valid, feats


class EvaluationPolicy:
    name = "base"
    def choose_action(self, obs: np.ndarray, info: Mapping[str, Any]) -> tuple[int, int]:
        raise NotImplementedError


class PPOPolicy(EvaluationPolicy):
    def __init__(self, agent, device: torch.device, sampled: bool, policy_seed: int | None):
        self.agent, self.device, self.sampled = agent, device, sampled
        self.name = "ppo_sampled" if sampled else "ppo_argmax"
        self.generator = None
        if sampled:
            self.generator = torch.Generator(device=device.type).manual_seed(int(policy_seed))

    def choose_action(self, obs, info):
        ids, valid, feats = legal_tensors(info)
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=self.device).reshape(1, -1)
        valid_t = torch.as_tensor(valid, dtype=torch.bool, device=self.device).reshape(1, -1)
        feats_t = torch.as_tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            dist = self.agent.get_action_distribution(
                x, legal_global_ids=ids_t, legal_action_valid_mask=valid_t,
                legal_action_features=feats_t,
            )
            if self.sampled:
                slot = int(torch.multinomial(dist.probs[0], 1, generator=self.generator).item())
            else:
                slot = int(torch.argmax(dist.logits[0]).item())
        if not valid[slot]:
            raise RuntimeError("PPO selected an invalid/padded legal slot")
        return int(ids[slot]), slot


class RandomLegalPolicy(EvaluationPolicy):
    name = "random_legal"
    def __init__(self, policy_seed: int):
        self.rng = np.random.default_rng(int(policy_seed))
    def choose_action(self, obs, info):
        ids, valid, _ = legal_tensors(info)
        slots = np.flatnonzero(valid)
        slot = int(self.rng.choice(slots))
        return int(ids[slot]), slot


class VisibleGreedyPolicy(EvaluationPolicy):
    """Deterministic heuristic using only PPO-visible observation and legal tensors."""
    name = "visible_greedy"
    def choose_action(self, obs, info):
        ids, valid, feats = legal_tensors(info)
        slots = np.flatnonzero(valid)
        capture = slots[feats[slots, FEATURE["is_capture"]] > 0.5]
        if capture.size:
            slot = int(capture[0])
        else:
            move = slots[feats[slots, FEATURE["is_move"]] > 0.5]
            if move.size:
                mf = feats[move]
                city_count = int(info.get("city_count", 1))
                if city_count < 2:
                    score = (5*mf[:,5] + 3*mf[:,1] + 2*mf[:,2] + mf[:,3] + 2*mf[:,7]
                             - 2*mf[:,4] - mf[:,8] - .5*mf[:,9])
                else:
                    score = 2*mf[:,1] + mf[:,2] + mf[:,5] - mf[:,4]
                slot = int(move[int(np.argmax(score))])
            else:
                warriors = slots[(feats[slots, FEATURE["is_train"]] > .5) & (feats[slots, FEATURE["unit_type_warrior"]] > .5)]
                resources = slots[feats[slots, FEATURE["is_resource"]] > .5]
                upgrades = slots[feats[slots, FEATURE["is_level_up"]] > .5]
                end_turn = slots[feats[slots, FEATURE["is_end_turn"]] > .5]
                slot = int(next((x[0] for x in (warriors, resources, upgrades, end_turn) if x.size), slots[0]))
        return int(ids[slot]), slot


def bootstrap_ci(values: Sequence[float], seed: int, samples: int = 5000) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return [math.nan, math.nan]
    if arr.size == 1:
        return [float(arr[0]), float(arr[0])]
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(arr, size=(samples, arr.size), replace=True), axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def distribution_stats(values: Sequence[float], ci_seed: int) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size), "mean": float(np.mean(arr)), "median": float(np.median(arr)),
        "stddev": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)), "max": float(np.max(arr)),
        "p05": float(np.percentile(arr, 5)), "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)), "p95": float(np.percentile(arr, 95)),
        "mean_ci95": bootstrap_ci(arr, ci_seed),
    }


def aggregate_results(episodes: Sequence[Mapping[str, Any]], seed: int) -> tuple[dict, list[dict]]:
    policies = sorted({str(e["policy"]) for e in episodes})
    summary, per_map = {}, []
    for policy in policies:
        rows = [e for e in episodes if e["policy"] == policy]
        grouped = {}
        for row in rows:
            grouped.setdefault(row["map_identity"], []).append(row)
        map_means = []
        for mid, group in sorted(grouped.items()):
            vals = [float(x["final_spt_t10"]) for x in group]
            map_row = {
                "policy": policy, "map_filename": group[0]["map_filename"], "map_identity": mid,
                "episodes": len(group), "mean_spt": float(np.mean(vals)), "median_spt": float(np.median(vals)),
                "min_spt": float(np.min(vals)), "max_spt": float(np.max(vals)),
                "stddev_spt": float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0,
            }
            per_map.append(map_row); map_means.append(map_row["mean_spt"])
        summary[policy] = {
            "maps": len(grouped), "episodes": len(rows),
            "episode_level": distribution_stats([e["final_spt_t10"] for e in rows], stable_seed(seed, policy, "episode-ci")),
            "map_level": distribution_stats(map_means, stable_seed(seed, policy, "map-ci")),
            "headline_level": "map_level",
        }
    return summary, per_map


def paired_stats(per_map: Sequence[Mapping[str, Any]], left: str, right: str, seed: int) -> dict | None:
    l = {r["map_identity"]: float(r["mean_spt"]) for r in per_map if r["policy"] == left}
    r = {x["map_identity"]: float(x["mean_spt"]) for x in per_map if x["policy"] == right}
    keys = sorted(set(l) & set(r))
    if not keys:
        return None
    diffs = np.asarray([l[k]-r[k] for k in keys])
    return {"left":left,"right":right,"maps":len(keys),"wins":int(np.sum(diffs>0)),"ties":int(np.sum(diffs==0)),
            "losses":int(np.sum(diffs<0)),"win_rate":float(np.mean(diffs>0)),"mean_difference":float(np.mean(diffs)),
            "median_difference":float(np.median(diffs)),"mean_difference_ci95":bootstrap_ci(diffs, stable_seed(seed,left,right,"paired-ci"))}


def comparison_rows(per_map: Sequence[Mapping[str, Any]]) -> list[dict]:
    by_map = {}
    for r in per_map: by_map.setdefault(r["map_identity"], {})[r["policy"]] = r
    out=[]
    for mid, policies in sorted(by_map.items()):
        row={"map_identity":mid,"map_filename":next(iter(policies.values()))["map_filename"]}
        for p in ALL_POLICIES: row[f"{p}_spt"] = policies.get(p,{}).get("mean_spt")
        row["ppo_argmax_minus_visible_greedy"] = _delta(row,"ppo_argmax_spt","visible_greedy_spt")
        row["ppo_sampled_minus_random_legal"] = _delta(row,"ppo_sampled_spt","random_legal_spt")
        row["ppo_argmax_minus_random_legal"] = _delta(row,"ppo_argmax_spt","random_legal_spt")
        out.append(row)
    return out


def _delta(row, a, b):
    return None if row.get(a) is None or row.get(b) is None else float(row[a]-row[b])


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def json_safe(value):
    if isinstance(value, dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,float) and (math.isnan(value) or math.isinf(value)): return None
    return value
