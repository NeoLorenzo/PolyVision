#!/usr/bin/env python3
"""Run and maintain the permanent Phase 1 human benchmark registry.

Ordinary usage selects a random human-benchmark map without a completed first
human attempt and plays it through the exact PPO-facing TribesGymWrapper:

    python tools/human_benchmark.py

Completed first attempts are immutable. Use ``--replay MAP`` for a deliberate
later attempt, or ``--summary`` to inspect the registry without playing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import secrets
import statistics
import subprocess
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol_env.Tribes.py.register_env import TribesGymWrapper
from tools.human_policy_interface import (
    EpisodeResult,
    choose_first_action,
    run_policy_visible_episode,
    utc_now,
)


SCHEMA_VERSION = 1
POOL_ROOT = REPO_ROOT / "pol_env" / "Tribes" / "levels" / "phase1_pool_bardur_real"
BENCHMARK_POOL = POOL_ROOT / "human_benchmark"
SPLIT_MANIFEST = POOL_ROOT / "split_manifest.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "human_benchmark"


class BenchmarkError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark_maps(
    pool_root: Path = POOL_ROOT,
    split_manifest_path: Path = SPLIT_MANIFEST,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark_pool = pool_root / "human_benchmark"
    if not benchmark_pool.is_dir():
        raise BenchmarkError(f"official human benchmark folder is missing: {benchmark_pool}")
    if not split_manifest_path.is_file():
        raise BenchmarkError(f"authoritative split manifest is missing: {split_manifest_path}")
    try:
        manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot read split manifest: {exc}") from exc
    entries = manifest.get("maps")
    if not isinstance(entries, list):
        raise BenchmarkError("split manifest has no map list")

    canonical_groups: dict[str, list[str]] = defaultdict(list)
    csv_groups: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        canonical_groups[str(entry.get("canonical_map_sha256", ""))].append(str(entry.get("relative_path", "")))
        csv_groups[str(entry.get("csv_sha256", ""))].append(str(entry.get("relative_path", "")))
    canonical_duplicates = {key: value for key, value in canonical_groups.items() if not key or len(value) > 1}
    csv_duplicates = {key: value for key, value in csv_groups.items() if not key or len(value) > 1}
    if canonical_duplicates or csv_duplicates:
        raise BenchmarkError(
            "split manifest is not disjoint by content identity: "
            f"canonical_duplicate_groups={len(canonical_duplicates)}, csv_duplicate_groups={len(csv_duplicates)}"
        )

    assigned = [entry for entry in entries if entry.get("pool") == "human_benchmark"]
    expected_count = int(manifest.get("pool_counts", {}).get("human_benchmark", -1))
    if expected_count <= 0 or len(assigned) != expected_count:
        raise BenchmarkError(
            f"split manifest human_benchmark count is inconsistent: entries={len(assigned)}, declared={expected_count}"
        )
    files = sorted(benchmark_pool.glob("*.csv"))
    if not files:
        raise BenchmarkError(f"official human benchmark folder is empty: {benchmark_pool}")
    assigned_names = {str(entry["filename"]) for entry in assigned}
    file_names = {path.name for path in files}
    if assigned_names != file_names:
        raise BenchmarkError(
            "human benchmark filesystem disagrees with split manifest: "
            f"missing={sorted(assigned_names - file_names)}, extra={sorted(file_names - assigned_names)}"
        )

    by_name = {str(entry["filename"]): entry for entry in assigned}
    records = []
    for path in files:
        entry = by_name[path.name]
        actual_hash = sha256_file(path)
        if actual_hash != str(entry["csv_sha256"]):
            raise BenchmarkError(f"{path.name}: CSV SHA-256 disagrees with split manifest")
        expected_relative = f"human_benchmark/{path.name}"
        if str(entry.get("relative_path")) != expected_relative:
            raise BenchmarkError(f"{path.name}: manifest relative path is not {expected_relative!r}")
        records.append(
            {
                "filename": path.name,
                "path": path.resolve(),
                "relative_path": expected_relative,
                "csv_sha256": actual_hash,
                "canonical_map_sha256": str(entry["canonical_map_sha256"]),
            }
        )
    return records, manifest


def _attempt_dir(output_root: Path) -> Path:
    return output_root / "attempts"


def load_attempts(output_root: Path) -> list[dict[str, Any]]:
    directory = _attempt_dir(output_root)
    if not directory.exists():
        return []
    attempts = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkError(f"cannot read attempt record {path}: {exc}") from exc
        if int(payload.get("benchmark_schema_version", -1)) != SCHEMA_VERSION:
            raise BenchmarkError(f"attempt record has unsupported schema: {path}")
        attempts.append(payload)
    return attempts


def completed_first_maps(attempts: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(attempt["map"]["filename"])
        for attempt in attempts
        if attempt.get("participant_kind") == "human"
        and attempt.get("status") == "completed"
        and bool(attempt.get("is_first_completed_attempt"))
    }


def select_unplayed_map(
    maps: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    selection_seed: int | None,
) -> dict[str, Any] | None:
    played = completed_first_maps(attempts)
    remaining = sorted((record for record in maps if record["filename"] not in played), key=lambda item: item["filename"])
    if not remaining:
        return None
    rng = random.Random(int(selection_seed)) if selection_seed is not None else secrets.SystemRandom()
    return rng.choice(remaining)


def resolve_map_id(maps: list[dict[str, Any]], map_id: str) -> dict[str, Any]:
    wanted = str(map_id).strip().lower()
    matches = [
        record
        for record in maps
        if wanted in {record["filename"].lower(), Path(record["filename"]).stem.lower()}
        or record["canonical_map_sha256"].lower().startswith(wanted)
    ]
    if len(matches) != 1:
        raise BenchmarkError(f"map identifier {map_id!r} matched {len(matches)} benchmark maps")
    return matches[0]


def git_provenance(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.strip()
        tracked_dirty = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"], cwd=repo_root, check=False
        ).returncode != 0
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {"commit_sha": commit, "dirty_worktree": bool(tracked_dirty or untracked)}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"commit_sha": None, "dirty_worktree": None, "provenance_error": str(exc)}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def append_registry_event(output_root: Path, payload: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    event = {
        "benchmark_schema_version": SCHEMA_VERSION,
        "recorded_at_utc": utc_now(),
        "attempt_id": payload["attempt_id"],
        "status": payload["status"],
        "participant_kind": payload["participant_kind"],
        "map_filename": payload["map"]["filename"],
        "attempt_number": payload["attempt_number"],
        "is_first_completed_attempt": payload["is_first_completed_attempt"],
        "attempt_file": f"attempts/{payload['attempt_id']}.json",
    }
    with (output_root / "results.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")


def persist_attempt(output_root: Path, payload: dict[str, Any], *, event: bool = True) -> None:
    path = _attempt_dir(output_root) / f"{payload['attempt_id']}.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            raise BenchmarkError(f"refusing to overwrite completed attempt {payload['attempt_id']}")
    atomic_write_json(path, payload)
    if event:
        append_registry_event(output_root, payload)


def _spt(attempt: dict[str, Any]) -> float | None:
    value = attempt.get("result", {}).get("final_t10_spt")
    return float(value) if value is not None else None


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_spt": None, "median_spt": None, "stddev_spt": None, "min_spt": None, "max_spt": None}
    return {
        "count": len(values),
        "mean_spt": statistics.fmean(values),
        "median_spt": statistics.median(values),
        "stddev_spt": statistics.pstdev(values),
        "min_spt": min(values),
        "max_spt": max(values),
    }


def rebuild_summary(
    output_root: Path,
    maps: list[dict[str, Any]],
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    attempts = load_attempts(output_root) if attempts is None else attempts
    human = [attempt for attempt in attempts if attempt.get("participant_kind") == "human"]
    completed = [attempt for attempt in human if attempt.get("status") == "completed" and _spt(attempt) is not None]
    all_by_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in human:
        all_by_map[str(attempt["map"]["filename"])].append(attempt)
    by_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in completed:
        by_map[str(attempt["map"]["filename"])].append(attempt)
    first_attempts = [
        attempt for attempt in completed if bool(attempt.get("is_first_completed_attempt"))
    ]
    latest_attempts = [max(items, key=lambda item: (int(item["attempt_number"]), str(item["ended_at_utc"]))) for items in by_map.values()]
    best_attempts = [max(items, key=lambda item: float(_spt(item))) for items in by_map.values()]
    played = {attempt["map"]["filename"] for attempt in first_attempts}
    status_counts = Counter(str(attempt.get("status")) for attempt in human)
    map_rows = []
    for record in sorted(maps, key=lambda item: item["filename"]):
        items = by_map.get(record["filename"], [])
        first = next((item for item in items if item.get("is_first_completed_attempt")), None)
        latest = max(items, key=lambda item: int(item["attempt_number"])) if items else None
        best = max(items, key=lambda item: float(_spt(item))) if items else None
        map_rows.append(
            {
                "filename": record["filename"],
                "csv_sha256": record["csv_sha256"],
                "canonical_map_sha256": record["canonical_map_sha256"],
                "completed_first_attempt": first is not None,
                "human_attempt_count": len(all_by_map.get(record["filename"], [])),
                "first_attempt_spt": _spt(first) if first else None,
                "latest_attempt_spt": _spt(latest) if latest else None,
                "best_attempt_spt": _spt(best) if best else None,
            }
        )
    summary = {
        "benchmark_schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "benchmark_map_count": len(maps),
        "completed_first_attempt_maps": len(played),
        "remaining_first_attempt_maps": len(maps) - len(played),
        "human_attempt_status_counts": dict(sorted(status_counts.items())),
        "first_attempt": _stats([float(_spt(attempt)) for attempt in first_attempts]),
        "latest_attempt": _stats([float(_spt(attempt)) for attempt in latest_attempts]),
        "best_attempt": _stats([float(_spt(attempt)) for attempt in best_attempts]),
        "maps": map_rows,
    }
    atomic_write_json(output_root / "summary.json", summary)
    return summary


def print_summary(summary: dict[str, Any], *, this_map_spt: float | None = None) -> None:
    print("\nHuman Benchmark")
    print(f"Completed: {summary['completed_first_attempt_maps']} / {summary['benchmark_map_count']}")
    print(f"Remaining: {summary['remaining_first_attempt_maps']}")
    if this_map_spt is not None:
        print(f"\nThis map:\nFinal T10 SPT: {this_map_spt:g}")
    first = summary["first_attempt"]
    print("\nFirst-attempt aggregate:")
    if first["count"] == 0:
        print("No completed human first attempts yet.")
    else:
        print(f"Mean SPT:   {first['mean_spt']:.2f}")
        print(f"Median SPT: {first['median_spt']:.2f}")
        print(f"Stddev SPT: {first['stddev_spt']:.2f}")
        print(f"Best:       {first['max_spt']:g}")
        print(f"Worst:      {first['min_spt']:g}")


@contextmanager
def official_environment(map_path: Path):
    overrides = {
        "POLYVISION_LEVEL_POOL_GLOB": str(map_path.resolve()),
        "POLYVISION_LEVEL_SELECTION_MODE": "round_robin",
        "POLYVISION_SOLO_NO_OPPONENT_MODE": "1",
        "POLYVISION_INFO_MODE": "fast",
        "POLYVISION_MAX_LEGAL_ACTIONS": str(TribesGymWrapper.MAX_LEGAL_ACTIONS_DEFAULT),
        "POLYVISION_VERBOSE_RESETS": "0",
        "POLYVISION_OPENING_GRID_DEBUG": "0",
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    env = None
    try:
        env = TribesGymWrapper(level_file=str(map_path.resolve()))
        if len(env._level_pool) != 1 or Path(env._level_pool[0]).resolve() != map_path.resolve():
            raise BenchmarkError("official environment did not resolve to exactly the selected benchmark map")
        yield env
    finally:
        if env is not None:
            env.close()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _final_spt(result: EpisodeResult) -> float | None:
    for key in ("terminal_final_spt", "spt"):
        value = result.final_info_metrics.get(key)
        if value is not None:
            return float(value)
    value = result.final_visible_metrics.get("spt")
    return float(value) if value is not None else None


def make_started_attempt(
    selected_map: dict[str, Any],
    attempts: list[dict[str, Any]],
    *,
    participant_kind: str,
    selection_mode: str,
    selection_seed: int | None,
    episode_seed: int,
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    existing = [
        attempt for attempt in attempts
        if attempt.get("participant_kind") == participant_kind
        and attempt.get("map", {}).get("filename") == selected_map["filename"]
    ]
    attempt_number = 1 + max((int(attempt.get("attempt_number", 0)) for attempt in existing), default=0)
    stamp = utc_now().replace(":", "").replace("-", "").replace(".", "")
    attempt_id = f"{Path(selected_map['filename']).stem}-{stamp}-{uuid.uuid4().hex[:8]}"
    return {
        "benchmark_schema_version": SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "participant_kind": participant_kind,
        "is_first_completed_attempt": False,
        "status": "started",
        "started_at_utc": utc_now(),
        "ended_at_utc": None,
        "map": {
            "filename": selected_map["filename"],
            "relative_path": selected_map["relative_path"],
            "csv_sha256": selected_map["csv_sha256"],
            "canonical_map_sha256": selected_map["canonical_map_sha256"],
            "split_dataset_contract": split_manifest.get("dataset_contract"),
            "split_seed": split_manifest.get("split_seed"),
            "split_assignment": "human_benchmark",
            "human_benchmark_pool_identity": split_manifest.get("pool_identities", {}).get("human_benchmark"),
        },
        "configuration": {
            "episode_seed": int(episode_seed),
            "selection_mode": selection_mode,
            "selection_seed": selection_seed,
            "level_selection_mode": "single_exact_map",
            "solo_no_opponent_mode": True,
        },
        "git": git_provenance(),
        "result": None,
    }


def finalize_attempt(
    output_root: Path,
    payload: dict[str, Any],
    result: EpisodeResult | None,
    *,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    if status not in {"completed", "aborted", "error"}:
        raise BenchmarkError(f"invalid final attempt status {status!r}")
    existing = load_attempts(output_root)
    payload = dict(payload)
    payload["status"] = status
    payload["ended_at_utc"] = result.ended_at_utc if result is not None else utc_now()
    if status == "completed" and payload["participant_kind"] == "human":
        earlier = [
            attempt for attempt in existing
            if attempt.get("attempt_id") != payload["attempt_id"]
            and attempt.get("participant_kind") == "human"
            and attempt.get("status") == "completed"
            and attempt.get("map", {}).get("filename") == payload["map"]["filename"]
        ]
        payload["is_first_completed_attempt"] = not earlier
    if result is not None:
        payload["result"] = {
            "final_t10_spt": _final_spt(result),
            "shaped_return": result.shaped_return,
            "decision_count": result.decision_count,
            "final_visible_metrics": result.final_visible_metrics,
            "final_info_metrics": result.final_info_metrics,
            "environment_contract": result.environment_contract,
            "action_history": result.action_history,
        }
    if error is not None:
        payload["error"] = error
    persist_attempt(output_root, payload)
    return payload


def execute(
    args: argparse.Namespace,
    *,
    episode_runner: Callable[..., EpisodeResult] = run_policy_visible_episode,
) -> int:
    pool_root = Path(args.pool_root).resolve()
    split_manifest_path = pool_root / "split_manifest.json"
    maps, split_manifest = load_benchmark_maps(pool_root, split_manifest_path)
    output_root = Path(args.output_root).resolve()
    attempts = load_attempts(output_root)
    summary = rebuild_summary(output_root, maps, attempts)
    if args.summary:
        print_summary(summary)
        return 0

    participant_kind = "synthetic_test" if args.synthetic_smoke else "human"
    if args.synthetic_smoke and output_root == DEFAULT_OUTPUT_ROOT.resolve():
        raise BenchmarkError("--synthetic-smoke requires an explicit non-canonical --output-root")
    if args.replay:
        selected = resolve_map_id(maps, args.replay)
        if selected["filename"] not in completed_first_maps(attempts):
            raise BenchmarkError("a deliberate replay requires an existing completed human first attempt")
        selection_mode = "deliberate_replay"
    elif args.synthetic_smoke:
        selected = resolve_map_id(maps, args.map) if args.map else maps[0]
        selection_mode = "synthetic_smoke"
    else:
        selected = select_unplayed_map(maps, attempts, args.selection_seed)
        if selected is None:
            print_summary(summary)
            print("\nAll current human benchmark maps have a completed first attempt. Use --replay MAP for a deliberate replay.")
            return 0
        selection_mode = "random_unplayed_first_attempt"

    print(f"Selected benchmark map: {selected['filename']}")
    print(f"Selection mode: {selection_mode}")
    payload = make_started_attempt(
        selected,
        attempts,
        participant_kind=participant_kind,
        selection_mode=selection_mode,
        selection_seed=args.selection_seed,
        episode_seed=args.episode_seed,
        split_manifest=split_manifest,
    )
    persist_attempt(output_root, payload)

    try:
        with official_environment(selected["path"]) as env:
            selector = choose_first_action if args.synthetic_smoke else None
            result = episode_runner(
                env,
                episode_seed=int(args.episode_seed),
                official=True,
                page_size=int(args.page_size),
                selector=selector,
            )
        final_status = result.status
        finalized = finalize_attempt(output_root, payload, result, status=final_status)
    except (KeyboardInterrupt, EOFError):
        print("\nBenchmark attempt aborted; the map remains eligible for its first completed attempt.")
        finalized = finalize_attempt(output_root, payload, None, status="aborted", error="interrupted")
    except Exception as exc:
        finalize_attempt(output_root, payload, None, status="error", error=f"{type(exc).__name__}: {exc}")
        rebuild_summary(output_root, maps)
        raise

    summary = rebuild_summary(output_root, maps)
    if finalized["status"] == "completed" and participant_kind == "human":
        print_summary(summary, this_map_spt=finalized["result"]["final_t10_spt"])
        print(f"\nSaved immutable attempt: attempts/{finalized['attempt_id']}.json")
    elif finalized["status"] == "aborted":
        print("Attempt recorded as aborted; it does not count and the map remains eligible.")
    else:
        print(f"Synthetic smoke recorded separately with status={finalized['status']}; human statistics were unchanged.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", metavar="MAP", help="Deliberately replay a map after its first human attempt")
    parser.add_argument("--selection-seed", type=int, help="Optional reproducible random unplayed-map selection seed")
    parser.add_argument("--episode-seed", type=int, default=42, help="Environment reset seed recorded for model comparison")
    parser.add_argument("--page-size", type=int, default=30)
    parser.add_argument("--summary", action="store_true", help="Rebuild and print the registry summary without playing")
    parser.add_argument("--pool-root", type=Path, default=POOL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Non-human automated workflow smoke; requires a non-canonical --output-root and never affects human statistics",
    )
    parser.add_argument("--map", help="Map selection for --synthetic-smoke only")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.map and not args.synthetic_smoke:
        raise BenchmarkError("--map is reserved for --synthetic-smoke; use --replay for deliberate human replays")
    try:
        return execute(args)
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
