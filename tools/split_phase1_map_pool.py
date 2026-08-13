#!/usr/bin/env python3
"""Establish or verify the frozen Phase 1 genuine-map dataset split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL_ROOT = REPO_ROOT / "pol_env" / "Tribes" / "levels" / "phase1_pool_bardur_real"
DEFAULT_CONVERSION_MANIFEST = REPO_ROOT / "data" / "polytopia_maps" / "conversion_manifest.csv"
MANIFEST_NAME = "split_manifest.json"
SCHEMA_VERSION = 1
DATASET_CONTRACT = "phase1-bardur-real-v1"
SPLIT_SEED = 20260813
POOL_COUNTS = {
    "train": 5000,
    "validation": 250,
    "test": 250,
    "human_benchmark": 17,
}
EXPECTED_TOTAL = sum(POOL_COUNTS.values())
VALID_TERRAINS = frozenset(".sdfmvc")
VALID_SUFFIXES = frozenset("achwofr")


class SplitError(RuntimeError):
    """Raised when the frozen dataset contract cannot be established or verified."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_rank(identity: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{identity}".encode("ascii")).hexdigest()


def validate_phase1_csv(content: bytes, label: str) -> None:
    try:
        rows = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise SplitError(f"{label}: CSV is not UTF-8: {exc}") from exc
    if len(rows) != 11:
        raise SplitError(f"{label}: expected 11 rows, found {len(rows)}")
    capitals = 0
    for y, row in enumerate(rows):
        tokens = row.split(",")
        if len(tokens) != 11:
            raise SplitError(f"{label}: row {y} has {len(tokens)} columns; expected 11")
        for x, token in enumerate(tokens):
            if token.count(":") != 1:
                raise SplitError(f"{label}: malformed token at ({x},{y}): {token!r}")
            terrain, suffix = token.split(":", 1)
            if terrain not in VALID_TERRAINS:
                raise SplitError(f"{label}: invalid terrain at ({x},{y}): {terrain!r}")
            if terrain == "c":
                if suffix != "2":
                    raise SplitError(f"{label}: capital at ({x},{y}) is not c:2")
                capitals += 1
            elif terrain == "v":
                if suffix:
                    raise SplitError(f"{label}: village at ({x},{y}) has suffix {suffix!r}")
            elif suffix and suffix not in VALID_SUFFIXES:
                raise SplitError(f"{label}: invalid suffix at ({x},{y}): {suffix!r}")
    if capitals != 1:
        raise SplitError(f"{label}: expected exactly one c:2 capital, found {capitals}")


def load_conversion_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise SplitError(f"conversion manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"csv_filename", "csv_sha256", "map_sha256"}
    if not rows or not required.issubset(rows[0]):
        raise SplitError(f"conversion manifest lacks required columns: {sorted(required)}")
    by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row["csv_filename"].strip()
        if not name or Path(name).name != name or not name.lower().endswith(".csv"):
            raise SplitError(f"invalid CSV filename in conversion manifest: {name!r}")
        if name in by_name:
            raise SplitError(f"duplicate filename in conversion manifest: {name}")
        by_name[name] = row
    return by_name


def fail_on_duplicates(entries: Iterable[dict], field: str, description: str) -> None:
    grouped: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry[field])].append(str(entry["filename"]))
    duplicates = {identity: names for identity, names in grouped.items() if len(names) > 1}
    if duplicates:
        examples = "; ".join(
            f"{identity}: {', '.join(names[:4])}" for identity, names in list(duplicates.items())[:5]
        )
        raise SplitError(f"duplicate {description} identities found ({len(duplicates)} groups): {examples}")


def planned_pool_identity(entries: list[dict]) -> tuple[str, list[dict]]:
    rows = [
        {
            "path": str(entry["relative_path"]),
            "size_bytes": int(entry["size_bytes"]),
            "sha256": str(entry["csv_sha256"]),
        }
        for entry in sorted(entries, key=lambda item: str(item["relative_path"]))
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8")), rows


def build_initial_manifest(pool_root: Path, conversion_manifest: Path) -> dict:
    loose = sorted(pool_root.glob("*.csv"))
    nested = sorted(path for pool in POOL_COUNTS for path in (pool_root / pool).glob("*.csv"))
    if nested:
        raise SplitError(
            "pool directories already contain CSVs but no authoritative split manifest exists; "
            "refusing to infer or rebalance a partial split"
        )
    if len(loose) != EXPECTED_TOTAL:
        raise SplitError(f"expected exactly {EXPECTED_TOTAL} loose CSV maps, found {len(loose)}")

    conversion_rows = load_conversion_rows(conversion_manifest)
    if len(conversion_rows) != EXPECTED_TOTAL:
        raise SplitError(
            f"expected {EXPECTED_TOTAL} conversion-manifest maps, found {len(conversion_rows)}"
        )
    loose_names = {path.name for path in loose}
    manifest_names = set(conversion_rows)
    if loose_names != manifest_names:
        missing = sorted(manifest_names - loose_names)[:10]
        extra = sorted(loose_names - manifest_names)[:10]
        raise SplitError(f"flat corpus and conversion manifest disagree; missing={missing}, extra={extra}")

    entries = []
    for path in loose:
        content = path.read_bytes()
        validate_phase1_csv(content, path.name)
        csv_hash = sha256_bytes(content)
        source = conversion_rows[path.name]
        recorded_csv_hash = source["csv_sha256"].strip().lower()
        canonical_hash = source["map_sha256"].strip().lower()
        if csv_hash != recorded_csv_hash:
            raise SplitError(
                f"{path.name}: CSV SHA-256 {csv_hash} disagrees with conversion manifest {recorded_csv_hash}"
            )
        if len(canonical_hash) != 64:
            raise SplitError(f"{path.name}: invalid or missing canonical map_sha256")
        entries.append(
            {
                "filename": path.name,
                "csv_sha256": csv_hash,
                "canonical_map_sha256": canonical_hash,
                "size_bytes": len(content),
            }
        )

    fail_on_duplicates(entries, "canonical_map_sha256", "canonical map content")
    fail_on_duplicates(entries, "csv_sha256", "exact CSV content")

    ranked = sorted(entries, key=lambda item: (canonical_rank(item["canonical_map_sha256"]), item["filename"]))
    cursor = 0
    for pool, count in POOL_COUNTS.items():
        for entry in ranked[cursor : cursor + count]:
            entry["pool"] = pool
            entry["relative_path"] = f"{pool}/{entry['filename']}"
        cursor += count
    if cursor != len(ranked):
        raise SplitError(f"assignment consumed {cursor} maps, expected {len(ranked)}")

    ranked.sort(key=lambda item: item["filename"])
    pool_identities = {}
    for pool in POOL_COUNTS:
        members = [entry for entry in ranked if entry["pool"] == pool]
        identity, _ = planned_pool_identity(members)
        pool_identities[pool] = identity

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_contract": DATASET_CONTRACT,
        "split_seed": SPLIT_SEED,
        "total_map_count": EXPECTED_TOTAL,
        "pool_counts": dict(POOL_COUNTS),
        "identity_method": "canonical_map_sha256 (conversion manifest); exact CSV SHA-256 also enforced unique",
        "assignment_method": "sort by SHA-256 of '<split_seed>:<canonical_map_sha256>', then fixed pool counts",
        "pool_identities": pool_identities,
        "maps": ranked,
    }


def atomic_write_json(path: Path, payload: dict) -> None:
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


def establish(pool_root: Path, conversion_manifest: Path) -> None:
    manifest_path = pool_root / MANIFEST_NAME
    if manifest_path.exists():
        raise SplitError(f"split manifest already exists; verify it instead: {manifest_path}")
    manifest = build_initial_manifest(pool_root, conversion_manifest)
    for pool in POOL_COUNTS:
        (pool_root / pool).mkdir(parents=False, exist_ok=True)

    moved: list[tuple[Path, Path]] = []
    try:
        for entry in manifest["maps"]:
            source = pool_root / entry["filename"]
            target = pool_root / entry["relative_path"]
            if target.exists():
                raise SplitError(f"refusing to overwrite existing split target: {target}")
            shutil.move(str(source), str(target))
            moved.append((source, target))
        atomic_write_json(manifest_path, manifest)
    except Exception:
        rollback_failures = []
        for source, target in reversed(moved):
            try:
                if target.exists() and not source.exists():
                    shutil.move(str(target), str(source))
            except OSError as exc:
                rollback_failures.append(f"{target} -> {source}: {exc}")
        if rollback_failures:
            raise SplitError("split failed and rollback was incomplete: " + "; ".join(rollback_failures))
        raise
    verify(pool_root)


def verify(pool_root: Path) -> None:
    manifest_path = pool_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise SplitError(f"split manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitError(f"cannot read split manifest: {exc}") from exc

    expected_header = {
        "schema_version": SCHEMA_VERSION,
        "dataset_contract": DATASET_CONTRACT,
        "split_seed": SPLIT_SEED,
        "total_map_count": EXPECTED_TOTAL,
        "pool_counts": POOL_COUNTS,
    }
    for field, expected in expected_header.items():
        if manifest.get(field) != expected:
            raise SplitError(f"manifest {field}={manifest.get(field)!r}; expected {expected!r}")
    entries = manifest.get("maps")
    if not isinstance(entries, list) or len(entries) != EXPECTED_TOTAL:
        raise SplitError(f"manifest must contain exactly {EXPECTED_TOTAL} map entries")

    required = {
        "filename", "pool", "csv_sha256", "canonical_map_sha256", "relative_path", "size_bytes"
    }
    seen_names = set()
    pool_counts = Counter()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise SplitError(f"manifest map entry {index} lacks required fields")
        name = str(entry["filename"])
        pool = str(entry["pool"])
        expected_relative = f"{pool}/{name}"
        if name in seen_names:
            raise SplitError(f"manifest repeats filename: {name}")
        if pool not in POOL_COUNTS:
            raise SplitError(f"{name}: unknown pool {pool!r}")
        if str(entry["relative_path"]) != expected_relative:
            raise SplitError(f"{name}: relative path must be {expected_relative!r}")
        seen_names.add(name)
        pool_counts[pool] += 1

    if dict(pool_counts) != POOL_COUNTS:
        raise SplitError(f"manifest pool counts {dict(pool_counts)}; expected {POOL_COUNTS}")
    fail_on_duplicates(entries, "canonical_map_sha256", "canonical map content")
    fail_on_duplicates(entries, "csv_sha256", "exact CSV content")

    loose = sorted(path.name for path in pool_root.glob("*.csv"))
    if loose:
        raise SplitError(f"unexpected loose CSV maps in pool root: {loose[:10]}")
    actual_csvs = sorted(path.relative_to(pool_root).as_posix() for path in pool_root.rglob("*.csv"))
    expected_csvs = sorted(str(entry["relative_path"]) for entry in entries)
    if actual_csvs != expected_csvs:
        missing = sorted(set(expected_csvs) - set(actual_csvs))[:10]
        extra = sorted(set(actual_csvs) - set(expected_csvs))[:10]
        raise SplitError(f"manifest/filesystem disagreement; missing={missing}, extra={extra}")

    for entry in entries:
        path = pool_root / str(entry["relative_path"])
        content = path.read_bytes()
        validate_phase1_csv(content, str(entry["relative_path"]))
        if len(content) != int(entry["size_bytes"]):
            raise SplitError(f"{entry['relative_path']}: file size does not match manifest")
        actual_hash = sha256_bytes(content)
        if actual_hash != str(entry["csv_sha256"]):
            raise SplitError(f"{entry['relative_path']}: CSV SHA-256 does not match manifest")

    computed_pool_identities = {}
    for pool in POOL_COUNTS:
        members = [entry for entry in entries if entry["pool"] == pool]
        identity, _ = planned_pool_identity(members)
        computed_pool_identities[pool] = identity
    if manifest.get("pool_identities") != computed_pool_identities:
        raise SplitError("aggregate pool identities do not match manifest")

    print("Phase 1 split verified")
    for pool, count in POOL_COUNTS.items():
        print(f"  {pool:16s} {count:5d}  {computed_pool_identities[pool]}")
    print(f"  {'total':16s} {EXPECTED_TOTAL:5d}")
    print("  duplicate identities: 0")
    print("  cross-pool overlaps:   0")
    print("  loose root CSVs:       0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", type=Path, default=DEFAULT_POOL_ROOT)
    parser.add_argument("--conversion-manifest", type=Path, default=DEFAULT_CONVERSION_MANIFEST)
    parser.add_argument(
        "--establish",
        action="store_true",
        help="create the initial frozen split; without this flag the command is verification-only",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pool_root = args.pool_root.resolve()
        if not pool_root.is_dir():
            raise SplitError(f"pool root not found: {pool_root}")
        if args.establish:
            establish(pool_root, args.conversion_manifest.resolve())
        else:
            verify(pool_root)
    except SplitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
