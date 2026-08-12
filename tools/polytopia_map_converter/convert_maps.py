#!/usr/bin/env python3
"""Convert canonical schema-v1 Polytopia maps into validated Tribes CSV maps."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable

SCHEMA_VERSION = 1

POLYTOPIA_TERRAIN = {1: "COAST", 2: "OCEAN", 3: "FIELD", 4: "MOUNTAIN", 5: "FOREST"}
POLYVISION_TERRAIN_CHAR = {"COAST": "s", "OCEAN": "d", "FIELD": ".", "MOUNTAIN": "m", "FOREST": "f"}
POLYTOPIA_RESOURCE = {1: "GAME", 2: "CROP", 3: "FISH", 4: "WHALE", 5: "METAL", 6: "FRUIT"}
POLYVISION_RESOURCE_CHAR = {"GAME": "a", "CROP": "c", "FISH": "h", "WHALE": "w", "METAL": "o", "FRUIT": "f"}
POLYTOPIA_IMPROVEMENT = {1: "CITY_OR_VILLAGE", 2: "RUIN", 47: "LIGHTHOUSE"}
POLYTOPIA_TRIBE = {1: "NATURE", 4: "BARDUR"}
POLYVISION_TRIBE_KEY = {"BARDUR": 2}

VALID_TERRAIN_CHARS = frozenset(POLYVISION_TERRAIN_CHAR.values()) | {"v", "c"}
VALID_SUFFIX_CHARS = frozenset(POLYVISION_RESOURCE_CHAR.values()) | {"r"}
MANIFEST_FIELDS = (
    "index", "source_json", "source_state_filename", "source_state_sha256", "map_sha256",
    "csv_filename", "csv_sha256", "width", "height", "game_version", "seed", "tribe",
    "capital_x", "capital_y", "validation_status",
)


class ConversionError(ValueError):
    """A canonical map cannot be converted without violating the CSV contract."""


def require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConversionError(f"{field} must be an integer")
    return value


def load_canonical(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read canonical JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConversionError("canonical JSON root must be an object")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ConversionError(f"Unsupported canonical map schema version: {version}")
    return data


def active_players(data: dict[str, Any]) -> list[dict[str, Any]]:
    game = data.get("game") or {}
    width = require_int(game.get("map_width"), "game.map_width")
    height = require_int(game.get("map_height"), "game.map_height")
    players = data.get("players")
    if not isinstance(players, list) or not players:
        raise ConversionError("at least one initial player is required")
    return [
        player for player in players
        if isinstance(player, dict)
        and isinstance(player.get("start_x"), int)
        and isinstance(player.get("start_y"), int)
        and 0 <= player["start_x"] < width
        and 0 <= player["start_y"] < height
    ]


def validate_canonical(data: dict[str, Any], source_name: str = "<map>") -> dict[str, Any]:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ConversionError(f"Unsupported canonical map schema version: {data.get('schema_version')}")
    game = data.get("game")
    if not isinstance(game, dict):
        raise ConversionError("game must be an object")
    width = require_int(game.get("map_width"), "game.map_width")
    height = require_int(game.get("map_height"), "game.map_height")
    if width <= 0 or height <= 0 or width != height:
        raise ConversionError(f"Phase-1 map must be positive and square, got {width}x{height}")
    require_int(game.get("game_version"), "game.game_version")
    require_int(game.get("seed"), "game.seed")

    players = data.get("players")
    if not isinstance(players, list) or not players:
        raise ConversionError("at least one initial player is required")
    active = active_players(data)
    if len(active) != 1:
        raise ConversionError(f"expected exactly one in-bounds starting player, found {len(active)}")
    player = active[0]
    player_id = require_int(player.get("player_id"), "player.player_id")
    tribe_id = require_int(player.get("tribe_id"), "player.tribe_id")
    tribe_name = POLYTOPIA_TRIBE.get(tribe_id)
    if tribe_name != "BARDUR":
        raise ConversionError(f"unsupported Phase-1 starting tribe ID {tribe_id} ({tribe_name or 'unknown'}); expected Bardur raw ID 4")
    player_ids = set()
    for i, candidate in enumerate(players):
        if not isinstance(candidate, dict):
            raise ConversionError(f"players[{i}] must be an object")
        candidate_id = require_int(candidate.get("player_id"), f"players[{i}].player_id")
        candidate_tribe = require_int(candidate.get("tribe_id"), f"players[{i}].tribe_id")
        if candidate_id in player_ids:
            raise ConversionError(f"duplicate player_id {candidate_id}")
        if candidate_tribe not in POLYTOPIA_TRIBE:
            raise ConversionError(f"unsupported player tribe ID {candidate_tribe}")
        player_ids.add(candidate_id)

    tiles = data.get("tiles")
    if not isinstance(tiles, list) or len(tiles) != width * height:
        actual = len(tiles) if isinstance(tiles, list) else "non-list"
        raise ConversionError(f"expected {width * height} tiles, found {actual}")
    by_coord: dict[tuple[int, int], dict[str, Any]] = {}
    capitals = []
    for index, tile in enumerate(tiles):
        if not isinstance(tile, dict):
            raise ConversionError(f"tiles[{index}] must be an object")
        x, y = require_int(tile.get("x"), f"tiles[{index}].x"), require_int(tile.get("y"), f"tiles[{index}].y")
        if not (0 <= x < width and 0 <= y < height):
            raise ConversionError(f"out-of-bounds coordinate ({x},{y})")
        if (x, y) in by_coord:
            raise ConversionError(f"duplicate coordinate ({x},{y})")
        by_coord[x, y] = tile
        terrain_id = require_int(tile.get("terrain_id"), f"tile ({x},{y}).terrain_id")
        if terrain_id not in POLYTOPIA_TERRAIN:
            raise ConversionError(f"tile ({x},{y}) has unsupported terrain ID {terrain_id}")
        owner_id = require_int(tile.get("owner_id"), f"tile ({x},{y}).owner_id")
        if owner_id not in (0, player_id):
            raise ConversionError(f"tile ({x},{y}) has unexpected owner ID {owner_id}")
        for flag_name in ("has_road", "has_water_route", "flooded"):
            if not isinstance(tile.get(flag_name), bool):
                raise ConversionError(f"tile ({x},{y}).{flag_name} must be boolean")
        if tile["has_road"] or tile["has_water_route"]:
            raise ConversionError(f"tile ({x},{y}) has a starting road/water route that Tribes CSV cannot encode")
        if tile["flooded"] or require_int(tile.get("flooded_value"), f"tile ({x},{y}).flooded_value") != 0:
            raise ConversionError(f"tile ({x},{y}) has flooding that Tribes CSV cannot encode")
        if require_int(tile.get("tile_skin"), f"tile ({x},{y}).tile_skin") != 0:
            raise ConversionError(f"tile ({x},{y}) has non-default tile skin that Tribes CSV cannot encode")
        resource = tile.get("resource")
        if resource is not None:
            if not isinstance(resource, dict):
                raise ConversionError(f"tile ({x},{y}) resource must be null or an object")
            resource_id = require_int(resource.get("id"), f"tile ({x},{y}).resource.id")
            if resource_id not in POLYTOPIA_RESOURCE:
                raise ConversionError(f"tile ({x},{y}) has unsupported resource ID {resource_id}")
        improvement = tile.get("improvement")
        improvement_id = None
        if improvement is not None:
            if not isinstance(improvement, dict):
                raise ConversionError(f"tile ({x},{y}) improvement must be null or an object")
            improvement_id = require_int(improvement.get("id"), f"tile ({x},{y}).improvement.id")
            if improvement_id not in POLYTOPIA_IMPROVEMENT:
                raise ConversionError(f"tile ({x},{y}) has unsupported improvement ID {improvement_id}")
        capital = tile.get("capital")
        if not isinstance(capital, bool):
            raise ConversionError(f"tile ({x},{y}).capital must be boolean")
        if capital:
            capitals.append(tile)
            if improvement_id != 1:
                raise ConversionError(f"capital tile ({x},{y}) must have city improvement ID 1")
            if resource is not None:
                raise ConversionError(f"capital tile ({x},{y}) unexpectedly contains a resource")
        if improvement_id == 1:
            if resource is not None:
                raise ConversionError(f"city/village tile ({x},{y}) unexpectedly contains a resource")
            if capital and owner_id != player_id:
                raise ConversionError(f"capital tile ({x},{y}) belongs to owner {owner_id}, expected {player_id}")
            if not capital and owner_id != 0:
                raise ConversionError(f"non-capital city improvement at ({x},{y}) is unexpectedly owned by {owner_id}")
        elif improvement_id == 2:
            if capital or owner_id != 0:
                raise ConversionError(f"ruin tile ({x},{y}) must be neutral and non-capital")
        elif improvement_id == 47:
            if (x, y) not in {(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)}:
                raise ConversionError(f"lighthouse improvement ID 47 must be on a corner, found ({x},{y})")
            if capital or owner_id != 0 or resource is not None:
                raise ConversionError(f"lighthouse tile ({x},{y}) must be neutral, non-capital, and resource-free")

    for y in range(height):
        for x in range(width):
            if (x, y) not in by_coord:
                raise ConversionError(f"missing coordinate ({x},{y})")
    if len(capitals) != 1:
        raise ConversionError(f"expected exactly one player-owned capital, found {len(capitals)}")
    capital = capitals[0]
    if capital["owner_id"] != player_id:
        raise ConversionError(f"capital owner {capital['owner_id']} does not match starting player {player_id}")
    if (capital["x"], capital["y"]) != (player["start_x"], player["start_y"]):
        raise ConversionError(
            f"capital coordinate ({capital['x']},{capital['y']}) does not match starting player coordinate "
            f"({player['start_x']},{player['start_y']})"
        )
    if (capital.get("capital_x"), capital.get("capital_y")) != (capital["x"], capital["y"]):
        raise ConversionError(f"capital metadata at ({capital['x']},{capital['y']}) is inconsistent")
    map_hash = data.get("map_sha256")
    if not isinstance(map_hash, str) or len(map_hash) != 64 or any(ch not in "0123456789abcdef" for ch in map_hash.lower()):
        raise ConversionError("map_sha256 must be a 64-character hexadecimal SHA-256")
    source = data.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("filename"), str):
        raise ConversionError("source identity is incomplete")
    source_hash = source.get("sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(ch not in "0123456789abcdef" for ch in source_hash.lower()):
        raise ConversionError("source.sha256 must be a 64-character hexadecimal SHA-256")
    return {"width": width, "height": height, "player": player, "capital": capital, "tiles": by_coord, "tribe": tribe_name, "source_name": source_name}


def convert_tile(tile: dict[str, Any], context: dict[str, Any]) -> str:
    improvement = tile.get("improvement")
    improvement_id = improvement["id"] if improvement else None
    if tile["capital"]:
        return f"c:{POLYVISION_TRIBE_KEY[context['tribe']]}"
    if improvement_id == 1:
        return "v:"
    terrain_name = POLYTOPIA_TERRAIN[tile["terrain_id"]]
    terrain_char = POLYVISION_TERRAIN_CHAR[terrain_name]
    if improvement_id == 2:
        return f"{terrain_char}:r"
    # Lighthouses (ID 47) cannot be represented by the Tribes CSV format.
    # They were strictly validated as neutral corner markers above; preserve
    # their underlying terrain and keep their full identity in canonical JSON.
    resource = tile.get("resource")
    if resource is not None:
        resource_name = POLYTOPIA_RESOURCE[resource["id"]]
        return f"{terrain_char}:{POLYVISION_RESOURCE_CHAR[resource_name]}"
    return f"{terrain_char}:"


def convert_map(data: dict[str, Any], source_name: str = "<map>") -> tuple[bytes, dict[str, Any]]:
    context = validate_canonical(data, source_name)
    rows = []
    for y in range(context["height"]):
        rows.append(",".join(convert_tile(context["tiles"][x, y], context) for x in range(context["width"])))
    content = ("\n".join(rows) + "\n").encode("utf-8")
    validate_csv_bytes(content, context["width"], context["height"])
    return content, context


def validate_csv_bytes(content: bytes, width: int, height: int) -> None:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConversionError(f"generated CSV is not UTF-8: {exc}") from exc
    rows = text.splitlines()
    if len(rows) != height:
        raise ConversionError(f"generated CSV has {len(rows)} rows; expected {height}")
    capitals = 0
    for y, row in enumerate(rows):
        tokens = row.split(",")
        if len(tokens) != width:
            raise ConversionError(f"generated CSV row {y} has {len(tokens)} columns; expected {width}")
        for x, token in enumerate(tokens):
            if token.count(":") != 1:
                raise ConversionError(f"generated CSV token ({x},{y}) is malformed: {token!r}")
            terrain, suffix = token.split(":", 1)
            if len(terrain) != 1 or terrain not in VALID_TERRAIN_CHARS:
                raise ConversionError(f"generated CSV token ({x},{y}) has invalid terrain {terrain!r}")
            if terrain == "c":
                if suffix != "2":
                    raise ConversionError(f"generated capital ({x},{y}) must be c:2, got {token!r}")
                capitals += 1
            elif terrain == "v":
                if suffix:
                    raise ConversionError(f"generated village ({x},{y}) cannot have a suffix")
            elif suffix and suffix not in VALID_SUFFIX_CHARS:
                raise ConversionError(f"generated CSV token ({x},{y}) has invalid resource {suffix!r}")
    if capitals != 1:
        raise ConversionError(f"generated CSV has {capitals} c:2 capitals; expected exactly one")


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def csv_name_for_json(path: Path) -> str:
    return f"{path.stem}.csv"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def manifest_row(index: int, json_path: Path, csv_path: Path, data: dict[str, Any], content: bytes, context: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "index": index, "source_json": json_path.name,
        "source_state_filename": data["source"]["filename"], "source_state_sha256": data["source"]["sha256"],
        "map_sha256": data["map_sha256"], "csv_filename": csv_path.name, "csv_sha256": sha256_bytes(content),
        "width": context["width"], "height": context["height"], "game_version": data["game"]["game_version"],
        "seed": data["game"]["seed"], "tribe": context["tribe"], "capital_x": context["capital"]["x"],
        "capital_y": context["capital"]["y"], "validation_status": status,
    }


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return {row["csv_filename"]: row for row in csv.DictReader(handle)}
    except (OSError, KeyError, csv.Error) as exc:
        raise ConversionError(f"cannot read existing manifest {path}: {exc}") from exc


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def counter_label(counter: collections.Counter[Any], names: dict[int, str] | None = None) -> list[str]:
    lines = []
    for key in sorted(counter, key=lambda value: str(value)):
        name = f" {names[key]}" if names and key in names else ""
        lines.append(f"  {key}{name}: {counter[key]}")
    return lines


def analyze_maps(loaded: list[tuple[Path, dict[str, Any]]]) -> tuple[dict[str, collections.Counter[Any]], dict[str, list[str]]]:
    counters = {name: collections.Counter() for name in ("schema", "versions", "dimensions", "tribes", "terrains", "resources", "improvements", "climates", "owners", "capital_flags")}
    hashes: dict[str, list[str]] = collections.defaultdict(list)
    for path, data in loaded:
        counters["schema"][data.get("schema_version")] += 1
        game = data.get("game") or {}
        counters["versions"][game.get("game_version")] += 1
        counters["dimensions"][f"{game.get('map_width')}x{game.get('map_height')}"] += 1
        for player in data.get("players") or []:
            if isinstance(player, dict): counters["tribes"][player.get("tribe_id")] += 1
        for tile in data.get("tiles") or []:
            if not isinstance(tile, dict): continue
            counters["terrains"][tile.get("terrain_id")] += 1; counters["climates"][tile.get("climate_id")] += 1
            counters["owners"][tile.get("owner_id")] += 1; counters["capital_flags"][tile.get("capital")] += 1
            if isinstance(tile.get("resource"), dict): counters["resources"][tile["resource"].get("id")] += 1
            if isinstance(tile.get("improvement"), dict): counters["improvements"][tile["improvement"].get("id")] += 1
        if isinstance(data.get("map_sha256"), str): hashes[data["map_sha256"]].append(path.name)
    return counters, {key: value for key, value in hashes.items() if len(value) > 1}


def print_analysis(loaded: list[tuple[Path, dict[str, Any]]], counters: dict[str, collections.Counter[Any]], duplicates: dict[str, list[str]]) -> None:
    print("POLYVISION — CANONICAL MAP ANALYSIS\n")
    print(f"Maps: {len(loaded)}")
    sections = (
        ("Schema versions", "schema", None), ("Game versions", "versions", None), ("Dimensions", "dimensions", None),
        ("Player tribe IDs", "tribes", POLYTOPIA_TRIBE), ("Terrain IDs", "terrains", POLYTOPIA_TERRAIN),
        ("Resource IDs", "resources", POLYTOPIA_RESOURCE), ("Improvement IDs", "improvements", POLYTOPIA_IMPROVEMENT),
        ("Climate IDs", "climates", None), ("Owner IDs", "owners", None), ("Capital flags", "capital_flags", None),
    )
    for title, key, names in sections:
        print(f"\n{title}:")
        print("\n".join(counter_label(counters[key], names)) or "  none")
    print(f"\nUnique map_sha256: {len(loaded) - sum(len(v) - 1 for v in duplicates.values())}")
    print(f"Duplicate hash groups: {len(duplicates)}")


def run_java_validation(repo_root: Path, input_dir: Path, output_dir: Path) -> None:
    tribes = repo_root / "pol_env" / "Tribes"
    helper = repo_root / "tools" / "polytopia_map_converter" / "java" / "core" / "game" / "CanonicalCsvValidator.java"
    engine_out, json_jar = tribes / "out", tribes / "lib" / "json.jar"
    if not engine_out.exists():
        raise ConversionError(f"Java engine classes are missing: {engine_out}; compile Tribes first")
    with tempfile.TemporaryDirectory(prefix="polyvision-java-validator-") as temp:
        compile_command = ["javac", "-cp", os.pathsep.join((str(engine_out), str(json_jar))), "-d", temp, str(helper)]
        result = subprocess.run(compile_command, cwd=repo_root, text=True, capture_output=True)
        if result.returncode:
            raise ConversionError(f"Java validator compilation failed: {(result.stderr or result.stdout).strip()}")
        classpath = os.pathsep.join((temp, str(engine_out), str(json_jar)))
        result = subprocess.run(
            ["java", "-cp", classpath, "core.game.CanonicalCsvValidator", str(input_dir), str(output_dir)],
            cwd=repo_root, text=True, capture_output=True,
        )
        if result.stdout: print(result.stdout.rstrip())
        if result.returncode:
            raise ConversionError(f"Java LevelLoader validation failed: {result.stderr.strip() or 'unknown error'}")


def discover_inputs(path: Path) -> tuple[list[Path], bool]:
    if path.is_file():
        if path.suffix.lower() != ".json": raise ConversionError("single input must have a .json extension")
        return [path], False
    if not path.is_dir(): raise ConversionError(f"input does not exist: {path}")
    return sorted((item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".json"), key=lambda item: item.name), True


def default_manifest_path(repo_root: Path) -> Path:
    return repo_root / "data" / "polytopia_maps" / "conversion_manifest.csv"


def execute(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    input_path, output_path = Path(args.input).resolve(), Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else default_manifest_path(repo_root)
    try:
        inputs, batch = discover_inputs(input_path)
    except ConversionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 1
    if not batch and output_path.exists() and output_path.is_dir(): output_path = output_path / csv_name_for_json(inputs[0])
    if batch: output_path.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(inputs)} canonical JSON files.\n")

    loaded, failures = [], []
    for path in inputs:
        try: loaded.append((path, load_canonical(path)))
        except ConversionError as exc: failures.append((path.name, str(exc)))
    counters, duplicates = analyze_maps(loaded)
    print_analysis(loaded, counters, duplicates)
    duplicate_files = {name for names in duplicates.values() for name in names}
    for hash_value, names in duplicates.items(): failures.append((", ".join(names), f"duplicate map_sha256 {hash_value}"))

    try: existing_manifest = read_manifest(manifest_path)
    except ConversionError as exc: print(f"ERROR: {exc}", file=sys.stderr); return 1
    rows, converted, skipped, validated = [], 0, 0, 0
    print("\nPOLYVISION — JSON TO TRIBES CSV\n")
    for index, (json_path, data) in enumerate(loaded, 1):
        if json_path.name in duplicate_files: continue
        csv_path = output_path / csv_name_for_json(json_path) if batch else output_path
        try:
            content, context = convert_map(data, json_path.name)
            validated += 1
            expected_hash = sha256_bytes(content)
            if csv_path.exists() and not args.overwrite:
                prior = existing_manifest.get(csv_path.name)
                actual_hash = sha256_bytes(csv_path.read_bytes())
                if not prior or prior.get("map_sha256") != data["map_sha256"] or prior.get("csv_sha256") != actual_hash or actual_hash != expected_hash:
                    raise ConversionError(f"existing output {csv_path} does not match this canonical map/manifest; use --overwrite only after review")
                validate_csv_bytes(csv_path.read_bytes(), context["width"], context["height"])
                skipped += 1; action = "skipped (verified existing)"
            else:
                atomic_write(csv_path, content)
                converted += 1; action = "converted"
            rows.append(manifest_row(index, json_path, csv_path, data, content, context, "PYTHON_VALIDATED"))
            if args.verbose or not batch: print(f"[{index}/{len(loaded)}] {json_path.name} -> {csv_path.name} {action}")
        except (ConversionError, OSError) as exc:
            failures.append((json_path.name, str(exc)))
            print(f"[{index}/{len(loaded)}] {json_path.name} FAILED: {exc}", file=sys.stderr)

    java_validated = False
    if args.java_validate and rows and not failures:
        if not batch:
            print("ERROR: --java-validate currently requires directory input/output", file=sys.stderr); failures.append((inputs[0].name, "Java validation requires batch mode"))
        else:
            try:
                run_java_validation(repo_root, input_path, output_path)
                java_validated = True
                for row in rows: row["validation_status"] = "JAVA_SEMANTIC_VALIDATED"
            except ConversionError as exc: failures.append(("Java LevelLoader", str(exc)))
    if rows:
        try: write_manifest(manifest_path, rows)
        except OSError as exc: failures.append((manifest_path.name, f"cannot write manifest: {exc}"))

    print("\nComplete.")
    print(f"\nFound:        {len(inputs)}\nValidated:    {validated}\nConverted:    {converted}\nSkipped:      {skipped}\nFailed:       {len(failures)}")
    print(f"Unique maps:  {len(loaded) - sum(len(v) - 1 for v in duplicates.values())}")
    print(f"Java checked: {len(rows) if java_validated else 0}")
    print(f"Output:       {output_path}\nManifest:     {manifest_path}")
    if failures:
        print("\nFailures:", file=sys.stderr)
        for name, reason in failures: print(f"  {name}: {reason}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="canonical schema-v1 JSON file or directory")
    parser.add_argument("--output", required=True, help="Tribes CSV file or output directory")
    parser.add_argument("--manifest", help="manifest CSV path (default: data/polytopia_maps/conversion_manifest.csv)")
    parser.add_argument("--overwrite", action="store_true", help="regenerate existing CSV files")
    parser.add_argument("--verbose", action="store_true", help="print each successful conversion")
    parser.add_argument("--java-validate", action="store_true", help="load and semantically compare all outputs through Java LevelLoader")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    return execute(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
