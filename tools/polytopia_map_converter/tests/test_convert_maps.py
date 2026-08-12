from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.polytopia_map_converter import convert_maps as converter


def make_map(size: int = 3) -> dict:
    tiles = []
    for y in range(size):
        for x in range(size):
            tiles.append({
                "x": x, "y": y, "terrain_id": 3, "climate_id": 4, "altitude": 1,
                "owner_id": 0, "capital": False, "capital_x": -1, "capital_y": -1,
                "resource": None, "improvement": None, "has_road": False,
                "has_water_route": False, "tile_skin": 0, "flooded": False, "flooded_value": 0,
            })
    capital = tiles[1 * size + 2]
    capital.update({
        "owner_id": 1, "capital": True, "capital_x": 2, "capital_y": 1,
        "improvement": {"id": 1},
    })
    return {
        "schema_version": 1,
        "source": {"filename": "fixture.state", "sha256": "a" * 64, "size_bytes": 1},
        "map_sha256": "b" * 64,
        "game": {"game_version": 122, "seed": 42, "map_width": size, "map_height": size},
        "players": [
            {"player_id": 1, "tribe_id": 4, "start_x": 2, "start_y": 1},
            {"player_id": 4, "tribe_id": 1, "start_x": -1, "start_y": -1},
        ],
        "tiles": tiles,
    }


class MappingTests(unittest.TestCase):
    def context(self) -> dict:
        return {"tribe": "BARDUR"}

    def tile(self, terrain: int, resource: int | None = None, improvement: int | None = None, capital: bool = False) -> dict:
        return {
            "terrain_id": terrain, "resource": None if resource is None else {"id": resource},
            "improvement": None if improvement is None else {"id": improvement}, "capital": capital,
        }

    def test_every_terrain_mapping(self):
        expected = {1: "s:", 2: "d:", 3: ".:", 4: "m:", 5: "f:"}
        for raw_id, token in expected.items():
            with self.subTest(raw_id=raw_id):
                self.assertEqual(token, converter.convert_tile(self.tile(raw_id), self.context()))

    def test_every_resource_mapping(self):
        expected = {1: "a", 2: "c", 3: "h", 4: "w", 5: "o", 6: "f"}
        for raw_id, suffix in expected.items():
            with self.subTest(raw_id=raw_id):
                self.assertEqual(f".:{suffix}", converter.convert_tile(self.tile(3, raw_id), self.context()))

    def test_representative_tokens_and_precedence(self):
        self.assertEqual("c:2", converter.convert_tile(self.tile(3, improvement=1, capital=True), self.context()))
        self.assertEqual("v:", converter.convert_tile(self.tile(3, improvement=1), self.context()))
        self.assertEqual("f:r", converter.convert_tile(self.tile(5, resource=1, improvement=2), self.context()))
        self.assertEqual(".:f", converter.convert_tile(self.tile(3, resource=6), self.context()))
        self.assertEqual("f:a", converter.convert_tile(self.tile(5, resource=1), self.context()))
        self.assertEqual("m:o", converter.convert_tile(self.tile(4, resource=5), self.context()))
        self.assertEqual("f:", converter.convert_tile(self.tile(5), self.context()))
        self.assertEqual(".:", converter.convert_tile(self.tile(3, improvement=47), self.context()))


class ValidationTests(unittest.TestCase):
    def assert_invalid(self, mutation, message: str):
        data = make_map()
        mutation(data)
        with self.assertRaisesRegex(converter.ConversionError, message):
            converter.convert_map(data)

    def test_schema_rejected(self):
        self.assert_invalid(lambda d: d.update(schema_version=2), "Unsupported canonical map schema version: 2")

    def test_unknown_ids_rejected(self):
        for field, value, message in (
            ("terrain_id", 999, "unsupported terrain ID 999"),
            ("resource", {"id": 999}, "unsupported resource ID 999"),
            ("improvement", {"id": 999}, "unsupported improvement ID 999"),
        ):
            with self.subTest(field=field):
                self.assert_invalid(lambda d, f=field, v=value: d["tiles"][0].update({f: v}), message)

    def test_zero_or_multiple_capitals_rejected(self):
        def remove(data):
            data["tiles"][5].update(owner_id=0, capital=False, capital_x=-1, capital_y=-1, improvement=None)
        self.assert_invalid(remove, "exactly one player-owned capital, found 0")

        def add(data):
            data["tiles"][0].update(owner_id=1, capital=True, capital_x=0, capital_y=0, improvement={"id": 1})
        self.assert_invalid(add, "exactly one player-owned capital, found 2")

    def test_lighthouse_is_strictly_corner_only(self):
        self.assert_invalid(lambda d: d["tiles"][4].update(improvement={"id": 47}), "must be on a corner")

    def test_unrepresentable_starting_tile_features_fail(self):
        self.assert_invalid(lambda d: d["tiles"][0].update(has_road=True), "road/water route")
        self.assert_invalid(lambda d: d["tiles"][0].update(flooded=True, flooded_value=1), "flooding")
        self.assert_invalid(lambda d: d["tiles"][0].update(tile_skin=3), "non-default tile skin")

    def test_coordinate_orientation_and_determinism(self):
        data = make_map()
        data["tiles"][0].update(terrain_id=5, resource={"id": 1})       # x=0,y=0 -> f:a
        data["tiles"][1].update(terrain_id=4, resource={"id": 5})       # x=1,y=0 -> m:o
        data["tiles"][3].update(terrain_id=3, resource={"id": 6})       # x=0,y=1 -> .:f
        first, _ = converter.convert_map(data)
        second, _ = converter.convert_map(copy.deepcopy(data))
        self.assertEqual(first, second)
        rows = first.decode().splitlines()
        self.assertEqual(["f:a", "m:o", ".:"], rows[0].split(","))
        self.assertEqual(".:f", rows[1].split(",")[0])
        self.assertEqual("c:2", rows[1].split(",")[2])

    def test_eleven_by_eleven_dimensions(self):
        content, _ = converter.convert_map(make_map(11))
        rows = content.decode().splitlines()
        self.assertEqual(11, len(rows))
        self.assertTrue(all(len(row.split(",")) == 11 for row in rows))

    def test_duplicate_hash_analysis(self):
        data = make_map()
        _, duplicates = converter.analyze_maps([(Path("a.json"), data), (Path("b.json"), copy.deepcopy(data))])
        self.assertEqual({"b" * 64: ["a.json", "b.json"]}, duplicates)


class CliTests(unittest.TestCase):
    def options(self, input_path: Path, output_path: Path, manifest: Path, overwrite: bool = False) -> argparse.Namespace:
        return argparse.Namespace(
            input=str(input_path), output=str(output_path), manifest=str(manifest), overwrite=overwrite,
            verbose=False, java_validate=False,
        )

    def test_existing_output_skip_and_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name); input_dir, output_dir = temp / "input", temp / "output"
            input_dir.mkdir(); source = input_dir / "map_000001.json"
            source.write_text(json.dumps(make_map()), encoding="utf-8")
            manifest = temp / "manifest.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, converter.execute(self.options(input_dir, output_dir, manifest)))
            csv_path = output_dir / "map_000001.csv"; original = csv_path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, converter.execute(self.options(input_dir, output_dir, manifest)))
                self.assertEqual(0, converter.execute(self.options(input_dir, output_dir, manifest, overwrite=True)))
            self.assertEqual(original, csv_path.read_bytes())
            rows = converter.read_manifest(manifest)
            self.assertEqual("b" * 64, rows["map_000001.csv"]["map_sha256"])
            self.assertEqual(converter.sha256_bytes(original), rows["map_000001.csv"]["csv_sha256"])

    def test_inconsistent_existing_output_fails(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name); source = temp / "map.json"; output = temp / "map.csv"; manifest = temp / "manifest.csv"
            source.write_text(json.dumps(make_map()), encoding="utf-8"); output.write_text("tampered\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(1, converter.execute(self.options(source, output, manifest)))

    def test_batch_continues_after_invalid_input(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name); input_dir, output_dir = temp / "input", temp / "output"; input_dir.mkdir()
            valid, invalid = make_map(), make_map(); invalid["tiles"][0]["terrain_id"] = 999
            (input_dir / "a_invalid.json").write_text(json.dumps(invalid), encoding="utf-8")
            (input_dir / "b_valid.json").write_text(json.dumps(valid), encoding="utf-8")
            # Give the invalid map a distinct hash so dataset duplicate detection
            # does not mask its tile-level validation failure.
            invalid["map_sha256"] = "c" * 64
            (input_dir / "a_invalid.json").write_text(json.dumps(invalid), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = converter.execute(self.options(input_dir, output_dir, temp / "manifest.csv"))
            self.assertEqual(1, code)
            self.assertTrue((output_dir / "b_valid.csv").exists())
            self.assertFalse((output_dir / "a_invalid.csv").exists())


class RealFixtureTests(unittest.TestCase):
    def test_real_map_000001_when_available(self):
        path = Path(__file__).resolve().parents[3] / "data" / "polytopia_maps" / "parsed" / "map_000001.json"
        if not path.exists(): self.skipTest("local canonical corpus unavailable")
        data = converter.load_canonical(path)
        content, context = converter.convert_map(data, path.name)
        self.assertEqual((11, 11), (context["width"], context["height"]))
        self.assertEqual(11, len(content.decode().splitlines()))
        self.assertEqual("c:2", content.decode().splitlines()[context["capital"]["y"]].split(",")[context["capital"]["x"]])


if __name__ == "__main__":
    unittest.main()
