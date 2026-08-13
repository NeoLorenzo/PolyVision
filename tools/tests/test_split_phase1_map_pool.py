import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import split_phase1_map_pool as split


def map_bytes(capital_x: int) -> bytes:
    rows = []
    for y in range(11):
        tokens = [".:" for _ in range(11)]
        if y == 0:
            tokens[capital_x] = "c:2"
        rows.append(",".join(tokens))
    return ("\n".join(rows) + "\n").encode("utf-8")


class SplitPhase1MapPoolTests(unittest.TestCase):
    def setUp(self):
        self.pool_counts = {"train": 1, "validation": 1, "test": 1, "human_benchmark": 1}
        self.patches = [
            mock.patch.object(split, "POOL_COUNTS", self.pool_counts),
            mock.patch.object(split, "EXPECTED_TOTAL", 4),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()

    def make_flat_corpus(self, root: Path) -> Path:
        manifest_path = root.parent / "conversion_manifest.csv"
        rows = []
        for index in range(4):
            name = f"map_{index + 1:06d}.csv"
            content = map_bytes(index)
            (root / name).write_bytes(content)
            rows.append(
                {
                    "csv_filename": name,
                    "csv_sha256": split.sha256_bytes(content),
                    "map_sha256": split.sha256_bytes(f"canonical-{index}".encode("ascii")),
                }
            )
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["csv_filename", "csv_sha256", "map_sha256"])
            writer.writeheader()
            writer.writerows(rows)
        return manifest_path

    def test_establish_is_deterministic_and_verify_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pool"
            root.mkdir()
            conversion_manifest = self.make_flat_corpus(root)
            split.establish(root, conversion_manifest)
            first = (root / split.MANIFEST_NAME).read_bytes()
            split.verify(root)
            with self.assertRaisesRegex(split.SplitError, "already exists"):
                split.establish(root, conversion_manifest)
            self.assertEqual(first, (root / split.MANIFEST_NAME).read_bytes())
            self.assertFalse(list(root.glob("*.csv")))
            self.assertEqual(
                {pool: len(list((root / pool).glob("*.csv"))) for pool in self.pool_counts},
                self.pool_counts,
            )

    def test_verify_rejects_content_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pool"
            root.mkdir()
            conversion_manifest = self.make_flat_corpus(root)
            split.establish(root, conversion_manifest)
            target = next((root / "train").glob("*.csv"))
            target.write_bytes(target.read_bytes() + b"\n")
            with self.assertRaisesRegex(split.SplitError, "rows|size|SHA-256"):
                split.verify(root)

    def test_initialization_rejects_duplicate_canonical_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pool"
            root.mkdir()
            conversion_manifest = self.make_flat_corpus(root)
            with conversion_manifest.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["map_sha256"] = rows[0]["map_sha256"]
            with conversion_manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(split.SplitError, "duplicate canonical map content"):
                split.build_initial_manifest(root, conversion_manifest)

    def test_partial_split_without_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pool"
            root.mkdir()
            conversion_manifest = self.make_flat_corpus(root)
            (root / "train").mkdir()
            (root / "map_000001.csv").replace(root / "train" / "map_000001.csv")
            with self.assertRaisesRegex(split.SplitError, "partial split"):
                split.build_initial_manifest(root, conversion_manifest)


if __name__ == "__main__":
    unittest.main()
