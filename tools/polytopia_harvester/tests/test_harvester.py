from __future__ import annotations

import csv
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


HARVESTER_DIR = Path(__file__).resolve().parents[1]
if str(HARVESTER_DIR) not in sys.path:
    sys.path.insert(0, str(HARVESTER_DIR))

from config import (  # noqa: E402
    BUTTON_SEQUENCE,
    ConfigError,
    HarvesterConfig,
    load_config,
    save_config,
    validate_screen_resolution,
)
from manifest import (  # noqa: E402
    DatasetManager,
    DuplicateStateError,
    sha256_file,
    state_filename,
)
from progress import ProgressTracker, estimate_remaining_seconds  # noqa: E402
from save_watcher import SaveWatcher  # noqa: E402
from harvest import harvest_one_map  # noqa: E402


class NamingAndResumeTests(unittest.TestCase):
    def test_state_filename_padding(self) -> None:
        self.assertEqual(state_filename(1), "map_000001.state")
        self.assertEqual(state_filename(999), "map_000999.state")

    def test_resume_and_target_calculation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = DatasetManager(Path(temporary))
            for index in (1, 2, 3):
                (manager.raw_dir / state_filename(index)).write_bytes(str(index).encode())
            self.assertEqual(manager.existing_count(), 3)
            self.assertEqual(manager.next_index(), 4)
            self.assertEqual(manager.remaining_to_target(10_000), 9_997)
            self.assertEqual(max(0, 10_000 - 4_382), 5_618)


class DatasetTests(unittest.TestCase):
    def test_hash_duplicate_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_a = root / "Game.state"
            source_b = root / "Reused.state"
            source_a.write_bytes(b"genuine-state-payload")
            source_b.write_bytes(b"genuine-state-payload")
            manager = DatasetManager(root / "dataset")

            record = manager.store_unique_state(source_a, 1, 7.92)
            self.assertEqual(record.filename, "map_000001.state")
            self.assertEqual(record.size_bytes, len(b"genuine-state-payload"))
            self.assertEqual(record.sha256, sha256_file(source_a))
            self.assertEqual(
                sha256_file(manager.raw_dir / record.filename), sha256_file(source_a)
            )

            with manager.manifest_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["index"], "1")
            self.assertEqual(rows[0]["source_filename"], "Game.state")
            self.assertEqual(rows[0]["sha256"], record.sha256)

            with self.assertRaises(DuplicateStateError):
                manager.store_unique_state(source_b, 2, 8.0)
            self.assertFalse((manager.raw_dir / "map_000002.state").exists())


class ProgressTests(unittest.TestCase):
    def test_eta_uses_known_cycle_average(self) -> None:
        estimate = estimate_remaining_seconds(10, [7.0, 8.0, 9.0])
        self.assertAlmostEqual(estimate or 0.0, 80.0)

    def test_console_average_uses_only_last_ten_cycles(self) -> None:
        tracker = ProgressTracker(100, historical_times=[100.0] * 5 + [5.0] * 10)
        self.assertEqual(len(tracker.recent), 10)
        self.assertAlmostEqual(tracker.average or 0.0, 5.0)
        tracker.record(15.0)
        self.assertAlmostEqual(tracker.average or 0.0, 6.0)


class ConfigTests(unittest.TestCase):
    def valid_config(self) -> HarvesterConfig:
        return HarvesterConfig(
            buttons={key: (100, 200) for key, _ in BUTTON_SEQUENCE},
            screen_width=1920,
            screen_height=1080,
            save_dir=r"C:\fake\Singleplayer",
            output_dir=r"C:\fake\dataset",
        )

    def test_config_round_trip_and_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            save_config(self.valid_config(), path)
            loaded = load_config(path)
            self.assertEqual(loaded.buttons["new_game"], (100, 200))
            self.assertEqual((loaded.screen_width, loaded.screen_height), (1920, 1080))

    def test_missing_button_is_rejected(self) -> None:
        config = self.valid_config()
        del config.buttons["ok"]
        with self.assertRaises(ConfigError):
            config.validate()

    def test_screen_resolution_guard(self) -> None:
        config = self.valid_config()
        validate_screen_resolution(config, (1920, 1080))
        with self.assertRaises(ConfigError):
            validate_screen_resolution(config, (2560, 1440))

    def test_start_game_delay_matches_observed_ok_boundary(self) -> None:
        self.assertEqual(HarvesterConfig().click_delays["start_game"], 3.5)

    def test_each_click_has_an_independent_delay(self) -> None:
        delays = HarvesterConfig().click_delays
        self.assertEqual(set(delays), {key for key, _ in BUTTON_SEQUENCE})
        self.assertEqual(delays["new_game"], 0.005)
        self.assertEqual(delays["bardur"], 0.25)
        self.assertEqual(delays["exit_to_menu"], 0.005)


class SaveWatcherTests(unittest.TestCase):
    def test_detects_existing_filename_modification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "Game.state"
            state.write_bytes(b"old")
            watcher = SaveWatcher(root, poll_interval=0.01, stability_seconds=0.03)
            baseline = watcher.snapshot()
            time.sleep(0.02)
            state.write_bytes(b"new-payload")
            changed = watcher.wait_for_changed_save(baseline, timeout_seconds=1.0)
            self.assertEqual(changed, state)

    def test_detects_new_save(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            watcher = SaveWatcher(root, poll_interval=0.01, stability_seconds=0.03)
            baseline = watcher.snapshot()
            state = root / "NewGame.state"
            state.write_bytes(b"new")
            self.assertEqual(
                watcher.wait_for_changed_save(baseline, timeout_seconds=1.0), state
            )

    def test_stability_waits_while_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "Game.state"
            state.write_bytes(b"a")
            watcher = SaveWatcher(root, poll_interval=0.01, stability_seconds=0.08)

            def mutate() -> None:
                for payload in (b"ab", b"abc", b"abcd"):
                    time.sleep(0.04)
                    state.write_bytes(payload)

            thread = threading.Thread(target=mutate)
            thread.start()
            started = time.monotonic()
            final_state = watcher.wait_until_stable(state, timeout_seconds=2.0)
            elapsed = time.monotonic() - started
            thread.join()
            self.assertEqual(final_state.size, 4)
            self.assertGreaterEqual(elapsed, 0.16)


class CycleOrderTests(unittest.TestCase):
    def test_exact_click_and_extraction_order(self) -> None:
        events = []

        class FakeUI:
            click_delays = {"settings": 0.0, "exit_to_menu": 0.0}

            def click(self, name):
                events.append(f"click:{name}")

        class FakeWatcher:
            def snapshot(self):
                events.append("snapshot")
                return {"old": "state"}

            def wait_for_changed_save(self, baseline, timeout_seconds):
                self_baseline = baseline
                self_timeout = timeout_seconds
                events.append("changed")
                return Path("Game.state")

            def wait_until_stable(self, path, timeout_seconds):
                events.append("stable")

        class FakeDataset:
            def store_unique_state(self, source, index, cycle_seconds):
                events.append("copy")
                return SimpleNamespace(filename="map_000001.state")

        timing = SimpleNamespace(
            save_timeout_seconds=1.0,
        )
        harvest_one_map(FakeUI(), FakeWatcher(), FakeDataset(), 1, timing)
        self.assertEqual(
            events,
            [
                "click:new_game",
                "click:creative",
                "click:bardur",
                "click:pick",
                "snapshot",
                "click:start_game",
                "click:ok",
                "changed",
                "stable",
                "copy",
                "click:settings",
                "click:exit_to_menu",
            ],
        )


if __name__ == "__main__":
    unittest.main()
