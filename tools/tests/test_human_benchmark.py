import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pol_env.Tribes.py.register_env import GlobalActionCatalog
from tools import human_benchmark as benchmark
from tools.human_policy_interface import (
    EpisodeResult,
    _print_actions,
    action_feature_annotations,
    extract_visible_units,
    move_direction,
    policy_visible_actions,
    run_policy_visible_episode,
    visible_map_lines,
    visible_state,
)
from tools.validate_human_benchmark_parity import assert_information_safety


class FakeEnv:
    MAX_TURNS = 10
    MAX_LEGAL_ACTIONS_DEFAULT = 4
    ALLOWED_ACTION_TYPES = {"END_TURN", "MOVE"}
    ACTION_FEATURE_DIM = 42
    LEGAL_ACTION_FEATURE_NAMES = tuple(f"feat_{i}" for i in range(42))

    def __init__(self):
        self.unwrapped = self
        self._catalog = GlobalActionCatalog(11, 11, ["HUNTING"], ["WARRIOR"], ["ANIMAL"], ["LUMBER_HUT"], ["WORKSHOP"])
        self.action_space = SimpleNamespace(n=self._catalog.total_size)
        self._terminal_spt_reward_enabled = False
        self._terminal_spt_base_weight = 1.0
        self._terminal_spt_over_10_weight = 2.0
        self._terminal_spt_over_15_weight = 3.0
        self._resource_gather_upgrade_filter_enabled = False
        self.stepped = []
        self.obs = np.zeros((586,), dtype=np.float32)
        self.obs[:121] = 7
        self.obs[121:363] = -1
        self.info = {
            "map_width": 11,
            "map_height": 11,
            "observation_dim": 586,
            "global_action_space_n": self._catalog.total_size,
            "max_legal_actions": 4,
            "catalog_version": "flat-v1",
            "phase1_opening_version": "v2_guaranteed_two_unit",
            "action_catalog_fingerprint": "catalog",
            "canonicalizer_version": "flat-v1-structured",
            "legal_action_feature_version": "features",
            "legal_action_feature_dim": 42,
            "info_mode": "fast",
            "legal_global_ids_padded": np.array([0, 0, 0, 0]),
            "legal_action_valid_mask": np.array([True, False, False, False]),
            "legal_action_features_padded": np.zeros((4, 42), dtype=np.float32),
            "legal_action_count": 1,
            "turn_count": 1,
        }

    def reset(self, seed):
        return self.obs.copy(), dict(self.info)

    def step(self, gid):
        self.stepped.append(int(gid))
        info = dict(self.info)
        info.update(
            {
                "selected_global_id": int(gid),
                "turn_count": 11,
                "spt": 17,
                "terminal_final_spt": 17,
                "stars": 4,
                "city_count": 2,
                "unit_count": 2,
            }
        )
        return self.obs.copy(), 1.5, False, True, info


def completed_result(spt=20):
    return EpisodeResult(
        status="completed",
        started_at_utc="2026-08-13T00:00:00Z",
        ended_at_utc="2026-08-13T00:01:00Z",
        episode_seed=42,
        shaped_return=3.0,
        decision_count=2,
        final_visible_metrics={"spt": spt},
        final_info_metrics={"terminal_final_spt": spt, "spt": spt},
        environment_contract={"wrapper_class": "test"},
        action_history=[{"global_id": 0}],
    )


class PolicyInterfaceTests(unittest.TestCase):
    def test_menu_selection_executes_exact_global_id(self):
        env = FakeEnv()
        result = run_policy_visible_episode(
            env,
            episode_seed=42,
            official=False,
            selector=lambda actions, _obs, _info, _step: actions[0]["global_id"],
            output_fn=lambda _text: None,
        )
        self.assertEqual(env.stepped, [0])
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.action_history[0]["global_id"], 0)
        self.assertEqual(result.action_history[0]["description"], "End turn")

    def test_action_feature_annotations_decoding(self):
        # Test move features
        feat_move = np.zeros((42,), dtype=np.float32)
        feat_move[0] = 1.0  # is_move
        feat_move[1] = 3.0 / 12.0  # newly revealed = 3
        feat_move[2] = 2.0 / 8.0  # adj fog after = 2
        feat_move[6] = 1.0  # has visible village
        feat_move[7] = 0.5  # closer to village
        feat_move[8] = 0.0  # no backtrack
        feat_move[10] = 1.0  # away from capital
        feat_move[11] = 1.0  # warrior
        ann = action_feature_annotations(feat_move, "MOVE")
        self.assertIn("reveal +3", ann)
        self.assertIn("adjacent fog 2", ann)
        self.assertIn("closer to village", ann)
        self.assertIn("away from capital", ann)

        # Test economy features
        feat_eco = np.zeros((42,), dtype=np.float32)
        feat_eco[16] = 1.0  # is_resource_gathering
        feat_eco[26] = 1.0  # resource_is_animal
        feat_eco[36] = 1.0 / 2.0  # pop_delta = +1
        feat_eco[37] = 0.0  # spt_delta = 0
        feat_eco[38] = 1.0  # makes_level_up_available = True
        feat_eco[40] = 0.67  # progress_before = 67%
        ann_eco = action_feature_annotations(feat_eco, "RESOURCE_GATHERING")
        self.assertIn("pop +1", ann_eco)
        self.assertIn("city progress 67%", ann_eco)
        self.assertIn("makes level-up ready", ann_eco)
        self.assertIn("resource: ANIMAL", ann_eco)

    def test_visible_state_extended_economy_decoding(self):
        obs = np.zeros((586,), dtype=np.float32)
        obs[363] = 8.0  # legacy stars
        obs[364] = 100.0  # score
        obs[365] = 3.0  # city count
        obs[490] = 8.0 / 50.0  # current stars norm
        obs[491] = 11.0 / 30.0  # current spt norm -> 11
        obs[492] = 6.0 / 10.0  # turn count norm -> 6
        obs[495] = 1.0  # tech organization
        obs[496] = 0.0  # tech forestry
        obs[497] = 3.0 / 24.0  # tech count -> 3
        obs[499] = 2.33 / 5.0  # avg city level -> 2.33
        obs[500] = 3.0 / 5.0  # max city level -> 3.0
        obs[501] = 0.58  # mean upgrade progress -> 0.58
        obs[502] = 0.75  # max upgrade progress -> 0.75
        obs[503] = 0.33  # upgrade ready frac -> 0.33
        obs[504] = 1.0  # any level up available -> True

        # City slot 0 (PARITY-001)
        obs[505] = 1.0  # present
        obs[506] = 3.0 / 10.0  # x = 3
        obs[507] = 5.0 / 10.0  # y = 5
        obs[508] = 2.0  # level = 2
        obs[509] = 1.0  # population = 1
        obs[510] = 3.0  # population_need = 3
        obs[511] = 4.0  # production = 4
        obs[512] = 2.0  # units = 2
        obs[513] = 3.0  # capacity = 3

        info = {"map_width": 11, "map_height": 11}
        st = visible_state(obs, info)
        self.assertEqual(st["turn"], 6)
        self.assertEqual(st["stars"], 8)
        self.assertEqual(st["spt"], 11)
        self.assertEqual(st["city_count"], 3)
        self.assertTrue(st["tech_organization"])
        self.assertFalse(st["tech_forestry"])
        self.assertEqual(st["tech_count"], 3)
        self.assertAlmostEqual(st["avg_city_level"], 2.33, places=2)
        self.assertAlmostEqual(st["max_city_level"], 3.0, places=1)
        self.assertAlmostEqual(st["mean_upgrade_progress"], 0.58, places=2)
        self.assertAlmostEqual(st["max_upgrade_progress"], 0.75, places=2)
        self.assertAlmostEqual(st["upgrade_ready_frac"], 0.33, places=2)
        self.assertTrue(st["any_level_up_available"])

        self.assertEqual(len(st["owned_cities"]), 1)
        self.assertEqual(st["owned_cities"][0]["x"], 3)
        self.assertEqual(st["owned_cities"][0]["y"], 5)
        self.assertEqual(st["owned_cities"][0]["level"], 2)
        self.assertEqual(st["owned_cities"][0]["population"], 1)
        self.assertEqual(st["owned_cities"][0]["population_need"], 3)
        self.assertEqual(st["owned_cities"][0]["production"], 4)
        self.assertEqual(st["owned_cities"][0]["supported_unit_count"], 2)
        self.assertEqual(st["owned_cities"][0]["unit_capacity"], 3)

    def test_deterministic_unit_numbering(self):
        st = {
            "width": 11,
            "height": 11,
            "unit_ids": np.zeros((121,), dtype=np.int64),
        }
        # Place units at (6, 4) and (6, 6)
        st["unit_ids"][6 * 11 + 4] = 101
        st["unit_ids"][6 * 11 + 6] = 102
        units = extract_visible_units(st)
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]["number"], 1)
        self.assertEqual(units[0]["pos"], (6, 4))
        self.assertEqual(units[1]["number"], 2)
        self.assertEqual(units[1]["pos"], (6, 6))

    def test_direction_arrows_and_ascii_derivation(self):
        src = (5, 5)
        # 8 cardinal and diagonal directions
        cases = [
            ((5, 4), "↑ ", "N "),
            ((5, 6), "↓ ", "S "),
            ((4, 5), "← ", "W "),
            ((6, 5), "→ ", "E "),
            ((4, 4), "↖", "NW"),
            ((6, 4), "↗", "NE"),
            ((4, 6), "↙", "SW"),
            ((6, 6), "↘", "SE"),
        ]
        for dst, expected_unicode, expected_ascii in cases:
            self.assertEqual(move_direction(src, dst, unicode_arrow=True), expected_unicode)
            self.assertEqual(move_direction(src, dst, unicode_arrow=False), expected_ascii)

    def test_movement_grouping_by_source_unit(self):
        st = {
            "width": 11,
            "height": 11,
            "unit_ids": np.zeros((121,), dtype=np.int64),
            "terrain": np.zeros((121,), dtype=np.int16),
            "city_ids": np.zeros((121,), dtype=np.int64),
            "resources": np.full((121,), -1, dtype=np.int16),
        }
        st["unit_ids"][6 * 11 + 4] = 1
        st["unit_ids"][6 * 11 + 6] = 2

        actions = [
            {
                "slot": 0,
                "padded_slot": 0,
                "global_id": 100,
                "type": "MOVE",
                "description": "Move unit (6, 4) -> (5, 3)",
                "src_xy": (6, 4),
                "dst_xy": (5, 3),
                "features": np.zeros((42,), dtype=np.float32),
                "annotations": ["reveal +4", "away from capital"],
            },
            {
                "slot": 1,
                "padded_slot": 1,
                "global_id": 101,
                "type": "MOVE",
                "description": "Move unit (6, 6) -> (5, 6)",
                "src_xy": (6, 6),
                "dst_xy": (5, 6),
                "features": np.zeros((42,), dtype=np.float32),
                "annotations": ["city bounds"],
            },
            {
                "slot": 2,
                "padded_slot": 2,
                "global_id": 0,
                "type": "END_TURN",
                "description": "End turn",
                "features": np.zeros((42,), dtype=np.float32),
                "annotations": [],
            },
        ]

        printed = []
        _print_actions(actions, 0, 30, output=printed.append, state=st, unicode_arrows=True)
        text = "\n".join(printed)

        self.assertIn("MOVES - UNIT 1 at (6, 4)", text)
        self.assertIn("MOVES - UNIT 2 at (6, 6)", text)
        self.assertIn("[  0] ↖ (5, 3)   reveal +4 | away from capital  (gid=100)", text)
        self.assertIn("[  1] ←  (5, 6)   city bounds  (gid=101)", text)
        self.assertIn("[TURN]", text)
        self.assertIn("[  2] End turn  (gid=0)", text)

    def test_visible_map_lines_ansi_and_monochrome(self):
        st = {
            "width": 11,
            "height": 11,
            "terrain": np.full((121,), 7, dtype=np.int16),
            "unit_ids": np.zeros((121,), dtype=np.int64),
            "city_ids": np.zeros((121,), dtype=np.int64),
            "resources": np.full((121,), -1, dtype=np.int16),
        }
        # Visible plain, forest, city, village, animal, fruit
        st["terrain"][6 * 11 + 4] = 6  # Forest
        st["terrain"][6 * 11 + 5] = 5  # City
        st["terrain"][6 * 11 + 6] = 4  # Village
        st["terrain"][5 * 11 + 4] = 0  # Plain
        st["resources"][5 * 11 + 4] = 2  # Animal on plain
        st["terrain"][5 * 11 + 5] = 0  # Plain
        st["resources"][5 * 11 + 5] = 1  # Fruit on plain
        st["terrain"][4 * 11 + 4] = 6  # Forest
        st["resources"][4 * 11 + 4] = 1  # Fruit on forest
        st["terrain"][4 * 11 + 5] = 6  # Forest (empty)
        st["unit_ids"][6 * 11 + 4] = 1  # Unit 1 on forest

        # ANSI mode
        ansi_lines = visible_map_lines(st, use_ansi=True)
        ansi_text = "\n".join(ansi_lines)
        self.assertIn("\033[", ansi_text)
        self.assertIn("Tactical Map (ANSI Color)", ansi_lines[0])
        self.assertIn("Units: [1] Warrior at (6, 4)", ansi_text)
        # Forest terrain cue in ANSI mode uses bold brown/tan text (33m) on green background (42m) with 'T'
        self.assertIn("\033[1;33m\033[42m T \033[0m", ansi_text)

        # Monochrome mode
        mono_lines = visible_map_lines(st, use_ansi=False)
        mono_text = "\n".join(mono_lines)
        self.assertNotIn("\033[", mono_text)
        self.assertIn("Tactical Map (Monochrome Fallback)", mono_lines[0])
        self.assertIn("Units: [1] Warrior at (6, 4)", mono_text)
        self.assertIn("T1 ", mono_text)  # Unit 1 on forest (T = forest)
        self.assertIn(" C ", mono_text)   # City
        self.assertIn(" V ", mono_text)   # Village
        self.assertIn(" A ", mono_text)   # Animal on plain
        self.assertIn(" F ", mono_text)   # Fruit on plain (F = fruit)
        self.assertIn("Tf ", mono_text)  # Fruit on forest
        self.assertIn(" T ", mono_text)   # Empty forest (T = forest)

    def test_policy_visible_actions_fails_on_missing_or_bad_features(self):
        env = FakeEnv()
        # Missing features key
        bad_info = dict(env.info)
        del bad_info["legal_action_features_padded"]
        with self.assertRaisesRegex(RuntimeError, "missing required info field"):
            policy_visible_actions(env, bad_info)

        # Mismatched dimension
        bad_info2 = dict(env.info)
        bad_info2["legal_action_features_padded"] = np.zeros((4, 20), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "feature dim"):
            policy_visible_actions(env, bad_info2)

    def test_official_presentation_source_has_no_privileged_api_references(self):
        assert_information_safety()


class RegistryWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.maps = [
            {
                "filename": f"map_{index:06d}.csv",
                "relative_path": f"human_benchmark/map_{index:06d}.csv",
                "csv_sha256": hashlib.sha256(f"csv-{index}".encode()).hexdigest(),
                "canonical_map_sha256": hashlib.sha256(f"map-{index}".encode()).hexdigest(),
            }
            for index in (1, 2)
        ]
        self.split = {
            "dataset_contract": "phase1-bardur-real-v1",
            "split_seed": 20260813,
            "pool_identities": {"human_benchmark": "pool"},
        }

    def started(self, selected, attempts, participant="human", mode="random_unplayed_first_attempt"):
        return benchmark.make_started_attempt(
            selected,
            attempts,
            participant_kind=participant,
            selection_mode=mode,
            selection_seed=7,
            episode_seed=42,
            split_manifest=self.split,
        )

    def test_abort_completion_replay_and_synthetic_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            first_map = self.maps[0]

            aborted = self.started(first_map, [])
            benchmark.persist_attempt(output, aborted)
            benchmark.finalize_attempt(output, aborted, None, status="aborted")
            attempts = benchmark.load_attempts(output)
            self.assertEqual(benchmark.select_unplayed_map([first_map], attempts, 1), first_map)

            first = self.started(first_map, attempts)
            benchmark.persist_attempt(output, first)
            completed = benchmark.finalize_attempt(output, first, completed_result(20), status="completed")
            self.assertTrue(completed["is_first_completed_attempt"])
            attempts = benchmark.load_attempts(output)
            self.assertIsNone(benchmark.select_unplayed_map([first_map], attempts, 1))

            replay = self.started(first_map, attempts, mode="deliberate_replay")
            benchmark.persist_attempt(output, replay)
            replayed = benchmark.finalize_attempt(output, replay, completed_result(25), status="completed")
            self.assertFalse(replayed["is_first_completed_attempt"])

            synthetic = self.started(self.maps[1], benchmark.load_attempts(output), participant="synthetic_test", mode="synthetic_smoke")
            benchmark.persist_attempt(output, synthetic)
            benchmark.finalize_attempt(output, synthetic, completed_result(99), status="completed")

            summary = benchmark.rebuild_summary(output, self.maps)
            self.assertEqual(summary["completed_first_attempt_maps"], 1)
            self.assertEqual(summary["remaining_first_attempt_maps"], 1)
            self.assertEqual(summary["first_attempt"]["mean_spt"], 20)
            self.assertEqual(summary["latest_attempt"]["mean_spt"], 25)
            self.assertEqual(summary["best_attempt"]["mean_spt"], 25)
            self.assertEqual(len(list((output / "attempts").glob("*.json"))), 4)

    def test_completed_attempt_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            payload = self.started(self.maps[0], [])
            benchmark.persist_attempt(output, payload)
            completed = benchmark.finalize_attempt(output, payload, completed_result(), status="completed")
            with self.assertRaisesRegex(benchmark.BenchmarkError, "refusing to overwrite"):
                benchmark.persist_attempt(output, completed)


if __name__ == "__main__":
    unittest.main()
