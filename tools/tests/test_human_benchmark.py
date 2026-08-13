import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pol_env.Tribes.py.register_env import GlobalActionCatalog
from tools import human_benchmark as benchmark
from tools.human_policy_interface import EpisodeResult, run_policy_visible_episode
from tools.validate_human_benchmark_parity import assert_information_safety


class FakeEnv:
    MAX_TURNS = 10
    MAX_LEGAL_ACTIONS_DEFAULT = 4
    ALLOWED_ACTION_TYPES = {"END_TURN"}

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
        self.obs = np.zeros((505,), dtype=np.float32)
        self.obs[:121] = 7
        self.obs[121:363] = -1
        self.info = {
            "map_width": 11,
            "map_height": 11,
            "observation_dim": 505,
            "global_action_space_n": self._catalog.total_size,
            "max_legal_actions": 4,
            "catalog_version": "flat-v1",
            "action_catalog_fingerprint": "catalog",
            "canonicalizer_version": "flat-v1-structured",
            "legal_action_feature_version": "features",
            "legal_action_feature_dim": 42,
            "info_mode": "fast",
            "legal_global_ids_padded": np.array([0, 0, 0, 0]),
            "legal_action_valid_mask": np.array([True, False, False, False]),
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
