#!/usr/bin/env python3
"""Py4J Metadata Equivalence Test Suite.

Audits whether observation-derived metadata strictly matches direct Java Py4J bridge
calls across hundreds/thousands of states on both single-tribe and multi-tribe maps:
  1. bool(obs["gameIsOver"]) == bool(self._env.isDone())
  2. int(obs["tick"]) == int(self._env.getTick())
  3. int(obs["activeTribeID"]) == int(self._env.getActiveTribeID())
  4. reconstructed scores == list(self._env.getScores())
"""

import glob
import json
import os
import random
import sys
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pol_env.Tribes.py.gym_env import make_default_env


class TestPy4jMetadataEquivalence(unittest.TestCase):
    """Test strict equivalence between observation payload and direct Py4J getters."""

    def setUp(self):
        self.env = make_default_env()

    def tearDown(self):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _audit_state(self, obs, counts):
        counts["states_tested"] += 1

        # Direct bridge calls
        ref_done = bool(self.env._env.isDone())
        ref_tick = int(self.env._env.getTick())
        ref_active_tribe_id = int(self.env._env.getActiveTribeID())
        ref_scores = list(self.env._env.getScores())

        # Observation-derived values (fail-closed)
        try:
            obs_done = bool(obs["gameIsOver"])
            obs_tick = int(obs["tick"])
            obs_active_tribe_id = int(obs["activeTribeID"])
            tribes_data = obs["tribes"]
            if not isinstance(tribes_data, dict):
                raise TypeError(f"obs['tribes'] must be dict, got {type(tribes_data)}")
            sorted_tribes = sorted(
                tribes_data.items(),
                key=lambda item: int(item[0]),
            )
            obs_scores = [int(v["score"]) for _, v in sorted_tribes]
        except (KeyError, TypeError, ValueError) as exc:
            self.fail(f"Observation metadata contract violation: {exc}")

        # Equivalence checks
        if obs_done != ref_done:
            counts["mismatches_done"] += 1
            self.assertEqual(obs_done, ref_done, f"isDone mismatch: obs={obs_done} vs java={ref_done}")

        if obs_tick != ref_tick:
            counts["mismatches_tick"] += 1
            self.assertEqual(obs_tick, ref_tick, f"getTick mismatch: obs={obs_tick} vs java={ref_tick}")

        if obs_active_tribe_id != ref_active_tribe_id:
            counts["mismatches_active_tribe"] += 1
            self.assertEqual(
                obs_active_tribe_id,
                ref_active_tribe_id,
                f"getActiveTribeID mismatch: obs={obs_active_tribe_id} vs java={ref_active_tribe_id}",
            )

        if obs_scores != ref_scores:
            counts["mismatches_scores"] += 1
            self.assertEqual(
                obs_scores,
                ref_scores,
                f"getScores mismatch: obs={obs_scores} vs java={ref_scores}",
            )

    def test_single_tribe_training_maps_metadata_equivalence(self):
        """Audit observation metadata on genuine Phase 1 single-tribe training maps."""
        train_maps = sorted(
            glob.glob(
                os.path.join(REPO_ROOT, "pol_env", "Tribes", "levels", "phase1_pool_bardur_real", "train", "*.csv")
            )
        )
        self.assertTrue(len(train_maps) > 0, "No training maps found")

        counts = {
            "states_tested": 0,
            "mismatches_done": 0,
            "mismatches_tick": 0,
            "mismatches_active_tribe": 0,
            "mismatches_scores": 0,
        }

        # Sample 30 training maps, running up to 30 steps per map with varied actions
        rng = random.Random(42)
        sample_maps = train_maps[:30]
        for map_file in sample_maps:
            obs = self.env.reset(map_file, seed=42)
            self._audit_state(obs, counts)

            for _ in range(30):
                actions = self.env.list_actions()
                if not actions:
                    break
                # Choose random action
                action = rng.choice(actions)
                action_idx = int(action.get("idx", 0))
                obs, reward, done, info = self.env.step(action_idx)
                self._audit_state(obs, counts)
                if done:
                    break

        print(f"\n[SINGLE-TRIBE AUDIT] States tested: {counts['states_tested']}")
        print(f"  done mismatches:         {counts['mismatches_done']}")
        print(f"  tick mismatches:         {counts['mismatches_tick']}")
        print(f"  active_tribe mismatches: {counts['mismatches_active_tribe']}")
        print(f"  scores mismatches:       {counts['mismatches_scores']}")

        self.assertGreater(counts["states_tested"], 300)
        self.assertEqual(counts["mismatches_done"], 0)
        self.assertEqual(counts["mismatches_tick"], 0)
        self.assertEqual(counts["mismatches_active_tribe"], 0)
        self.assertEqual(counts["mismatches_scores"], 0)

    def test_multi_tribe_maps_metadata_and_score_ordering_equivalence(self):
        """Audit observation metadata and tribe score array ordering on synthetic multi-tribe maps."""
        # Create a 2-tribe map (Bardur=1, Imperius=0) and a 3-tribe map
        # Format: C:<tribeKey> represents city for tribe.
        # Types.TRIBE enum: IMPERIUS=0, BARDUR=1, OUMAJI=2, KICKOO=3, etc.
        # Lines: comma-separated cells
        lines_2tribe = [
            "c:0,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,c:1",
        ]
        lines_3tribe = [
            "c:0,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,c:2,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,.:",
            ".:,.:,.:,.:,.:,.:,.:,.:,.:,.:,c:1",
        ]

        counts = {
            "states_tested": 0,
            "mismatches_done": 0,
            "mismatches_tick": 0,
            "mismatches_active_tribe": 0,
            "mismatches_scores": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path_2t = os.path.join(tmpdir, "map_2tribe.csv")
            with open(path_2t, "w", encoding="utf-8") as f:
                f.write("\n".join(lines_2tribe) + "\n")

            path_3t = os.path.join(tmpdir, "map_3tribe.csv")
            with open(path_3t, "w", encoding="utf-8") as f:
                f.write("\n".join(lines_3tribe) + "\n")

            rng = random.Random(123)
            for map_file, expected_tribes in [(path_2t, 2), (path_3t, 3)]:
                obs = self.env.reset(map_file, seed=42)
                self.assertEqual(len(obs["tribes"]), expected_tribes)
                self.assertEqual(len(self.env._env.getScores()), expected_tribes)
                self._audit_state(obs, counts)

                for _ in range(40):
                    actions = self.env.list_actions()
                    if not actions:
                        break
                    action = rng.choice(actions)
                    action_idx = int(action.get("idx", 0))
                    obs, reward, done, info = self.env.step(action_idx)
                    self._audit_state(obs, counts)
                    if done:
                        break

        print(f"\n[MULTI-TRIBE AUDIT] States tested: {counts['states_tested']}")
        print(f"  done mismatches:         {counts['mismatches_done']}")
        print(f"  tick mismatches:         {counts['mismatches_tick']}")
        print(f"  active_tribe mismatches: {counts['mismatches_active_tribe']}")
        print(f"  scores mismatches:       {counts['mismatches_scores']}")

        self.assertGreater(counts["states_tested"], 40)
        self.assertEqual(counts["mismatches_done"], 0)
        self.assertEqual(counts["mismatches_tick"], 0)
        self.assertEqual(counts["mismatches_active_tribe"], 0)
        self.assertEqual(counts["mismatches_scores"], 0)


if __name__ == "__main__":
    unittest.main()
