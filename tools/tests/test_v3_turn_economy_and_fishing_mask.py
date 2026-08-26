import os
import sys
import unittest
import copy
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"
os.environ["POLYVISION_INFO_MODE"] = "debug"

from pol_env.Tribes.py.register_env import TribesGymWrapper, GlobalActionCatalog
from pol_env.Tribes.py.environment_contract import (
    PHASE1_ENVIRONMENT_VERSION,
    environment_compatibility_metadata,
    validate_checkpoint_compatibility,
    CheckpointCompatibilityError,
)
from tools.human_policy_interface import policy_visible_actions


class Phase1V3TurnEconomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map_path = os.path.join(REPO_ROOT, "pol_env", "Tribes", "levels", "phase1_pool_bardur_real", "train", "map_000001.csv")

    def setUp(self):
        os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"
        os.environ["POLYVISION_LEVEL_POOL_GLOB"] = self.map_path.replace("\\", "/")
        os.environ["POLYVISION_LEVEL_SELECTION_MODE"] = "fixed"
        self.env = TribesGymWrapper()

    def tearDown(self):
        self.env.close()

    def test_environment_version_and_handoff_state(self):
        obs, info = self.env.reset(seed=42)
        self.assertEqual(self.env.PHASE1_ENVIRONMENT_VERSION, "v5_human_information_parity")
        self.assertEqual(info.get("phase1_environment_version"), "v5_human_information_parity")
        self.assertEqual(info.get("phase1_opening_version"), "v2_guaranteed_two_unit")
        self.assertEqual(info.get("turn_count"), 2)
        self.assertEqual(info.get("stars"), 7)
        self.assertEqual(float(info.get("spt")), 4.0)
        self.assertEqual(int(info.get("city_count")), 1)
        raw_obs = self.env.tribes_env._last_obs
        self.assertEqual(int(raw_obs.get("tick")), 2)
        self.assertFalse(self.env.tribes_env._env.isDone())
        t0 = raw_obs.get("tribes", {}).get("0", {})
        self.assertEqual(int(t0.get("winner")), 2)

    def test_no_spending_star_accumulation_and_tick_progression(self):
        obs, info = self.env.reset(seed=42)
        initial_stars = int(info["stars"])
        self.assertEqual(initial_stars, 7)

        current_stars = initial_stars
        end_turn_global = self.env._catalog.id_end_turn()

        for expected_turn in range(3, 7):
            valid_slots = self.env._current_legal_id_to_raw_index
            if end_turn_global in valid_slots:
                chosen = end_turn_global
            else:
                legal_acts = self.env._current_legal_actions
                chosen = None
                for gid, raw_idx in valid_slots.items():
                    act_type = str(legal_acts[raw_idx].get("type", "")).upper()
                    if act_type not in ("RESEARCH_TECH", "BUILD", "TRAIN", "SPAWN", "RESOURCE_GATHERING"):
                        chosen = gid
                        break
                if chosen is None:
                    chosen = list(valid_slots.keys())[0]

            raw_obs_before = self.env.tribes_env._last_obs
            tick_before = int(raw_obs_before.get("tick"))

            obs, reward, term, trunc, step_info = self.env.step(chosen)
            raw_obs_after = self.env.tribes_env._last_obs

            if chosen == end_turn_global:
                tick_after = int(raw_obs_after.get("tick"))
                self.assertEqual(tick_after, tick_before + 1)
                spt_after = float(self.env._compute_bardur_spt(raw_obs_after))
                stars_after = int(self.env._get_bardur_stars(raw_obs_after))
                self.assertEqual(stars_after, current_stars + int(spt_after))
                current_stars = stars_after
                self.assertFalse(term)
                t0 = raw_obs_after.get("tribes", {}).get("0", {})
                self.assertEqual(int(t0.get("winner")), 2)

    def test_spending_plus_income(self):
        obs = self.env.tribes_env.reset(self.map_path, 42)
        raw_env = self.env.tribes_env._env
        self.assertEqual(raw_env.getTick(), 0)
        acts = self.env.tribes_env.list_actions()
        end_idx = next(i for i, a in enumerate(acts) if a.get("type") == "END_TURN")
        self.env.tribes_env.step(end_idx)
        self.assertEqual(raw_env.getTick(), 1)
        obs1 = self.env.tribes_env.get_observation()
        t0_stars = int(obs1.get("tribes", {}).get("0", {}).get("star"))
        self.assertEqual(t0_stars, 7)


class Phase1DrylandsFishingMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drylands_map = os.path.join(REPO_ROOT, "pol_env", "Tribes", "levels", "phase1_pool_bardur_real", "train", "map_000001.csv")

    def test_fishing_masked_on_drylands(self):
        os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"
        os.environ["POLYVISION_LEVEL_POOL_GLOB"] = self.drylands_map.replace("\\", "/")
        os.environ["POLYVISION_LEVEL_SELECTION_MODE"] = "fixed"
        os.environ["POLYVISION_MAP_TYPE"] = "DRYLANDS"

        env = TribesGymWrapper(map_type="DRYLANDS")
        try:
            obs, info = env.reset(seed=42)
            raw_obs = env.tribes_env._last_obs
            self.assertEqual(env._current_map_type, "DRYLANDS")

            raw_actions = env._current_legal_actions
            raw_has_fishing = any(
                str(a.get("type", "")).upper() == "RESEARCH_TECH"
                and "FISHING" in str(a.get("repr", "")).upper()
                for a in raw_actions
            )
            self.assertTrue(raw_has_fishing, "Expected raw Java actions to include FISHING")

            catalog = env._catalog
            fishing_gid = catalog.id_research("FISHING")
            self.assertIsNotNone(fishing_gid)

            valid_gids = set(env._current_legal_id_to_raw_index.keys())
            self.assertNotIn(fishing_gid, valid_gids)

            filtered_indices = env._filter_allowed_raw_indices(raw_actions, raw_obs)
            filtered_fishing = any(
                str(raw_actions[i].get("type", "")).upper() == "RESEARCH_TECH"
                and "FISHING" in str(raw_actions[i].get("repr", "")).upper()
                for i in filtered_indices
            )
            self.assertFalse(filtered_fishing, "Fishing must not pass _filter_allowed_raw_indices on DRYLANDS")

            mask = info.get("action_mask", env._current_action_mask)
            if mask is not None:
                self.assertEqual(mask[fishing_gid], 0)

            valid_slots = info["legal_action_valid_mask"]
            padded_gids = info["legal_global_ids_padded"]
            active_gids = set(padded_gids[valid_slots])
            self.assertNotIn(fishing_gid, active_gids)

            human_actions = policy_visible_actions(env, info)
            human_action_reprs = [a.get("repr", "") for a in human_actions]
            self.assertFalse(any("FISHING" in r for r in human_action_reprs))
        finally:
            env.close()

    def test_fishing_allowed_on_non_drylands(self):
        os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"
        os.environ["POLYVISION_LEVEL_POOL_GLOB"] = self.drylands_map.replace("\\", "/")
        os.environ["POLYVISION_LEVEL_SELECTION_MODE"] = "fixed"

        non_drylands_profiles = ["LAKES", "CONTINENTS", "ARCHIPELAGO", "WATERWORLD", "PANGEA"]
        for profile in non_drylands_profiles:
            with self.subTest(map_type=profile):
                env = TribesGymWrapper(map_type=profile)
                try:
                    obs, info = env.reset(seed=42)
                    raw_obs = env.tribes_env._last_obs
                    self.assertEqual(env._current_map_type, profile)

                    catalog = env._catalog
                    fishing_gid = catalog.id_research("FISHING")
                    self.assertIsNotNone(fishing_gid)

                    raw_actions = env._current_legal_actions
                    fishing_raw_idx = next(
                        (i for i, a in enumerate(raw_actions)
                         if str(a.get("type", "")).upper() == "RESEARCH_TECH" and "FISHING" in str(a.get("repr", "")).upper()),
                        None
                    )
                    self.assertIsNotNone(fishing_raw_idx, f"Raw actions must contain fishing on {profile}")

                    non_village_obs = copy.deepcopy(raw_obs)
                    t0 = non_village_obs.setdefault("tribes", {}).setdefault("0", {})
                    t0["citiesID"] = [0, 1]
                    allowed = env._filter_allowed_raw_indices(raw_actions, non_village_obs)
                    self.assertIn(fishing_raw_idx, allowed, f"Fishing must be allowed on {profile}")
                finally:
                    env.close()

    def test_global_action_catalog_integrity(self):
        env = TribesGymWrapper()
        try:
            env.reset(seed=42)
            catalog = env._catalog
            self.assertEqual(catalog.total_size, 63913)
            self.assertIn("FISHING", catalog.tech_types)
            fishing_id = catalog.id_research("FISHING")
            self.assertIsNotNone(fishing_id)
            self.assertEqual(catalog.fingerprint(), "c849a4abf7b0bee073ccc56b63ae65917ea30e77068ad648c472130693dfe6e4")
        finally:
            env.close()


class Phase1V3CompatibilityTests(unittest.TestCase):
    def test_v2_checkpoint_rejected_against_v3_environment(self):
        wrapper = TribesGymWrapper()
        try:
            env_meta = environment_compatibility_metadata(wrapper, actor_mode="legal_features", max_legal_actions=256)
            v2_meta = dict(env_meta)
            del v2_meta["phase1_environment_version"]
            with self.assertRaises(CheckpointCompatibilityError):
                validate_checkpoint_compatibility(v2_meta, env_meta)

            v2_meta_explicit = dict(env_meta)
            v2_meta_explicit["phase1_environment_version"] = "v2_broken_turn_economy"
            with self.assertRaises(CheckpointCompatibilityError):
                validate_checkpoint_compatibility(v2_meta_explicit, env_meta)

            v3_meta = dict(env_meta)
            validate_checkpoint_compatibility(v3_meta, env_meta)
        finally:
            wrapper.close()


if __name__ == "__main__":
    unittest.main()
