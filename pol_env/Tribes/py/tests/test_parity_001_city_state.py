import unittest
from types import SimpleNamespace
from unittest import mock
import numpy as np

from pol_env.Tribes.py.environment_contract import (
    CheckpointCompatibilityError,
    CITY_BLOCK_DIM,
    CITY_SLOT_FEATURE_DIM,
    CITY_SLOT_FEATURE_NAMES,
    MAX_OWNED_CITIES,
    ObservationContractError,
    ObservationLayout,
    OwnedCityState,
    PHASE1_ENVIRONMENT_VERSION,
    decode_owned_city_slots,
    encode_owned_city_slots,
    extract_owned_cities,
    observation_layout,
    validate_checkpoint_compatibility,
)
from pol_env.Tribes.py.register_env import TribesGymWrapper


class TestParity001CityState(unittest.TestCase):
    def setUp(self):
        self.width = 11
        self.height = 11

    # --- Test A: Exact City State Test ---
    def test_a_exact_city_state_roundtrip(self):
        """Construct a state with multiple owned cities and verify exact roundtrip correspondence."""
        pov_obs = {
            "city": {
                "101": {
                    "x": 3,
                    "y": 5,
                    "level": 2,
                    "population": 2,
                    "population_need": 3,
                    "production": 3,
                    "tribeID": 0,
                    "isCapital": True,
                    "units": [201, 202],
                },
                "102": {
                    "x": 7,
                    "y": 1,
                    "level": 1,
                    "population": 0,
                    "population_need": 2,
                    "production": 2,
                    "tribeID": 0,
                    "isCapital": False,
                    "units": [203],
                },
            }
        }

        cities = extract_owned_cities(pov_obs, tribe_id=0)
        self.assertEqual(len(cities), 2)
        # Expected spatial ordering: (3, 5) then (7, 1)
        self.assertEqual((cities[0].x, cities[0].y), (3, 5))
        self.assertEqual(cities[0].level, 2)
        self.assertEqual(cities[0].population, 2)
        self.assertEqual(cities[0].population_need, 3)
        self.assertEqual(cities[0].production, 3)
        self.assertEqual(cities[0].supported_unit_count, 2)
        self.assertEqual(cities[0].unit_capacity, 3)  # level + 1

        self.assertEqual((cities[1].x, cities[1].y), (7, 1))
        self.assertEqual(cities[1].level, 1)
        self.assertEqual(cities[1].population, 0)
        self.assertEqual(cities[1].population_need, 2)
        self.assertEqual(cities[1].production, 2)
        self.assertEqual(cities[1].supported_unit_count, 1)
        self.assertEqual(cities[1].unit_capacity, 2)  # level + 1

        encoded = encode_owned_city_slots(cities, width=self.width, height=self.height)
        self.assertEqual(len(encoded), MAX_OWNED_CITIES * CITY_SLOT_FEATURE_DIM)

        decoded = decode_owned_city_slots(encoded, width=self.width, height=self.height)
        self.assertEqual(len(decoded), 2)
        for original, dec in zip(cities, decoded):
            self.assertEqual(dec["x"], original.x)
            self.assertEqual(dec["y"], original.y)
            self.assertEqual(dec["level"], original.level)
            self.assertEqual(dec["population"], original.population)
            self.assertEqual(dec["population_need"], original.population_need)
            self.assertEqual(dec["production"], original.production)
            self.assertEqual(dec["supported_unit_count"], original.supported_unit_count)
            self.assertEqual(dec["unit_capacity"], original.unit_capacity)

    # --- Test B: Multi-City Distinguishability Test ---
    def test_b_multi_city_distinguishability(self):
        """Two states with identical global aggregates but different city assignments must yield different city tensors."""
        # State A: City 1 = Lv1, pop 1/2; City 2 = Lv3, pop 2/4
        state_a = {
            "city": {
                "1": {"x": 2, "y": 2, "level": 1, "population": 1, "population_need": 2, "production": 2, "tribeID": 0, "units": [10]},
                "2": {"x": 8, "y": 8, "level": 3, "population": 2, "population_need": 4, "production": 4, "tribeID": 0, "units": [11, 12, 13]},
            }
        }
        # State B: City 1 = Lv2, pop 2/3; City 2 = Lv2, pop 1/3
        state_b = {
            "city": {
                "1": {"x": 2, "y": 2, "level": 2, "population": 2, "population_need": 3, "production": 3, "tribeID": 0, "units": [10, 11]},
                "2": {"x": 8, "y": 8, "level": 2, "population": 1, "population_need": 3, "production": 3, "tribeID": 0, "units": [12, 13]},
            }
        }

        # Both have:
        # avg level = (1+3)/2 = 2.0 vs (2+2)/2 = 2.0
        # total pop = 3, total production = 6, total units = 4
        cities_a = extract_owned_cities(state_a, tribe_id=0)
        cities_b = extract_owned_cities(state_b, tribe_id=0)

        tensor_a = encode_owned_city_slots(cities_a, width=self.width, height=self.height)
        tensor_b = encode_owned_city_slots(cities_b, width=self.width, height=self.height)

        self.assertFalse(np.array_equal(tensor_a, tensor_b), "Distinct city states must not produce identical tensors")

    # --- Test C: Deterministic Slot Ordering Test ---
    def test_c_deterministic_slot_ordering(self):
        """Permuting dictionary key insertion order produces identical city-slot tensor sorted by (x, y)."""
        c1 = {"x": 1, "y": 8, "level": 1, "population": 0, "population_need": 2, "production": 1, "tribeID": 0, "units": []}
        c2 = {"x": 4, "y": 2, "level": 2, "population": 1, "population_need": 3, "production": 3, "tribeID": 0, "units": [1]}
        c3 = {"x": 4, "y": 7, "level": 1, "population": 1, "population_need": 2, "production": 2, "tribeID": 0, "units": [2]}

        # Order 1
        obs_1 = {"city": {"actor_c3": c3, "actor_c1": c1, "actor_c2": c2}}
        # Order 2
        obs_2 = {"city": {"actor_c2": c2, "actor_c3": c3, "actor_c1": c1}}

        tensor_1 = encode_owned_city_slots(extract_owned_cities(obs_1, tribe_id=0), width=self.width, height=self.height)
        tensor_2 = encode_owned_city_slots(extract_owned_cities(obs_2, tribe_id=0), width=self.width, height=self.height)

        self.assertEqual(tensor_1, tensor_2)
        decoded = decode_owned_city_slots(tensor_1, width=self.width, height=self.height)
        self.assertEqual([(c["x"], c["y"]) for c in decoded], [(1, 8), (4, 2), (4, 7)])

    # --- Test D: Actor-ID Independence Test ---
    def test_d_actor_id_independence(self):
        """Semantically identical cities with different Java actor IDs must produce identical city-slot tensors."""
        c1 = {"x": 3, "y": 3, "level": 2, "population": 1, "population_need": 3, "production": 3, "tribeID": 0, "units": [100]}
        obs_1 = {"city": {"9999": c1}}
        obs_2 = {"city": {"12": c1}}

        tensor_1 = encode_owned_city_slots(extract_owned_cities(obs_1, tribe_id=0), width=self.width, height=self.height)
        tensor_2 = encode_owned_city_slots(extract_owned_cities(obs_2, tribe_id=0), width=self.width, height=self.height)

        self.assertEqual(tensor_1, tensor_2)
        # Ensure raw actor ID values (9999, 12, 100) do NOT appear in the tensor
        self.assertNotIn(9999.0, tensor_1)
        self.assertNotIn(100.0, tensor_1)

    # --- Test E: Empty Slot Padding Test ---
    def test_e_empty_slot_padding(self):
        """For K < MAX_OWNED_CITIES, populated slots have present=1.0 and empty slots have present=0.0 and zero padding."""
        obs = {
            "city": {
                "1": {"x": 5, "y": 5, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": []}
            }
        }
        tensor = np.asarray(
            encode_owned_city_slots(extract_owned_cities(obs, tribe_id=0), width=self.width, height=self.height),
            dtype=np.float32,
        )

        # Slot 0 is populated
        self.assertEqual(tensor[0], 1.0)
        # Slots 1..8 must be exactly 0.0
        for slot in range(1, MAX_OWNED_CITIES):
            slot_slice = tensor[slot * CITY_SLOT_FEATURE_DIM : (slot + 1) * CITY_SLOT_FEATURE_DIM]
            self.assertTrue(np.all(slot_slice == 0.0), f"Slot {slot} is not zero-padded: {slot_slice}")

    # --- Test F: Overflow Test & Fallback Prevention ---
    def test_f_overflow_raises_contract_error_and_no_fallback(self):
        """More than MAX_OWNED_CITIES raises ObservationContractError and wrapper re-raises without fallback."""
        cities = [
            OwnedCityState(x=i % 10, y=i // 10, level=1, population=0, population_need=2, production=1, supported_unit_count=0, unit_capacity=2, is_capital=(i == 0))
            for i in range(MAX_OWNED_CITIES + 1)
        ]
        with self.assertRaises(ObservationContractError):
            encode_owned_city_slots(cities, width=self.width, height=self.height)

        # Test that TribesGymWrapper re-raises ObservationContractError during initialization
        wrapper = object.__new__(TribesGymWrapper)
        wrapper._turn_count = 0
        wrapper.MAX_TURNS = 10
        wrapper._researched_techs_t10 = set()
        wrapper._controlled_tribe_id = 0
        overflow_obs = {
            "board": {"terrain": [[0] * 11 for _ in range(11)]},
            "city": {
                str(i): {"x": i % 10, "y": i // 10, "level": 1, "population": 0, "population_need": 2, "production": 1, "tribeID": 0, "units": [], "isCapital": (i == 0)}
                for i in range(MAX_OWNED_CITIES + 1)
            },
        }
        with self.assertRaises(ObservationContractError):
            wrapper._dict_to_array(overflow_obs)

    # --- Test G: Fog / POV Provenance Test ---
    def test_g_pov_provenance_and_enemy_filtering(self):
        """Enemy cities (tribeID != 0) are excluded from owned city state."""
        obs = {
            "city": {
                "1": {"x": 2, "y": 2, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [], "isCapital": True},
                "2": {"x": 9, "y": 9, "level": 3, "population": 1, "population_need": 4, "production": 5, "tribeID": 1, "units": [50, 51], "isCapital": True},
            }
        }
        owned = extract_owned_cities(obs, tribe_id=0)
        self.assertEqual(len(owned), 1)
        self.assertEqual((owned[0].x, owned[0].y), (2, 2))
        self.assertTrue(owned[0].is_capital)

    # --- Test H: Observation Dimension Contract Test ---
    def test_h_observation_dimension_contract(self):
        """Assert exact 5335 observation dimensions for 11x11 Phase-1 contract."""
        layout = observation_layout(11, 11)
        self.assertEqual(layout.terrain_start, 0)
        self.assertEqual(layout.terrain_end, 121)
        self.assertEqual(layout.unit_types_start, 121)
        self.assertEqual(layout.unit_types_end, 1573)
        self.assertEqual(layout.city_territory_start, 1573)
        self.assertEqual(layout.city_territory_end, 2662)
        self.assertEqual(layout.road_start, 2662)
        self.assertEqual(layout.road_end, 2783)
        self.assertEqual(layout.buildings_start, 2783)
        self.assertEqual(layout.buildings_end, 5082)
        self.assertEqual(layout.resource_start, 5082)
        self.assertEqual(layout.resource_end, 5203)
        self.assertEqual(layout.legacy_scalar_start, 5203)
        self.assertEqual(layout.legacy_scalar_end, 5209)
        self.assertEqual(layout.economy_scalar_start, 5209)
        self.assertEqual(layout.economy_scalar_end, 5221)
        self.assertEqual(layout.tech_vector_start, 5221)
        self.assertEqual(layout.tech_vector_end, 5245)
        self.assertEqual(layout.city_block_start, 5245)
        self.assertEqual(layout.city_block_end, 5335)
        self.assertEqual(layout.expected_obs_dim, 5335)
        self.assertEqual(CITY_BLOCK_DIM, 90)
        self.assertEqual(len(CITY_SLOT_FEATURE_NAMES), 10)

    # --- Test I: Checkpoint Incompatibility Test ---
    def test_i_historical_505_checkpoint_is_rejected(self):
        """A historical 505-dim checkpoint metadata must fail validation under v5."""
        historical_meta = {
            "map_width": 11,
            "map_height": 11,
            "observation_dim": 505,
            "action_space_n": 63913,
            "action_catalog_fingerprint": "catalog-11x11",
            "actor_mode": "legal_features",
            "legal_action_feature_version": "v1_3_move_focus_plus_semantic_econ",
            "legal_action_feature_dim": 42,
            "catalog_version": "flat-v1",
            "canonicalizer_version": "flat-v1-structured",
            "phase1_opening_version": "v2_guaranteed_two_unit",
            "phase1_environment_version": "v3_corrected_turn_economy",
            "max_legal_actions": 256,
        }

        v5_env_meta = dict(historical_meta)
        v5_env_meta["observation_dim"] = 5335
        v5_env_meta["phase1_environment_version"] = "v5_human_information_parity"
        v5_env_meta["legal_action_feature_version"] = "v1_4_parity_spatial_and_cost"
        v5_env_meta["legal_action_feature_dim"] = 47

        with self.assertRaises(CheckpointCompatibilityError) as ctx:
            validate_checkpoint_compatibility(historical_meta, v5_env_meta)
        err_msg = str(ctx.exception)
        self.assertIn("observation_dim", err_msg)
        self.assertIn("phase1_environment_version", err_msg)

    # --- Test J: Capital Identity Inclusion in PARITY-002 ---
    def test_j_capital_identity_inclusion(self):
        """Assert OwnedCityState and decoded slots expose is_capital (STATE-MAP-003 is resolved)."""
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(OwnedCityState)]
        self.assertIn("is_capital", field_names, "OwnedCityState must contain is_capital")
        self.assertEqual(len(field_names), 9)

        # Test decoded slots contain is_capital
        encoded = encode_owned_city_slots(
            [OwnedCityState(x=3, y=4, level=1, population=0, population_need=2, production=2, supported_unit_count=0, unit_capacity=2, is_capital=True)],
            width=self.width,
            height=self.height,
        )
        decoded = decode_owned_city_slots(encoded, width=self.width, height=self.height)
        self.assertIn("is_capital", decoded[0], "Decoded slot dictionary must expose is_capital")
        self.assertTrue(decoded[0]["is_capital"])

    # --- Test K: Decision-Time Human Visibility ---
    def test_k_decision_time_human_visibility(self):
        """Verify human presentation displays exact unnormalized city primitives at every decision state."""
        from tools.human_policy_interface import run_policy_visible_episode
        from tools.tests.test_human_benchmark import FakeEnv

        env = FakeEnv()
        # Put 2 cities in the FakeEnv observation:
        # Slot 0: (3, 4), Lv 1, pop 0/2, prod 2, units 1/2, is_capital=True
        env.obs[5245] = 1.0
        env.obs[5246] = 3.0 / 10.0
        env.obs[5247] = 4.0 / 10.0
        env.obs[5248] = 1.0
        env.obs[5249] = 0.0
        env.obs[5250] = 2.0
        env.obs[5251] = 2.0
        env.obs[5252] = 1.0
        env.obs[5253] = 2.0
        env.obs[5254] = 1.0

        # Slot 1: (6, 8), Lv 2, pop 2/3, prod 3, units 2/3, is_capital=False
        env.obs[5255] = 1.0
        env.obs[5256] = 6.0 / 10.0
        env.obs[5257] = 8.0 / 10.0
        env.obs[5258] = 2.0
        env.obs[5259] = 2.0
        env.obs[5260] = 3.0
        env.obs[5261] = 3.0
        env.obs[5262] = 2.0
        env.obs[5263] = 3.0
        env.obs[5264] = 0.0

        captured_lines: list[str] = []
        observed_states: list[dict] = []

        def state_recorder(_env, _obs, info):
            from tools.human_policy_interface import visible_state
            observed_states.append(visible_state(_obs, info))

        run_policy_visible_episode(
            env,
            episode_seed=42,
            official=False,
            selector=lambda actions, _obs, _info, _step: actions[0]["global_id"],
            output_fn=captured_lines.append,
            state_callback=state_recorder,
        )

        # Confirm state_callback observed exact owned_cities at decision time
        self.assertTrue(len(observed_states) > 0)
        first_state = observed_states[0]
        self.assertIn("owned_cities", first_state)
        self.assertEqual(len(first_state["owned_cities"]), 2)
        self.assertEqual(first_state["owned_cities"][0]["x"], 3)
        self.assertEqual(first_state["owned_cities"][0]["y"], 4)
        self.assertEqual(first_state["owned_cities"][0]["level"], 1)
        self.assertEqual(first_state["owned_cities"][0]["population"], 0)
        self.assertEqual(first_state["owned_cities"][0]["population_need"], 2)
        self.assertEqual(first_state["owned_cities"][0]["production"], 2)
        self.assertEqual(first_state["owned_cities"][0]["supported_unit_count"], 1)
        self.assertEqual(first_state["owned_cities"][0]["unit_capacity"], 2)
        self.assertTrue(first_state["owned_cities"][0]["is_capital"])

        # Confirm output_fn printed the human-readable owned cities block
        full_text = "\n".join(captured_lines)
        self.assertIn("Owned Cities (2/9):", full_text)
        self.assertIn("City #1 @ (3, 4): Level 1 | Pop 0/2 | Production +2 SPT | Units 1/2 | Capital", full_text)
        self.assertIn("City #2 @ (6, 8): Level 2 | Pop 2/3 | Production +3 SPT | Units 2/3", full_text)

    # --- Test L: Live End-to-End City Parity Invariant ---
    def test_l_live_end_to_end_city_parity_invariant(self):
        """Verify canonical == decoded_policy == human_visible parity helper function."""
        from tools.validate_human_benchmark_parity import assert_live_city_parity

        canonical = [
            OwnedCityState(x=2, y=3, level=1, population=1, population_need=2, production=2, supported_unit_count=1, unit_capacity=2, is_capital=True),
            OwnedCityState(x=5, y=5, level=2, population=0, population_need=3, production=3, supported_unit_count=0, unit_capacity=3, is_capital=False),
        ]
        decoded = [
            {"slot": 0, "x": 2, "y": 3, "level": 1, "population": 1, "population_need": 2, "production": 2, "supported_unit_count": 1, "unit_capacity": 2, "is_capital": True},
            {"slot": 1, "x": 5, "y": 5, "level": 2, "population": 0, "population_need": 3, "production": 3, "supported_unit_count": 0, "unit_capacity": 2, "is_capital": False},
        ]
        human = list(decoded)

        # Perfect match must pass (with equal unit_capacity)
        decoded[1]["unit_capacity"] = 3
        human[1]["unit_capacity"] = 3
        assert_live_city_parity(canonical, decoded, human)

        # Discrepancy must raise RuntimeError
        decoded_bad = list(decoded)
        decoded_bad[0] = dict(decoded[0])
        decoded_bad[0]["level"] = 99
        with self.assertRaises(RuntimeError) as ctx:
            assert_live_city_parity(canonical, decoded_bad, human)
        self.assertIn("PARITY-001 Live Discrepancy", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
