import dataclasses
import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from pol_env.Tribes.py.environment_contract import (
    ACTION_STAR_COST_SCALE,
    CheckpointCompatibilityError,
    CITY_BLOCK_DIM,
    CITY_SLOT_FEATURE_DIM,
    CITY_SLOT_FEATURE_NAMES,
    MAX_OWNED_CITIES,
    NUM_BUILDING_CHANNELS,
    NUM_CITY_TERRITORY_CHANNELS,
    NUM_TECHNOLOGIES,
    NUM_UNIT_HOME_CITY_CHANNELS,
    NUM_UNIT_TYPE_CHANNELS,
    ObservationContractError,
    ObservationLayout,
    OwnedCityState,
    PHASE1_ENVIRONMENT_VERSION,
    SUPPORTED_BUILDINGS,
    SUPPORTED_UNIT_TYPES,
    TECHNOLOGY_ORDER,
    decode_owned_city_slots,
    encode_owned_city_slots,
    extract_owned_cities,
    observation_layout,
    validate_checkpoint_compatibility,
)
from pol_env.Tribes.py.register_env import GlobalActionCatalog, TribesGymWrapper


REPO_ROOT = Path(__file__).resolve().parents[4]
TYPES_JAVA_PATH = REPO_ROOT / "pol_env" / "Tribes" / "src" / "core" / "Types.java"


class TestParity002HumanInformationParity(unittest.TestCase):
    def setUp(self):
        self.width = 11
        self.height = 11
        self.layout = observation_layout(self.width, self.height)

    # --- Test 1: Cross-Language Semantic Ordering & Contract Validation ---
    def test_01_cross_language_contract_vocabularies_match_java(self):
        """Assert Java Types.java enum definitions, lengths, ordinals and keys match Python contract 1:1."""
        self.assertTrue(TYPES_JAVA_PATH.is_file(), f"Types.java not found at {TYPES_JAVA_PATH}")
        with open(TYPES_JAVA_PATH, "r", encoding="utf-8") as f:
            java_src = f.read()

        # 1. TECHNOLOGY enum order and length
        tech_match = re.search(r"public\s+enum\s+TECHNOLOGY\s*\{([\s\S]+?);\s*\n", java_src)
        self.assertIsNotNone(tech_match, "Could not find enum TECHNOLOGY in Types.java")
        tech_body = tech_match.group(1)
        java_techs = []
        for line in tech_body.split("\n"):
            line = line.strip()
            m = re.match(r"^([A-Z_]+)\s*(?:\([^)]*\))?,?$", line)
            if m:
                java_techs.append(m.group(1))

        self.assertEqual(len(java_techs), len(TECHNOLOGY_ORDER), f"Java tech enum count {len(java_techs)} != Python {len(TECHNOLOGY_ORDER)}")
        self.assertEqual(tuple(java_techs), TECHNOLOGY_ORDER)
        self.assertEqual(len(TECHNOLOGY_ORDER), 24)

        # 2. BUILDING enum order / keys / length
        bld_match = re.search(r"public\s+enum\s+BUILDING\s*\{([\s\S]+?);\s*\n", java_src)
        self.assertIsNotNone(bld_match, "Could not find enum BUILDING in Types.java")
        bld_body = bld_match.group(1)
        java_buildings = []
        for line in bld_body.split("\n"):
            line = line.strip()
            m = re.match(r"^([A-Z_]+)\s*\(\s*(\d+)", line)
            if m:
                java_buildings.append((m.group(1), int(m.group(2))))

        self.assertEqual(len(java_buildings), len(SUPPORTED_BUILDINGS), f"Java building count {len(java_buildings)} != Python {len(SUPPORTED_BUILDINGS)}")
        self.assertEqual(len(SUPPORTED_BUILDINGS), 19)
        self.assertEqual(tuple(b[0] for b in java_buildings), SUPPORTED_BUILDINGS)
        self.assertEqual(tuple(b[1] for b in java_buildings), tuple(range(19)))

        # 3. UNIT enum order / keys / length
        unit_match = re.search(r"public\s+enum\s+UNIT\s*\{([\s\S]+?);\s*\n", java_src)
        self.assertIsNotNone(unit_match, "Could not find enum UNIT in Types.java")
        unit_body = unit_match.group(1)
        java_units = []
        for line in unit_body.split("\n"):
            line = line.strip()
            m = re.match(r"^([A-Z_]+)\s*\(\s*(\d+)", line)
            if m:
                java_units.append((m.group(1), int(m.group(2))))

        self.assertEqual(len(java_units), len(SUPPORTED_UNIT_TYPES), f"Java unit count {len(java_units)} != Python {len(SUPPORTED_UNIT_TYPES)}")
        self.assertEqual(len(SUPPORTED_UNIT_TYPES), 12)
        self.assertEqual(tuple(u[0] for u in java_units), SUPPORTED_UNIT_TYPES)
        self.assertEqual(tuple(u[1] for u in java_units), tuple(range(12)))

        # 4. ACTION_STAR_COST_SCALE constant
        self.assertEqual(ACTION_STAR_COST_SCALE, 50.0)

    # --- Test 2: Observation Dimension Math & Layout Slices ---
    def test_02_observation_dimension_exact_math(self):
        """Assert exact 6,424 total observation dimension and contiguous slice partitions."""
        n = 121
        self.assertEqual(self.layout.expected_obs_dim, 6424)
        self.assertEqual(self.layout.terrain_end - self.layout.terrain_start, n)
        self.assertEqual(self.layout.unit_types_end - self.layout.unit_types_start, 12 * n)
        self.assertEqual(self.layout.city_territory_end - self.layout.city_territory_start, 9 * n)
        self.assertEqual(self.layout.unit_home_city_end - self.layout.unit_home_city_start, 9 * n)
        self.assertEqual(self.layout.road_end - self.layout.road_start, n)
        self.assertEqual(self.layout.buildings_end - self.layout.buildings_start, 19 * n)
        self.assertEqual(self.layout.resource_end - self.layout.resource_start, n)
        self.assertEqual(self.layout.legacy_scalar_end - self.layout.legacy_scalar_start, 6)
        self.assertEqual(self.layout.economy_scalar_end - self.layout.economy_scalar_start, 12)
        self.assertEqual(self.layout.tech_vector_end - self.layout.tech_vector_start, 24)
        self.assertEqual(self.layout.city_block_end - self.layout.city_block_start, 90)

        # Verify exact slice continuity
        self.assertEqual(self.layout.terrain_start, 0)
        self.assertEqual(self.layout.unit_types_start, self.layout.terrain_end)
        self.assertEqual(self.layout.city_territory_start, self.layout.unit_types_end)
        self.assertEqual(self.layout.unit_home_city_start, self.layout.city_territory_end)
        self.assertEqual(self.layout.road_start, self.layout.unit_home_city_end)
        self.assertEqual(self.layout.buildings_start, self.layout.road_end)
        self.assertEqual(self.layout.resource_start, self.layout.buildings_end)
        self.assertEqual(self.layout.legacy_scalar_start, self.layout.resource_end)
        self.assertEqual(self.layout.economy_scalar_start, self.layout.legacy_scalar_end)
        self.assertEqual(self.layout.tech_vector_start, self.layout.economy_scalar_end)
        self.assertEqual(self.layout.city_block_start, self.layout.tech_vector_end)
        self.assertEqual(self.layout.city_block_end, self.layout.expected_obs_dim)

    # --- Test 3: Raw Actor-ID Invariance (STATE-ID-001) ---
    def test_03_actor_id_invariance_multi_city_multi_unit(self):
        """Permuting raw actor IDs across multiple cities/units yields bitwise identical observation tensors."""
        wrapper = object.__new__(TribesGymWrapper)
        wrapper._turn_count = 0
        wrapper.MAX_TURNS = 10
        wrapper._researched_techs_t10 = set()
        wrapper._controlled_tribe_id = 0
        wrapper._board_dimensions_from_obs = lambda obs: (11, 11)
        wrapper._get_bardur_stars = lambda obs: 5.0
        wrapper._compute_bardur_spt = lambda obs: 4.0
        wrapper._get_effective_researched_techs = lambda obs, tribe_id=0: {"HUNTING", "RIDING"}

        # State 1: City 101 at (2, 2), City 102 at (7, 7); Unit 501 at (2, 3), Unit 502 at (7, 8)
        city_id_grid_1 = [[-1] * 11 for _ in range(11)]
        city_id_grid_1[2][2] = 101
        city_id_grid_1[2][3] = 101
        city_id_grid_1[7][7] = 102
        city_id_grid_1[7][8] = 102

        unit_id_grid_1 = [[-1] * 11 for _ in range(11)]
        unit_id_grid_1[2][3] = 501
        unit_id_grid_1[7][8] = 502

        obs_1 = {
            "board": {
                "terrain": [[0] * 11 for _ in range(11)],
                "unitID": unit_id_grid_1,
                "cityID": city_id_grid_1,
                "building": [[-1] * 11 for _ in range(11)],
                "road": [[0] * 11 for _ in range(11)],
                "resource": [[-1] * 11 for _ in range(11)],
                "actorIDcounter": 9999,
            },
            "unit": {
                "501": {"x": 2, "y": 3, "type": 0, "tribeId": 0, "cityID": 101, "currentHP": 10},
                "502": {"x": 7, "y": 8, "type": 1, "tribeId": 0, "cityID": 102, "currentHP": 10},
            },
            "city": {
                "101": {"x": 2, "y": 2, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [501], "isCapital": True},
                "102": {"x": 7, "y": 7, "level": 1, "population": 1, "population_need": 2, "production": 2, "tribeID": 0, "units": [502], "isCapital": False},
            },
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [101, 102], "nKills": 0}},
            "tick": 1,
            "activeTribeID": 0,
        }

        # State 2: Permuted actor IDs: 101 -> 8888, 102 -> 4444, 501 -> 3333, 502 -> 1111, counter -> 55
        city_id_grid_2 = [[-1] * 11 for _ in range(11)]
        city_id_grid_2[2][2] = 8888
        city_id_grid_2[2][3] = 8888
        city_id_grid_2[7][7] = 4444
        city_id_grid_2[7][8] = 4444

        unit_id_grid_2 = [[-1] * 11 for _ in range(11)]
        unit_id_grid_2[2][3] = 3333
        unit_id_grid_2[7][8] = 1111

        obs_2 = {
            "board": {
                "terrain": [[0] * 11 for _ in range(11)],
                "unitID": unit_id_grid_2,
                "cityID": city_id_grid_2,
                "building": [[-1] * 11 for _ in range(11)],
                "road": [[0] * 11 for _ in range(11)],
                "resource": [[-1] * 11 for _ in range(11)],
                "actorIDcounter": 55,
            },
            "unit": {
                "3333": {"x": 2, "y": 3, "type": 0, "tribeId": 0, "cityID": 8888, "currentHP": 10},
                "1111": {"x": 7, "y": 8, "type": 1, "tribeId": 0, "cityID": 4444, "currentHP": 10},
            },
            "city": {
                "8888": {"x": 2, "y": 2, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [3333], "isCapital": True},
                "4444": {"x": 7, "y": 7, "level": 1, "population": 1, "population_need": 2, "production": 2, "tribeID": 0, "units": [1111], "isCapital": False},
            },
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [8888, 4444], "nKills": 0}},
            "tick": 1,
            "activeTribeID": 0,
        }

        tensor_1 = wrapper._dict_to_array(obs_1)
        tensor_2 = wrapper._dict_to_array(obs_2)

        self.assertTrue(np.array_equal(tensor_1, tensor_2), "Observations with permuted actor IDs must be bitwise identical")
        self.assertNotIn(501.0, tensor_1)
        self.assertNotIn(101.0, tensor_1)
        self.assertNotIn(8888.0, tensor_2)
        self.assertNotIn(3333.0, tensor_2)
        self.assertNotIn(9999.0, tensor_1)

    # --- Test 4: Explicit Terrain-Fog Masking Defense-in-Depth Across All Channels ---
    def test_04_python_fog_masking_defense_in_depth_all_spatial_channels(self):
        """Even if hidden values exist in serialized JSON on fog tiles (terrain=7), Python masks all channels to 0.0."""
        wrapper = object.__new__(TribesGymWrapper)
        wrapper._turn_count = 0
        wrapper.MAX_TURNS = 10
        wrapper._researched_techs_t10 = set()
        wrapper._controlled_tribe_id = 0
        wrapper._board_dimensions_from_obs = lambda obs: (11, 11)
        wrapper._get_bardur_stars = lambda obs: 5.0
        wrapper._compute_bardur_spt = lambda obs: 2.0
        wrapper._get_effective_researched_techs = lambda obs, tribe_id=0: set()

        terrain = [[0] * 11 for _ in range(11)]
        terrain[5][5] = 7  # Fog

        building = [[-1] * 11 for _ in range(11)]
        building[5][5] = 3  # Farm (leaked)

        road = [[0] * 11 for _ in range(11)]
        road[5][5] = 1  # Road (leaked)

        resource = [[-1] * 11 for _ in range(11)]
        resource[5][5] = 2  # Animal (leaked)

        city_id_grid = [[-1] * 11 for _ in range(11)]
        city_id_grid[5][5] = 10  # Leaked city territory

        obs = {
            "board": {
                "terrain": terrain,
                "unitID": [[-1] * 11 for _ in range(11)],
                "cityID": city_id_grid,
                "building": building,
                "road": road,
                "resource": resource,
            },
            "unit": {
                "1": {"x": 5, "y": 5, "type": 0, "tribeId": 0, "cityID": 10, "currentHP": 10},  # Unit inside fog
            },
            "city": {
                "10": {"x": 5, "y": 5, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [1], "isCapital": True},
            },
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [10], "nKills": 0}},
            "tick": 1,
            "activeTribeID": 0,
        }

        tensor = wrapper._dict_to_array(obs)
        fog_tile_idx = 5 * 11 + 5

        # 1. Unit type channels (12 channels)
        for u_idx in range(NUM_UNIT_TYPE_CHANNELS):
            u_chan_start = self.layout.unit_types_start + u_idx * 121
            self.assertEqual(tensor[u_chan_start + fog_tile_idx], 0.0)

        # 2. City territory channels (9 channels)
        for c_idx in range(NUM_CITY_TERRITORY_CHANNELS):
            c_chan_start = self.layout.city_territory_start + c_idx * 121
            self.assertEqual(tensor[c_chan_start + fog_tile_idx], 0.0)

        # 3. Unit home city channels (9 channels)
        for uhc_idx in range(NUM_UNIT_HOME_CITY_CHANNELS):
            uhc_chan_start = self.layout.unit_home_city_start + uhc_idx * 121
            self.assertEqual(tensor[uhc_chan_start + fog_tile_idx], 0.0)

        # 4. Road channel (1 channel)
        self.assertEqual(tensor[self.layout.road_start + fog_tile_idx], 0.0)

        # 5. Building channels (19 channels)
        for b_idx in range(NUM_BUILDING_CHANNELS):
            b_chan_start = self.layout.buildings_start + b_idx * 121
            self.assertEqual(tensor[b_chan_start + fog_tile_idx], 0.0)

        # 6. Resource channel (1 channel)
        self.assertEqual(tensor[self.layout.resource_start + fog_tile_idx], 0.0)

    # --- Test 5: Categorical Unit-Type Representation (STATE-UNIT-001) ---
    def test_05_categorical_unit_types(self):
        """Visible units map to their exact categorical unit-type channel."""
        wrapper = object.__new__(TribesGymWrapper)
        wrapper._turn_count = 0
        wrapper.MAX_TURNS = 10
        wrapper._researched_techs_t10 = set()
        wrapper._controlled_tribe_id = 0
        wrapper._board_dimensions_from_obs = lambda obs: (11, 11)
        wrapper._get_bardur_stars = lambda obs: 5.0
        wrapper._compute_bardur_spt = lambda obs: 2.0
        wrapper._get_effective_researched_techs = lambda obs, tribe_id=0: set()

        # Place Warrior (type 0) at (2, 3), Rider (type 1) at (4, 5), Archer (type 4) at (6, 7)
        obs = {
            "board": {
                "terrain": [[0] * 11 for _ in range(11)],
                "unitID": [[-1] * 11 for _ in range(11)],
                "cityID": [[-1] * 11 for _ in range(11)],
                "building": [[-1] * 11 for _ in range(11)],
                "road": [[0] * 11 for _ in range(11)],
                "resource": [[-1] * 11 for _ in range(11)],
            },
            "unit": {
                "1": {"x": 2, "y": 3, "type": 0, "tribeId": 0, "cityID": 10},
                "2": {"x": 4, "y": 5, "type": 1, "tribeId": 0, "cityID": 10},
                "3": {"x": 6, "y": 7, "type": 4, "tribeId": 0, "cityID": 10},
            },
            "city": {
                "10": {"x": 0, "y": 0, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [1, 2, 3], "isCapital": True},
            },
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [10], "nKills": 0}},
            "tick": 1,
            "activeTribeID": 0,
        }

        tensor = wrapper._dict_to_array(obs)

        # Warrior channel (idx 0 in SUPPORTED_UNIT_TYPES)
        w_chan = tensor[self.layout.unit_types_start : self.layout.unit_types_start + 121]
        self.assertEqual(w_chan[2 * 11 + 3], 1.0)
        self.assertEqual(np.sum(w_chan), 1.0)

        # Rider channel (idx 1 in SUPPORTED_UNIT_TYPES)
        r_chan = tensor[self.layout.unit_types_start + 121 : self.layout.unit_types_start + 2 * 121]
        self.assertEqual(r_chan[4 * 11 + 5], 1.0)
        self.assertEqual(np.sum(r_chan), 1.0)

        # Archer channel (idx 4 in SUPPORTED_UNIT_TYPES)
        a_chan = tensor[self.layout.unit_types_start + 4 * 121 : self.layout.unit_types_start + 5 * 121]
        self.assertEqual(a_chan[6 * 11 + 7], 1.0)
        self.assertEqual(np.sum(a_chan), 1.0)

    # --- Test 6: Unit Home-City Association Channels (STATE-UNIT-003) ---
    def test_06_unit_home_city_association_channels(self):
        """Visible owned units map to the deterministic city slot of their supporting home city."""
        wrapper = object.__new__(TribesGymWrapper)
        wrapper._turn_count = 0
        wrapper.MAX_TURNS = 10
        wrapper._researched_techs_t10 = set()
        wrapper._controlled_tribe_id = 0
        wrapper._board_dimensions_from_obs = lambda obs: (11, 11)
        wrapper._get_bardur_stars = lambda obs: 5.0
        wrapper._compute_bardur_spt = lambda obs: 4.0
        wrapper._get_effective_researched_techs = lambda obs, tribe_id=0: set()

        # Two cities: City A at (1, 1) -> Slot 0, City B at (8, 8) -> Slot 1
        # Unit 1 at (3, 3) supported by City A (Slot 0)
        # Unit 2 at (6, 6) supported by City B (Slot 1)
        obs = {
            "board": {
                "terrain": [[0] * 11 for _ in range(11)],
                "unitID": [[-1] * 11 for _ in range(11)],
                "cityID": [[-1] * 11 for _ in range(11)],
                "building": [[-1] * 11 for _ in range(11)],
                "road": [[0] * 11 for _ in range(11)],
                "resource": [[-1] * 11 for _ in range(11)],
            },
            "unit": {
                "10": {"x": 3, "y": 3, "type": 0, "tribeId": 0, "cityID": 100},
                "20": {"x": 6, "y": 6, "type": 0, "tribeId": 0, "cityID": 200},
            },
            "city": {
                "100": {"x": 1, "y": 1, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [10], "isCapital": True},
                "200": {"x": 8, "y": 8, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [20], "isCapital": False},
            },
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [100, 200], "nKills": 0}},
            "tick": 1,
            "activeTribeID": 0,
        }

        tensor = wrapper._dict_to_array(obs)

        # Slot 0 home-city channel (City A at (1, 1)): Unit at (3, 3)
        uhc_slot0 = tensor[self.layout.unit_home_city_start : self.layout.unit_home_city_start + 121]
        self.assertEqual(uhc_slot0[3 * 11 + 3], 1.0)
        self.assertEqual(uhc_slot0[6 * 11 + 6], 0.0)
        self.assertEqual(np.sum(uhc_slot0), 1.0)

        # Slot 1 home-city channel (City B at (8, 8)): Unit at (6, 6)
        uhc_slot1 = tensor[self.layout.unit_home_city_start + 121 : self.layout.unit_home_city_start + 2 * 121]
        self.assertEqual(uhc_slot1[6 * 11 + 6], 1.0)
        self.assertEqual(uhc_slot1[3 * 11 + 3], 0.0)
        self.assertEqual(np.sum(uhc_slot1), 1.0)

        # All other 7 home-city slots are empty
        for s in range(2, MAX_OWNED_CITIES):
            uhc_slot = tensor[self.layout.unit_home_city_start + s * 121 : self.layout.unit_home_city_start + (s + 1) * 121]
            self.assertEqual(np.sum(uhc_slot), 0.0)

        # Actor-ID invariance: permuting city IDs 100/200 -> 999/888 yields identical observation tensor
        obs_permuted = {
            "board": {
                "terrain": [[0] * 11 for _ in range(11)],
                "unitID": [[-1] * 11 for _ in range(11)],
                "cityID": [[-1] * 11 for _ in range(11)],
                "building": [[-1] * 11 for _ in range(11)],
                "road": [[0] * 11 for _ in range(11)],
                "resource": [[-1] * 11 for _ in range(11)],
            },
            "unit": {
                "77": {"x": 3, "y": 3, "type": 0, "tribeId": 0, "cityID": 999},
                "88": {"x": 6, "y": 6, "type": 0, "tribeId": 0, "cityID": 888},
            },
            "city": {
                "999": {"x": 1, "y": 1, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [77], "isCapital": True},
                "888": {"x": 8, "y": 8, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [88], "isCapital": False},
            },
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [999, 888], "nKills": 0}},
            "tick": 1,
            "activeTribeID": 0,
        }
        tensor_permuted = wrapper._dict_to_array(obs_permuted)
        np.testing.assert_array_equal(tensor, tensor_permuted)

        # Legitimate NO_HOME_CITY unit: visible owned unit with tribeId=0, cityID=-1, type=SUPERUNIT (idx 11)
        obs_no_home_city = {
            "board": {
                "terrain": [[0] * 11 for _ in range(11)],
                "unitID": [[-1] * 11 for _ in range(11)],
                "cityID": [[-1] * 11 for _ in range(11)],
                "building": [[-1] * 11 for _ in range(11)],
                "road": [[0] * 11 for _ in range(11)],
                "resource": [[-1] * 11 for _ in range(11)],
            },
            "unit": {
                "30": {"x": 3, "y": 6, "type": 11, "tribeId": 0, "cityID": -1},  # SUPERUNIT extra unit
            },
            "city": {
                "100": {"x": 1, "y": 1, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [], "isCapital": True},
            },
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [100], "nKills": 0}},
            "tick": 1,
            "activeTribeID": 0,
        }
        tensor_no_home_city = wrapper._dict_to_array(obs_no_home_city)

        # Correct SUPERUNIT spatial channel is 1.0 at (3, 6)
        superunit_idx = SUPPORTED_UNIT_TYPES.index("SUPERUNIT")
        su_chan = tensor_no_home_city[
            self.layout.unit_types_start + superunit_idx * 121 : self.layout.unit_types_start + (superunit_idx + 1) * 121
        ]
        self.assertEqual(su_chan[3 * 11 + 6], 1.0)
        self.assertEqual(np.sum(su_chan), 1.0)

        # All 9 home-city channels at (3, 6) are exactly 0.0
        for s in range(MAX_OWNED_CITIES):
            uhc_slot = tensor_no_home_city[
                self.layout.unit_home_city_start + s * 121 : self.layout.unit_home_city_start + (s + 1) * 121
            ]
            self.assertEqual(uhc_slot[3 * 11 + 6], 0.0)
            self.assertEqual(np.sum(uhc_slot), 0.0)

        # Fail closed: visible owned unit with unmapped non--1 cityID raises ObservationContractError
        obs_unmapped = {
            "board": {
                "terrain": [[0] * 11 for _ in range(11)],
                "unitID": [[-1] * 11 for _ in range(11)],
                "cityID": [[-1] * 11 for _ in range(11)],
                "building": [[-1] * 11 for _ in range(11)],
                "road": [[0] * 11 for _ in range(11)],
                "resource": [[-1] * 11 for _ in range(11)],
            },
            "unit": {
                "10": {"x": 3, "y": 3, "type": 0, "tribeId": 0, "cityID": 999},  # 999 not in city dict!
            },
            "city": {
                "100": {"x": 1, "y": 1, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [10], "isCapital": True},
            },
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [100], "nKills": 0}},
            "tick": 1,
            "activeTribeID": 0,
        }
        with self.assertRaises(ObservationContractError):
            wrapper._dict_to_array(obs_unmapped)

        # Fog defense-in-depth: unit under fog (with unmapped cityID or cityID=-1) does not raise and its channels remain 0.0
        obs_fogged = {
            "board": {
                "terrain": [[7] * 11 for _ in range(11)],  # all fog
                "unitID": [[-1] * 11 for _ in range(11)],
                "cityID": [[-1] * 11 for _ in range(11)],
                "building": [[-1] * 11 for _ in range(11)],
                "road": [[0] * 11 for _ in range(11)],
                "resource": [[-1] * 11 for _ in range(11)],
            },
            "unit": {
                "10": {"x": 3, "y": 3, "type": 0, "tribeId": 0, "cityID": 999},
                "30": {"x": 3, "y": 6, "type": 11, "tribeId": 0, "cityID": -1},
            },
            "city": {},
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [], "nKills": 0}},
            "tick": 1,
            "activeTribeID": 0,
        }
        tensor_fogged = wrapper._dict_to_array(obs_fogged)
        uhc_all_fog = tensor_fogged[self.layout.unit_home_city_start : self.layout.unit_home_city_end]
        self.assertEqual(np.sum(uhc_all_fog), 0.0)
        su_chan_fog = tensor_fogged[
            self.layout.unit_types_start + superunit_idx * 121 : self.layout.unit_types_start + (superunit_idx + 1) * 121
        ]
        self.assertEqual(np.sum(su_chan_fog), 0.0)

    # --- Test 7: Strict City Territory Association ---
    def test_07_strict_city_territory_association_and_unmapped_assertion(self):
        """Every visible city territory tile maps to its deterministic city slot; unmapped raises error."""
        wrapper = object.__new__(TribesGymWrapper)
        wrapper._turn_count = 0
        wrapper.MAX_TURNS = 10
        wrapper._researched_techs_t10 = set()
        wrapper._controlled_tribe_id = 0
        wrapper._board_dimensions_from_obs = lambda obs: (11, 11)
        wrapper._get_bardur_stars = lambda obs: 5.0
        wrapper._compute_bardur_spt = lambda obs: 2.0
        wrapper._get_effective_researched_techs = lambda obs, tribe_id=0: set()

        city_id_grid = [[-1] * 11 for _ in range(11)]
        city_id_grid[3][4] = 10  # City 10 territory
        city_id_grid[3][5] = 10

        obs_valid = {
            "board": {
                "terrain": [[0] * 11 for _ in range(11)],
                "unitID": [[-1] * 11 for _ in range(11)],
                "cityID": city_id_grid,
                "building": [[-1] * 11 for _ in range(11)],
                "road": [[0] * 11 for _ in range(11)],
                "resource": [[-1] * 11 for _ in range(11)],
            },
            "unit": {},
            "city": {
                "10": {"x": 3, "y": 4, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [], "isCapital": True},
            },
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [10], "nKills": 0}},
            "tick": 1,
            "activeTribeID": 0,
        }

        tensor = wrapper._dict_to_array(obs_valid)
        slot0_territory = tensor[self.layout.city_territory_start : self.layout.city_territory_start + 121]
        self.assertEqual(slot0_territory[3 * 11 + 4], 1.0)
        self.assertEqual(slot0_territory[3 * 11 + 5], 1.0)
        self.assertEqual(np.sum(slot0_territory), 2.0)

        # Unmapped territory (e.g. cityID 99 on board but no city record for 99) raises ObservationContractError
        obs_invalid = dict(obs_valid)
        invalid_grid = [[-1] * 11 for _ in range(11)]
        invalid_grid[7][7] = 99  # Unmatched city
        obs_invalid["board"] = dict(obs_valid["board"])
        obs_invalid["board"]["cityID"] = invalid_grid

        with self.assertRaises(ObservationContractError):
            wrapper._dict_to_array(obs_invalid)

    # --- Test 8: Full 24-Tech Vector (STATE-TECH-001) ---
    def test_08_full_24_tech_vector(self):
        """Full 24 technologies are represented as a binary vector in TECHNOLOGY_ORDER."""
        wrapper = object.__new__(TribesGymWrapper)
        wrapper._turn_count = 0
        wrapper.MAX_TURNS = 10
        wrapper._researched_techs_t10 = set()
        wrapper._controlled_tribe_id = 0
        wrapper._board_dimensions_from_obs = lambda obs: (11, 11)
        wrapper._get_bardur_stars = lambda obs: 5.0
        wrapper._compute_bardur_spt = lambda obs: 2.0
        wrapper._get_effective_researched_techs = lambda obs, tribe_id=0: {"HUNTING", "FORESTRY", "MATHEMATICS"}

        obs = {
            "board": {
                "terrain": [[0] * 11 for _ in range(11)],
                "unitID": [[-1] * 11 for _ in range(11)],
                "cityID": [[-1] * 11 for _ in range(11)],
                "building": [[-1] * 11 for _ in range(11)],
                "road": [[0] * 11 for _ in range(11)],
                "resource": [[-1] * 11 for _ in range(11)],
            },
            "unit": {},
            "city": {},
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [], "nKills": 0}},
            "tick": 1,
            "activeTribeID": 0,
        }

        tensor = wrapper._dict_to_array(obs)
        tech_slice = tensor[self.layout.tech_vector_start : self.layout.tech_vector_end]
        self.assertEqual(len(tech_slice), 24)

        for idx, tech_name in enumerate(TECHNOLOGY_ORDER):
            if tech_name in ("HUNTING", "FORESTRY", "MATHEMATICS"):
                self.assertEqual(tech_slice[idx], 1.0, f"Expected {tech_name} to be 1.0")
            else:
                self.assertEqual(tech_slice[idx], 0.0, f"Expected {tech_name} to be 0.0")

    # --- Test 9: Spatial Action Families Normalization (FEAT-SPAT-001) ---
    def test_09_all_spatial_action_families_normalization(self):
        """Verify spatial source/target normalization x/(width-1), y/(height-1) across all Phase-1 action families."""
        wrapper = object.__new__(TribesGymWrapper)
        wrapper.ACTION_FEATURE_DIM = 47
        wrapper.ACTION_STAR_COST_SCALE = 50.0
        wrapper.REVEAL_CLIP = 12.0
        wrapper.ADJ_FOG_MAX = 8.0
        wrapper._unit_previous_tiles = {}
        wrapper._board_dimensions_from_obs = lambda obs: (11, 11)
        wrapper._extract_move_components = lambda action, obs: (action.get("unit_id"), action.get("src_x"), action.get("src_y"), action.get("dst_x"), action.get("dst_y"))
        wrapper._estimate_newly_revealed_tiles_if_move = lambda obs, uid, dx, dy: 0
        wrapper._count_adjacent_fog_tiles = lambda obs, cx, cy: 0
        wrapper._get_visible_uncaptured_village_positions = lambda obs: set()
        wrapper._is_inside_owned_city_bounds = lambda obs, cx, cy: False
        wrapper._get_capital_position = lambda obs, tribe_id=0: (3, 3)
        wrapper._resolve_action_tech_type = lambda a: a.get("tech_type")
        wrapper._resolve_action_resource_type = lambda a: a.get("resource_type")
        wrapper._resolve_action_building_type = lambda a: a.get("building_type")
        wrapper._resolve_action_levelup_choice = lambda a: a.get("levelup_choice")
        wrapper._summarize_action_economy_expectation = lambda a, obs: {
            "expected_population_delta": 0,
            "expected_immediate_spt_delta": 0,
            "makes_level_up_available": False,
            "is_level_up_claim": False,
            "progress_before": 0,
            "ready_before": False,
        }
        wrapper._catalog = None

        obs = {
            "unit": {"1": {"x": 2, "y": 4, "type": 0}},
            "city": {"10": {"x": 3, "y": 7}},
        }

        # 1. MOVE: src (2, 4) -> dst (2, 5)
        a_move = {"type": "MOVE", "unit_id": 1, "src_x": 2, "src_y": 4, "dst_x": 2, "dst_y": 5}
        f_move = wrapper._compute_legal_action_feature_vector_reference(a_move, obs)
        self.assertAlmostEqual(f_move[43], 2.0 / 10.0)
        self.assertAlmostEqual(f_move[44], 4.0 / 10.0)
        self.assertAlmostEqual(f_move[45], 2.0 / 10.0)
        self.assertAlmostEqual(f_move[46], 5.0 / 10.0)

        # 2. CAPTURE: unit at (5, 6)
        a_cap = {"type": "CAPTURE", "src_x": 5, "src_y": 6, "target_x": 5, "target_y": 6}
        f_cap = wrapper._compute_legal_action_feature_vector_reference(a_cap, obs)
        self.assertAlmostEqual(f_cap[43], 5.0 / 10.0)
        self.assertAlmostEqual(f_cap[44], 6.0 / 10.0)
        self.assertAlmostEqual(f_cap[45], 5.0 / 10.0)
        self.assertAlmostEqual(f_cap[46], 6.0 / 10.0)

        # 3. EXAMINE: unit at (1, 9)
        a_ex = {"type": "EXAMINE", "src_x": 1, "src_y": 9}
        f_ex = wrapper._compute_legal_action_feature_vector_reference(a_ex, obs)
        self.assertAlmostEqual(f_ex[43], 1.0 / 10.0)
        self.assertAlmostEqual(f_ex[44], 9.0 / 10.0)
        self.assertAlmostEqual(f_ex[45], 1.0 / 10.0)
        self.assertAlmostEqual(f_ex[46], 9.0 / 10.0)

        # 4. SPAWN: city at (3, 7)
        a_sp = {"type": "SPAWN", "city_id": 10, "city_x": 3, "city_y": 7}
        f_sp = wrapper._compute_legal_action_feature_vector_reference(a_sp, obs)
        self.assertAlmostEqual(f_sp[43], 3.0 / 10.0)
        self.assertAlmostEqual(f_sp[44], 7.0 / 10.0)
        self.assertAlmostEqual(f_sp[45], 3.0 / 10.0)
        self.assertAlmostEqual(f_sp[46], 7.0 / 10.0)

        # 5. RESOURCE_GATHERING: city at (3, 7), target (3, 8)
        a_rg = {"type": "RESOURCE_GATHERING", "city_id": 10, "city_x": 3, "city_y": 7, "target_x": 3, "target_y": 8}
        f_rg = wrapper._compute_legal_action_feature_vector_reference(a_rg, obs)
        self.assertAlmostEqual(f_rg[43], 3.0 / 10.0)
        self.assertAlmostEqual(f_rg[44], 7.0 / 10.0)
        self.assertAlmostEqual(f_rg[45], 3.0 / 10.0)
        self.assertAlmostEqual(f_rg[46], 8.0 / 10.0)

        # 6. CLEAR_FOREST: city at (3, 7), target (4, 7)
        a_cf = {"type": "CLEAR_FOREST", "city_id": 10, "city_x": 3, "city_y": 7, "target_x": 4, "target_y": 7}
        f_cf = wrapper._compute_legal_action_feature_vector_reference(a_cf, obs)
        self.assertAlmostEqual(f_cf[43], 3.0 / 10.0)
        self.assertAlmostEqual(f_cf[44], 7.0 / 10.0)
        self.assertAlmostEqual(f_cf[45], 4.0 / 10.0)
        self.assertAlmostEqual(f_cf[46], 7.0 / 10.0)

        # 7. GROW_FOREST: city at (3, 7), target (2, 7)
        a_gf = {"type": "GROW_FOREST", "city_id": 10, "city_x": 3, "city_y": 7, "target_x": 2, "target_y": 7}
        f_gf = wrapper._compute_legal_action_feature_vector_reference(a_gf, obs)
        self.assertAlmostEqual(f_gf[43], 3.0 / 10.0)
        self.assertAlmostEqual(f_gf[44], 7.0 / 10.0)
        self.assertAlmostEqual(f_gf[45], 2.0 / 10.0)
        self.assertAlmostEqual(f_gf[46], 7.0 / 10.0)

        # 8. LEVEL_UP: city at (3, 7)
        a_lu = {"type": "LEVEL_UP", "city_id": 10, "city_x": 3, "city_y": 7}
        f_lu = wrapper._compute_legal_action_feature_vector_reference(a_lu, obs)
        self.assertAlmostEqual(f_lu[43], 3.0 / 10.0)
        self.assertAlmostEqual(f_lu[44], 7.0 / 10.0)
        self.assertAlmostEqual(f_lu[45], 3.0 / 10.0)
        self.assertAlmostEqual(f_lu[46], 7.0 / 10.0)

        # 9. BUILD: city at (3, 7), target (3, 6)
        a_bd = {"type": "BUILD", "city_id": 10, "city_x": 3, "city_y": 7, "target_x": 3, "target_y": 6}
        f_bd = wrapper._compute_legal_action_feature_vector_reference(a_bd, obs)
        self.assertAlmostEqual(f_bd[43], 3.0 / 10.0)
        self.assertAlmostEqual(f_bd[44], 7.0 / 10.0)
        self.assertAlmostEqual(f_bd[45], 3.0 / 10.0)
        self.assertAlmostEqual(f_bd[46], 6.0 / 10.0)

        # 10. Non-spatial (RESEARCH_TECH, END_TURN): (0, 0, 0, 0)
        a_rt = {"type": "RESEARCH_TECH", "tech_type": "CLIMBING"}
        f_rt = wrapper._compute_legal_action_feature_vector_reference(a_rt, obs)
        self.assertEqual((f_rt[43], f_rt[44], f_rt[45], f_rt[46]), (0.0, 0.0, 0.0, 0.0))

        a_et = {"type": "END_TURN"}
        f_et = wrapper._compute_legal_action_feature_vector_reference(a_et, obs)
        self.assertEqual((f_et[43], f_et[44], f_et[45], f_et[46]), (0.0, 0.0, 0.0, 0.0))

    # --- Test 10: Unit Turn Status Legality Signal Parity ---
    def test_10_unit_turn_status_legality_signal_parity(self):
        """In Phase 1, unit turn status (FRESH vs non-FRESH) is losslessly recoverable from the complete legal-action interface."""
        map_files = sorted(Path(REPO_ROOT / "pol_env" / "Tribes" / "levels" / "phase1_pool_bardur_real").glob("**/*.csv"))
        if not map_files:
            self.skipTest("No levels found in levels/phase1_pool_bardur_real")

        os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"
        os.environ["POLYVISION_INFO_MODE"] = "debug"
        os.environ["POLYVISION_LEVEL_POOL_GLOB"] = str(map_files[0]).replace("\\", "/")
        os.environ["POLYVISION_LEVEL_SELECTION_MODE"] = "fixed"

        env = TribesGymWrapper()
        try:
            obs, info = env.reset(seed=42)
            jvm = env.tribes_env._jvm
            gs = env.tribes_env._env.getGameState()

            # Step 1: Initial state (FRESH unit): legal MOVE actions exist
            raw_actions_fresh = env.tribes_env.list_actions()
            move_actions_fresh = [a for a in raw_actions_fresh if a.get("type") == "MOVE"]
            self.assertGreater(len(move_actions_fresh), 0, "Fresh unit must have legal MOVE actions")

            # Step 2: Controlled engine comparison: setting target unit's status to MOVED disables MOVE actions for that unit
            t0 = gs.getTribe(0)
            units = list(gs.getUnits(0))
            self.assertGreater(len(units), 0)
            target_unit = units[0]
            target_x = int(move_actions_fresh[0]["src_x"])
            target_y = int(move_actions_fresh[0]["src_y"])
            orig_status = target_unit.getStatus()
            self.assertEqual(str(orig_status), "FRESH")

            # Change status to MOVED in Java engine and recompute available player actions
            target_unit.setStatus(jvm.core.Types.TURN_STATUS.MOVED)
            env.tribes_env._env.recomputePlayerActions(0)
            raw_actions_moved = env.tribes_env.list_actions()
            move_actions_target = [a for a in raw_actions_moved if a.get("type") == "MOVE" and int(a.get("src_x", -1)) == target_x and int(a.get("src_y", -1)) == target_y]
            self.assertEqual(len(move_actions_target), 0, "Controlled MOVED status must produce zero legal MOVE actions for that unit")

            # Non-move actions (e.g. END_TURN, RESEARCH_TECH, SPAWN) remain legal
            end_turn_actions = [a for a in raw_actions_moved if a.get("type") == "END_TURN"]
            self.assertEqual(len(end_turn_actions), 1)

            # Restore FRESH status and verify MOVE actions return
            target_unit.setStatus(orig_status)
            env.tribes_env._env.recomputePlayerActions(0)
            raw_actions_restored = env.tribes_env.list_actions()
            move_actions_restored = [a for a in raw_actions_restored if a.get("type") == "MOVE" and int(a.get("src_x", -1)) == target_x and int(a.get("src_y", -1)) == target_y]
            self.assertGreater(len(move_actions_restored), 0, "Restored FRESH status must restore legal MOVE actions")

            # Step 3: Natural execution transition: executing a MOVE transitions unit to MOVED/FINISHED
            first_move = move_actions_fresh[0]
            res = env._canonicalize_action_to_global_id(first_move, env.tribes_env._last_obs)
            move_gid = int(res[0] if isinstance(res, tuple) else res)
            self.assertIsNotNone(move_gid)
            obs_after, rew, done, trun, info_after = env.step(move_gid)

            raw_actions_stepped = env.tribes_env.list_actions()
            unit_map = env.tribes_env._last_obs.get("unit", {})
            for u in unit_map.values():
                if u.get("status") in ("MOVED", "FINISHED"):
                    moved_unit_moves = [a for a in raw_actions_stepped if a.get("type") == "MOVE" and a.get("src_x") == u.get("x") and a.get("src_y") == u.get("y")]
                    self.assertEqual(len(moved_unit_moves), 0, "Stepped MOVED unit must generate zero legal MOVE actions")
        finally:
            env.close()

    # --- Test 11: Java Engine Authoritative Star Cost & Dynamic Research Cost Integration ---
    def test_11_java_engine_authoritative_star_cost_dynamic_and_static_integration(self):
        """Verify Java computeActionStarCost emits exact dynamic cost for RESEARCH_TECH (scaling with city count) and static costs."""
        map_files = sorted(Path(REPO_ROOT / "pol_env" / "Tribes" / "levels" / "phase1_pool_bardur_real").glob("**/*.csv"))
        if not map_files:
            self.skipTest("No levels found in levels/phase1_pool_bardur_real")

        os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"
        os.environ["POLYVISION_INFO_MODE"] = "debug"
        os.environ["POLYVISION_LEVEL_POOL_GLOB"] = str(map_files[0]).replace("\\", "/")
        os.environ["POLYVISION_LEVEL_SELECTION_MODE"] = "fixed"

        env = TribesGymWrapper()
        try:
            obs, info = env.reset(seed=42)
            jvm = env.tribes_env._jvm
            gs = env.tribes_env._env.getGameState()
            t0 = gs.getTribe(0)
            tech_tree = t0.getTechTree()

            # --- Part 1: 1-City State Dynamic Tech Cost ---
            num_cities_1 = t0.getNumCities()
            self.assertEqual(num_cities_1, 1)
            raw_actions_1 = env.tribes_env.list_actions()

            observed_families = set()
            tech_costs_1 = {}
            for a in raw_actions_1:
                a_type = a.get("type")
                observed_families.add(a_type)
                if a_type == "RESEARCH_TECH":
                    t_name = a.get("tech_type")
                    tech_enum = getattr(jvm.core.Types.TECHNOLOGY, t_name)
                    expected_cost = tech_enum.getCost(num_cities_1, tech_tree)
                    actual_cost = a.get("star_cost")
                    self.assertEqual(actual_cost, expected_cost)
                    tech_costs_1[t_name] = actual_cost
                elif a_type == "SPAWN":
                    u_name = a.get("unit_type")
                    u_enum = getattr(jvm.core.Types.UNIT, u_name)
                    self.assertEqual(a.get("star_cost"), u_enum.getCost())
                elif a_type == "RESOURCE_GATHERING":
                    r_name = a.get("resource_type")
                    r_enum = getattr(jvm.core.Types.RESOURCE, r_name)
                    self.assertEqual(a.get("star_cost"), r_enum.getCost())

            self.assertIn("ORGANIZATION", tech_costs_1)
            self.assertEqual(tech_costs_1["ORGANIZATION"], jvm.core.Types.TECHNOLOGY.ORGANIZATION.getCost(1, tech_tree))

            # Verify action feature index 42 matches exact serialized star cost for all legal slots
            legal_feats_1 = info["legal_action_features_padded"][:info["legal_action_count"]]
            legal_gids_1 = info["legal_global_ids_padded"][:info["legal_action_count"]]
            for slot_idx, feat in enumerate(legal_feats_1):
                gid = int(legal_gids_1[slot_idx])
                matching = [a for a in raw_actions_1 if env._canonicalize_action_to_global_id(a, env.tribes_env._last_obs)[0] == gid]
                if matching:
                    expected_star_cost = float(matching[0].get("star_cost", 0))
                    reconstructed_cost = feat[42] * ACTION_STAR_COST_SCALE
                    self.assertAlmostEqual(
                        reconstructed_cost,
                        expected_star_cost,
                        places=4,
                        msg=f"Feature 42 cost {reconstructed_cost} != serialized cost {expected_star_cost} for action {matching[0]}",
                    )

            # --- Part 2: 2-City State Dynamic Tech Cost Scaling ---
            # Construct a legitimate 2nd city in GameState for Tribe 0
            c2 = jvm.core.actors.City(5, 5, 0)
            c2_id = 9999
            gs.getBoard().addActor(c2, c2_id)
            t0.addCity(c2_id)
            env.tribes_env._env.recomputePlayerActions(0)

            num_cities_2 = t0.getNumCities()
            self.assertEqual(num_cities_2, 2)

            raw_actions_2 = env.tribes_env.list_actions()
            tech_costs_2 = {}
            for a in raw_actions_2:
                a_type = a.get("type")
                observed_families.add(a_type)
                if a_type == "RESEARCH_TECH":
                    t_name = a.get("tech_type")
                    tech_enum = getattr(jvm.core.Types.TECHNOLOGY, t_name)
                    expected_cost = tech_enum.getCost(num_cities_2, tech_tree)
                    actual_cost = a.get("star_cost")
                    self.assertEqual(actual_cost, expected_cost)
                    tech_costs_2[t_name] = actual_cost

            self.assertIn("ORGANIZATION", tech_costs_2)
            self.assertEqual(tech_costs_2["ORGANIZATION"], jvm.core.Types.TECHNOLOGY.ORGANIZATION.getCost(2, tech_tree))

            # Assert dynamic scaling on the SAME technology
            self.assertNotEqual(tech_costs_1["ORGANIZATION"], tech_costs_2["ORGANIZATION"])
            self.assertEqual(tech_costs_1["ORGANIZATION"], 5)  # 4 + 1*1
            self.assertEqual(tech_costs_2["ORGANIZATION"], 6)  # 4 + 1*2
            if "FORESTRY" in tech_costs_1 and "FORESTRY" in tech_costs_2:
                self.assertEqual(tech_costs_1["FORESTRY"], 6)  # 4 + 2*1
                self.assertEqual(tech_costs_2["FORESTRY"], 8)  # 4 + 2*2

            # Place ANIMAL resource in City 2 territory (5, 6) to make RESOURCE_GATHERING reachable (Bardur starts with HUNTING)
            gs.getBoard().setCityIdAt(5, 6, c2_id)
            gs.getBoard().setResourceAt(5, 6, jvm.core.Types.RESOURCE.ANIMAL)
            t0.setStars(20)
            env.tribes_env._env.recomputePlayerActions(0)
            raw_actions_rg = env.tribes_env.list_actions()
            for a in raw_actions_rg:
                a_type = a.get("type")
                observed_families.add(a_type)
                if a_type == "RESOURCE_GATHERING":
                    r_name = a.get("resource_type")
                    r_enum = getattr(jvm.core.Types.RESOURCE, r_name)
                    self.assertEqual(a.get("star_cost"), r_enum.getCost())

            # Research FORESTRY and place a FOREST tile in City 2 territory (5, 7) to make BUILD LUMBER_HUT reachable
            gs.getBoard().setCityIdAt(5, 7, c2_id)
            gs.getBoard().setTerrainAt(5, 7, jvm.core.Types.TERRAIN.FOREST)
            t0.getTechTree().doResearchInit(jvm.core.Types.TECHNOLOGY.FORESTRY)
            t0.setStars(20)
            env.tribes_env._env.recomputePlayerActions(0)
            raw_actions_forestry = env.tribes_env.list_actions()
            for a in raw_actions_forestry:
                a_type = a.get("type")
                observed_families.add(a_type)
                if a_type == "BUILD":
                    b_name = a.get("building_type")
                    b_enum = getattr(jvm.core.Types.BUILDING, b_name)
                    self.assertEqual(a.get("star_cost"), b_enum.getCost())

            # Research SPIRITUALISM and place a PLAIN tile in City 2 territory (5, 8) to make GROW_FOREST reachable
            gs.getBoard().setCityIdAt(5, 8, c2_id)
            gs.getBoard().setTerrainAt(5, 8, jvm.core.Types.TERRAIN.PLAIN)
            t0.getTechTree().doResearchInit(jvm.core.Types.TECHNOLOGY.SPIRITUALISM)
            t0.setStars(20)
            env.tribes_env._env.recomputePlayerActions(0)
            raw_actions_spiritualism = env.tribes_env.list_actions()
            for a in raw_actions_spiritualism:
                a_type = a.get("type")
                observed_families.add(a_type)
                if a_type == "GROW_FOREST":
                    self.assertEqual(a.get("star_cost"), jvm.core.TribesConfig.GROW_FOREST_COST)

            self.assertEqual(jvm.core.Types.UNIT.WARRIOR.getCost(), 2)
            self.assertEqual(jvm.core.Types.UNIT.RIDER.getCost(), 3)
            self.assertEqual(jvm.core.Types.RESOURCE.FRUIT.getCost(), 2)
            self.assertEqual(jvm.core.Types.RESOURCE.ANIMAL.getCost(), 2)
            self.assertEqual(jvm.core.Types.RESOURCE.ORE.getCost(), 0)
            self.assertEqual(jvm.core.TribesConfig.GROW_FOREST_COST, 5)
            self.assertEqual(jvm.core.TribesConfig.BURN_FOREST_COST, 5)
            self.assertEqual(jvm.core.TribesConfig.ROAD_COST, 2)
            self.assertEqual(jvm.core.Types.BUILDING.PORT.getCost(), 10)
            self.assertEqual(jvm.core.Types.BUILDING.MINE.getCost(), 5)
            self.assertEqual(jvm.core.Types.BUILDING.LUMBER_HUT.getCost(), 2)
            self.assertEqual(jvm.core.Types.BUILDING.SAWMILL.getCost(), 5)
            self.assertEqual(jvm.core.Types.BUILDING.FORGE.getCost(), 5)
            self.assertEqual(jvm.core.Types.BUILDING.FARM.getCost(), 5)
            self.assertEqual(jvm.core.Types.BUILDING.WINDMILL.getCost(), 5)
            self.assertEqual(jvm.core.Types.BUILDING.CUSTOMS_HOUSE.getCost(), 5)
            self.assertEqual(jvm.core.Types.BUILDING.TEMPLE.getCost(), 20)

            # Confirm observed required action families in state
            self.assertIn("RESEARCH_TECH", observed_families)
            self.assertIn("SPAWN", observed_families)
            self.assertIn("RESOURCE_GATHERING", observed_families)
            self.assertIn("BUILD", observed_families)
            self.assertIn("GROW_FOREST", observed_families)
        finally:
            env.close()

    # --- Test 12: Ruin Examine SuperUnit Engine Contract & No-Home-City Parity ---
    def test_12_ruin_examine_superunit_no_home_city_contract_regression(self):
        """Verify Java engine ExamineCommand creates SuperUnit with cityId == -1 (extra unit) and wrapper encodes it losslessly."""
        examine_java_path = REPO_ROOT / "pol_env" / "Tribes" / "src" / "core" / "actions" / "unitactions" / "command" / "ExamineCommand.java"
        board_java_path = REPO_ROOT / "pol_env" / "Tribes" / "src" / "core" / "game" / "Board.java"
        self.assertTrue(examine_java_path.is_file(), f"ExamineCommand.java not found at {examine_java_path}")
        self.assertTrue(board_java_path.is_file(), f"Board.java not found at {board_java_path}")

        with open(examine_java_path, "r", encoding="utf-8") as f:
            examine_src = f.read()
        with open(board_java_path, "r", encoding="utf-8") as f:
            board_src = f.read()

        # Confirm ExamineCommand constructs superunit with cityId = -1
        self.assertIn("Types.UNIT.createUnit(spawnPos, 0, false, -1, unit.getTribeId(), unitType)", examine_src)

        # Confirm Board.addUnit treats cityId == -1 as Tribe extra unit
        self.assertIn("if(u.getCityId() != -1)", board_src)
        self.assertIn("tribes[u.getTribeId()].addExtraUnit(u)", board_src)

        # Dynamic wrapper test: Full observation containing both a normal warrior in City 10 and a ruin SuperUnit with cityID = -1
        wrapper = object.__new__(TribesGymWrapper)
        wrapper._turn_count = 0
        wrapper.MAX_TURNS = 10
        wrapper._researched_techs_t10 = set()
        wrapper._controlled_tribe_id = 0
        wrapper._board_dimensions_from_obs = lambda obs: (11, 11)
        wrapper._get_bardur_stars = lambda obs: 5.0
        wrapper._compute_bardur_spt = lambda obs: 2.0
        wrapper._get_effective_researched_techs = lambda obs, tribe_id=0: set()

        obs_ruin_superunit = {
            "board": {
                "terrain": [[0] * 11 for _ in range(11)],
                "unitID": [[-1] * 11 for _ in range(11)],
                "cityID": [[-1] * 11 for _ in range(11)],
                "building": [[-1] * 11 for _ in range(11)],
                "road": [[0] * 11 for _ in range(11)],
                "resource": [[-1] * 11 for _ in range(11)],
            },
            "unit": {
                "1": {"x": 2, "y": 2, "type": 0, "tribeId": 0, "cityID": 10},   # Warrior supported by City 10
                "2": {"x": 5, "y": 7, "type": 11, "tribeId": 0, "cityID": -1},  # Ruin SuperUnit (no home city)
            },
            "city": {
                "10": {"x": 2, "y": 2, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [1], "isCapital": True},
            },
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [10], "nKills": 0}},
            "tick": 3,
            "activeTribeID": 0,
        }

        tensor = wrapper._dict_to_array(obs_ruin_superunit)
        self.assertEqual(tensor.shape, (6424,))

        # Warrior spatial plane (idx 0) has 1.0 at (2, 2)
        warrior_chan = tensor[self.layout.unit_types_start : self.layout.unit_types_start + 121]
        self.assertEqual(warrior_chan[2 * 11 + 2], 1.0)
        self.assertEqual(np.sum(warrior_chan), 1.0)

        # SuperUnit spatial plane (idx 11) has 1.0 at (5, 7)
        superunit_idx = SUPPORTED_UNIT_TYPES.index("SUPERUNIT")
        su_chan = tensor[
            self.layout.unit_types_start + superunit_idx * 121 : self.layout.unit_types_start + (superunit_idx + 1) * 121
        ]
        self.assertEqual(su_chan[5 * 11 + 7], 1.0)
        self.assertEqual(np.sum(su_chan), 1.0)

        # Home-city slot 0 (City 10) has 1.0 at (2, 2) and 0.0 at (5, 7)
        uhc_slot0 = tensor[self.layout.unit_home_city_start : self.layout.unit_home_city_start + 121]
        self.assertEqual(uhc_slot0[2 * 11 + 2], 1.0)
        self.assertEqual(uhc_slot0[5 * 11 + 7], 0.0)
        self.assertEqual(np.sum(uhc_slot0), 1.0)

        # All remaining home-city slots 1..8 are 0.0 everywhere
        for s in range(1, MAX_OWNED_CITIES):
            uhc_slot = tensor[
                self.layout.unit_home_city_start + s * 121 : self.layout.unit_home_city_start + (s + 1) * 121
            ]
            self.assertEqual(np.sum(uhc_slot), 0.0)


if __name__ == "__main__":
    unittest.main()
