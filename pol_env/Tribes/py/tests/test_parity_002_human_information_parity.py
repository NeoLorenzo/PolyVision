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
        """Assert Java Types.java enum definitions and ordinals match Python contract data 1:1."""
        self.assertTrue(TYPES_JAVA_PATH.is_file(), f"Types.java not found at {TYPES_JAVA_PATH}")
        with open(TYPES_JAVA_PATH, "r", encoding="utf-8") as f:
            java_src = f.read()

        # 1. TECHNOLOGY enum order
        tech_match = re.search(r"public\s+enum\s+TECHNOLOGY\s*\{([^}]+)\}", java_src)
        self.assertIsNotNone(tech_match, "Could not find enum TECHNOLOGY in Types.java")
        tech_body = tech_match.group(1)
        java_techs = [
            m.group(1) for m in re.finditer(r"([A-Z_]+)\s*(?:\([^)]*\))?,?", tech_body)
            if m.group(1) not in ("NULL", "SEMICOLON")
        ]
        # First 24 elements before semicolon/methods
        java_tech_names = [t for t in java_techs if t in TECHNOLOGY_ORDER]
        self.assertEqual(tuple(java_tech_names), TECHNOLOGY_ORDER)
        self.assertEqual(len(TECHNOLOGY_ORDER), 24)

        # 2. BUILDING enum order / keys
        bld_match = re.search(r"public\s+enum\s+BUILDING\s*\{([\s\S]+?);\s*\n", java_src)
        self.assertIsNotNone(bld_match, "Could not find enum BUILDING in Types.java")
        bld_body = bld_match.group(1)
        java_buildings = []
        for line in bld_body.split("\n"):
            line = line.strip()
            m = re.match(r"([A-Z_]+)\s*\(\s*(\d+)", line)
            if m:
                java_buildings.append((m.group(1), int(m.group(2))))
        for expected_name, (actual_name, actual_key) in zip(SUPPORTED_BUILDINGS, java_buildings):
            self.assertEqual(expected_name, actual_name)
            self.assertEqual(SUPPORTED_BUILDINGS.index(expected_name), actual_key)
        self.assertEqual(len(SUPPORTED_BUILDINGS), 19)

        # 3. UNIT enum order / keys
        unit_match = re.search(r"public\s+enum\s+UNIT\s*\{([\s\S]+?);\s*\n", java_src)
        self.assertIsNotNone(unit_match, "Could not find enum UNIT in Types.java")
        unit_body = unit_match.group(1)
        java_units = []
        for line in unit_body.split("\n"):
            line = line.strip()
            m = re.match(r"([A-Z_]+)\s*\(\s*(\d+)", line)
            if m:
                java_units.append((m.group(1), int(m.group(2))))
        for expected_name, (actual_name, actual_key) in zip(SUPPORTED_UNIT_TYPES, java_units):
            self.assertEqual(expected_name, actual_name)
            self.assertEqual(SUPPORTED_UNIT_TYPES.index(expected_name), actual_key)
        self.assertEqual(len(SUPPORTED_UNIT_TYPES), 12)

        # 4. ACTION_STAR_COST_SCALE constant
        self.assertEqual(ACTION_STAR_COST_SCALE, 50.0)

    # --- Test 2: Observation Dimension Math & Layout Slices ---
    def test_02_observation_dimension_exact_math(self):
        """Assert exact 5,335 total observation dimension and contiguous slice partitions."""
        n = 121
        self.assertEqual(self.layout.expected_obs_dim, 5335)
        self.assertEqual(self.layout.terrain_end - self.layout.terrain_start, n)
        self.assertEqual(self.layout.unit_types_end - self.layout.unit_types_start, 12 * n)
        self.assertEqual(self.layout.city_territory_end - self.layout.city_territory_start, 9 * n)
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
        self.assertEqual(self.layout.road_start, self.layout.city_territory_end)
        self.assertEqual(self.layout.buildings_start, self.layout.road_end)
        self.assertEqual(self.layout.resource_start, self.layout.buildings_end)
        self.assertEqual(self.layout.legacy_scalar_start, self.layout.resource_end)
        self.assertEqual(self.layout.economy_scalar_start, self.layout.legacy_scalar_end)
        self.assertEqual(self.layout.tech_vector_start, self.layout.economy_scalar_end)
        self.assertEqual(self.layout.city_block_start, self.layout.tech_vector_end)
        self.assertEqual(self.layout.city_block_end, self.layout.expected_obs_dim)

    # --- Test 3: Raw Actor-ID Invariance (STATE-ID-001) ---
    def test_03_actor_id_invariance(self):
        """Permuting raw actor IDs yields identical observation tensors with zero raw ID exposure."""
        wrapper = object.__new__(TribesGymWrapper)
        wrapper._turn_count = 0
        wrapper.MAX_TURNS = 10
        wrapper._researched_techs_t10 = set()
        wrapper._controlled_tribe_id = 0
        wrapper._board_dimensions_from_obs = lambda obs: (11, 11)
        wrapper._get_bardur_stars = lambda obs: 5.0
        wrapper._compute_bardur_spt = lambda obs: 2.0
        wrapper._get_effective_researched_techs = lambda obs, tribe_id=0: {"HUNTING"}

        obs_1 = {
            "board": {
                "terrain": [[0] * 11 for _ in range(11)],
                "unitID": [[-1] * 11 for _ in range(11)],
                "cityID": [[101 if (x, y) == (3, 3) else -1 for y in range(11)] for x in range(11)],
                "building": [[-1] * 11 for _ in range(11)],
                "road": [[0] * 11 for _ in range(11)],
                "resource": [[-1] * 11 for _ in range(11)],
                "actorIDcounter": 9999,
            },
            "unit": {
                "501": {"x": 2, "y": 2, "type": 0, "tribeId": 0, "currentHP": 10},
            },
            "city": {
                "101": {"x": 3, "y": 3, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [501], "isCapital": True},
            },
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [101], "nKills": 0}},
            "tick": 1,
            "activeTribeID": 0,
        }

        # Same state with completely permuted raw actor IDs: 501 -> 88888, 101 -> 77777, counter 9999 -> 123
        obs_2 = {
            "board": {
                "terrain": [[0] * 11 for _ in range(11)],
                "unitID": [[-1] * 11 for _ in range(11)],
                "cityID": [[77777 if (x, y) == (3, 3) else -1 for y in range(11)] for x in range(11)],
                "building": [[-1] * 11 for _ in range(11)],
                "road": [[0] * 11 for _ in range(11)],
                "resource": [[-1] * 11 for _ in range(11)],
                "actorIDcounter": 123,
            },
            "unit": {
                "88888": {"x": 2, "y": 2, "type": 0, "tribeId": 0, "currentHP": 10},
            },
            "city": {
                "77777": {"x": 3, "y": 3, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [88888], "isCapital": True},
            },
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [77777], "nKills": 0}},
            "tick": 1,
            "activeTribeID": 0,
        }

        tensor_1 = wrapper._dict_to_array(obs_1)
        tensor_2 = wrapper._dict_to_array(obs_2)

        self.assertTrue(np.array_equal(tensor_1, tensor_2), "Observations with permuted actor IDs must be bitwise identical")
        self.assertNotIn(501.0, tensor_1)
        self.assertNotIn(101.0, tensor_1)
        self.assertNotIn(88888.0, tensor_2)
        self.assertNotIn(77777.0, tensor_2)
        self.assertNotIn(9999.0, tensor_1)

    # --- Test 4: Explicit Terrain-Fog Masking Defense-in-Depth ---
    def test_04_python_fog_masking_defense_in_depth(self):
        """Even if hidden values are present in serialized JSON on fog tiles (terrain=7), Python masks them to 0.0."""
        wrapper = object.__new__(TribesGymWrapper)
        wrapper._turn_count = 0
        wrapper.MAX_TURNS = 10
        wrapper._researched_techs_t10 = set()
        wrapper._controlled_tribe_id = 0
        wrapper._board_dimensions_from_obs = lambda obs: (11, 11)
        wrapper._get_bardur_stars = lambda obs: 5.0
        wrapper._compute_bardur_spt = lambda obs: 2.0
        wrapper._get_effective_researched_techs = lambda obs, tribe_id=0: set()

        # Create state where tile (5, 5) is FOG (terrain=7), but leaked road=1, building=3 (FARM), unit=WARRIOR, resource=2 (ANIMAL)
        terrain = [[0] * 11 for _ in range(11)]
        terrain[5][5] = 7  # Fog

        building = [[-1] * 11 for _ in range(11)]
        building[5][5] = 3  # Farm (leaked)

        road = [[0] * 11 for _ in range(11)]
        road[5][5] = 1  # Road (leaked)

        resource = [[-1] * 11 for _ in range(11)]
        resource[5][5] = 2  # Animal (leaked)

        obs = {
            "board": {
                "terrain": terrain,
                "unitID": [[-1] * 11 for _ in range(11)],
                "cityID": [[-1] * 11 for _ in range(11)],
                "building": building,
                "road": road,
                "resource": resource,
            },
            "unit": {
                "1": {"x": 5, "y": 5, "type": 0, "tribeId": 0, "currentHP": 10},  # Unit inside fog
            },
            "city": {},
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [], "nKills": 0}},
            "tick": 1,
            "activeTribeID": 0,
        }

        tensor = wrapper._dict_to_array(obs)
        fog_tile_idx = 5 * 11 + 5

        # Check unit channels: all 12 channels must be 0.0 at fog tile
        for u_idx in range(NUM_UNIT_TYPE_CHANNELS):
            u_chan_start = self.layout.unit_types_start + u_idx * 121
            self.assertEqual(tensor[u_chan_start + fog_tile_idx], 0.0)

        # Check road channel: must be 0.0 at fog tile
        self.assertEqual(tensor[self.layout.road_start + fog_tile_idx], 0.0)

        # Check building channels: all 19 channels must be 0.0 at fog tile
        for b_idx in range(NUM_BUILDING_CHANNELS):
            b_chan_start = self.layout.buildings_start + b_idx * 121
            self.assertEqual(tensor[b_chan_start + fog_tile_idx], 0.0)

        # Check resource channel: must be 0.0 at fog tile
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
                "1": {"x": 2, "y": 3, "type": 0, "tribeId": 0},
                "2": {"x": 4, "y": 5, "type": 1, "tribeId": 0},
                "3": {"x": 6, "y": 7, "type": 4, "tribeId": 0},
            },
            "city": {},
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [], "nKills": 0}},
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

    # --- Test 6: Strict City Territory Association ---
    def test_06_strict_city_territory_association_and_unmapped_assertion(self):
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

    # --- Test 7: Full 24-Tech Vector (STATE-TECH-001) ---
    def test_07_full_24_tech_vector(self):
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

    # --- Test 8: Legal Action Feature Vector Math (FEAT-MISS-001 & FEAT-SPAT-001) ---
    def test_08_legal_action_features_star_cost_and_spatial_coordinates(self):
        """Legal action features row has 47 dims with normalized star cost and spatial coordinates."""
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
            "unit": {"1": {"x": 3, "y": 4, "type": 0}},
            "city": {"10": {"x": 2, "y": 8}},
        }

        # Case A: MOVE action from (3, 4) to (3, 5), star cost = 0
        move_action = {
            "type": "MOVE",
            "unit_id": 1,
            "src_x": 3,
            "src_y": 4,
            "dst_x": 3,
            "dst_y": 5,
            "star_cost": 0,
        }
        feat_move = wrapper._compute_legal_action_feature_vector_reference(move_action, obs)
        self.assertEqual(len(feat_move), 47)
        self.assertEqual(feat_move[0], 1.0)  # is_move
        self.assertEqual(feat_move[42], 0.0)  # star_cost / 50
        self.assertAlmostEqual(feat_move[43], 3.0 / 10.0)  # src_x_norm
        self.assertAlmostEqual(feat_move[44], 4.0 / 10.0)  # src_y_norm
        self.assertAlmostEqual(feat_move[45], 3.0 / 10.0)  # target_x_norm
        self.assertAlmostEqual(feat_move[46], 5.0 / 10.0)  # target_y_norm

        # Case B: SPAWN action at city (2, 8), unit type WARRIOR, star cost = 2
        spawn_action = {
            "type": "SPAWN",
            "city_id": 10,
            "city_x": 2,
            "city_y": 8,
            "star_cost": 2,
        }
        feat_spawn = wrapper._compute_legal_action_feature_vector_reference(spawn_action, obs)
        self.assertEqual(len(feat_spawn), 47)
        self.assertEqual(feat_spawn[14], 1.0)  # is_train_or_spawn
        self.assertAlmostEqual(feat_spawn[42], 2.0 / 50.0)  # star_cost_norm
        self.assertAlmostEqual(feat_spawn[43], 2.0 / 10.0)  # src_x_norm
        self.assertAlmostEqual(feat_spawn[44], 8.0 / 10.0)  # src_y_norm
        self.assertAlmostEqual(feat_spawn[45], 2.0 / 10.0)  # target_x_norm
        self.assertAlmostEqual(feat_spawn[46], 8.0 / 10.0)  # target_y_norm

        # Case C: BUILD action at city (2, 8), target (2, 7), building LUMBER_HUT, star cost = 2
        build_action = {
            "type": "BUILD",
            "city_id": 10,
            "city_x": 2,
            "city_y": 8,
            "target_x": 2,
            "target_y": 7,
            "building_type": "LUMBER_HUT",
            "star_cost": 2,
        }
        feat_build = wrapper._compute_legal_action_feature_vector_reference(build_action, obs)
        self.assertEqual(len(feat_build), 47)
        self.assertEqual(feat_build[18], 1.0)  # is_build
        self.assertAlmostEqual(feat_build[42], 2.0 / 50.0)  # star_cost_norm
        self.assertAlmostEqual(feat_build[43], 2.0 / 10.0)  # src_x_norm (city)
        self.assertAlmostEqual(feat_build[44], 8.0 / 10.0)  # src_y_norm (city)
        self.assertAlmostEqual(feat_build[45], 2.0 / 10.0)  # target_x_norm
        self.assertAlmostEqual(feat_build[46], 7.0 / 10.0)  # target_y_norm

        # Case D: RESEARCH_TECH action, non-spatial, star cost = 5
        research_action = {
            "type": "RESEARCH_TECH",
            "tech_type": "ORGANIZATION",
            "star_cost": 5,
        }
        feat_research = wrapper._compute_legal_action_feature_vector_reference(research_action, obs)
        self.assertEqual(len(feat_research), 47)
        self.assertEqual(feat_research[15], 1.0)  # is_research
        self.assertAlmostEqual(feat_research[42], 5.0 / 50.0)  # star_cost_norm
        self.assertEqual(feat_research[43], 0.0)  # non-spatial
        self.assertEqual(feat_research[44], 0.0)
        self.assertEqual(feat_research[45], 0.0)
        self.assertEqual(feat_research[46], 0.0)

    # --- Test 9: Exact Road Grid (STATE-MAP-002) ---
    def test_09_exact_road_grid_representation(self):
        """Visible roads map to binary road plane in the observation tensor."""
        wrapper = object.__new__(TribesGymWrapper)
        wrapper._turn_count = 0
        wrapper.MAX_TURNS = 10
        wrapper._researched_techs_t10 = set()
        wrapper._controlled_tribe_id = 0
        wrapper._board_dimensions_from_obs = lambda obs: (11, 11)
        wrapper._get_bardur_stars = lambda obs: 5.0
        wrapper._compute_bardur_spt = lambda obs: 2.0
        wrapper._get_effective_researched_techs = lambda obs, tribe_id=0: set()

        road_grid = [[0] * 11 for _ in range(11)]
        road_grid[2][3] = 1
        road_grid[2][4] = 1
        road_grid[2][5] = 1

        obs = {
            "board": {
                "terrain": [[0] * 11 for _ in range(11)],
                "unitID": [[-1] * 11 for _ in range(11)],
                "cityID": [[-1] * 11 for _ in range(11)],
                "building": [[-1] * 11 for _ in range(11)],
                "road": road_grid,
                "resource": [[-1] * 11 for _ in range(11)],
            },
            "unit": {},
            "city": {},
            "tribes": {"0": {"star": 5, "score": 100, "citiesID": [], "nKills": 0}},
            "tick": 1,
            "activeTribeID": 0,
        }

        tensor = wrapper._dict_to_array(obs)
        road_slice = tensor[self.layout.road_start : self.layout.road_end]
        self.assertEqual(len(road_slice), 121)
        self.assertEqual(road_slice[2 * 11 + 3], 1.0)
        self.assertEqual(road_slice[2 * 11 + 4], 1.0)
        self.assertEqual(road_slice[2 * 11 + 5], 1.0)
        self.assertEqual(np.sum(road_slice), 3.0)

    # --- Test 10: Exact Capital Identity (STATE-MAP-003) ---
    def test_10_exact_capital_identity(self):
        """Capital city identity is explicitly represented in city slot feature index 9."""
        pov_obs = {
            "city": {
                "1": {"x": 2, "y": 2, "level": 1, "population": 0, "population_need": 2, "production": 2, "tribeID": 0, "units": [], "isCapital": True},
                "2": {"x": 5, "y": 5, "level": 2, "population": 1, "population_need": 3, "production": 3, "tribeID": 0, "units": [], "isCapital": False},
            }
        }
        cities = extract_owned_cities(pov_obs, tribe_id=0)
        self.assertEqual(len(cities), 2)
        self.assertTrue(cities[0].is_capital)
        self.assertFalse(cities[1].is_capital)

        encoded = encode_owned_city_slots(cities, width=11, height=11)
        self.assertEqual(len(encoded), 90)
        self.assertEqual(encoded[9], 1.0)   # slot 0 is_capital
        self.assertEqual(encoded[19], 0.0)  # slot 1 is_capital

        decoded = decode_owned_city_slots(encoded, width=11, height=11)
        self.assertEqual(len(decoded), 2)
        self.assertTrue(decoded[0]["is_capital"])
        self.assertFalse(decoded[1]["is_capital"])


if __name__ == "__main__":
    unittest.main()
