import gymnasium as gym
import glob
import hashlib
import json
import numpy as np
import os
import pickle
import re
import time
from gymnasium.envs.registration import register
from .gym_env import TribesGymEnv, make_default_env


class GlobalActionCatalog:
    """Deterministic flat global action-ID catalog for fixed map dimensions."""

    def __init__(self, width, height, tech_types, train_unit_types, resource_types, building_types, levelup_choices):
        self.width = int(width)
        self.height = int(height)
        self.n_tiles = int(width * height)
        self.tech_types = sorted(set(tech_types))
        self.train_unit_types = sorted(set(train_unit_types))
        self.resource_types = sorted(set(resource_types))
        self.building_types = sorted(set(building_types))
        self.levelup_choices = sorted(set(levelup_choices))

        self.tech_to_idx = {k: i for i, k in enumerate(self.tech_types)}
        self.unit_to_idx = {k: i for i, k in enumerate(self.train_unit_types)}
        self.resource_to_idx = {k: i for i, k in enumerate(self.resource_types)}
        self.building_to_idx = {k: i for i, k in enumerate(self.building_types)}
        self.levelup_to_idx = {k: i for i, k in enumerate(self.levelup_choices)}

        self.offsets = {}
        cur = 0
        self.offsets["END_TURN"] = cur
        cur += 1

        self.offsets["MOVE"] = cur
        cur += self.n_tiles * self.n_tiles

        self.offsets["CAPTURE"] = cur
        # 3 capture modes max (CITY/VILLAGE/UNKNOWN) x src x target.
        cur += 3 * self.n_tiles * self.n_tiles

        self.offsets["TRAIN"] = cur
        cur += max(1, len(self.train_unit_types)) * self.n_tiles

        self.offsets["RESOURCE_GATHERING"] = cur
        cur += max(1, len(self.resource_types)) * self.n_tiles

        self.offsets["CLEAR_FOREST"] = cur
        cur += self.n_tiles

        self.offsets["GROW_FOREST"] = cur
        cur += self.n_tiles

        self.offsets["BUILD"] = cur
        cur += max(1, len(self.building_types)) * self.n_tiles

        self.offsets["RESEARCH_TECH"] = cur
        cur += max(1, len(self.tech_types))

        self.offsets["LEVEL_UP"] = cur
        cur += max(1, len(self.levelup_choices)) * self.n_tiles

        self.offsets["EXAMINE"] = cur
        cur += self.n_tiles

        self.total_size = int(cur)
        self.end_turn_id = int(self.offsets["END_TURN"])

    def tile_id(self, x, y):
        x = int(x)
        y = int(y)
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return None
        return x * self.height + y

    def id_end_turn(self):
        return self.end_turn_id

    def id_move(self, src_tile, dst_tile):
        if src_tile is None or dst_tile is None:
            return None
        return int(self.offsets["MOVE"] + src_tile * self.n_tiles + dst_tile)

    def id_capture(self, src_tile, target_tile, capture_type):
        if src_tile is None or target_tile is None:
            return None
        t_idx = 2
        cap = str(capture_type or "").upper()
        if "CITY" in cap:
            t_idx = 0
        elif "VILLAGE" in cap:
            t_idx = 1
        return int(self.offsets["CAPTURE"] + t_idx * self.n_tiles * self.n_tiles + src_tile * self.n_tiles + target_tile)

    def id_train(self, unit_type, city_tile):
        if city_tile is None:
            return None
        u_idx = self.unit_to_idx.get(str(unit_type or "").upper())
        if u_idx is None:
            return None
        return int(self.offsets["TRAIN"] + u_idx * self.n_tiles + city_tile)

    def id_resource(self, resource_type, resource_tile):
        if resource_tile is None:
            return None
        r_idx = self.resource_to_idx.get(str(resource_type or "").upper())
        if r_idx is None:
            return None
        return int(self.offsets["RESOURCE_GATHERING"] + r_idx * self.n_tiles + resource_tile)

    def id_clear_forest(self, forest_tile):
        if forest_tile is None:
            return None
        return int(self.offsets["CLEAR_FOREST"] + forest_tile)

    def id_grow_forest(self, target_tile):
        if target_tile is None:
            return None
        return int(self.offsets["GROW_FOREST"] + target_tile)

    def id_build(self, building_type, target_tile):
        if target_tile is None:
            return None
        b_idx = self.building_to_idx.get(str(building_type or "").upper())
        if b_idx is None:
            return None
        return int(self.offsets["BUILD"] + b_idx * self.n_tiles + target_tile)

    def id_research(self, tech_type):
        t_idx = self.tech_to_idx.get(str(tech_type or "").upper())
        if t_idx is None:
            return None
        return int(self.offsets["RESEARCH_TECH"] + t_idx)

    def id_levelup(self, choice, city_tile):
        if city_tile is None:
            return None
        l_idx = self.levelup_to_idx.get(str(choice or "").upper())
        if l_idx is None:
            return None
        return int(self.offsets["LEVEL_UP"] + l_idx * self.n_tiles + city_tile)

    def id_examine(self, unit_tile):
        if unit_tile is None:
            return None
        return int(self.offsets["EXAMINE"] + unit_tile)

    def fingerprint(self):
        payload = {
            "width": self.width,
            "height": self.height,
            "offsets": self.offsets,
            "tech_types": self.tech_types,
            "train_unit_types": self.train_unit_types,
            "resource_types": self.resource_types,
            "building_types": self.building_types,
            "levelup_choices": self.levelup_choices,
            "total_size": self.total_size,
        }
        dumped = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

# wrapper to make it gym-compatible
class TribesGymWrapper(gym.Env):
    PHASE1_LEVEL_FILE = "levels/phase1_12x12_2bardur.csv"
    DEFAULT_LEVEL_POOL_GLOB = "levels/phase1_pool/*.csv"
    MAX_TURNS = 10
    TERMINAL_SPT_REWARD_ENABLED_DEFAULT = False
    TERMINAL_SPT_BASE_WEIGHT_DEFAULT = 1.0
    TERMINAL_SPT_OVER_10_WEIGHT_DEFAULT = 2.0
    TERMINAL_SPT_OVER_15_WEIGHT_DEFAULT = 3.0
    RESOURCE_GATHER_UPGRADE_FILTER_ENABLED_DEFAULT = False

    # Phase1 village/city shaping.
    REVEAL_UNCAPTURED_VILLAGE_REWARD = 1.0
    MOVE_CLOSER_TO_VISIBLE_VILLAGE_REWARD = 0.5
    MOVE_ONTO_VILLAGE_REWARD = 1.0
    MOVE_ONTO_VISIBLE_NEUTRAL_VILLAGE_REWARD = 5.0
    MOVE_MISS_VISIBLE_NEUTRAL_VILLAGE_PENALTY = 2.0
    CAPTURE_CITY_BONUS_MIN = 3.0
    CAPTURE_CITY_BONUS_MAX = 6.0
    SPT_INCREASE_REWARD_MULTIPLIER = 5.0
    SPT_NONPOSITIVE_REWARD_MULTIPLIER = 1.0
    VILLAGE_BREADCRUMB_REWARD = 0.5
    # Disabled for diagnostics: holding behavior should be measured without these
    # extra shaping terms.
    HOLD_NEUTRAL_VILLAGE_END_TURN_REWARD = 0.0
    MOVE_OFF_NEUTRAL_VILLAGE_WHEN_CAPTURE_ILLEGAL_PENALTY = 0.0
    FOG_CLEAR_REWARD_PER_TILE = 0.08
    FOG_CLEAR_REWARD_MAX_TILES = 5
    USELESS_MOVE_FOG_MISS_PENALTY = 0.35
    ALLOWED_ACTION_TYPES = {
        "END_TURN",
        "MOVE",
        "CAPTURE",
        "EXAMINE",
        "SPAWN",
        "RESOURCE_GATHERING",
        "CLEAR_FOREST",
        "GROW_FOREST",
        "LEVEL_UP",
        "RESEARCH_TECH",
        "BUILD",
    }
    CATALOG_VERSION = "flat-v1"
    CANONICALIZER_VERSION = "flat-v1-structured"
    MAX_LEGAL_ACTIONS_DEFAULT = 1024
    LEGAL_ACTION_FEATURE_VERSION = "v1_3_move_focus_plus_semantic_econ"
    REVEAL_CLIP = 12.0
    ADJ_FOG_MAX = 8.0
    LEGAL_ACTION_FEATURE_NAMES = (
        "is_move",
        "newly_revealed_tiles_if_move_norm",
        "adjacent_fog_count_after_move_norm",
        "adjacent_fog_delta_norm",
        "is_zero_reveal_move",
        "target_contains_visible_uncaptured_village",
        "has_visible_uncaptured_village",
        "distance_delta_to_nearest_visible_uncaptured_village_norm",
        "is_immediate_backtrack",
        "target_inside_owned_city_bounds",
        "distance_from_capital_delta_norm",
        "unit_type_warrior",
        "is_end_turn",
        "is_capture",
        "is_train_or_spawn",
        "is_research",
        "is_resource_gathering",
        "is_level_up",
        "is_build",
        "is_clear_forest",
        "is_grow_forest",
        "is_other",
        "research_tech_id_norm",
        "research_is_organization",
        "research_is_forestry",
        "resource_id_norm",
        "resource_is_animal",
        "resource_is_fruit",
        "resource_is_fish",
        "resource_is_crop",
        "resource_is_metal",
        "build_id_norm",
        "build_is_lumber_hut",
        "build_is_sawmill",
        "levelup_choice_id_norm",
        "levelup_is_workshop",
        "expected_population_delta_norm",
        "expected_immediate_spt_delta_norm",
        "makes_level_up_available",
        "is_level_up_claim",
        "action_city_upgrade_progress_before_norm",
        "action_city_upgrade_ready_before",
    )
    ACTION_FEATURE_DIM = len(LEGAL_ACTION_FEATURE_NAMES)
    TRIBE_STARTING_TECH_BY_TYPE = {
        0: "CLIMBING",      # XIN_XI
        1: "ORGANIZATION",  # IMPERIUS
        2: "HUNTING",       # BARDUR
        3: "RIDING",        # OUMAJI
        4: "FISHING",       # KICKOO
        5: "ARCHERY",       # HOODRICK
        6: None,            # LUXIDOOR
        7: "SMITHERY",      # VENGIR
        8: "FARMING",       # ZEBASI
        9: "MEDITATION",    # AI_MO
        10: "SHIELDS",      # QUETZALI
        11: "ROADS",        # YADAKK
    }

    def __init__(self, level_file=None):
        self.tribes_env = make_default_env()
        self.level_file = level_file or self.PHASE1_LEVEL_FILE
        self._level_selection_mode = str(
            os.environ.get("POLYVISION_LEVEL_SELECTION_MODE", "round_robin")
        ).strip().lower()
        if self._level_selection_mode not in ("round_robin", "seeded_random"):
            self._level_selection_mode = "round_robin"
        self._level_pool = self._resolve_level_pool(self.level_file)
        self._level_pool_size = len(self._level_pool)
        self._level_pool_offset = 0
        self._level_pool_rng = None
        self._episode_index = 0
        self._current_level_file = self.level_file
        self._current_level_index = 0
        self._last_reset_seed = None
        self._seed_stream = None
        self._seed_stream_base = self._parse_int_env("POLYVISION_BASE_SEED", default=42)
        self.verbose_resets = os.environ.get("POLYVISION_VERBOSE_RESETS", "0").lower() in ("1", "true", "yes", "on")
        self.debug_opening_grid = os.environ.get("POLYVISION_OPENING_GRID_DEBUG", "0").lower() in ("1", "true", "yes", "on")
        self.render_mode = "rgb_array"        # Initialize the environment to get the actual action space size
        self._turn_count = 0
        self._starting_city_count = 1
        self._last_city_count = 1
        self._moved_on_t0 = False
        self._visible_village_streak_turns = 0
        self._queued_village_capture_unit_ids = set()
        self._unit_previous_tiles = {}
        self._initial_visible_tiles = 0
        self._episode_fog_tiles_cleared = 0
        self._turn_first_uncaptured_village_visible = None
        self._turn_second_city_captured = None
        self._turn_forestry_researched = None
        self._turn_organization_researched = None
        self._researched_techs_t10 = set()
        self._animals_harvested_t10 = 0
        self._fruit_harvested_t10 = 0
        self._lumber_huts_built_t10 = 0
        self._sawmills_built_t10 = 0
        self._forests_cleared_t10 = 0
        self._terminal_spt_bonus_applied_this_episode = False
        self._terminal_spt_reward_enabled = self._parse_bool_env(
            "POLYVISION_TERMINAL_SPT_REWARD_ENABLED",
            default=self.TERMINAL_SPT_REWARD_ENABLED_DEFAULT,
        )
        self._terminal_spt_base_weight = self._parse_float_env(
            "POLYVISION_TERMINAL_SPT_BASE_WEIGHT",
            default=self.TERMINAL_SPT_BASE_WEIGHT_DEFAULT,
        )
        self._terminal_spt_over_10_weight = self._parse_float_env(
            "POLYVISION_TERMINAL_SPT_OVER_10_WEIGHT",
            default=self.TERMINAL_SPT_OVER_10_WEIGHT_DEFAULT,
        )
        self._terminal_spt_over_15_weight = self._parse_float_env(
            "POLYVISION_TERMINAL_SPT_OVER_15_WEIGHT",
            default=self.TERMINAL_SPT_OVER_15_WEIGHT_DEFAULT,
        )
        self._resource_gather_upgrade_filter_enabled = self._parse_bool_env(
            "POLYVISION_RESOURCE_GATHER_UPGRADE_FILTER_ENABLED",
            default=self.RESOURCE_GATHER_UPGRADE_FILTER_ENABLED_DEFAULT,
        )
        self._catalog = None
        self._catalog_fingerprint = ""
        self._tech_name_to_obs_idx = {}
        self._obs_idx_to_tech_name = {}
        self._last_step_canonical_diag = {}
        self._illegal_sample_count = 0
        self._fallback_end_turn_count = 0
        self._total_action_decisions = 0
        self._validation_mode = os.environ.get("POLYVISION_ACTION_VALIDATION_MODE", "0").lower() in ("1", "true", "yes", "on")
        self._info_mode = str(os.environ.get("POLYVISION_INFO_MODE", "fast")).strip().lower()
        if self._info_mode not in ("fast", "debug", "train"):
            self._info_mode = "fast"
        self._profile_sps_enabled = self._parse_bool_env("POLYVISION_PROFILE_SPS", default=False)
        self._profile_every_n_steps = max(1, self._parse_int_env("POLYVISION_PROFILE_EVERY_N_STEPS", default=1000))
        self._last_step_feature_profile = {}
        self._profile_feature_build_active = False
        self._profile_feature_build_repr_parse_s = 0.0
        self._feature_equiv_check_enabled = self._parse_bool_env("POLYVISION_FEATURE_EQUIV_CHECK", default=False)
        self._legal_summary_equiv_check_enabled = self._parse_bool_env(
            "POLYVISION_LEGAL_SUMMARY_EQUIV_CHECK",
            default=False,
        )
        self._legal_summary_equiv_check_every_n = max(
            1,
            self._parse_int_env("POLYVISION_LEGAL_SUMMARY_EQUIV_CHECK_EVERY_N_STEPS", default=50),
        )
        self._filter_equiv_check_enabled = self._parse_bool_env(
            "POLYVISION_FILTER_EQUIV_CHECK",
            default=False,
        )
        self._filter_equiv_check_every_n = max(
            1,
            self._parse_int_env("POLYVISION_FILTER_EQUIV_CHECK_EVERY_N_STEPS", default=50),
        )
        self._batch_legal_fetch_equiv_check_enabled = self._parse_bool_env(
            "POLYVISION_BATCH_LEGAL_FETCH_EQUIV_CHECK",
            default=False,
        )
        self._batch_legal_fetch_equiv_check_every_n = max(
            1,
            self._parse_int_env("POLYVISION_BATCH_LEGAL_FETCH_EQUIV_CHECK_EVERY_N_STEPS", default=50),
        )
        self._feature_action_meta_cache = {}
        self._last_selected_predicted_population_delta = 0.0
        self._last_selected_predicted_spt_delta = 0.0
        self._last_selected_city_id = None
        self._max_legal_actions = max(
            1,
            self._parse_int_env("POLYVISION_MAX_LEGAL_ACTIONS", default=self.MAX_LEGAL_ACTIONS_DEFAULT),
        )
        self._current_legal_actions = []
        self._current_action_mask = None
        self._current_legal_id_to_raw_index = {}
        self._current_diag = {}
        self._current_raw_valid_actions = 0
        try:
            bootstrap_seed = self._resolve_episode_seed(seed=42)
            bootstrap_level, bootstrap_idx = self._select_level_for_reset(bootstrap_seed)
            self._current_level_file = bootstrap_level
            self._current_level_index = bootstrap_idx
            obs = self.tribes_env.reset(self._current_level_file, bootstrap_seed)
            dims = self._board_dimensions_from_obs(obs)
            if dims is None:
                raise RuntimeError("Cannot infer board dimensions for action catalog.")
            width, height = dims
            vocab = self._load_action_vocab()
            self._catalog = GlobalActionCatalog(
                width=width,
                height=height,
                tech_types=vocab["TECHNOLOGY"],
                train_unit_types=[u for u in vocab["UNIT"] if u not in ("BOAT", "SHIP", "BATTLESHIP", "SUPERUNIT")],
                resource_types=[r for r in vocab["RESOURCE"] if r in ("ANIMAL", "FISH", "WHALES", "FRUIT", "CROPS", "ORE")],
                building_types=vocab["BUILDING"],
                levelup_choices=vocab["CITY_LEVEL_UP"],
            )
            self._tech_name_to_obs_idx = {
                str(name).upper(): int(i)
                for i, name in enumerate(vocab.get("TECHNOLOGY", []))
            }
            self._obs_idx_to_tech_name = {
                int(i): str(name).upper()
                for i, name in enumerate(vocab.get("TECHNOLOGY", []))
            }
            self._catalog_fingerprint = self._catalog.fingerprint()
            self.action_space = gym.spaces.Discrete(self._catalog.total_size)
            if int(self.action_space.n) != int(self._catalog.total_size):
                raise RuntimeError(
                    f"Action space mismatch: action_space.n={self.action_space.n} "
                    f"catalog_total={self._catalog.total_size}"
                )
            
            # Get observation dimensions from actual observation
            obs_array = self._dict_to_array(obs)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=obs_array.shape, 
                dtype=np.float32
            )
        except Exception as e:
            # Fallback to placeholders if initialization fails
            print(f"Warning: Could not initialize environment properly: {e}")
            self.action_space = gym.spaces.Discrete(200)  # safe fallback
            self.observation_space = gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(1000,), 
                dtype=np.float32
            )
    
    def reset(self, seed=None, options=None):
        t_reset_start = time.perf_counter() if self._profile_sps_enabled else None
        t_reset_java = 0.0
        t_reset_opening = 0.0
        t_reset_legal_generation = 0.0
        t_reset_slot_mask_build = 0.0
        t_reset_feature_build = 0.0
        t_reset_info_build = 0.0
        t_reset_obs_flatten = 0.0
        t_reset_sanitize = 0.0
        episode_seed = self._resolve_episode_seed(seed=seed)
        level_file, level_index = self._select_level_for_reset(episode_seed)
        self._current_level_file = level_file
        self._current_level_index = int(level_index)
        self._last_reset_seed = int(episode_seed)
        t_java0 = time.perf_counter() if self._profile_sps_enabled else None
        obs = self.tribes_env.reset(self._current_level_file, self._last_reset_seed)
        if self._profile_sps_enabled:
            t_reset_java += time.perf_counter() - t_java0
        self._episode_index += 1
        self._turn_count = 0
        self._unit_previous_tiles = {}
        t_opening0 = time.perf_counter() if self._profile_sps_enabled else None
        obs = self._apply_bardur_opening(obs)
        if self._profile_sps_enabled:
            t_reset_opening += time.perf_counter() - t_opening0
        self._starting_city_count = self._get_city_count(obs)
        self._last_city_count = self._starting_city_count
        self._moved_on_t0 = False
        self._visible_village_streak_turns = 0
        self._queued_village_capture_unit_ids = set()
        self._initial_visible_tiles = int(self._count_visible_tiles(obs))
        self._episode_fog_tiles_cleared = 0
        self._turn_first_uncaptured_village_visible = None
        self._turn_second_city_captured = None
        self._turn_forestry_researched = None
        self._turn_organization_researched = None
        self._researched_techs_t10 = set()
        self._animals_harvested_t10 = 0
        self._fruit_harvested_t10 = 0
        self._lumber_huts_built_t10 = 0
        self._sawmills_built_t10 = 0
        self._forests_cleared_t10 = 0
        self._terminal_spt_bonus_applied_this_episode = False
        self._initialize_episode_researched_tech_state(obs)
        
        # Log action space info for debugging
        t_legal0 = time.perf_counter() if self._profile_sps_enabled else None
        legal_actions = self.tribes_env.list_actions()
        action_mask, legal_id_to_raw_index, diag = self._build_action_mask_and_mapping(legal_actions, obs=obs)
        if self._profile_sps_enabled:
            t_reset_legal_generation += time.perf_counter() - t_legal0
        self._current_legal_actions = legal_actions
        self._current_action_mask = action_mask
        self._current_legal_id_to_raw_index = legal_id_to_raw_index
        self._current_diag = diag
        self._current_raw_valid_actions = int(len(legal_actions))
        if self.verbose_resets:
            print(
                f"Reset: Available actions = {self._current_raw_valid_actions}, canonicalized = {diag.get('canonicalized_legal_actions', 0)}, "
                f"map={os.path.basename(self._current_level_file)}, seed={self._last_reset_seed}"
            )
        
        # convert your dict obs to numpy array here
        t_info0 = time.perf_counter() if self._profile_sps_enabled else None
        info = {
            "valid_actions": int(np.sum(action_mask)),
            "raw_valid_actions": int(self._current_raw_valid_actions),
            "turn_count": self._turn_count,
            "info_mode": self._info_mode,
            "catalog_version": self.CATALOG_VERSION,
            "canonicalizer_version": self.CANONICALIZER_VERSION,
            "map_width": int(self._catalog.width) if self._catalog is not None else None,
            "map_height": int(self._catalog.height) if self._catalog is not None else None,
            "global_action_space_n": int(self.action_space.n),
            "action_offset_table_hash": self._catalog_fingerprint,
            "max_legal_actions": int(self._max_legal_actions),
        }
        t_slot0 = time.perf_counter() if self._profile_sps_enabled else None
        legal_global_ids_padded, legal_action_valid_mask, legal_action_count = self._build_legal_slot_tensors(action_mask)
        if self._profile_sps_enabled:
            t_reset_slot_mask_build += time.perf_counter() - t_slot0
        t_feat0 = time.perf_counter() if self._profile_sps_enabled else None
        legal_action_features_padded = self._build_legal_action_features_padded(
            legal_global_ids_padded,
            legal_action_valid_mask,
            legal_id_to_raw_index,
            legal_actions,
            obs,
        )
        info["legal_global_ids_padded"] = legal_global_ids_padded
        info["legal_action_valid_mask"] = legal_action_valid_mask
        info["legal_action_count"] = int(legal_action_count)
        info["legal_action_features_padded"] = legal_action_features_padded
        if self._profile_sps_enabled:
            t_reset_feature_build += time.perf_counter() - t_feat0
        info["legal_action_feature_dim"] = int(self.ACTION_FEATURE_DIM)
        info["legal_action_feature_version"] = str(self.LEGAL_ACTION_FEATURE_VERSION)
        info["animals_harvested_t10"] = int(self._animals_harvested_t10)
        info["fruit_harvested_t10"] = int(self._fruit_harvested_t10)
        info["lumber_huts_built_t10"] = int(self._lumber_huts_built_t10)
        info["sawmills_built_t10"] = int(self._sawmills_built_t10)
        info["forests_cleared_t10"] = int(self._forests_cleared_t10)
        if self._is_debug_info_mode():
            info["action_mask"] = action_mask
        info.update(self._diag_for_info(diag))
        if self._is_debug_info_mode():
            info["map_path"] = self._current_level_file
            info["map_id"] = os.path.basename(self._current_level_file)
            info["map_pool_index"] = int(self._current_level_index)
            info["map_pool_size"] = int(self._level_pool_size)
            info["episode_seed"] = int(self._last_reset_seed)
            info["level_selection_mode"] = self._level_selection_mode
            info["initial_visible_tiles"] = int(self._initial_visible_tiles)
        if self._profile_sps_enabled:
            t_reset_info_build += time.perf_counter() - t_info0
            info["profile_env_reset_java_reset_s"] = float(t_reset_java)
            info["profile_env_reset_opening_script_s"] = float(t_reset_opening)
            info["profile_env_reset_legal_generation_s"] = float(t_reset_legal_generation)
            info["profile_env_reset_slot_mask_build_s"] = float(t_reset_slot_mask_build)
            info["profile_env_reset_feature_build_s"] = float(t_reset_feature_build)
            info["profile_env_reset_info_build_s"] = float(t_reset_info_build)
            info["profile_env_reset_total_s"] = float(time.perf_counter() - t_reset_start)
        t_obs0 = time.perf_counter() if self._profile_sps_enabled else None
        obs_arr = self._dict_to_array(obs)
        if self._profile_sps_enabled:
            t_reset_obs_flatten += time.perf_counter() - t_obs0
            info["profile_env_reset_obs_flatten_s"] = float(t_reset_obs_flatten)
        t_sanitize0 = time.perf_counter() if self._profile_sps_enabled else None
        safe_info = self._sanitize_info_for_multiprocessing(info)
        if self._profile_sps_enabled:
            t_reset_sanitize += time.perf_counter() - t_sanitize0
            safe_info["profile_env_reset_sanitize_info_s"] = float(t_reset_sanitize)
        return obs_arr, safe_info
    
    def render(self, **kwargs):
        data = np.array(self.tribes_env.render("rgb_image"))
        return data

    def step(self, action):
        t_step_start = time.perf_counter() if self._profile_sps_enabled else None
        t_step_pre_fast_forward = 0.0
        t_step_pre_legal_generation = 0.0
        t_step_action_decode = 0.0
        t_step_java_apply = 0.0
        t_step_post_fast_forward = 0.0
        t_step_reward_calc = 0.0
        t_step_post_legal_generation = 0.0
        t_step_slot_mask_build = 0.0
        t_step_feature_build = 0.0
        t_step_info_build = 0.0
        t_step_diag_build = 0.0
        t_step_obs_flatten = 0.0
        t_step_sanitize = 0.0
        t_java_apply_action_serialize = 0.0
        t_java_apply_step_call = 0.0
        t_java_apply_response_parse = 0.0
        t_java_apply_obs_fetch = 0.0
        t_java_apply_done_fetch = 0.0
        t_java_apply_scores_fetch = 0.0
        t_java_apply_tick_fetch = 0.0
        t_java_apply_active_fetch = 0.0
        t_java_apply_spt_compute = 0.0
        t_java_apply_active_check_pre = 0.0
        t_java_apply_active_check_post = 0.0
        t_java_apply_solo_ff_pre = 0.0
        t_java_apply_solo_ff_post = 0.0
        t_reward_spt_delta = 0.0
        t_reward_city_capture_delta = 0.0
        t_reward_fog_reward = 0.0
        t_reward_village_reveal = 0.0
        t_reward_move_village_shaping = 0.0
        t_reward_tactical_diag = 0.0
        t_reward_resource_upgrade_checks = 0.0
        t_reward_legal_action_scans = 0.0
        t_reward_board_scans = 0.0
        t_reward_java_calls = 0.0
        t_post_legal_java_fetch = 0.0
        t_post_legal_filter = 0.0
        t_post_legal_canonicalization = 0.0
        t_post_legal_collision_check = 0.0
        t_post_legal_legal_id_build = 0.0
        t_post_legal_mask_build = 0.0
        t_post_legal_diag_build = 0.0
        t_post_legal_filter_allowed_type = 0.0
        t_post_legal_filter_oob_move = 0.0
        t_post_legal_filter_resource_upgrade = 0.0
        t_post_legal_filter_city_count_tactical = 0.0
        t_post_legal_filter_capture_priority = 0.0
        t_post_legal_filter_move_visible_village = 0.0
        t_post_legal_filter_closest_reduce_distance = 0.0
        t_post_legal_filter_early_backtrack = 0.0
        t_post_legal_filter_board_city_village_scans = 0.0
        post_legal_raw_actions_count = 0
        post_legal_allowed_after_base_count = 0
        post_legal_allowed_after_tactical_count = 0
        post_legal_allowed_final_count = 0
        t_post_legal_java_compute_bridge = 0.0
        t_post_legal_java_list_materialize = 0.0
        t_post_legal_java_json_parse = 0.0
        post_legal_java_raw_chars = 0
        t_post_legal_info_attach = 0.0
        t_post_legal_terminal_path = 0.0
        batch_equiv_ref_actions = None
        batch_equiv_ref_action_mask = None
        batch_equiv_ref_legal_id_to_raw_index = None
        batch_equiv_ref_diag = None
        start_obs = getattr(self.tribes_env, "_last_obs", None)
        if not isinstance(start_obs, dict):
            start_obs = {}
        chosen_move_unit_id = None
        chosen_move_dest = None

        forced_pre_end_turns = 0
        t_pre_ff0 = time.perf_counter() if self._profile_sps_enabled else None
        t_active_pre0 = time.perf_counter() if self._profile_sps_enabled else None
        start_active_tribe_id = self._get_active_tribe_id(start_obs)
        if self._profile_sps_enabled:
            t_java_apply_active_check_pre += time.perf_counter() - t_active_pre0
        t_ff_pre0 = time.perf_counter() if self._profile_sps_enabled else None
        if start_active_tribe_id != 0:
            start_obs, forced_pre_end_turns, _ = self._force_non_bardur_turns_to_end(start_obs)
        if self._profile_sps_enabled:
            t_java_apply_solo_ff_pre += time.perf_counter() - t_ff_pre0
        if self._profile_sps_enabled:
            t_step_pre_fast_forward += time.perf_counter() - t_pre_ff0

        legal_actions = self._current_legal_actions if self._current_legal_actions is not None else []
        raw_count = int(self._current_raw_valid_actions)
        action_mask = self._current_action_mask
        legal_id_to_raw_index = self._current_legal_id_to_raw_index
        diag = self._current_diag

        if raw_count == 0 or not legal_actions or action_mask is None or not legal_id_to_raw_index:
            # Recover from any stale cache by rebuilding once.
            t_pre_legal0 = time.perf_counter() if self._profile_sps_enabled else None
            legal_actions = self.tribes_env.list_actions()
            raw_count = len(legal_actions)
            action_mask, legal_id_to_raw_index, diag = self._build_action_mask_and_mapping(legal_actions, obs=start_obs)
            if self._profile_sps_enabled:
                t_step_pre_legal_generation += time.perf_counter() - t_pre_legal0
            self._current_legal_actions = legal_actions
            self._current_action_mask = action_mask
            self._current_legal_id_to_raw_index = legal_id_to_raw_index
            self._current_diag = diag
            self._current_raw_valid_actions = int(raw_count)

        if raw_count == 0:
            raise RuntimeError("No legal actions available from Java environment.")

        sampled_action = int(action)
        t_decode0 = time.perf_counter() if self._profile_sps_enabled else None
        selected_global_id = sampled_action
        illegal_sampled_global_id = False
        fallback_to_end_turn = False
        if selected_global_id in legal_id_to_raw_index:
            selected_raw_action = int(legal_id_to_raw_index[selected_global_id])
        else:
            illegal_sampled_global_id = True
            self._illegal_sample_count += 1
            raise RuntimeError(
                f"Illegal selected global action ID {selected_global_id}; "
                f"valid_count={len(legal_id_to_raw_index)} "
                f"sample_valid_ids={list(sorted(legal_id_to_raw_index.keys()))[:20]}"
            )

        selected_action_type = legal_actions[selected_raw_action].get("type", "UNKNOWN")
        self._total_action_decisions += 1
        if self._profile_sps_enabled:
            t_step_action_decode += time.perf_counter() - t_decode0

        prev_bardur_spt = self._compute_bardur_spt(start_obs)

        deferred_capture_count = 0
        # If END_TURN is selected and we have queued village captures (city_count < 2),
        # execute those captures first, then end the turn.
        if selected_action_type == "END_TURN" and self._queued_village_capture_unit_ids:
            queued_units = set(self._queued_village_capture_unit_ids)
            for _ in range(max(1, len(queued_units))):
                cur_obs = getattr(self.tribes_env, "_last_obs", {})
                if self._get_city_count(cur_obs) >= 2:
                    break
                legal_pre_end = self.tribes_env.list_actions()
                capture_idx = None
                capture_unit_id = None
                for idx, a in enumerate(legal_pre_end):
                    if a.get("type") != "CAPTURE":
                        continue
                    if not self._is_capture_of_village(a, cur_obs):
                        continue
                    unit_id = self._parse_unit_id_from_action_repr(str(a.get("repr", "")))
                    if unit_id is None or unit_id not in queued_units:
                        continue
                    capture_idx = idx
                    capture_unit_id = unit_id
                    break
                if capture_idx is None:
                    break
                self.tribes_env.step(capture_idx)
                deferred_capture_count += 1
                if capture_unit_id in queued_units:
                    queued_units.remove(capture_unit_id)
            # Re-resolve END_TURN after deferred captures since action indices may shift.
            legal_actions = self.tribes_env.list_actions()
            raw_count = len(legal_actions)
            end_turn_idx = next((i for i, a in enumerate(legal_actions) if a.get("type") == "END_TURN"), 0)
            selected_raw_action = end_turn_idx
            selected_action_type = "END_TURN"
            selected_global_id = int(self._catalog.id_end_turn()) if self._catalog is not None else selected_global_id

        selected_action = legal_actions[selected_raw_action]
        selected_eco = self._summarize_action_economy_expectation(selected_action, start_obs)
        selected_city_id_for_delta = selected_eco.get("city_id", None)
        selected_city_pop_before = None
        selected_city_prod_before = None
        if selected_city_id_for_delta is not None:
            selected_city_info = (start_obs.get("city", {}) or {}).get(str(int(selected_city_id_for_delta)), None)
            if isinstance(selected_city_info, dict):
                try:
                    selected_city_pop_before = int(selected_city_info.get("population", 0))
                except Exception:
                    selected_city_pop_before = None
                try:
                    selected_city_prod_before = float(selected_city_info.get("production", 0.0))
                except Exception:
                    selected_city_prod_before = None
        t_reward_resource0 = time.perf_counter() if self._profile_sps_enabled else None
        pre_city_count = int(self._get_city_count(start_obs))
        combat_disabled = "ATTACK" not in self.ALLOWED_ACTION_TYPES
        tactical_window_active = (
            bool(combat_disabled)
            and int(self._turn_count) <= int(self.MAX_TURNS)
            and int(pre_city_count) < 4
        )
        if self._profile_sps_enabled:
            t_reward_resource_upgrade_checks += time.perf_counter() - t_reward_resource0

        t_reward_legal0 = time.perf_counter() if self._profile_sps_enabled else None
        visible_villages_before = self._get_visible_uncaptured_village_positions(start_obs)
        legal_summary = self._build_step_legal_action_summary(
            legal_actions,
            start_obs,
            visible_villages_before=visible_villages_before,
        )
        if (
            self._legal_summary_equiv_check_enabled
            and int(self._total_action_decisions) % int(self._legal_summary_equiv_check_every_n) == 0
        ):
            self._assert_legal_summary_equivalence(
                legal_actions,
                start_obs,
                visible_villages_before,
                legal_summary,
            )
        legal_capture_exists = bool(legal_summary.get("legal_capture_exists", False))
        legal_level_up_exists = bool(legal_summary.get("legal_level_up_exists", False))
        completion_gather_available = bool(legal_summary.get("completion_gather_available", False))
        legal_move_onto_visible_village_exists = bool(
            legal_summary.get("legal_move_onto_visible_village_exists", False)
        )
        legal_useful_move_exists = bool(legal_summary.get("legal_useful_move_exists", False))
        if self._profile_sps_enabled:
            t_reward_legal_action_scans += time.perf_counter() - t_reward_legal0

        selected_move_onto_visible_village = (
            selected_action_type == "MOVE"
            and int(selected_raw_action) in legal_summary.get("move_to_visible_village_raw_indices", set())
        )
        selected_completes_city_upgrade = (
            selected_action_type == "RESOURCE_GATHERING"
            and int(selected_raw_action) in legal_summary.get("completion_gather_raw_indices", set())
        )
        if (
            self._legal_summary_equiv_check_enabled
            and int(self._total_action_decisions) % int(self._legal_summary_equiv_check_every_n) == 0
        ):
            legacy_selected_move_onto_visible_village = (
                selected_action_type == "MOVE"
                and self._is_move_to_visible_uncaptured_village(selected_action, start_obs)
            )
            legacy_selected_completes_city_upgrade = (
                selected_action_type == "RESOURCE_GATHERING"
                and self._resource_gather_action_completes_city_upgrade(selected_action, start_obs)
            )
            if bool(selected_move_onto_visible_village) != bool(legacy_selected_move_onto_visible_village):
                raise RuntimeError("LEGAL_SUMMARY_EQUIV: selected_move_onto_visible_village mismatch")
            if bool(selected_completes_city_upgrade) != bool(legacy_selected_completes_city_upgrade):
                raise RuntimeError("LEGAL_SUMMARY_EQUIV: selected_completes_city_upgrade mismatch")

        tactical_missed_move_onto_visible_village_num = int(
            tactical_window_active
            and legal_move_onto_visible_village_exists
            and selected_action_type == "MOVE"
            and not selected_move_onto_visible_village
        )
        tactical_missed_move_onto_visible_village_den = int(
            tactical_window_active and legal_move_onto_visible_village_exists
        )
        tactical_ignored_capture_num = int(legal_capture_exists and selected_action_type != "CAPTURE")
        tactical_ignored_capture_den = int(legal_capture_exists)
        tactical_end_turn_with_capture_available_num = int(
            legal_capture_exists and selected_action_type == "END_TURN"
        )
        tactical_end_turn_with_capture_available_den = int(legal_capture_exists)
        tactical_end_turn_with_level_up_num = int(legal_level_up_exists and selected_action_type == "END_TURN")
        tactical_end_turn_with_level_up_den = int(legal_level_up_exists)
        tactical_missed_city_upgrade_completion_num = int(
            completion_gather_available
            and selected_action_type == "RESOURCE_GATHERING"
            and not selected_completes_city_upgrade
        )
        tactical_missed_city_upgrade_completion_den = int(completion_gather_available)
        tactical_end_turn_with_useful_move_num = int(
            tactical_window_active
            and legal_useful_move_exists
            and selected_action_type == "END_TURN"
        )
        tactical_end_turn_with_useful_move_den = int(
            tactical_window_active and legal_useful_move_exists
        )

        if selected_action_type == "END_TURN":
            self._turn_count += 1
        elif selected_action_type == "MOVE":
            unit_id, src_x, src_y, dst_x, dst_y = self._extract_move_components(selected_action, start_obs)
            if unit_id is not None:
                chosen_move_unit_id = int(unit_id)
            if dst_x is not None and dst_y is not None:
                chosen_move_dest = (int(dst_x), int(dst_y))
            if unit_id is not None and src_x is not None and src_y is not None:
                self._unit_previous_tiles[int(unit_id)] = (int(src_x), int(src_y))
            if not self._is_move_destination_within_board(selected_action, start_obs):
                raise RuntimeError(
                    f"Selected MOVE action has out-of-bounds destination: "
                    f"global_id={selected_global_id}, raw_idx={selected_raw_action}, repr={selected_action.get('repr', '')}"
                )

        t_java0 = time.perf_counter() if self._profile_sps_enabled else None
        obs, _, done, info = self.tribes_env.step(selected_raw_action)
        if self._profile_sps_enabled:
            t_step_java_apply += time.perf_counter() - t_java0
            java_prof = dict(getattr(self.tribes_env, "_last_step_profile", {}) or {})
            t_java_apply_action_serialize += float(java_prof.get("java_action_serialize_s", 0.0))
            t_java_apply_step_call += float(java_prof.get("java_step_call_s", 0.0))
            t_java_apply_response_parse += float(java_prof.get("java_response_parse_s", 0.0))
            t_java_apply_obs_fetch += float(java_prof.get("java_observation_fetch_s", 0.0))
            t_java_apply_done_fetch += float(java_prof.get("java_done_fetch_s", 0.0))
            t_java_apply_scores_fetch += float(java_prof.get("java_scores_fetch_s", 0.0))
            t_java_apply_tick_fetch += float(java_prof.get("java_tick_fetch_s", 0.0))
            t_java_apply_active_fetch += float(java_prof.get("java_active_tribe_fetch_s", 0.0))
            t_java_apply_spt_compute += float(java_prof.get("java_spt_compute_s", 0.0))
        obs_after_selected = obs
        java_done = bool(done)
        # Update wrapper-level economy and fallback tech state only after Java has
        # successfully applied the selected legal action.
        self._update_economy_counters_from_action(selected_action_type, selected_action)
        actual_population_delta_selected = 0.0
        actual_immediate_spt_delta_selected = 0.0
        if selected_city_id_for_delta is not None:
            city_after = (obs_after_selected.get("city", {}) or {}).get(str(int(selected_city_id_for_delta)), None)
            if isinstance(city_after, dict):
                if selected_city_pop_before is not None:
                    try:
                        actual_population_delta_selected = float(int(city_after.get("population", 0)) - int(selected_city_pop_before))
                    except Exception:
                        actual_population_delta_selected = 0.0
                if selected_city_prod_before is not None:
                    try:
                        actual_immediate_spt_delta_selected = float(
                            float(city_after.get("production", 0.0)) - float(selected_city_prod_before)
                        )
                    except Exception:
                        pass
        self._assert_all_units_in_bounds(obs, context="post_selected_action")

        forced_post_end_turns = 0
        t_post_ff0 = time.perf_counter() if self._profile_sps_enabled else None
        t_active_post0 = time.perf_counter() if self._profile_sps_enabled else None
        post_active_tribe_id = self._get_active_tribe_id(obs)
        if self._profile_sps_enabled:
            t_java_apply_active_check_post += time.perf_counter() - t_active_post0
        t_ff_post0 = time.perf_counter() if self._profile_sps_enabled else None
        if post_active_tribe_id != 0:
            obs, forced_post_end_turns, forced_done = self._force_non_bardur_turns_to_end(obs)
            java_done = java_done or bool(forced_done)
            self._assert_all_units_in_bounds(obs, context="post_forced_end_turns")
        if self._profile_sps_enabled:
            t_java_apply_solo_ff_post += time.perf_counter() - t_ff_post0
        if self._profile_sps_enabled:
            t_step_post_fast_forward += time.perf_counter() - t_post_ff0

        t_reward0 = time.perf_counter() if self._profile_sps_enabled else None
        t_reward_spt0 = time.perf_counter() if self._profile_sps_enabled else None
        current_bardur_spt = self._compute_bardur_spt(obs)
        base_delta_spt = float(current_bardur_spt - prev_bardur_spt)
        delta_spt_reward = float(base_delta_spt)
        if base_delta_spt > 0:
            delta_spt_reward = float(base_delta_spt) * float(self.SPT_INCREASE_REWARD_MULTIPLIER)
        else:
            delta_spt_reward = float(base_delta_spt) * float(self.SPT_NONPOSITIVE_REWARD_MULTIPLIER)
        current_city_count = self._get_city_count(obs)
        if self._profile_sps_enabled:
            t_reward_spt_delta += time.perf_counter() - t_reward_spt0
        reward_adjustment = 0.0

        t_reward_city0 = time.perf_counter() if self._profile_sps_enabled else None
        capture_city_bonus = 0.0
        if current_city_count > self._last_city_count:
            n_new_cities = current_city_count - self._last_city_count
            # +3.0 per newly captured city in this step, capped at +6.0.
            capture_city_bonus = min(
                self.CAPTURE_CITY_BONUS_MAX,
                max(0, n_new_cities) * self.CAPTURE_CITY_BONUS_MIN,
            )
        if self._turn_second_city_captured is None and int(current_city_count) >= 2:
            self._turn_second_city_captured = int(self._turn_count)
        reward_adjustment += capture_city_bonus
        if self._profile_sps_enabled:
            t_reward_city_capture_delta += time.perf_counter() - t_reward_city0

        reveal_uncaptured_village_reward = 0.0
        move_closer_to_visible_village_reward = 0.0
        move_onto_village_reward = 0.0
        village_breadcrumb_reward = 0.0
        fog_clearance_reward = 0.0
        useless_move_fog_miss_penalty = 0.0
        move_onto_visible_neutral_village_tactical_reward = 0.0
        fog_tiles_cleared = 0
        newly_revealed_tiles = 0
        unit_had_any_legal_fog_revealing_move = False

        t_reward_village0 = time.perf_counter() if self._profile_sps_enabled else None
        visible_villages_before = self._get_visible_uncaptured_village_positions(start_obs)
        visible_villages_after_selected = self._get_visible_uncaptured_village_positions(obs_after_selected)
        newly_revealed_villages = visible_villages_after_selected - visible_villages_before
        visible_uncaptured_villages_before = int(len(visible_villages_before))
        uncaptured_villages_count = int(len(visible_villages_after_selected))
        captured_villages_t10 = max(0, int(current_city_count) - int(self._starting_city_count))
        capturable_villages_total = max(0, int(captured_villages_t10) + int(uncaptured_villages_count))
        village_capture_pct_t10 = (
            (100.0 * float(captured_villages_t10) / float(capturable_villages_total))
            if int(capturable_villages_total) > 0
            else 0.0
        )
        if (
            self._turn_first_uncaptured_village_visible is None
            and len(visible_villages_after_selected) > 0
        ):
            self._turn_first_uncaptured_village_visible = int(self._turn_count)
        if len(newly_revealed_villages) > 0:
            reveal_uncaptured_village_reward = self.REVEAL_UNCAPTURED_VILLAGE_REWARD * float(len(newly_revealed_villages))
        if self._profile_sps_enabled:
            t_reward_village_reveal += time.perf_counter() - t_reward_village0

        visible_uncaptured_village = self._has_visible_uncaptured_village(obs)
        unit_on_visible_uncaptured_village = self._has_unit_on_visible_uncaptured_village(obs)

        t_reward_legal1 = time.perf_counter() if self._profile_sps_enabled else None
        units_on_neutral_village_capture_illegal_all = self._owned_units_on_visible_uncaptured_village_without_capture_from_sets(
            start_obs,
            visible_villages_before,
            legal_summary.get("capture_unit_ids", set()),
        )
        units_on_neutral_village_capture_illegal = set()
        hold_neutral_village_end_turn_reward = 0.0
        move_off_neutral_village_capture_illegal_penalty = 0.0
        if (
            combat_disabled
            and int(self._turn_count) <= int(self.MAX_TURNS)
            and int(pre_city_count) < 2
        ):
            units_on_neutral_village_capture_illegal = set(units_on_neutral_village_capture_illegal_all)
        if self._profile_sps_enabled:
            t_reward_legal_action_scans += time.perf_counter() - t_reward_legal1

        if selected_action_type == "MOVE":
            t_reward_fog0 = time.perf_counter() if self._profile_sps_enabled else None
            vis_before = self._count_visible_tiles(start_obs)
            vis_after = self._count_visible_tiles(obs_after_selected)
            fog_tiles_cleared = max(0, vis_after - vis_before)
            newly_revealed_tiles = int(fog_tiles_cleared)
            if fog_tiles_cleared > 0:
                fog_clearance_reward = min(
                    self.FOG_CLEAR_REWARD_MAX_TILES,
                    int(fog_tiles_cleared),
                ) * self.FOG_CLEAR_REWARD_PER_TILE
                self._episode_fog_tiles_cleared += int(fog_tiles_cleared)
            if chosen_move_unit_id is not None:
                unit_had_any_legal_fog_revealing_move = bool(
                    int(chosen_move_unit_id) in legal_summary.get("unit_ids_with_adjacent_fog_move", set())
                )
                if (
                    self._legal_summary_equiv_check_enabled
                    and int(self._total_action_decisions) % int(self._legal_summary_equiv_check_every_n) == 0
                ):
                    legacy_unit_had_any_legal_fog_revealing_move = self._unit_had_any_legal_fog_revealing_move(
                        legal_actions,
                        start_obs,
                        chosen_move_unit_id,
                    )
                    if bool(unit_had_any_legal_fog_revealing_move) != bool(legacy_unit_had_any_legal_fog_revealing_move):
                        raise RuntimeError("LEGAL_SUMMARY_EQUIV: unit_had_any_legal_fog_revealing_move mismatch")
            if self._profile_sps_enabled:
                t_reward_fog_reward += time.perf_counter() - t_reward_fog0

            t_reward_shape0 = time.perf_counter() if self._profile_sps_enabled else None
            moved_unit_before = self._get_unit_position(start_obs, chosen_move_unit_id) if chosen_move_unit_id is not None else None
            moved_unit_after = self._get_unit_position(obs_after_selected, chosen_move_unit_id) if chosen_move_unit_id is not None else None
            dist_before = self._min_manhattan_distance(moved_unit_before, visible_villages_before)
            dist_after = self._min_manhattan_distance(moved_unit_after, visible_villages_after_selected)
            if dist_before is not None and dist_after is not None and dist_after < dist_before:
                move_closer_to_visible_village_reward = self.MOVE_CLOSER_TO_VISIBLE_VILLAGE_REWARD

            if moved_unit_after is not None and moved_unit_after in visible_villages_after_selected:
                move_onto_village_reward = self.MOVE_ONTO_VILLAGE_REWARD

            if (
                combat_disabled
                and int(self._turn_count) <= int(self.MAX_TURNS)
                and int(current_city_count) < 2
                and int(visible_uncaptured_villages_before) == 0
                and int(newly_revealed_tiles) == 0
                and bool(unit_had_any_legal_fog_revealing_move)
            ):
                useless_move_fog_miss_penalty = -float(self.USELESS_MOVE_FOG_MISS_PENALTY)
            if self._profile_sps_enabled:
                t_reward_move_village_shaping += time.perf_counter() - t_reward_shape0

        if selected_action_type == "END_TURN":
            self._visible_village_streak_turns = 0
            if unit_on_visible_uncaptured_village:
                village_breadcrumb_reward = self.VILLAGE_BREADCRUMB_REWARD
            if units_on_neutral_village_capture_illegal:
                hold_neutral_village_end_turn_reward = float(self.HOLD_NEUTRAL_VILLAGE_END_TURN_REWARD)

        if (
            selected_action_type == "MOVE"
            and chosen_move_unit_id is not None
            and int(chosen_move_unit_id) in units_on_neutral_village_capture_illegal
        ):
            move_off_neutral_village_capture_illegal_penalty = -float(
                self.MOVE_OFF_NEUTRAL_VILLAGE_WHEN_CAPTURE_ILLEGAL_PENALTY
            )

        t_reward_tact0 = time.perf_counter() if self._profile_sps_enabled else None
        if tactical_window_active and legal_move_onto_visible_village_exists:
            if selected_move_onto_visible_village:
                move_onto_visible_neutral_village_tactical_reward = float(
                    self.MOVE_ONTO_VISIBLE_NEUTRAL_VILLAGE_REWARD
                )
            elif selected_action_type == "MOVE":
                move_onto_visible_neutral_village_tactical_reward = -float(
                    self.MOVE_MISS_VISIBLE_NEUTRAL_VILLAGE_PENALTY
                )
        if self._profile_sps_enabled:
            t_reward_tactical_diag += time.perf_counter() - t_reward_tact0

        tactical_move_off_neutral_village_before_capture_num = int(
            selected_action_type == "MOVE"
            and chosen_move_unit_id is not None
            and int(chosen_move_unit_id) in units_on_neutral_village_capture_illegal_all
        )
        tactical_move_off_neutral_village_before_capture_den = int(
            len(units_on_neutral_village_capture_illegal_all) > 0
        )

        reward_adjustment += reveal_uncaptured_village_reward
        reward_adjustment += move_closer_to_visible_village_reward
        reward_adjustment += move_onto_village_reward
        reward_adjustment += village_breadcrumb_reward
        reward_adjustment += hold_neutral_village_end_turn_reward
        reward_adjustment += move_off_neutral_village_capture_illegal_penalty
        reward_adjustment += fog_clearance_reward
        reward_adjustment += useless_move_fog_miss_penalty
        reward_adjustment += move_onto_visible_neutral_village_tactical_reward

        # Phase 1 override: ignore Java terminal state and control horizon purely in Python.
        terminated = False
        # Turn index starts at 2 after the hardcoded Bardur opening in reset().
        # Truncate only after finishing Bardur Turn 10 (i.e., when count advances to 11).
        internal_t10_truncation = bool(self._turn_count > self.MAX_TURNS)
        truncated = bool(internal_t10_truncation)

        terminal_spt_bonus = 0.0
        terminal_spt_base_component = 0.0
        terminal_spt_over_10_component = 0.0
        terminal_spt_over_15_component = 0.0
        terminal_final_spt = None
        if (
            bool(self._terminal_spt_reward_enabled)
            and bool(internal_t10_truncation)
            and not bool(self._terminal_spt_bonus_applied_this_episode)
        ):
            terminal_final_spt = float(current_bardur_spt)
            terminal_spt_base_component = float(self._terminal_spt_base_weight) * float(terminal_final_spt)
            terminal_spt_over_10_component = (
                float(self._terminal_spt_over_10_weight) * max(0.0, float(terminal_final_spt) - 10.0)
            )
            terminal_spt_over_15_component = (
                float(self._terminal_spt_over_15_weight) * max(0.0, float(terminal_final_spt) - 15.0)
            )
            terminal_spt_bonus = (
                float(terminal_spt_base_component)
                + float(terminal_spt_over_10_component)
                + float(terminal_spt_over_15_component)
            )
            self._terminal_spt_bonus_applied_this_episode = True

        reward = float(delta_spt_reward) + reward_adjustment + float(terminal_spt_bonus)
        if self._profile_sps_enabled:
            t_reward_board_scans += (t_reward_village_reveal + t_reward_fog_reward)
            t_reward_java_calls += 0.0
            t_step_reward_calc += time.perf_counter() - t_reward0

        # Build the next-state legal mask and mapping diagnostics (for policy step t+1).
        t_post_legal0 = time.perf_counter() if self._profile_sps_enabled else None
        t_post_fetch0 = time.perf_counter() if self._profile_sps_enabled else None
        post_legal_actions = self.tribes_env.list_actions()
        if self._profile_sps_enabled:
            t_post_legal_java_fetch += time.perf_counter() - t_post_fetch0
            _la_prof = dict(getattr(self.tribes_env, "_last_list_actions_profile", {}) or {})
            t_post_legal_java_compute_bridge += float(_la_prof.get("java_compute_bridge_s", 0.0))
            t_post_legal_java_list_materialize += float(_la_prof.get("python_list_materialize_s", 0.0))
            t_post_legal_java_json_parse += float(_la_prof.get("python_json_parse_s", 0.0))
            post_legal_java_raw_chars += int(_la_prof.get("raw_action_total_chars", 0))
        post_raw_count = len(post_legal_actions)
        post_legal_raw_actions_count = int(post_raw_count)
        post_mask_profile = {} if self._profile_sps_enabled else None
        post_action_mask, post_legal_id_to_raw_index, post_diag = self._build_action_mask_and_mapping(
            post_legal_actions,
            obs=obs,
            profile=post_mask_profile,
        )
        if self._profile_sps_enabled and isinstance(post_mask_profile, dict):
            t_post_legal_filter += float(post_mask_profile.get("filter_allowed_indices_s", 0.0))
            t_post_legal_canonicalization += float(post_mask_profile.get("canonicalize_global_id_s", 0.0))
            t_post_legal_collision_check += float(post_mask_profile.get("collision_check_s", 0.0))
            t_post_legal_legal_id_build += float(post_mask_profile.get("legal_id_list_construct_s", 0.0))
            t_post_legal_mask_build += float(post_mask_profile.get("mask_build_s", 0.0))
            t_post_legal_diag_build += float(post_mask_profile.get("diag_build_s", 0.0))
            t_post_legal_filter_allowed_type += float(post_mask_profile.get("filter_allowed_type_s", 0.0))
            t_post_legal_filter_oob_move += float(post_mask_profile.get("filter_oob_move_s", 0.0))
            t_post_legal_filter_resource_upgrade += float(post_mask_profile.get("filter_resource_upgrade_s", 0.0))
            t_post_legal_filter_city_count_tactical += float(post_mask_profile.get("filter_city_count_tactical_s", 0.0))
            t_post_legal_filter_capture_priority += float(post_mask_profile.get("filter_capture_priority_s", 0.0))
            t_post_legal_filter_move_visible_village += float(post_mask_profile.get("filter_move_visible_village_s", 0.0))
            t_post_legal_filter_closest_reduce_distance += float(post_mask_profile.get("filter_closest_reduce_distance_s", 0.0))
            t_post_legal_filter_early_backtrack += float(post_mask_profile.get("filter_early_backtrack_s", 0.0))
            t_post_legal_filter_board_city_village_scans += float(post_mask_profile.get("filter_board_city_village_scans_s", 0.0))
            post_legal_allowed_after_base_count = int(post_mask_profile.get("allowed_after_base_count", 0))
            post_legal_allowed_after_tactical_count = int(post_mask_profile.get("allowed_after_tactical_count", 0))
            post_legal_allowed_final_count = int(post_mask_profile.get("allowed_final_count", 0))
        run_batch_fetch_equiv_check = bool(
            self._batch_legal_fetch_equiv_check_enabled
            and int(self._total_action_decisions) % int(self._batch_legal_fetch_equiv_check_every_n) == 0
        )
        if run_batch_fetch_equiv_check:
            legacy_fetch = getattr(self.tribes_env, "_list_actions_legacy", None)
            batch_fetch = getattr(self.tribes_env, "_list_actions_batch", None)
            use_batch = bool(getattr(self.tribes_env, "_batch_legal_action_fetch_enabled", False))
            if callable(legacy_fetch) and callable(batch_fetch):
                batch_equiv_ref_actions = legacy_fetch(None) if use_batch else batch_fetch(None)
                if post_legal_actions != batch_equiv_ref_actions:
                    raise RuntimeError(
                        "BATCH_LEGAL_FETCH_EQUIV: raw post-step actions mismatch "
                        f"current_count={len(post_legal_actions)} ref_count={len(batch_equiv_ref_actions)}"
                    )
                batch_equiv_ref_action_mask, batch_equiv_ref_legal_id_to_raw_index, batch_equiv_ref_diag = (
                    self._build_action_mask_and_mapping(batch_equiv_ref_actions, obs=obs, profile=None)
                )
                if not np.array_equal(np.asarray(post_action_mask), np.asarray(batch_equiv_ref_action_mask)):
                    raise RuntimeError("BATCH_LEGAL_FETCH_EQUIV: post action mask mismatch")
                if dict(post_legal_id_to_raw_index) != dict(batch_equiv_ref_legal_id_to_raw_index):
                    raise RuntimeError("BATCH_LEGAL_FETCH_EQUIV: legal_id_to_raw_index mismatch")
                if list(post_legal_id_to_raw_index.keys()) != list(batch_equiv_ref_legal_id_to_raw_index.keys()):
                    raise RuntimeError("BATCH_LEGAL_FETCH_EQUIV: ordered canonical IDs mismatch")
                if dict(post_diag.get("legal_action_count_by_type", {})) != dict(
                    batch_equiv_ref_diag.get("legal_action_count_by_type", {})
                ):
                    raise RuntimeError("BATCH_LEGAL_FETCH_EQUIV: legal_action_count_by_type mismatch")
        if self._profile_sps_enabled:
            t_step_post_legal_generation += time.perf_counter() - t_post_legal0
            if bool(terminated) or bool(truncated):
                t_post_legal_terminal_path += time.perf_counter() - t_post_legal0
        self._current_legal_actions = post_legal_actions
        self._current_action_mask = post_action_mask
        self._current_legal_id_to_raw_index = post_legal_id_to_raw_index
        self._current_diag = post_diag
        self._current_raw_valid_actions = int(post_raw_count)

        t_info0 = time.perf_counter() if self._profile_sps_enabled else None
        t_info_scalar = 0.0
        t_info_diag = 0.0
        t_info_legal_summary = 0.0
        t_info_large_payload = 0.0
        t_info_action_repr = 0.0
        t_info_terminal_episode = 0.0
        t_info_metric_packaging = 0.0
        t_info_ws = 0.0
        if self._profile_sps_enabled:
            t_info_ws = time.perf_counter()
        info["valid_actions"] = int(np.sum(post_action_mask))
        info["raw_valid_actions"] = int(post_raw_count)
        info["info_mode"] = self._info_mode
        if self._is_debug_info_mode():
            info["action_mask"] = post_action_mask
        else:
            info["legal_global_ids"] = np.flatnonzero(post_action_mask).astype(np.int32).tolist()
        if self._profile_sps_enabled:
            t_info_legal_summary += time.perf_counter() - t_info_ws
            t_info_ws = time.perf_counter()
        info["sampled_action"] = sampled_action
        info["selected_global_id"] = int(selected_global_id)
        info["selected_raw_action"] = selected_raw_action
        info["selected_raw_java_index"] = int(selected_raw_action)
        info["selected_action_type"] = selected_action_type
        info["illegal_sampled_global_id"] = bool(illegal_sampled_global_id)
        info["fallback_to_end_turn"] = bool(fallback_to_end_turn)
        info["total_action_decisions"] = int(self._total_action_decisions)
        info["illegal_sample_count"] = int(self._illegal_sample_count)
        info["fallback_end_turn_count"] = int(self._fallback_end_turn_count)
        info["illegal_sample_rate"] = float(self._illegal_sample_count / max(1, self._total_action_decisions))
        info["fallback_end_turn_rate"] = float(self._fallback_end_turn_count / max(1, self._total_action_decisions))
        info["catalog_version"] = self.CATALOG_VERSION
        info["canonicalizer_version"] = self.CANONICALIZER_VERSION
        info["map_width"] = int(self._catalog.width) if self._catalog is not None else None
        info["map_height"] = int(self._catalog.height) if self._catalog is not None else None
        info["global_action_space_n"] = int(self.action_space.n)
        info["action_offset_table_hash"] = self._catalog_fingerprint
        info["max_legal_actions"] = int(self._max_legal_actions)
        info["turn_count"] = self._turn_count
        info["city_count"] = current_city_count
        info["avg_city_level"] = float(self._get_avg_city_level(obs, tribe_id=0))
        if self._profile_sps_enabled:
            t_info_scalar += time.perf_counter() - t_info_ws
            t_info_ws = time.perf_counter()
        effective_techs = self._get_effective_researched_techs(obs, tribe_id=0)
        info["techs_researched"] = int(len(effective_techs))
        info["forestry_researched"] = int("FORESTRY" in effective_techs)
        info["organization_researched"] = int("ORGANIZATION" in effective_techs)
        info["turn_first_uncaptured_village_visible"] = (
            int(self._turn_first_uncaptured_village_visible)
            if self._turn_first_uncaptured_village_visible is not None
            else -1
        )
        info["turn_second_city_captured"] = (
            int(self._turn_second_city_captured)
            if self._turn_second_city_captured is not None
            else -1
        )
        info["turn_forestry_researched"] = (
            int(self._turn_forestry_researched)
            if self._turn_forestry_researched is not None
            else -1
        )
        info["turn_organization_researched"] = (
            int(self._turn_organization_researched)
            if self._turn_organization_researched is not None
            else -1
        )
        if self._profile_sps_enabled:
            t_info_terminal_episode += time.perf_counter() - t_info_ws
            t_info_ws = time.perf_counter()
        info["fog_tiles_cleared_total"] = int(self._episode_fog_tiles_cleared)
        info["delta_spt"] = float(base_delta_spt)
        info["delta_spt_reward"] = float(delta_spt_reward)
        info["terminal_spt_reward_enabled"] = bool(self._terminal_spt_reward_enabled)
        info["terminal_spt_bonus"] = float(terminal_spt_bonus)
        info["terminal_final_spt"] = float(terminal_final_spt) if terminal_final_spt is not None else None
        info["terminal_spt_base_component"] = float(terminal_spt_base_component)
        info["terminal_spt_over_10_component"] = float(terminal_spt_over_10_component)
        info["terminal_spt_over_15_component"] = float(terminal_spt_over_15_component)
        info["resource_gather_upgrade_filter_enabled"] = bool(self._resource_gather_upgrade_filter_enabled)
        if self._profile_sps_enabled:
            t_info_terminal_episode += time.perf_counter() - t_info_ws
            t_info_ws = time.perf_counter()
        info["spt"] = float(current_bardur_spt)
        info["reward"] = float(reward)
        info["turn"] = int(self._turn_count)
        info["unit_count"] = int(self._get_owned_unit_count(obs))
        info["stars"] = int(self._get_tribe_stars(obs, tribe_id=0))
        info["selected_expected_population_delta"] = float(selected_eco.get("expected_population_delta", 0.0))
        info["selected_actual_population_delta"] = float(actual_population_delta_selected)
        info["selected_population_delta_abs_error"] = float(
            abs(float(selected_eco.get("expected_population_delta", 0.0)) - float(actual_population_delta_selected))
        )
        info["selected_expected_immediate_spt_delta"] = float(selected_eco.get("expected_immediate_spt_delta", 0.0))
        info["selected_actual_immediate_spt_delta"] = float(actual_immediate_spt_delta_selected)
        info["selected_immediate_spt_delta_abs_error"] = float(
            abs(float(selected_eco.get("expected_immediate_spt_delta", 0.0)) - float(actual_immediate_spt_delta_selected))
        )
        info["selected_expected_city_id"] = int(selected_city_id_for_delta) if selected_city_id_for_delta is not None else -1
        if self._profile_sps_enabled:
            t_info_metric_packaging += time.perf_counter() - t_info_ws
        t_slot0 = time.perf_counter() if self._profile_sps_enabled else None
        post_legal_global_ids_padded, post_legal_action_valid_mask, post_legal_action_count = self._build_legal_slot_tensors(post_action_mask)
        if self._profile_sps_enabled:
            t_step_slot_mask_build += time.perf_counter() - t_slot0
        t_feat0 = time.perf_counter() if self._profile_sps_enabled else None
        post_legal_action_features_padded = self._build_legal_action_features_padded(
            post_legal_global_ids_padded,
            post_legal_action_valid_mask,
            post_legal_id_to_raw_index,
            post_legal_actions,
            obs,
        )
        if (
            run_batch_fetch_equiv_check
            and batch_equiv_ref_actions is not None
            and batch_equiv_ref_action_mask is not None
            and batch_equiv_ref_legal_id_to_raw_index is not None
        ):
            ref_ids_padded, ref_valid_mask, ref_count = self._build_legal_slot_tensors(batch_equiv_ref_action_mask)
            if not np.array_equal(np.asarray(post_legal_global_ids_padded), np.asarray(ref_ids_padded)):
                raise RuntimeError("BATCH_LEGAL_FETCH_EQUIV: legal_global_ids_padded mismatch")
            if not np.array_equal(np.asarray(post_legal_action_valid_mask), np.asarray(ref_valid_mask)):
                raise RuntimeError("BATCH_LEGAL_FETCH_EQUIV: legal_action_valid_mask mismatch")
            if int(post_legal_action_count) != int(ref_count):
                raise RuntimeError("BATCH_LEGAL_FETCH_EQUIV: legal_action_count mismatch")
            ref_features = self._build_legal_action_features_padded(
                ref_ids_padded,
                ref_valid_mask,
                batch_equiv_ref_legal_id_to_raw_index,
                batch_equiv_ref_actions,
                obs,
            )
            if np.asarray(ref_features).shape != np.asarray(post_legal_action_features_padded).shape:
                raise RuntimeError("BATCH_LEGAL_FETCH_EQUIV: legal feature shape mismatch")
            if not np.allclose(
                np.asarray(post_legal_action_features_padded, dtype=np.float32),
                np.asarray(ref_features, dtype=np.float32),
                rtol=1e-6,
                atol=1e-6,
            ):
                raise RuntimeError("BATCH_LEGAL_FETCH_EQUIV: legal feature values mismatch")
        t_large0 = time.perf_counter() if self._profile_sps_enabled else None
        t_attach0 = time.perf_counter() if self._profile_sps_enabled else None
        info["legal_global_ids_padded"] = post_legal_global_ids_padded
        info["legal_action_valid_mask"] = post_legal_action_valid_mask
        info["legal_action_count"] = int(post_legal_action_count)
        info["legal_action_features_padded"] = post_legal_action_features_padded
        if self._profile_sps_enabled:
            t_post_legal_info_attach += time.perf_counter() - t_attach0
            t_step_feature_build += time.perf_counter() - t_feat0
            t_info_large_payload += time.perf_counter() - t_large0
            t_info_ws = time.perf_counter()
        info["legal_action_feature_dim"] = int(self.ACTION_FEATURE_DIM)
        info["legal_action_feature_version"] = str(self.LEGAL_ACTION_FEATURE_VERSION)
        info["animals_harvested_t10"] = int(self._animals_harvested_t10)
        info["fruit_harvested_t10"] = int(self._fruit_harvested_t10)
        info["lumber_huts_built_t10"] = int(self._lumber_huts_built_t10)
        info["sawmills_built_t10"] = int(self._sawmills_built_t10)
        info["forests_cleared_t10"] = int(self._forests_cleared_t10)
        info["captured_villages_t10"] = int(captured_villages_t10)
        info["capturable_villages_total"] = int(capturable_villages_total)
        info["village_capture_pct_t10"] = float(village_capture_pct_t10)
        if self._profile_sps_enabled:
            t_info_terminal_episode += time.perf_counter() - t_info_ws
            t_info_ws = time.perf_counter()
        info["tm_missed_move_onto_visible_village_num"] = int(tactical_missed_move_onto_visible_village_num)
        info["tm_move_onto_visible_village_available_den"] = int(tactical_missed_move_onto_visible_village_den)
        info["tm_ignored_capture_num"] = int(tactical_ignored_capture_num)
        info["tm_capture_available_den"] = int(tactical_ignored_capture_den)
        info["tm_end_turn_with_capture_available_num"] = int(tactical_end_turn_with_capture_available_num)
        info["tm_end_turn_with_capture_available_den"] = int(tactical_end_turn_with_capture_available_den)
        info["tm_end_turn_with_level_up_num"] = int(tactical_end_turn_with_level_up_num)
        info["tm_level_up_available_den"] = int(tactical_end_turn_with_level_up_den)
        info["tm_missed_city_upgrade_completion_num"] = int(tactical_missed_city_upgrade_completion_num)
        info["tm_completion_gather_available_den"] = int(tactical_missed_city_upgrade_completion_den)
        info["tm_move_off_neutral_village_before_capture_num"] = int(tactical_move_off_neutral_village_before_capture_num)
        info["tm_unit_on_neutral_village_capture_illegal_den"] = int(tactical_move_off_neutral_village_before_capture_den)
        info["tm_end_turn_with_useful_move_num"] = int(tactical_end_turn_with_useful_move_num)
        info["tm_useful_move_available_den"] = int(tactical_end_turn_with_useful_move_den)
        if self._profile_sps_enabled:
            t_info_diag += time.perf_counter() - t_info_ws
            t_info_ws = time.perf_counter()
        info.update(self._diag_for_info(post_diag))
        if self._profile_sps_enabled:
            t_info_legal_summary += time.perf_counter() - t_info_ws

        t_diag0 = time.perf_counter() if self._profile_sps_enabled else None
        if self._is_debug_info_mode() and selected_action_type == "MOVE" and chosen_move_unit_id is not None and chosen_move_dest is not None:
            actual_pos = self._get_unit_position(obs, chosen_move_unit_id)
            dims = self._board_dimensions_from_obs(obs)
            in_bounds = False
            if actual_pos is not None and dims is not None:
                width, height = dims
                in_bounds = (
                    0 <= int(actual_pos[0]) < int(width)
                    and 0 <= int(actual_pos[1]) < int(height)
                )
            info["move_verify_unit_id"] = int(chosen_move_unit_id)
            info["move_verify_requested_x"] = int(chosen_move_dest[0])
            info["move_verify_requested_y"] = int(chosen_move_dest[1])
            info["move_verify_actual_x"] = int(actual_pos[0]) if actual_pos is not None else -1
            info["move_verify_actual_y"] = int(actual_pos[1]) if actual_pos is not None else -1
            info["move_verify_dest_match"] = bool(actual_pos == chosen_move_dest if actual_pos is not None else False)
            info["move_verify_actual_in_bounds"] = bool(in_bounds)

            if actual_pos is not None and not in_bounds:
                raise RuntimeError(
                    f"Unit moved out of board bounds: unit={chosen_move_unit_id} "
                    f"requested={chosen_move_dest} actual={actual_pos} dims={dims}"
                )
        if self._is_debug_info_mode():
            info["pre_step_raw_valid_actions"] = int(raw_count)
            info["pre_step_mask_ones"] = int(np.sum(action_mask))
            info["reward_adjustment"] = float(reward_adjustment)
            info["starting_city_count"] = int(self._starting_city_count)
            info["visible_uncaptured_village"] = bool(visible_uncaptured_village)
            info["unit_on_visible_uncaptured_village"] = bool(unit_on_visible_uncaptured_village)
            info["visible_village_streak_turns"] = int(self._visible_village_streak_turns)
            info["moved_on_t0"] = bool(self._moved_on_t0)
            info["reward_capture_city_bonus"] = float(capture_city_bonus)
            info["reward_second_village_delay_penalty"] = 0.0
            info["reward_village_breadcrumb"] = float(village_breadcrumb_reward)
            info["reward_fog_clearance"] = float(fog_clearance_reward)
            info["reward_reveal_uncaptured_village"] = float(reveal_uncaptured_village_reward)
            info["reward_move_closer_to_visible_village"] = float(move_closer_to_visible_village_reward)
            info["reward_move_onto_village"] = float(move_onto_village_reward)
            info["reward_hold_neutral_village_end_turn"] = float(hold_neutral_village_end_turn_reward)
            info["reward_move_off_neutral_village_when_capture_illegal"] = float(
                move_off_neutral_village_capture_illegal_penalty
            )
            info["reward_useless_move_fog_miss_penalty"] = float(useless_move_fog_miss_penalty)
            info["reward_move_onto_visible_neutral_village_tactical"] = float(
                move_onto_visible_neutral_village_tactical_reward
            )
            info["newly_revealed_uncaptured_villages"] = int(len(newly_revealed_villages))
            info["visible_uncaptured_villages_before_move"] = int(visible_uncaptured_villages_before)
            info["newly_revealed_tiles"] = int(newly_revealed_tiles)
            info["unit_on_neutral_village_capture_illegal"] = bool(len(units_on_neutral_village_capture_illegal) > 0)
            info["unit_on_neutral_village_capture_illegal_count"] = int(len(units_on_neutral_village_capture_illegal))
            info["end_turn_while_unit_on_neutral_village"] = bool(
                selected_action_type == "END_TURN" and len(units_on_neutral_village_capture_illegal) > 0
            )
            info["moved_off_neutral_village_before_capture"] = bool(
                selected_action_type == "MOVE"
                and chosen_move_unit_id is not None
                and int(chosen_move_unit_id) in units_on_neutral_village_capture_illegal
            )
            info["unit_had_any_legal_fog_revealing_move"] = bool(unit_had_any_legal_fog_revealing_move)
            info["fog_tiles_cleared_step"] = int(fog_tiles_cleared)
            info["visible_tiles"] = int(self._count_visible_tiles(obs))
            info["initial_visible_tiles"] = int(self._initial_visible_tiles)
            info["java_done"] = bool(java_done)
            info["terminated_overridden"] = True
            info["forced_pre_end_turns"] = int(forced_pre_end_turns)
            info["forced_post_end_turns"] = int(forced_post_end_turns)
            info["deferred_village_captures_before_end_turn"] = int(deferred_capture_count)
            info["queued_village_capture_unit_ids"] = list(sorted(int(u) for u in self._queued_village_capture_unit_ids))
            info["map_path"] = self._current_level_file
            info["map_id"] = os.path.basename(self._current_level_file)
            info["map_pool_index"] = int(self._current_level_index)
            info["map_pool_size"] = int(self._level_pool_size)
            info["episode_seed"] = int(self._last_reset_seed) if self._last_reset_seed is not None else None
            info["level_selection_mode"] = self._level_selection_mode
            info["activeTribeID"] = int(self._get_active_tribe_id(obs))
        if self._profile_sps_enabled:
            t_step_diag_build += time.perf_counter() - t_diag0
            t_step_info_build += time.perf_counter() - t_info0
        self._last_city_count = current_city_count
        if selected_action_type == "MOVE" and self._turn_count == 0:
            self._moved_on_t0 = True
        if current_city_count >= 2:
            self._queued_village_capture_unit_ids = set()

        if self._profile_sps_enabled:
            info["profile_env_step_pre_fast_forward_s"] = float(t_step_pre_fast_forward)
            info["profile_env_step_pre_legal_generation_s"] = float(t_step_pre_legal_generation)
            info["profile_env_step_action_decode_s"] = float(t_step_action_decode)
            info["profile_env_step_java_apply_s"] = float(t_step_java_apply)
            info["profile_env_step_post_fast_forward_s"] = float(t_step_post_fast_forward)
            info["profile_env_step_reward_calc_s"] = float(t_step_reward_calc)
            info["profile_env_step_post_legal_generation_s"] = float(t_step_post_legal_generation)
            info["profile_env_step_slot_mask_build_s"] = float(t_step_slot_mask_build)
            info["profile_env_step_feature_build_s"] = float(t_step_feature_build)
            info["profile_env_step_info_build_s"] = float(t_step_info_build)
            info["profile_env_step_diag_build_s"] = float(t_step_diag_build)
            info["profile_env_step_info_build_scalar_s"] = float(t_info_scalar)
            info["profile_env_step_info_build_diag_counter_s"] = float(t_info_diag)
            info["profile_env_step_info_build_legal_summary_s"] = float(t_info_legal_summary)
            info["profile_env_step_info_build_large_payload_s"] = float(t_info_large_payload)
            info["profile_env_step_info_build_action_repr_s"] = float(t_info_action_repr)
            info["profile_env_step_info_build_terminal_episode_s"] = float(t_info_terminal_episode)
            info["profile_env_step_info_build_metric_packaging_s"] = float(t_info_metric_packaging)
            _feat_prof = dict(self._last_step_feature_profile) if isinstance(self._last_step_feature_profile, dict) else {}
            info["profile_env_step_feature_alloc_zero_fill_s"] = float(_feat_prof.get("feature_alloc_zero_fill_s", 0.0))
            info["profile_env_step_feature_mask_construct_s"] = float(_feat_prof.get("feature_mask_construct_s", 0.0))
            info["profile_env_step_feature_precompute_s"] = float(_feat_prof.get("feature_step_precompute_s", 0.0))
            info["profile_env_step_feature_loop_iteration_s"] = float(_feat_prof.get("feature_loop_iteration_s", 0.0))
            info["profile_env_step_feature_gid_decode_s"] = float(_feat_prof.get("feature_gid_decode_s", 0.0))
            info["profile_env_step_feature_canonical_lookup_s"] = float(_feat_prof.get("feature_canonical_lookup_s", 0.0))
            info["profile_env_step_feature_static_metadata_s"] = float(_feat_prof.get("feature_static_metadata_extract_s", 0.0))
            info["profile_env_step_feature_dynamic_state_s"] = float(_feat_prof.get("feature_dynamic_state_extract_s", 0.0))
            info["profile_env_step_feature_padding_write_s"] = float(_feat_prof.get("feature_padding_write_s", 0.0))
            info["profile_env_step_feature_action_repr_s"] = float(_feat_prof.get("feature_repr_string_s", 0.0))
            info["profile_env_step_feature_java_calls_s"] = float(_feat_prof.get("feature_java_calls_s", 0.0))
            info["profile_env_step_reward_spt_delta_s"] = float(t_reward_spt_delta)
            info["profile_env_step_reward_city_capture_delta_s"] = float(t_reward_city_capture_delta)
            info["profile_env_step_reward_fog_calc_s"] = float(t_reward_fog_reward)
            info["profile_env_step_reward_village_reveal_s"] = float(t_reward_village_reveal)
            info["profile_env_step_reward_move_village_shaping_s"] = float(t_reward_move_village_shaping)
            info["profile_env_step_reward_tactical_diag_s"] = float(t_reward_tactical_diag)
            info["profile_env_step_reward_resource_upgrade_checks_s"] = float(t_reward_resource_upgrade_checks)
            info["profile_env_step_reward_legal_action_scans_s"] = float(t_reward_legal_action_scans)
            info["profile_env_step_reward_board_scans_s"] = float(t_reward_board_scans)
            info["profile_env_step_reward_java_calls_s"] = float(t_reward_java_calls)
            info["profile_env_step_post_legal_java_fetch_s"] = float(t_post_legal_java_fetch)
            info["profile_env_step_post_legal_java_compute_bridge_s"] = float(t_post_legal_java_compute_bridge)
            info["profile_env_step_post_legal_java_list_materialize_s"] = float(t_post_legal_java_list_materialize)
            info["profile_env_step_post_legal_java_json_parse_s"] = float(t_post_legal_java_json_parse)
            info["profile_env_step_post_legal_action_filter_s"] = float(t_post_legal_filter)
            info["profile_env_step_post_legal_filter_allowed_type_s"] = float(t_post_legal_filter_allowed_type)
            info["profile_env_step_post_legal_filter_oob_move_s"] = float(t_post_legal_filter_oob_move)
            info["profile_env_step_post_legal_filter_resource_upgrade_s"] = float(t_post_legal_filter_resource_upgrade)
            info["profile_env_step_post_legal_filter_city_count_tactical_s"] = float(t_post_legal_filter_city_count_tactical)
            info["profile_env_step_post_legal_filter_capture_priority_s"] = float(t_post_legal_filter_capture_priority)
            info["profile_env_step_post_legal_filter_move_visible_village_s"] = float(t_post_legal_filter_move_visible_village)
            info["profile_env_step_post_legal_filter_closest_reduce_distance_s"] = float(t_post_legal_filter_closest_reduce_distance)
            info["profile_env_step_post_legal_filter_early_backtrack_s"] = float(t_post_legal_filter_early_backtrack)
            info["profile_env_step_post_legal_filter_board_city_village_scans_s"] = float(t_post_legal_filter_board_city_village_scans)
            info["profile_env_step_post_legal_canonicalize_s"] = float(t_post_legal_canonicalization)
            info["profile_env_step_post_legal_collision_checks_s"] = float(t_post_legal_collision_check)
            info["profile_env_step_post_legal_id_list_build_s"] = float(t_post_legal_legal_id_build)
            info["profile_env_step_post_legal_mask_build_s"] = float(t_post_legal_mask_build)
            info["profile_env_step_post_legal_diag_build_s"] = float(t_post_legal_diag_build)
            info["profile_env_step_post_legal_padding_mask_s"] = float(t_step_slot_mask_build)
            info["profile_env_step_post_legal_feature_build_s"] = float(t_step_feature_build)
            info["profile_env_step_post_legal_info_attach_s"] = float(t_post_legal_info_attach)
            info["profile_env_step_post_legal_terminal_path_s"] = float(t_post_legal_terminal_path)
            info["profile_env_step_post_legal_raw_actions_count"] = int(post_legal_raw_actions_count)
            info["profile_env_step_post_legal_canonical_actions_count"] = int(len(post_legal_id_to_raw_index))
            info["profile_env_step_post_legal_allowed_after_base_count"] = int(post_legal_allowed_after_base_count)
            info["profile_env_step_post_legal_allowed_after_tactical_count"] = int(post_legal_allowed_after_tactical_count)
            info["profile_env_step_post_legal_allowed_final_count"] = int(post_legal_allowed_final_count)
            info["profile_env_step_post_legal_raw_action_total_chars"] = int(post_legal_java_raw_chars)
            info["profile_env_step_java_apply_action_serialize_s"] = float(t_java_apply_action_serialize)
            info["profile_env_step_java_apply_call_s"] = float(t_java_apply_step_call)
            info["profile_env_step_java_apply_response_parse_s"] = float(t_java_apply_response_parse)
            info["profile_env_step_java_apply_obs_fetch_s"] = float(t_java_apply_obs_fetch)
            info["profile_env_step_java_apply_done_fetch_s"] = float(t_java_apply_done_fetch)
            info["profile_env_step_java_apply_scores_fetch_s"] = float(t_java_apply_scores_fetch)
            info["profile_env_step_java_apply_tick_fetch_s"] = float(t_java_apply_tick_fetch)
            info["profile_env_step_java_apply_active_fetch_s"] = float(t_java_apply_active_fetch)
            info["profile_env_step_java_apply_spt_compute_s"] = float(t_java_apply_spt_compute)
            info["profile_env_step_java_apply_active_check_pre_s"] = float(t_java_apply_active_check_pre)
            info["profile_env_step_java_apply_active_check_post_s"] = float(t_java_apply_active_check_post)
            info["profile_env_step_java_apply_solo_fast_forward_pre_s"] = float(t_java_apply_solo_ff_pre)
            info["profile_env_step_java_apply_solo_fast_forward_post_s"] = float(t_java_apply_solo_ff_post)
        t_obs0 = time.perf_counter() if self._profile_sps_enabled else None
        obs_arr = self._dict_to_array(obs)
        if self._profile_sps_enabled:
            t_step_obs_flatten += time.perf_counter() - t_obs0
            info["profile_env_step_obs_flatten_s"] = float(t_step_obs_flatten)
            info["profile_env_step_total_s"] = float(time.perf_counter() - t_step_start)
        t_sanitize0 = time.perf_counter() if self._profile_sps_enabled else None
        safe_info = self._sanitize_info_for_multiprocessing(info)
        if self._profile_sps_enabled:
            t_step_sanitize += time.perf_counter() - t_sanitize0
            safe_info["profile_env_step_sanitize_info_s"] = float(t_step_sanitize)
        return obs_arr, reward, terminated, truncated, safe_info

    def _is_debug_info_mode(self):
        return self._info_mode == "debug"

    def _is_train_info_mode(self):
        return self._info_mode == "train"

    def _diag_for_info(self, diag):
        if not isinstance(diag, dict):
            return {}
        if self._is_debug_info_mode():
            return dict(diag)
        if self._is_train_info_mode():
            keep = (
                "legal_actions_total",
                "canonicalized_legal_actions",
                "uncanonicalized_legal_actions",
                "duplicate_global_id_collisions",
                "mask_ones",
                "unique_legal_global_ids",
            )
            return {k: diag[k] for k in keep if k in diag}
        # Fast mode keeps only fields required by trainer safety gates and validators.
        keep = (
            "legal_actions_total",
            "canonicalized_legal_actions",
            "uncanonicalized_legal_actions",
            "duplicate_global_id_collisions",
            "mask_ones",
            "unique_legal_global_ids",
            "legal_action_count_by_type",
        )
        return {k: diag[k] for k in keep if k in diag}

    def _build_legal_slot_tensors(self, action_mask):
        if action_mask is None:
            raise RuntimeError("action_mask is required to build legal slot tensors.")
        legal_ids = np.flatnonzero(np.asarray(action_mask, dtype=np.int8)).astype(np.int32)
        if legal_ids.size > int(self._max_legal_actions):
            raise RuntimeError(
                f"legal_action_count={int(legal_ids.size)} exceeded max_legal_actions={int(self._max_legal_actions)}"
            )
        if legal_ids.size > 0:
            unique_size = np.unique(legal_ids).size
            if int(unique_size) != int(legal_ids.size):
                raise RuntimeError(
                    f"Duplicate legal global IDs detected: count={int(legal_ids.size)} unique={int(unique_size)}"
                )
        padded = np.zeros((int(self._max_legal_actions),), dtype=np.int32)
        valid_mask = np.zeros((int(self._max_legal_actions),), dtype=np.bool_)
        n = int(legal_ids.size)
        if n > 0:
            padded[:n] = legal_ids
            valid_mask[:n] = True
        return padded, valid_mask, n

    def _build_legal_action_features_sparse(self, legal_global_ids, legal_id_to_raw_index, legal_actions, obs):
        ids_arr = np.asarray(legal_global_ids, dtype=np.int64).reshape(-1)
        out = np.zeros((int(ids_arr.shape[0]), int(self.ACTION_FEATURE_DIM)), dtype=np.float32)
        if not isinstance(legal_id_to_raw_index, dict) or not isinstance(legal_actions, list):
            return out
        for i, gid in enumerate(ids_arr):
            raw_idx = legal_id_to_raw_index.get(int(gid), None)
            if raw_idx is None:
                continue
            if int(raw_idx) < 0 or int(raw_idx) >= len(legal_actions):
                continue
            out[i, :] = self._compute_legal_action_feature_vector(legal_actions[int(raw_idx)], obs)
        return out

    def _update_economy_counters_from_action(self, action_type, action):
        if int(self._turn_count) > int(self.MAX_TURNS):
            return
        a_type = str(action_type or "").upper()
        if a_type == "RESEARCH_TECH":
            tech_type = self._action_str(action, "tech_type", None)
            if tech_type is None:
                tech_type = self._parse_tech_type_from_action_repr(str(action.get("repr", "")))
            tech_type = str(tech_type or "").upper()
            if tech_type:
                self._researched_techs_t10.add(tech_type)
                if tech_type == "FORESTRY" and self._turn_forestry_researched is None:
                    self._turn_forestry_researched = int(self._turn_count)
                if tech_type == "ORGANIZATION" and self._turn_organization_researched is None:
                    self._turn_organization_researched = int(self._turn_count)
            return
        if a_type == "RESOURCE_GATHERING":
            resource_type = self._action_str(action, "resource_type", None)
            if resource_type is None:
                resource_type = self._parse_resource_type_from_action_repr(str(action.get("repr", "")))
            resource_type = str(resource_type or "").upper()
            if resource_type == "ANIMAL":
                self._animals_harvested_t10 += 1
            elif resource_type == "FRUIT":
                self._fruit_harvested_t10 += 1
        elif a_type == "BUILD":
            building_type = self._action_str(action, "building_type", None)
            if building_type is None:
                building_type = self._parse_building_type_from_action_repr(str(action.get("repr", "")))
            building_type = str(building_type or "").upper()
            if building_type == "LUMBER_HUT":
                self._lumber_huts_built_t10 += 1
            elif building_type == "SAWMILL":
                self._sawmills_built_t10 += 1
        elif a_type == "CLEAR_FOREST":
            self._forests_cleared_t10 += 1

    def _parse_tech_type_from_action_repr(self, action_repr):
        if not isinstance(action_repr, str):
            return None
        m = re.search(r"RESEARCH_TECH.*:\s*([A-Z_]+)\s*$", action_repr.strip(), flags=re.IGNORECASE)
        if m is None:
            return None
        return str(m.group(1)).upper()

    def _build_legal_action_features_padded_reference(self, legal_global_ids_padded, legal_action_valid_mask, legal_id_to_raw_index, legal_actions, obs):
        features = np.zeros((int(self._max_legal_actions), int(self.ACTION_FEATURE_DIM)), dtype=np.float32)
        if legal_global_ids_padded is None or legal_action_valid_mask is None:
            return features
        if not isinstance(legal_id_to_raw_index, dict) or not isinstance(legal_actions, list):
            return features

        ids = np.asarray(legal_global_ids_padded, dtype=np.int64).reshape(-1)
        valid = np.asarray(legal_action_valid_mask, dtype=bool).reshape(-1)
        max_slots = min(ids.shape[0], valid.shape[0], int(self._max_legal_actions))
        for slot in range(max_slots):
            if not bool(valid[slot]):
                continue
            gid = int(ids[slot])
            raw_idx = legal_id_to_raw_index.get(gid, None)
            if raw_idx is None:
                continue
            if int(raw_idx) < 0 or int(raw_idx) >= len(legal_actions):
                continue
            action = legal_actions[int(raw_idx)]
            features[slot, :] = self._compute_legal_action_feature_vector_reference(action, obs)
        return features

    def _build_feature_step_context(self, obs):
        ctx = {
            "dims": self._board_dimensions_from_obs(obs),
            "terrain_arr": None,
            "fog_mask": None,
            "mountain_mask": None,
            "adj_fog_counts": None,
            "reveal_range1": None,
            "reveal_range2": None,
            "visible_villages": set(),
            "visible_village_mask": None,
            "has_visible_village": False,
            "dist_to_visible_village": None,
            "city_bounds_mask": None,
            "capital_pos": self._get_capital_position(obs, tribe_id=0),
            "dist_norm": 1.0,
            "unit_info": {},
        }
        dims = ctx["dims"]
        if dims is None:
            return ctx
        width, height = int(dims[0]), int(dims[1])
        if width <= 0 or height <= 0:
            return ctx

        board = obs.get("board", {}) if isinstance(obs, dict) else {}
        terrain_raw = board.get("terrain", []) if isinstance(board, dict) else []
        try:
            terrain_arr = np.asarray(terrain_raw, dtype=np.int16)
        except Exception:
            terrain_arr = np.full((width, height), -1, dtype=np.int16)
        if terrain_arr.shape != (width, height):
            fixed = np.full((width, height), -1, dtype=np.int16)
            try:
                wx = min(width, terrain_arr.shape[0])
                hy = min(height, terrain_arr.shape[1]) if terrain_arr.ndim == 2 else 0
                if hy > 0:
                    fixed[:wx, :hy] = terrain_arr[:wx, :hy]
            except Exception:
                pass
            terrain_arr = fixed
        ctx["terrain_arr"] = terrain_arr
        fog_mask = terrain_arr == 7
        ctx["fog_mask"] = fog_mask
        ctx["mountain_mask"] = terrain_arr == 3
        ctx["dist_norm"] = float(max(width, height)) if max(width, height) > 0 else 1.0

        adj = np.zeros((width, height), dtype=np.int16)
        for x in range(width):
            x0 = max(0, x - 1)
            x1 = min(width, x + 2)
            for y in range(height):
                y0 = max(0, y - 1)
                y1 = min(height, y + 2)
                local = fog_mask[x0:x1, y0:y1]
                adj[x, y] = int(np.count_nonzero(local)) - (1 if fog_mask[x, y] else 0)
        ctx["adj_fog_counts"] = adj

        reveal1 = np.zeros((width, height), dtype=np.int16)
        reveal2 = np.zeros((width, height), dtype=np.int16)
        for x in range(width):
            x01 = max(0, x - 1)
            x11 = min(width, x + 2)
            x02 = max(0, x - 2)
            x12 = min(width, x + 3)
            for y in range(height):
                y01 = max(0, y - 1)
                y11 = min(height, y + 2)
                y02 = max(0, y - 2)
                y12 = min(height, y + 3)
                reveal1[x, y] = int(np.count_nonzero(fog_mask[x01:x11, y01:y11]))
                reveal2[x, y] = int(np.count_nonzero(fog_mask[x02:x12, y02:y12]))
        ctx["reveal_range1"] = reveal1
        ctx["reveal_range2"] = reveal2

        visible_villages = self._get_visible_uncaptured_village_positions(obs)
        ctx["visible_villages"] = visible_villages
        ctx["has_visible_village"] = len(visible_villages) > 0
        village_mask = np.zeros((width, height), dtype=np.bool_)
        for vx, vy in visible_villages:
            if 0 <= int(vx) < width and 0 <= int(vy) < height:
                village_mask[int(vx), int(vy)] = True
        ctx["visible_village_mask"] = village_mask

        city_bounds = np.zeros((width, height), dtype=np.bool_)
        city_map = obs.get("city", {}) if isinstance(obs, dict) else {}
        if isinstance(city_map, dict):
            for city in city_map.values():
                if not isinstance(city, dict):
                    continue
                try:
                    if int(city.get("tribeID", -1)) != 0:
                        continue
                    cx = int(city.get("x", -1))
                    cy = int(city.get("y", -1))
                    bound = int(city.get("bound", -1))
                except Exception:
                    continue
                if cx < 0 or cy < 0 or bound < 0:
                    continue
                x0 = max(0, cx - bound)
                x1 = min(width, cx + bound + 1)
                y0 = max(0, cy - bound)
                y1 = min(height, cy + bound + 1)
                for x in range(x0, x1):
                    for y in range(y0, y1):
                        if max(abs(x - cx), abs(y - cy)) <= bound:
                            city_bounds[x, y] = True
        ctx["city_bounds_mask"] = city_bounds

        if ctx["has_visible_village"]:
            dist_map = np.full((width, height), -1, dtype=np.int16)
            villages = list(visible_villages)
            for x in range(width):
                for y in range(height):
                    best = None
                    for vx, vy in villages:
                        d = abs(int(x) - int(vx)) + abs(int(y) - int(vy))
                        if best is None or d < best:
                            best = d
                    dist_map[x, y] = int(best) if best is not None else -1
            ctx["dist_to_visible_village"] = dist_map

        units = obs.get("unit", {}) if isinstance(obs, dict) else {}
        if isinstance(units, dict):
            unit_info = {}
            for uid_s, unit in units.items():
                if not isinstance(unit, dict):
                    continue
                try:
                    uid = int(uid_s)
                except Exception:
                    continue
                try:
                    ux = int(unit.get("x", -1))
                    uy = int(unit.get("y", -1))
                except Exception:
                    ux, uy = -1, -1
                try:
                    utype = int(unit.get("type", -1))
                except Exception:
                    utype = -1
                unit_info[uid] = {
                    "x": ux,
                    "y": uy,
                    "type": utype,
                }
            ctx["unit_info"] = unit_info
        return ctx

    def _get_cached_move_action_meta(self, gid, action):
        gid_i = int(gid)
        a_type = str(action.get("type", "UNKNOWN")).upper()
        sig = (
            a_type,
            action.get("unit_id", None),
            action.get("src_x", None),
            action.get("src_y", None),
            action.get("dst_x", None),
            action.get("dst_y", None),
            action.get("repr", ""),
        )
        entry = self._feature_action_meta_cache.get(gid_i, None)
        if isinstance(entry, dict) and entry.get("sig", None) == sig:
            return entry.get("meta", {})

        unit_id = self._action_int(action, "unit_id", None)
        src_x = self._action_int(action, "src_x", None)
        src_y = self._action_int(action, "src_y", None)
        dst_x = self._action_int(action, "dst_x", None)
        dst_y = self._action_int(action, "dst_y", None)
        if (dst_x is None or dst_y is None) or (unit_id is None):
            t_repr_parse0 = time.perf_counter() if self._profile_feature_build_active else None
            parsed_move = self._parse_move_unit_and_dest_from_action_repr(str(action.get("repr", "")))
            if self._profile_feature_build_active:
                self._profile_feature_build_repr_parse_s += time.perf_counter() - t_repr_parse0
            if parsed_move is not None:
                if unit_id is None:
                    unit_id = int(parsed_move[0])
                if dst_x is None:
                    dst_x = int(parsed_move[1])
                if dst_y is None:
                    dst_y = int(parsed_move[2])
        meta = {
            "unit_id": unit_id,
            "src_x": src_x,
            "src_y": src_y,
            "dst_x": dst_x,
            "dst_y": dst_y,
        }
        self._feature_action_meta_cache[gid_i] = {"sig": sig, "meta": meta}
        return meta

    def _compute_legal_action_feature_vector_cached(self, action, obs, ctx, gid):
        feat = np.zeros((int(self.ACTION_FEATURE_DIM),), dtype=np.float32)
        a_type = str(action.get("type", "UNKNOWN")).upper()
        is_move = a_type == "MOVE"
        feat[0] = 1.0 if is_move else 0.0

        if is_move:
            meta = self._get_cached_move_action_meta(gid, action)
            unit_id = meta.get("unit_id", None)
            src_x = meta.get("src_x", None)
            src_y = meta.get("src_y", None)
            dst_x = meta.get("dst_x", None)
            dst_y = meta.get("dst_y", None)

            unit_info = ctx.get("unit_info", {})
            uinfo = unit_info.get(int(unit_id), None) if unit_id is not None else None
            if (src_x is None or src_y is None) and isinstance(uinfo, dict):
                sx = int(uinfo.get("x", -1))
                sy = int(uinfo.get("y", -1))
                if sx >= 0 and sy >= 0:
                    src_x, src_y = sx, sy

            dims = ctx.get("dims", None)
            if dst_x is not None and dst_y is not None and dims is not None:
                width, height = int(dims[0]), int(dims[1])
                dx = int(dst_x)
                dy = int(dst_y)
                in_bounds_dst = 0 <= dx < width and 0 <= dy < height
                if in_bounds_dst:
                    reveal1 = ctx.get("reveal_range1", None)
                    reveal2 = ctx.get("reveal_range2", None)
                    mountain_mask = ctx.get("mountain_mask", None)
                    unit_type = int(uinfo.get("type", -1)) if isinstance(uinfo, dict) else -1
                    if unit_type == 10:
                        revealed = int(reveal2[dx, dy]) if reveal2 is not None else 0
                    else:
                        on_mountain = bool(mountain_mask[dx, dy]) if mountain_mask is not None else False
                        revealed = int(reveal2[dx, dy]) if (on_mountain and reveal2 is not None) else int(reveal1[dx, dy] if reveal1 is not None else 0)
                    feat[1] = float(min(float(self.REVEAL_CLIP), max(0.0, float(revealed))) / max(1.0, float(self.REVEAL_CLIP)))

                    adj_fog = ctx.get("adj_fog_counts", None)
                    adj_after = int(adj_fog[dx, dy]) if adj_fog is not None else 0
                    feat[2] = float(min(float(self.ADJ_FOG_MAX), max(0.0, float(adj_after))) / max(1.0, float(self.ADJ_FOG_MAX)))

                    adj_before = 0
                    if src_x is not None and src_y is not None:
                        sx = int(src_x)
                        sy = int(src_y)
                        if 0 <= sx < width and 0 <= sy < height and adj_fog is not None:
                            adj_before = int(adj_fog[sx, sy])
                    adj_delta = float(adj_after - adj_before)
                    feat[3] = float(np.clip(adj_delta / max(1.0, float(self.ADJ_FOG_MAX)), -1.0, 1.0))
                    feat[4] = 1.0 if revealed == 0 else 0.0

                    has_visible_village = bool(ctx.get("has_visible_village", False))
                    feat[6] = 1.0 if has_visible_village else 0.0
                    if has_visible_village:
                        village_mask = ctx.get("visible_village_mask", None)
                        if village_mask is not None and village_mask[dx, dy]:
                            feat[5] = 1.0
                        dist_map = ctx.get("dist_to_visible_village", None)
                        if src_x is not None and src_y is not None and dist_map is not None:
                            sx = int(src_x)
                            sy = int(src_y)
                            if 0 <= sx < width and 0 <= sy < height:
                                dist_before = int(dist_map[sx, sy])
                                dist_after = int(dist_map[dx, dy])
                                if dist_before >= 0 and dist_after >= 0:
                                    dist_norm = float(ctx.get("dist_norm", 1.0))
                                    if dist_norm <= 0:
                                        dist_norm = 1.0
                                    feat[7] = float(np.clip((float(dist_before) - float(dist_after)) / dist_norm, -1.0, 1.0))

                    if unit_id is not None:
                        prev_tile = self._unit_previous_tiles.get(int(unit_id), None)
                        if prev_tile is not None and (dx, dy) == (int(prev_tile[0]), int(prev_tile[1])):
                            feat[8] = 1.0
                        if isinstance(uinfo, dict):
                            feat[11] = 1.0 if int(uinfo.get("type", -1)) == 0 else 0.0

                    city_bounds_mask = ctx.get("city_bounds_mask", None)
                    if city_bounds_mask is not None and city_bounds_mask[dx, dy]:
                        feat[9] = 1.0

                    capital_pos = ctx.get("capital_pos", None)
                    if capital_pos is not None and src_x is not None and src_y is not None:
                        sx = int(src_x)
                        sy = int(src_y)
                        before = abs(sx - int(capital_pos[0])) + abs(sy - int(capital_pos[1]))
                        after = abs(dx - int(capital_pos[0])) + abs(dy - int(capital_pos[1]))
                        dist_norm = float(ctx.get("dist_norm", 1.0))
                        if dist_norm <= 0:
                            dist_norm = 1.0
                        feat[10] = float(np.clip((float(after) - float(before)) / dist_norm, -1.0, 1.0))

        feat[12] = 1.0 if a_type == "END_TURN" else 0.0
        feat[13] = 1.0 if a_type == "CAPTURE" else 0.0
        feat[14] = 1.0 if a_type in ("SPAWN", "TRAIN") else 0.0
        feat[15] = 1.0 if a_type == "RESEARCH_TECH" else 0.0
        feat[16] = 1.0 if a_type == "RESOURCE_GATHERING" else 0.0
        feat[17] = 1.0 if a_type == "LEVEL_UP" else 0.0
        feat[18] = 1.0 if a_type == "BUILD" else 0.0
        feat[19] = 1.0 if a_type == "CLEAR_FOREST" else 0.0
        feat[20] = 1.0 if a_type == "GROW_FOREST" else 0.0
        known = is_move or bool(feat[12] or feat[13] or feat[14] or feat[15] or feat[16] or feat[17] or feat[18] or feat[19] or feat[20])
        feat[21] = 0.0 if known else 1.0

        tech_type = self._resolve_action_tech_type(action)
        if tech_type and self._catalog is not None:
            feat[22] = self._normalize_vocab_index(
                self._catalog.tech_to_idx.get(tech_type, -1),
                len(self._catalog.tech_types),
            )
        feat[23] = 1.0 if tech_type == "ORGANIZATION" else 0.0
        feat[24] = 1.0 if tech_type == "FORESTRY" else 0.0

        resource_type = self._resolve_action_resource_type(action)
        if resource_type and self._catalog is not None:
            feat[25] = self._normalize_vocab_index(
                self._catalog.resource_to_idx.get(resource_type, -1),
                len(self._catalog.resource_types),
            )
        feat[26] = 1.0 if resource_type == "ANIMAL" else 0.0
        feat[27] = 1.0 if resource_type == "FRUIT" else 0.0
        feat[28] = 1.0 if resource_type == "FISH" else 0.0
        feat[29] = 1.0 if resource_type == "CROPS" else 0.0
        feat[30] = 1.0 if resource_type == "ORE" else 0.0

        building_type = self._resolve_action_building_type(action)
        if building_type and self._catalog is not None:
            feat[31] = self._normalize_vocab_index(
                self._catalog.building_to_idx.get(building_type, -1),
                len(self._catalog.building_types),
            )
        feat[32] = 1.0 if building_type == "LUMBER_HUT" else 0.0
        feat[33] = 1.0 if building_type == "SAWMILL" else 0.0

        levelup_choice = self._resolve_action_levelup_choice(action)
        if levelup_choice and self._catalog is not None:
            feat[34] = self._normalize_vocab_index(
                self._catalog.levelup_to_idx.get(levelup_choice, -1),
                len(self._catalog.levelup_choices),
            )
        feat[35] = 1.0 if levelup_choice == "WORKSHOP" else 0.0

        eco = self._summarize_action_economy_expectation(action, obs)
        feat[36] = float(np.clip(float(eco["expected_population_delta"]) / 2.0, -1.0, 1.0))
        feat[37] = float(np.clip(float(eco["expected_immediate_spt_delta"]) / 5.0, -1.0, 1.0))
        feat[38] = 1.0 if bool(eco["makes_level_up_available"]) else 0.0
        feat[39] = 1.0 if bool(eco["is_level_up_claim"]) else 0.0
        feat[40] = float(np.clip(float(eco["progress_before"]), 0.0, 1.0))
        feat[41] = 1.0 if bool(eco["ready_before"]) else 0.0
        return feat

    def _build_legal_action_features_padded(self, legal_global_ids_padded, legal_action_valid_mask, legal_id_to_raw_index, legal_actions, obs):
        profile = {
            "feature_alloc_zero_fill_s": 0.0,
            "feature_mask_construct_s": 0.0,
            "feature_step_precompute_s": 0.0,
            "feature_loop_iteration_s": 0.0,
            "feature_gid_decode_s": 0.0,
            "feature_canonical_lookup_s": 0.0,
            "feature_static_metadata_extract_s": 0.0,
            "feature_dynamic_state_extract_s": 0.0,
            "feature_padding_write_s": 0.0,
            "feature_repr_string_s": 0.0,
            "feature_java_calls_s": 0.0,
        }
        t_alloc0 = time.perf_counter() if self._profile_sps_enabled else None
        features = np.zeros((int(self._max_legal_actions), int(self.ACTION_FEATURE_DIM)), dtype=np.float32)
        if self._profile_sps_enabled:
            profile["feature_alloc_zero_fill_s"] += time.perf_counter() - t_alloc0
        self._last_step_feature_profile = profile
        if legal_global_ids_padded is None or legal_action_valid_mask is None:
            return features
        if not isinstance(legal_id_to_raw_index, dict) or not isinstance(legal_actions, list):
            return features

        t_mask0 = time.perf_counter() if self._profile_sps_enabled else None
        ids = np.asarray(legal_global_ids_padded, dtype=np.int64).reshape(-1)
        valid = np.asarray(legal_action_valid_mask, dtype=bool).reshape(-1)
        if self._profile_sps_enabled:
            profile["feature_mask_construct_s"] += time.perf_counter() - t_mask0
        max_slots = min(ids.shape[0], valid.shape[0], int(self._max_legal_actions))

        t_ctx0 = time.perf_counter() if self._profile_sps_enabled else None
        ctx = self._build_feature_step_context(obs)
        if self._profile_sps_enabled:
            profile["feature_step_precompute_s"] += time.perf_counter() - t_ctx0
        self._profile_feature_build_active = bool(self._profile_sps_enabled)
        self._profile_feature_build_repr_parse_s = 0.0
        for slot in range(max_slots):
            t_loop0 = time.perf_counter() if self._profile_sps_enabled else None
            if not bool(valid[slot]):
                if self._profile_sps_enabled:
                    profile["feature_loop_iteration_s"] += time.perf_counter() - t_loop0
                continue
            t_gid0 = time.perf_counter() if self._profile_sps_enabled else None
            gid = int(ids[slot])
            if self._profile_sps_enabled:
                profile["feature_gid_decode_s"] += time.perf_counter() - t_gid0
            t_lookup0 = time.perf_counter() if self._profile_sps_enabled else None
            raw_idx = legal_id_to_raw_index.get(gid, None)
            if self._profile_sps_enabled:
                profile["feature_canonical_lookup_s"] += time.perf_counter() - t_lookup0
            if raw_idx is None:
                if self._profile_sps_enabled:
                    profile["feature_loop_iteration_s"] += time.perf_counter() - t_loop0
                continue
            if int(raw_idx) < 0 or int(raw_idx) >= len(legal_actions):
                if self._profile_sps_enabled:
                    profile["feature_loop_iteration_s"] += time.perf_counter() - t_loop0
                continue
            t_static0 = time.perf_counter() if self._profile_sps_enabled else None
            action = legal_actions[int(raw_idx)]
            _a_type = str(action.get("type", "UNKNOWN")).upper()
            if self._profile_sps_enabled:
                profile["feature_static_metadata_extract_s"] += time.perf_counter() - t_static0
            t_dynamic0 = time.perf_counter() if self._profile_sps_enabled else None
            vec = self._compute_legal_action_feature_vector_cached(action, obs, ctx, gid)
            if self._profile_sps_enabled:
                profile["feature_dynamic_state_extract_s"] += time.perf_counter() - t_dynamic0
            t_write0 = time.perf_counter() if self._profile_sps_enabled else None
            features[slot, :] = vec
            if self._profile_sps_enabled:
                profile["feature_padding_write_s"] += time.perf_counter() - t_write0
                profile["feature_loop_iteration_s"] += time.perf_counter() - t_loop0
        self._profile_feature_build_active = False
        if self._profile_sps_enabled:
            profile["feature_repr_string_s"] += float(self._profile_feature_build_repr_parse_s)
            self._last_step_feature_profile = profile

        if self._feature_equiv_check_enabled:
            if self._current_action_mask is not None:
                ids_check, mask_check, _count_check = self._build_legal_slot_tensors(self._current_action_mask)
                if np.asarray(ids_check).shape != np.asarray(legal_global_ids_padded).shape:
                    raise RuntimeError(
                        f"feature_equiv_ids_shape_mismatch: "
                        f"recomputed={np.asarray(ids_check).shape} input={np.asarray(legal_global_ids_padded).shape}"
                    )
                if np.asarray(mask_check).shape != np.asarray(legal_action_valid_mask).shape:
                    raise RuntimeError(
                        f"feature_equiv_mask_shape_mismatch: "
                        f"recomputed={np.asarray(mask_check).shape} input={np.asarray(legal_action_valid_mask).shape}"
                    )
                if not np.array_equal(np.asarray(ids_check), np.asarray(legal_global_ids_padded)):
                    raise RuntimeError("feature_equiv_ids_value_mismatch between recomputed and input legal ids.")
                if not np.array_equal(np.asarray(mask_check), np.asarray(legal_action_valid_mask)):
                    raise RuntimeError("feature_equiv_mask_value_mismatch between recomputed and input legal masks.")
            ref = self._build_legal_action_features_padded_reference(
                legal_global_ids_padded,
                legal_action_valid_mask,
                legal_id_to_raw_index,
                legal_actions,
                obs,
            )
            if features.shape != ref.shape:
                raise RuntimeError(
                    f"feature_equiv_shape_mismatch: new={features.shape} ref={ref.shape}"
                )
            if not np.allclose(features, ref, atol=1e-6, rtol=1e-6):
                diff = np.abs(features - ref)
                max_idx = np.unravel_index(np.argmax(diff), diff.shape)
                raise RuntimeError(
                    "feature_equiv_value_mismatch: "
                    f"max_abs_diff={float(np.max(diff)):.8f} at_slot={int(max_idx[0])} feature_idx={int(max_idx[1])}"
                )
        return features

    def _compute_legal_action_feature_vector_reference(self, action, obs):
        feat = np.zeros((int(self.ACTION_FEATURE_DIM),), dtype=np.float32)
        a_type = str(action.get("type", "UNKNOWN")).upper()
        is_move = a_type == "MOVE"
        feat[0] = 1.0 if is_move else 0.0

        if is_move:
            unit_id, src_x, src_y, dst_x, dst_y = self._extract_move_components(action, obs)
            if dst_x is not None and dst_y is not None:
                revealed = int(self._estimate_newly_revealed_tiles_if_move(obs, unit_id, int(dst_x), int(dst_y)))
                feat[1] = float(min(float(self.REVEAL_CLIP), max(0.0, float(revealed))) / max(1.0, float(self.REVEAL_CLIP)))

                adj_after = int(self._count_adjacent_fog_tiles(obs, int(dst_x), int(dst_y)))
                feat[2] = float(min(float(self.ADJ_FOG_MAX), max(0.0, float(adj_after))) / max(1.0, float(self.ADJ_FOG_MAX)))

                adj_before = 0
                if src_x is not None and src_y is not None:
                    adj_before = int(self._count_adjacent_fog_tiles(obs, int(src_x), int(src_y)))
                adj_delta = float(adj_after - adj_before)
                feat[3] = float(np.clip(adj_delta / max(1.0, float(self.ADJ_FOG_MAX)), -1.0, 1.0))
                feat[4] = 1.0 if revealed == 0 else 0.0

                visible_villages = self._get_visible_uncaptured_village_positions(obs)
                has_visible_village = len(visible_villages) > 0
                feat[6] = 1.0 if has_visible_village else 0.0
                if has_visible_village:
                    target_xy = (int(dst_x), int(dst_y))
                    # Use the same coordinate convention as legal MOVE dst_x/dst_y
                    # and the neutral-village detector: direct (x, y) only.
                    if target_xy in visible_villages:
                        feat[5] = 1.0
                    if src_x is not None and src_y is not None:
                        dist_before = self._min_manhattan_distance((int(src_x), int(src_y)), visible_villages)
                        dist_after = self._min_manhattan_distance((int(dst_x), int(dst_y)), visible_villages)
                        if dist_before is not None and dist_after is not None:
                            dims = self._board_dimensions_from_obs(obs)
                            dist_norm = float(max(dims)) if dims is not None else 1.0
                            if dist_norm <= 0:
                                dist_norm = 1.0
                            feat[7] = float(np.clip((float(dist_before) - float(dist_after)) / dist_norm, -1.0, 1.0))

                if unit_id is not None:
                    prev_tile = self._unit_previous_tiles.get(int(unit_id), None)
                    if prev_tile is not None and (int(dst_x), int(dst_y)) == (int(prev_tile[0]), int(prev_tile[1])):
                        feat[8] = 1.0
                    unit = (obs.get("unit", {}) or {}).get(str(int(unit_id)), {})
                    try:
                        feat[11] = 1.0 if int(unit.get("type", -1)) == 0 else 0.0
                    except Exception:
                        feat[11] = 0.0

                feat[9] = 1.0 if self._is_inside_owned_city_bounds(obs, int(dst_x), int(dst_y)) else 0.0

                capital_pos = self._get_capital_position(obs, tribe_id=0)
                if capital_pos is not None and src_x is not None and src_y is not None:
                    before = abs(int(src_x) - int(capital_pos[0])) + abs(int(src_y) - int(capital_pos[1]))
                    after = abs(int(dst_x) - int(capital_pos[0])) + abs(int(dst_y) - int(capital_pos[1]))
                    dims = self._board_dimensions_from_obs(obs)
                    dist_norm = float(max(dims)) if dims is not None else 1.0
                    if dist_norm <= 0:
                        dist_norm = 1.0
                    feat[10] = float(np.clip((float(after) - float(before)) / dist_norm, -1.0, 1.0))

        feat[12] = 1.0 if a_type == "END_TURN" else 0.0
        feat[13] = 1.0 if a_type == "CAPTURE" else 0.0
        feat[14] = 1.0 if a_type in ("SPAWN", "TRAIN") else 0.0
        feat[15] = 1.0 if a_type == "RESEARCH_TECH" else 0.0
        feat[16] = 1.0 if a_type == "RESOURCE_GATHERING" else 0.0
        feat[17] = 1.0 if a_type == "LEVEL_UP" else 0.0
        feat[18] = 1.0 if a_type == "BUILD" else 0.0
        feat[19] = 1.0 if a_type == "CLEAR_FOREST" else 0.0
        feat[20] = 1.0 if a_type == "GROW_FOREST" else 0.0
        known = is_move or bool(feat[12] or feat[13] or feat[14] or feat[15] or feat[16] or feat[17] or feat[18] or feat[19] or feat[20])
        feat[21] = 0.0 if known else 1.0

        tech_type = self._resolve_action_tech_type(action)
        if tech_type and self._catalog is not None:
            feat[22] = self._normalize_vocab_index(
                self._catalog.tech_to_idx.get(tech_type, -1),
                len(self._catalog.tech_types),
            )
        feat[23] = 1.0 if tech_type == "ORGANIZATION" else 0.0
        feat[24] = 1.0 if tech_type == "FORESTRY" else 0.0

        resource_type = self._resolve_action_resource_type(action)
        if resource_type and self._catalog is not None:
            feat[25] = self._normalize_vocab_index(
                self._catalog.resource_to_idx.get(resource_type, -1),
                len(self._catalog.resource_types),
            )
        feat[26] = 1.0 if resource_type == "ANIMAL" else 0.0
        feat[27] = 1.0 if resource_type == "FRUIT" else 0.0
        feat[28] = 1.0 if resource_type == "FISH" else 0.0
        feat[29] = 1.0 if resource_type == "CROPS" else 0.0
        feat[30] = 1.0 if resource_type == "ORE" else 0.0

        building_type = self._resolve_action_building_type(action)
        if building_type and self._catalog is not None:
            feat[31] = self._normalize_vocab_index(
                self._catalog.building_to_idx.get(building_type, -1),
                len(self._catalog.building_types),
            )
        feat[32] = 1.0 if building_type == "LUMBER_HUT" else 0.0
        feat[33] = 1.0 if building_type == "SAWMILL" else 0.0

        levelup_choice = self._resolve_action_levelup_choice(action)
        if levelup_choice and self._catalog is not None:
            feat[34] = self._normalize_vocab_index(
                self._catalog.levelup_to_idx.get(levelup_choice, -1),
                len(self._catalog.levelup_choices),
            )
        feat[35] = 1.0 if levelup_choice == "WORKSHOP" else 0.0

        eco = self._summarize_action_economy_expectation(action, obs)
        feat[36] = float(np.clip(float(eco["expected_population_delta"]) / 2.0, -1.0, 1.0))
        feat[37] = float(np.clip(float(eco["expected_immediate_spt_delta"]) / 5.0, -1.0, 1.0))
        feat[38] = 1.0 if bool(eco["makes_level_up_available"]) else 0.0
        feat[39] = 1.0 if bool(eco["is_level_up_claim"]) else 0.0
        feat[40] = float(np.clip(float(eco["progress_before"]), 0.0, 1.0))
        feat[41] = 1.0 if bool(eco["ready_before"]) else 0.0
        return feat

    def _compute_legal_action_feature_vector(self, action, obs):
        return self._compute_legal_action_feature_vector_reference(action, obs)

    def _count_adjacent_fog_tiles(self, obs, center_x, center_y):
        dims = self._board_dimensions_from_obs(obs)
        if dims is None:
            return 0
        width, height = dims
        cx = int(center_x)
        cy = int(center_y)
        cnt = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                x = cx + int(dx)
                y = cy + int(dy)
                if x < 0 or y < 0 or x >= int(width) or y >= int(height):
                    continue
                if self._board_get_int_by_java_coord(obs, "terrain", int(x), int(y), default=-1) == 7:
                    cnt += 1
        return int(cnt)

    def _estimate_newly_revealed_tiles_if_move(self, obs, unit_id, dst_x, dst_y):
        dims = self._board_dimensions_from_obs(obs)
        if dims is None:
            return 0
        width, height = dims
        cx = int(dst_x)
        cy = int(dst_y)
        if cx < 0 or cy < 0 or cx >= int(width) or cy >= int(height):
            return 0

        clear_range = 1
        unit_type = None
        if unit_id is not None:
            unit = (obs.get("unit", {}) or {}).get(str(int(unit_id)), {})
            try:
                unit_type = int(unit.get("type", -1))
            except Exception:
                unit_type = None
        if unit_type == 10:
            clear_range += 1
        else:
            if self._board_get_int_by_java_coord(obs, "terrain", int(cx), int(cy), default=-1) == 3:
                clear_range += 1

        revealed = 0
        for x in range(cx - clear_range, cx + clear_range + 1):
            if x < 0 or x >= int(width):
                continue
            for y in range(cy - clear_range, cy + clear_range + 1):
                if y < 0 or y >= int(height):
                    continue
                if self._board_get_int_by_java_coord(obs, "terrain", int(x), int(y), default=-1) == 7:
                    revealed += 1
        return int(revealed)

    def _get_capital_position(self, obs, tribe_id=0):
        tribes = obs.get("tribes", {})
        if not isinstance(tribes, dict):
            return None
        tribe = tribes.get(str(int(tribe_id)), {})
        if not isinstance(tribe, dict):
            return None
        capital_id = tribe.get("capitalID", None)
        if capital_id is None:
            return None
        city = (obs.get("city", {}) or {}).get(str(int(capital_id)), {})
        if not isinstance(city, dict):
            return None
        try:
            return int(city.get("x", -1)), int(city.get("y", -1))
        except Exception:
            return None

    def _is_inside_owned_city_bounds(self, obs, x, y):
        city_map = obs.get("city", {})
        if not isinstance(city_map, dict):
            return False
        tx = int(x)
        ty = int(y)
        for city in city_map.values():
            if not isinstance(city, dict):
                continue
            try:
                if int(city.get("tribeID", -1)) != 0:
                    continue
                cx = int(city.get("x", -1))
                cy = int(city.get("y", -1))
                bound = int(city.get("bound", -1))
            except Exception:
                continue
            if cx < 0 or cy < 0 or bound < 0:
                continue
            if max(abs(tx - cx), abs(ty - cy)) <= bound:
                return True
        return False

    def _sanitize_info_value(self, value):
        # Keep common scalar types as-is.
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.generic):
            try:
                return value.item()
            except Exception:
                return str(value)
        if isinstance(value, np.ndarray):
            # Object arrays are the most likely to contain non-picklables.
            if value.dtype == object:
                return [self._sanitize_info_value(v) for v in value.tolist()]
            return value
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize_info_value(v) for v in value]
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                out[str(k)] = self._sanitize_info_value(v)
            return out
        # Fallback for custom/foreign objects (e.g., Py4J wrappers, locks, etc.).
        return str(value)

    def _sanitize_info_for_multiprocessing(self, info):
        if not isinstance(info, dict):
            return {"info_error": "non_dict_info", "info_repr": str(info)}
        safe = {str(k): self._sanitize_info_value(v) for k, v in info.items()}
        # Final defensive check: if still unpicklable, degrade to string-only payload.
        try:
            pickle.dumps(safe, protocol=pickle.HIGHEST_PROTOCOL)
            return safe
        except Exception as exc:
            fallback = {}
            for k, v in safe.items():
                try:
                    pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
                    fallback[k] = v
                except Exception:
                    fallback[k] = str(v)
            fallback["info_pickle_warning"] = str(exc)
            return fallback

    def _parse_int_env(self, key, default):
        raw = os.environ.get(key, None)
        if raw is None:
            return int(default)
        try:
            return int(str(raw).strip())
        except Exception:
            return int(default)

    def _parse_float_env(self, key, default):
        raw = os.environ.get(key, None)
        if raw is None:
            return float(default)
        try:
            return float(str(raw).strip())
        except Exception:
            return float(default)

    def _parse_bool_env(self, key, default):
        raw = os.environ.get(key, None)
        if raw is None:
            return bool(default)
        s = str(raw).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        return bool(default)

    def _load_action_vocab(self):
        types_path = os.path.join(self._tribes_root_dir(), "src", "core", "Types.java")
        out = {
            "TECHNOLOGY": [],
            "UNIT": [],
            "RESOURCE": [],
            "BUILDING": [],
            "CITY_LEVEL_UP": [],
        }
        for enum_name in list(out.keys()):
            out[enum_name] = self._extract_enum_names_from_types_java(types_path, enum_name)
        return out

    def _extract_enum_names_from_types_java(self, path, enum_name):
        if not os.path.exists(path):
            return []
        try:
            lines = open(path, "r", encoding="utf-8").read().splitlines()
        except Exception:
            return []
        start = None
        marker = f"enum {enum_name}"
        for i, ln in enumerate(lines):
            if marker in ln:
                start = i
                break
        if start is None:
            return []

        names = []
        depth = 0
        entered = False
        constants_closed = False
        for ln in lines[start:]:
            depth += ln.count("{")
            if "{" in ln:
                entered = True
            if entered and not constants_closed and depth == 1:
                if ";" in ln:
                    constants_closed = True
                m = re.match(r"^\s*([A-Z][A-Z0-9_]+)\s*(\(|,)", ln)
                if m is not None:
                    names.append(m.group(1).strip().upper())
            depth -= ln.count("}")
            if entered and depth <= 0:
                break
        return names

    def _tribes_root_dir(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _resolve_level_pool(self, fallback_level):
        root = self._tribes_root_dir()
        pool_glob = os.environ.get("POLYVISION_LEVEL_POOL_GLOB", "").strip()
        if not pool_glob:
            default_glob_abs = os.path.join(root, self.DEFAULT_LEVEL_POOL_GLOB)
            found = sorted(glob.glob(default_glob_abs))
        else:
            pattern = pool_glob
            if not os.path.isabs(pattern):
                pattern = os.path.join(root, pattern)
            found = sorted(glob.glob(pattern))

        if found:
            return found

        if os.path.isabs(fallback_level):
            return [fallback_level]
        return [os.path.join(root, fallback_level)]

    def _ensure_seed_stream_initialized(self, seed):
        seed_i = int(seed)
        self._seed_stream = np.random.default_rng(seed_i)
        # Separate stream for map choice so map-selection mode is deterministic per run
        # but independent from episode-seed draws.
        self._level_pool_rng = np.random.default_rng(seed_i ^ 0x9E3779B9)
        if self._level_pool_size > 0:
            self._level_pool_offset = abs(seed_i) % self._level_pool_size
        else:
            self._level_pool_offset = 0

    def _resolve_episode_seed(self, seed=None):
        if seed is not None:
            resolved = int(seed)
            self._ensure_seed_stream_initialized(resolved)
            self._last_reset_seed = resolved
            return resolved

        if self._seed_stream is None:
            self._ensure_seed_stream_initialized(self._seed_stream_base)

        resolved = int(self._seed_stream.integers(0, 2**31 - 1))
        self._last_reset_seed = resolved
        return resolved

    def _select_level_for_reset(self, episode_seed):
        if self._level_pool_size <= 1:
            return self._level_pool[0], 0

        if self._level_selection_mode == "seeded_random":
            if self._level_pool_rng is None:
                self._ensure_seed_stream_initialized(episode_seed)
            idx = int(self._level_pool_rng.integers(0, self._level_pool_size))
            return self._level_pool[idx], idx

        idx = int((self._level_pool_offset + self._episode_index) % self._level_pool_size)
        return self._level_pool[idx], idx

    def _filter_allowed_raw_indices(
        self,
        legal_actions,
        obs,
        profile=None,
        optimize_visible_village_filter=True,
    ):
        use_profile = isinstance(profile, dict)
        if use_profile:
            profile["filter_allowed_type_s"] = 0.0
            profile["filter_oob_move_s"] = 0.0
            profile["filter_resource_upgrade_s"] = 0.0
            profile["filter_city_count_tactical_s"] = 0.0
            profile["filter_capture_priority_s"] = 0.0
            profile["filter_move_visible_village_s"] = 0.0
            profile["filter_closest_reduce_distance_s"] = 0.0
            profile["filter_early_backtrack_s"] = 0.0
            profile["filter_board_city_village_scans_s"] = 0.0
            profile["raw_actions_count"] = 0
            profile["allowed_after_base_count"] = 0
            profile["allowed_after_tactical_count"] = 0
            profile["allowed_final_count"] = 0
        allowed_indices = []
        if use_profile:
            profile["raw_actions_count"] = int(len(legal_actions)) if isinstance(legal_actions, list) else 0
        for idx, a in enumerate(legal_actions):
            t_type0 = time.perf_counter() if use_profile else None
            a_type = str(a.get("type", "")).upper()
            if use_profile:
                profile["filter_allowed_type_s"] += time.perf_counter() - t_type0
            if a_type not in self.ALLOWED_ACTION_TYPES:
                continue
            t_oob0 = time.perf_counter() if use_profile else None
            if a_type == "MOVE" and not self._is_move_destination_within_board(a, obs):
                if use_profile:
                    profile["filter_oob_move_s"] += time.perf_counter() - t_oob0
                continue
            if use_profile:
                profile["filter_oob_move_s"] += time.perf_counter() - t_oob0
            t_rg0 = time.perf_counter() if use_profile else None
            if (
                a_type == "RESOURCE_GATHERING"
                and bool(self._resource_gather_upgrade_filter_enabled)
                and not self._is_resource_gather_legal_for_upgrade(a, legal_actions, obs)
            ):
                if use_profile:
                    profile["filter_resource_upgrade_s"] += time.perf_counter() - t_rg0
                continue
            if use_profile:
                profile["filter_resource_upgrade_s"] += time.perf_counter() - t_rg0
            allowed_indices.append(idx)
        if use_profile:
            profile["allowed_after_base_count"] = int(len(allowed_indices))

        # Hard guardrail: if <2 cities, prioritize village-capture lines.
        t_city_gate0 = time.perf_counter() if use_profile else None
        if self._get_city_count(obs) < 2 and allowed_indices:
            if use_profile:
                profile["filter_city_count_tactical_s"] += time.perf_counter() - t_city_gate0
            forced_village_captures = []
            t_capture0 = time.perf_counter() if use_profile else None
            for idx in allowed_indices:
                a = legal_actions[idx]
                if a.get("type") != "CAPTURE":
                    continue
                if self._is_capture_of_village(a, obs):
                    forced_village_captures.append(idx)
            if use_profile:
                profile["filter_capture_priority_s"] += time.perf_counter() - t_capture0
            if forced_village_captures:
                t_capture0 = time.perf_counter() if use_profile else None
                frozen_units = set()
                for idx in forced_village_captures:
                    u = self._parse_unit_id_from_action_repr(str(legal_actions[idx].get("repr", "")))
                    if u is not None:
                        frozen_units.add(int(u))
                self._queued_village_capture_unit_ids = set(frozen_units)

                filtered_indices = []
                for idx in allowed_indices:
                    a = legal_actions[idx]
                    a_type = a.get("type")
                    if a_type not in ("MOVE", "CAPTURE"):
                        filtered_indices.append(idx)
                        continue
                    u = self._parse_unit_id_from_action_repr(str(a.get("repr", "")))
                    if u is None or int(u) not in frozen_units:
                        filtered_indices.append(idx)
                if filtered_indices:
                    allowed_indices = filtered_indices
                else:
                    end_turn_idx = next((i for i, a in enumerate(legal_actions) if a.get("type") == "END_TURN"), None)
                    allowed_indices = [end_turn_idx] if end_turn_idx is not None else []
                if use_profile:
                    profile["filter_capture_priority_s"] += time.perf_counter() - t_capture0
            else:
                self._queued_village_capture_unit_ids = set()

            t_scan0 = time.perf_counter() if use_profile else None
            visible_village_positions = self._get_visible_uncaptured_village_positions(obs)
            if use_profile:
                profile["filter_board_city_village_scans_s"] += time.perf_counter() - t_scan0
            village_mask = None
            if bool(optimize_visible_village_filter):
                village_mask = self._build_village_lookup_mask(obs, visible_village_positions)
            forced_village_moves = []
            if not forced_village_captures:
                t_move_visible0 = time.perf_counter() if use_profile else None
                for idx in allowed_indices:
                    a = legal_actions[idx]
                    if a.get("type") != "MOVE":
                        continue
                    if bool(optimize_visible_village_filter):
                        if self._is_move_to_visible_uncaptured_village(
                            a,
                            obs,
                            village_positions=visible_village_positions,
                            village_mask=village_mask,
                        ):
                            forced_village_moves.append(idx)
                    else:
                        if self._is_move_to_visible_uncaptured_village(a, obs):
                            forced_village_moves.append(idx)
                if use_profile:
                    profile["filter_move_visible_village_s"] += time.perf_counter() - t_move_visible0
                if forced_village_moves:
                    allowed_indices = forced_village_moves
                elif visible_village_positions:
                    t_closest0 = time.perf_counter() if use_profile else None
                    closest_unit_id, _closest_pos, _closest_dist = self._closest_owned_unit_to_targets(
                        obs,
                        visible_village_positions,
                    )
                    forced_progress_moves = []
                    if closest_unit_id is not None:
                        for idx in allowed_indices:
                            a = legal_actions[idx]
                            if str(a.get("type", "")).upper() != "MOVE":
                                continue
                            if self._is_move_reducing_distance_to_targets(
                                a,
                                obs,
                                closest_unit_id,
                                visible_village_positions,
                            ):
                                forced_progress_moves.append(idx)
                    if forced_progress_moves:
                        allowed_indices = forced_progress_moves
                    if use_profile:
                        profile["filter_closest_reduce_distance_s"] += time.perf_counter() - t_closest0
        else:
            if use_profile:
                profile["filter_city_count_tactical_s"] += time.perf_counter() - t_city_gate0
            self._queued_village_capture_unit_ids = set()
        if use_profile:
            profile["allowed_after_tactical_count"] = int(len(allowed_indices))

        # Early-turn constraint: only block unit 2 immediate backtracking on T1/T2.
        t_backtrack0 = time.perf_counter() if use_profile else None
        allowed_indices = self._apply_t1_t2_unit2_backtrack_mask(allowed_indices, legal_actions, obs)
        if use_profile:
            profile["filter_early_backtrack_s"] += time.perf_counter() - t_backtrack0
            profile["allowed_final_count"] = int(len(allowed_indices))

        return allowed_indices

    def _apply_t1_t2_unit2_backtrack_mask(self, allowed_indices, legal_actions, obs):
        if not allowed_indices:
            return allowed_indices
        if int(self._turn_count) not in (1, 2):
            return allowed_indices

        target_unit_id = 2
        prev_tile = self._unit_previous_tiles.get(int(target_unit_id), None)
        if prev_tile is None:
            return allowed_indices
        prev_tile = (int(prev_tile[0]), int(prev_tile[1]))

        filtered = []
        removed_any = False
        for idx in allowed_indices:
            a = legal_actions[idx]
            if str(a.get("type", "")).upper() != "MOVE":
                filtered.append(idx)
                continue

            unit_id, _sx, _sy, dst_x, dst_y = self._extract_move_components(a, obs)
            if unit_id is None or dst_x is None or dst_y is None:
                filtered.append(idx)
                continue

            if int(unit_id) == int(target_unit_id) and (int(dst_x), int(dst_y)) == prev_tile:
                removed_any = True
                continue

            filtered.append(idx)

        # Safety: never collapse legal set to empty from this one-tile rule.
        if removed_any and len(filtered) == 0:
            return allowed_indices
        return filtered

    def _extract_move_components(self, action, obs):
        unit_id = self._action_int(action, "unit_id", None)
        src_x = self._action_int(action, "src_x", None)
        src_y = self._action_int(action, "src_y", None)
        dst_x = self._action_int(action, "dst_x", None)
        dst_y = self._action_int(action, "dst_y", None)
        if (dst_x is None or dst_y is None) or (unit_id is None):
            t_repr_parse0 = time.perf_counter() if self._profile_feature_build_active else None
            parsed_move = self._parse_move_unit_and_dest_from_action_repr(str(action.get("repr", "")))
            if self._profile_feature_build_active:
                self._profile_feature_build_repr_parse_s += time.perf_counter() - t_repr_parse0
            if parsed_move is not None:
                if unit_id is None:
                    unit_id = int(parsed_move[0])
                if dst_x is None:
                    dst_x = int(parsed_move[1])
                if dst_y is None:
                    dst_y = int(parsed_move[2])
        if (src_x is None or src_y is None) and unit_id is not None:
            pos = self._unit_position_by_id(obs, unit_id)
            if pos is not None:
                src_x, src_y = int(pos[0]), int(pos[1])
        return unit_id, src_x, src_y, dst_x, dst_y

    def _is_move_reducing_distance_to_targets(self, action, obs, target_unit_id, targets):
        if target_unit_id is None or not targets:
            return False
        unit_id, src_x, src_y, dst_x, dst_y = self._extract_move_components(action, obs)
        if unit_id is None or int(unit_id) != int(target_unit_id):
            return False
        if dst_x is None or dst_y is None:
            return False

        if src_x is None or src_y is None:
            pos = self._unit_position_by_id(obs, unit_id)
            if pos is None:
                return False
            src_x, src_y = int(pos[0]), int(pos[1])

        dist_before = self._min_manhattan_distance((int(src_x), int(src_y)), targets)
        dist_after = self._min_manhattan_distance((int(dst_x), int(dst_y)), targets)
        if dist_before is None or dist_after is None:
            return False
        return int(dist_after) < int(dist_before)

    def _build_action_mask_and_mapping(self, legal_actions=None, obs=None, profile=None):
        use_profile = bool(self._profile_sps_enabled and isinstance(profile, dict))
        if use_profile:
            profile["filter_allowed_indices_s"] = 0.0
            profile["filter_allowed_type_s"] = 0.0
            profile["filter_oob_move_s"] = 0.0
            profile["filter_resource_upgrade_s"] = 0.0
            profile["filter_city_count_tactical_s"] = 0.0
            profile["filter_capture_priority_s"] = 0.0
            profile["filter_move_visible_village_s"] = 0.0
            profile["filter_closest_reduce_distance_s"] = 0.0
            profile["filter_early_backtrack_s"] = 0.0
            profile["filter_board_city_village_scans_s"] = 0.0
            profile["raw_actions_count"] = 0
            profile["allowed_after_base_count"] = 0
            profile["allowed_after_tactical_count"] = 0
            profile["allowed_final_count"] = 0
            profile["canonicalize_global_id_s"] = 0.0
            profile["collision_check_s"] = 0.0
            profile["legal_id_list_construct_s"] = 0.0
            profile["mask_build_s"] = 0.0
            profile["diag_build_s"] = 0.0
        if legal_actions is None:
            legal_actions = self.tribes_env.list_actions()
        if obs is None:
            obs = getattr(self.tribes_env, "_last_obs", {})
        if not isinstance(obs, dict):
            obs = {}
        if self._catalog is None:
            raise RuntimeError("Global action catalog is not initialized.")

        t_filter0 = time.perf_counter() if use_profile else None
        allowed_indices = self._filter_allowed_raw_indices(legal_actions, obs, profile=profile if use_profile else None)
        if use_profile:
            profile["filter_allowed_indices_s"] += time.perf_counter() - t_filter0
        run_filter_equiv = bool(
            self._filter_equiv_check_enabled
            and int(self._total_action_decisions) % int(self._filter_equiv_check_every_n) == 0
        )
        legacy_allowed_indices = None
        if run_filter_equiv:
            legacy_allowed_indices = self._filter_allowed_raw_indices(
                legal_actions,
                obs,
                profile=None,
                optimize_visible_village_filter=False,
            )
        legal_id_to_raw_index = {}
        uncanonicalized = []
        collisions = []
        per_type_counts = {}

        for raw_idx in allowed_indices:
            action = legal_actions[raw_idx]
            a_type = str(action.get("type", "UNKNOWN")).upper()
            per_type_counts[a_type] = per_type_counts.get(a_type, 0) + 1
            t_can0 = time.perf_counter() if use_profile else None
            gid, reason = self._canonicalize_action_to_global_id(action, obs)
            if use_profile:
                profile["canonicalize_global_id_s"] += time.perf_counter() - t_can0
            if gid is None:
                uncanonicalized.append({"raw_idx": int(raw_idx), "type": a_type, "reason": str(reason), "repr": str(action.get("repr", ""))})
                continue
            t_collision0 = time.perf_counter() if use_profile else None
            if gid in legal_id_to_raw_index and legal_id_to_raw_index[gid] != raw_idx:
                collisions.append({"global_id": int(gid), "raw_a": int(legal_id_to_raw_index[gid]), "raw_b": int(raw_idx), "type": a_type})
                if use_profile:
                    profile["collision_check_s"] += time.perf_counter() - t_collision0
            else:
                if use_profile:
                    profile["collision_check_s"] += time.perf_counter() - t_collision0
                t_lid0 = time.perf_counter() if use_profile else None
                legal_id_to_raw_index[int(gid)] = int(raw_idx)
                if use_profile:
                    profile["legal_id_list_construct_s"] += time.perf_counter() - t_lid0

        t_mask0 = time.perf_counter() if use_profile else None
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for gid in legal_id_to_raw_index.keys():
            if 0 <= int(gid) < self.action_space.n:
                mask[int(gid)] = 1
        if use_profile:
            profile["mask_build_s"] += time.perf_counter() - t_mask0

        t_diag0 = time.perf_counter() if use_profile else None
        diag = {
            "legal_actions_total": int(len(allowed_indices)),
            "canonicalized_legal_actions": int(len(legal_id_to_raw_index)),
            "uncanonicalized_legal_actions": int(len(uncanonicalized)),
            "uncanonicalized_legal_actions_by_type": self._summarize_by_type(uncanonicalized),
            "uncanonicalized_repr_examples": [u.get("repr", "") for u in uncanonicalized[:20]],
            "duplicate_global_id_collisions": int(len(collisions)),
            "mask_ones": int(np.sum(mask)),
            "unique_legal_global_ids": int(len(legal_id_to_raw_index)),
            "legal_action_count_by_type": per_type_counts,
            "global_id_collisions": collisions[:20],
        }
        if use_profile:
            profile["diag_build_s"] += time.perf_counter() - t_diag0

        if run_filter_equiv and legacy_allowed_indices is not None:
            if list(allowed_indices) != list(legacy_allowed_indices):
                raise RuntimeError(
                    "FILTER_EQUIV: allowed_indices mismatch "
                    f"new_n={len(allowed_indices)} legacy_n={len(legacy_allowed_indices)}"
                )
            new_types = [str(legal_actions[i].get("type", "UNKNOWN")).upper() for i in allowed_indices]
            legacy_types = [str(legal_actions[i].get("type", "UNKNOWN")).upper() for i in legacy_allowed_indices]
            if new_types != legacy_types:
                raise RuntimeError("FILTER_EQUIV: allowed action type order mismatch")
            new_ordered_gids = []
            for raw_idx in allowed_indices:
                gid, _reason = self._canonicalize_action_to_global_id(legal_actions[raw_idx], obs)
                if gid is not None:
                    new_ordered_gids.append(int(gid))

            legacy_map = {}
            legacy_uncanonicalized = []
            legacy_collisions = []
            legacy_per_type_counts = {}
            legacy_ordered_gids = []
            for raw_idx in legacy_allowed_indices:
                action = legal_actions[raw_idx]
                a_type = str(action.get("type", "UNKNOWN")).upper()
                legacy_per_type_counts[a_type] = legacy_per_type_counts.get(a_type, 0) + 1
                gid, reason = self._canonicalize_action_to_global_id(action, obs)
                if gid is None:
                    legacy_uncanonicalized.append(
                        {
                            "raw_idx": int(raw_idx),
                            "type": a_type,
                            "reason": str(reason),
                            "repr": str(action.get("repr", "")),
                        }
                    )
                    continue
                legacy_ordered_gids.append(int(gid))
                if gid in legacy_map and legacy_map[gid] != raw_idx:
                    legacy_collisions.append(
                        {
                            "global_id": int(gid),
                            "raw_a": int(legacy_map[gid]),
                            "raw_b": int(raw_idx),
                            "type": a_type,
                        }
                    )
                else:
                    legacy_map[int(gid)] = int(raw_idx)
            legacy_mask = np.zeros(self.action_space.n, dtype=np.int8)
            for gid in legacy_map.keys():
                if 0 <= int(gid) < self.action_space.n:
                    legacy_mask[int(gid)] = 1

            if dict(legacy_map) != dict(legal_id_to_raw_index):
                raise RuntimeError("FILTER_EQUIV: canonical legal_id_to_raw_index mismatch")
            if new_ordered_gids != legacy_ordered_gids:
                raise RuntimeError("FILTER_EQUIV: ordered canonical IDs mismatch")
            if not np.array_equal(np.asarray(mask), np.asarray(legacy_mask)):
                raise RuntimeError("FILTER_EQUIV: mask mismatch")
            if dict(per_type_counts) != dict(legacy_per_type_counts):
                raise RuntimeError("FILTER_EQUIV: legal_action_count_by_type mismatch")
            if len(uncanonicalized) != len(legacy_uncanonicalized):
                raise RuntimeError("FILTER_EQUIV: uncanonicalized count mismatch")
            if len(collisions) != len(legacy_collisions):
                raise RuntimeError("FILTER_EQUIV: collision count mismatch")

        # Fail-fast: never silently continue on canonicalization or collision gaps.
        if len(collisions) > 0:
            raise RuntimeError(f"[ACTION_CATALOG] Global-ID collision detected: {json.dumps(diag, default=str)}")
        if len(uncanonicalized) > 0:
            raise RuntimeError(f"[ACTION_CATALOG] Uncanonicalized allowed legal actions detected: {json.dumps(diag, default=str)}")

        return mask, legal_id_to_raw_index, diag

    def _summarize_by_type(self, rows):
        out = {}
        for r in rows:
            t = str(r.get("type", "UNKNOWN"))
            out[t] = out.get(t, 0) + 1
        return out

    def _action_int(self, action, key, default=None):
        try:
            if key not in action:
                return default
            v = action.get(key)
            if v is None:
                return default
            return int(v)
        except Exception:
            return default

    def _action_str(self, action, key, default=None):
        try:
            if key not in action:
                return default
            v = action.get(key)
            if v is None:
                return default
            return str(v).upper()
        except Exception:
            return default

    def _unit_position_by_id(self, obs, unit_id):
        units = obs.get("unit", {})
        if not isinstance(units, dict):
            return None
        unit = units.get(str(unit_id), None)
        if not isinstance(unit, dict):
            return None
        try:
            return int(unit.get("x", -1)), int(unit.get("y", -1))
        except Exception:
            return None

    def _city_position_by_id(self, obs, city_id):
        cities = obs.get("city", {})
        if not isinstance(cities, dict):
            return None
        city = cities.get(str(city_id), None)
        if not isinstance(city, dict):
            return None
        try:
            return int(city.get("x", -1)), int(city.get("y", -1))
        except Exception:
            return None

    def _parse_target_xy_from_action_repr(self, action_repr):
        if not isinstance(action_repr, str):
            return None
        m = re.search(r"\bat\s+(-?\d+)\s*:\s*(-?\d+)", action_repr, flags=re.IGNORECASE)
        if m is not None:
            try:
                return int(m.group(1)), int(m.group(2))
            except Exception:
                return None
        return None

    def _canonicalize_action_to_global_id(self, action, obs):
        if self._catalog is None:
            return None, "catalog_uninitialized"
        a_type = str(action.get("type", "UNKNOWN")).upper()
        repr_s = str(action.get("repr", ""))

        if a_type == "END_TURN":
            return self._catalog.id_end_turn(), None

        if a_type == "MOVE":
            src_x = self._action_int(action, "src_x", None)
            src_y = self._action_int(action, "src_y", None)
            dst_x = self._action_int(action, "dst_x", None)
            dst_y = self._action_int(action, "dst_y", None)
            unit_id = self._action_int(action, "unit_id", None)
            if (src_x is None or src_y is None) and unit_id is not None:
                pos = self._unit_position_by_id(obs, unit_id)
                if pos is not None:
                    src_x, src_y = pos
            if dst_x is None or dst_y is None:
                parsed = self._parse_move_unit_and_dest_from_action_repr(repr_s)
                if parsed is not None:
                    _, dst_x, dst_y = parsed
            src_tile = self._catalog.tile_id(src_x, src_y)
            dst_tile = self._catalog.tile_id(dst_x, dst_y)
            gid = self._catalog.id_move(src_tile, dst_tile)
            return gid, "missing_move_coords" if gid is None else None

        if a_type == "CAPTURE":
            unit_id = self._action_int(action, "unit_id", None)
            src_x = self._action_int(action, "src_x", None)
            src_y = self._action_int(action, "src_y", None)
            if (src_x is None or src_y is None) and unit_id is not None:
                pos = self._unit_position_by_id(obs, unit_id)
                if pos is not None:
                    src_x, src_y = pos
            tgt_x = self._action_int(action, "target_x", None)
            tgt_y = self._action_int(action, "target_y", None)
            if tgt_x is None or tgt_y is None:
                tgt_x, tgt_y = src_x, src_y
            src_tile = self._catalog.tile_id(src_x, src_y)
            tgt_tile = self._catalog.tile_id(tgt_x, tgt_y)
            cap_type = self._action_str(action, "capture_type", "UNKNOWN")
            gid = self._catalog.id_capture(src_tile, tgt_tile, cap_type)
            return gid, "missing_capture_coords" if gid is None else None

        if a_type in ("SPAWN", "TRAIN"):
            unit_type = self._action_str(action, "unit_type", None)
            city_x = self._action_int(action, "city_x", None)
            city_y = self._action_int(action, "city_y", None)
            if city_x is None or city_y is None:
                city_id = self._action_int(action, "city_id", None)
                if city_id is not None:
                    pos = self._city_position_by_id(obs, city_id)
                    if pos is not None:
                        city_x, city_y = pos
            city_tile = self._catalog.tile_id(city_x, city_y)
            gid = self._catalog.id_train(unit_type, city_tile)
            return gid, "missing_spawn_fields" if gid is None else None

        if a_type == "RESOURCE_GATHERING":
            r_type = self._action_str(action, "resource_type", None)
            tx = self._action_int(action, "target_x", None)
            ty = self._action_int(action, "target_y", None)
            if tx is None or ty is None:
                parsed = self._parse_target_xy_from_action_repr(repr_s)
                if parsed is not None:
                    tx, ty = parsed
            res_tile = self._catalog.tile_id(tx, ty)
            gid = self._catalog.id_resource(r_type, res_tile)
            return gid, "missing_resource_fields" if gid is None else None

        if a_type == "CLEAR_FOREST":
            tx = self._action_int(action, "target_x", None)
            ty = self._action_int(action, "target_y", None)
            if tx is None or ty is None:
                parsed = self._parse_target_xy_from_action_repr(repr_s)
                if parsed is not None:
                    tx, ty = parsed
            tile = self._catalog.tile_id(tx, ty)
            gid = self._catalog.id_clear_forest(tile)
            return gid, "missing_clear_forest_target" if gid is None else None

        if a_type == "GROW_FOREST":
            tx = self._action_int(action, "target_x", None)
            ty = self._action_int(action, "target_y", None)
            if tx is None or ty is None:
                parsed = self._parse_target_xy_from_action_repr(repr_s)
                if parsed is not None:
                    tx, ty = parsed
            tile = self._catalog.tile_id(tx, ty)
            gid = self._catalog.id_grow_forest(tile)
            return gid, "missing_grow_forest_target" if gid is None else None

        if a_type == "BUILD":
            b_type = self._action_str(action, "building_type", None)
            tx = self._action_int(action, "target_x", None)
            ty = self._action_int(action, "target_y", None)
            if tx is None or ty is None:
                parsed = self._parse_target_xy_from_action_repr(repr_s)
                if parsed is not None:
                    tx, ty = parsed
            tile = self._catalog.tile_id(tx, ty)
            gid = self._catalog.id_build(b_type, tile)
            return gid, "missing_build_fields" if gid is None else None

        if a_type == "RESEARCH_TECH":
            tech = self._action_str(action, "tech_type", None)
            gid = self._catalog.id_research(tech)
            return gid, "missing_research_tech" if gid is None else None

        if a_type == "LEVEL_UP":
            choice = self._action_str(action, "levelup_choice", None)
            city_x = self._action_int(action, "city_x", None)
            city_y = self._action_int(action, "city_y", None)
            if city_x is None or city_y is None:
                city_id = self._action_int(action, "city_id", None)
                if city_id is not None:
                    pos = self._city_position_by_id(obs, city_id)
                    if pos is not None:
                        city_x, city_y = pos
            city_tile = self._catalog.tile_id(city_x, city_y)
            gid = self._catalog.id_levelup(choice, city_tile)
            return gid, "missing_levelup_fields" if gid is None else None

        if a_type == "EXAMINE":
            src_x = self._action_int(action, "src_x", None)
            src_y = self._action_int(action, "src_y", None)
            if src_x is None or src_y is None:
                unit_id = self._action_int(action, "unit_id", None)
                if unit_id is not None:
                    pos = self._unit_position_by_id(obs, unit_id)
                    if pos is not None:
                        src_x, src_y = pos
            unit_tile = self._catalog.tile_id(src_x, src_y)
            gid = self._catalog.id_examine(unit_tile)
            return gid, "missing_examine_source" if gid is None else None

        return None, f"unsupported_action_type:{a_type}"

    def _apply_bardur_opening(self, obs):
        """Force deterministic opening through the start of Turn 2."""
        def find_action_idx(predicate):
            legal = self.tribes_env.list_actions()
            for idx, act in enumerate(legal):
                if predicate(act):
                    return idx
            return None

        def parse_move_repr(action_repr):
            # Supports common formats such as:
            # "MOVE by unit U to X : Y" or "MOVE by unit U from A : B to X : Y".
            if not isinstance(action_repr, str):
                return None
            nums = [int(n) for n in re.findall(r"-?\d+", action_repr)]
            if len(nums) < 3:
                return None
            unit_id = nums[0]
            if len(nums) >= 5:
                current_x, current_y, dest_x, dest_y = nums[-4], nums[-3], nums[-2], nums[-1]
            else:
                current_x, current_y, dest_x, dest_y = None, None, nums[-2], nums[-1]
            return unit_id, current_x, current_y, dest_x, dest_y

        def get_unit_pos_from_obs(local_obs, unit_id):
            units = local_obs.get("unit", {})
            if not isinstance(units, dict):
                return None
            for key, unit in units.items():
                if not isinstance(unit, dict):
                    continue
                try:
                    key_as_int = int(key)
                except Exception:
                    key_as_int = None
                if key_as_int == unit_id:
                    return int(unit.get("x", -1)), int(unit.get("y", -1))
            return None

        def get_capital_pos(local_obs):
            try:
                tribes = local_obs.get("tribes", {})
                city_map = local_obs.get("city", {})
                tribe0 = tribes.get("0", {}) if isinstance(tribes, dict) else {}
                capital_id = tribe0.get("capitalID", None) if isinstance(tribe0, dict) else None
                if capital_id is None:
                    return None
                city = city_map.get(str(capital_id), None) if isinstance(city_map, dict) else None
                if isinstance(city, dict):
                    return int(city.get("x", -1)), int(city.get("y", -1))
            except Exception:
                return None
            return None

        def score_move(
            action_repr,
            capital_x,
            capital_y,
            other_unit_pos=None,
            current_override=None,
            map_size=15,
        ):
            parsed = parse_move_repr(action_repr)
            if parsed is None:
                return -1e9

            _, current_x, current_y, dest_x, dest_y = parsed
            if (current_x is None or current_y is None) and current_override is not None:
                current_x, current_y = current_override
            score = 0.0

            # Reward 1: diagonal movement.
            if current_x is not None and current_y is not None:
                if abs(dest_x - current_x) > 0 and abs(dest_y - current_y) > 0:
                    score += 3.0

            # Reward 2: move toward map center (or away from capital if center unknown).
            center = (map_size - 1) / 2.0
            if current_x is not None and current_y is not None:
                current_dist_center = abs(current_x - center) + abs(current_y - center)
                dest_dist_center = abs(dest_x - center) + abs(dest_y - center)
                score += max(0.0, current_dist_center - dest_dist_center) * 1.0
            else:
                # Fallback center-pressure using distance away from capital/edge.
                score += abs(dest_x - capital_x) * 0.2 + abs(dest_y - capital_y) * 0.2
                edge_dist = min(dest_x, dest_y, map_size - 1 - dest_x, map_size - 1 - dest_y)
                score += max(0, edge_dist) * 0.2

            # Reward 3: dispersion from other unit position.
            if other_unit_pos is not None:
                ox, oy = other_unit_pos
                score += (abs(dest_x - ox) + abs(dest_y - oy)) * 0.7

            return score

        def ensure_bardur_turn(local_obs):
            # Fast-forward non-Bardur turns with END_TURN so scripted actions always
            # execute for tribe 0. Defensive caps prevent infinite loops.
            for _ in range(6):
                try:
                    if int(local_obs.get("activeTribeID", -1)) == 0:
                        return local_obs
                except Exception:
                    return local_obs
                idx = find_action_idx(lambda a: a.get("type") == "END_TURN")
                if idx is None:
                    return local_obs
                local_obs, _, _, _ = self.tribes_env.step(idx)
            return local_obs

        def choose_and_execute_best_move(local_obs, target_unit_id=None, other_unit_pos=None):
            legal = self.tribes_env.list_actions()
            move_candidates = []
            for idx, act in enumerate(legal):
                if act.get("type") != "MOVE":
                    continue
                parsed = parse_move_repr(act.get("repr", ""))
                if parsed is None:
                    continue
                unit_id = parsed[0]
                if target_unit_id is not None and unit_id != target_unit_id:
                    continue
                move_candidates.append((idx, act, parsed))
            if not move_candidates:
                return local_obs

            capital_pos = get_capital_pos(local_obs)
            if capital_pos is None:
                capital_pos = (0, 0)
            cap_x, cap_y = capital_pos
            board = local_obs.get("board", {})
            terrain = board.get("terrain", [])
            map_size = len(terrain) if terrain else 15

            best = None
            best_score = -1e18
            scored_moves = []
            for idx, act, parsed in move_candidates:
                try:
                    unit_id, cur_x, cur_y, dest_x, dest_y = parsed
                    effective_cur = (cur_x, cur_y)
                    if (cur_x is None or cur_y is None) and unit_id is not None:
                        fallback_pos = get_unit_pos_from_obs(local_obs, unit_id)
                        if fallback_pos is not None:
                            effective_cur = fallback_pos
                            sc = score_move(
                                act.get("repr", ""),
                                cap_x,
                                cap_y,
                                other_unit_pos=other_unit_pos,
                                current_override=fallback_pos,
                                map_size=map_size,
                            )
                        else:
                            sc = score_move(
                                act.get("repr", ""),
                                cap_x,
                                cap_y,
                                other_unit_pos=other_unit_pos,
                                map_size=map_size,
                            )
                    else:
                        sc = score_move(
                            act.get("repr", ""),
                            cap_x,
                            cap_y,
                            other_unit_pos=other_unit_pos,
                            map_size=map_size,
                        )
                    dx = None
                    dy = None
                    cur_valid = effective_cur[0] is not None and effective_cur[1] is not None
                    if cur_valid:
                        dx = int(dest_x) - int(effective_cur[0])
                        dy = int(dest_y) - int(effective_cur[1])
                    scored_moves.append(
                        {
                            "score": float(sc),
                            "idx": idx,
                            "unit_id": unit_id,
                            "cur": (effective_cur[0], effective_cur[1]),
                            "dest": (int(dest_x), int(dest_y)),
                            "dx": dx,
                            "dy": dy,
                            "cur_valid": cur_valid,
                        }
                    )
                    if sc > best_score:
                        best_score = sc
                        best = idx
                except Exception:
                    continue

            if self.debug_opening_grid and scored_moves:
                anchor = None
                if best is not None:
                    anchor = next((m for m in scored_moves if m["idx"] == best), None)
                if anchor is None:
                    anchor = max(scored_moves, key=lambda m: m["score"])

                if anchor and anchor["cur_valid"]:
                    anchor_unit = anchor["unit_id"]
                    anchor_cur = anchor["cur"]

                    # Map relative offsets -> score for moves of the selected unit from selected origin.
                    rel_scores = {}
                    max_delta = 1
                    for m in scored_moves:
                        if m["unit_id"] != anchor_unit or m["cur"] != anchor_cur or not m["cur_valid"]:
                            continue
                        key = (int(m["dx"]), int(m["dy"]))
                        rel_scores[key] = float(m["score"])
                        max_delta = max(max_delta, abs(int(m["dx"])), abs(int(m["dy"])))

                    radius = max(2, max_delta)
                    print(f"OPENING_MOVE_GRID: unit={anchor_unit} centered at current tile")
                    print("  (numbers = move score, X = invalid/unavailable, U = unit)")
                    for rel_y in range(-radius, radius + 1):
                        row = []
                        for rel_x in range(-radius, radius + 1):
                            if rel_x == 0 and rel_y == 0:
                                row.append("  U  ")
                                continue

                            # Display-grid orientation is mapped to match the Java
                            # viewer orientation:
                            # shown horizontal axis -> engine Y delta
                            # shown vertical axis   -> engine X delta
                            engine_dx = rel_y
                            engine_dy = rel_x

                            world_x = int(anchor_cur[0]) + engine_dx
                            world_y = int(anchor_cur[1]) + engine_dy
                            off_board = world_x < 0 or world_y < 0 or world_x >= map_size or world_y >= map_size
                            if off_board:
                                row.append("  X  ")
                                continue

                            key = (engine_dx, engine_dy)
                            if key in rel_scores:
                                row.append(f"{rel_scores[key]:5.1f}")
                            else:
                                row.append("  X  ")
                        print(" ".join(row))
                else:
                    print("OPENING_MOVE_GRID: unavailable (could not resolve unit origin).")

            if best is None:
                return local_obs
            try:
                chosen = next((m for m in scored_moves if m.get("idx", None) == best), None)
                if isinstance(chosen, dict):
                    unit_id = chosen.get("unit_id", None)
                    cur = chosen.get("cur", (None, None))
                    cur_x = cur[0] if isinstance(cur, tuple) and len(cur) > 0 else None
                    cur_y = cur[1] if isinstance(cur, tuple) and len(cur) > 1 else None
                    if unit_id is not None and (cur_x is None or cur_y is None):
                        fallback_pos = get_unit_pos_from_obs(local_obs, int(unit_id))
                        if fallback_pos is not None:
                            cur_x, cur_y = int(fallback_pos[0]), int(fallback_pos[1])
                    if unit_id is not None and cur_x is not None and cur_y is not None:
                        self._unit_previous_tiles[int(unit_id)] = (int(cur_x), int(cur_y))
                new_obs, _, _, _ = self.tribes_env.step(best)
                return new_obs
            except Exception:
                return local_obs

        # ---- Turn 0 ----
        obs = ensure_bardur_turn(obs)

        # Harvest 1
        idx = find_action_idx(lambda a: a.get("type") == "RESOURCE_GATHERING" and "ANIMAL" in a.get("repr", ""))
        if idx is None:
            raise RuntimeError("Bardur opening failed: missing first ANIMAL harvest action.")
        obs, _, _, _ = self.tribes_env.step(idx)

        # Harvest 2
        idx = find_action_idx(lambda a: a.get("type") == "RESOURCE_GATHERING" and "ANIMAL" in a.get("repr", ""))
        if idx is None:
            raise RuntimeError("Bardur opening failed: missing second ANIMAL harvest action.")
        obs, _, _, _ = self.tribes_env.step(idx)

        # Level-up workshop
        idx = find_action_idx(lambda a: a.get("type") == "LEVEL_UP" and "WORKSHOP" in a.get("repr", ""))
        if idx is None:
            raise RuntimeError("Bardur opening failed: missing WORKSHOP level-up action.")
        obs, _, _, _ = self.tribes_env.step(idx)

        # Best move for starting warrior.
        try:
            obs = choose_and_execute_best_move(obs)
        except Exception:
            pass

        # End Turn 0.
        idx = find_action_idx(lambda a: a.get("type") == "END_TURN")
        if idx is None:
            raise RuntimeError("Bardur opening failed: missing END_TURN on Turn 0.")
        obs, _, _, _ = self.tribes_env.step(idx)

        # ---- Turn 1 ----
        obs = ensure_bardur_turn(obs)

        # Move first warrior again.
        first_unit_id = None
        try:
            own_units = []
            for key, unit in (obs.get("unit", {}) or {}).items():
                if isinstance(unit, dict) and int(unit.get("tribeId", -1)) == 0:
                    own_units.append((int(key), int(unit.get("x", -1)), int(unit.get("y", -1))))
            own_units.sort(key=lambda t: t[0])
            if own_units:
                first_unit_id = own_units[0][0]
        except Exception:
            first_unit_id = None

        try:
            obs = choose_and_execute_best_move(obs, target_unit_id=first_unit_id)
        except Exception:
            pass

        # Train/spawn second warrior.
        idx = find_action_idx(
            lambda a: (
                a.get("type") in ("SPAWN", "TRAIN")
                and "WARRIOR" in a.get("repr", "")
            )
        )
        if idx is None:
            # Fallback: any warrior spawn-like action.
            idx = find_action_idx(lambda a: "WARRIOR" in a.get("repr", ""))
        if idx is not None:
            obs, _, _, _ = self.tribes_env.step(idx)

        # End Turn 1.
        idx = find_action_idx(lambda a: a.get("type") == "END_TURN")
        if idx is None:
            raise RuntimeError("Bardur opening failed: missing END_TURN on Turn 1.")
        obs, _, _, _ = self.tribes_env.step(idx)

        # Bring environment to Bardur turn (start of Turn 2 for our side).
        obs = ensure_bardur_turn(obs)
        self._turn_count = 2

        return obs

    def _get_city_count(self, obs):
        tribes = obs.get("tribes", {})
        if isinstance(tribes, dict):
            tribe0 = tribes.get("0", {})
            if isinstance(tribe0, dict):
                return len(tribe0.get("citiesID", []))
        return 0

    def _get_avg_city_level(self, obs, tribe_id=0):
        city_map = obs.get("city", {})
        if not isinstance(city_map, dict):
            return 0.0
        levels = []
        for city in city_map.values():
            if not isinstance(city, dict):
                continue
            try:
                if int(city.get("tribeID", -1)) != int(tribe_id):
                    continue
                levels.append(int(city.get("level", 0)))
            except Exception:
                continue
        if not levels:
            return 0.0
        return float(sum(levels) / len(levels))

    def _get_tribe_type(self, obs, tribe_id=0):
        tribes = obs.get("tribes", {})
        if not isinstance(tribes, dict):
            return None
        tribe = tribes.get(str(int(tribe_id)), {})
        if not isinstance(tribe, dict):
            return None
        try:
            return int(tribe.get("type", -1))
        except Exception:
            return None

    def _raw_obs_researched_techs(self, obs, tribe_id=0):
        tribes = obs.get("tribes", {})
        if not isinstance(tribes, dict):
            return None
        tribe = tribes.get(str(int(tribe_id)), {})
        if not isinstance(tribe, dict):
            return None
        technology = tribe.get("technology", {})
        if not isinstance(technology, dict):
            return None
        researched = technology.get("researched", [])
        if not isinstance(researched, list):
            return None
        if len(researched) <= 0:
            return None
        out = set()
        for idx, flag in enumerate(researched):
            try:
                if not bool(flag):
                    continue
            except Exception:
                continue
            tech_name = self._obs_idx_to_tech_name.get(int(idx), None)
            if tech_name is None:
                continue
            out.add(str(tech_name).upper())
        return out

    def _get_effective_researched_techs(self, obs, tribe_id=0):
        raw = self._raw_obs_researched_techs(obs, tribe_id=tribe_id)
        if raw is not None:
            return set(raw)
        if int(tribe_id) == 0:
            return set(self._researched_techs_t10)
        return set()

    def _initialize_episode_researched_tech_state(self, obs):
        self._researched_techs_t10 = set()
        raw = self._raw_obs_researched_techs(obs, tribe_id=0)
        if raw is not None and len(raw) > 0:
            self._researched_techs_t10.update(str(t).upper() for t in raw)
            return
        tribe_type = self._get_tribe_type(obs, tribe_id=0)
        if tribe_type is None:
            return
        initial_tech = self.TRIBE_STARTING_TECH_BY_TYPE.get(int(tribe_type), None)
        if isinstance(initial_tech, str) and initial_tech:
            self._researched_techs_t10.add(str(initial_tech).upper())

    def _get_researched_tech_count(self, obs, tribe_id=0):
        return int(len(self._get_effective_researched_techs(obs, tribe_id=tribe_id)))

    def _has_researched_tech(self, obs, tech_name, tribe_id=0):
        if not isinstance(tech_name, str):
            return False
        target = str(tech_name).upper()
        return target in self._get_effective_researched_techs(obs, tribe_id=tribe_id)

    def _get_active_tribe_id(self, obs):
        try:
            return int(obs.get("activeTribeID", -1))
        except Exception:
            return -1

    def _compute_bardur_spt(self, obs):
        city_map = obs.get("city", {})
        spt = 0.0
        if not isinstance(city_map, dict):
            return spt
        for city in city_map.values():
            if not isinstance(city, dict):
                continue
            try:
                if int(city.get("tribeID", -1)) != 0:
                    continue
                spt += float(city.get("production", 0))
            except Exception:
                continue
        return spt

    def _get_bardur_stars(self, obs):
        tribes = obs.get("tribes", {})
        if not isinstance(tribes, dict):
            return 0.0
        tribe0 = tribes.get("0", {})
        if not isinstance(tribe0, dict):
            return 0.0
        try:
            return float(tribe0.get("star", 0.0))
        except Exception:
            return 0.0

    def _parse_city_id_from_action_repr(self, action_repr):
        if not isinstance(action_repr, str):
            return None
        m = re.search(r"by city\s+(-?\d+)", action_repr)
        if m is None:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _parse_resource_type_from_action_repr(self, action_repr):
        if not isinstance(action_repr, str):
            return None
        m = re.search(r":\s*([A-Z_]+)\s*$", action_repr.strip())
        if m is None:
            return None
        return m.group(1)

    def _parse_building_type_from_action_repr(self, action_repr):
        if not isinstance(action_repr, str):
            return None
        m = re.search(r":\s*([A-Z_]+)\s*$", action_repr.strip())
        if m is None:
            return None
        return m.group(1)

    def _parse_levelup_choice_from_action_repr(self, action_repr):
        if not isinstance(action_repr, str):
            return None
        m = re.search(r"\bbonus\s+([A-Z_]+)\s*$", action_repr.strip(), flags=re.IGNORECASE)
        if m is None:
            return None
        return str(m.group(1)).upper()

    def _normalize_vocab_index(self, idx, count):
        try:
            idx_i = int(idx)
            n = int(count)
        except Exception:
            return 0.0
        if n <= 1:
            return 0.0
        if idx_i < 0 or idx_i >= n:
            return 0.0
        return float(idx_i) / float(n - 1)

    def _resolve_action_resource_type(self, action):
        r_type = self._action_str(action, "resource_type", None)
        if r_type is None:
            r_type = self._parse_resource_type_from_action_repr(str(action.get("repr", "")))
        return str(r_type or "").upper()

    def _resolve_action_building_type(self, action):
        b_type = self._action_str(action, "building_type", None)
        if b_type is None:
            b_type = self._parse_building_type_from_action_repr(str(action.get("repr", "")))
        return str(b_type or "").upper()

    def _resolve_action_tech_type(self, action):
        tech_type = self._action_str(action, "tech_type", None)
        if tech_type is None:
            tech_type = self._parse_tech_type_from_action_repr(str(action.get("repr", "")))
        return str(tech_type or "").upper()

    def _resolve_action_levelup_choice(self, action):
        choice = self._action_str(action, "levelup_choice", None)
        if choice is None:
            choice = self._parse_levelup_choice_from_action_repr(str(action.get("repr", "")))
        return str(choice or "").upper()

    def _resolve_action_city_id(self, action, obs):
        city_id = self._action_int(action, "city_id", None)
        if city_id is not None:
            return int(city_id)
        city_x = self._action_int(action, "city_x", None)
        city_y = self._action_int(action, "city_y", None)
        if city_x is not None and city_y is not None:
            city_map = obs.get("city", {})
            if isinstance(city_map, dict):
                for cid_s, city in city_map.items():
                    if not isinstance(city, dict):
                        continue
                    try:
                        if int(city.get("x", -1)) == int(city_x) and int(city.get("y", -1)) == int(city_y):
                            return int(cid_s)
                    except Exception:
                        continue
        repr_s = str(action.get("repr", ""))
        parsed_city_id = self._parse_city_id_from_action_repr(repr_s)
        if parsed_city_id is not None:
            return int(parsed_city_id)
        return None

    def _resolve_action_city_info(self, action, obs):
        city_id = self._resolve_action_city_id(action, obs)
        if city_id is None:
            return None, None
        city_info = (obs.get("city", {}) or {}).get(str(int(city_id)), None)
        if not isinstance(city_info, dict):
            return int(city_id), None
        return int(city_id), city_info

    def _city_upgrade_progress_from_city_info(self, city_info):
        if not isinstance(city_info, dict):
            return 0.0, False, 0, 0
        try:
            pop = int(city_info.get("population", 0))
            pop_need = int(city_info.get("population_need", 0))
        except Exception:
            pop, pop_need = 0, 0
        denom = max(1, int(pop_need))
        progress = float(np.clip(float(pop) / float(denom), 0.0, 1.0))
        ready = int(pop) >= int(pop_need) if int(pop_need) > 0 else True
        return progress, bool(ready), int(pop), int(pop_need)

    def _expected_population_delta_from_action(self, action):
        a_type = str(action.get("type", "")).upper()
        if a_type == "RESOURCE_GATHERING":
            r_type = self._resolve_action_resource_type(action)
            meta = self._resource_cost_and_population_bonus(r_type)
            if meta is None:
                return 0
            _cost, bonus = meta
            return int(bonus)
        # Conservative default for uncertain rules.
        return 0

    def _expected_immediate_spt_delta_from_action(self, action):
        a_type = str(action.get("type", "")).upper()
        if a_type == "LEVEL_UP":
            choice = self._resolve_action_levelup_choice(action)
            # Deterministic from rules: WORKSHOP grants +1 production.
            if choice == "WORKSHOP":
                return 1.0
        # Conservative default when uncertain.
        return 0.0

    def _summarize_action_economy_expectation(self, action, obs):
        city_id, city_info = self._resolve_action_city_info(action, obs)
        progress_before, ready_before, pop_before, pop_need_before = self._city_upgrade_progress_from_city_info(city_info)
        pop_delta = int(self._expected_population_delta_from_action(action))
        spt_delta = float(self._expected_immediate_spt_delta_from_action(action))
        makes_level_up_available = False
        if city_info is not None and not bool(ready_before):
            post_pop = int(pop_before) + int(pop_delta)
            makes_level_up_available = int(post_pop) >= int(pop_need_before)
        is_level_up_claim = str(action.get("type", "")).upper() == "LEVEL_UP"
        if is_level_up_claim:
            makes_level_up_available = False
        return {
            "city_id": city_id,
            "progress_before": float(progress_before),
            "ready_before": bool(ready_before),
            "expected_population_delta": int(pop_delta),
            "expected_immediate_spt_delta": float(spt_delta),
            "makes_level_up_available": bool(makes_level_up_available),
            "is_level_up_claim": bool(is_level_up_claim),
        }

    def _resource_cost_and_population_bonus(self, resource_type):
        # Costs/bonuses aligned with TribesConfig defaults.
        table = {
            "ANIMAL": (2.0, 1),
            "FRUIT": (2.0, 1),
            "FISH": (2.0, 1),
            "WHALES": (0.0, 0),
            "ORE": (0.0, 0),
            "CROPS": (0.0, 0),
        }
        return table.get(resource_type, None)

    def _is_resource_gather_legal_for_upgrade(self, action, legal_actions, obs):
        repr_s = str(action.get("repr", ""))
        city_id = self._parse_city_id_from_action_repr(repr_s)
        if city_id is None:
            return True  # Defensive fallback: don't mask unknown format.

        city_info = (obs.get("city", {}) or {}).get(str(city_id), None)
        if not isinstance(city_info, dict):
            return True

        if int(city_info.get("tribeID", -1)) != 0:
            return False

        try:
            pop = int(city_info.get("population", 0))
            pop_need = int(city_info.get("population_need", 0))
        except Exception:
            return True

        missing_pop = max(0, pop_need - pop)
        if missing_pop <= 0:
            return True

        stars = float(self._get_bardur_stars(obs))
        candidates = []
        for la in legal_actions:
            if la.get("type") != "RESOURCE_GATHERING":
                continue
            la_repr = str(la.get("repr", ""))
            la_city_id = self._parse_city_id_from_action_repr(la_repr)
            if la_city_id != city_id:
                continue
            r_type = self._parse_resource_type_from_action_repr(la_repr)
            meta = self._resource_cost_and_population_bonus(r_type) if r_type is not None else None
            if meta is None:
                return True  # Unknown resource semantics; keep legal rather than over-mask.
            cost, bonus = meta
            candidates.append((float(cost), int(bonus)))

        if not candidates:
            return False

        max_pop_gain = sum(max(0, b) for _, b in candidates)
        if max_pop_gain < missing_pop:
            return False

        # 0/1 knapsack DP: minimum star cost needed to reach each population gain.
        inf = 1e9
        dp = [inf] * (max_pop_gain + 1)
        dp[0] = 0.0
        for cost, bonus in candidates:
            if bonus <= 0:
                continue
            for p in range(max_pop_gain - bonus, -1, -1):
                if dp[p] >= inf:
                    continue
                new_cost = dp[p] + cost
                if new_cost < dp[p + bonus]:
                    dp[p + bonus] = new_cost

        min_cost = min(dp[missing_pop:]) if missing_pop <= max_pop_gain else inf
        return min_cost <= stars + 1e-9

    def _resource_gather_action_completes_city_upgrade(self, action, obs):
        if str(action.get("type", "")).upper() != "RESOURCE_GATHERING":
            return False
        repr_s = str(action.get("repr", ""))
        city_id = self._parse_city_id_from_action_repr(repr_s)
        if city_id is None:
            return False
        city_info = (obs.get("city", {}) or {}).get(str(city_id), None)
        if not isinstance(city_info, dict):
            return False
        try:
            if int(city_info.get("tribeID", -1)) != 0:
                return False
            pop = int(city_info.get("population", 0))
            pop_need = int(city_info.get("population_need", 0))
        except Exception:
            return False
        missing_pop = max(0, int(pop_need) - int(pop))
        if missing_pop <= 0:
            return True
        r_type = self._parse_resource_type_from_action_repr(repr_s)
        meta = self._resource_cost_and_population_bonus(r_type) if r_type is not None else None
        if meta is None:
            return False
        _cost, bonus = meta
        return int(bonus) >= int(missing_pop)

    def _move_action_reveals_any_fog(self, action, obs):
        if str(action.get("type", "")).upper() != "MOVE":
            return False
        unit_id, _sx, _sy, dst_x, dst_y = self._extract_move_components(action, obs)
        if unit_id is None or dst_x is None or dst_y is None:
            return False
        try:
            revealed = int(self._estimate_newly_revealed_tiles_if_move(obs, int(unit_id), int(dst_x), int(dst_y)))
        except Exception:
            return False
        return int(revealed) >= 1

    def _parse_move_unit_and_dest_from_action_repr(self, action_repr):
        if not isinstance(action_repr, str):
            return None
        m = re.search(
            r"by unit\s+(-?\d+).*?\bto\s+(-?\d+)\s*:\s*(-?\d+)",
            action_repr,
            flags=re.IGNORECASE,
        )
        if m is not None:
            try:
                unit_id = int(m.group(1))
                dest_x = int(m.group(2))
                dest_y = int(m.group(3))
                return unit_id, dest_x, dest_y
            except Exception:
                return None

        nums = re.findall(r"-?\d+", action_repr)
        if len(nums) < 3:
            return None
        try:
            unit_id = int(nums[0])
            dest_x = int(nums[-2])
            dest_y = int(nums[-1])
            return unit_id, dest_x, dest_y
        except Exception:
            return None

    def _parse_unit_id_from_action_repr(self, action_repr):
        if not isinstance(action_repr, str):
            return None
        m = re.search(r"by unit\s+(-?\d+)", action_repr, flags=re.IGNORECASE)
        if m is None:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _board_dimensions_from_obs(self, obs):
        """Return board dimensions in Java runtime frame.

        Coordinate frames:
        - Java/runtime frame: (x, y). This is what unit positions, city positions,
          and legal action coordinates use.
        - Python index frame: (row, col) == (y, x), convenient for list indexing.

        In observation JSON, board arrays are emitted as board[x][y] (outer axis x),
        mirroring Java runtime coordinates. This helper returns (width_x, height_y).
        """
        if not isinstance(obs, dict):
            return None
        board = obs.get("board", {})
        terrain = board.get("terrain", []) if isinstance(board, dict) else []
        if not isinstance(terrain, list) or not terrain:
            return None

        width = len(terrain)
        height = 0
        for col in terrain:
            if isinstance(col, (list, tuple)):
                height = max(height, len(col))
        if width <= 0 or height <= 0:
            return None
        return width, height

    def java_to_py_coord(self, coord):
        """Convert Java/runtime coord (x, y) -> Python index coord (row=y, col=x)."""
        if coord is None:
            return None
        x, y = int(coord[0]), int(coord[1])
        return int(y), int(x)

    def py_to_java_coord(self, coord):
        """Convert Python index coord (row=y, col=x) -> Java/runtime coord (x, y)."""
        if coord is None:
            return None
        row, col = int(coord[0]), int(coord[1])
        return int(col), int(row)

    def _board_get_by_java_coord(self, obs, board_key, x, y, default=None):
        board = obs.get("board", {}) if isinstance(obs, dict) else {}
        grid = board.get(board_key, []) if isinstance(board, dict) else []
        if not isinstance(grid, list):
            return default
        dims = self._board_dimensions_from_obs(obs)
        if dims is None:
            return default
        width, height = dims
        jx = int(x)
        jy = int(y)
        if jx < 0 or jy < 0 or jx >= int(width) or jy >= int(height):
            return default
        try:
            col = grid[jx]
            if not isinstance(col, (list, tuple)):
                return default
            return col[jy] if jy < len(col) else default
        except Exception:
            return default

    def _board_get_int_by_java_coord(self, obs, board_key, x, y, default=None):
        v = self._board_get_by_java_coord(obs, board_key, x, y, default=default)
        if v is None:
            return default
        try:
            return int(v)
        except Exception:
            return default

    def _board_get_by_py_coord(self, obs, board_key, row, col, default=None):
        java_xy = self.py_to_java_coord((row, col))
        if java_xy is None:
            return default
        return self._board_get_by_java_coord(obs, board_key, int(java_xy[0]), int(java_xy[1]), default=default)

    def _get_unit_position(self, obs, unit_id):
        if not isinstance(obs, dict):
            return None
        units = obs.get("unit", {})
        if not isinstance(units, dict):
            return None
        unit = units.get(str(unit_id), None)
        if not isinstance(unit, dict):
            return None
        try:
            return int(unit.get("x", -1)), int(unit.get("y", -1))
        except Exception:
            return None

    def _is_move_destination_within_board(self, action, obs):
        parsed = self._parse_move_unit_and_dest_from_action_repr(str(action.get("repr", "")))
        if parsed is None:
            return False

        _, dest_x, dest_y = parsed
        dims = self._board_dimensions_from_obs(obs)
        if dims is None:
            # If dimensions are unavailable, avoid over-masking legal actions.
            return True
        width, height = dims
        return 0 <= int(dest_x) < width and 0 <= int(dest_y) < height

    def _get_visible_uncaptured_village_positions(self, obs):
        dims = self._board_dimensions_from_obs(obs)
        if dims is None:
            return set()
        width, height = dims

        # City occupancy in board.cityID is not always reliable for captured-village
        # tiles, so also exclude any coordinate present in runtime city actors.
        occupied_city_coords = set()
        city_map = obs.get("city", {})
        if isinstance(city_map, dict):
            for c in city_map.values():
                if not isinstance(c, dict):
                    continue
                try:
                    cx = int(c.get("x", -1))
                    cy = int(c.get("y", -1))
                except Exception:
                    continue
                if cx >= 0 and cy >= 0:
                    occupied_city_coords.add((cx, cy))

        village_positions = set()
        for x in range(int(width)):
            for y in range(int(height)):
                t_val = self._board_get_int_by_java_coord(obs, "terrain", int(x), int(y), default=-1)
                c_val = self._board_get_int_by_java_coord(obs, "cityID", int(x), int(y), default=-1)
                if t_val == 4 and c_val == -1 and (int(x), int(y)) not in occupied_city_coords:
                    village_positions.add((int(x), int(y)))
        self._validate_visible_uncaptured_villages(obs, village_positions)
        return village_positions

    def _validate_visible_uncaptured_villages(self, obs, village_positions):
        if not village_positions:
            return
        strict = os.environ.get("POLYVISION_STRICT_COORD_ASSERT", "0").lower() in ("1", "true", "yes", "on")
        legal_actions = self._current_legal_actions if isinstance(self._current_legal_actions, list) and len(self._current_legal_actions) > 0 else self.tribes_env.list_actions()
        capture_unit_ids = self._legal_capture_unit_ids(legal_actions)
        units = obs.get("unit", {}) if isinstance(obs, dict) else {}
        issues = []
        for vx, vy in village_positions:
            t_val = self._board_get_int_by_java_coord(obs, "terrain", int(vx), int(vy), default=-1)
            c_val = self._board_get_int_by_java_coord(obs, "cityID", int(vx), int(vy), default=-1)
            city_actor = self._city_actor_at_java_coord(obs, int(vx), int(vy))
            if t_val != 4:
                issues.append(f"terrain_not_village@{vx},{vy}: terrain={t_val}")
            if c_val != -1:
                issues.append(f"city_id_not_neutral@{vx},{vy}: cityID={c_val}")
            if city_actor is not None:
                issues.append(f"occupied_by_city_actor@{vx},{vy}: city={city_actor}")

            # If an owned fresh-by-legality unit is standing here, CAPTURE should exist.
            if isinstance(units, dict):
                for unit_id_s, unit in units.items():
                    if not isinstance(unit, dict):
                        continue
                    try:
                        if int(unit.get("tribeId", -1)) != 0:
                            continue
                        ux = int(unit.get("x", -1))
                        uy = int(unit.get("y", -1))
                        uid = int(unit_id_s)
                    except Exception:
                        continue
                    if ux != int(vx) or uy != int(vy):
                        continue
                    fresh_proxy = self._unit_has_any_legal_move_or_capture(uid, obs, legal_actions=legal_actions)
                    if fresh_proxy and uid not in capture_unit_ids:
                        issues.append(f"fresh_owned_unit_without_capture@{vx},{vy}: unit_id={uid}")
        if issues:
            preview = "; ".join(issues[:8])
            if strict:
                raise RuntimeError(f"Visible village coordinate validation failed: {preview}")
            if self._is_debug_info_mode():
                print(f"[COORD_VALIDATE] {preview}")

    def _city_actor_at_java_coord(self, obs, x, y):
        city_map = obs.get("city", {}) if isinstance(obs, dict) else {}
        if not isinstance(city_map, dict):
            return None
        for cid, c in city_map.items():
            if not isinstance(c, dict):
                continue
            try:
                cx = int(c.get("x", -1))
                cy = int(c.get("y", -1))
            except Exception:
                continue
            if cx == int(x) and cy == int(y):
                out = dict(c)
                out["city_id"] = str(cid)
                return out
        return None

    def _unit_has_any_legal_move_or_capture(self, unit_id, obs, legal_actions=None):
        if unit_id is None:
            return False
        legal = legal_actions
        if not isinstance(legal, list):
            legal = self._current_legal_actions if isinstance(self._current_legal_actions, list) and len(self._current_legal_actions) > 0 else self.tribes_env.list_actions()
        target_uid = int(unit_id)
        for a in legal:
            a_type = str(a.get("type", "")).upper()
            if a_type not in ("MOVE", "CAPTURE"):
                continue
            uid = self._action_int(a, "unit_id", None)
            if uid is None:
                uid = self._parse_unit_id_from_action_repr(str(a.get("repr", "")))
            if uid is None:
                continue
            if int(uid) == target_uid:
                return True
        return False

    def _build_village_lookup_mask(self, obs, village_positions):
        dims = self._board_dimensions_from_obs(obs)
        if dims is None:
            return None
        width, height = int(dims[0]), int(dims[1])
        if width <= 0 or height <= 0:
            return None
        mask = np.zeros((height, width), dtype=np.bool_)
        for vx, vy in village_positions:
            if 0 <= int(vx) < width and 0 <= int(vy) < height:
                mask[int(vy), int(vx)] = True
        return mask

    def _is_move_to_visible_uncaptured_village(self, action, obs, village_positions=None, village_mask=None):
        parsed = self._parse_move_unit_and_dest_from_action_repr(str(action.get("repr", "")))
        if parsed is None:
            return False
        unit_id, dest_x, dest_y = parsed

        units = obs.get("unit", {})
        unit = units.get(str(unit_id), None) if isinstance(units, dict) else None
        if not isinstance(unit, dict):
            return False
        if int(unit.get("tribeId", -1)) != 0:
            return False

        if village_positions is None:
            village_positions = self._get_visible_uncaptured_village_positions(obs)
        if not village_positions:
            return False

        if isinstance(village_mask, np.ndarray) and village_mask.ndim == 2:
            h, w = village_mask.shape
            dx = int(dest_x)
            dy = int(dest_y)
            if 0 <= dx < int(w) and 0 <= dy < int(h):
                return bool(village_mask[int(dy), int(dx)])
            return False

        return (int(dest_x), int(dest_y)) in village_positions

    def _is_capture_of_village(self, action, obs):
        try:
            action_repr = str(action.get("repr", ""))
            if "VILLAGE" not in action_repr.upper():
                return False

            unit_id = self._parse_unit_id_from_action_repr(action_repr)
            if unit_id is None:
                return False

            units = obs.get("unit", {})
            unit = units.get(str(unit_id), None) if isinstance(units, dict) else None
            if not isinstance(unit, dict):
                return False
            if int(unit.get("tribeId", -1)) != 0:
                return False

            ux = int(unit.get("x", -1))
            uy = int(unit.get("y", -1))
            if ux < 0 or uy < 0:
                return False
            village_positions = self._get_visible_uncaptured_village_positions(obs)
            return (int(ux), int(uy)) in village_positions
        except Exception:
            return False

    def _force_non_bardur_turns_to_end(self, obs, max_loops=16):
        forced = 0
        java_done = False
        local_obs = obs
        while self._get_active_tribe_id(local_obs) not in (0, -1) and forced < max_loops:
            legal = self.tribes_env.list_actions()
            end_idx = next((i for i, a in enumerate(legal) if a.get("type") == "END_TURN"), None)
            if end_idx is None:
                break
            local_obs, _, done, _ = self.tribes_env.step(end_idx)
            java_done = java_done or bool(done)
            forced += 1
        return local_obs, forced, java_done

    def _get_owned_unit_tiles(self, obs):
        out = set()
        units = obs.get("unit", {})
        if not isinstance(units, dict):
            return out
        for unit in units.values():
            if not isinstance(unit, dict):
                continue
            if int(unit.get("tribeId", -1)) != 0:
                continue
            x = int(unit.get("x", -1))
            y = int(unit.get("y", -1))
            if x >= 0 and y >= 0:
                out.add((x, y))
        return out

    def _get_owned_unit_count(self, obs):
        return len(self._get_owned_unit_tiles(obs))

    def _get_tribe_stars(self, obs, tribe_id=0):
        tribes = obs.get("tribes", {})
        if not isinstance(tribes, dict):
            return 0
        t = tribes.get(str(int(tribe_id)), {})
        if not isinstance(t, dict):
            return 0
        try:
            return int(t.get("star", 0))
        except Exception:
            return 0

    def _has_visible_uncaptured_village(self, obs):
        return len(self._get_visible_uncaptured_village_positions(obs)) > 0

    def _has_unit_on_visible_uncaptured_village(self, obs):
        village_positions = self._get_visible_uncaptured_village_positions(obs)
        if not village_positions:
            return False

        for ux, uy in self._get_owned_unit_tiles(obs):
            if (int(ux), int(uy)) in village_positions:
                return True
        return False

    def _legal_capture_unit_ids(self, legal_actions):
        out = set()
        if not isinstance(legal_actions, list):
            return out
        for a in legal_actions:
            if str(a.get("type", "")).upper() != "CAPTURE":
                continue
            uid = self._action_int(a, "unit_id", None)
            if uid is None:
                uid = self._parse_unit_id_from_action_repr(str(a.get("repr", "")))
            if uid is None:
                continue
            out.add(int(uid))
        return out

    def _owned_units_on_visible_uncaptured_village_without_capture_from_sets(
        self,
        obs,
        village_positions,
        capture_unit_ids,
    ):
        if not village_positions:
            return set()
        out = set()
        units = obs.get("unit", {})
        if not isinstance(units, dict):
            return out
        for unit_id_s, unit in units.items():
            if not isinstance(unit, dict):
                continue
            try:
                if int(unit.get("tribeId", -1)) != 0:
                    continue
                uid = int(unit_id_s)
                ux = int(unit.get("x", -1))
                uy = int(unit.get("y", -1))
            except Exception:
                continue
            if ux < 0 or uy < 0:
                continue
            if (ux, uy) in village_positions and int(uid) not in capture_unit_ids:
                out.add(int(uid))
        return out

    def _owned_units_on_visible_uncaptured_village_without_capture(self, obs, legal_actions):
        village_positions = self._get_visible_uncaptured_village_positions(obs)
        capture_unit_ids = self._legal_capture_unit_ids(legal_actions)
        return self._owned_units_on_visible_uncaptured_village_without_capture_from_sets(
            obs,
            village_positions,
            capture_unit_ids,
        )

    def _build_step_legal_action_summary(self, legal_actions, obs, visible_villages_before=None):
        if not isinstance(legal_actions, list):
            legal_actions = []
        if not isinstance(obs, dict):
            obs = {}
        if visible_villages_before is None:
            visible_villages_before = self._get_visible_uncaptured_village_positions(obs)
        visible_villages_before = set(visible_villages_before or set())

        per_type_counts = {}
        capture_unit_ids = set()
        completion_gather_raw_indices = set()
        move_to_visible_village_raw_indices = set()
        move_reveals_fog_raw_indices = set()
        move_reduces_village_distance_raw_indices = set()
        useful_move_raw_indices = set()
        unit_ids_with_adjacent_fog_move = set()

        units = obs.get("unit", {})
        if not isinstance(units, dict):
            units = {}

        for raw_idx, action in enumerate(legal_actions):
            a_type = str(action.get("type", "UNKNOWN")).upper()
            per_type_counts[a_type] = int(per_type_counts.get(a_type, 0) + 1)

            if a_type == "CAPTURE":
                uid = self._action_int(action, "unit_id", None)
                if uid is None:
                    uid = self._parse_unit_id_from_action_repr(str(action.get("repr", "")))
                if uid is not None:
                    capture_unit_ids.add(int(uid))
                continue

            if a_type == "RESOURCE_GATHERING":
                if self._resource_gather_action_completes_city_upgrade(action, obs):
                    completion_gather_raw_indices.add(int(raw_idx))
                continue

            if a_type != "MOVE":
                continue

            move_uid, src_x, src_y, dst_x, dst_y = self._extract_move_components(action, obs)
            if move_uid is None or dst_x is None or dst_y is None:
                continue
            uid_key = str(int(move_uid))
            unit = units.get(uid_key, None)
            owns_unit = isinstance(unit, dict) and int(unit.get("tribeId", -1)) == 0

            move_to_visible_village = bool(
                owns_unit and (int(dst_x), int(dst_y)) in visible_villages_before
            )
            if move_to_visible_village:
                move_to_visible_village_raw_indices.add(int(raw_idx))

            move_reveals_fog = self._move_action_reveals_any_fog(action, obs)
            if move_reveals_fog:
                move_reveals_fog_raw_indices.add(int(raw_idx))
            if self._tile_has_adjacent_fog(obs, int(dst_x), int(dst_y)):
                unit_ids_with_adjacent_fog_move.add(int(move_uid))

            move_reduces_village_distance = False
            if visible_villages_before:
                if src_x is None or src_y is None:
                    pos = self._unit_position_by_id(obs, move_uid)
                    if pos is not None:
                        src_x, src_y = int(pos[0]), int(pos[1])
                dist_before = self._min_manhattan_distance(
                    (int(src_x), int(src_y)),
                    visible_villages_before,
                ) if src_x is not None and src_y is not None else None
                dist_after = self._min_manhattan_distance((int(dst_x), int(dst_y)), visible_villages_before)
                if dist_before is not None and dist_after is not None:
                    move_reduces_village_distance = int(dist_after) < int(dist_before)
            if move_reduces_village_distance:
                move_reduces_village_distance_raw_indices.add(int(raw_idx))

            if move_to_visible_village or move_reveals_fog or move_reduces_village_distance:
                useful_move_raw_indices.add(int(raw_idx))

        return {
            "per_type_counts": per_type_counts,
            "capture_unit_ids": capture_unit_ids,
            "legal_capture_exists": bool("CAPTURE" in per_type_counts),
            "legal_level_up_exists": bool("LEVEL_UP" in per_type_counts),
            "completion_gather_raw_indices": completion_gather_raw_indices,
            "completion_gather_available": bool(len(completion_gather_raw_indices) > 0),
            "move_to_visible_village_raw_indices": move_to_visible_village_raw_indices,
            "move_reveals_fog_raw_indices": move_reveals_fog_raw_indices,
            "move_reduces_village_distance_raw_indices": move_reduces_village_distance_raw_indices,
            "useful_move_raw_indices": useful_move_raw_indices,
            "legal_move_onto_visible_village_exists": bool(len(move_to_visible_village_raw_indices) > 0),
            "legal_useful_move_exists": bool(len(useful_move_raw_indices) > 0),
            "unit_ids_with_adjacent_fog_move": unit_ids_with_adjacent_fog_move,
            "visible_villages_before": visible_villages_before,
        }

    def _assert_legal_summary_equivalence(self, legal_actions, obs, visible_villages_before, summary):
        legacy_capture_exists = any(str(a.get("type", "")).upper() == "CAPTURE" for a in legal_actions)
        legacy_level_up_exists = any(str(a.get("type", "")).upper() == "LEVEL_UP" for a in legal_actions)
        legacy_completion_indices = set(
            int(i)
            for i, a in enumerate(legal_actions)
            if str(a.get("type", "")).upper() == "RESOURCE_GATHERING"
            and self._resource_gather_action_completes_city_upgrade(a, obs)
        )
        legacy_move_to_visible = set()
        legacy_useful = set()
        legacy_adjacent_fog_unit_ids = set()
        for i, a in enumerate(legal_actions):
            if str(a.get("type", "")).upper() != "MOVE":
                continue
            move_to_visible = self._is_move_to_visible_uncaptured_village(a, obs)
            if move_to_visible:
                legacy_move_to_visible.add(int(i))
            move_reveals_fog = self._move_action_reveals_any_fog(a, obs)
            move_uid, _sx, _sy, dst_x, dst_y = self._extract_move_components(a, obs)
            if move_uid is not None and dst_x is not None and dst_y is not None:
                if self._tile_has_adjacent_fog(obs, int(dst_x), int(dst_y)):
                    legacy_adjacent_fog_unit_ids.add(int(move_uid))
            move_reduces = False
            if visible_villages_before:
                if move_uid is not None:
                    move_reduces = self._is_move_reducing_distance_to_targets(
                        a,
                        obs,
                        int(move_uid),
                        visible_villages_before,
                    )
            if move_to_visible or move_reveals_fog or move_reduces:
                legacy_useful.add(int(i))

        legacy_capture_unit_ids = self._legal_capture_unit_ids(legal_actions)
        if bool(summary.get("legal_capture_exists", False)) != bool(legacy_capture_exists):
            raise RuntimeError("LEGAL_SUMMARY_EQUIV: legal_capture_exists mismatch")
        if bool(summary.get("legal_level_up_exists", False)) != bool(legacy_level_up_exists):
            raise RuntimeError("LEGAL_SUMMARY_EQUIV: legal_level_up_exists mismatch")
        if set(summary.get("completion_gather_raw_indices", set())) != legacy_completion_indices:
            raise RuntimeError("LEGAL_SUMMARY_EQUIV: completion_gather_raw_indices mismatch")
        if bool(summary.get("completion_gather_available", False)) != bool(len(legacy_completion_indices) > 0):
            raise RuntimeError("LEGAL_SUMMARY_EQUIV: completion_gather_available mismatch")
        if set(summary.get("move_to_visible_village_raw_indices", set())) != legacy_move_to_visible:
            raise RuntimeError("LEGAL_SUMMARY_EQUIV: move_to_visible_village_raw_indices mismatch")
        if bool(summary.get("legal_move_onto_visible_village_exists", False)) != bool(len(legacy_move_to_visible) > 0):
            raise RuntimeError("LEGAL_SUMMARY_EQUIV: legal_move_onto_visible_village_exists mismatch")
        if set(summary.get("useful_move_raw_indices", set())) != legacy_useful:
            new_useful = set(summary.get("useful_move_raw_indices", set()))
            only_new = sorted(list(new_useful - legacy_useful))[:10]
            only_old = sorted(list(legacy_useful - new_useful))[:10]
            raise RuntimeError(
                "LEGAL_SUMMARY_EQUIV: useful_move_raw_indices mismatch "
                f"only_new={only_new} only_old={only_old} "
                f"new_n={len(new_useful)} old_n={len(legacy_useful)}"
            )
        if bool(summary.get("legal_useful_move_exists", False)) != bool(len(legacy_useful) > 0):
            raise RuntimeError("LEGAL_SUMMARY_EQUIV: legal_useful_move_exists mismatch")
        if set(summary.get("capture_unit_ids", set())) != legacy_capture_unit_ids:
            raise RuntimeError("LEGAL_SUMMARY_EQUIV: capture_unit_ids mismatch")
        if set(summary.get("unit_ids_with_adjacent_fog_move", set())) != legacy_adjacent_fog_unit_ids:
            raise RuntimeError("LEGAL_SUMMARY_EQUIV: unit_ids_with_adjacent_fog_move mismatch")

    def _min_manhattan_distance(self, origin, targets):
        if origin is None or not targets:
            return None
        ox, oy = origin
        best = None
        for tx, ty in targets:
            d = abs(int(ox) - int(tx)) + abs(int(oy) - int(ty))
            if best is None or d < best:
                best = d
        return best

    def _closest_owned_unit_to_targets(self, obs, targets):
        if not isinstance(obs, dict) or not targets:
            return None, None, None
        units = obs.get("unit", {})
        if not isinstance(units, dict):
            return None, None, None

        best_unit_id = None
        best_pos = None
        best_dist = None
        for unit_id_s, unit in units.items():
            if not isinstance(unit, dict):
                continue
            try:
                if int(unit.get("tribeId", -1)) != 0:
                    continue
                unit_id = int(unit_id_s)
                ux = int(unit.get("x", -1))
                uy = int(unit.get("y", -1))
            except Exception:
                continue
            if ux < 0 or uy < 0:
                continue
            d = self._min_manhattan_distance((ux, uy), targets)
            if d is None:
                continue
            if best_dist is None or d < best_dist:
                best_dist = int(d)
                best_unit_id = int(unit_id)
                best_pos = (int(ux), int(uy))
        return best_unit_id, best_pos, best_dist

    def _count_visible_tiles(self, obs):
        board = obs.get("board", {})
        terrain = board.get("terrain", [])
        if not terrain:
            return 0
        visible = 0
        for row in terrain:
            for val in row:
                try:
                    # TERRAIN 7 = FOG in this env; everything else is visible.
                    if int(val) != 7:
                        visible += 1
                except Exception:
                    continue
        return visible

    def _tile_has_adjacent_fog(self, obs, center_x, center_y):
        dims = self._board_dimensions_from_obs(obs)
        if dims is None:
            return False
        width, height = dims
        cx = int(center_x)
        cy = int(center_y)
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            x = cx + int(dx)
            y = cy + int(dy)
            if x < 0 or y < 0 or x >= int(width) or y >= int(height):
                continue
            if self._board_get_int_by_java_coord(obs, "terrain", int(x), int(y), default=-1) == 7:
                return True
        return False

    def _unit_had_any_legal_fog_revealing_move(self, legal_actions, obs, unit_id):
        if unit_id is None:
            return False
        if not isinstance(legal_actions, list):
            return False

        target_unit_id = int(unit_id)
        for action in legal_actions:
            if str(action.get("type", "")).upper() != "MOVE":
                continue
            move_unit_id, _sx, _sy, dst_x, dst_y = self._extract_move_components(action, obs)
            if move_unit_id is None or int(move_unit_id) != target_unit_id:
                continue
            if dst_x is None or dst_y is None:
                continue
            if self._tile_has_adjacent_fog(obs, int(dst_x), int(dst_y)):
                return True
        return False

    def _assert_all_units_in_bounds(self, obs, context="unknown"):
        dims = self._board_dimensions_from_obs(obs)
        if dims is None:
            return
        width, height = dims
        units = obs.get("unit", {})
        if not isinstance(units, dict):
            return

        offenders = []
        for unit_id, unit in units.items():
            if not isinstance(unit, dict):
                continue
            try:
                x = int(unit.get("x", -1))
                y = int(unit.get("y", -1))
            except Exception:
                continue
            if x < 0 or y < 0 or x >= width or y >= height:
                offenders.append((unit_id, x, y))

        if offenders:
            detail = ", ".join(f"id={uid} pos={x}:{y}" for uid, x, y in offenders)
            raise RuntimeError(
                f"Detected out-of-bounds unit(s) ({context}) on board {width}x{height}: {detail}"
            )
    
    def _dict_to_array(self, obs_dict):
        # Keep the original 438-entry observation prefix unchanged for compatibility.
        features = []
        board = obs_dict.get("board", {})

        terrain = board.get("terrain", [])
        if terrain:
            features.extend(np.array(terrain).flatten())

        unit_ids = board.get("unitID", [])
        if unit_ids:
            features.extend(np.array(unit_ids).flatten())

        city_ids = board.get("cityID", [])
        if city_ids:
            features.extend(np.array(city_ids).flatten())

        tribes_info = obs_dict.get("tribes", {})
        if isinstance(tribes_info, dict):
            tribe0 = tribes_info.get("0", {})
            if isinstance(tribe0, dict):
                features.append(tribe0.get("star", 0))
                features.append(tribe0.get("score", 0))
                features.append(len(tribe0.get("citiesID", [])))
                features.append(tribe0.get("nKills", 0))

        features.append(obs_dict.get("tick", 0))
        features.append(obs_dict.get("activeTribeID", 0))

        # Append visible-only resource map and normalized economy/state features.
        dims = self._board_dimensions_from_obs(obs_dict)
        if dims is not None:
            width, height = int(dims[0]), int(dims[1])
        else:
            width, height = 0, 0
        terrain_arr = np.full((width, height), -1, dtype=np.int16)
        resource_arr = np.full((width, height), -1, dtype=np.int16)
        try:
            terrain_raw = np.asarray(board.get("terrain", []), dtype=np.int16)
            if terrain_raw.shape == (width, height):
                terrain_arr = terrain_raw
        except Exception:
            pass
        try:
            resource_raw = np.asarray(board.get("resource", []), dtype=np.int16)
            if resource_raw.shape == (width, height):
                resource_arr = resource_raw
        except Exception:
            pass
        if width > 0 and height > 0:
            fog_mask = terrain_arr == 7
            masked_resource = np.array(resource_arr, copy=True)
            masked_resource[fog_mask] = -1
            resource_norm = np.clip((masked_resource.astype(np.float32) + 1.0) / 8.0, 0.0, 1.0)
            features.extend(resource_norm.flatten())

        current_stars = float(self._get_bardur_stars(obs_dict))
        current_spt = float(self._compute_bardur_spt(obs_dict))
        turn_count = float(self._turn_count)
        max_turns = float(max(1, int(self.MAX_TURNS)))
        turns_remaining_after_current = float(np.clip((max_turns - turn_count) / max_turns, 0.0, 1.0))
        turns_remaining_including_current = float(np.clip((max_turns - turn_count + 1.0) / max_turns, 0.0, 1.0))
        tech_has_organization = 1.0 if self._has_researched_tech(obs_dict, "ORGANIZATION", tribe_id=0) else 0.0
        tech_has_forestry = 1.0 if self._has_researched_tech(obs_dict, "FORESTRY", tribe_id=0) else 0.0
        tech_researched_count = float(self._get_researched_tech_count(obs_dict, tribe_id=0))

        city_map = obs_dict.get("city", {})
        owned_cities = []
        if isinstance(city_map, dict):
            for city in city_map.values():
                if not isinstance(city, dict):
                    continue
                try:
                    if int(city.get("tribeID", -1)) != 0:
                        continue
                except Exception:
                    continue
                owned_cities.append(city)
        city_count = len(owned_cities)
        city_levels = []
        city_progress = []
        ready_count = 0
        for city in owned_cities:
            try:
                city_levels.append(int(city.get("level", 0)))
            except Exception:
                city_levels.append(0)
            progress, ready, _pop, _need = self._city_upgrade_progress_from_city_info(city)
            city_progress.append(float(progress))
            if bool(ready):
                ready_count += 1

        avg_city_level = float(np.mean(city_levels)) if city_levels else 0.0
        max_city_level = float(np.max(city_levels)) if city_levels else 0.0
        mean_upgrade_progress = float(np.mean(city_progress)) if city_progress else 0.0
        max_upgrade_progress = float(np.max(city_progress)) if city_progress else 0.0
        upgrade_ready_frac = float(ready_count / max(1, city_count))
        any_level_up_available = 1.0 if ready_count > 0 else 0.0

        features.append(float(np.clip(current_stars / 50.0, 0.0, 1.0)))
        features.append(float(np.clip(current_spt / 30.0, 0.0, 1.0)))
        features.append(float(np.clip(turn_count / max_turns, 0.0, 1.0)))
        features.append(turns_remaining_after_current)
        features.append(turns_remaining_including_current)
        features.append(tech_has_organization)
        features.append(tech_has_forestry)
        features.append(float(np.clip(tech_researched_count / 24.0, 0.0, 1.0)))
        features.append(float(np.clip(float(city_count) / 6.0, 0.0, 1.0)))
        features.append(float(np.clip(avg_city_level / 5.0, 0.0, 1.0)))
        features.append(float(np.clip(max_city_level / 5.0, 0.0, 1.0)))
        features.append(float(np.clip(mean_upgrade_progress, 0.0, 1.0)))
        features.append(float(np.clip(max_upgrade_progress, 0.0, 1.0)))
        features.append(float(np.clip(upgrade_ready_frac, 0.0, 1.0)))
        features.append(any_level_up_available)

        return np.array(features, dtype=np.float32)
    
    def close(self):
        """Close the underlying environment"""
        if hasattr(self.tribes_env, 'close'):
            self.tribes_env.close()

register(
    id="Tribes-v0",
    entry_point="pol_env.Tribes.py.register_env:TribesGymWrapper",
    max_episode_steps=1000,
)
