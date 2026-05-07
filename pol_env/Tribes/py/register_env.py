import gymnasium as gym
import glob
import hashlib
import json
import numpy as np
import os
import pickle
import re
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

    # Phase1 village/city shaping.
    REVEAL_UNCAPTURED_VILLAGE_REWARD = 1.0
    MOVE_CLOSER_TO_VISIBLE_VILLAGE_REWARD = 0.5
    MOVE_ONTO_VILLAGE_REWARD = 1.0
    CAPTURE_CITY_BONUS_MIN = 4.0
    CAPTURE_CITY_BONUS_MAX = 8.0
    VISIBLE_VILLAGE_NEGLECT_PENALTY = -0.5
    VISIBLE_VILLAGE_NEGLECT_GRACE_TURNS = 2
    VILLAGE_BREADCRUMB_REWARD = 0.5
    FOG_CLEAR_REWARD_PER_TILE = 0.08
    FOG_CLEAR_REWARD_MAX_TILES = 5
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
        self._initial_visible_tiles = 0
        self._episode_fog_tiles_cleared = 0
        self._catalog = None
        self._catalog_fingerprint = ""
        self._last_step_canonical_diag = {}
        self._illegal_sample_count = 0
        self._fallback_end_turn_count = 0
        self._total_action_decisions = 0
        self._validation_mode = os.environ.get("POLYVISION_ACTION_VALIDATION_MODE", "0").lower() in ("1", "true", "yes", "on")
        self._info_mode = str(os.environ.get("POLYVISION_INFO_MODE", "fast")).strip().lower()
        if self._info_mode not in ("fast", "debug"):
            self._info_mode = "fast"
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
        episode_seed = self._resolve_episode_seed(seed=seed)
        level_file, level_index = self._select_level_for_reset(episode_seed)
        self._current_level_file = level_file
        self._current_level_index = int(level_index)
        self._last_reset_seed = int(episode_seed)
        obs = self.tribes_env.reset(self._current_level_file, self._last_reset_seed)
        self._episode_index += 1
        self._turn_count = 0
        obs = self._apply_bardur_opening(obs)
        self._starting_city_count = self._get_city_count(obs)
        self._last_city_count = self._starting_city_count
        self._moved_on_t0 = False
        self._visible_village_streak_turns = 0
        self._queued_village_capture_unit_ids = set()
        self._initial_visible_tiles = int(self._count_visible_tiles(obs))
        self._episode_fog_tiles_cleared = 0
        
        # Log action space info for debugging
        legal_actions = self.tribes_env.list_actions()
        action_mask, legal_id_to_raw_index, diag = self._build_action_mask_and_mapping(legal_actions, obs=obs)
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
        }
        if self._is_debug_info_mode():
            info["action_mask"] = action_mask
        else:
            info["legal_global_ids"] = np.flatnonzero(action_mask).astype(np.int32).tolist()
        info.update(self._diag_for_info(diag))
        if self._is_debug_info_mode():
            info["map_path"] = self._current_level_file
            info["map_id"] = os.path.basename(self._current_level_file)
            info["map_pool_index"] = int(self._current_level_index)
            info["map_pool_size"] = int(self._level_pool_size)
            info["episode_seed"] = int(self._last_reset_seed)
            info["level_selection_mode"] = self._level_selection_mode
            info["initial_visible_tiles"] = int(self._initial_visible_tiles)
        return self._dict_to_array(obs), self._sanitize_info_for_multiprocessing(info)
    
    def render(self, **kwargs):
        data = np.array(self.tribes_env.render("rgb_image"))
        return data

    def step(self, action):
        start_obs = getattr(self.tribes_env, "_last_obs", None)
        if not isinstance(start_obs, dict):
            start_obs = {}
        chosen_move_unit_id = None
        chosen_move_dest = None

        forced_pre_end_turns = 0
        if self._get_active_tribe_id(start_obs) != 0:
            start_obs, forced_pre_end_turns, _ = self._force_non_bardur_turns_to_end(start_obs)

        legal_actions = self._current_legal_actions if self._current_legal_actions is not None else []
        raw_count = int(self._current_raw_valid_actions)
        action_mask = self._current_action_mask
        legal_id_to_raw_index = self._current_legal_id_to_raw_index
        diag = self._current_diag

        if raw_count == 0 or not legal_actions or action_mask is None or not legal_id_to_raw_index:
            # Recover from any stale cache by rebuilding once.
            legal_actions = self.tribes_env.list_actions()
            raw_count = len(legal_actions)
            action_mask, legal_id_to_raw_index, diag = self._build_action_mask_and_mapping(legal_actions, obs=start_obs)
            self._current_legal_actions = legal_actions
            self._current_action_mask = action_mask
            self._current_legal_id_to_raw_index = legal_id_to_raw_index
            self._current_diag = diag
            self._current_raw_valid_actions = int(raw_count)

        if raw_count == 0:
            raise RuntimeError("No legal actions available from Java environment.")

        sampled_action = int(action)
        selected_global_id = sampled_action
        illegal_sampled_global_id = False
        fallback_to_end_turn = False
        if selected_global_id in legal_id_to_raw_index:
            selected_raw_action = int(legal_id_to_raw_index[selected_global_id])
        else:
            illegal_sampled_global_id = True
            fallback_to_end_turn = True
            self._illegal_sample_count += 1
            self._fallback_end_turn_count += 1
            selected_global_id = int(self._catalog.id_end_turn()) if self._catalog is not None else 0
            selected_raw_action = legal_id_to_raw_index.get(selected_global_id, None)
            if selected_raw_action is None:
                end_turn_idx = next((i for i, a in enumerate(legal_actions) if a.get("type") == "END_TURN"), 0)
                selected_raw_action = int(end_turn_idx)

        selected_action_type = legal_actions[selected_raw_action].get("type", "UNKNOWN")
        self._total_action_decisions += 1

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

        if selected_action_type == "END_TURN":
            self._turn_count += 1
        elif selected_action_type == "MOVE":
            selected_action = legal_actions[selected_raw_action]
            parsed_move = self._parse_move_unit_and_dest_from_action_repr(str(selected_action.get("repr", "")))
            if parsed_move is not None:
                chosen_move_unit_id = int(parsed_move[0])
                chosen_move_dest = (int(parsed_move[1]), int(parsed_move[2]))
            if not self._is_move_destination_within_board(selected_action, start_obs):
                end_turn_idx = next((i for i, a in enumerate(legal_actions) if a.get("type") == "END_TURN"), 0)
                selected_raw_action = end_turn_idx
                selected_action_type = legal_actions[selected_raw_action].get("type", "UNKNOWN")
                selected_global_id = int(self._catalog.id_end_turn()) if self._catalog is not None else selected_global_id
                fallback_to_end_turn = True
                self._fallback_end_turn_count += 1
                if selected_action_type == "END_TURN":
                    self._turn_count += 1

        obs, _, done, info = self.tribes_env.step(selected_raw_action)
        obs_after_selected = obs
        java_done = bool(done)
        self._assert_all_units_in_bounds(obs, context="post_selected_action")

        forced_post_end_turns = 0
        if self._get_active_tribe_id(obs) != 0:
            obs, forced_post_end_turns, forced_done = self._force_non_bardur_turns_to_end(obs)
            java_done = java_done or bool(forced_done)
            self._assert_all_units_in_bounds(obs, context="post_forced_end_turns")

        current_bardur_spt = self._compute_bardur_spt(obs)
        base_delta_spt = float(current_bardur_spt - prev_bardur_spt)
        current_city_count = self._get_city_count(obs)
        reward_adjustment = 0.0

        capture_city_bonus = 0.0
        if current_city_count > self._last_city_count:
            n_new_cities = current_city_count - self._last_city_count
            # +4.0 per newly captured city in this step, capped at +8.0.
            capture_city_bonus = min(
                self.CAPTURE_CITY_BONUS_MAX,
                max(0, n_new_cities) * self.CAPTURE_CITY_BONUS_MIN,
            )
        reward_adjustment += capture_city_bonus

        reveal_uncaptured_village_reward = 0.0
        move_closer_to_visible_village_reward = 0.0
        move_onto_village_reward = 0.0
        visible_village_neglect_penalty = 0.0
        village_breadcrumb_reward = 0.0
        fog_clearance_reward = 0.0
        fog_tiles_cleared = 0

        visible_villages_before = self._get_visible_uncaptured_village_positions(start_obs)
        visible_villages_after_selected = self._get_visible_uncaptured_village_positions(obs_after_selected)
        newly_revealed_villages = visible_villages_after_selected - visible_villages_before
        if len(newly_revealed_villages) > 0:
            reveal_uncaptured_village_reward = self.REVEAL_UNCAPTURED_VILLAGE_REWARD * float(len(newly_revealed_villages))

        visible_uncaptured_village = self._has_visible_uncaptured_village(obs)
        has_second_village = current_city_count >= (self._starting_city_count + 1)
        unit_on_visible_uncaptured_village = self._has_unit_on_visible_uncaptured_village(obs)

        if selected_action_type == "MOVE":
            vis_before = self._count_visible_tiles(start_obs)
            vis_after = self._count_visible_tiles(obs_after_selected)
            fog_tiles_cleared = max(0, vis_after - vis_before)
            if fog_tiles_cleared > 0:
                fog_clearance_reward = min(
                    self.FOG_CLEAR_REWARD_MAX_TILES,
                    int(fog_tiles_cleared),
                ) * self.FOG_CLEAR_REWARD_PER_TILE
                self._episode_fog_tiles_cleared += int(fog_tiles_cleared)

            moved_unit_before = self._get_unit_position(start_obs, chosen_move_unit_id) if chosen_move_unit_id is not None else None
            moved_unit_after = self._get_unit_position(obs_after_selected, chosen_move_unit_id) if chosen_move_unit_id is not None else None
            dist_before = self._min_manhattan_distance(moved_unit_before, visible_villages_before)
            dist_after = self._min_manhattan_distance(moved_unit_after, visible_villages_after_selected)
            if dist_before is not None and dist_after is not None and dist_after < dist_before:
                move_closer_to_visible_village_reward = self.MOVE_CLOSER_TO_VISIBLE_VILLAGE_REWARD

            if moved_unit_after is not None and (
                moved_unit_after in visible_villages_after_selected
                or (moved_unit_after[1], moved_unit_after[0]) in visible_villages_after_selected
            ):
                move_onto_village_reward = self.MOVE_ONTO_VILLAGE_REWARD

        if selected_action_type == "END_TURN":
            if visible_uncaptured_village and not has_second_village:
                self._visible_village_streak_turns += 1
            else:
                self._visible_village_streak_turns = 0

            if self._visible_village_streak_turns > self.VISIBLE_VILLAGE_NEGLECT_GRACE_TURNS:
                visible_village_neglect_penalty = self.VISIBLE_VILLAGE_NEGLECT_PENALTY

            if unit_on_visible_uncaptured_village:
                village_breadcrumb_reward = self.VILLAGE_BREADCRUMB_REWARD

        reward_adjustment += reveal_uncaptured_village_reward
        reward_adjustment += move_closer_to_visible_village_reward
        reward_adjustment += move_onto_village_reward
        reward_adjustment += visible_village_neglect_penalty
        reward_adjustment += village_breadcrumb_reward
        reward_adjustment += fog_clearance_reward

        # Phase 1 override: ignore Java terminal state and control horizon purely in Python.
        terminated = False
        truncated = self._turn_count >= self.MAX_TURNS

        reward = float(base_delta_spt) + reward_adjustment

        # Build the next-state legal mask and mapping diagnostics (for policy step t+1).
        post_legal_actions = self.tribes_env.list_actions()
        post_raw_count = len(post_legal_actions)
        post_action_mask, post_legal_id_to_raw_index, post_diag = self._build_action_mask_and_mapping(post_legal_actions, obs=obs)
        self._current_legal_actions = post_legal_actions
        self._current_action_mask = post_action_mask
        self._current_legal_id_to_raw_index = post_legal_id_to_raw_index
        self._current_diag = post_diag
        self._current_raw_valid_actions = int(post_raw_count)

        info["valid_actions"] = int(np.sum(post_action_mask))
        info["raw_valid_actions"] = int(post_raw_count)
        info["info_mode"] = self._info_mode
        if self._is_debug_info_mode():
            info["action_mask"] = post_action_mask
        else:
            info["legal_global_ids"] = np.flatnonzero(post_action_mask).astype(np.int32).tolist()
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
        info["turn_count"] = self._turn_count
        info["city_count"] = current_city_count
        info["fog_tiles_cleared_total"] = int(self._episode_fog_tiles_cleared)
        info["delta_spt"] = float(base_delta_spt)
        info["spt"] = float(current_bardur_spt)
        info["reward"] = float(reward)
        info["turn"] = int(self._turn_count)
        info["unit_count"] = int(self._get_owned_unit_count(obs))
        info["stars"] = int(self._get_tribe_stars(obs, tribe_id=0))
        info.update(self._diag_for_info(post_diag))

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
            info["reward_visible_village_neglect_penalty"] = float(visible_village_neglect_penalty)
            info["reward_village_breadcrumb"] = float(village_breadcrumb_reward)
            info["reward_fog_clearance"] = float(fog_clearance_reward)
            info["reward_reveal_uncaptured_village"] = float(reveal_uncaptured_village_reward)
            info["reward_move_closer_to_visible_village"] = float(move_closer_to_visible_village_reward)
            info["reward_move_onto_village"] = float(move_onto_village_reward)
            info["newly_revealed_uncaptured_villages"] = int(len(newly_revealed_villages))
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
        self._last_city_count = current_city_count
        if selected_action_type == "MOVE" and self._turn_count == 0:
            self._moved_on_t0 = True
        if current_city_count >= 2:
            self._queued_village_capture_unit_ids = set()

        info = self._sanitize_info_for_multiprocessing(info)
        return self._dict_to_array(obs), reward, terminated, truncated, info

    def _is_debug_info_mode(self):
        return self._info_mode == "debug"

    def _diag_for_info(self, diag):
        if not isinstance(diag, dict):
            return {}
        if self._is_debug_info_mode():
            return dict(diag)
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

    def _filter_allowed_raw_indices(self, legal_actions, obs):
        allowed_indices = []
        for idx, a in enumerate(legal_actions):
            a_type = str(a.get("type", "")).upper()
            if a_type not in self.ALLOWED_ACTION_TYPES:
                continue
            if a_type == "MOVE" and not self._is_move_destination_within_board(a, obs):
                continue
            if a_type == "RESOURCE_GATHERING":
                if not self._is_resource_gather_legal_for_upgrade(a, legal_actions, obs):
                    continue
            allowed_indices.append(idx)

        # Hard guardrail: if <2 cities, prioritize village-capture lines.
        if self._get_city_count(obs) < 2 and allowed_indices:
            forced_village_captures = []
            for idx in allowed_indices:
                a = legal_actions[idx]
                if a.get("type") != "CAPTURE":
                    continue
                if self._is_capture_of_village(a, obs):
                    forced_village_captures.append(idx)
            if forced_village_captures:
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
            else:
                self._queued_village_capture_unit_ids = set()

            forced_village_moves = []
            if not forced_village_captures:
                for idx in allowed_indices:
                    a = legal_actions[idx]
                    if a.get("type") != "MOVE":
                        continue
                    if self._is_move_to_visible_uncaptured_village(a, obs):
                        forced_village_moves.append(idx)
                if forced_village_moves:
                    allowed_indices = forced_village_moves
        else:
            self._queued_village_capture_unit_ids = set()

        return allowed_indices

    def _build_action_mask_and_mapping(self, legal_actions=None, obs=None):
        if legal_actions is None:
            legal_actions = self.tribes_env.list_actions()
        if obs is None:
            obs = getattr(self.tribes_env, "_last_obs", {})
        if not isinstance(obs, dict):
            obs = {}
        if self._catalog is None:
            raise RuntimeError("Global action catalog is not initialized.")

        allowed_indices = self._filter_allowed_raw_indices(legal_actions, obs)
        legal_id_to_raw_index = {}
        uncanonicalized = []
        collisions = []
        per_type_counts = {}

        for raw_idx in allowed_indices:
            action = legal_actions[raw_idx]
            a_type = str(action.get("type", "UNKNOWN")).upper()
            per_type_counts[a_type] = per_type_counts.get(a_type, 0) + 1
            gid, reason = self._canonicalize_action_to_global_id(action, obs)
            if gid is None:
                uncanonicalized.append({"raw_idx": int(raw_idx), "type": a_type, "reason": str(reason), "repr": str(action.get("repr", ""))})
                continue
            if gid in legal_id_to_raw_index and legal_id_to_raw_index[gid] != raw_idx:
                collisions.append({"global_id": int(gid), "raw_a": int(legal_id_to_raw_index[gid]), "raw_b": int(raw_idx), "type": a_type})
            else:
                legal_id_to_raw_index[int(gid)] = int(raw_idx)

        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for gid in legal_id_to_raw_index.keys():
            if 0 <= int(gid) < self.action_space.n:
                mask[int(gid)] = 1

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
        if not isinstance(obs, dict):
            return None
        board = obs.get("board", {})
        terrain = board.get("terrain", []) if isinstance(board, dict) else []
        if not isinstance(terrain, list) or not terrain:
            return None

        height = len(terrain)
        width = 0
        for row in terrain:
            if isinstance(row, (list, tuple)):
                width = max(width, len(row))
        if width <= 0 or height <= 0:
            return None
        return width, height

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
        board = obs.get("board", {})
        terrain = board.get("terrain", [])
        city_ids = board.get("cityID", [])
        if not terrain or not city_ids:
            return set()

        village_positions = set()
        for y in range(len(terrain)):
            row_t = terrain[y]
            row_c = city_ids[y] if y < len(city_ids) else []
            for x in range(len(row_t)):
                try:
                    t_val = int(row_t[x])
                    c_val = int(row_c[x]) if x < len(row_c) else -1
                except Exception:
                    continue
                if t_val == 4 and c_val == -1:
                    village_positions.add((x, y))
        return village_positions

    def _is_move_to_visible_uncaptured_village(self, action, obs):
        parsed = self._parse_move_unit_and_dest_from_action_repr(str(action.get("repr", "")))
        if parsed is None:
            return False
        unit_id, dest_a, dest_b = parsed

        units = obs.get("unit", {})
        unit = units.get(str(unit_id), None) if isinstance(units, dict) else None
        if not isinstance(unit, dict):
            return False
        if int(unit.get("tribeId", -1)) != 0:
            return False

        village_positions = self._get_visible_uncaptured_village_positions(obs)
        if not village_positions:
            return False

        # Some action repr variants encode destination as "x:y" while others
        # effectively behave as "y:x" relative to board arrays in Python.
        return (dest_a, dest_b) in village_positions or (dest_b, dest_a) in village_positions

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
            return (ux, uy) in village_positions or (uy, ux) in village_positions
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
            if (ux, uy) in village_positions or (uy, ux) in village_positions:
                return True
        return False

    def _min_manhattan_distance(self, origin, targets):
        if origin is None or not targets:
            return None
        ox, oy = origin
        best = None
        for tx, ty in targets:
            d_xy = abs(int(ox) - int(tx)) + abs(int(oy) - int(ty))
            d_yx = abs(int(ox) - int(ty)) + abs(int(oy) - int(tx))
            d = min(d_xy, d_yx)
            if best is None or d < best:
                best = d
        return best

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
        # convert your complex dict observation to flat array
        # this is the key missing piece - you need to flatten your board state
        features = []
        
        board = obs_dict.get("board", {})
        
        # Extract terrain as flattened array
        terrain = board.get("terrain", [])
        if terrain:
            features.extend(np.array(terrain).flatten())
        
        # Extract unit IDs as flattened array  
        unit_ids = board.get("unitID", [])
        if unit_ids:
            features.extend(np.array(unit_ids).flatten())
            
        # Extract city IDs as flattened array
        city_ids = board.get("cityID", [])
        if city_ids:
            features.extend(np.array(city_ids).flatten())
            
        # Add tribe information
        tribes_info = obs_dict.get("tribes", {})
        if isinstance(tribes_info, dict):
            tribe0 = tribes_info.get("0", {})
            if isinstance(tribe0, dict):
                # Add some key tribe stats as features
                features.append(tribe0.get("star", 0))
                features.append(tribe0.get("score", 0))
                features.append(len(tribe0.get("citiesID", [])))
                features.append(tribe0.get("nKills", 0))
            
        # Add game state info
        features.append(obs_dict.get("tick", 0))
        features.append(obs_dict.get("activeTribeID", 0))
        
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
