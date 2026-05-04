import gymnasium as gym
import numpy as np
import os
import math
from gymnasium.envs.registration import register
from .gym_env import TribesGymEnv, make_default_env

# wrapper to make it gym-compatible
class TribesGymWrapper(gym.Env):
    PHASE1_LEVEL_FILE = "levels/phase1_12x12_2bardur.csv"
    MAX_TURNS = 10
    SECOND_VILLAGE_BY_T10_PENALTY = -3.0

    # Phase1-Learning-004 shaping (learning-only; no gameplay-rule changes).
    EXPLORATION_REWARD_PER_NEW_TILE = 0.02
    EXPLORATION_REWARD_CAP_PER_TURN = 0.20
    CAPTURE_REWARD_BASE = 1.5
    CAPTURE_REWARD_DECAY_K = 0.35
    SECOND_VILLAGE_DELAY_BASE = 0.10
    SECOND_VILLAGE_DELAY_K = 0.35
    VISIBLE_VILLAGE_NEGLECT_GRACE_TURNS = 3
    VISIBLE_VILLAGE_NEGLECT_BASE = 0.30
    VISIBLE_VILLAGE_NEGLECT_K = 0.45
    T0_NO_MOVE_PENALTY = -3.0
    ALLOWED_ACTION_TYPES = {
        "END_TURN",
        "MOVE",
        "CAPTURE",
        "EXAMINE",
        "RESOURCE_GATHERING",
        "CLEAR_FOREST",
        "GROW_FOREST",
        "LEVEL_UP",
        "RESEARCH_TECH",
        "BUILD",
    }

    def __init__(self, level_file=None):
        self.tribes_env = make_default_env()
        self.level_file = level_file or self.PHASE1_LEVEL_FILE
        self.verbose_resets = os.environ.get("POLYVISION_VERBOSE_RESETS", "0").lower() in ("1", "true", "yes", "on")
        self.render_mode = "rgb_array"        # Initialize the environment to get the actual action space size
        self._turn_count = 0
        self._starting_city_count = 1
        self._last_city_count = 1
        self._seen_owned_unit_tiles = set()
        self._turn_exploration_reward_accum = 0.0
        self._moved_on_t0 = False
        self._visible_village_streak_turns = 0
        try:
            obs = self.tribes_env.reset(self.level_file, seed=42)
            
            # Use a fixed large action space to handle dynamic action counts
            # Most games will have fewer actions, but this provides a safe upper bound
            max_actions = 200  # Reasonable upper bound for this game
            self.action_space = gym.spaces.Discrete(max_actions)
            
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
            self.action_space = gym.spaces.Discrete(200)  # reasonable default
            self.observation_space = gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(1000,), 
                dtype=np.float32
            )
    
    def reset(self, seed=None, options=None):
        obs = self.tribes_env.reset(self.level_file, seed or 42)
        self._turn_count = 0
        obs = self._apply_bardur_opening(obs)
        self._starting_city_count = self._get_city_count(obs)
        self._last_city_count = self._starting_city_count
        self._seen_owned_unit_tiles = self._get_owned_unit_tiles(obs)
        self._turn_exploration_reward_accum = 0.0
        self._moved_on_t0 = False
        self._visible_village_streak_turns = 0
        
        # Log action space info for debugging
        action_count = self.tribes_env.action_space_n
        action_mask, allowed_indices = self._build_action_mask_and_indices()
        if self.verbose_resets:
            print(f"Reset: Available actions = {action_count}, allowed = {len(allowed_indices)}")
        
        # convert your dict obs to numpy array here
        return self._dict_to_array(obs), {
            "valid_actions": len(allowed_indices),
            "raw_valid_actions": action_count,
            "turn_count": self._turn_count,
            "action_mask": action_mask,
        }
    
    def render(self, **kwargs):
        data = np.array(self.tribes_env.render("rgb_image"))
        return data

    def step(self, action):
        legal_actions = self.tribes_env.list_actions()
        raw_count = len(legal_actions)
        action_mask, allowed_indices = self._build_action_mask_and_indices(legal_actions)

        if raw_count == 0:
            raise RuntimeError("No legal actions available from Java environment.")

        sampled_action = int(action)
        if len(allowed_indices) == 0:
            # Fallback: if whitelist eliminates everything, force END_TURN if available.
            end_turn_idx = next((i for i, a in enumerate(legal_actions) if a.get("type") == "END_TURN"), 0)
            selected_raw_action = end_turn_idx
        else:
            selected_allowed_pos = sampled_action % len(allowed_indices)
            selected_raw_action = allowed_indices[selected_allowed_pos]

        selected_action_type = legal_actions[selected_raw_action].get("type", "UNKNOWN")
        if selected_action_type == "END_TURN":
            self._turn_count += 1

        obs, reward, done, info = self.tribes_env.step(selected_raw_action)
        current_city_count = self._get_city_count(obs)
        current_turn = int(self._turn_count)
        reward_adjustment = 0.0

        exploration_reward = self._compute_exploration_reward(obs)
        reward_adjustment += exploration_reward

        capture_decay_bonus = 0.0
        if current_city_count > self._last_city_count:
            n_new_cities = current_city_count - self._last_city_count
            for _ in range(max(0, n_new_cities)):
                capture_decay_bonus += self.CAPTURE_REWARD_BASE * math.exp(-self.CAPTURE_REWARD_DECAY_K * current_turn)
        reward_adjustment += capture_decay_bonus

        second_village_delay_penalty = 0.0
        visible_village_neglect_penalty = 0.0
        t0_no_move_penalty = 0.0

        visible_uncaptured_village = self._has_visible_uncaptured_village(obs)
        has_second_village = current_city_count >= (self._starting_city_count + 1)

        if selected_action_type == "END_TURN":
            if current_turn >= 3 and not has_second_village:
                second_village_delay_penalty = -(
                    self.SECOND_VILLAGE_DELAY_BASE * math.exp(self.SECOND_VILLAGE_DELAY_K * (current_turn - 3))
                )

            if visible_uncaptured_village and not has_second_village:
                self._visible_village_streak_turns += 1
            else:
                self._visible_village_streak_turns = 0

            if self._visible_village_streak_turns > self.VISIBLE_VILLAGE_NEGLECT_GRACE_TURNS:
                overdue = self._visible_village_streak_turns - self.VISIBLE_VILLAGE_NEGLECT_GRACE_TURNS
                visible_village_neglect_penalty = -(
                    self.VISIBLE_VILLAGE_NEGLECT_BASE * math.exp(self.VISIBLE_VILLAGE_NEGLECT_K * overdue)
                )

            if current_turn == 1 and not self._moved_on_t0 and self._has_legal_move_action():
                t0_no_move_penalty = self.T0_NO_MOVE_PENALTY

            self._turn_exploration_reward_accum = 0.0

        reward_adjustment += second_village_delay_penalty
        reward_adjustment += visible_village_neglect_penalty
        reward_adjustment += t0_no_move_penalty

        # Phase 1 override: ignore Java terminal state and control horizon purely in Python.
        terminated = False
        truncated = self._turn_count >= self.MAX_TURNS
        if truncated and current_city_count <= self._starting_city_count:
            reward_adjustment += self.SECOND_VILLAGE_BY_T10_PENALTY

        reward = float(reward) + reward_adjustment

        info["valid_actions"] = len(allowed_indices)
        info["raw_valid_actions"] = raw_count
        info["sampled_action"] = sampled_action
        info["selected_raw_action"] = selected_raw_action
        info["selected_action_type"] = selected_action_type
        info["turn_count"] = self._turn_count
        info["city_count"] = current_city_count
        info["reward_adjustment"] = float(reward_adjustment)
        info["starting_city_count"] = int(self._starting_city_count)
        info["visible_uncaptured_village"] = bool(visible_uncaptured_village)
        info["visible_village_streak_turns"] = int(self._visible_village_streak_turns)
        info["moved_on_t0"] = bool(self._moved_on_t0)
        info["reward_exploration"] = float(exploration_reward)
        info["reward_capture_decay_bonus"] = float(capture_decay_bonus)
        info["reward_second_village_delay_penalty"] = float(second_village_delay_penalty)
        info["reward_visible_village_neglect_penalty"] = float(visible_village_neglect_penalty)
        info["reward_t0_no_move_penalty"] = float(t0_no_move_penalty)
        info["action_mask"] = action_mask
        info["java_done"] = bool(done)
        info["terminated_overridden"] = True
        self._last_city_count = current_city_count
        if selected_action_type == "MOVE" and self._turn_count == 0:
            self._moved_on_t0 = True

        return self._dict_to_array(obs), reward, terminated, truncated, info

    def _build_action_mask_and_indices(self, legal_actions=None):
        if legal_actions is None:
            legal_actions = self.tribes_env.list_actions()
        allowed_indices = [
            idx for idx, a in enumerate(legal_actions)
            if a.get("type") in self.ALLOWED_ACTION_TYPES
        ]
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for pos in range(min(len(allowed_indices), self.action_space.n)):
            mask[pos] = 1
        return mask, allowed_indices

    def _apply_bardur_opening(self, obs):
        """Force deterministic T0 opening: harvest 2 animals, then choose Workshop level-up.

        This executes directly against the Java bridge before the agent takes any action.
        """
        def find_action_idx(predicate):
            legal = self.tribes_env.list_actions()
            for idx, act in enumerate(legal):
                if predicate(act):
                    return idx
            return None

        # 1) Harvest animal #1
        idx = find_action_idx(
            lambda a: a.get("type") == "RESOURCE_GATHERING" and "ANIMAL" in a.get("repr", "")
        )
        if idx is None:
            actions_debug = self.tribes_env.list_actions()
            print("DEBUG_T0_ACTIONS_START")
            for i, action in enumerate(actions_debug):
                print(f"{i}: {action}")
            print("DEBUG_T0_ACTIONS_END")
            raise RuntimeError("Bardur opening failed: missing first ANIMAL harvest action.")
        obs, _, _, _ = self.tribes_env.step(idx)

        # 2) Harvest animal #2
        idx = find_action_idx(
            lambda a: a.get("type") == "RESOURCE_GATHERING" and "ANIMAL" in a.get("repr", "")
        )
        if idx is None:
            raise RuntimeError("Bardur opening failed: missing second ANIMAL harvest action.")
        obs, _, _, _ = self.tribes_env.step(idx)

        # 3) Choose Workshop on city level-up
        idx = find_action_idx(
            lambda a: a.get("type") == "LEVEL_UP" and "WORKSHOP" in a.get("repr", "")
        )
        if idx is None:
            raise RuntimeError("Bardur opening failed: missing WORKSHOP level-up action.")
        obs, _, _, _ = self.tribes_env.step(idx)

        return obs

    def _get_city_count(self, obs):
        tribes = obs.get("tribes", {})
        if isinstance(tribes, dict):
            tribe0 = tribes.get("0", {})
            if isinstance(tribe0, dict):
                return len(tribe0.get("citiesID", []))
        return 0

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

    def _compute_exploration_reward(self, obs):
        current_tiles = self._get_owned_unit_tiles(obs)
        new_tiles = current_tiles.difference(self._seen_owned_unit_tiles)
        self._seen_owned_unit_tiles.update(current_tiles)

        remaining_cap = max(0.0, self.EXPLORATION_REWARD_CAP_PER_TURN - self._turn_exploration_reward_accum)
        raw_reward = len(new_tiles) * self.EXPLORATION_REWARD_PER_NEW_TILE
        granted = min(remaining_cap, raw_reward)
        self._turn_exploration_reward_accum += granted
        return granted

    def _has_visible_uncaptured_village(self, obs):
        board = obs.get("board", {})
        terrain = board.get("terrain", [])
        city_ids = board.get("cityID", [])
        if not terrain or not city_ids:
            return False

        for i in range(len(terrain)):
            row_t = terrain[i]
            row_c = city_ids[i] if i < len(city_ids) else []
            for j in range(len(row_t)):
                t_val = int(row_t[j])
                c_val = int(row_c[j]) if j < len(row_c) else -1
                if t_val == 4 and c_val == -1:
                    return True
        return False

    def _has_legal_move_action(self):
        legal_actions = self.tribes_env.list_actions()
        for act in legal_actions:
            if act.get("type") == "MOVE":
                return True
        return False
    
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
