import gymnasium as gym
import numpy as np
import os
from gymnasium.envs.registration import register
from .gym_env import TribesGymEnv, make_default_env

# wrapper to make it gym-compatible
class TribesGymWrapper(gym.Env):
    PHASE1_LEVEL_FILE = "levels/phase1_bardur_drylands.csv"
    MAX_TURNS = 10
    ALLOWED_ACTION_TYPES = {
        "END_TURN",
        "MOVE",
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
        # Phase 1 override: ignore Java terminal state and control horizon purely in Python.
        terminated = False
        truncated = self._turn_count >= self.MAX_TURNS

        info["valid_actions"] = len(allowed_indices)
        info["raw_valid_actions"] = raw_count
        info["sampled_action"] = sampled_action
        info["selected_raw_action"] = selected_raw_action
        info["selected_action_type"] = selected_action_type
        info["turn_count"] = self._turn_count
        info["action_mask"] = action_mask
        info["java_done"] = bool(done)
        info["terminated_overridden"] = True

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
        tribe_info = obs_dict.get("tribe", {})
        if isinstance(tribe_info, dict):
            # Add some key tribe stats as features
            features.append(tribe_info.get("stars", 0))
            features.append(tribe_info.get("score", 0))
            features.append(len(tribe_info.get("citiesID", [])))
            features.append(tribe_info.get("nKills", 0))
            
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
