import gymnasium as gym
import numpy as np
import os
import re
from gymnasium.envs.registration import register
from .gym_env import TribesGymEnv, make_default_env

# wrapper to make it gym-compatible
class TribesGymWrapper(gym.Env):
    PHASE1_LEVEL_FILE = "levels/phase1_12x12_2bardur.csv"
    MAX_TURNS = 10
    SECOND_VILLAGE_BY_T10_PENALTY = -3.0

    # Phase1-Learning-006 shaping (flattened, lower-variance rewards/penalties).
    CAPTURE_CITY_BONUS = 2.0
    SECOND_VILLAGE_DELAY_PENALTY = -0.2
    SECOND_VILLAGE_DELAY_START_TURN = 4
    VISIBLE_VILLAGE_NEGLECT_PENALTY = -0.5
    VISIBLE_VILLAGE_NEGLECT_GRACE_TURNS = 2
    VILLAGE_BREADCRUMB_REWARD = 0.5
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

        capture_city_bonus = 0.0
        if current_city_count > self._last_city_count:
            n_new_cities = current_city_count - self._last_city_count
            for _ in range(max(0, n_new_cities)):
                capture_city_bonus += self.CAPTURE_CITY_BONUS
        reward_adjustment += capture_city_bonus

        second_village_delay_penalty = 0.0
        visible_village_neglect_penalty = 0.0
        village_breadcrumb_reward = 0.0

        visible_uncaptured_village = self._has_visible_uncaptured_village(obs)
        has_second_village = current_city_count >= (self._starting_city_count + 1)
        unit_on_visible_uncaptured_village = self._has_unit_on_visible_uncaptured_village(obs)

        if selected_action_type == "END_TURN":
            if current_turn >= self.SECOND_VILLAGE_DELAY_START_TURN and not has_second_village:
                second_village_delay_penalty = self.SECOND_VILLAGE_DELAY_PENALTY

            if visible_uncaptured_village and not has_second_village:
                self._visible_village_streak_turns += 1
            else:
                self._visible_village_streak_turns = 0

            if self._visible_village_streak_turns > self.VISIBLE_VILLAGE_NEGLECT_GRACE_TURNS:
                visible_village_neglect_penalty = self.VISIBLE_VILLAGE_NEGLECT_PENALTY

            if unit_on_visible_uncaptured_village:
                village_breadcrumb_reward = self.VILLAGE_BREADCRUMB_REWARD

        reward_adjustment += second_village_delay_penalty
        reward_adjustment += visible_village_neglect_penalty
        reward_adjustment += village_breadcrumb_reward

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
        info["unit_on_visible_uncaptured_village"] = bool(unit_on_visible_uncaptured_village)
        info["visible_village_streak_turns"] = int(self._visible_village_streak_turns)
        info["moved_on_t0"] = bool(self._moved_on_t0)
        info["reward_capture_city_bonus"] = float(capture_city_bonus)
        info["reward_second_village_delay_penalty"] = float(second_village_delay_penalty)
        info["reward_visible_village_neglect_penalty"] = float(visible_village_neglect_penalty)
        info["reward_village_breadcrumb"] = float(village_breadcrumb_reward)
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

            if scored_moves:
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
            actions_debug = self.tribes_env.list_actions()
            print("DEBUG_T0_ACTIONS_START")
            for i, action in enumerate(actions_debug):
                print(f"{i}: {action}")
            print("DEBUG_T0_ACTIONS_END")
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

    def _has_unit_on_visible_uncaptured_village(self, obs):
        board = obs.get("board", {})
        terrain = board.get("terrain", [])
        city_ids = board.get("cityID", [])
        if not terrain or not city_ids:
            return False

        village_positions = set()
        for i in range(len(terrain)):
            row_t = terrain[i]
            row_c = city_ids[i] if i < len(city_ids) else []
            for j in range(len(row_t)):
                t_val = int(row_t[j])
                c_val = int(row_c[j]) if j < len(row_c) else -1
                if t_val == 4 and c_val == -1:
                    # Board arrays are indexed [row=y][col=x], while units store (x, y).
                    village_positions.add((j, i))

        if not village_positions:
            return False

        for ux, uy in self._get_owned_unit_tiles(obs):
            if (ux, uy) in village_positions:
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
