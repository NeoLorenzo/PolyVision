import os
import random
import traceback
from collections import Counter, defaultdict

import gymnasium as gym
import numpy as np

import importlib.util
import sys

# Ensure repo root is importable and the environment registration side effect runs.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
try:
    import pol_env.Tribes.py.register_env as register_env  # noqa: F401
except Exception:
    _fpath = os.path.join(_repo_root, "pol_env", "Tribes", "py", "register_env.py")
    spec = importlib.util.spec_from_file_location("register_env", _fpath)
    register_env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(register_env)


LEGACY_OBS_DIM = 438
RESOURCE_BLOCK_DIM = 144
EXPECTED_OBS_DIM = 597
EXPECTED_LEGAL_FEATURE_DIM = 42
UNKNOWN_RESOURCE_NORM = 0.0  # clip((-1 + 1) / 8, 0, 1)

LEGACY_FEATURE_NAMES_22 = (
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
)


IDX = {
    "research_tech_id_norm": 22,
    "research_is_organization": 23,
    "research_is_forestry": 24,
    "resource_id_norm": 25,
    "resource_is_animal": 26,
    "resource_is_fruit": 27,
    "resource_is_fish": 28,
    "resource_is_crop": 29,
    "resource_is_metal": 30,
    "build_id_norm": 31,
    "build_is_lumber_hut": 32,
    "build_is_sawmill": 33,
    "levelup_choice_id_norm": 34,
    "levelup_is_workshop": 35,
    "expected_population_delta_norm": 36,
    "expected_immediate_spt_delta_norm": 37,
    "makes_level_up_available": 38,
    "is_level_up_claim": 39,
    "action_city_upgrade_progress_before_norm": 40,
    "action_city_upgrade_ready_before": 41,
}


OBS_IDX = {
    "resource_start": LEGACY_OBS_DIM,
    "resource_end": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM,
    "current_stars_norm": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM + 0,
    "current_spt_norm": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM + 1,
    "turn_count_norm": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM + 2,
    "turns_remaining_after_current_norm": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM + 3,
    "turns_remaining_including_current_norm": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM + 4,
    "tech_has_organization": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM + 5,
    "tech_has_forestry": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM + 6,
    "tech_researched_count_norm": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM + 7,
    "city_count_norm": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM + 8,
    "avg_city_level_norm": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM + 9,
    "max_city_level_norm": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM + 10,
    "mean_upgrade_progress_norm": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM + 11,
    "max_upgrade_progress_norm": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM + 12,
    "upgrade_ready_frac": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM + 13,
    "any_level_up_available": LEGACY_OBS_DIM + RESOURCE_BLOCK_DIM + 14,
}


CATEGORIES = [
    "1_shape_checks",
    "2_legacy_prefix_preservation",
    "3_slot_alignment",
    "4_research_feature_checks",
    "5_resource_feature_checks",
    "6_build_feature_checks",
    "7_levelup_feature_checks",
    "8_city_upgrade_feature_checks",
    "9_fog_safe_resource_obs",
    "10_tech_state_lifecycle",
    "11_predicted_vs_actual_deltas",
    "12_sampling_coverage",
]


def set_required_env():
    os.environ["POLYVISION_LEVEL_POOL_GLOB"] = "levels/phase1_pool_bardur_solo/*.csv"
    os.environ["POLYVISION_LEVEL_SELECTION_MODE"] = "round_robin"
    os.environ["POLYVISION_INFO_MODE"] = "fast"
    os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"
    os.environ["POLYVISION_BATCH_LEGAL_ACTION_FETCH"] = "1"


class Validator:
    def __init__(self):
        self.category_pass = {k: True for k in CATEGORIES}
        self.failures = []
        self.total_failures = 0
        self.max_failure_examples = 120

        self.total_valid_slots_inspected = 0
        self.legal_action_type_counts = Counter()
        self.selected_action_type_counts = Counter()
        self.research_counts = Counter()
        self.resource_counts = Counter()
        self.build_counts = Counter()
        self.levelup_counts = Counter()

        self.delta_abs_err_by_type = defaultdict(lambda: {"pop": [], "spt": []})

        self.coverage_seen = {
            "RESEARCH_TECH": False,
            "RESOURCE_GATHERING": False,
            "BUILD": False,
            "CAPTURE": False,
            "MOVE": False,
            "LEVEL_UP": False,
        }

        self.lifecycle = {
            "org_toggled_on_steps": 0,
            "forestry_toggled_on_steps": 0,
            "invalid_research_attempt_checks": 0,
            "invalid_research_attempt_failures": 0,
            "reset_checks": 0,
        }

    def fail(self, category, message, context=None):
        self.category_pass[category] = False
        self.total_failures += 1
        if len(self.failures) < self.max_failure_examples:
            entry = {"category": category, "message": message}
            if context is not None:
                entry["context"] = context
            self.failures.append(entry)

    def check(self, condition, category, message, context=None):
        if not bool(condition):
            self.fail(category, message, context)

    @staticmethod
    def _action_subtype(uw, action):
        a_type = str(action.get("type", "")).upper()
        if a_type == "RESEARCH_TECH":
            return uw._resolve_action_tech_type(action)
        if a_type == "RESOURCE_GATHERING":
            return uw._resolve_action_resource_type(action)
        if a_type == "BUILD":
            return uw._resolve_action_building_type(action)
        if a_type == "LEVEL_UP":
            return uw._resolve_action_levelup_choice(action)
        return ""

    @staticmethod
    def _resource_norm(v):
        return float(np.clip((float(v) + 1.0) / 8.0, 0.0, 1.0))

    def _slot_to_action(self, uw, slot, ids):
        gid = int(ids[slot])
        raw_idx = uw._current_legal_id_to_raw_index.get(gid, None)
        if raw_idx is None:
            return gid, None, None
        if int(raw_idx) < 0 or int(raw_idx) >= len(uw._current_legal_actions):
            return gid, raw_idx, None
        return gid, int(raw_idx), uw._current_legal_actions[int(raw_idx)]

    def validate_state(self, obs, info, uw, episode_idx, step_idx):
        category = "1_shape_checks"
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.check(
            tuple(obs_arr.shape) == (EXPECTED_OBS_DIM,),
            category,
            f"obs shape mismatch: got={obs_arr.shape}, expected={(EXPECTED_OBS_DIM,)}",
            {"episode": episode_idx, "step": step_idx},
        )

        feats = np.asarray(info.get("legal_action_features_padded"), dtype=np.float32)
        ids = np.asarray(info.get("legal_global_ids_padded"), dtype=np.int64).reshape(-1)
        valid = np.asarray(info.get("legal_action_valid_mask"), dtype=bool).reshape(-1)

        self.check(
            feats.ndim == 2 and feats.shape[1] == EXPECTED_LEGAL_FEATURE_DIM,
            category,
            f"legal feature dim mismatch: got shape {feats.shape}, expected second dim {EXPECTED_LEGAL_FEATURE_DIM}",
            {"episode": episode_idx, "step": step_idx},
        )
        self.check(
            feats.shape[0] == ids.shape[0] == valid.shape[0],
            category,
            "slot tensor first-dim mismatch",
            {"episode": episode_idx, "step": step_idx, "feats0": feats.shape[0], "ids0": ids.shape[0], "valid0": valid.shape[0]},
        )

        if feats.size > 0 and valid.size > 0:
            invalid_rows = feats[~valid]
            if invalid_rows.size > 0:
                self.check(
                    np.allclose(invalid_rows, 0.0, atol=1e-8),
                    category,
                    "invalid/padded slot features are not all zeros",
                    {"episode": episode_idx, "step": step_idx},
                )

        category = "2_legacy_prefix_preservation"
        self.check(
            tuple(uw.LEGAL_ACTION_FEATURE_NAMES[:22]) == LEGACY_FEATURE_NAMES_22,
            category,
            "first 22 LEGAL_ACTION_FEATURE_NAMES changed",
            {"episode": episode_idx, "step": step_idx},
        )

        category = "3_slot_alignment"
        valid_indices = np.where(valid)[0]
        self.total_valid_slots_inspected += int(valid_indices.size)
        valid_gids = [int(ids[k]) for k in valid_indices]
        self.check(
            len(valid_gids) == len(set(valid_gids)),
            category,
            "duplicate global ids among valid slots",
            {"episode": episode_idx, "step": step_idx},
        )

        raw_obs = uw.tribes_env._last_obs
        for slot in valid_indices:
            gid, raw_idx, action = self._slot_to_action(uw, int(slot), ids)
            self.check(
                action is not None,
                category,
                f"slot {slot} gid {gid} unresolved raw action",
                {"episode": episode_idx, "step": step_idx},
            )
            if action is None:
                continue

            expected = np.asarray(uw._compute_legal_action_feature_vector_reference(action, raw_obs), dtype=np.float32)
            got = feats[int(slot)]
            self.check(
                np.allclose(expected, got, atol=1e-6, rtol=1e-6),
                category,
                f"slot {slot} feature mismatch vs recomputed action",
                {"episode": episode_idx, "step": step_idx, "gid": gid, "raw_idx": raw_idx, "repr": str(action.get('repr', ''))[:180]},
            )

            self.check(
                np.allclose(expected[:22], got[:22], atol=1e-6, rtol=1e-6),
                "2_legacy_prefix_preservation",
                f"legacy first22 mismatch slot={slot}",
                {"episode": episode_idx, "step": step_idx, "gid": gid, "raw_idx": raw_idx},
            )

            a_type = str(action.get("type", "")).upper()
            self.legal_action_type_counts[a_type] += 1
            if a_type in self.coverage_seen:
                self.coverage_seen[a_type] = True

            # Category 4: research features
            if a_type == "RESEARCH_TECH":
                tech = uw._resolve_action_tech_type(action)
                self.research_counts[tech] += 1
                self.check(float(got[15]) > 0.5, "4_research_feature_checks", "is_research old feature != 1", {"tech": tech, "slot": int(slot)})
                if tech == "FORESTRY":
                    self.check(float(got[IDX["research_is_forestry"]]) > 0.5, "4_research_feature_checks", "research_is_forestry != 1", {"slot": int(slot)})
                    self.check(float(got[IDX["research_is_organization"]]) < 0.5, "4_research_feature_checks", "research_is_organization != 0 for Forestry", {"slot": int(slot)})
                    norm = float(got[IDX["research_tech_id_norm"]])
                    if not (norm > 0.0):
                        idx = uw._catalog.tech_to_idx.get("FORESTRY", -1)
                        n = len(uw._catalog.tech_types)
                        self.check(idx == 0 or n <= 1, "4_research_feature_checks", "research_tech_id_norm not >0 for Forestry but not a valid zero-index normalization case", {"norm": norm, "idx": idx, "n": n})
                if tech == "ORGANIZATION":
                    self.check(float(got[IDX["research_is_organization"]]) > 0.5, "4_research_feature_checks", "research_is_organization != 1", {"slot": int(slot)})
                    self.check(float(got[IDX["research_is_forestry"]]) < 0.5, "4_research_feature_checks", "research_is_forestry != 0 for Organization", {"slot": int(slot)})

            # Category 5: resource features
            if a_type == "RESOURCE_GATHERING":
                r = uw._resolve_action_resource_type(action)
                self.resource_counts[r] += 1
                self.check(float(got[16]) > 0.5, "5_resource_feature_checks", "old is_resource_gathering != 1", {"resource": r, "slot": int(slot)})
                if r == "ANIMAL":
                    self.check(float(got[IDX["resource_is_animal"]]) > 0.5, "5_resource_feature_checks", "resource_is_animal != 1", {"slot": int(slot)})
                    self.check(float(got[IDX["resource_is_fruit"]]) < 0.5, "5_resource_feature_checks", "resource_is_fruit != 0 for animal", {"slot": int(slot)})
                    self.check(float(got[IDX["expected_population_delta_norm"]]) > 0.0, "5_resource_feature_checks", "expected_population_delta_norm <= 0 for ANIMAL", {"slot": int(slot)})
                elif r == "FRUIT":
                    self.check(float(got[IDX["resource_is_fruit"]]) > 0.5, "5_resource_feature_checks", "resource_is_fruit != 1", {"slot": int(slot)})
                    self.check(float(got[IDX["resource_is_animal"]]) < 0.5, "5_resource_feature_checks", "resource_is_animal != 0 for fruit", {"slot": int(slot)})
                    self.check(float(got[IDX["expected_population_delta_norm"]]) > 0.0, "5_resource_feature_checks", "expected_population_delta_norm <= 0 for FRUIT", {"slot": int(slot)})
                elif r == "FISH":
                    self.check(float(got[IDX["resource_is_fish"]]) > 0.5, "5_resource_feature_checks", "resource_is_fish != 1", {"slot": int(slot)})
                elif r == "CROPS":
                    self.check(float(got[IDX["resource_is_crop"]]) > 0.5, "5_resource_feature_checks", "resource_is_crop != 1", {"slot": int(slot)})
                elif r == "ORE":
                    self.check(float(got[IDX["resource_is_metal"]]) > 0.5, "5_resource_feature_checks", "resource_is_metal != 1", {"slot": int(slot)})

            # Category 6: build features
            if a_type == "BUILD":
                b = uw._resolve_action_building_type(action)
                self.build_counts[b] += 1
                if b == "LUMBER_HUT":
                    self.check(float(got[18]) > 0.5, "6_build_feature_checks", "old is_build != 1 for LUMBER_HUT", {"slot": int(slot)})
                    self.check(float(got[IDX["build_is_lumber_hut"]]) > 0.5, "6_build_feature_checks", "build_is_lumber_hut != 1", {"slot": int(slot)})
                    self.check(float(got[IDX["build_is_sawmill"]]) < 0.5, "6_build_feature_checks", "build_is_sawmill != 0 for LUMBER_HUT", {"slot": int(slot)})
                elif b == "SAWMILL":
                    self.check(float(got[18]) > 0.5, "6_build_feature_checks", "old is_build != 1 for SAWMILL", {"slot": int(slot)})
                    self.check(float(got[IDX["build_is_sawmill"]]) > 0.5, "6_build_feature_checks", "build_is_sawmill != 1", {"slot": int(slot)})

            # Category 7: level-up features
            if a_type == "LEVEL_UP":
                choice = uw._resolve_action_levelup_choice(action)
                self.levelup_counts[choice] += 1
                self.check(float(got[17]) > 0.5, "7_levelup_feature_checks", "old is_level_up != 1", {"slot": int(slot)})
                self.check(float(got[IDX["is_level_up_claim"]]) > 0.5, "7_levelup_feature_checks", "is_level_up_claim != 1", {"slot": int(slot)})
                if choice == "WORKSHOP":
                    self.check(float(got[IDX["levelup_is_workshop"]]) > 0.5, "7_levelup_feature_checks", "levelup_is_workshop != 1 for WORKSHOP", {"slot": int(slot)})
                    self.check(float(got[IDX["expected_immediate_spt_delta_norm"]]) > 0.0, "7_levelup_feature_checks", "expected_immediate_spt_delta_norm <= 0 for WORKSHOP", {"slot": int(slot)})
                else:
                    self.check(float(got[IDX["levelup_is_workshop"]]) < 0.5, "7_levelup_feature_checks", "levelup_is_workshop != 0 for non-WORKSHOP", {"slot": int(slot), "choice": choice})

            # Category 8: city-upgrade feature checks
            if a_type == "RESOURCE_GATHERING":
                eco = uw._summarize_action_economy_expectation(action, raw_obs)
                prog = float(got[IDX["action_city_upgrade_progress_before_norm"]])
                ready_before_f = float(got[IDX["action_city_upgrade_ready_before"]]) > 0.5
                self.check(0.0 <= prog <= 1.0, "8_city_upgrade_feature_checks", "action_city_upgrade_progress_before_norm out of [0,1]", {"slot": int(slot), "progress": prog})

                city_id = eco.get("city_id", None)
                expected_delta = int(eco.get("expected_population_delta", 0))
                if city_id is not None and expected_delta > 0:
                    city_info = (raw_obs.get("city", {}) or {}).get(str(int(city_id)), None)
                    if isinstance(city_info, dict):
                        pop = int(city_info.get("population", 0))
                        need = int(city_info.get("population_need", 0))
                        expected_ready_before = (pop >= need) if need > 0 else True
                        expected_makes = (pop < need) and ((pop + expected_delta) >= need)
                        makes_feat = float(got[IDX["makes_level_up_available"]]) > 0.5
                        self.check(
                            makes_feat == expected_makes,
                            "8_city_upgrade_feature_checks",
                            "makes_level_up_available mismatch vs pre-action city threshold logic",
                            {"slot": int(slot), "city_id": int(city_id), "pop": pop, "need": need, "expected_delta": expected_delta, "feat": makes_feat, "expected": expected_makes},
                        )
                        self.check(
                            ready_before_f == expected_ready_before,
                            "8_city_upgrade_feature_checks",
                            "action_city_upgrade_ready_before mismatch",
                            {"slot": int(slot), "city_id": int(city_id), "feat": ready_before_f, "expected": expected_ready_before},
                        )

        # Category 9: fog-safe resource observation block
        category = "9_fog_safe_resource_obs"
        resource_slice = obs_arr[OBS_IDX["resource_start"] : OBS_IDX["resource_end"]]
        dims = uw._board_dimensions_from_obs(raw_obs)
        if dims is not None:
            width, height = int(dims[0]), int(dims[1])
            try:
                terrain_arr = np.asarray((raw_obs.get("board", {}) or {}).get("terrain", []), dtype=np.int16)
                resource_arr = np.asarray((raw_obs.get("board", {}) or {}).get("resource", []), dtype=np.int16)
                if terrain_arr.shape == (width, height) and resource_arr.shape == (width, height):
                    expected_resource = np.array(resource_arr, copy=True)
                    expected_resource[terrain_arr == 7] = -1
                    expected_norm = np.clip((expected_resource.astype(np.float32) + 1.0) / 8.0, 0.0, 1.0).flatten()
                    self.check(
                        np.allclose(resource_slice, expected_norm, atol=1e-6, rtol=1e-6),
                        category,
                        "visible_resource_flat_norm mismatch with fog-safe expected encoding",
                        {"episode": episode_idx, "step": step_idx},
                    )
                    fog_flat = (terrain_arr.flatten() == 7)
                    if np.any(fog_flat):
                        self.check(
                            np.allclose(resource_slice[fog_flat], UNKNOWN_RESOURCE_NORM, atol=1e-8),
                            category,
                            "fog tiles resource encoding not unknown/empty",
                            {"episode": episode_idx, "step": step_idx},
                        )
            except Exception as exc:
                self.fail(category, f"exception while checking fog-safe resource block: {exc}", {"episode": episode_idx, "step": step_idx})

        return feats, ids, valid

    def choose_action(self, info, uw, rng):
        ids = np.asarray(info["legal_global_ids_padded"], dtype=np.int64).reshape(-1)
        valid = np.asarray(info["legal_action_valid_mask"], dtype=bool).reshape(-1)
        valid_slots = np.where(valid)[0]
        if valid_slots.size == 0:
            return None, None

        # Build candidate buckets.
        buckets = defaultdict(list)
        for slot in valid_slots:
            gid, _raw_idx, action = self._slot_to_action(uw, int(slot), ids)
            if action is None:
                continue
            a_type = str(action.get("type", "")).upper()
            subtype = self._action_subtype(uw, action)
            buckets[(a_type, subtype)].append(int(gid))
            buckets[(a_type, "")].append(int(gid))

        # Targeted exploration first to maximize coverage.
        target_order = [
            ("RESEARCH_TECH", "FORESTRY"),
            ("RESEARCH_TECH", "ORGANIZATION"),
            ("RESOURCE_GATHERING", "ANIMAL"),
            ("RESOURCE_GATHERING", "FRUIT"),
            ("RESOURCE_GATHERING", "FISH"),
            ("RESOURCE_GATHERING", "CROPS"),
            ("RESOURCE_GATHERING", "ORE"),
            ("BUILD", "LUMBER_HUT"),
            ("BUILD", "SAWMILL"),
            ("LEVEL_UP", "WORKSHOP"),
            ("LEVEL_UP", ""),
            ("CAPTURE", ""),
            ("MOVE", ""),
        ]
        if rng.random() < 0.7:
            for key in target_order:
                vals = buckets.get(key, [])
                if vals:
                    return int(rng.choice(vals)), key

        # Fallback random legal action.
        gid = int(rng.choice([int(ids[s]) for s in valid_slots]))
        return gid, None

    def summarize(self):
        cat_status = {k: ("PASS" if self.category_pass[k] else "FAIL") for k in CATEGORIES}

        # Coverage category (12).
        missing_coverage = [k for k, seen in self.coverage_seen.items() if not seen]
        if missing_coverage:
            self.category_pass["12_sampling_coverage"] = False
            cat_status["12_sampling_coverage"] = "FAIL"
        else:
            cat_status["12_sampling_coverage"] = "PASS"

        delta_summary = {}
        for a_type, vals in sorted(self.delta_abs_err_by_type.items()):
            pop_arr = np.asarray(vals["pop"], dtype=np.float32)
            spt_arr = np.asarray(vals["spt"], dtype=np.float32)
            delta_summary[a_type] = {
                "n": int(max(pop_arr.size, spt_arr.size)),
                "pop_abs_err_mean": float(np.mean(pop_arr)) if pop_arr.size > 0 else 0.0,
                "pop_abs_err_max": float(np.max(pop_arr)) if pop_arr.size > 0 else 0.0,
                "spt_abs_err_mean": float(np.mean(spt_arr)) if spt_arr.size > 0 else 0.0,
                "spt_abs_err_max": float(np.max(spt_arr)) if spt_arr.size > 0 else 0.0,
            }

        print("=== Bardur Feature Validation Report ===")
        for k in CATEGORIES:
            print(f"{k}: {cat_status[k]}")
        print(f"total_valid_legal_actions_inspected: {self.total_valid_slots_inspected}")
        print(f"legal_action_type_counts: {dict(self.legal_action_type_counts)}")
        print(f"selected_action_type_counts: {dict(self.selected_action_type_counts)}")
        print(f"research_counts: {dict(self.research_counts)}")
        print(f"resource_counts: {dict(self.resource_counts)}")
        print(f"build_counts: {dict(self.build_counts)}")
        print(f"levelup_counts: {dict(self.levelup_counts)}")
        print(f"missing_coverage_action_types: {missing_coverage}")
        print(f"invalid_research_attempt_checks: {self.lifecycle['invalid_research_attempt_checks']}")
        print(f"invalid_research_attempt_failures: {self.lifecycle['invalid_research_attempt_failures']}")
        print("predicted_vs_actual_delta_error_summary_by_selected_action_type:")
        for a_type, stats in delta_summary.items():
            print(f"  {a_type}: {stats}")
        print(f"total_failures: {self.total_failures}")
        print("failure_examples:")
        if not self.failures:
            print("  []")
        else:
            for item in self.failures[:30]:
                print(f"  - {item}")

        safe = all(self.category_pass.values())
        recommendation = "safe to train" if safe else "not safe to train"
        print(f"final_recommendation: {recommendation}")
        return safe


def run_validation(num_episodes=100, max_steps_per_episode=20, seed=1234):
    set_required_env()
    rng = random.Random(seed)
    validator = Validator()

    env = gym.make("Tribes-v0")
    try:
        prev_episode_had_org = False
        prev_episode_had_forestry = False

        for ep in range(int(num_episodes)):
            obs, info = env.reset(seed=seed + ep)
            uw = env.unwrapped
            validator.lifecycle["reset_checks"] += 1

            # Lifecycle checks at reset.
            techs = set(uw._researched_techs_t10)
            org_flag = float(np.asarray(obs)[OBS_IDX["tech_has_organization"]]) > 0.5
            forestry_flag = float(np.asarray(obs)[OBS_IDX["tech_has_forestry"]]) > 0.5
            validator.check("HUNTING" in techs, "10_tech_state_lifecycle", "Bardur starting tech HUNTING missing after reset", {"episode": ep, "techs": sorted(list(techs))})
            validator.check(not org_flag, "10_tech_state_lifecycle", "Organization flag unexpectedly on at reset", {"episode": ep})
            validator.check(not forestry_flag, "10_tech_state_lifecycle", "Forestry flag unexpectedly on at reset", {"episode": ep})
            if prev_episode_had_org:
                validator.check(not org_flag, "10_tech_state_lifecycle", "Organization did not reset cleanly across episodes", {"episode": ep})
            if prev_episode_had_forestry:
                validator.check(not forestry_flag, "10_tech_state_lifecycle", "Forestry did not reset cleanly across episodes", {"episode": ep})

            # Invalid action attempt check (should not update research state).
            if ep < 3:
                before = set(uw._researched_techs_t10)
                validator.lifecycle["invalid_research_attempt_checks"] += 1
                try:
                    env.step(-1)
                    validator.lifecycle["invalid_research_attempt_failures"] += 1
                    validator.fail("10_tech_state_lifecycle", "invalid action step(-1) unexpectedly succeeded", {"episode": ep})
                except Exception:
                    after = set(uw._researched_techs_t10)
                    if before != after:
                        validator.lifecycle["invalid_research_attempt_failures"] += 1
                        validator.fail(
                            "10_tech_state_lifecycle",
                            "tech state changed after invalid action attempt",
                            {"episode": ep, "before": sorted(list(before)), "after": sorted(list(after))},
                        )

            for step in range(int(max_steps_per_episode)):
                try:
                    validator.validate_state(obs, info, uw, ep, step)
                except Exception as exc:
                    validator.fail(
                        "1_shape_checks",
                        f"unexpected exception in state validation: {exc}",
                        {"episode": ep, "step": step, "trace": traceback.format_exc()[:1200]},
                    )

                action_gid, _ = validator.choose_action(info, uw, rng)
                if action_gid is None:
                    validator.fail("12_sampling_coverage", "no valid action available", {"episode": ep, "step": step})
                    break

                prev_org = float(np.asarray(obs)[OBS_IDX["tech_has_organization"]]) > 0.5
                prev_forestry = float(np.asarray(obs)[OBS_IDX["tech_has_forestry"]]) > 0.5

                # Identify selected action semantics from pre-step slot alignment.
                pre_ids = np.asarray(info["legal_global_ids_padded"], dtype=np.int64).reshape(-1)
                pre_valid = np.asarray(info["legal_action_valid_mask"], dtype=bool).reshape(-1)
                slot_sel = None
                for s in np.where(pre_valid)[0]:
                    if int(pre_ids[s]) == int(action_gid):
                        slot_sel = int(s)
                        break
                selected_type = None
                selected_subtype = ""
                if slot_sel is not None:
                    _, _, selected_action = validator._slot_to_action(uw, slot_sel, pre_ids)
                    if selected_action is not None:
                        selected_type = str(selected_action.get("type", "")).upper()
                        selected_subtype = validator._action_subtype(uw, selected_action)

                try:
                    obs_next, _reward, terminated, truncated, info_next = env.step(int(action_gid))
                except Exception as exc:
                    validator.fail(
                        "3_slot_alignment",
                        f"env.step failed for chosen legal gid {action_gid}: {exc}",
                        {"episode": ep, "step": step},
                    )
                    break

                selected_type_post = str(info_next.get("selected_action_type", selected_type or "UNKNOWN")).upper()
                validator.selected_action_type_counts[selected_type_post] += 1

                # Category 10: tech lifecycle transitions only after successful matching research action.
                post_org = float(np.asarray(obs_next)[OBS_IDX["tech_has_organization"]]) > 0.5
                post_forestry = float(np.asarray(obs_next)[OBS_IDX["tech_has_forestry"]]) > 0.5
                if (not prev_org) and post_org:
                    validator.lifecycle["org_toggled_on_steps"] += 1
                    validator.check(
                        selected_type == "RESEARCH_TECH" and selected_subtype == "ORGANIZATION",
                        "10_tech_state_lifecycle",
                        "Organization toggled on without selecting successful RESEARCH_TECH Organization",
                        {"episode": ep, "step": step, "selected_type": selected_type, "selected_subtype": selected_subtype},
                    )
                if (not prev_forestry) and post_forestry:
                    validator.lifecycle["forestry_toggled_on_steps"] += 1
                    validator.check(
                        selected_type == "RESEARCH_TECH" and selected_subtype == "FORESTRY",
                        "10_tech_state_lifecycle",
                        "Forestry toggled on without selecting successful RESEARCH_TECH Forestry",
                        {"episode": ep, "step": step, "selected_type": selected_type, "selected_subtype": selected_subtype},
                    )

                # Category 11: predicted-vs-actual deltas.
                pop_abs_err = info_next.get("selected_population_delta_abs_error", None)
                spt_abs_err = info_next.get("selected_immediate_spt_delta_abs_error", None)
                if pop_abs_err is None or spt_abs_err is None:
                    validator.fail("11_predicted_vs_actual_deltas", "missing predicted-vs-actual delta diagnostics in info", {"episode": ep, "step": step})
                else:
                    try:
                        validator.delta_abs_err_by_type[selected_type_post]["pop"].append(float(pop_abs_err))
                        validator.delta_abs_err_by_type[selected_type_post]["spt"].append(float(spt_abs_err))
                    except Exception:
                        validator.fail("11_predicted_vs_actual_deltas", "non-numeric delta diagnostics in info", {"episode": ep, "step": step})

                obs, info = obs_next, info_next
                if bool(terminated) or bool(truncated):
                    break

            # Track whether techs were learned in this episode for next reset-cleanliness check.
            prev_episode_had_org = bool(float(np.asarray(obs)[OBS_IDX["tech_has_organization"]]) > 0.5)
            prev_episode_had_forestry = bool(float(np.asarray(obs)[OBS_IDX["tech_has_forestry"]]) > 0.5)

    finally:
        env.close()

    safe = validator.summarize()
    return 0 if safe else 1


if __name__ == "__main__":
    exit_code = run_validation(num_episodes=100, max_steps_per_episode=20, seed=1234)
    raise SystemExit(exit_code)
