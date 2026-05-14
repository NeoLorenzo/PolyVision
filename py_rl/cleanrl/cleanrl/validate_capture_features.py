import os
import random
import sys
import traceback

import gymnasium as gym
import numpy as np

import importlib.util


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


CAPTURE_TARGET_ACTIONS = 50
MAX_EPISODES = 100
MAX_STEPS_PER_EPISODE = 20
EXPECTED_OBS_DIM = 597
EXPECTED_FEAT_DIM = 42

IDX_IS_CAPTURE_OLD = 13
APPENDED_RANGE = list(range(22, 42))


def set_required_env():
    os.environ["POLYVISION_LEVEL_POOL_GLOB"] = "levels/phase1_pool_bardur_solo/*.csv"
    os.environ["POLYVISION_LEVEL_SELECTION_MODE"] = "round_robin"
    os.environ["POLYVISION_INFO_MODE"] = "fast"
    os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"
    os.environ["POLYVISION_BATCH_LEGAL_ACTION_FETCH"] = "1"


class CaptureValidator:
    def __init__(self, seed=123):
        self.rng = random.Random(seed)
        self.failures = []
        self.max_fail_examples = 80
        self.pass_all = True

        self.episodes_run = 0
        self.states_with_capture_legal = 0
        self.capture_actions_inspected = 0
        self.successful_capture_exec = 0
        self.action_type_seen = {}

    def fail(self, msg, ctx=None):
        self.pass_all = False
        if len(self.failures) < self.max_fail_examples:
            row = {"message": msg}
            if ctx is not None:
                row["context"] = ctx
            self.failures.append(row)

    def check(self, cond, msg, ctx=None):
        if not bool(cond):
            self.fail(msg, ctx)

    def iter_valid_slots(self, info, uw):
        feats = np.asarray(info["legal_action_features_padded"], dtype=np.float32)
        ids = np.asarray(info["legal_global_ids_padded"], dtype=np.int64).reshape(-1)
        valid = np.asarray(info["legal_action_valid_mask"], dtype=bool).reshape(-1)

        self.check(tuple(np.asarray(info.get("legal_action_features_padded")).shape[1:2]) == (EXPECTED_FEAT_DIM,),
                   "feature_dim mismatch", {"shape": np.asarray(info.get("legal_action_features_padded")).shape})
        self.check(tuple(np.asarray(info.get("legal_global_ids_padded")).shape) == tuple(np.asarray(info.get("legal_action_valid_mask")).shape),
                   "ids/valid shape mismatch")
        self.check(feats.shape[0] == ids.shape[0] == valid.shape[0], "slot first-dim mismatch")

        for s in np.where(valid)[0]:
            gid = int(ids[s])
            raw_idx = uw._current_legal_id_to_raw_index.get(gid, None)
            if raw_idx is None or int(raw_idx) < 0 or int(raw_idx) >= len(uw._current_legal_actions):
                self.fail("valid slot does not map to legal action", {"slot": int(s), "gid": gid})
                continue
            action = uw._current_legal_actions[int(raw_idx)]
            yield int(s), int(gid), int(raw_idx), action, feats[int(s)]

    @staticmethod
    def _min_dist(x, y, targets):
        best = None
        for tx, ty in targets:
            d = abs(int(x) - int(tx)) + abs(int(y) - int(ty))
            if best is None or d < best:
                best = d
        return best

    def choose_action_gid(self, info, uw, raw_obs):
        # Build candidates from current valid slots for deterministic selection.
        capture_gids = []
        move_candidates = []  # tuple(priority_bucket, score, gid)
        fallback_gids = []

        villages = uw._get_visible_uncaptured_village_positions(raw_obs)
        has_villages = len(villages) > 0

        for _slot, gid, _raw_idx, action, _feat in self.iter_valid_slots(info, uw):
            a_type = str(action.get("type", "")).upper()
            fallback_gids.append(gid)
            if a_type == "CAPTURE":
                capture_gids.append(gid)
                continue
            if a_type != "MOVE":
                continue

            unit_id, src_x, src_y, dst_x, dst_y = uw._extract_move_components(action, raw_obs)
            if dst_x is None or dst_y is None:
                continue

            # 1) Moves landing on visible uncaptured villages.
            if has_villages and (int(dst_x), int(dst_y)) in villages:
                move_candidates.append((0, 0.0, gid))
                continue

            # 2) Moves reducing distance to visible uncaptured villages.
            if has_villages:
                if src_x is None or src_y is None:
                    if unit_id is not None:
                        pos = uw._unit_position_by_id(raw_obs, unit_id)
                        if pos is not None:
                            src_x, src_y = int(pos[0]), int(pos[1])
                if src_x is not None and src_y is not None:
                    before = self._min_dist(src_x, src_y, villages)
                    after = self._min_dist(dst_x, dst_y, villages)
                    if before is not None and after is not None and int(after) < int(before):
                        # Larger distance reduction first.
                        move_candidates.append((1, float(before - after), gid))
                        continue

            # 3) High-fog-reveal moves.
            revealed = uw._estimate_newly_revealed_tiles_if_move(raw_obs, unit_id, int(dst_x), int(dst_y)) if unit_id is not None else 0
            move_candidates.append((2, float(revealed), gid))

        if capture_gids:
            return int(sorted(capture_gids)[0]), "capture"

        if move_candidates:
            # Stable deterministic sort by bucket then score desc then gid asc.
            move_candidates_sorted = sorted(
                move_candidates,
                key=lambda t: (int(t[0]), -float(t[1]), int(t[2])),
            )
            return int(move_candidates_sorted[0][2]), "move_heuristic"

        if fallback_gids:
            return int(sorted(fallback_gids)[0]), "fallback"
        return None, "none"

    def validate_capture_slots(self, info, uw, raw_obs, episode_idx, step_idx):
        capture_slots = []
        for slot, gid, raw_idx, action, feat in self.iter_valid_slots(info, uw):
            a_type = str(action.get("type", "")).upper()
            if a_type == "CAPTURE":
                capture_slots.append((slot, gid, raw_idx, action, feat))

        if not capture_slots:
            return

        self.states_with_capture_legal += 1

        for slot, gid, raw_idx, action, feat in capture_slots:
            self.capture_actions_inspected += 1

            self.check(float(feat[IDX_IS_CAPTURE_OLD]) > 0.5,
                       "CAPTURE slot old is_capture feature != 1",
                       {"ep": episode_idx, "step": step_idx, "slot": slot, "gid": gid})

            self.check(str(action.get("type", "")).upper() == "CAPTURE",
                       "decoded action type for CAPTURE slot is not CAPTURE",
                       {"ep": episode_idx, "step": step_idx, "slot": slot, "gid": gid, "type": action.get("type")})

            # Appended research/resource/build/levelup-specific flags should be zero for CAPTURE.
            nonzero = []
            for idx in APPENDED_RANGE:
                v = float(feat[idx])
                if abs(v) > 1e-8:
                    nonzero.append((idx, v))
            self.check(
                len(nonzero) == 0,
                "CAPTURE slot has non-zero appended semantic/economy features",
                {"ep": episode_idx, "step": step_idx, "slot": slot, "gid": gid, "nonzero": nonzero[:10]},
            )

            # Slot alignment check by recomputation.
            expected = np.asarray(uw._compute_legal_action_feature_vector_reference(action, raw_obs), dtype=np.float32)
            self.check(
                np.allclose(expected, feat, atol=1e-6, rtol=1e-6),
                "CAPTURE slot feature not aligned with decoded action",
                {"ep": episode_idx, "step": step_idx, "slot": slot, "gid": gid, "raw_idx": raw_idx},
            )

    def run(self):
        set_required_env()
        env = gym.make("Tribes-v0")
        try:
            for ep in range(MAX_EPISODES):
                if self.capture_actions_inspected >= CAPTURE_TARGET_ACTIONS:
                    break
                self.episodes_run += 1
                obs, info = env.reset(seed=1000 + ep)
                uw = env.unwrapped

                self.check(tuple(np.asarray(obs).shape) == (EXPECTED_OBS_DIM,),
                           "obs dim mismatch at reset",
                           {"ep": ep, "shape": tuple(np.asarray(obs).shape)})

                for step in range(MAX_STEPS_PER_EPISODE):
                    raw_obs = uw.tribes_env._last_obs
                    self.validate_capture_slots(info, uw, raw_obs, ep, step)
                    if self.capture_actions_inspected >= CAPTURE_TARGET_ACTIONS:
                        break

                    before_city_count = int(uw._get_city_count(raw_obs))
                    before_capture_village_count = int(info.get("captured_villages_t10", 0))

                    chosen_gid, choose_mode = self.choose_action_gid(info, uw, raw_obs)
                    self.check(chosen_gid is not None, "failed to choose a legal action gid", {"ep": ep, "step": step, "mode": choose_mode})
                    if chosen_gid is None:
                        break

                    # Find selected action metadata pre-step for capture transition checks.
                    selected_action = None
                    selected_slot = None
                    ids = np.asarray(info["legal_global_ids_padded"], dtype=np.int64).reshape(-1)
                    valid = np.asarray(info["legal_action_valid_mask"], dtype=bool).reshape(-1)
                    for s in np.where(valid)[0]:
                        if int(ids[s]) == int(chosen_gid):
                            selected_slot = int(s)
                            raw_idx = uw._current_legal_id_to_raw_index.get(int(chosen_gid), None)
                            if raw_idx is not None and 0 <= int(raw_idx) < len(uw._current_legal_actions):
                                selected_action = uw._current_legal_actions[int(raw_idx)]
                            break

                    pre_capture_unit_pos = None
                    pre_capture_was_village = False
                    if isinstance(selected_action, dict) and str(selected_action.get("type", "")).upper() == "CAPTURE":
                        uid = uw._action_int(selected_action, "unit_id", None)
                        if uid is None:
                            uid = uw._parse_unit_id_from_action_repr(str(selected_action.get("repr", "")))
                        if uid is not None:
                            pos = uw._unit_position_by_id(raw_obs, uid)
                            if pos is not None:
                                pre_capture_unit_pos = (int(pos[0]), int(pos[1]))
                                pre_capture_was_village = pre_capture_unit_pos in uw._get_visible_uncaptured_village_positions(raw_obs)

                    try:
                        obs_next, _rew, terminated, truncated, info_next = env.step(int(chosen_gid))
                    except Exception as exc:
                        self.fail("env.step failed on chosen legal gid", {"ep": ep, "step": step, "gid": int(chosen_gid), "err": str(exc)})
                        break

                    selected_type_post = str(info_next.get("selected_action_type", "UNKNOWN")).upper()
                    self.action_type_seen[selected_type_post] = int(self.action_type_seen.get(selected_type_post, 0) + 1)

                    if selected_type_post == "CAPTURE":
                        after_raw_obs = uw.tribes_env._last_obs
                        after_city_count = int(uw._get_city_count(after_raw_obs))
                        after_capture_village_count = int(info_next.get("captured_villages_t10", before_capture_village_count))

                        transition_success = False
                        if after_city_count > before_city_count:
                            transition_success = True
                        elif after_capture_village_count > before_capture_village_count:
                            transition_success = True
                        elif pre_capture_unit_pos is not None and pre_capture_was_village:
                            post_villages = uw._get_visible_uncaptured_village_positions(after_raw_obs)
                            if pre_capture_unit_pos not in post_villages:
                                transition_success = True

                        self.check(
                            transition_success,
                            "CAPTURE execution did not show city-count increase or expected capture transition",
                            {
                                "ep": ep,
                                "step": step,
                                "before_city_count": before_city_count,
                                "after_city_count": after_city_count,
                                "before_captured_villages_t10": before_capture_village_count,
                                "after_captured_villages_t10": after_capture_village_count,
                                "pre_capture_unit_pos": pre_capture_unit_pos,
                                "selected_slot": selected_slot,
                            },
                        )
                        if transition_success:
                            self.successful_capture_exec += 1

                    obs, info = obs_next, info_next
                    if bool(terminated) or bool(truncated):
                        break
        finally:
            env.close()

        print("=== Focused CAPTURE Validation Report ===")
        print(f"PASS: {self.pass_all}")
        print(f"episodes_run: {self.episodes_run}")
        print(f"states_with_capture_legal: {self.states_with_capture_legal}")
        print(f"capture_actions_inspected: {self.capture_actions_inspected}")
        print(f"successful_capture_executions: {self.successful_capture_exec}")
        print(f"selected_action_type_counts: {self.action_type_seen}")
        print(f"assertion_failures: {len(self.failures)}")
        if self.failures:
            print("failure_examples:")
            for row in self.failures[:20]:
                print(f"  - {row}")

        return 0 if self.pass_all else 1


if __name__ == "__main__":
    cv = CaptureValidator(seed=123)
    rc = cv.run()
    if rc != 0:
        # Helpful trace marker for CI logs.
        print("capture_validation_result: FAIL")
    else:
        print("capture_validation_result: PASS")
    raise SystemExit(rc)

