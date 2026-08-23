import os
import sys

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Set environment variables for Phase 1 single-tribe training path
os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"
os.environ["POLYVISION_INFO_MODE"] = "debug"
training_map = os.path.join("levels", "phase1_pool_bardur_real", "train", "map_000001.csv")
os.environ["POLYVISION_LEVEL_POOL_GLOB"] = training_map.replace("\\", "/")
os.environ["POLYVISION_LEVEL_SELECTION_MODE"] = "fixed"

import gymnasium as gym
from pol_env.Tribes.py.register_env import TribesGymWrapper

def run_diagnostic():
    print("=" * 80)
    print("PHASE 1 TURN ECONOMY DIAGNOSTIC")
    print("=" * 80)
    print(f"Target Map: {training_map}")
    print(f"POLYVISION_SOLO_NO_OPPONENT_MODE: {os.environ.get('POLYVISION_SOLO_NO_OPPONENT_MODE')}")
    print()

    env = gym.make("Tribes-v0")
    raw_wrapper = env.unwrapped if hasattr(env, "unwrapped") else env

    obs_arr, info = env.reset(seed=42)

    def get_state_snapshot():
        raw_obs = getattr(raw_wrapper.tribes_env, "_last_obs", {}) or {}
        java_env = raw_wrapper.tribes_env._env
        java_tick = int(java_env.getTick())
        active_tribe = int(java_env.getActiveTribeID())
        java_done = bool(java_env.isDone())
        wrapper_turn = int(raw_wrapper._turn_count)
        stars = int(raw_wrapper._get_bardur_stars(raw_obs))
        spt = float(raw_wrapper._compute_bardur_spt(raw_obs))
        city_count = int(raw_wrapper._get_city_count(raw_obs))

        # Sum of owned city productions
        cities_dict = raw_obs.get("city", {}) or {}
        owned_city_prod = sum(
            float(c.get("production", 0))
            for c in cities_dict.values()
            if int(c.get("tribeID", -1)) == 0
        )
        queued_captures = list(raw_wrapper._queued_village_capture_unit_ids)
        solo_continuation = bool(java_env.getSoloNoOpponentMode())
        return {
            "wrapper_turn": wrapper_turn,
            "java_tick": java_tick,
            "active_tribe": active_tribe,
            "stars": stars,
            "spt": spt,
            "owned_city_prod": owned_city_prod,
            "city_count": city_count,
            "java_done": java_done,
            "queued_captures": queued_captures,
            "solo_continuation": solo_continuation,
        }

    snap0 = get_state_snapshot()
    print("INITIAL STATE AFTER RESET (HANDOFF TO AGENT):")
    print(f"  Wrapper Turn:     {snap0['wrapper_turn']}")
    print(f"  Java Tick:        {snap0['java_tick']}")
    print(f"  Active Tribe ID:  {snap0['active_tribe']}")
    print(f"  Stars:            {snap0['stars']}")
    print(f"  Bardur SPT:       {snap0['spt']}")
    print(f"  City Production:  {snap0['owned_city_prod']}")
    print(f"  City Count:       {snap0['city_count']}")
    print(f"  Java isDone():    {snap0['java_done']}")
    print(f"  Solo Continuation:{snap0['solo_continuation']}")
    print("-" * 80)

    timeline = []
    timeline.append({
        "step": 0,
        "event": "RESET_HANDOFF",
        "action_type": "-",
        "global_id": "-",
        "wrapper_turn": snap0["wrapper_turn"],
        "java_tick": snap0["java_tick"],
        "stars": snap0["stars"],
        "spt": snap0["spt"],
        "city_prod": snap0["owned_city_prod"],
        "city_count": snap0["city_count"],
        "java_done": snap0["java_done"],
    })

    # Step through at least 3 turn transitions without spending stars
    step_count = 0
    max_steps = 30

    while step_count < max_steps and snap0["wrapper_turn"] <= 6:
        step_count += 1
        valid_slots = raw_wrapper._current_legal_id_to_raw_index
        legal_actions = raw_wrapper._current_legal_actions

        # Look for END_TURN first
        end_turn_global = raw_wrapper._catalog.id_end_turn()
        selected_global = None

        if end_turn_global in valid_slots:
            selected_global = end_turn_global
        else:
            # Pick a valid non-spending action (e.g. MOVE)
            for gid in sorted(valid_slots.keys()):
                raw_idx = valid_slots[gid]
                act = legal_actions[raw_idx]
                act_type = str(act.get("type", "")).upper()
                if act_type not in ("RESEARCH_TECH", "BUILD", "TRAIN", "SPAWN", "RESOURCE_GATHERING"):
                    selected_global = gid
                    break
            if selected_global is None:
                selected_global = list(valid_slots.keys())[0]

        raw_idx = valid_slots[selected_global]
        action_obj = legal_actions[raw_idx]
        action_type = str(action_obj.get("type", "UNKNOWN")).upper()
        action_repr = str(action_obj.get("repr", ""))

        snap_before = get_state_snapshot()

        obs_arr, reward, terminated, truncated, step_info = env.step(selected_global)

        snap_after = get_state_snapshot()

        is_turn_boundary = (action_type == "END_TURN")

        timeline.append({
            "step": step_count,
            "event": "TURN_TRANSITION" if is_turn_boundary else "INTRA_TURN_ACTION",
            "action_type": action_type,
            "action_repr": action_repr,
            "global_id": selected_global,
            "wrapper_turn": snap_after["wrapper_turn"],
            "java_tick": snap_after["java_tick"],
            "stars": snap_after["stars"],
            "spt": snap_after["spt"],
            "city_prod": snap_after["owned_city_prod"],
            "city_count": snap_after["city_count"],
            "java_done": snap_after["java_done"],
            "terminated": terminated,
            "truncated": truncated,
        })

        if is_turn_boundary:
            print(f"[Step {step_count:2d}] END_TURN executed:")
            print(f"   Before: wrapper_turn={snap_before['wrapper_turn']}, java_tick={snap_before['java_tick']}, stars={snap_before['stars']}, spt={snap_before['spt']}")
            print(f"   After:  wrapper_turn={snap_after['wrapper_turn']}, java_tick={snap_after['java_tick']}, stars={snap_after['stars']}, spt={snap_after['spt']}")
            print(f"   Java isDone: {snap_after['java_done']}, Terminated: {terminated}, Truncated: {truncated}")
            print()

        if terminated or truncated:
            print(f"Episode ended at step {step_count} (term={terminated}, trunc={truncated})")
            break

    env.close()

    print("=" * 80)
    print("DIAGNOSTIC TIMELINE TABLE")
    print("=" * 80)
    header = f"{'Step':<5} | {'Action Type':<12} | {'Global ID':<9} | {'Turn':<5} | {'Tick':<5} | {'Stars':<6} | {'SPT':<5} | {'CityProd':<8} | {'Cities':<6} | {'JavaDone':<8}"
    print(header)
    print("-" * len(header))
    for t in timeline:
        row = (
            f"{t['step']:<5} | "
            f"{t['action_type'][:12]:<12} | "
            f"{str(t['global_id']):<9} | "
            f"{t['wrapper_turn']:<5} | "
            f"{t['java_tick']:<5} | "
            f"{t['stars']:<6} | "
            f"{t['spt']:<5.1f} | "
            f"{t['city_prod']:<8.1f} | "
            f"{t['city_count']:<6} | "
            f"{str(t['java_done']):<8}"
        )
        print(row)

    print("=" * 80)
    print("TURN / ECONOMY INVARIANT VERIFICATION")
    print("=" * 80)
    # Check that for all turn transitions, stars accumulated properly
    verified_transitions = 0
    for i in range(1, len(timeline)):
        t_prev = timeline[i - 1]
        t_cur = timeline[i]
        if t_cur["event"] == "TURN_TRANSITION":
            expected_tick = t_prev["java_tick"] + 1
            if t_cur["java_tick"] != expected_tick:
                raise AssertionError(f"Java tick mismatch at step {t_cur['step']}: expected {expected_tick}, got {t_cur['java_tick']}")
            # If no spending action was taken at this step, stars must equal previous stars + new turn city production
            expected_stars = t_prev["stars"] + int(t_cur["city_prod"])
            if t_cur["stars"] != expected_stars:
                raise AssertionError(
                    f"Star accumulation mismatch at step {t_cur['step']}: "
                    f"previous stars={t_prev['stars']} + new turn city_prod={t_cur['city_prod']} = expected {expected_stars}, "
                    f"got {t_cur['stars']}"
                )
            verified_transitions += 1
            print(f"  Turn {t_prev['wrapper_turn']} -> {t_cur['wrapper_turn']} verified: stars {t_prev['stars']} + prod {int(t_cur['city_prod'])} -> {t_cur['stars']} (Java tick {t_prev['java_tick']} -> {t_cur['java_tick']})")
    print()
    print(f"Status: ALL {verified_transitions} TURN TRANSITIONS VERIFIED ACCUMULATING ACCURATELY.")
    print("=" * 80)

if __name__ == "__main__":
    run_diagnostic()
