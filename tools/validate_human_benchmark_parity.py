#!/usr/bin/env python3
"""Validate official human-menu parity with the PPO-facing Phase 1 wrapper."""

from __future__ import annotations

import argparse
import ast
import inspect
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import human_policy_interface
from tools.human_benchmark import POOL_ROOT, load_benchmark_maps, official_environment
from tools.human_policy_interface import capture_environment_contract, policy_visible_actions, policy_visible_ids


FORBIDDEN_OFFICIAL_ATTRIBUTES = {
    "tribes_env",
    "_last_obs",
    "_current_legal_actions",
    "_current_legal_id_to_raw_index",
    "get_observation",
    "render",
    "observationJsonFull",
}


def assert_information_safety() -> None:
    source = inspect.getsource(human_policy_interface)
    tree = ast.parse(source)
    offenders = sorted(
        {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_OFFICIAL_ATTRIBUTES}
    )
    if offenders:
        raise RuntimeError(f"official presentation module references privileged/raw APIs: {offenders}")


def assert_state_parity(human_env, model_env, human_obs, model_obs, human_info, model_info) -> list[int]:
    if not np.array_equal(np.asarray(human_obs), np.asarray(model_obs)):
        raise RuntimeError("human/model flattened observations differ")
    human_contract = capture_environment_contract(human_env, human_info)
    model_contract = capture_environment_contract(model_env, model_info)
    if human_contract != model_contract:
        raise RuntimeError("human/model environment contracts differ")
    human_ids = [int(action["global_id"]) for action in policy_visible_actions(human_env, human_info)]
    model_ids = policy_visible_ids(model_info)
    if human_ids != model_ids:
        raise RuntimeError("human menu differs from legal_global_ids_padded[legal_action_valid_mask]")

    wrapper = human_env.unwrapped
    for gid in human_ids:
        if gid not in wrapper._current_legal_id_to_raw_index:
            raise RuntimeError(f"human menu global ID {gid} has no wrapper raw-action mapping")
        raw_index = int(wrapper._current_legal_id_to_raw_index[gid])
        raw_action = wrapper._current_legal_actions[raw_index]
        canonical_gid, reason = wrapper._canonicalize_action_to_global_id(raw_action, wrapper.tribes_env._last_obs)
        if reason is not None or int(canonical_gid) != gid:
            raise RuntimeError(f"global ID {gid} does not resolve to the same raw Java action")
    return human_ids


def validate_map(record: dict, seed: int, max_states: int, require_horizon: bool) -> tuple[int, int]:
    with official_environment(record["path"]) as human_env:
        with official_environment(record["path"]) as model_env:
            human_obs, human_info = human_env.reset(seed=seed)
            model_obs, model_info = model_env.reset(seed=seed)
            if int(human_info["turn_count"]) != int(model_info["turn_count"]):
                raise RuntimeError("scripted opening produced different starting turns")
            checked = 0
            filtered_raw = 0
            terminated = truncated = False
            cap = 500 if require_horizon else max_states
            while checked < cap:
                ids = assert_state_parity(human_env, model_env, human_obs, model_obs, human_info, model_info)
                filtered_raw += max(0, int(human_env.unwrapped._current_raw_valid_actions) - len(ids))
                selected = int(ids[0])
                expected_human_raw = int(human_env.unwrapped._current_legal_id_to_raw_index[selected])
                expected_model_raw = int(model_env.unwrapped._current_legal_id_to_raw_index[selected])
                human_action = human_env.unwrapped._current_legal_actions[expected_human_raw]
                model_action = model_env.unwrapped._current_legal_actions[expected_model_raw]
                if (human_action.get("type"), human_action.get("repr")) != (
                    model_action.get("type"), model_action.get("repr")
                ):
                    raise RuntimeError("human/model global ID resolved to different Java action semantics")
                human_obs, human_reward, human_terminated, human_truncated, human_info = human_env.step(selected)
                model_obs, model_reward, model_terminated, model_truncated, model_info = model_env.step(selected)
                if float(human_reward) != float(model_reward):
                    raise RuntimeError("human/model rewards differ for the same global ID")
                if (bool(human_terminated), bool(human_truncated)) != (
                    bool(model_terminated), bool(model_truncated)
                ):
                    raise RuntimeError("human/model horizon signals differ")
                if int(human_info.get("selected_global_id", -1)) != selected:
                    raise RuntimeError("human env did not execute the selected global ID")
                deferred_end_turn = human_action.get("type") == "END_TURN"
                if not deferred_end_turn and int(human_info.get("selected_raw_java_index", -1)) != expected_human_raw:
                    raise RuntimeError(
                        "human env executed a different raw Java action: "
                        f"gid={selected}, expected={expected_human_raw}, "
                        f"actual={human_info.get('selected_raw_java_index')}, action={human_action}"
                    )
                if not deferred_end_turn and int(model_info.get("selected_raw_java_index", -1)) != expected_model_raw:
                    raise RuntimeError("model env raw-action resolution differs")
                checked += 1
                terminated, truncated = bool(human_terminated), bool(human_truncated)
                if terminated or truncated:
                    break
            if require_horizon:
                if not truncated or int(human_info.get("turn_count", -1)) <= int(human_env.unwrapped.MAX_TURNS):
                    raise RuntimeError("episode did not terminate through the wrapper's Turn-10 truncation contract")
            elif checked != max_states and not (terminated or truncated):
                raise RuntimeError("parity sampling ended unexpectedly")
            return checked, filtered_raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=int, default=3)
    parser.add_argument("--states-per-map", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-root", type=Path, default=POOL_ROOT)
    args = parser.parse_args()

    assert_information_safety()
    maps, _manifest = load_benchmark_maps(args.pool_root.resolve(), args.pool_root.resolve() / "split_manifest.json")
    selected = maps[: max(1, min(len(maps), int(args.maps)))]
    total_states = 0
    filtered_raw = 0
    for index, record in enumerate(selected):
        checked, filtered = validate_map(
            record,
            seed=int(args.seed),
            max_states=max(1, int(args.states_per_map)),
            require_horizon=index == 0,
        )
        total_states += checked
        filtered_raw += filtered
        print(f"PASS {record['filename']}: states={checked}, filtered_raw_actions_observed={filtered}")
    print(f"Human/model parity passed: maps={len(selected)}, states={total_states}")
    print(f"Java-legal actions excluded by the policy interface across sampled states: {filtered_raw}")
    print("Official information-safety source audit: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
