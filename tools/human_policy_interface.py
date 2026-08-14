"""Human-readable presentation of the authoritative PPO-facing Phase 1 interface.

Official benchmark code in this module consumes only the flattened policy
observation, the legal-slot tensors returned in ``info``, and stable global-ID
catalog metadata. It never reads the Java observation, raw action dictionaries,
``_last_obs``, debug info, or a renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import numpy as np

from pol_env.Tribes.py.environment_contract import observation_layout


TERRAIN_SYMBOLS = {0: ".", 1: "~", 2: "D", 3: "M", 4: "V", 5: "C", 6: "F", 7: "?"}
RESOURCE_SYMBOLS = {0: "h", 1: "f", 2: "a", 3: "w", 5: "o", 6: "c", 7: "r"}
SAFE_INFO_MODE = "fast"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item() if value.size else default
    return value


def policy_visible_ids(info: dict[str, Any]) -> list[int]:
    ids = np.asarray(info.get("legal_global_ids_padded", []), dtype=np.int64).reshape(-1)
    valid = np.asarray(info.get("legal_action_valid_mask", []), dtype=bool).reshape(-1)
    if ids.shape != valid.shape:
        raise RuntimeError(f"legal-slot shape mismatch: ids={ids.shape}, mask={valid.shape}")
    selected = [int(value) for value in ids[valid]]
    if len(selected) != len(set(selected)):
        raise RuntimeError("policy-visible legal global IDs contain duplicates")
    declared = int(info.get("legal_action_count", len(selected)))
    if declared != len(selected):
        raise RuntimeError(f"legal_action_count={declared} but visible slot count={len(selected)}")
    return selected


def _tile_xy(catalog: Any, tile: int) -> tuple[int, int]:
    return int(tile) // int(catalog.height), int(tile) % int(catalog.height)


def decode_global_action(env: Any, global_id: int) -> tuple[str, str]:
    """Decode a stable global ID without consulting a raw Java action object."""
    wrapper = env.unwrapped
    catalog = wrapper._catalog
    gid = int(global_id)
    offsets = catalog.offsets
    ordered = sorted((int(offset), name) for name, offset in offsets.items())
    family = None
    end = int(catalog.total_size)
    for index, (start, name) in enumerate(ordered):
        next_start = ordered[index + 1][0] if index + 1 < len(ordered) else end
        if start <= gid < next_start:
            family = name
            local = gid - start
            break
    if family is None:
        raise RuntimeError(f"global ID {gid} is outside the catalog")

    n = int(catalog.n_tiles)
    if family == "END_TURN":
        return family, "End turn"
    if family == "MOVE":
        src, dst = divmod(local, n)
        return family, f"Move unit {_tile_xy(catalog, src)} -> {_tile_xy(catalog, dst)}"
    if family == "CAPTURE":
        capture_index, remainder = divmod(local, n * n)
        src, dst = divmod(remainder, n)
        capture_name = ("city", "village", "target")[min(capture_index, 2)]
        return family, f"Capture {capture_name} from {_tile_xy(catalog, src)} at {_tile_xy(catalog, dst)}"
    if family == "TRAIN":
        vocab_index, tile = divmod(local, n)
        unit = catalog.train_unit_types[vocab_index]
        return family, f"Train {unit} at city {_tile_xy(catalog, tile)}"
    if family == "RESOURCE_GATHERING":
        vocab_index, tile = divmod(local, n)
        resource = catalog.resource_types[vocab_index]
        return family, f"Gather {resource} at {_tile_xy(catalog, tile)}"
    if family == "CLEAR_FOREST":
        return family, f"Clear forest at {_tile_xy(catalog, local)}"
    if family == "GROW_FOREST":
        return family, f"Grow forest at {_tile_xy(catalog, local)}"
    if family == "BUILD":
        vocab_index, tile = divmod(local, n)
        building = catalog.building_types[vocab_index]
        return family, f"Build {building} at {_tile_xy(catalog, tile)}"
    if family == "RESEARCH_TECH":
        return family, f"Research {catalog.tech_types[local]}"
    if family == "LEVEL_UP":
        vocab_index, tile = divmod(local, n)
        choice = catalog.levelup_choices[vocab_index]
        return family, f"Level up city {_tile_xy(catalog, tile)} with {choice}"
    if family == "EXAMINE":
        return family, f"Examine unit at {_tile_xy(catalog, local)}"
    raise RuntimeError(f"unsupported catalog family {family!r}")


def policy_visible_actions(env: Any, info: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"slot": slot, "global_id": gid, "type": action_type, "description": description}
        for slot, gid in enumerate(policy_visible_ids(info))
        for action_type, description in [decode_global_action(env, gid)]
    ]


def visible_state(observation: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
    width = int(info["map_width"])
    height = int(info["map_height"])
    layout = observation_layout(width, height)
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    if obs.size != layout.expected_obs_dim:
        raise RuntimeError(f"observation has {obs.size} values; expected {layout.expected_obs_dim}")
    n = layout.n_tiles
    legacy = 3 * n
    scalar = layout.scalar_start
    terrain = np.rint(obs[:n]).astype(np.int16)
    units = np.rint(obs[n : 2 * n]).astype(np.int64)
    cities = np.rint(obs[2 * n : 3 * n]).astype(np.int64)
    resources = np.rint(obs[layout.resource_start : layout.resource_end] * 8.0 - 1.0).astype(np.int16)
    return {
        "width": width,
        "height": height,
        "terrain": terrain,
        "unit_ids": units,
        "city_ids": cities,
        "resources": resources,
        "stars": int(round(float(obs[legacy]))),
        "score": int(round(float(obs[legacy + 1]))),
        "city_count": int(round(float(obs[legacy + 2]))),
        "turn": int(round(float(obs[scalar + 2]) * 10.0)),
        "spt": int(round(float(obs[scalar + 1]) * 30.0)),
        "tech_organization": bool(obs[scalar + 5] >= 0.5),
        "tech_forestry": bool(obs[scalar + 6] >= 0.5),
        "tech_count": int(round(float(obs[scalar + 7]) * 24.0)),
        "avg_city_level": float(obs[scalar + 9]) * 5.0,
        "max_city_level": float(obs[scalar + 10]) * 5.0,
        "visible_unit_count": int(np.sum(units > 0)),
    }


def visible_map_lines(state: dict[str, Any]) -> list[str]:
    width, height = int(state["width"]), int(state["height"])
    terrain = state["terrain"]
    units = state["unit_ids"]
    resources = state["resources"]
    lines = ["Visible map (terrain/unit/resource; ? is fog)"]
    lines.append("    " + " ".join(f"{x:>3}" for x in range(width)))
    for y in range(height):
        cells = []
        for x in range(width):
            tile = x * height + y
            terrain_id = int(terrain[tile])
            if terrain_id == 7:
                cells.append("???")
                continue
            terr = TERRAIN_SYMBOLS.get(terrain_id, str(terrain_id)[-1])
            unit = "U" if int(units[tile]) > 0 else "."
            resource = RESOURCE_SYMBOLS.get(int(resources[tile]), ".")
            cells.append(f"{terr}{unit}{resource}")
        lines.append(f"{y:>3} " + " ".join(cells))
    lines.append("Legend: .=plain ~=water D=deep M=mountain V=village C=city F=forest; U=visible unit")
    lines.append("Resources: a=animal f=fruit h=fish w=whale o=ore c=crops r=ruin")
    return lines


def visible_metrics(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: state[key]
        for key in (
            "turn", "stars", "spt", "city_count", "visible_unit_count", "tech_count",
            "tech_organization", "tech_forestry", "avg_city_level", "max_city_level",
        )
    }


def capture_environment_contract(env: Any, info: dict[str, Any]) -> dict[str, Any]:
    wrapper = env.unwrapped
    map_width = info["map_width"] if "map_width" in info else wrapper._catalog.width
    map_height = info["map_height"] if "map_height" in info else wrapper._catalog.height
    observation_dim = (
        info["observation_dim"] if "observation_dim" in info else wrapper.observation_space.shape[0]
    )
    global_action_space_n = (
        info["global_action_space_n"] if "global_action_space_n" in info else wrapper.action_space.n
    )
    max_legal_actions = (
        info["max_legal_actions"] if "max_legal_actions" in info else wrapper._max_legal_actions
    )
    catalog_version = info["catalog_version"] if "catalog_version" in info else wrapper.CATALOG_VERSION
    catalog_fingerprint = (
        info["action_catalog_fingerprint"]
        if "action_catalog_fingerprint" in info
        else wrapper._catalog_fingerprint
    )
    canonicalizer_version = (
        info["canonicalizer_version"] if "canonicalizer_version" in info else wrapper.CANONICALIZER_VERSION
    )
    opening_version = (
        info["phase1_opening_version"]
        if "phase1_opening_version" in info
        else wrapper.PHASE1_OPENING_VERSION
    )
    feature_version = (
        info["legal_action_feature_version"]
        if "legal_action_feature_version" in info
        else wrapper.LEGAL_ACTION_FEATURE_VERSION
    )
    feature_dim = (
        info["legal_action_feature_dim"] if "legal_action_feature_dim" in info else wrapper.ACTION_FEATURE_DIM
    )
    info_mode = info["info_mode"] if "info_mode" in info else wrapper._info_mode
    return {
        "wrapper_class": f"{type(wrapper).__module__}.{type(wrapper).__name__}",
        "map_width": int(map_width),
        "map_height": int(map_height),
        "observation_dim": int(observation_dim),
        "global_action_space_n": int(global_action_space_n),
        "max_legal_actions": int(max_legal_actions),
        "catalog_version": str(catalog_version),
        "catalog_fingerprint": str(catalog_fingerprint),
        "canonicalizer_version": str(canonicalizer_version),
        "phase1_opening_version": str(opening_version),
        "legal_action_feature_version": str(feature_version),
        "legal_action_feature_dim": int(feature_dim),
        "max_turns": int(wrapper.MAX_TURNS),
        "controlled_tribe_id": 0,
        "solo_no_opponent_mode": True,
        "info_mode": str(info_mode),
        "allowed_action_types": sorted(str(value) for value in wrapper.ALLOWED_ACTION_TYPES),
        "terminal_spt_reward_enabled": bool(wrapper._terminal_spt_reward_enabled),
        "terminal_spt_base_weight": float(wrapper._terminal_spt_base_weight),
        "terminal_spt_over_10_weight": float(wrapper._terminal_spt_over_10_weight),
        "terminal_spt_over_15_weight": float(wrapper._terminal_spt_over_15_weight),
        "resource_gather_upgrade_filter_enabled": bool(wrapper._resource_gather_upgrade_filter_enabled),
    }


def validate_official_contract(env: Any, info: dict[str, Any]) -> None:
    wrapper = env.unwrapped
    if type(wrapper).__name__ != "TribesGymWrapper":
        raise RuntimeError(f"official benchmark requires TribesGymWrapper, got {type(wrapper).__name__}")
    if str(info.get("info_mode")) != SAFE_INFO_MODE:
        raise RuntimeError(f"official benchmark requires info_mode={SAFE_INFO_MODE!r}")
    if int(info.get("max_legal_actions", -1)) != int(wrapper.MAX_LEGAL_ACTIONS_DEFAULT):
        raise RuntimeError(
            "official benchmark legal-slot capacity differs from the current Phase 1 default: "
            f"{info.get('max_legal_actions')} != {wrapper.MAX_LEGAL_ACTIONS_DEFAULT}"
        )
    if int(info.get("global_action_space_n", -1)) != int(env.action_space.n):
        raise RuntimeError("global action-space metadata disagrees with the environment")
    policy_visible_ids(info)


def _print_actions(actions: list[dict[str, Any]], page: int, page_size: int, output: Callable[[str], None]) -> int:
    pages = max(1, (len(actions) + page_size - 1) // page_size)
    page = max(0, min(page, pages - 1))
    start, end = page * page_size, min(len(actions), (page + 1) * page_size)
    output(f"Legal policy actions {start}-{end - 1} of {len(actions) - 1} (page {page + 1}/{pages})")
    last_type = None
    for index in range(start, end):
        action = actions[index]
        if action["type"] != last_type:
            output(f"  [{action['type']}]")
            last_type = action["type"]
        output(f"  {index:>3}: gid={action['global_id']:>6}  {action['description']}")
    return page


@dataclass
class EpisodeResult:
    status: str
    started_at_utc: str
    ended_at_utc: str
    episode_seed: int
    shaped_return: float
    decision_count: int
    final_visible_metrics: dict[str, Any]
    final_info_metrics: dict[str, Any]
    environment_contract: dict[str, Any]
    action_history: list[dict[str, Any]]


SAFE_FINAL_INFO_KEYS = (
    "turn_count", "spt", "terminal_final_spt", "stars", "city_count", "avg_city_level",
    "unit_count", "techs_researched", "forestry_researched", "organization_researched",
    "fog_tiles_cleared_total", "captured_villages_t10", "capturable_villages_total",
    "village_capture_pct_t10", "animals_harvested_t10", "fruit_harvested_t10",
    "lumber_huts_built_t10", "sawmills_built_t10", "forests_cleared_t10",
)


def run_policy_visible_episode(
    env: Any,
    *,
    episode_seed: int,
    official: bool,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    page_size: int = 30,
    selector: Callable[[list[dict[str, Any]], np.ndarray, dict[str, Any], int], int | None] | None = None,
    state_callback: Callable[[Any, np.ndarray, dict[str, Any]], None] | None = None,
) -> EpisodeResult:
    started = utc_now()
    observation, info = env.reset(seed=int(episode_seed))
    if official:
        validate_official_contract(env, info)
    contract = capture_environment_contract(env, info)
    history: list[dict[str, Any]] = []
    shaped_return = 0.0
    page = 0
    status = "aborted"

    while True:
        actions = policy_visible_actions(env, info)
        if not actions:
            raise RuntimeError("the policy-visible legal action set is empty")
        state = visible_state(observation, info)
        if state_callback is not None:
            state_callback(env, observation, info)
        output_fn("")
        output_fn(
            f"Turn {state['turn']} | stars={state['stars']} | SPT={state['spt']} | "
            f"cities={state['city_count']} | visible units={state['visible_unit_count']}"
        )
        for line in visible_map_lines(state):
            output_fn(line)
        page = _print_actions(actions, page, max(1, int(page_size)), output_fn)

        selected_gid: int | None = None
        if selector is not None:
            selected_gid = selector(actions, observation, info, len(history))
        else:
            while selected_gid is None:
                command = input_fn("Choose action index or [n/p page, g <gid>, q abort]: ").strip()
                lowered = command.lower()
                if lowered in {"q", "quit", "exit"}:
                    status = "aborted"
                    return EpisodeResult(
                        status, started, utc_now(), int(episode_seed), shaped_return, len(history),
                        visible_metrics(state), {}, contract, history,
                    )
                if lowered == "n":
                    page = _print_actions(actions, page + 1, max(1, int(page_size)), output_fn)
                    continue
                if lowered == "p":
                    page = _print_actions(actions, page - 1, max(1, int(page_size)), output_fn)
                    continue
                if lowered.startswith("g "):
                    try:
                        candidate = int(lowered.split(maxsplit=1)[1])
                    except ValueError:
                        output_fn("Invalid global ID.")
                        continue
                    if candidate not in {action["global_id"] for action in actions}:
                        output_fn("That global ID is not currently policy-visible/legal.")
                        continue
                    selected_gid = candidate
                    continue
                try:
                    index = int(command)
                    if not 0 <= index < len(actions):
                        raise IndexError(index)
                    selected_gid = int(actions[index]["global_id"])
                except (ValueError, IndexError):
                    output_fn("Invalid action index.")

        if selected_gid is None:
            status = "aborted"
            return EpisodeResult(
                status, started, utc_now(), int(episode_seed), shaped_return, len(history),
                visible_metrics(state), {}, contract, history,
            )
        by_gid = {int(action["global_id"]): action for action in actions}
        if int(selected_gid) not in by_gid:
            raise RuntimeError(f"selector returned non-policy-visible global ID {selected_gid}")
        selected = by_gid[int(selected_gid)]
        observation, reward, terminated, truncated, next_info = env.step(int(selected_gid))
        if int(next_info.get("selected_global_id", -1)) != int(selected_gid):
            raise RuntimeError("wrapper did not execute the selected stable global ID")
        shaped_return += float(reward)
        result_state = visible_state(observation, next_info)
        history.append(
            {
                "step": len(history),
                "global_id": int(selected_gid),
                "type": str(selected["type"]),
                "description": str(selected["description"]),
                "reward": float(reward),
                "resulting_visible_metrics": visible_metrics(result_state),
            }
        )
        info = next_info
        if bool(terminated or truncated):
            status = "completed"
            final_info = {key: _scalar(info.get(key)) for key in SAFE_FINAL_INFO_KEYS if key in info}
            return EpisodeResult(
                status=status,
                started_at_utc=started,
                ended_at_utc=utc_now(),
                episode_seed=int(episode_seed),
                shaped_return=float(shaped_return),
                decision_count=len(history),
                final_visible_metrics=visible_metrics(result_state),
                final_info_metrics=final_info,
                environment_contract=contract,
                action_history=history,
            )


def choose_first_action(
    actions: list[dict[str, Any]], _observation: np.ndarray, _info: dict[str, Any], _step: int
) -> int:
    """Synthetic test selector. Never use its results as human benchmark evidence."""
    return int(actions[0]["global_id"])


def choose_then_abort(
    actions: list[dict[str, Any]], _observation: np.ndarray, _info: dict[str, Any], step: int
) -> int | None:
    return int(actions[0]["global_id"]) if step == 0 else None
