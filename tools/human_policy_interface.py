"""Human-readable presentation of the authoritative PPO-facing Phase 1 interface.

Official benchmark code in this module consumes only the flattened policy
observation, the legal-slot tensors returned in ``info``, and stable global-ID
catalog metadata. It never reads the Java observation, raw action dictionaries,
``_last_obs``, debug info, or a renderer.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import numpy as np

from pol_env.Tribes.py.environment_contract import (
    MAX_OWNED_CITIES,
    SUPPORTED_UNIT_TYPES,
    SUPPORTED_BUILDINGS,
    TECHNOLOGY_ORDER,
    decode_owned_city_slots,
    observation_layout,
)


HUMAN_INTERFACE_VERSION = "v5_human_information_parity"
TERRAIN_SYMBOLS = {0: ".", 1: "~", 2: "D", 3: "M", 4: "V", 5: "C", 6: "T", 7: "?"}
RESOURCE_SYMBOLS = {0: "h", 1: "f", 2: "a", 3: "w", 5: "o", 6: "c", 7: "r"}
SAFE_INFO_MODE = "fast"


# ANSI Escape Sequences
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"

# Standard 16-color ANSI backgrounds
BG_FOG = "\033[40m"          # Black
BG_PLAIN = "\033[49m"        # Default / neutral
BG_FOREST = "\033[42m"       # Green
BG_MOUNTAIN = "\033[100m"    # Dark Gray / Bright Black
BG_WATER = "\033[44m"        # Blue
BG_DEEP_WATER = "\033[44m"   # Blue
BG_VILLAGE = "\033[43m"      # Yellow/Gold
BG_CITY = "\033[46m"         # Cyan

# Foregrounds
FG_UNIT = "\033[1;97m"       # Bold Bright White
FG_CITY = "\033[1;97m"       # Bold Bright White
FG_VILLAGE = "\033[1;30m"    # Bold Black
FG_FOG = "\033[90m"          # Dark Gray
FG_ANIMAL = "\033[1;93m"     # Bold Bright Yellow
FG_FRUIT = "\033[1;91m"      # Bold Bright Red
FG_FISH = "\033[1;96m"       # Bold Bright Cyan
FG_WHALE = "\033[1;97m"      # Bold Bright White
FG_ORE = "\033[1;93m"        # Bold Bright Yellow
FG_CROPS = "\033[1;92m"      # Bold Bright Green
FG_RUIN = "\033[1;95m"       # Bold Bright Magenta
FG_MOUNTAIN = "\033[1;97m"   # Bold White
FG_FOREST = "\033[1;33m"     # Bold Brown/Tan on Green background
FG_WATER = "\033[96m"        # Cyan
FG_PLAIN = "\033[90m"        # Dim Gray
FG_TERRITORY = "\033[36m"    # Cyan


DIRECTION_UNICODE = {
    (0, -1): "↑ ",
    (0, 1): "↓ ",
    (-1, 0): "← ",
    (1, 0): "→ ",
    (-1, -1): "↖",
    (1, -1): "↗",
    (-1, 1): "↙",
    (1, 1): "↘",
}

DIRECTION_ASCII = {
    (0, -1): "N ",
    (0, 1): "S ",
    (-1, 0): "W ",
    (1, 0): "E ",
    (-1, -1): "NW",
    (1, -1): "NE",
    (-1, 1): "SW",
    (1, 1): "SE",
}


SECTION_DEFINITIONS: list[tuple[str, tuple[str, ...]]] = [
    ("MOVEMENT", ("MOVE",)),
    ("CAPTURE", ("CAPTURE",)),
    ("ECONOMY / RESOURCES", ("RESOURCE_GATHERING", "CLEAR_FOREST", "GROW_FOREST")),
    ("BUILD", ("BUILD",)),
    ("LEVEL UP", ("LEVEL_UP",)),
    ("RESEARCH", ("RESEARCH_TECH",)),
    ("TRAINING", ("TRAIN", "SPAWN")),
    ("OTHER", ("EXAMINE",)),
    ("TURN", ("END_TURN",)),
]


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


def supports_ansi_color(stream: Any = None) -> bool:
    """Detect whether ANSI color escapes should be used."""
    if os.environ.get("POLYVISION_FORCE_COLOR") in ("1", "true", "yes", "on"):
        return True
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("POLYVISION_NO_COLOR") in ("1", "true", "yes", "on"):
        return False
    if stream is None:
        stream = sys.stdout
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return True


def supports_unicode(stream: Any = None) -> bool:
    """Detect whether Unicode directional arrows and glyphs are supported."""
    if os.environ.get("POLYVISION_ASCII_ONLY") in ("1", "true", "yes", "on"):
        return False
    if os.environ.get("POLYVISION_FORCE_UNICODE") in ("1", "true", "yes", "on"):
        return True
    if stream is None:
        stream = sys.stdout
    encoding = getattr(stream, "encoding", None) or sys.getdefaultencoding() or ""
    return "utf" in encoding.lower()


def move_direction(src: tuple[int, int], dst: tuple[int, int], unicode_arrow: bool = True) -> str:
    """Derive directional arrow / compass string from source and destination coordinates."""
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    mapping = DIRECTION_UNICODE if unicode_arrow else DIRECTION_ASCII
    return mapping.get((dx, dy), "? ")


def extract_visible_units(
    state: dict[str, Any],
    actions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract visible units deterministically in (x, y) spatial tile order from PPO observation."""
    width = int(state["width"])
    height = int(state["height"])
    unit_block = state.get("unit_types_block")
    unit_ids = state.get("unit_ids")

    units = []
    unit_num = 1
    for x in range(width):
        for y in range(height):
            tile = x * height + y
            if unit_block is not None and unit_block.shape[0] == len(SUPPORTED_UNIT_TYPES):
                type_idx = int(np.argmax(unit_block[:, tile]))
                if unit_block[type_idx, tile] > 0.5:
                    u_type = SUPPORTED_UNIT_TYPES[type_idx].title()
                    units.append({
                        "number": unit_num,
                        "pos": (x, y),
                        "type": u_type,
                        "tile": tile,
                    })
                    unit_num += 1
            elif unit_ids is not None and int(unit_ids[tile]) > 0:
                units.append({
                    "number": unit_num,
                    "pos": (x, y),
                    "type": "Warrior",
                    "tile": tile,
                })
                unit_num += 1
    return units


def action_feature_annotations(feature_vector: np.ndarray | Iterable[float], action_type: str | None = None) -> list[str]:
    """Turn the model-visible 47-feature row into concise, human-readable annotations."""
    feat = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
    if feat.shape != (47,):
        raise RuntimeError(f"action feature row shape mismatch: expected (47,), got {feat.shape}")

    lines: list[str] = []
    if len(feat) > 42:
        cost = int(round(float(feat[42]) * 50.0))
        if cost > 0:
            lines.append(f"{cost}*")

    a_type = (action_type or "").upper()
    is_move = a_type == "MOVE" or bool(feat[0] >= 0.5)

    if is_move:
        revealed_norm = float(feat[1])
        revealed = int(round(revealed_norm * 12.0))
        if revealed_norm >= 1.0:
            lines.append("reveal +12+")
        elif not bool(feat[4] >= 0.5) and revealed > 0:
            lines.append(f"reveal +{revealed}")

        adj_fog_norm = float(feat[2])
        adj_fog = int(round(adj_fog_norm * 8.0))
        if adj_fog_norm >= 1.0:
            lines.append("adjacent fog 8+")
        elif adj_fog > 0:
            lines.append(f"adjacent fog {adj_fog}")

        adj_delta = int(round(float(feat[3]) * 8.0))
        if adj_delta > 0:
            lines.append(f"fog change +{adj_delta}")
        elif adj_delta < 0:
            lines.append(f"fog change {adj_delta}")

        if bool(feat[5] >= 0.5):
            lines.append("target: uncaptured village")
        elif bool(feat[6] >= 0.5):
            dist_delta = float(feat[7])
            if dist_delta > 0.001:
                lines.append("closer to village")
            elif dist_delta < -0.001:
                lines.append("further from village")

        if bool(feat[8] >= 0.5):
            lines.append("immediate backtrack")

        if bool(feat[9] >= 0.5):
            lines.append("city bounds")

        capital_delta = float(feat[10])
        if capital_delta > 0.001:
            lines.append("away from capital")
        elif capital_delta < -0.001:
            lines.append("toward capital")

    elif a_type in ("RESOURCE_GATHERING", "BUILD", "CLEAR_FOREST", "GROW_FOREST") or bool(
        feat[16] >= 0.5 or feat[18] >= 0.5 or feat[19] >= 0.5 or feat[20] >= 0.5
    ):
        pop_delta = int(round(float(feat[36]) * 2.0))
        if pop_delta != 0:
            lines.append(f"pop {pop_delta:+d}")

        spt_delta = int(round(float(feat[37]) * 5.0))
        if spt_delta != 0:
            lines.append(f"SPT {spt_delta:+d}")

        progress_before = int(round(float(feat[40]) * 100.0))
        if progress_before > 0 or bool(feat[41] >= 0.5):
            lines.append(f"city progress {progress_before}%")

        if bool(feat[41] >= 0.5):
            lines.append("city upgrade ready")

        if bool(feat[38] >= 0.5):
            lines.append("makes level-up ready")

        if bool(feat[26] >= 0.5):
            lines.append("resource: ANIMAL")
        elif bool(feat[27] >= 0.5):
            lines.append("resource: FRUIT")
        elif bool(feat[28] >= 0.5):
            lines.append("resource: FISH")
        elif bool(feat[29] >= 0.5):
            lines.append("resource: CROPS")
        elif bool(feat[30] >= 0.5):
            lines.append("resource: ORE")

        if bool(feat[32] >= 0.5):
            lines.append("building: LUMBER_HUT")
        elif bool(feat[33] >= 0.5):
            lines.append("building: SAWMILL")

    elif a_type == "LEVEL_UP" or bool(feat[17] >= 0.5):
        pop_delta = int(round(float(feat[36]) * 2.0))
        if pop_delta != 0:
            lines.append(f"pop {pop_delta:+d}")

        spt_delta = int(round(float(feat[37]) * 5.0))
        if spt_delta != 0:
            lines.append(f"SPT {spt_delta:+d}")

        if bool(feat[35] >= 0.5):
            lines.append("choice: WORKSHOP")

        if bool(feat[39] >= 0.5):
            lines.append("level-up claim")

    elif a_type == "RESEARCH_TECH" or bool(feat[15] >= 0.5):
        if bool(feat[23] >= 0.5):
            lines.append("tech: ORGANIZATION")
        elif bool(feat[24] >= 0.5):
            lines.append("tech: FORESTRY")

    elif a_type == "CAPTURE" or bool(feat[13] >= 0.5):
        spt_delta = int(round(float(feat[37]) * 5.0))
        if spt_delta != 0:
            lines.append(f"SPT {spt_delta:+d}")

    elif a_type in ("TRAIN", "SPAWN") or bool(feat[14] >= 0.5):
        if bool(feat[11] >= 0.5):
            lines.append("unit: warrior")

    return lines


def policy_visible_ids(info: dict[str, Any]) -> list[int]:
    ids = np.asarray(info.get("legal_global_ids_padded", []), dtype=np.int64).reshape(-1)
    valid = np.asarray(info.get("legal_action_valid_mask", []), dtype=bool).reshape(-1)
    if ids.shape != valid.shape:
        raise RuntimeError(f"legal-slot shape mismatch: ids={ids.shape}, mask={valid.shape}")
    valid_indices = np.nonzero(valid)[0]
    selected = [int(ids[slot_idx]) for slot_idx in valid_indices]
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
    wrapper = getattr(env, "unwrapped", env)
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
    ids = np.asarray(info.get("legal_global_ids_padded", []), dtype=np.int64).reshape(-1)
    valid = np.asarray(info.get("legal_action_valid_mask", []), dtype=bool).reshape(-1)
    if ids.shape != valid.shape:
        raise RuntimeError(f"legal-slot shape mismatch: ids={ids.shape}, mask={valid.shape}")

    features = info.get("legal_action_features_padded", None)
    if features is None:
        raise RuntimeError("missing required info field 'legal_action_features_padded'")
    features_arr = np.asarray(features, dtype=np.float32)
    if features_arr.ndim != 2:
        raise RuntimeError(f"expected 2D legal_action_features_padded, got shape={features_arr.shape}")
    if features_arr.shape[0] != ids.shape[0]:
        raise RuntimeError(
            f"legal_action_features_padded row count {features_arr.shape[0]} != slot count {ids.shape[0]}"
        )

    wrapper = getattr(env, "unwrapped", env)
    expected_dim = getattr(wrapper, "ACTION_FEATURE_DIM", None)
    if expected_dim is None:
        expected_dim = int(info.get("legal_action_feature_dim", 42))
    else:
        expected_dim = int(expected_dim)

    if features_arr.shape[1] != expected_dim:
        raise RuntimeError(
            f"legal_action_features_padded feature dim {features_arr.shape[1]} != expected {expected_dim}"
        )

    expected_names = getattr(wrapper, "LEGAL_ACTION_FEATURE_NAMES", ())
    if expected_names and len(expected_names) != expected_dim:
        raise RuntimeError(
            f"LEGAL_ACTION_FEATURE_NAMES length {len(expected_names)} != feature dim {expected_dim}"
        )

    valid_slot_indices = np.nonzero(valid)[0]
    selected_ids = [int(ids[slot_idx]) for slot_idx in valid_slot_indices]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("policy-visible legal global IDs contain duplicates")

    declared = int(info.get("legal_action_count", len(selected_ids)))
    if declared != len(selected_ids):
        raise RuntimeError(f"legal_action_count={declared} but visible slot count={len(selected_ids)}")

    catalog = getattr(wrapper, "_catalog", None)
    n_tiles = int(catalog.n_tiles) if catalog is not None else 121
    move_offset = int(catalog.offsets["MOVE"]) if catalog is not None and "MOVE" in catalog.offsets else None

    actions: list[dict[str, Any]] = []
    for menu_slot, slot_idx in enumerate(valid_slot_indices):
        gid = int(ids[slot_idx])
        action_type, description = decode_global_action(env, gid)
        feat_row = features_arr[slot_idx]
        annotations = action_feature_annotations(feat_row, action_type)

        src_xy = None
        dst_xy = None
        if action_type == "MOVE" and catalog is not None and move_offset is not None:
            local = gid - move_offset
            src_t, dst_t = divmod(local, n_tiles)
            src_xy = _tile_xy(catalog, src_t)
            dst_xy = _tile_xy(catalog, dst_t)

        actions.append(
            {
                "slot": int(menu_slot),
                "padded_slot": int(slot_idx),
                "global_id": int(gid),
                "type": str(action_type),
                "description": str(description),
                "features": feat_row,
                "annotations": annotations,
                "src_xy": src_xy,
                "dst_xy": dst_xy,
            }
        )
    return actions


def visible_state(observation: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
    width = int(info["map_width"])
    height = int(info["map_height"])
    layout = observation_layout(width, height)
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    if obs.size != layout.expected_obs_dim:
        raise RuntimeError(f"observation has {obs.size} values; expected {layout.expected_obs_dim}")
    n = layout.n_tiles

    terrain = np.rint(obs[layout.terrain_start : layout.terrain_end]).astype(np.int16)
    unit_types_block = obs[layout.unit_types_start : layout.unit_types_end].reshape(len(SUPPORTED_UNIT_TYPES), n)
    city_territory_block = obs[layout.city_territory_start : layout.city_territory_end].reshape(layout.city_slots, n)
    roads = np.rint(obs[layout.road_start : layout.road_end]).astype(np.int16)
    buildings_block = obs[layout.buildings_start : layout.buildings_end].reshape(len(SUPPORTED_BUILDINGS), n)
    resources = np.rint(obs[layout.resource_start : layout.resource_end] * 8.0 - 1.0).astype(np.int16)

    legacy = layout.legacy_scalar_start
    scalar = layout.economy_scalar_start
    tech_start = layout.tech_vector_start

    city_block = obs[layout.city_block_start : layout.city_block_end]
    owned_cities = decode_owned_city_slots(city_block, width=width, height=height, max_cities=layout.city_slots)

    unit_present = np.any(unit_types_block > 0.5, axis=0)
    visible_unit_count = int(np.sum(unit_present))

    researched_techs = [
        tech_name for i, tech_name in enumerate(TECHNOLOGY_ORDER)
        if obs[tech_start + i] >= 0.5
    ]

    return {
        "width": width,
        "height": height,
        "terrain": terrain,
        "unit_types_block": unit_types_block,
        "city_territory_block": city_territory_block,
        "roads": roads,
        "buildings_block": buildings_block,
        "resources": resources,
        "owned_cities": owned_cities,
        "researched_techs": researched_techs,
        "stars": int(round(float(obs[legacy]))),
        "score": int(round(float(obs[legacy + 1]))),
        "city_count": int(round(float(obs[legacy + 2]))),
        "kills": int(round(float(obs[legacy + 3]))),
        "turn": int(round(float(obs[scalar + 2]) * 10.0)),
        "spt": int(round(float(obs[scalar + 1]) * 30.0)),
        "turns_remaining_after_current": float(obs[scalar + 3]),
        "turns_remaining_including_current": float(obs[scalar + 4]),
        "tech_organization": "ORGANIZATION" in researched_techs,
        "tech_forestry": "FORESTRY" in researched_techs,
        "tech_count": len(researched_techs),
        "avg_city_level": float(obs[scalar + 6]) * 5.0,
        "max_city_level": float(obs[scalar + 7]) * 5.0,
        "mean_upgrade_progress": float(obs[scalar + 8]),
        "max_upgrade_progress": float(obs[scalar + 9]),
        "upgrade_ready_frac": float(obs[scalar + 10]),
        "any_level_up_available": bool(obs[scalar + 11] >= 0.5),
        "visible_unit_count": visible_unit_count,
        "unit_ids": unit_present.astype(np.int64),
        "city_ids": np.zeros(n, dtype=np.int64),
    }


def _legacy_compact_map_lines(state: dict[str, Any]) -> list[str]:
    width, height = int(state["width"]), int(state["height"])
    terrain = state["terrain"]
    units = state["unit_ids"]
    resources = state["resources"]
    lines = ["Visible map: [Terrain Unit Resource] (? = fog)"]
    header_cells = [f"{x:^5}" for x in range(width)]
    lines.append("     " + " ".join(header_cells))
    lines.append("    +" + "-----+" * width)
    for y in range(height):
        row_cells = []
        for x in range(width):
            tile = x * height + y
            terrain_id = int(terrain[tile])
            if terrain_id == 7:
                row_cells.append(" ??? ")
                continue
            terr = TERRAIN_SYMBOLS.get(terrain_id, str(terrain_id)[-1])
            unit = "U" if int(units[tile]) > 0 else "."
            res_id = int(resources[tile])
            res = RESOURCE_SYMBOLS.get(res_id, ".") if res_id >= 0 else "."
            row_cells.append(f" {terr}{unit}{res} ")
        lines.append(f"{y:>3} |" + "|".join(row_cells) + "|")
        lines.append("    +" + "-----+" * width)
    lines.append("Terrain: .=plain ~=water D=deep M=mountain V=village C=city T=forest")
    lines.append("Occupants: U=visible unit; Resources: a=animal f=fruit h=fish w=whale o=ore c=crops r=ruin")
    return lines


def visible_map_lines(
    state: dict[str, Any],
    actions: list[dict[str, Any]] | None = None,
    use_ansi: bool | None = None,
    unicode_chars: bool | None = None,
    mode: str = "tactical",
) -> list[str]:
    """Generate human-readable map lines using colored tactical display or monochrome fallback."""
    if mode == "compact_symbolic":
        return _legacy_compact_map_lines(state)

    if use_ansi is None:
        use_ansi = supports_ansi_color()
    if unicode_chars is None:
        unicode_chars = supports_unicode()

    width, height = int(state["width"]), int(state["height"])
    terrain = state["terrain"]
    city_ids = state["city_ids"]
    resources = state["resources"]

    unit_list = extract_visible_units(state, actions)
    unit_map = {u["pos"]: u["number"] for u in unit_list}

    lines = [f"Tactical Map ({'ANSI Color' if use_ansi else 'Monochrome Fallback'}):"]
    header_cells = [f"{x:>3}" for x in range(width)]
    lines.append("     " + " ".join(header_cells))
    lines.append("    +" + "---+" * width)

    territory_dot = "·" if unicode_chars else "."

    for y in range(height):
        row_cells = []
        for x in range(width):
            tile = x * height + y
            terrain_id = int(terrain[tile])
            res_id = int(resources[tile])
            u_num = unit_map.get((x, y))
            c_id = int(city_ids[tile])

            if terrain_id == 7:  # Fog
                if use_ansi:
                    row_cells.append(f"{FG_FOG}{BG_FOG} ? {ANSI_RESET}")
                else:
                    row_cells.append(" ? ")
                continue

            # Visible Unit
            if u_num is not None:
                u_str = f"{u_num}" if u_num < 10 else f"{u_num % 10}"
                if use_ansi:
                    if terrain_id == 5:
                        bg = BG_CITY
                    elif terrain_id == 4:
                        bg = BG_VILLAGE
                    elif terrain_id == 6:
                        bg = BG_FOREST
                    elif terrain_id == 3:
                        bg = BG_MOUNTAIN
                    elif terrain_id in (1, 2):
                        bg = BG_WATER
                    else:
                        bg = BG_PLAIN
                    row_cells.append(f"{FG_UNIT}{bg} {u_str} {ANSI_RESET}")
                else:
                    if terrain_id == 5:
                        row_cells.append(f"C{u_str} ")
                    elif terrain_id == 4:
                        row_cells.append(f"V{u_str} ")
                    elif terrain_id == 6:
                        row_cells.append(f"T{u_str} ")
                    elif terrain_id == 3:
                        row_cells.append(f"M{u_str} ")
                    elif terrain_id in (1, 2):
                        row_cells.append(f"~{u_str} ")
                    elif res_id == 2:
                        row_cells.append(f"A{u_str} ")
                    elif res_id == 1:
                        row_cells.append(f"F{u_str} ")
                    else:
                        row_cells.append(f" {u_str} ")
                continue

            # City Center
            if terrain_id == 5:
                if use_ansi:
                    row_cells.append(f"{FG_CITY}{BG_CITY} C {ANSI_RESET}")
                else:
                    row_cells.append(" C ")
                continue

            # Village
            if terrain_id == 4:
                if use_ansi:
                    row_cells.append(f"{FG_VILLAGE}{BG_VILLAGE} V {ANSI_RESET}")
                else:
                    row_cells.append(" V ")
                continue

            # Resources
            if res_id == 2:  # Animal
                if use_ansi:
                    bg = BG_FOREST if terrain_id == 6 else BG_PLAIN
                    row_cells.append(f"{FG_ANIMAL}{bg} A {ANSI_RESET}")
                else:
                    row_cells.append("Ta " if terrain_id == 6 else " A ")
                continue
            elif res_id == 1:  # Fruit
                if use_ansi:
                    bg = BG_FOREST if terrain_id == 6 else BG_PLAIN
                    row_cells.append(f"{FG_FRUIT}{bg} F {ANSI_RESET}")
                else:
                    row_cells.append("Tf " if terrain_id == 6 else " F ")
                continue
            elif res_id == 0:  # Fish
                if use_ansi:
                    row_cells.append(f"{FG_FISH}{BG_WATER} H {ANSI_RESET}")
                else:
                    row_cells.append(" H ")
                continue
            elif res_id == 3:  # Whale
                if use_ansi:
                    row_cells.append(f"{FG_WHALE}{BG_WATER} W {ANSI_RESET}")
                else:
                    row_cells.append(" W ")
                continue
            elif res_id == 5:  # Ore
                if use_ansi:
                    bg = BG_MOUNTAIN if terrain_id == 3 else BG_PLAIN
                    row_cells.append(f"{FG_ORE}{bg} O {ANSI_RESET}")
                else:
                    row_cells.append("Mo " if terrain_id == 3 else " O ")
                continue
            elif res_id == 6:  # Crops
                if use_ansi:
                    row_cells.append(f"{FG_CROPS}{BG_PLAIN} P {ANSI_RESET}")
                else:
                    row_cells.append(" P ")
                continue
            elif res_id == 7:  # Ruin
                if use_ansi:
                    bg = BG_FOREST if terrain_id == 6 else BG_PLAIN
                    row_cells.append(f"{FG_RUIN}{bg} R {ANSI_RESET}")
                else:
                    row_cells.append("Tr " if terrain_id == 6 else " R ")
                continue

            # Empty Terrain
            if terrain_id == 6:  # Forest
                if use_ansi:
                    row_cells.append(f"{FG_FOREST}{BG_FOREST} T {ANSI_RESET}")
                else:
                    row_cells.append(" T ")
            elif terrain_id == 3:  # Mountain
                if use_ansi:
                    row_cells.append(f"{FG_MOUNTAIN}{BG_MOUNTAIN} M {ANSI_RESET}")
                else:
                    row_cells.append(" M ")
            elif terrain_id == 1:  # Shallow Water
                if use_ansi:
                    row_cells.append(f"{FG_WATER}{BG_WATER} ~ {ANSI_RESET}")
                else:
                    row_cells.append(" ~ ")
            elif terrain_id == 2:  # Deep Water
                if use_ansi:
                    row_cells.append(f"{FG_WATER}{BG_DEEP_WATER} D {ANSI_RESET}")
                else:
                    row_cells.append(" D ")
            else:  # Plain (0)
                if c_id > 0:  # City territory
                    if use_ansi:
                        row_cells.append(f"{FG_TERRITORY}{BG_PLAIN} {territory_dot} {ANSI_RESET}")
                    else:
                        row_cells.append(" . ")
                else:
                    if use_ansi:
                        row_cells.append(f"{FG_PLAIN}{BG_PLAIN} . {ANSI_RESET}")
                    else:
                        row_cells.append(" . ")

        lines.append(f"{y:>3} |" + "|".join(row_cells) + "|")
        lines.append("    +" + "---+" * width)

    if use_ansi:
        lines.append("Terrain (bg): plain=neutral forest=green water=blue mountain=gray city=cyan village=gold fog=dark")
        lines.append("Glyphs: 1,2..=unit T=forest C=city V=village A=animal F=fruit H=fish W=whale O=ore P=crops R=ruin ?=fog")
    else:
        lines.append("Terrain: .=plain T=forest M=mountain ~=water D=deep water C=city V=village ?=fog")
        lines.append("Glyphs: 1,2..=unit A=animal F=fruit H=fish W=whale O=ore P=crops R=ruin")

    if unit_list:
        unit_descs = [f"[{u['number']}] {u['type']} at {u['pos']}" for u in unit_list]
        lines.append("Units: " + " | ".join(unit_descs))
    else:
        lines.append("Units: none")

    return lines


def visible_metrics(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: state[key]
        for key in (
            "turn",
            "stars",
            "spt",
            "city_count",
            "visible_unit_count",
            "tech_count",
            "tech_organization",
            "tech_forestry",
            "avg_city_level",
            "max_city_level",
            "mean_upgrade_progress",
            "max_upgrade_progress",
            "upgrade_ready_frac",
            "any_level_up_available",
        )
        if key in state
    }


def capture_environment_contract(env: Any, info: dict[str, Any]) -> dict[str, Any]:
    wrapper = getattr(env, "unwrapped", env)
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
    wrapper = getattr(env, "unwrapped", env)
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
    policy_visible_actions(env, info)


def _print_actions(
    actions: list[dict[str, Any]],
    page: int,
    page_size: int,
    output: Callable[[str], None],
    state: dict[str, Any] | None = None,
    unicode_arrows: bool | None = None,
) -> int:
    pages = max(1, (len(actions) + page_size - 1) // page_size)
    page = max(0, min(page, pages - 1))
    start, end = page * page_size, min(len(actions), (page + 1) * page_size)
    output(f"--- Legal Policy Actions ({start} to {end - 1} of {len(actions) - 1}; Page {page + 1}/{pages}) ---")

    if unicode_arrows is None:
        unicode_arrows = supports_unicode()

    unit_list = extract_visible_units(state, actions) if state is not None else []

    groups: list[tuple[str, str, list[int]]] = []

    move_indices = [idx for idx, a in enumerate(actions) if a["type"] == "MOVE"]
    if move_indices:
        moves_by_src: dict[tuple[int, int], list[int]] = {}
        for idx in move_indices:
            src = actions[idx].get("src_xy")
            if src is not None:
                moves_by_src.setdefault(src, []).append(idx)

        handled_moves: set[int] = set()
        for u in unit_list:
            pos = u["pos"]
            if pos in moves_by_src:
                u_moves = moves_by_src[pos]
                groups.append((
                    f"MOVE_UNIT_{u['number']}",
                    f"MOVES - UNIT {u['number']} at {pos}",
                    u_moves,
                ))
                handled_moves.update(u_moves)

        unhandled = [idx for idx in move_indices if idx not in handled_moves]
        if unhandled:
            groups.append(("MOVE_OTHER", "MOVES - OTHER", unhandled))

    for sec_name, sec_types in SECTION_DEFINITIONS:
        if sec_name == "MOVEMENT":
            continue
        sec_indices = [idx for idx, a in enumerate(actions) if a["type"] in sec_types]
        if sec_indices:
            groups.append((sec_name, sec_name, sec_indices))

    classified_indices = {idx for _, _, idcs in groups for idx in idcs}
    remaining = [idx for idx in range(len(actions)) if idx not in classified_indices]
    if remaining:
        groups.append(("OTHER", "OTHER", remaining))

    flattened_slots: list[tuple[str, int]] = []
    for _gkey, gheader, idcs in groups:
        for idx in idcs:
            flattened_slots.append((gheader, idx))

    page_items = flattened_slots[start:end]

    current_header = None
    for gheader, action_idx in page_items:
        action = actions[action_idx]
        if gheader != current_header:
            current_header = gheader
            output(f"\n[{gheader}]")

        a_type = action["type"]
        if a_type == "MOVE":
            src = action.get("src_xy")
            dst = action.get("dst_xy")
            if src is not None and dst is not None:
                arrow = move_direction(src, dst, unicode_arrow=unicode_arrows)
                ann_str = " | ".join(action.get("annotations", []))
                if ann_str:
                    output(f"  [{action['slot']:>3}] {arrow} {dst}   {ann_str}  (gid={action['global_id']})")
                else:
                    output(f"  [{action['slot']:>3}] {arrow} {dst}  (gid={action['global_id']})")
            else:
                output(f"  [{action['slot']:>3}] {action['description']}  (gid={action['global_id']})")
                ann_str = " | ".join(action.get("annotations", []))
                if ann_str:
                    output(f"        * {ann_str}")
        else:
            output(f"  [{action['slot']:>3}] {action['description']}  (gid={action['global_id']})")
            ann_str = " | ".join(action.get("annotations", []))
            if ann_str:
                output(f"        * {ann_str}")

    output("")
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
            f"=== Turn {state['turn']}/10 | Stars {state['stars']} | SPT {state['spt']} | "
            f"Cities {state['city_count']} | Visible units {state['visible_unit_count']} ==="
        )
        for line in visible_map_lines(state, actions=actions):
            output_fn(line)
        output_fn("")
        owned_cities = state.get("owned_cities", [])
        if owned_cities:
            output_fn(f"Owned Cities ({len(owned_cities)}/{MAX_OWNED_CITIES}):")
            for c in owned_cities:
                slot_num = c["slot"] + 1
                cap_suffix = " | Capital" if c.get("is_capital") else ""
                output_fn(
                    f"  City #{slot_num} @ ({c['x']}, {c['y']}): Level {c['level']} | "
                    f"Pop {c['population']}/{c['population_need']} | "
                    f"Production +{c['production']} SPT | "
                    f"Units {c['supported_unit_count']}/{c['unit_capacity']}{cap_suffix}"
                )
        output_fn("Economy & Aggregates:")
        output_fn(
            f"  Avg city level:        {state['avg_city_level']:.2f}  | Max city level:       {state['max_city_level']:.0f}\n"
            f"  Mean upgrade progress: {int(round(state['mean_upgrade_progress'] * 100))}%  | Max upgrade progress: {int(round(state['max_upgrade_progress'] * 100))}%\n"
            f"  Cities upgrade-ready:  {int(round(state['upgrade_ready_frac'] * 100))}%  | Level-up available:   {'yes' if state['any_level_up_available'] else 'no'}\n"
            f"  Techs researched:      {state['tech_count']}   (Organization: {'yes' if state['tech_organization'] else 'no'}, Forestry: {'yes' if state['tech_forestry'] else 'no'})"
        )
        output_fn("")
        page = _print_actions(actions, page, max(1, int(page_size)), output_fn, state=state)

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
                    page = _print_actions(actions, page + 1, max(1, int(page_size)), output_fn, state=state)
                    continue
                if lowered == "p":
                    page = _print_actions(actions, page - 1, max(1, int(page_size)), output_fn, state=state)
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
