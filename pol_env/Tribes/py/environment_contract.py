"""Dimension and checkpoint contracts shared by the Tribes environment tooling."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterable, Mapping


class MapGeometryError(RuntimeError):
    """Raised when a level cannot satisfy an environment's fixed geometry."""


class CheckpointCompatibilityError(RuntimeError):
    """Raised before loading model tensors when interface metadata is incompatible."""


class ObservationContractError(RuntimeError):
    """Raised when an observation violates the environment interface contract."""


PHASE1_ENVIRONMENT_VERSION = "v5_human_information_parity"

ACTION_STAR_COST_SCALE = 50.0

MAX_OWNED_CITIES = 9
CITY_SLOT_FEATURE_DIM = 10
CITY_SLOT_FEATURE_NAMES = (
    "city_present",
    "city_x",
    "city_y",
    "city_level",
    "city_population",
    "city_population_need",
    "city_production",
    "city_supported_unit_count",
    "city_unit_capacity",
    "city_is_capital",
)
CITY_BLOCK_DIM = MAX_OWNED_CITIES * CITY_SLOT_FEATURE_DIM  # 90

SUPPORTED_UNIT_TYPES = (
    "WARRIOR",
    "RIDER",
    "DEFENDER",
    "SWORDMAN",
    "ARCHER",
    "CATAPULT",
    "KNIGHT",
    "MIND_BENDER",
    "BOAT",
    "SHIP",
    "BATTLESHIP",
    "SUPERUNIT",
)
NUM_UNIT_TYPE_CHANNELS = len(SUPPORTED_UNIT_TYPES)  # 12

SUPPORTED_BUILDINGS = (
    "PORT",
    "MINE",
    "FORGE",
    "FARM",
    "WINDMILL",
    "CUSTOMS_HOUSE",
    "LUMBER_HUT",
    "SAWMILL",
    "TEMPLE",
    "WATER_TEMPLE",
    "FOREST_TEMPLE",
    "MOUNTAIN_TEMPLE",
    "ALTAR_OF_PEACE",
    "EMPERORS_TOMB",
    "EYE_OF_GOD",
    "GATE_OF_POWER",
    "GRAND_BAZAR",
    "PARK_OF_FORTUNE",
    "TOWER_OF_WISDOM",
)
NUM_BUILDING_CHANNELS = len(SUPPORTED_BUILDINGS)  # 19

TECHNOLOGY_ORDER = (
    "CLIMBING",
    "FISHING",
    "HUNTING",
    "ORGANIZATION",
    "RIDING",
    "ARCHERY",
    "FARMING",
    "FORESTRY",
    "FREE_SPIRIT",
    "MEDITATION",
    "MINING",
    "ROADS",
    "SAILING",
    "SHIELDS",
    "WHALING",
    "AQUATISM",
    "CHIVALRY",
    "CONSTRUCTION",
    "MATHEMATICS",
    "NAVIGATION",
    "SMITHERY",
    "SPIRITUALISM",
    "TRADE",
    "PHILOSOPHY",
)
NUM_TECHNOLOGIES = len(TECHNOLOGY_ORDER)  # 24

NUM_CITY_TERRITORY_CHANNELS = MAX_OWNED_CITIES  # 9
NUM_UNIT_HOME_CITY_CHANNELS = MAX_OWNED_CITIES  # 9
LEGACY_SCALAR_DIM = 6
ECONOMY_SCALAR_DIM = 12


@dataclass(frozen=True)
class OwnedCityState:
    x: int
    y: int
    level: int
    population: int
    population_need: int
    production: int
    supported_unit_count: int
    unit_capacity: int
    is_capital: bool


def extract_owned_cities(
    obs_dict: Mapping,
    tribe_id: int,
) -> list[OwnedCityState]:
    """Extract canonical owned-city states from fog-respecting POV observation dictionary.

    Cities are deterministically sorted by spatial coordinates (x, y) ascending.
    """
    controlled_tribe_id = int(tribe_id)
    cities_map = obs_dict.get("city", {})
    if not isinstance(cities_map, dict):
        return []

    collected: list[OwnedCityState] = []
    for _key, city_data in cities_map.items():
        if not isinstance(city_data, dict):
            continue
        try:
            city_tribe_id = int(city_data.get("tribeID", -1))
        except (TypeError, ValueError):
            continue
        if city_tribe_id != controlled_tribe_id:
            continue

        try:
            x = int(city_data["x"])
            y = int(city_data["y"])
            level = int(city_data["level"])
            population = int(city_data["population"])
            population_need = int(city_data["population_need"])
            production = int(city_data["production"])
            units_list = city_data.get("units", [])
            supported_unit_count = len(units_list) if isinstance(units_list, (list, tuple)) else 0
            unit_capacity = level + 1
            is_capital = bool(city_data.get("isCapital", False))
        except (KeyError, TypeError, ValueError) as exc:
            raise ObservationContractError(f"Malformed city payload in observation: {city_data}") from exc

        collected.append(
            OwnedCityState(
                x=x,
                y=y,
                level=level,
                population=population,
                population_need=population_need,
                production=production,
                supported_unit_count=supported_unit_count,
                unit_capacity=unit_capacity,
                is_capital=is_capital,
            )
        )

    # Sort strictly by (x, y) ascending for deterministic, actor-ID-independent slot mapping
    collected.sort(key=lambda c: (c.x, c.y))
    return collected


def encode_owned_city_slots(
    owned_cities: Sequence[OwnedCityState],
    *,
    width: int,
    height: int,
    max_cities: int = MAX_OWNED_CITIES,
) -> list[float]:
    """Encode owned cities into a fixed-size normalized flat float vector of length max_cities * CITY_SLOT_FEATURE_DIM."""
    if len(owned_cities) > max_cities:
        raise ObservationContractError(
            f"Observed {len(owned_cities)} owned cities, exceeding maximum slot capacity of {max_cities}"
        )

    max_x = float(max(1, width - 1))
    max_y = float(max(1, height - 1))
    encoded: list[float] = []

    for i in range(max_cities):
        if i < len(owned_cities):
            c = owned_cities[i]
            encoded.extend(
                [
                    1.0,  # city_present
                    float(c.x) / max_x,  # city_x
                    float(c.y) / max_y,  # city_y
                    float(c.level),  # city_level
                    float(c.population),  # city_population
                    float(c.population_need),  # city_population_need
                    float(c.production),  # city_production
                    float(c.supported_unit_count),  # city_supported_unit_count
                    float(c.unit_capacity),  # city_unit_capacity
                    1.0 if c.is_capital else 0.0,  # city_is_capital
                ]
            )
        else:
            encoded.extend([0.0] * CITY_SLOT_FEATURE_DIM)

    return encoded


def decode_owned_city_slots(
    city_block: Sequence[float],
    *,
    width: int,
    height: int,
    max_cities: int = MAX_OWNED_CITIES,
) -> list[dict]:
    """Decode a fixed-size city slot tensor back into human-interpretable city dictionaries."""
    expected_dim = max_cities * CITY_SLOT_FEATURE_DIM
    if len(city_block) != expected_dim:
        raise ValueError(f"Expected city block of length {expected_dim}, got {len(city_block)}")

    max_x = float(max(1, width - 1))
    max_y = float(max(1, height - 1))
    cities: list[dict] = []

    for i in range(max_cities):
        offset = i * CITY_SLOT_FEATURE_DIM
        block = city_block
        present = bool(block[offset + 0] >= 0.5)
        if not present:
            continue
        x = int(round(float(block[offset + 1]) * max_x))
        y = int(round(float(block[offset + 2]) * max_y))
        level = int(round(float(block[offset + 3])))
        population = int(round(float(block[offset + 4])))
        population_need = int(round(float(block[offset + 5])))
        production = int(round(float(block[offset + 6])))
        supported_unit_count = int(round(float(block[offset + 7])))
        unit_capacity = int(round(float(block[offset + 8])))
        is_capital = bool(block[offset + 9] >= 0.5)
        cities.append(
            {
                "slot": i,
                "x": x,
                "y": y,
                "level": level,
                "population": population,
                "population_need": population_need,
                "production": production,
                "supported_unit_count": supported_unit_count,
                "unit_capacity": unit_capacity,
                "is_capital": is_capital,
            }
        )
    return cities


@dataclass(frozen=True)
class ObservationLayout:
    width: int
    height: int
    n_tiles: int
    terrain_start: int
    terrain_end: int
    unit_types_start: int
    unit_types_end: int
    city_territory_start: int
    city_territory_end: int
    unit_home_city_start: int
    unit_home_city_end: int
    road_start: int
    road_end: int
    buildings_start: int
    buildings_end: int
    resource_start: int
    resource_end: int
    legacy_scalar_start: int
    legacy_scalar_end: int
    economy_scalar_start: int
    economy_scalar_end: int
    tech_vector_start: int
    tech_vector_end: int
    city_block_start: int
    city_block_end: int
    expected_obs_dim: int
    city_slots: int = MAX_OWNED_CITIES
    city_slot_dim: int = CITY_SLOT_FEATURE_DIM


def observation_layout(width: int, height: int) -> ObservationLayout:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid observation geometry: {width}x{height}")
    n_tiles = width * height

    cur = 0
    terrain_start = cur
    cur += n_tiles
    terrain_end = cur

    unit_types_start = cur
    cur += NUM_UNIT_TYPE_CHANNELS * n_tiles
    unit_types_end = cur

    city_territory_start = cur
    cur += NUM_CITY_TERRITORY_CHANNELS * n_tiles
    city_territory_end = cur

    unit_home_city_start = cur
    cur += NUM_UNIT_HOME_CITY_CHANNELS * n_tiles
    unit_home_city_end = cur

    road_start = cur
    cur += n_tiles
    road_end = cur

    buildings_start = cur
    cur += NUM_BUILDING_CHANNELS * n_tiles
    buildings_end = cur

    resource_start = cur
    cur += n_tiles
    resource_end = cur

    legacy_scalar_start = cur
    cur += LEGACY_SCALAR_DIM
    legacy_scalar_end = cur

    economy_scalar_start = cur
    cur += ECONOMY_SCALAR_DIM
    economy_scalar_end = cur

    tech_vector_start = cur
    cur += NUM_TECHNOLOGIES
    tech_vector_end = cur

    city_block_start = cur
    cur += MAX_OWNED_CITIES * CITY_SLOT_FEATURE_DIM
    city_block_end = cur

    expected_obs_dim = cur

    return ObservationLayout(
        width=width,
        height=height,
        n_tiles=n_tiles,
        terrain_start=terrain_start,
        terrain_end=terrain_end,
        unit_types_start=unit_types_start,
        unit_types_end=unit_types_end,
        city_territory_start=city_territory_start,
        city_territory_end=city_territory_end,
        unit_home_city_start=unit_home_city_start,
        unit_home_city_end=unit_home_city_end,
        road_start=road_start,
        road_end=road_end,
        buildings_start=buildings_start,
        buildings_end=buildings_end,
        resource_start=resource_start,
        resource_end=resource_end,
        legacy_scalar_start=legacy_scalar_start,
        legacy_scalar_end=legacy_scalar_end,
        economy_scalar_start=economy_scalar_start,
        economy_scalar_end=economy_scalar_end,
        tech_vector_start=tech_vector_start,
        tech_vector_end=tech_vector_end,
        city_block_start=city_block_start,
        city_block_end=city_block_end,
        expected_obs_dim=expected_obs_dim,
        city_slots=MAX_OWNED_CITIES,
        city_slot_dim=CITY_SLOT_FEATURE_DIM,
    )


def validate_fixed_square_geometry(
    loaded_width: int,
    loaded_height: int,
    expected_width: int | None = None,
    expected_height: int | None = None,
    *,
    level_path: str | None = None,
) -> tuple[int, int]:
    loaded_width = int(loaded_width)
    loaded_height = int(loaded_height)
    where = f"\n  level: {level_path}" if level_path else ""
    if loaded_width <= 0 or loaded_height <= 0:
        raise MapGeometryError(
            f"Invalid map geometry: {loaded_width}x{loaded_height}.{where}"
        )
    if loaded_width != loaded_height:
        raise MapGeometryError(
            "Rectangular maps are unsupported.\n"
            f"  loaded map geometry: {loaded_width}x{loaded_height}{where}\n\n"
            "Use a square, dimension-homogeneous level pool."
        )
    if expected_width is not None and expected_height is not None:
        expected_width = int(expected_width)
        expected_height = int(expected_height)
        if loaded_width != expected_width or loaded_height != expected_height:
            raise MapGeometryError(
                "Map geometry mismatch:\n"
                f"  environment geometry: {expected_width}x{expected_height}\n"
                f"  loaded map geometry: {loaded_width}x{loaded_height}{where}\n\n"
                "Mixed-dimension level pools are unsupported."
            )
    return loaded_width, loaded_height


CHECKPOINT_REQUIRED_FIELDS = (
    "map_width",
    "map_height",
    "observation_dim",
    "action_space_n",
    "action_catalog_fingerprint",
    "actor_mode",
    "legal_action_feature_version",
    "legal_action_feature_dim",
    "catalog_version",
    "canonicalizer_version",
    "phase1_opening_version",
    "phase1_environment_version",
    "max_legal_actions",
)

_INTEGER_FIELDS = {
    "map_width",
    "map_height",
    "observation_dim",
    "action_space_n",
    "legal_action_feature_dim",
    "max_legal_actions",
}


def environment_compatibility_metadata(
    wrapper,
    *,
    actor_mode: str,
    max_legal_actions: int | None = None,
) -> dict:
    catalog = getattr(wrapper, "_catalog", None)
    if catalog is None:
        raise RuntimeError("Environment action catalog is not initialized.")
    obs_shape = tuple(int(v) for v in wrapper.observation_space.shape)
    if len(obs_shape) != 1:
        raise RuntimeError(f"Expected a flat observation space, got shape={obs_shape}")
    metadata = {
        "map_width": int(catalog.width),
        "map_height": int(catalog.height),
        "observation_dim": int(obs_shape[0]),
        "action_space_n": int(wrapper.action_space.n),
        "action_catalog_fingerprint": str(wrapper._catalog_fingerprint),
        "actor_mode": str(actor_mode).strip().lower(),
        "legal_action_feature_version": str(wrapper.LEGAL_ACTION_FEATURE_VERSION),
        "legal_action_feature_dim": int(wrapper.ACTION_FEATURE_DIM),
        "catalog_version": str(wrapper.CATALOG_VERSION),
        "canonicalizer_version": str(wrapper.CANONICALIZER_VERSION),
        "phase1_opening_version": str(wrapper.PHASE1_OPENING_VERSION),
        "phase1_environment_version": str(getattr(wrapper, "PHASE1_ENVIRONMENT_VERSION", PHASE1_ENVIRONMENT_VERSION)),
    }
    actual_max_legal_actions = getattr(wrapper, "_max_legal_actions", max_legal_actions)
    if actual_max_legal_actions is not None:
        metadata["max_legal_actions"] = int(actual_max_legal_actions)
    return metadata


def read_checkpoint_metadata(model_path: str) -> dict:
    meta_path = model_path + ".action_interface.json"
    if not os.path.isfile(meta_path):
        raise CheckpointCompatibilityError(
            "Checkpoint metadata is insufficient to establish compatibility.\n"
            f"Missing metadata file: {meta_path}"
        )
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise CheckpointCompatibilityError(
            "Checkpoint metadata is insufficient to establish compatibility.\n"
            f"Could not read {meta_path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise CheckpointCompatibilityError(
            "Checkpoint metadata is insufficient to establish compatibility.\n"
            f"Expected a JSON object in: {meta_path}"
        )
    return data


def _normalized_value(field: str, value):
    if field in _INTEGER_FIELDS:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    return str(value).strip().lower() if field == "actor_mode" else str(value)


def validate_checkpoint_compatibility(
    checkpoint: Mapping,
    environment: Mapping,
) -> None:
    missing_checkpoint = [k for k in CHECKPOINT_REQUIRED_FIELDS if checkpoint.get(k) is None]
    missing_environment = [k for k in CHECKPOINT_REQUIRED_FIELDS if environment.get(k) is None]
    if missing_checkpoint or missing_environment:
        details = []
        if missing_checkpoint:
            details.append("checkpoint missing: " + ", ".join(missing_checkpoint))
        if missing_environment:
            details.append("environment missing: " + ", ".join(missing_environment))
        raise CheckpointCompatibilityError(
            "Checkpoint metadata is insufficient to establish compatibility.\n"
            + "\n".join(f"  {line}" for line in details)
        )

    fields = list(CHECKPOINT_REQUIRED_FIELDS)
    differences = []
    for field in fields:
        checkpoint_value = _normalized_value(field, checkpoint.get(field))
        environment_value = _normalized_value(field, environment.get(field))
        if checkpoint_value != environment_value:
            differences.append((field, checkpoint_value, environment_value))
    if differences:
        rows = "\n".join(
            f"  {field}: checkpoint={checkpoint_value!r}, environment={environment_value!r}"
            for field, checkpoint_value, environment_value in differences
        )
        raise CheckpointCompatibilityError(
            "Checkpoint is incompatible with the current environment.\n\n"
            f"Differences:\n{rows}\n\n"
            "Use a checkpoint trained for the same environment geometry/action interface."
        )


def compute_level_pool_identity(
    level_paths: Iterable[str],
    *,
    relative_to: str | None = None,
) -> tuple[str, list[dict]]:
    root = os.path.abspath(relative_to) if relative_to else None
    rows = []
    for path in sorted({os.path.abspath(p) for p in level_paths}):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        label = os.path.relpath(path, root) if root else os.path.basename(path)
        rows.append(
            {
                "path": label.replace("\\", "/"),
                "size_bytes": int(os.path.getsize(path)),
                "sha256": h.hexdigest(),
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), rows
