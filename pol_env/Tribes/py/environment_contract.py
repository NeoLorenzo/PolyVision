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


PHASE1_ENVIRONMENT_VERSION = "v4_exact_per_city_state"

MAX_OWNED_CITIES = 9
CITY_SLOT_FEATURE_DIM = 9
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
)
CITY_BLOCK_DIM = MAX_OWNED_CITIES * CITY_SLOT_FEATURE_DIM  # 81


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


def extract_owned_cities(
    obs_dict: Mapping,
    tribe_id: int,
) -> list[OwnedCityState]:
    """Extract canonical owned-city states from fog-respecting POV observation dictionary.

    Cities are deterministically sorted by spatial coordinates (x, y) ascending.
    Raw actor IDs are never exposed in the returned records.
    """
    if not isinstance(obs_dict, Mapping):
        raise ObservationContractError(f"obs_dict must be a Mapping, got {type(obs_dict).__name__}")
    tribe_id = int(tribe_id)
    city_map = obs_dict.get("city", {})
    if not isinstance(city_map, Mapping):
        return []
    owned: list[OwnedCityState] = []
    for raw_actor_id, city_info in city_map.items():
        if not isinstance(city_info, Mapping):
            continue
        try:
            c_tribe = int(city_info.get("tribeID", -1))
        except Exception:
            continue
        if c_tribe != tribe_id:
            continue
        try:
            x = int(city_info.get("x", -1))
            y = int(city_info.get("y", -1))
            level = int(city_info.get("level", 1))
            population = int(city_info.get("population", 0))
            population_need = int(city_info.get("population_need", level + 1))
            production = int(city_info.get("production", 1))
            units = city_info.get("units", [])
            supported_unit_count = len(units) if isinstance(units, (list, tuple)) else 0
            unit_capacity = level + 1
        except Exception as exc:
            raise ObservationContractError(
                f"Malformed city record for actor {raw_actor_id}: {exc}"
            ) from exc

        owned.append(
            OwnedCityState(
                x=x,
                y=y,
                level=level,
                population=population,
                population_need=population_need,
                production=production,
                supported_unit_count=supported_unit_count,
                unit_capacity=unit_capacity,
            )
        )
    # Sort deterministically by spatial coordinates (x, y) ascending: primary key x, secondary key y
    owned.sort(key=lambda c: (c.x, c.y))
    return owned


def encode_owned_city_slots(
    owned_cities: list[OwnedCityState],
    width: int,
    height: int,
    max_cities: int = MAX_OWNED_CITIES,
) -> list[float]:
    """Encode owned cities into a fixed-width, exact, non-clipping 1D feature list.

    Raises ObservationContractError if owned city count exceeds max_cities or coordinates
    are out of bounds.
    """
    if len(owned_cities) > max_cities:
        raise ObservationContractError(
            f"Owned city count {len(owned_cities)} exceeds MAX_OWNED_CITIES={max_cities}"
        )
    max_x = max(1, width - 1)
    max_y = max(1, height - 1)
    features: list[float] = []
    for i in range(max_cities):
        if i < len(owned_cities):
            city = owned_cities[i]
            if city.x < 0 or city.x >= width or city.y < 0 or city.y >= height:
                raise ObservationContractError(
                    f"City coordinates ({city.x}, {city.y}) out of bounds for board {width}x{height}"
                )
            features.extend([
                1.0,  # city_present
                float(city.x) / float(max_x),
                float(city.y) / float(max_y),
                float(city.level),
                float(city.population),
                float(city.population_need),
                float(city.production),
                float(city.supported_unit_count),
                float(city.unit_capacity),
            ])
        else:
            features.extend([0.0] * CITY_SLOT_FEATURE_DIM)
    return features


def decode_owned_city_slots(
    city_block: Iterable[float],
    width: int,
    height: int,
    max_cities: int = MAX_OWNED_CITIES,
) -> list[dict[str, Any]]:
    """Decode exact integer city primitives from a flattened city-slot feature block."""
    import numpy as np

    block = np.asarray(city_block, dtype=np.float32).reshape(-1)
    expected_size = max_cities * CITY_SLOT_FEATURE_DIM
    if block.size != expected_size:
        raise ObservationContractError(
            f"Expected city block size {expected_size}, got {block.size}"
        )
    max_x = max(1, width - 1)
    max_y = max(1, height - 1)
    cities: list[dict[str, Any]] = []
    for i in range(max_cities):
        offset = i * CITY_SLOT_FEATURE_DIM
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
            }
        )
    return cities


@dataclass(frozen=True)
class ObservationLayout:
    width: int
    height: int
    n_tiles: int
    legacy_obs_dim: int
    resource_block_dim: int
    scalar_start: int
    scalar_end: int
    city_block_start: int
    city_block_end: int
    expected_obs_dim: int
    resource_start: int
    resource_end: int
    city_slots: int = MAX_OWNED_CITIES
    city_slot_dim: int = CITY_SLOT_FEATURE_DIM


def observation_layout(width: int, height: int) -> ObservationLayout:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid observation geometry: {width}x{height}")
    n_tiles = width * height
    legacy_obs_dim = 3 * n_tiles + 6
    resource_block_dim = n_tiles
    scalar_start = legacy_obs_dim + resource_block_dim
    scalar_dim = 15
    scalar_end = scalar_start + scalar_dim
    city_block_start = scalar_end
    city_block_dim = MAX_OWNED_CITIES * CITY_SLOT_FEATURE_DIM  # 81
    city_block_end = city_block_start + city_block_dim
    expected_obs_dim = city_block_end
    return ObservationLayout(
        width=width,
        height=height,
        n_tiles=n_tiles,
        legacy_obs_dim=legacy_obs_dim,
        resource_block_dim=resource_block_dim,
        scalar_start=scalar_start,
        scalar_end=scalar_end,
        city_block_start=city_block_start,
        city_block_end=city_block_end,
        expected_obs_dim=expected_obs_dim,
        resource_start=legacy_obs_dim,
        resource_end=scalar_start,
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
