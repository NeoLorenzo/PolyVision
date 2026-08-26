# Observations

```text
PARITY-002 — Phase-1 Human Information Parity (PARITY-002.1 Hardened)
Status: CLOSED
Environment contract: v5_human_information_parity
Observation dimension: 6424
```

`Tribes-v0` exposes a one-dimensional `float32` observation tensor. For the active 11×11 Phase-1 contract (`v5_human_information_parity`), its dimension is:

```text
spatial_dim  = (1 terrain + 12 unit types + 9 city territory + 9 unit home city + 1 road + 19 buildings + 1 resource) × (11 × 11)
             = 52 channels × 121 tiles
             = 6292

scalar_dim   = 6 legacy + 12 economy + 24 technologies + (9 city slots × 10 features)
             = 132

observation_dim = 6292 + 132 = 6424
```

The active 11×11 pool produces **6,424 values**.

## Layout

The vector layout contains 52 fog-masked spatial channels followed by scalar and structured entity blocks:

| Block | Channel Count | Range (11×11) | Size | Contents |
|---|:---:|---|---:|---|
| **Terrain** | 1 | `0..120` | 121 | Terrain IDs (0..7, where 7 = FOG). |
| **Categorical Unit Types** | 12 | `121..1572` | 1452 | One binary spatial channel per unit type in `SUPPORTED_UNIT_TYPES` order (`WARRIOR`, `RIDER`, `DEFENDER`, `SWORDMAN`, `ARCHER`, `CATAPULT`, `KNIGHT`, `MIND_BENDER`, `BOAT`, `SHIP`, `BATTLESHIP`, `SUPERUNIT`). |
| **Deterministic City Territory** | 9 | `1573..2661` | 1089 | One binary spatial channel per deterministic city slot (0..8) representing claimed territory tiles. |
| **Unit Home City Association** | 9 | `2662..3750` | 1089 | One binary spatial channel per deterministic city slot (0..8) mapping unit locations to their supporting home city. |
| **Road Grid** | 1 | `3751..3871` | 121 | Binary road plane (`1.0` if visible road present on tile, `0.0` otherwise). |
| **Categorical Buildings** | 19 | `3872..6170` | 2299 | One binary spatial channel per building type in `SUPPORTED_BUILDINGS` order (`PORT`, `MINE`, `TEMPLE`, `WATER_TEMPLE`, `FOREST_TEMPLE`, `MOUNTAIN_TEMPLE`, `FARM`, `WINDMILL`, `SAWMILL`, `CUSTOMS_HOUSE`, `FORGE`, `LUMBER_HUT`, `ALTAR_OF_PEACE`, `GRAND_BAZAR`, `EMPERORS_TOMB`, `EYE_OF_GOD`, `GATE_OF_POWER`, `PARK_OF_FORTUNE`, `TOWER_OF_WISDOM`). |
| **Visible Resources** | 1 | `6171..6291` | 121 | Resource IDs normalized to `[0,1]`; fogged tiles masked to `0.0`. |
| **Legacy Scalars** | — | `6292..6297` | 6 | Stars, score, city count, kills, engine tick, active tribe ID. |
| **Economy Scalars** | — | `6298..6309` | 12 | Normalized stars/SPT/turn timing, city count, average/max city level, mean/max upgrade progress, upgrade ready fraction, level-up available flag. |
| **Researched Technologies** | 24 | `6310..6333` | 24 | Full binary indicator vector for all 24 technologies in `TECHNOLOGY_ORDER` order. |
| **Exact Per-City Slots** | 9 slots | `6334..6423` | 90 | 9 deterministic city slots (10 features each) preserving exact unclipped entity state and capital identity. |

## Exact Per-City State Block (`PARITY-001` & `PARITY-002`)

Under `v5_human_information_parity`, exact state for every owned city is preserved across 9 fixed slots (`MAX_OWNED_CITIES = 9`).

### Slot Ordering
Owned cities are extracted exclusively from the fog-respecting POV observation JSON (`obs_dict["city"]` where `tribeID == controlled_tribe_id`) and sorted deterministically by spatial coordinates $(x, y)$ ascending (primary key $x$, secondary key $y$). Raw engine actor IDs are completely excluded.

### Slot Schema (10 Features per Slot)
For each slot $i \in \{0 \dots 8\}$:
1. `city_present`: `1.0` if city exists in slot $i$, `0.0` if empty slot.
2. `city_x`: `float(x) / float(width - 1)` (normalized to $[0, 1]$ by board geometry).
3. `city_y`: `float(y) / float(height - 1)` (normalized to $[0, 1]$ by board geometry).
4. `city_level`: Exact unclipped `float(level)`.
5. `city_population`: Exact unclipped `float(population)`.
6. `city_population_need`: Exact unclipped `float(population_need)`.
7. `city_production`: Exact unclipped `float(production)` (city SPT contribution).
8. `city_supported_unit_count`: Exact unclipped `float(len(city["units"]))`.
9. `city_unit_capacity`: Exact unclipped `float(level + 1)`.
10. `city_is_capital`: `1.0` if capital city, `0.0` otherwise (`STATE-MAP-003`).

Unused slots ($K < 9$) are strictly padded with `0.0` across all 10 values. Any observation producing more than 9 owned cities violates the contract and raises `ObservationContractError`.

## Defense-in-Depth Fog Masking

All spatial channels (unit types, city territory, unit home city, road, buildings, resources) are explicitly fog-masked in Python (`fog_mask = (terrain == 7)`). No value contained in hidden tiles within Java serialization can influence the observation tensor.

## Visibility Boundary & Compatibility

Policy observations come from Java's fog-respecting `observationJson()`. Changing geometry or interface contract changes observation dimension and catalog fingerprint, so checkpoints are interface-specific. Historical checkpoints (`v3_corrected_turn_economy` with 505 dims, and `v4_exact_per_city_state` with 586 dims) are cleanly rejected by `validate_checkpoint_compatibility()`.

See [Actions](actions.md) for the policy interface and [Reproducibility](reproducibility.md) for checkpoint contract fields.
