# Observations

```text
PARITY-001 — Exact Per-City State
Status: CLOSED / FROZEN
Environment contract: v4_exact_per_city_state
Observation dimension: 586
```

`Tribes-v0` exposes a one-dimensional `float32` observation. For the active 11×11 Phase-1 contract (`v4_exact_per_city_state`), its dimension is:

```text
observation_dim = 4 × (width × height) + 21 + (MAX_OWNED_CITIES × CITY_SLOT_FEATURE_DIM)
                = 4 × 121 + 21 + (9 × 9)
                = 484 + 21 + 81
                = 586
```

The active 11×11 pool therefore produces **586 values**.

## Layout

The vector preserves the legacy and scalar prefix and appends the exact per-city state block introduced in `PARITY-001`:

| Block | Range (11×11) | Size | Contents |
|---|---|---:|---|
| Terrain | `0..120` | 121 | Java observation terrain IDs, including fog markers. |
| Unit IDs | `121..241` | 121 | Board unit identifiers from the visible observation. |
| City IDs | `242..362` | 121 | Board city identifiers from the visible observation. |
| Legacy scalars | `363..368` | 6 | Bardur stars, score, city count, kills, engine tick, and active tribe ID. |
| Visible resources | `369..489` | 121 | Resource IDs normalized to `[0,1]`; fogged tiles are masked before normalization. |
| Economy/state scalars | `490..504` | 15 | Normalized stars/SPT/turn timing, key technologies, research count, city levels, and upgrade progress/readiness. |
| **Exact Per-City Slots** | `505..585` | 81 | 9 deterministic city slots (9 features each) preserving exact unclipped entity state. |

## Exact Per-City State Block (`PARITY-001`)

Under the `v4_exact_per_city_state` contract, exact state for every owned city is preserved across 9 fixed slots (`MAX_OWNED_CITIES = 9`, matching the empirical maximum on 11×11 maps).

### Slot Ordering
Owned cities are extracted exclusively from the fog-respecting POV observation JSON (`obs_dict["city"]` where `tribeID == controlled_tribe_id`) and sorted deterministically by spatial coordinates $(x, y)$ ascending (primary key $x$, secondary key $y$). Raw engine actor IDs are never exposed in the city slots.

### Slot Schema (9 Features per Slot)
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

Unused slots ($K < 9$) are strictly padded with `0.0` across all 9 values. Any observation producing more than 9 owned cities violates the contract and raises `ObservationContractError`.

## Visibility boundary

Policy observations come from Java's normal `observationJson()` and respect fog of war. The visible resource block explicitly masks fogged resources. `TribesGymEnv.get_observation(full_visibility=True)` exists for diagnostics and audits; it must not be treated as the training observation or used by a fair visible-information baseline.

## Geometry and compatibility

The wrapper derives the observation space and global action catalog from the map pool and enforces square, dimension-homogeneous pools. Changing geometry or interface contract changes observation dimension and catalog fingerprint, so checkpoints are interface-specific. Historical 505-dimensional checkpoints (`v3_corrected_turn_economy`) are cleanly rejected by `validate_checkpoint_compatibility()`.

See [Actions](actions.md) for the policy interface and [Reproducibility](reproducibility.md) for checkpoint contract fields.
