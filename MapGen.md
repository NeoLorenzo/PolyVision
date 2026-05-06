# Polytopia Map Generation Reference (Cleaned)

This document is a cleaned, data-focused reference of Polytopia map generation behavior from the provided wiki text.
All UI/navigation/media embeds were removed.

## Parity Legend (Tribes Java Env)

- `FULL`: Implemented and matched in current Tribes generation logic.
- `PARTIAL`: Implemented in part, approximate, or missing map-type-specific details.
- `NOT IMPLEMENTED`: Not yet implemented in Tribes generation logic.
- `SOURCE NOTE`: Reference behavior/quirk from source text (not a direct implementation target).

Snapshot date: `2026-05-06`

## 1) Core Model

Parity status: `PARTIAL`

- Maps are square and randomly generated.
- Terrain/resource distributions vary by tribe via modifiers.
- Initial land is distributed roughly equally among tribes.
- Generation order varies by map type, but generally follows:
  - capitals
  - villages
  - terrain
  - resources
  - ruins/starfish

## 2) Map Sizes

Parity status: `PARTIAL`

| Size | Tiles | Dimensions |
|---|---:|---:|
| Tiny | 121 | 11x11 |
| Small | 196 | 14x14 |
| Normal | 256 | 16x16 |
| Large | 324 | 18x18 |
| Huge | 400 | 20x20 |
| Massive | 900 | 30x30 |

## 3) Water Profiles (Map Types)

Parity status: `PARTIAL`

| Map Type | Wetness | Effect |
|---|---|---|
| Drylands | 0-10% | Almost no water |
| Lakes | 25-30% | Inland lakes; edge land-bridge behavior |
| Continents | 40-70% | Larger ocean, shrinking continents as wetness rises |
| Pangea | 40-60% | Central landmass + surrounding ocean |
| Archipelago | 60-80% | More fragmented islands/coasts |
| Waterworld | 90-100% | Almost all water/ocean; land forced for city viability |

Notes:
- Drylands exception: Kickoo and Aquarion capitals include two water tiles with fish.
- Lakes note from source: each player is guaranteed land connections to at least two villages.

## 4) Base Spawn Rates (Land Tiles)

Parity status: `PARTIAL`

Implemented in Tribes now:
- Quota-normalized placement (exact-count allocation) per tribe and band, instead of purely per-tile random rolls.
- Inner/outer band resource quotas for field/forest/mountain and fish/whales water quotas.

Remaining gap:
- Exact global percentages and map-type-specific post-processing still need calibration against Polytopia outputs.

"Inner city" = adjacent to a city or village.
"Outer city" = not adjacent to a city or village.

| Category | Inner City | Outer City |
|---|---:|---:|
| Field (total) | 48% | 48% |
| Fruit | 18% | 6% |
| Crop | 18% | 6% |
| Empty field | 12% | 36% |
| Forest (total) | 38% | 38% |
| Animal | 19% | 6% |
| Empty forest | 19% | 32% |
| Mountain (total) | 14% | 14% |
| Metal | 11% | 3% |
| Empty mountain | 3% | 11% |

Additional base rules:
- Fish base spawn is 50% on shallow water tiles.
- Fish can also spawn in some ocean tiles (including some outside borders reachable via border growth).
- Starfish can spawn on any water tiles.
- Resources spawn only within two tiles of cities/villages.

## 5) Tribe Spawn Modifiers

Parity status: `PARTIAL`

Implemented in Tribes now:
- Tribe multipliers are applied during quota generation for terrain/resources.
- Modifier application order for mountain/forest/field residual is implemented.

Remaining gap:
- Some tribe-specific special cases (for example water replacement quirks and faction-specific exceptions) are not fully mirrored.

Modifiers are multiplicative on base rates.

| Tribe | Modifiers |
|---|---|
| Xin-xi | 1.5x mountain, 1.5x metal |
| Imperius | 0.5x animal, 2.0x fruit |
| Bardur | 0.8x forest, 0x crop |
| Oumaji | 0.2x forest, 0.2x animal, 0.5x mountain, 0.5x water (source note) |
| Kickoo | 0.5x mountain, 1.5x fish, 2.0x water (source note) |
| Hoodrick | 0.5x mountain, 1.5x forest |
| Luxidoor | base rates |
| Vengir | 2.0x metal, 0.1x animal, 0.1x fruit, 0.1x fish |
| Zebasi | 0.5x mountain, 0.5x forest, 0.5x fruit |
| Ai-Mo | 1.5x mountain, 0.1x crop |
| Quetzali | 2.0x fruit, 0.1x crop |
| Yadakk | 0.5x mountain, 0.5x forest, 1.5x fruit |
| Aquarion | 0.5x forest, 1.5x water (source note) |
| Elyrion | 0.5x mountain, 1.5x crop |
| Polaris | same as non-Polaris opponents; else base |
| Cymanti | 1.2x mountain, crop replaced by spore; cannot spawn crop |

### Modifier Application Order

Parity status: `FULL`

1. Apply mountain modifier first (affects mountain directly; indirectly shifts forest/field share).
2. Apply forest modifier second (affects forest directly; then field remainder).
3. Field is residual:

`Field% = 100% - Mountain% - Forest%`

## 6) Known Water-Modifier Quirk (from source text)

Parity status: `SOURCE NOTE`

- Water-reduction logic appears in source but is stated as not impacting actual world generation.
- Stated bug: on Continents, Kickoo/Aquarion post-initial water replacement (40% / 30%) is not applied.

## 7) Capital Placement

Parity status: `PARTIAL`

For Drylands/Lakes/Archipelago/Waterworld:
- Map is divided into domains ("quadrants") to separate starts.
- Domain count by player count:
  - 1-4 players: 4 domains (`FULL`)
  - 5-9 players: 9 domains (`FULL`)
  - 10-16 players: 16 domains (`FULL`)
- Capitals placed into unoccupied domains with weighted randomness.

For Continents/Pangea:
- No quadrant system (different pipeline; see section 9).

## 8) Village Systems

Parity status: `PARTIAL`

Village categories:
- Suburbs (Lakes/Archipelago only)
- Pre-terrain villages (primary)
- Post-terrain villages (secondary)

### 8.1 Feature Matrix by Map Type

Parity status: `PARTIAL`

| Feature | Drylands | Archipelago | Lakes | Waterworld | Continents/Pangea |
|---|---|---|---|---|---|
| Quadrants for capitals | Yes | Yes | Yes | Yes | No |
| Suburbs | No | Yes | Yes | No | No |
| Pre-terrain villages | No | Yes | Yes | Yes | No |
| Post-terrain villages | Yes | Yes | Yes | Yes | Yes |
| Tiny island villages | No | No | No | Yes | Yes |
| Pre-terrain village coefficient | None | 0.3 | 0.3 | 0.1 | None |

### 8.2 Suburbs

Parity status: `PARTIAL`

- Used only on Lakes and Archipelago.
- Up to 2 villages associated with each capital after capital placement.
- Can be 0/1/2; larger maps tend to produce 2 more often.
- Placed before terrain; often near capital in practice.

### 8.3 Pre-Terrain Villages

Parity status: `PARTIAL`

Used on Archipelago, Lakes, Waterworld.

Formula:

`PreTerrainVillages = ((floor(mapWidth / 3)^2) - (capitals + suburbs)) * densityCoefficient`

Density coefficient:
- Lakes: 0.3
- Archipelago: 0.3
- Waterworld: 0.1

Constraints:
- At least two tiles from other villages.
- At least one tile from map edge.

### 8.4 Post-Terrain Villages

Parity status: `PARTIAL`

After terrain generation, villages are added repeatedly until no valid tile remains.

Stated constraints from source text:
- Not within two tiles of map edge.
- Not within two tiles of other villages.
- "Not be three tiles from the edge" (source wording preserved; ambiguous).

## 9) Continents and Pangea (Special Pipeline)

Parity status: `PARTIAL`

Implemented in Tribes now:
- Dedicated land-generation branches for Continents and Pangea.
- Village-first flow with later capital conversion for these map types.

Remaining gap:
- Current implementation is an approximation, not exact Polytopia continent/noise behavior and placement constraints.

General:
- No suburbs.
- No pre-terrain villages.
- Villages are added after land generation.
- Capitals are converted from villages later.

### 9.1 Pangea

Parity status: `PARTIAL`

- Land seeded from a central start point.
- Land/water ratio determined by wetness (about half water typical per source).
- Villages added one at a time on land until no valid placement remains (>=2 tile separation).
- Some villages are then converted to capitals.
- Capital conversion preferences:
  - maximize distance between capitals
  - prefer adjacency to water (coastal bias)
- Tiny-island villages generated after mainland setup.

### 9.2 Continents

Parity status: `PARTIAL`

- Landmasses generated by noise process.
- Landmass count depends on players, wetness, map size.
- Typical total water around half (source statement).
- Reported continent sizes roughly 30-200 tiles; continents at least one tile apart.
- Villages placed per continent with >=2 tile separation.
- At least one village per continent.
- Capitals then converted from villages, trying to separate capitals across landmasses if possible.

## 10) Tiny Island Villages

Parity status: `PARTIAL`

Implemented in Tribes now:
- Tiny-island village spawning pass exists for Continents/Pangea.

Remaining gap:
- Count/distribution constraints are approximated and still need strict parity calibration.

For Continents and Pangea, fixed count by map size:

| Map Size | Tiles | Tiny Island Villages |
|---|---:|---:|
| Tiny | 121 | 0 |
| Small | 196 | 1 |
| Normal | 256 | 2 |
| Large | 324 | 3 |
| Huge | 400 | 4 |
| Massive | 900 | 9 |

## 11) Ruins

Parity status: `PARTIAL`

- Spawn after villages/resources/lighthouses (per source text ordering).
- Can spawn on mountains, forests, fields, or deep ocean.
- Cannot spawn adjacent to another ruin or a village.
- Count scales with map size.
- Lakes-specific cap: at most one third of ruins on water.

| Map Size | Tiles | Ruins |
|---|---:|---:|
| Tiny | 121 | 4 |
| Small | 196 | 5 |
| Normal | 256 | 7 |
| Large | 324 | 9 |
| Huge | 400 | 11 |
| Massive | 900 | 23 |

## 12) Starfish

Parity status: `NOT IMPLEMENTED`

- Approximate rate: 1 starfish per 25 water tiles.
- Can spawn in shallow or deep water.
- Cannot be adjacent to another starfish, a lighthouse, or a city.

## 13) Special Exception Mentioned in Source

Parity status: `NOT IMPLEMENTED`

- Aquarion: claiming a deep-water ruin can create a Lost City (level 3) and nearby water resources (fish/aquacrops).
