# Baseline Human–AI Information Parity Audit

**Authoritative Baseline Audit & Candidate Specification**  
**Document Version:** `2.1.0`  
**Date:** 2026-08-26  
**Target Environment:** `Tribes-v0` (`v5_human_information_parity`, `flat-v1-structured`, `v1_4_parity_spatial_and_cost`)  
**Scope:** Normative information parity audit comparing normal human-visible Polytopia information against PolyVision PPO policy visibility. Fully reconciled across baseline `v3`, `PARITY-001`, `PARITY-002`, and `PARITY-002.1` milestones.

---

## 1. Purpose & Conceptual Foundations

### 1.1 Definition of Human–AI Information Parity
In reinforcement learning research for complex strategy games, **Human–AI Information Parity** requires that:
1. The artificial agent has access to **all legitimate information** that a normal human player can observe or directly inspect through the standard game interface.
2. The artificial agent has access to **no privileged or hidden information** (such as fog-of-war state, invisible opponent assets, or internal engine indices) that a human player cannot legitimately observe.
3. The legal choices and action metadata available to the agent are governed strictly by **fog-respecting game legality** without leaking unobserved state through action availability or precomputed features.

### 1.2 Human Reference Target
> **Normative Baseline Target:**  
> PolyVision aims to reproduce the informational conditions of a human playing Polytopia-style strategy games, applied directly to the mechanics simulated by the Tribes engine. The goal of this audit is not to replicate every commercial Polytopia patch feature from August 2026, but to establish full informational parity for the game rules as implemented. Where the Tribes engine differs intentionally or historically from commercial Polytopia, such differences are explicitly documented as `ENGINE_IMPLEMENTATION_DIFFERENCE`.

### 1.3 Benchmark-Interface Parity vs. Ordinary-Human Parity
PolyVision maintains a verified parity standard between the PPO policy and the official human benchmark interface (`tools/human_policy_interface.py` / `tools/human_benchmark.py`). Under `v5_human_information_parity`, both human and agent receive identical full entity-level and spatial observations (6,424 floats) and 47-dimensional legal-action feature tensors.

```
[Target Information Parity — Realized in v5_human_information_parity]
Ordinary Human Interface (Polytopia)  ============================  Parity Policy Interface (TribesGymWrapper)
                                                                                  ==
                                                                     Human Benchmark UI (tools/human_policy_interface.py)
```

The standard established in this audit is **not** to restrict the human player, but to **expand and structure the AI interface until it achieves informational parity with an unrestricted human playing the defined task**.

### 1.4 Strict Distinction Between Problem Classes
Throughout this audit, every disparity is strictly separated into one of three distinct problem classes:
1. **Information Availability Problem:** The AI genuinely lacks legitimate information that a human player can inspect (or the AI receives hidden information).
2. **Representation Problem:** The information is technically present across policy inputs, but it has been destructively aggregated, flattened, poorly encoded, or coupled to arbitrary scalar values.
3. **Task-Definition Difference:** The environment intentionally restricts the action or state space as a defined research constraint of Phase 1 (e.g. Turn-10 termination, combat exclusion), distinct from an accidental informational disparity.

---

## 2. Parity Classification & Provenance Taxonomy

### 2.1 Primary Parity Classification
Every audited item is assigned **exactly one** primary classification:

| Primary Classification | Code | Definition |
|---|:---:|---|
| **PARITY** | `PARITY` | The human and AI have materially equivalent access to the primitive information. |
| **AI DEFICIT** | `AI_DEFICIT` | A normal human has direct UI access to information that the policy cannot access across any of its inputs. |
| **AI PRIVILEGE** | `AI_PRIVILEGE` | The AI receives or exploits hidden information that a normal human cannot legitimately obtain under fog of war. |
| **REPRESENTATION LOSS** | `REPRESENTATION_LOSS` | The information is available across policy inputs, but in a destructively aggregated, flattened, or distorted form that loses entity-level resolution. |
| **DERIVED ADVANTAGE** | `DERIVED_ADVANTAGE` | The AI receives a precomputed calculation derived from public state and rules that a human must calculate manually. |
| **TASK DIFFERENCE** | `TASK_DIFFERENCE` | An intentional restriction of the defined Phase-1 task scope rather than an accidental informational defect. |
| **UNVERIFIED** | `UNVERIFIED` | Insufficient static or empirical evidence exists to classify the item with high confidence. |

### 2.2 Secondary Descriptive Flags
- `REPRESENTATION_HAZARD`: Input structure introduces spurious mathematical or numerical inductive bias (e.g., treating categorical actor IDs as continuous floats).
- `CROSS_LAYER_PARTIAL`: Absent from state observation, but partially recoverable through legal action identity or action features.
- `INDIRECT_LEGALITY_SIGNAL`: Information not explicitly represented as a primitive feature but partially or fully inferable from the presence/absence of legal actions.
- `DORMANT_PHASE1`: Code contains hidden-state dependence, but the bug is dormant in solo no-combat Phase 1.
- `FULL_GAME_ONLY`: Relevant only for multi-tribe or full-game combat settings.
- `STATIC_RULE_KNOWLEDGE`: Public rule learnable into model parameters vs. dynamic runtime state.
- `ENGINE_IMPLEMENTATION_DIFFERENCE`: The Tribes Java engine mechanics differ from commercial Polytopia mechanics.
- `HUMAN_UI_UNVERIFIED`: Polytopia commercial UI behavior requires further empirical verification.

### 2.3 Policy Information Provenance Model
Every policy-visible feature traces its provenance to one of five mutually exclusive data origins:
- **Source H0 (Human State Primitive):** Directly observable runtime state from the visible board or UI (e.g., city population dots, unit HP).
- **Source H1 (Static Game Rule):** Public game rule or constant invariant (e.g., Warrior cost = 2, base tech cost = 4).
- **Source H2 (Public Deterministic Derivation):** Deterministic mathematical combination of H0 and H1 (e.g., remaining population needed to level up = $\text{need} - \text{pop}$).
- **Source T (Task Metadata):** Defined environment task parameters (e.g., Phase-1 Turn-10 horizon countdown).
- **Source P (Privileged / Hidden State):** State concealed by fog of war or internal engine internals. **Must never reach the policy.**

---

## 3. The Three Information Availability Boundaries

To avoid conflating internal engine fields with serialized bridge observations and policy tensor inputs, the audit enforces strict separation across three layers:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. JAVA ENGINE AUTHORITATIVE STATE (GameState gs)                                            │
│    Contains full game state (all actors, full board, hidden fog tiles, internal counters).  │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │ gs.copy(activeTribeID) -> Board.copy(partialObs=true)
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. FOG-RESPECTING JAVA POV SERIALIZATION (PythonEnv.observationJson())                      │
│    Masks fog to TERRAIN.FOG(7), hides unrevealed actors. Serializes JSON dictionary.        │
│    Contains board, unit, city, tribes, tick, and activeTribeID primitives.                  │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │ TribesGymWrapper._dict_to_array() & info tensors
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. PPO POLICY VISIBILITY (Union of all policy-visible inputs in v5)                         │
│    ├── A. 6,424-dim State Observation (Vectorized 1D array: 52 spatial planes + 132 scalars)│
│    ├── B. Legal Action Global IDs (Discrete categorical catalog slots)                      │
│    └── C. 47-dim Legal Action Feature Rows (Per-slot semantic/spatial/economic vector)      │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Layer A: Game-State Information Audit

### 4.1 Board & Map Spatial State

| Information Element | Human UI Access | Engine State Has It? | POV JSON Serializes It? | PPO State Observation (v5) | Whole-Policy Visibility | Primary Class | Secondary Flags | Evidence / Code Source | Human UI Evidence & Notes |
|---|---|:---:|:---:|---|---|:---:|---|---|---|
| **Terrain Geometry** | Visible per revealed tile | Yes (`Board.terrains`) | Yes (`board.terrain`) | Yes (121 floats, $[0..7]$) | Full | `PARITY` | — | `Board.java:197`, `PythonEnv.java:341`, `environment_contract.py` | Rendered on game board (`polytopia.io/gameplay`). Plain(0), ShallowWater(1), DeepWater(2), Mountain(3), Village(4), City(5), Forest(6), Fog(7). |
| **Fog of War** | Clouds/fog on unrevealed tiles | Yes (`Tribe.obsGrid`) | Yes (Masked to `7`) | Yes (121 floats, value `7.0` + Python fog mask) | Full | `PARITY` | — | `Board.java:203`, `register_env.py:5165` | Unrevealed tiles covered by cloud fog graphics (`polytopia.io/gameplay`). |
| **Visible Resources** | Visible on revealed tiles if tech researched | Yes (`Board.resources`) | Yes (`board.resource`, masked) | Yes (121 floats, normalized $[0..1]$, fog-masked) | Full | `PARITY` | `ENGINE_IMPLEMENTATION_DIFFERENCE` | `Board.java:198,251-278`, `register_env.py:5220` | Visible resource sprites on tile. **Engine Difference:** `Board.maskResource()` masks Crops without Organization, Ore without Climbing, Whales without Fishing. Animal/Fruit/Fish are always visible out of fog. |
| **Neutral Villages** | Distinct village huts on unowned tiles | Yes (`TERRAIN.VILLAGE`) | Yes (`terrain == 4`) | Yes (Value `4.0` in terrain plane) | Full | `PARITY` | — | `Types.java:152`, `register_env.py:5165` | Rendered with distinct neutral village hut sprite (`polytopia.io/gameplay`). |
| **Ruins** | Stone ruin structure on tile | Yes (`RESOURCE.RUINS`) | Yes (`resource == 7`) | Yes (Value `1.0` in resource plane) | Full | `PARITY` | — | `Types.java:178`, `register_env.py:5220` | Rendered as ancient stone ruins sprite (`polytopia.io/gameplay`). |
| **City Tile Ownership** | Colored territorial borders | Yes (`board.tileCityId`) | Yes (`board.cityID`) | Yes (9 binary spatial channels, fog-masked) | Full | `PARITY` | — | `Board.java:200`, `register_env.py:5190` | Colored borders mark city territory. Mapped to deterministic city slot channels (Resolved PARITY-002). |
| **Building Placement & Type** | Visible structures on tiles | Yes (`board.buildings`) | Yes (`board.building`) | Yes (19 binary spatial channels, fog-masked) | Full | `PARITY` | — | `PythonEnv.java:345`, `register_env.py:5215` | Lumber Huts, Sawmills, Ports rendered on board. 19 categorical spatial channels matching `SUPPORTED_BUILDINGS` (Resolved PARITY-002). |
| **Road Grid** | Visible road segments | Yes (`board.isRoad()`) | Yes (`board.road`) | Yes (1 binary spatial channel, fog-masked) | Full | `PARITY` | — | `Board.java:1010`, `register_env.py:5205` | Cobblestone/dirt paths rendered between tiles (Resolved PARITY-002). |
| **City Connection Indicator** | Connection halo / network popup | Inferred from trade network | Inferred | Yes (Represented via road grid) | Full | `PARITY` | `TASK_DIFFERENCE` | `TradeNetwork.java:77`, `register_env.py:5205` | Road connectivity visible on road plane; BuildRoad action filtered in Phase 1. |
| **Internal Trade Network (`networkTiles`)** | N/A (Internal engine graph) | Yes (`TradeNetwork`) | Yes (`board.network`) | Dropped | None | `PARITY` | `ENGINE_IMPLEMENTATION_DIFFERENCE` | `TradeNetwork.java:21`, `PythonEnv.java:346` | Internal engine boolean array. Not a human UI primitive; dropping it is not an AI deficit. |
| **Capital Identity** | Star crown emblem on city banner | Yes (`City.isCapital`) | Yes (`city.isCapital`, `tribe.capitalID`) | Yes (City slot feature index 9) | Full | `PARITY` | — | `City.java:381`, `register_env.py:5250` | Crown/star emblem on capital city banner (Resolved PARITY-002). |
| **City Walls** | Stone wall around city center | Yes (`City.hasWalls`) | Yes (`city.hasWalls`) | N/A | None | `PARITY` | `TASK_DIFFERENCE` | `City.java:407`, `register_env.py:2739` | `CITY_WALL` choice stripped from LevelUp action filter (Workshop forced); strictly unreachable in Phase 1. |

---

### 4.2 City Entity State

> **Authoritative City Unit Capacity Rule:**  
> In PolyVision's engine (`core.actors.City.canAddUnit()`, `pol_env/Tribes/src/core/actors/City.java:295-297`):
> ```java
> public boolean canAddUnit(){
>     return unitsID.size() < (level + 1);
> }
> ```
> Therefore, **$\text{City Unit Capacity} = \text{City Level} + 1$**, and **$\text{Remaining Unit Capacity} = (\text{Level} + 1) - \text{Current Supported Units}$**.  
> This aligns with standard Polytopia rules (*Official Polytopia Rulebook & Wiki*): Level 1 cities support 2 units; Level 2 cities support 3 units; Level $L$ cities support $L + 1$ units.

| Information Element | Human UI Access | Engine State Has It? | POV JSON Serializes It? | PPO State Observation (v5) | Whole-Policy Visibility | Primary Class | Secondary Flags | Evidence / Code Source | Human UI Evidence & Notes |
|---|---|:---:|:---:|---|---|:---:|---|---|---|
| **City Position** | Visible on map | Yes (`City.position`) | Yes (`city.x`, `city.y`) | Yes (City slot features 1 & 2 + territory plane) | Spatial grid & slots | `PARITY` | — | `City.java:52`, `environment_contract.py:166` | Normalized $(x/(W-1), y/(H-1))$ in slot features 1 & 2 (Resolved PARITY-001). |
| **City Level (Per City)** | Displayed above city bar (Lv 1, 2..) | Yes (`City.level`) | Yes (`city.level`) | Yes (City slot feature 3, exact unclipped float) | Exact per-city slot | `PARITY` | — | `City.java:372`, `environment_contract.py:168` | Number shown above city name banner (Resolved PARITY-001). |
| **Current Population (Per City)** | Filled dots in city status bar | Yes (`City.population`) | Yes (`city.population`) | Yes (City slot feature 4, exact unclipped float) | Exact per-city slot | `PARITY` | — | `City.java:376`, `environment_contract.py:169` | Blue/cyan dots in city status bar (Resolved PARITY-001). |
| **Population Needed (Per City)** | Total dots in city bar ($= L + 1$) | Yes (`City.population_need`) | Yes (`city.population_need`) | Yes (City slot feature 5, exact unclipped float) | Exact per-city slot | `PARITY` | — | `City.java:390`, `environment_contract.py:170` | Total dot pips in city status bar (Resolved PARITY-001). |
| **Remaining Population to Level** | Unfilled dots in city bar | Derived: `need - pop` | Derived | Exact ($\text{need} - \text{pop}$) | Exact per-city slot | `PARITY` | — | `City.java:376,390`, `environment_contract.py:171` | Visual empty pips in city status bar (Resolved PARITY-001). |
| **City Production / SPT (Per City)** | Displayed on banner ($+X$ stars) | Yes (`City.production`) | Yes (`city.production`) | Yes (City slot feature 6, exact unclipped float) | Exact per-city slot | `PARITY` | — | `City.java:28,391`, `environment_contract.py:172` | Direct $+X$ display on city banner (Resolved PARITY-001). |
| **City Unit Capacity** | Total unit pips under banner ($= L + 1$) | Inferred (`level + 1`) | Inferred from level | Yes (City slot feature 8, exact unclipped float) | Exact per-city slot | `PARITY` | — | `City.java:296,372`, `environment_contract.py:174` | Dot pips displayed beneath city banner (Resolved PARITY-001). |
| **Supported Units Count** | Filled unit pips under banner | Yes (`unitsID.size()`) | Yes (`city.units`) | Yes (City slot feature 7, exact unclipped float) | Exact per-city slot | `PARITY` | — | `City.java:418-425`, `environment_contract.py:173` | Filled circular pips beneath city banner (Resolved PARITY-001). |
| **Remaining Unit Capacity** | Empty unit pips under banner | Derived: $(L+1) - N_{\text{units}}$ | Inferred | Exact $(\text{cap} - \text{units})$ | Exact per-city slot | `PARITY` | — | `City.java:296`, `environment_contract.py:175` | Empty circular pips beneath city banner (Resolved PARITY-001). |
| **Level-Up Readiness (Per City)** | Glowing city banner / Level-up prompt | Yes (`pop >= need`) | Derived | Exact ($\text{pop} \ge \text{need}$) | Exact per-city slot | `PARITY` | — | `register_env.py:5051`, `environment_contract.py:160` | Flashing banner & level-up prompt activate (Resolved PARITY-001). |
| **City Buildings List** | Inspectable on map / city view | Yes (`City.buildings`) | Yes (`city.buildings`) | Yes (19 spatial building channels) | Full 2D map | `PARITY` | — | `City.java:431`, `register_env.py:5215` | Visible structures within borders (Resolved PARITY-002). |

---

### 4.3 Unit Entity State

| Information Element | Human UI Access | Engine State Has It? | POV JSON Serializes It? | PPO State Observation (v5) | Whole-Policy Visibility | Primary Class | Secondary Flags | Evidence / Code Source | Human UI Evidence & Notes |
|---|---|:---:|:---:|---|---|:---:|---|---|---|
| **Unit Spatial Position** | Visible on map | Yes (`Unit.position`) | Yes (`unit.x`, `unit.y`) | Yes (12 unit-type spatial channels, fog-masked) | Full 2D map | `PARITY` | — | `Unit.java:37`, `PythonEnv.java:369`, `register_env.py:5180` | Unit rendered on tile (Resolved PARITY-002). |
| **Unit Type** | Distinct sprite | Yes (`Unit.getType()`) | Yes (`unit.type`) | Yes (12 binary spatial channels matching `SUPPORTED_UNIT_TYPES`) | Full 2D map | `PARITY` | — | `Unit.java:85`, `PythonEnv.java:361`, `register_env.py:5180` | Sprite clearly shows Warrior, Rider, Archer, etc. (Resolved PARITY-002). |
| **Unit Tribe / Owner** | Tribe color / border | Yes (`Unit.tribeId`) | Yes (`unit.tribeId`) | Invariant (Tribe 0) | Full | `PARITY` | `DORMANT_PHASE1` | `Unit.java:41`, `PythonEnv.java:374` | Unit color scheme indicates owner. In Phase 1 solo mode, all units belong to Tribe 0. |
| **Unit Current & Max HP** | Health bar / number over unit | Yes (`Unit.currentHP`, `maxHP`) | Yes (`unit.currentHP`) | Invariant (10/10 HP) | Full | `PARITY` | `DORMANT_PHASE1` | `Unit.java:51-53`, `PythonEnv.java:375` | In Phase 1 solo mode, `ATTACK` is excluded; units spawn at 10/10 HP and take 0 damage. Strictly invariant in Phase 1. |
| **Veteran Status & Kills** | Crown badge / kill count | Yes (`Unit.isVeteran`, `kills`) | Yes (`unit.isVeteran`, `kill`) | Invariant (0 kills, non-veteran) | Full | `PARITY` | `DORMANT_PHASE1` | `Unit.java:60-71`, `PythonEnv.java:371` | In Phase 1 solo mode, units cannot fight or earn kills. Strictly invariant in Phase 1. |
| **Home City ID** | Selecting unit highlights home city | Yes (`Unit.cityId`) | Yes (`unit.cityID`) | Yes (9 binary spatial channels `unit_home_city_slot_0..8`, fog-masked) | Full 2D map | `PARITY` | — | `Unit.java:79`, `PythonEnv.java:373`, `register_env.py:5198` | Direct 9-channel spatial mapping from unit position to deterministic home-city slot (Resolved PARITY-002.1). |
| **Unit Turn Status (Fresh/Moved)** | Dimmed sprite if moved; bright if fresh | Yes (`Unit.status`) | Yes (`unit.status`) | Complete via legal MOVE candidates | Complete via legal MOVE candidates | `PARITY` | `INDIRECT_LEGALITY_SIGNAL` | `Unit.java:87`, `PythonEnv.java:420`, `MoveFactory.java:30` | MoveFactory only generates moves for FRESH units; legal MOVE actions uniquely identify fresh units. |
| **Raw Actor IDs (`unitID`, `cityID`)** | **Invisible to Human** | Yes (`Actor.actorId`) | Yes (`board.unitID`, `cityID`) | **NO (Eliminated)** | None | `PARITY` | — | `Board.java:887`, `register_env.py:5150-5260` | Completely removed from observation tensor in v5 (Resolved PARITY-002). |

---

### 4.4 Technology State Audit

| Information Element | Human UI Access | Engine State Has It? | POV JSON Serializes It? | PPO State Observation (v5) | Whole-Policy Visibility | Primary Class | Secondary Flags | Evidence / Code Source | Human UI Evidence & Notes |
|---|---|:---:|:---:|---|---|:---:|---|---|---|
| **Researched Tech Vector (Full 24)** | Full 24-tech directed tree | Yes (`TechnologyTree.isResearched`) | Reconstructed in Python wrapper | Yes (24 binary features `tech_vector_start..tech_vector_end`) | Full tech vector | `PARITY` | — | `TechnologyTree.java:55`, `register_env.py:5235` | Full 24 technologies in `TECHNOLOGY_ORDER` enum sequence (Resolved PARITY-002). |
| **Dynamic Research Cost** | Star cost displayed on tech node | Yes (`tech.getCost()`) | Yes (`star_cost` in action candidates) | Yes (Action feature index 42: `star_cost / 50.0`) | Per-action feature row | `PARITY` | — | `Types.java:72-80`, `register_env.py:2285` | Authoritative Java formula $4 + \text{tier} \times N_{\text{cities}}$ exposed in action features (Resolved PARITY-002). |
| **Tech Affordability** | Enabled vs greyed tech node | Yes (`stars >= cost`) | Inferred | Yes (Masked by action legality) | Exact legality mask | `PARITY` | `INDIRECT_LEGALITY_SIGNAL` | `ResearchTech.java:30` | Disabled if insufficient stars. Affordability is fully signaled by legality. |

---

## 5. Layer B: Legal-Action Generation & Side-Channel Audit

### 5.1 Action Pipeline & Curriculum Filters

```
[Authoritative GameState] ──> [Raw Java Legal Actions] ──> [_filter_allowed_raw_indices()] ──> [GlobalActionCatalog] ──> [PPO Legal Slots (256)]
```

| Action Filter / Rule | Action Families | Human Allowed? | PPO Allowed? | Primary Class | Secondary Flags | Code Source | Rationale & Effect |
|---|---|:---:|:---:|:---:|---|---|---|
| **Phase 1 Allowlist** | ATTACK, DISBAND, CONVERT, HEAL, BUILD_ROAD | Yes | **NO** | `TASK_DIFFERENCE` | — | `register_env.py:196` | Phase 1 evaluates solo economic expansion without military combat. |
| **City Wall Level-Up Filter** | LEVEL_UP (`CITY_WALL`) | Yes | **NO** | `TASK_DIFFERENCE` | — | `register_env.py:2739-2742` | Forces economic level-up choices (Workshop). |
| **Drylands Fishing Filter** | RESEARCH_TECH (`FISHING`) | Yes | **NO** | `TASK_DIFFERENCE` | — | `register_env.py:2744-2747` | Prevents wasting stars on water tech on dry maps. |
| **Pre-2-City Village Capture Priority** | MOVE, SPAWN | Yes | **NO** | `TASK_DIFFERENCE` | — | `register_env.py:2772-2813` | If village capture is legal, freezes other moves to prioritize expansion. |
| **Pre-2-City Visible Village Move Priority** | Non-village MOVE | Yes | **NO** | `TASK_DIFFERENCE` | — | `register_env.py:2821-2865` | Forces units to advance toward visible neutral villages. |
| **T1/T2 Unit 2 Backtrack Mask** | Early MOVE | Yes | **NO** | `TASK_DIFFERENCE` | — | `register_env.py:2873-2916` | Prevents starting warrior from immediate oscillation. |

---

### 5.2 Legal-Action Hidden-State Dependence & Human-Interface Equivalence

> **The Human-Interface Equivalence Invariant:**  
> If two game states $S_1$ and $S_2$ produce identical complete human-visible interfaces for player $i$, then they must also produce identical complete AI-visible policy interfaces:
> $$\text{HumanInterface}(S_1, \text{player}_i) = \text{HumanInterface}(S_2, \text{player}_i) \implies \text{PolicyInterface}(S_1, \text{player}_i) = \text{PolicyInterface}(S_2, \text{player}_i)$$

1. **Movement Pathfinding & Zone of Control (`StepMove.java:64-70`):** Checks 8-neighborhood tiles around candidate destinations. In Phase 1 solo mode, no opponent tribes exist; thus, this code path is strictly dormant and invariant.
2. **Attack Target Search (`AttackFactory.java:28-37`):** Neutralized in Phase 1 because `ATTACK` is excluded from the action allowlist.
3. **Destination Tile Visibility (`StepMove.java:73-75`):** Explicitly checks `gs.getTribe(unit.getTribeId()).isVisible(tile.x, tile.y)`. Ground units cannot move directly into fog.

---

## 6. Layer C: Legal-Action Feature Audit

The 47-dimensional legal-action feature vector (`v1_4_parity_spatial_and_cost`) is computed in `register_env.py:2255-2385`.

### 6.1 Exhaustive 47-Feature Enumeration

| Idx | Feature Name | Action Family | Prov. Source | P-Level | Primary Class | Underlying Data Sources | Parity Status |
|:---:|---|---|:---:|:---:|:---:|---|:---:|
| **0** | `is_move` | All | `H0` | `P1` | `PARITY` | `action.type == "MOVE"` | `PARITY` |
| **1** | `newly_revealed_tiles_if_move_norm` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | Raycast fog reveal from dst / 12.0 | `PARITY` (P2) |
| **2** | `adjacent_fog_count_after_move_norm` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | Fog count in 8-adj of dst / 8.0 | `PARITY` (P2) |
| **3** | `adjacent_fog_delta_norm` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | `(adj_fog(dst) - adj_fog(src)) / 8.0` | `PARITY` (P2) |
| **4** | `is_zero_reveal_move` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | `revealed == 0` | `PARITY` (P2) |
| **5** | `target_contains_visible_uncaptured_village` | MOVE | `H0` | `P0/P1` | `PARITY` | `dst in visible_villages` | `PARITY` |
| **6** | `has_visible_uncaptured_village` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | `len(visible_villages) > 0` | `PARITY` (P2) |
| **7** | `distance_delta_to_nearest_visible_uncaptured_village_norm` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | $\Delta \text{ManhattanDist}(V)$ normalized | `PARITY` (P2) |
| **8** | `is_immediate_backtrack` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | `dst == unit_previous_tile` | `PARITY` (P2) |
| **9** | `target_inside_owned_city_bounds` | MOVE | `H0` | `P0/P1` | `PARITY` | `tileCityId[dst] == owned_city` | `PARITY` |
| **10** | `distance_from_capital_delta_norm` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | $\Delta \text{ManhattanDist}(\text{Capital})$ normalized | `PARITY` (P2) |
| **11** | `unit_type_warrior` | MOVE | `H0` | `P0/P1` | `PARITY` | `unit.type == WARRIOR` | `PARITY` |
| **12** | `is_end_turn` | All | `H0` | `P1` | `PARITY` | `action.type == "END_TURN"` | `PARITY` |
| **13** | `is_capture` | All | `H0` | `P1` | `PARITY` | `action.type == "CAPTURE"` | `PARITY` |
| **14** | `is_train_or_spawn` | All | `H0` | `P1` | `PARITY` | `action.type in ("SPAWN", "TRAIN")` | `PARITY` |
| **15** | `is_research` | All | `H0` | `P1` | `PARITY` | `action.type == "RESEARCH_TECH"` | `PARITY` |
| **16** | `is_resource_gathering` | All | `H0` | `P1` | `PARITY` | `action.type == "RESOURCE_GATHERING"` | `PARITY` |
| **17** | `is_level_up` | All | `H0` | `P1` | `PARITY` | `action.type == "LEVEL_UP"` | `PARITY` |
| **18** | `is_build` | All | `H0` | `P1` | `PARITY` | `action.type == "BUILD"` | `PARITY` |
| **19** | `is_clear_forest` | All | `H0` | `P1` | `PARITY` | `action.type == "CLEAR_FOREST"` | `PARITY` |
| **20** | `is_grow_forest` | All | `H0` | `P1` | `PARITY` | `action.type == "GROW_FOREST"` | `PARITY` |
| **21** | `is_other` | All | `H0` | `P1` | `PARITY` | Fallback type indicator | `PARITY` |
| **22** | `research_tech_id_norm` | RESEARCH | `H0` | `P1` | `PARITY` | Normalized catalog tech index | `PARITY` |
| **23** | `research_is_organization` | RESEARCH | `H0` | `P0/P1` | `PARITY` | `tech == "ORGANIZATION"` | `PARITY` |
| **24** | `research_is_forestry` | RESEARCH | `H0` | `P0/P1` | `PARITY` | `tech == "FORESTRY"` | `PARITY` |
| **25** | `resource_id_norm` | RESOURCE | `H0` | `P1` | `PARITY` | Normalized catalog resource index | `PARITY` |
| **26** | `resource_is_animal` | RESOURCE | `H0` | `P0/P1` | `PARITY` | `resource == "ANIMAL"` | `PARITY` |
| **27** | `resource_is_fruit` | RESOURCE | `H0` | `P0/P1` | `PARITY` | `resource == "FRUIT"` | `PARITY` |
| **28** | `resource_is_fish` | RESOURCE | `H0` | `P0/P1` | `PARITY` | `resource == "FISH"` | `PARITY` |
| **29** | `resource_is_crop` | RESOURCE | `H0` | `P0/P1` | `PARITY` | `resource == "CROPS"` | `PARITY` |
| **30** | `resource_is_metal` | RESOURCE | `H0` | `P0/P1` | `PARITY` | `resource == "ORE"` | `PARITY` |
| **31** | `build_id_norm` | BUILD | `H0` | `P1` | `PARITY` | Normalized catalog building index | `PARITY` |
| **32** | `build_is_lumber_hut` | BUILD | `H0` | `P0/P1` | `PARITY` | `building == "LUMBER_HUT"` | `PARITY` |
| **33** | `build_is_sawmill` | BUILD | `H0` | `P0/P1` | `PARITY` | `building == "SAWMILL"` | `PARITY` |
| **34** | `levelup_choice_id_norm` | LEVEL_UP | `H0` | `P1` | `PARITY` | Normalized catalog level-up index | `PARITY` |
| **35** | `levelup_is_workshop` | LEVEL_UP | `H0` | `P0/P1` | `PARITY` | `choice == "WORKSHOP"` | `PARITY` |
| **36** | `expected_population_delta_norm` | ECON/BUILD | `H2` | `P2` | `DERIVED_ADVANTAGE` | Precomputed pop yield / 2.0 | `PARITY` (P2) |
| **37** | `expected_immediate_spt_delta_norm` | ECON/BUILD | `H2` | `P2` | `DERIVED_ADVANTAGE` | Precomputed immediate SPT yield / 5.0 | `PARITY` (P2) |
| **38** | `makes_level_up_available` | ECON/BUILD | `H2` | `P2` | `DERIVED_ADVANTAGE` | `pop + \Delta pop >= need` | `PARITY` (P2) |
| **39** | `is_level_up_claim` | LEVEL_UP | `H0` | `P0/P1` | `PARITY` | `action.type == "LEVEL_UP"` | `PARITY` |
| **40** | `action_city_upgrade_progress_before_norm` | ECON/BUILD | `H2` | `P2` | `DERIVED_ADVANTAGE` | Pre-action city progress fraction | `PARITY` (P2) |
| **41** | `action_city_upgrade_ready_before` | ECON/BUILD | `H2` | `P2` | `DERIVED_ADVANTAGE` | `pop >= need` for target city | `PARITY` (P2) |
| **42** | `action_star_cost_norm` | All | `H1/H2` | `P0/P1` | `PARITY` | Authoritative Java `star_cost / 50.0` | `PARITY` (Resolved PARITY-002) |
| **43** | `action_src_x_norm` | Spatial | `H0` | `P0/P1` | `PARITY` | `src_x / (width - 1.0)` | `PARITY` (Resolved PARITY-002) |
| **44** | `action_src_y_norm` | Spatial | `H0` | `P0/P1` | `PARITY` | `src_y / (height - 1.0)` | `PARITY` (Resolved PARITY-002) |
| **45** | `action_target_x_norm` | Spatial | `H0` | `P0/P1` | `PARITY` | `target_x / (width - 1.0)` | `PARITY` (Resolved PARITY-002) |
| **46** | `action_target_y_norm` | Spatial | `H0` | `P0/P1` | `PARITY` | `target_y / (height - 1.0)` | `PARITY` (Resolved PARITY-002) |

---

## 7. Layer D: Public Game Knowledge Audit

### 7.1 Public Game Knowledge Parity Policy

| Game Rule Domain | Nature of Knowledge | PolyVision PPO Representation (v5) | Parity Assessment | Parity Status |
|---|---|---|:---:|:---:|
| **Unit Base Statistics (ATK, DEF, MOV, RANGE)** | Static Invariant | Implicit in neural network weights | `PARITY` | `PARITY` |
| **Unit Purchase Costs** | Static Invariant | Explicit in action feature 42 (`star_cost / 50.0`) | `PARITY` | `PARITY` |
| **Building Costs & Requirements** | Static Invariant | Explicit in action feature 42 (`star_cost / 50.0`) | `PARITY` | `PARITY` |
| **Building Economic Yields** | Static Rule ($+1$ pop, $+1$ SPT) | Precomputed in action features 36–37 | `DERIVED_ADVANTAGE` | `PARITY` (P2) |
| **Tech Tree Directed Graph** | Static Invariant | Implicit in weights; enforced by Java legal mask | `PARITY` | `PARITY` |
| **Dynamic Tech Cost Formula** | Dynamic ($4 + \text{tier} \times N_{\text{cities}}$) | Explicit in action feature 42 (`star_cost / 50.0`) | `PARITY` | `PARITY` |
| **City Unit Capacity Rule** | Dynamic Rule ($N_{\text{units}} < L + 1$) | Explicit in city slot features 7 & 8 | `PARITY` | `PARITY` |
| **City Level-Up Formula** | Static Rule ($\text{need} = L + 1$) | Explicit in city slot features 4 & 5 | `PARITY` | `PARITY` |
| **Terrain Movement Costs & Roads** | Static Rule | Explicit on road plane; pathfinding enforced | `PARITY` | `PARITY` |
| **Sight & Vision Mechanics** | Static Rule (1-tile, Mt=2) | Precomputed in action features 1–4 | `DERIVED_ADVANTAGE` | `PARITY` (P2) |

---

## 8. Whole-Policy Cross-Layer Reconciliation

| Information Element | Present in State (6,424)? | Encoded in Action Identity? | Present in Action Features (47)? | Whole Policy Possesses It? | Human UI Has It? | Final Whole-Policy Parity Status | Cross-Layer Reconciliation Summary |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Terrain Layout** | Yes (121) | Implicit | Implicit | **YES** | Yes | `PARITY` | Fully visible in state vector. |
| **Fog of War** | Yes (121) | Implicit | Feat 1–4 | **YES** | Yes | `PARITY` | Fully visible in state vector with Python fog mask defense-in-depth. |
| **Visible Resources** | Yes (121) | Yes (Res IDs) | Feat 25–30 | **YES** | Yes | `PARITY` | Fully visible in state vector. |
| **Building Placement & Type** | Yes (19 planes = 2,299) | Yes (Build IDs) | Feat 31–33 | **YES** | Yes | `PARITY` | Full 19-channel categorical spatial grid (Resolved PARITY-002). |
| **Road Network** | Yes (121) | Implicit | Implicit | **YES** | Yes | `PARITY` | Full binary spatial road plane (Resolved PARITY-002). |
| **Capital Identity** | Yes (Slot feature 9) | Implicit | Feat 10 ($\Delta \text{Cap}$) | **YES** | Yes | `PARITY` | Explicit capital flag per city slot (Resolved PARITY-002). |
| **Per-City Level** | Yes (Slot feature 3) | Implicit | Implicit | **YES** | Yes | `PARITY` | Exact level per owned city slot (Resolved PARITY-001). |
| **Per-City Population & Need** | Yes (Slot features 4, 5) | Implicit | Feat 40, 41 | **YES** | Yes | `PARITY` | Exact population and need per city slot (Resolved PARITY-001). |
| **City Unit Capacity & Count** | Yes (Slot features 7, 8) | Implicit | Implicit | **YES** | Yes | `PARITY` | Exact supported units and capacity per city slot (Resolved PARITY-001). |
| **Unit Type & Location** | Yes (12 planes = 1,452) | Implicit | Feat 11, 43–46 | **YES** | Yes | `PARITY` | Full 12-channel categorical unit-type spatial grid (Resolved PARITY-002). |
| **Unit Home-City Association** | Yes (9 planes = 1,089) | Implicit | Implicit | **YES** | Yes | `PARITY` | 9-channel spatial mapping to deterministic city slot (Resolved PARITY-002.1). |
| **Unit Current & Max HP** | Invariant (10/10 HP) | Implicit | Implicit | **YES** | Yes | `PARITY` | Invariant in Phase 1 (no combat). Deferred for full-game combat. |
| **Unit Turn Status** | Signaled by MOVE legality | Yes (MOVE mask) | Implicit | **YES** | Yes | `PARITY` | MoveFactory only generates moves for FRESH units; complete signal. |
| **Researched Techs (Full 24)** | Yes (24 binary flags) | Implicit | Feat 22–24 | **YES** | Yes | `PARITY` | Full 24-element binary vector in `TECHNOLOGY_ORDER` (Resolved PARITY-002). |
| **Action Star Cost** | Implicit in Stars | Implicit | Yes (Feat 42) | **YES** | Yes | `PARITY` | Normalized authoritative star cost in feature row (Resolved PARITY-002). |
| **Action Spatial Coordinates** | Spatial planes | Yes (Global ID) | Yes (Feat 43–46) | **YES** | Yes | `PARITY` | Explicit normalized source and destination coordinates (Resolved PARITY-002). |

---

## 9. Consolidated Disparity Register

| Disparity ID | Layer | Information / Behavior | Human Access | AI Access | Primary Class | Phase-1 Severity | Full-Game Severity | Current Status |
|:---:|:---:|---|---|---|:---:|:---:|:---:|:---:|
| `STATE-CITY-001` | State | Per-City Level | Displayed over city | Exact per-city slot 3 | `PARITY` | **Critical** | **Critical** | **RESOLVED (PARITY-001)** |
| `STATE-CITY-002` | State | Per-City Population & Need | Displayed in status bar | Exact per-city slots 4 & 5 | `PARITY` | **Critical** | **Critical** | **RESOLVED (PARITY-001)** |
| `STATE-CITY-003` | State | Per-City SPT Contribution | Displayed on banner | Exact per-city slot 6 | `PARITY` | **High** | **High** | **RESOLVED (PARITY-001)** |
| `STATE-CITY-004` | State | City Unit Count & Capacity | Displayed pips under city | Exact per-city slots 7 & 8 | `PARITY` | **Critical** | **Critical** | **RESOLVED (PARITY-001)** |
| `STATE-MAP-001` | State | Building Placement & Type | Inspectable on board | 19 categorical spatial channels | `PARITY` | **High** | **High** | **RESOLVED (PARITY-002)** |
| `STATE-MAP-002` | State | Road Grid on Map | Inspectable on board | Binary road spatial channel | `PARITY` | **Medium** | **High** | **RESOLVED (PARITY-002)** |
| `STATE-MAP-003` | State | Capital City Identity | Star icon on city banner | Explicit `is_capital` in slot 9 | `PARITY` | **Medium** | **High** | **RESOLVED (PARITY-002)** |
| `STATE-UNIT-001` | State | Unit Type on Map | Distinct sprite | 12 categorical unit spatial channels | `PARITY` | **High** | **Critical** | **RESOLVED (PARITY-002)** |
| `STATE-UNIT-002` | State | Unit Current & Max HP | Health bar over unit | Invariant 10/10 HP in solo Phase 1 | `PARITY` | **Low (Solo)** | **Critical** | **PARITY / INVARIANT (Phase 1)** |
| `STATE-UNIT-003` | State | Unit Home-City Association | Pulse city dot on unit select | 9 unit home-city spatial channels | `PARITY` | **Medium** | **High** | **RESOLVED (PARITY-002.1)** |
| `STATE-ID-001` | State | Raw Actor IDs as Floats | Invisible / N/A | Completely removed from tensor | `PARITY` | **High** | **High** | **RESOLVED (PARITY-002)** |
| `STATE-TECH-001` | State | Full Researched Tech Tree | Full 24-tech UI tree | Full 24-element binary vector | `PARITY` | **High** | **High** | **RESOLVED (PARITY-002)** |
| `FEAT-MISS-001` | Features | Action Star Cost | Displayed on buttons | Normalized star cost (idx 42) | `PARITY` | **High** | **High** | **RESOLVED (PARITY-002)** |
| `FEAT-SPAT-001` | Features | Action Spatial Coordinates | Spatial grid click | Normalized src/dst coords (43..46) | `PARITY` | **Medium** | **Medium** | **RESOLVED (PARITY-002)** |
| `ACTION-SIDE-001` | Legality | Zone of Control Fog Leak | Hidden by fog | Invariant (No enemies in Phase 1) | `PARITY` | **Dormant** | **High** | **DORMANT / INVARIANT (Phase 1)** |
| `ACTION-SIDE-002` | Legality | Attack Target Fog Leak | Hidden by fog | Neutralized (ATTACK filtered) | `PARITY` | **Dormant** | **Critical** | **DORMANT / INVARIANT (Phase 1)** |
| `ACTION-FILT-001` | Legality | Pre-2-City Expansion Filters | Full player freedom | Heuristically forced to village | `TASK_DIFFERENCE` | **Medium** | **N/A** | **RESOLVED (Curriculum Rule)** |

---

## 10. Diagnostic Verification Protocols

### Protocol 1: Actor ID Invariance Verification (`STATE-ID-001`)
- **Verification Rule:** Permuting raw Java actor IDs across units, cities, and counters produces bitwise identical observation tensors with zero raw ID exposure.
- **Empirical Status:** `VERIFIED` (`pol_env/Tribes/py/tests/test_parity_002_human_information_parity.py:test_03_actor_id_invariance_multi_city_multi_unit`).

### Protocol 2: Unit Turn Status Legality Signal (`STATE-UNIT-STATUS`)
- **Verification Rule:** Sequential step execution transitions unit status from FRESH to MOVED/FINISHED; MoveFactory generates zero MOVE actions for MOVED units.
- **Empirical Status:** `VERIFIED` (`pol_env/Tribes/py/tests/test_parity_002_human_information_parity.py:test_10_unit_turn_status_legality_signal_parity`).

### Protocol 3: Cross-Language Semantic Ordering (`STATE-VOCAB-001`)
- **Verification Rule:** Java `Types.java` enums (`TECHNOLOGY`, `BUILDING`, `UNIT`) match Python constants (`TECHNOLOGY_ORDER`, `SUPPORTED_BUILDINGS`, `SUPPORTED_UNIT_TYPES`) in length, order, and integer keys 1:1.
- **Empirical Status:** `VERIFIED` (`pol_env/Tribes/py/tests/test_parity_002_human_information_parity.py:test_01_cross_language_contract_vocabularies_match_java`).

---

## 11. Prioritized Roadmap & Milestone Summary

| Priority | Disparity Item | Phase-1 Status | Milestone |
|:---:|---|:---:|:---:|
| **P1** | Per-City Level, Pop & Need (`STATE-CITY-001`, `002`) | **CLOSED** | PARITY-001 |
| **P1** | City Unit Count & Capacity (`STATE-CITY-004`) | **CLOSED** | PARITY-001 |
| **P1** | Action Star Cost (`FEAT-MISS-001`) | **CLOSED** | PARITY-002 |
| **P1** | Building Placement & Type (`STATE-MAP-001`) | **CLOSED** | PARITY-002 |
| **P2** | Full 24-Tech Tree Vector (`STATE-TECH-001`) | **CLOSED** | PARITY-002 |
| **P2** | Per-City SPT Contribution (`STATE-CITY-003`) | **CLOSED** | PARITY-001 |
| **P2** | Eliminate Raw Float Actor IDs (`STATE-ID-001`) | **CLOSED** | PARITY-002 |
| **P3** | Capital City Indicator (`STATE-MAP-003`) | **CLOSED** | PARITY-002 |
| **P3** | Unit Type Spatial Channels (`STATE-UNIT-001`) | **CLOSED** | PARITY-002 |
| **P3** | Unit Home-City Association Channels (`STATE-UNIT-003`) | **CLOSED** | PARITY-002.1 |
| **P4** | Multi-Agent Unit Combat HP / Veteran (`STATE-UNIT-002`) | **INVARIANT (Phase 1)** | Deferred for Full-Game Multi-Agent |
| **P4** | Zone of Control Fog Leak (`ACTION-SIDE-001`) | **INVARIANT (Phase 1)** | Deferred for Full-Game Multi-Agent |

---

## 12. Methodological Conclusions

1. **Information Parity Achieved for Phase 1:** All legitimate human-visible game state in Phase 1 solo economic expansion is explicitly, losslessly, and actor-ID-invariantly represented in the policy observation tensor (6,424 floats) and legal-action feature vector (47 floats).
2. **Actor-ID Invariance Enforced:** The observation tensor contains zero raw engine actor IDs, preventing catastrophic generalization failure.
3. **Fog Defense-in-Depth:** Every spatial channel applies an explicit terrain-fog mask in Python, ensuring zero hidden state leaks under fog.
4. **Clean Checkpoint Rejection:** Checkpoints trained on older contracts (`v3`, `v4`) are safely rejected with clear dimensional mismatch diagnostics.

---

## 13. Target Parity Principles for Implementation

1. **Strict Human-Interface Equivalence Invariant:**  
   If $\text{HumanInterface}(S_1, \text{player}) = \text{HumanInterface}(S_2, \text{player})$, then $\text{PolicyInterface}(S_1, \text{player}) = \text{PolicyInterface}(S_2, \text{player})$.
2. **Preserve Entity-Level Granularity:** Maintain individual entity resolution for all owned and visible assets without destructive global pooling.
3. **Eliminate Arbitrary Engine ID Numerics:** Never feed raw monotonic engine counters as continuous scalar inputs.
4. **Explicit Provenance Tracking:** Every policy-visible feature traces back to legitimate human-visible state (H0) or public rules (H1/H2).
5. **Decouple Task Scoping from Information Blinding:** Phase-1 task constraints (e.g. Turn-10 horizon, combat exclusion) are enforced through action availability, never by blinding the policy to visible state.

---

## 14. Audit Closure Status

### 14.1 Resolved Milestones Summary
- **PARITY-001 (Exact Per-City State — Contract `v4_exact_per_city_state`):**  
  Closed `STATE-CITY-001`, `STATE-CITY-002`, `STATE-CITY-003`, and `STATE-CITY-004`.
- **PARITY-002 (Phase-1 Human Information Parity — Contract `v5_human_information_parity`):**  
  Closed `STATE-MAP-001`, `STATE-MAP-002`, `STATE-MAP-003`, `FEAT-MISS-001`, `STATE-TECH-001`, `STATE-ID-001`, `STATE-UNIT-001`, and `FEAT-SPAT-001`.
- **PARITY-002.1 (Audit Reconciliation & Strict Closure Review):**  
  Closed `STATE-UNIT-003` (9-channel unit home-city spatial mapping), hardened cross-language enum tests, dynamic star cost integration tests, and reconciled all audit tables and documentation to 6,424 observation dimensions.

```text
================================================================================
PHASE-1 HUMAN INFORMATION PARITY: COMPLETE & DEFICIT-FREE
Environment contract: v5_human_information_parity
Total observation dimension: 6,424 floats (52 spatial channels x 121 tiles + 132 structured scalars)
Legal action feature dimension: 47 floats
================================================================================
```

---

## 15. Evidence & Provenance Index

### 15.1 Human Interface / Commercial Polytopia Evidence
- **Official Polytopia Website & Gameplay Manual:** [`https://polytopia.io/`](https://polytopia.io/)
- **Polytopia Reference Community Wiki:** [`https://polytopia.fandom.com/wiki/The_Battle_of_Polytopia_Wiki`](https://polytopia.fandom.com/wiki/The_Battle_of_Polytopia_Wiki)

### 15.2 Java Engine Core
- [`pol_env/Tribes/src/core/game/PythonEnv.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/game/PythonEnv.java) — Observation JSON serialization (`observationJsonFromState()`).
- [`pol_env/Tribes/src/core/game/GameState.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/game/GameState.java) — Partial observation deep copy (`copy(int playerIdx)`).
- [`pol_env/Tribes/src/core/game/Board.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/game/Board.java) — Fog masking (`copy()`), resource masking (`maskResource()`), actor ID allocation (`addActor()`).
- [`pol_env/Tribes/src/core/actors/City.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/actors/City.java) — City unit capacity (`canAddUnit()`), population, production, building lists.
- [`pol_env/Tribes/src/core/actors/units/Unit.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/actors/units/Unit.java) — Unit state and partial copy hiding (`hide()`).
- [`pol_env/Tribes/src/core/actors/Tribe.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/actors/Tribe.java) — Tech tree, score, stars, visibility grid (`obsGrid`).
- [`pol_env/Tribes/src/core/TechnologyTree.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/TechnologyTree.java) — Tech tree prerequisites and research state.
- [`pol_env/Tribes/src/core/Types.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/Types.java) — Enums and dynamic tech cost formula (`Types.TECHNOLOGY.getCost()`).

### 15.3 Java Action Factories
- [`pol_env/Tribes/src/core/actions/unitactions/StepMove.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/actions/unitactions/StepMove.java) — Movement pathfinding.
- [`pol_env/Tribes/src/core/actions/unitactions/factory/MoveFactory.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/actions/unitactions/factory/MoveFactory.java) — Move action generation.
- [`pol_env/Tribes/src/core/actions/unitactions/factory/AttackFactory.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/actions/unitactions/factory/AttackFactory.java) — Attack action generation.
- [`pol_env/Tribes/src/core/actions/cityactions/factory/BuildFactory.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/actions/cityactions/factory/BuildFactory.java) — Building action generation.
- [`pol_env/Tribes/src/core/actions/cityactions/factory/ResourceGatheringFactory.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/actions/cityactions/factory/ResourceGatheringFactory.java) — Resource gathering generation.
- [`pol_env/Tribes/src/core/actions/cityactions/factory/SpawnFactory.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/actions/cityactions/factory/SpawnFactory.java) — Unit training generation.

### 15.4 Python Bridge, Wrapper & Tooling
- [`pol_env/Tribes/py/gym_env.py`](file:///c:/PolyVision/pol_env/Tribes/py/gym_env.py) — Py4J bridge, step execution, observation retrieval.
- [`pol_env/Tribes/py/register_env.py`](file:///c:/PolyVision/pol_env/Tribes/py/register_env.py) — `TribesGymWrapper`, observation vectorization (`_dict_to_array()`), curriculum filters (`_filter_allowed_raw_indices()`), action features (`_compute_legal_action_feature_vector_reference()`).
- [`pol_env/Tribes/py/environment_contract.py`](file:///c:/PolyVision/pol_env/Tribes/py/environment_contract.py) — Observation layout dimension definitions and checkpoint contracts.
- [`tools/human_policy_interface.py`](file:///c:/PolyVision/tools/human_policy_interface.py) — Human benchmark presentation layer and action decoding.
- [`tools/human_benchmark.py`](file:///c:/PolyVision/tools/human_benchmark.py) — Persistent benchmark runner.
- [`tools/validate_human_benchmark_parity.py`](file:///c:/PolyVision/tools/validate_human_benchmark_parity.py) — Human/PPO interface parity validator.y.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/actions/cityactions/factory/ResourceGatheringFactory.java) — Resource gathering generation.
- [`pol_env/Tribes/src/core/actions/cityactions/factory/SpawnFactory.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/actions/cityactions/factory/SpawnFactory.java) — Unit training generation.

### 15.4 Python Bridge, Wrapper & Tooling
- [`pol_env/Tribes/py/gym_env.py`](file:///c:/PolyVision/pol_env/Tribes/py/gym_env.py) — Py4J bridge, step execution, observation retrieval.
- [`pol_env/Tribes/py/register_env.py`](file:///c:/PolyVision/pol_env/Tribes/py/register_env.py) — `TribesGymWrapper`, observation vectorization (`_dict_to_array()`), curriculum filters (`_filter_allowed_raw_indices()`), action features (`_compute_legal_action_feature_vector_reference()`).
- [`pol_env/Tribes/py/environment_contract.py`](file:///c:/PolyVision/pol_env/Tribes/py/environment_contract.py) — Observation layout dimension definitions and checkpoint contracts.
- [`tools/human_policy_interface.py`](file:///c:/PolyVision/tools/human_policy_interface.py) — Human benchmark presentation layer and action decoding.
- [`tools/human_benchmark.py`](file:///c:/PolyVision/tools/human_benchmark.py) — Persistent benchmark runner.
- [`tools/validate_human_benchmark_parity.py`](file:///c:/PolyVision/tools/validate_human_benchmark_parity.py) — Human/PPO interface parity validator.
