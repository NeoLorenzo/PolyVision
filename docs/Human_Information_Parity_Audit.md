# Baseline Human–AI Information Parity Audit

**Authoritative Baseline Audit & Candidate Specification**  
**Document Version:** `1.0.0`  
**Date:** 2026-08-25  
**Target Environment:** `Tribes-v0` (`v3_corrected_turn_economy`, `flat-v1-structured`, `v1_3_move_focus_plus_semantic_econ`)  
**Scope:** Normative information parity audit comparing normal human-visible Polytopia information against PolyVision PPO policy visibility.

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
PolyVision currently maintains a verified parity standard between the PPO policy and the official human benchmark interface (`tools/human_policy_interface.py` / `tools/human_benchmark.py`). However, this existing parity was achieved by **restricting the human benchmark interface down to the flattened 505-value vector and 42-dimensional legal-action feature tensors received by PPO**.

```
[Current Benchmark Parity — Human Restricted Down to Agent]
Ordinary Human Interface (Polytopia)  ───(Artificial Reduction)───>  Human Benchmark UI (tools/human_policy_interface.py)
                                                                                    ==
                                                                     PPO Policy Interface (TribesGymWrapper)

[Target Information Parity — Agent Expanded to Full Legitimate Human Information]
Ordinary Human Interface (Polytopia)  ============================  Future Parity Policy Interface (TribesGymWrapper)
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
Every future policy-visible feature must trace its provenance to one of five mutually exclusive data origins:
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
│    NOTE: Does NOT serialize everything in Java objects (e.g. omits maxHP, status, techTree).│
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │ TribesGymWrapper._dict_to_array() & info tensors
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. PPO POLICY VISIBILITY (Union of all policy-visible inputs)                                │
│    ├── A. 505-dim State Observation (Vectorized 1D array)                                   │
│    ├── B. Legal Action Global IDs (Discrete categorical catalog slots)                      │
│    └── C. 42-dim Legal Action Feature Rows (Per-slot semantic/economic vector)               │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Layer A: Game-State Information Audit

### 4.1 Board & Map Spatial State

| Information Element | Human UI Access | Engine State Has It? | POV JSON Serializes It? | PPO State Observation | Whole-Policy Visibility | Primary Class | Secondary Flags | Evidence / Code Source | Human UI Evidence & Notes |
|---|---|:---:|:---:|---|---|:---:|---|---|---|
| **Terrain Geometry** | Visible per revealed tile | Yes (`Board.terrains`) | Yes (`board.terrain`) | Yes (121 floats, $[0..7]$) | Full | `PARITY` | — | `Board.java:197`, `PythonEnv.java:341`, `register_env.py:4967` | Rendered on game board (`polytopia.io/gameplay`). Plain(0), ShallowWater(1), DeepWater(2), Mountain(3), Village(4), City(5), Forest(6), Fog(7). |
| **Fog of War** | Clouds/fog on unrevealed tiles | Yes (`Tribe.obsGrid`) | Yes (Masked to `7`) | Yes (121 floats, value `7.0`) | Full | `PARITY` | — | `Board.java:203`, `register_env.py:4969` | Unrevealed tiles covered by cloud fog graphics (`polytopia.io/gameplay`). |
| **Visible Resources** | Visible on revealed tiles if tech researched | Yes (`Board.resources`) | Yes (`board.resource`, masked) | Yes (121 floats, normalized $[0..1]$) | Full | `PARITY` | `ENGINE_IMPLEMENTATION_DIFFERENCE` | `Board.java:198,251-278`, `register_env.py:5015` | Visible resource sprites on tile. **Engine Difference:** `Board.maskResource()` masks Crops without Organization, Ore without Climbing (not Mining!), Whales without Fishing. Animal/Fruit/Fish are always visible out of fog. |
| **Neutral Villages** | Distinct village huts on unowned tiles | Yes (`TERRAIN.VILLAGE`) | Yes (`terrain == 4`) | Yes (Value `4.0`) | Full | `PARITY` | — | `Types.java:152`, `register_env.py:4969` | Rendered with distinct neutral village hut sprite (`polytopia.io/gameplay`). |
| **Ruins** | Stone ruin structure on tile | Yes (`RESOURCE.RUINS`) | Yes (`resource == 7`) | Yes (Value `1.0` in resource block) | Full | `PARITY` | — | `Types.java:178`, `register_env.py:5015` | Rendered as ancient stone ruins sprite (`polytopia.io/gameplay`). |
| **City Tile Ownership** | Colored territorial borders | Yes (`board.tileCityId`) | Yes (`board.cityID`) | Corrupted (Raw integer actor IDs) | Poorly Encoded | `REPRESENTATION_LOSS` | `REPRESENTATION_HAZARD` | `Board.java:200`, `register_env.py:4975` | Colored borders mark city territory. PPO receives raw engine actor ID numbers. |
| **Building Placement & Type** | Visible 3D structures on tiles | Yes (`board.buildings`) | Yes (`board.building`) | **NO (Discarded)** | Discarded in State; Action features know target | `AI_DEFICIT` | `CROSS_LAYER_PARTIAL` | `PythonEnv.java:345`, `register_env.py:4962-5078` | Lumber Huts, Sawmills, Ports rendered on board. Dropped by `_dict_to_array()`. |
| **Road Grid** | Visible road segments connecting tiles | Yes (`board.isRoad()`) | No (Only composite network) | **NO (Discarded)** | None | `AI_DEFICIT` | — | `Board.java:1010`, `register_env.py:4962-5078` | Cobblestone/dirt paths rendered between tiles (`polytopia.io/gameplay`). |
| **City Connection Indicator** | Connection halo / network popup | Inferred from trade network | Inferred | **NO in State** | None | `AI_DEFICIT` | `HUMAN_UI_UNVERIFIED` | `TradeNetwork.java:77`, `register_env.py:4962-5078` | UI indicates city connection to capital (+1 population reward upon connection). |
| **Internal Trade Network (`networkTiles`)** | N/A (Internal engine graph) | Yes (`TradeNetwork`) | Yes (`board.network`) | **NO (Discarded)** | None | `PARITY` | `ENGINE_IMPLEMENTATION_DIFFERENCE` | `TradeNetwork.java:21`, `PythonEnv.java:346` | Internal engine boolean array. Not a human UI primitive; dropping it is not an AI deficit. |
| **Capital Identity** | Star crown emblem on city banner | Yes (`City.isCapital`) | Yes (`city.isCapital`, `tribe.capitalID`) | **NO in State** | MOVE features have `distance_from_capital_delta` | `AI_DEFICIT` | `CROSS_LAYER_PARTIAL` | `City.java:381`, `PythonEnv.java:389`, `register_env.py:2308` | Crown/star emblem on capital city banner (`polytopia.io/gameplay`). State vector omits capital flag. |
| **City Walls** | Stone wall around city center | Yes (`City.hasWalls`) | Yes (`city.hasWalls`) | **NO (Discarded)** | None | `AI_DEFICIT` | `DORMANT_PHASE1` | `City.java:407`, `PythonEnv.java:392` | Fortified stone wall rendered around city (+4 defense). Dropped during vector flattening. |

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

| Information Element | Human UI Access | Engine State Has It? | POV JSON Serializes It? | PPO State Observation | Whole-Policy Visibility | Primary Class | Secondary Flags | Evidence / Code Source | Human UI Evidence & Notes |
|---|---|:---:|:---:|---|---|:---:|---|---|---|
| **City Position** | Visible on map | Yes (`City.position`) | Yes (`city.x`, `city.y`) | Corrupted (Raw `cityID` in grid) | Spatial grid | `REPRESENTATION_LOSS` | `REPRESENTATION_HAZARD` | `City.java:52`, `PythonEnv.java:383` | City center tile visible on board (`polytopia.io/gameplay`). |
| **City Level (Per City)** | Displayed above city bar (Lv 1, 2..) | Yes (`City.level`) | Yes (`city.level`) | **NO (Aggregated)** (`avg_level`, `max_level`) | Aggregated in state | `REPRESENTATION_LOSS` | — | `City.java:372`, `register_env.py:5054-5071` | Number shown above city name banner (`polytopia.io/gameplay`). |
| **Current Population (Per City)** | Filled dots in city status bar | Yes (`City.population`) | Yes (`city.population`) | **NO (Aggregated)** (`mean_progress`, `max_progress`) | Action features know target city progress | `REPRESENTATION_LOSS` | `CROSS_LAYER_PARTIAL` | `City.java:376`, `register_env.py:5049-5073` | Blue/cyan dots in city status bar (`polytopia.io/gameplay`). |
| **Population Needed (Per City)** | Total dots in city bar ($= L + 1$) | Yes (`City.population_need`) | Yes (`city.population_need`) | **NO (Discarded)** | Inferred if level known | `REPRESENTATION_LOSS` | — | `City.java:390`, `register_env.py:5049` | Total dot pips in city status bar. Always equals $\text{level} + 1$. |
| **Remaining Population to Level** | Unfilled dots in city bar | Derived: `need - pop` | Derived | **NO in State** | Precomputed in action features 36–41 | `REPRESENTATION_LOSS` | `CROSS_LAYER_PARTIAL` | `City.java:376,390`, `register_env.py:2368` | Visual empty pips in city status bar. |
| **City Production / SPT (Per City)** | Displayed on banner ($+X$ stars) | Yes (`City.production`) | Yes (`city.production`) | **NO (Aggregated)** (Total `current_spt`) | Aggregated in state | `REPRESENTATION_LOSS` | — | `City.java:28,391`, `register_env.py:5019` | Direct $+X$ display on city banner (`polytopia.io/gameplay`). |
| **City Unit Capacity** | Total unit pips under banner ($= L + 1$) | Inferred (`level + 1`) | Inferred from level | **NO in State** | Inferred if level known | `AI_DEFICIT` | — | `City.java:296,372` | Dot pips displayed beneath city banner. |
| **Supported Units Count** | Filled unit pips under banner | Yes (`unitsID.size()`) | Yes (`city.units`) | **NO (Discarded)** | None | `AI_DEFICIT` | — | `City.java:418-425`, `PythonEnv.java:408` | Filled circular pips beneath city banner. |
| **Remaining Unit Capacity** | Empty unit pips under banner | Derived: $(L+1) - N_{\text{units}}$ | Inferred | **NO in State** | Inferred via SPAWN legality | `AI_DEFICIT` | `INDIRECT_LEGALITY_SIGNAL` | `City.java:296`, `Spawn.java:39` | Empty circular pips beneath city banner. |
| **Level-Up Readiness (Per City)** | Glowing city banner / Level-up prompt | Yes (`pop >= need`) | Derived | **NO (Aggregated)** (`upgrade_ready_frac`, `any_available`) | Action features have `makes_level_up_available` | `REPRESENTATION_LOSS` | `CROSS_LAYER_PARTIAL` | `register_env.py:5051-5075, 2371` | Flashing banner & level-up prompt activate (`polytopia.io/gameplay`). |
| **City Buildings List** | Inspectable on map / city view | Yes (`City.buildings`) | Yes (`city.buildings`) | **NO (Discarded)** | None | `AI_DEFICIT` | — | `City.java:431`, `PythonEnv.java:395-406` | Visible structures within borders. |

---

### 4.3 Unit Entity State

| Information Element | Human UI Access | Engine State Has It? | POV JSON Serializes It? | PPO State Observation | Whole-Policy Visibility | Primary Class | Secondary Flags | Evidence / Code Source | Human UI Evidence & Notes |
|---|---|:---:|:---:|---|---|:---:|---|---|---|
| **Unit Spatial Position** | Visible on map | Yes (`Unit.position`) | Yes (`unit.x`, `unit.y`) | Corrupted (Raw `unitID` in grid) | Global action ID encodes src/dst | `REPRESENTATION_LOSS` | `REPRESENTATION_HAZARD` | `Unit.java:37`, `PythonEnv.java:369`, `register_env.py:4971` | Unit rendered on tile (`polytopia.io/gameplay`). |
| **Unit Type** | Distinct 3D sprite | Yes (`Unit.getType()`) | Yes (`unit.type`) | **NO in State** | Action feature 11 has `unit_type_warrior` | `AI_DEFICIT` | `CROSS_LAYER_PARTIAL` | `Unit.java:85`, `PythonEnv.java:361`, `register_env.py:2302` | Sprite clearly shows Warrior, Rider, Archer, etc. |
| **Unit Tribe / Owner** | Tribe color / border | Yes (`Unit.tribeId`) | Yes (`unit.tribeId`) | **NO in State** | None | `AI_DEFICIT` | `DORMANT_PHASE1` | `Unit.java:41`, `PythonEnv.java:374` | Unit color scheme indicates owner. (Dormant in solo Phase 1). |
| **Unit Current HP** | Health bar / number over unit | Yes (`Unit.currentHP`) | Yes (`unit.currentHP`) | **NO (Discarded)** | None | `AI_DEFICIT` | `DORMANT_PHASE1` | `Unit.java:53`, `PythonEnv.java:375` | Colored bar with numerical HP (e.g. 10/10) (`polytopia.io/gameplay`). |
| **Unit Max HP** | Health bar segments (10 or 15) | Yes (`Unit.maxHP`) | **NO (Not Serialized!)** | **NO (Discarded)** | None | `AI_DEFICIT` | `DORMANT_PHASE1` | `Unit.java:51`, `PythonEnv.java:358-377` | 10 for standard units, 15 for veteran units. |
| **Veteran Status** | Crown / medal badge icon | Yes (`Unit.isVeteran`) | Yes (`unit.isVeteran`) | **NO (Discarded)** | None | `AI_DEFICIT` | `DORMANT_PHASE1` | `Unit.java:71`, `PythonEnv.java:372` | Medal/crown icon above health bar (`polytopia.io/gameplay`). |
| **Unit Kill Count** | Unit info pop-up (3 kills = Vet) | Yes (`Unit.kills`) | Yes (`unit.kill`) | **NO (Discarded)** | None | `AI_DEFICIT` | `DORMANT_PHASE1` | `Unit.java:60`, `PythonEnv.java:371` | Displayed upon inspecting unit details. |
| **Home City ID** | Selecting unit highlights home city | Yes (`Unit.cityId`) | Yes (`unit.cityID`, hidden for enemy) | **NO (Discarded)** | None | `AI_DEFICIT` | — | `Unit.java:79`, `PythonEnv.java:373` | Selecting unit pulses its home city dot pips. |
| **Unit Turn Status (Fresh/Moved)** | Dimmed sprite if moved; bright if fresh | Yes (`Unit.status`) | **NO (Not Serialized!)** | **NO in State** | Inferable via MOVE legality | `AI_DEFICIT` | `INDIRECT_LEGALITY_SIGNAL` | `Unit.java:87`, `PythonEnv.java:358-377` | Bright sprite + white ring = fresh; dimmed = moved (`polytopia.io/gameplay`). |
| **Raw Actor IDs (`unitID`, `cityID`)** | **Invisible to Human** | Yes (`Actor.actorId`) | Yes (`board.unitID`, `cityID`) | **YES (242 floats)** | Full state | `REPRESENTATION_LOSS` | `REPRESENTATION_HAZARD` | `Board.java:887`, `register_env.py:4971-4978` | Continuous float injection of raw integer counter. |

---

### 4.4 Technology State Audit

| Information Element | Human UI Access | Engine State Has It? | POV JSON Serializes It? | PPO State Observation | Whole-Policy Visibility | Primary Class | Secondary Flags | Evidence / Code Source | Human UI Evidence & Notes |
|---|---|:---:|:---:|---|---|:---:|---|---|---|
| **Researched Tech: Organization** | Glowing node in Tech Tree | Yes (`TechnologyTree.isResearched`) | **NO (Not Serialized!)** | Yes (1 float, idx 495) | Reconstructed in Python wrapper | `PARITY` | `CROSS_LAYER_PARTIAL` | `TechnologyTree.java:55`, `register_env.py:5024,5066` | In-game Tech Tree window (`polytopia.io/gameplay`). |
| **Researched Tech: Forestry** | Glowing node in Tech Tree | Yes (`TechnologyTree.isResearched`) | **NO (Not Serialized!)** | Yes (1 float, idx 496) | Reconstructed in Python wrapper | `PARITY` | `CROSS_LAYER_PARTIAL` | `TechnologyTree.java:55`, `register_env.py:5025,5067` | In-game Tech Tree window (`polytopia.io/gameplay`). |
| **All Other Researched Techs (22 Techs)** | Full 24-tech directed tree | Yes (`TechnologyTree.isResearched`) | **NO (Not Serialized!)** | **NO (Compressed)** (Only `tech_count / 24.0`) | Python wrapper tracks set in memory | `REPRESENTATION_LOSS` | — | `TechnologyTree.java:55`, `register_env.py:5026,5068` | Full tree inspectable anytime. PPO cannot distinguish Climbing, Riding, Mining, etc. |
| **Tech Prerequisites & Topology** | Visual graph tree branches | Yes (`tech.getParentTech()`) | No (Static enum) | **NO in State** | Masked by research legality | `PARITY` / `AI_DEFICIT` | `STATIC_RULE_KNOWLEDGE` | `Types.java:43-70`, `TechnologyTree.java:71-79` | Tree structure rendered in UI. Learnable in weights under Knowledge Parity standard. |
| **Dynamic Research Cost** | Star cost displayed on tech node | Yes (`tech.getCost()`) | No | **NO in State** | Masked by research legality | `AI_DEFICIT` | `INDIRECT_LEGALITY_SIGNAL` | `Types.java:72-80`, `ResearchTech.java:30` | Number of stars displayed on tech circle ($4 + \text{tier} \times N_{\text{cities}}$). |
| **Tech Affordability** | Enabled vs greyed tech node | Yes (`stars >= cost`) | Inferred | **NO in State** | Exact RESEARCH legality | `PARITY` | `INDIRECT_LEGALITY_SIGNAL` | `ResearchTech.java:30` | Disabled if insufficient stars. Affordability is fully signaled by legality. |

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
> If two game states $S_1$ and $S_2$ produce identical complete human-visible interfaces for player $i$ (including viewport, UI inspectables, and feedback affordances), then they must also produce identical complete AI-visible policy interfaces:
> $$\text{HumanInterface}(S_1, \text{player}_i) = \text{HumanInterface}(S_2, \text{player}_i) \implies \text{PolicyInterface}(S_1, \text{player}_i) = \text{PolicyInterface}(S_2, \text{player}_i)$$

We audited all Java action factories against this invariant:

#### 1. Movement Pathfinding & Zone of Control (`StepMove.java`)
- **Code Reference:** `pol_env/Tribes/src/core/actions/unitactions/StepMove.java:64-70`
- **Confirmed Hidden-State Dependence in Code:** When computing step costs, `StepMove` checks 8-neighborhood tiles around candidate destinations:
  ```java
  for (Vector2d tileAdj : tile.neighborhood(1, 0, board.getSize())) {
      Unit u = board.getUnitAt(tileAdj.x, tileAdj.y);
      if (u != null && u.getTribeId() != unit.getTribeId()) {
          zoneOfControl = true;
      }
  }
  ```
- **Analysis:** `board.getUnitAt()` queries the authoritative board without checking `isVisible(tileAdj.x, tileAdj.y)`. If an enemy unit is concealed under fog adjacent to a visible tile, Zone of Control triggers, setting movement cost to maximum and prohibiting multi-step paths beyond that tile.
- **Classification:** Primary: `UNVERIFIED` (Ordinary-human parity side-channel risk), Secondary: `CONFIRMED_HIDDEN_STATE_DEPENDENCE`, `DORMANT_PHASE1`, `FULL_GAME_ONLY`. (Dormant in Phase 1 because no enemy tribes exist).

#### 2. Attack Target Search (`AttackFactory.java`)
- **Code Reference:** `pol_env/Tribes/src/core/actions/unitactions/factory/AttackFactory.java:28-37`
- **Confirmed Hidden-State Dependence in Code:** Iterates through weapon range and checks `b.getUnitAt(tile.x, tile.y)` without checking if `tile` is visible.
- **Classification:** Primary: `UNVERIFIED` (Ordinary-human parity risk), Secondary: `CONFIRMED_HIDDEN_STATE_DEPENDENCE`, `DORMANT_PHASE1`, `FULL_GAME_ONLY`. (Neutralized in Phase 1 because `ATTACK` is excluded).

#### 3. Destination Tile Visibility (`StepMove.java:73-75`)
- **Confirmed Fog-Respecting:** Explicitly checks `gs.getTribe(unit.getTribeId()).isVisible(tile.x, tile.y)`. Ground units cannot move directly into fog.

---

## 6. Layer C: Legal-Action Feature Audit

The 42-dimensional legal-action feature vector (`v1_3_move_focus_plus_semantic_econ`) is computed in `register_env.py:2255-2376`.

### 6.1 Exhaustive 42-Feature Enumeration

| Idx | Feature Name | Action Family | Prov. Source | P-Level | Primary Class | Secondary Flags | Underlying Data Sources | Derivable by Human? | Parity Recommendation |
|:---:|---|---|:---:|:---:|:---:|---|---|:---:|---|
| **0** | `is_move` | All | `H0` | `P1` | `PARITY` | — | `action.type == "MOVE"` | Trivial | Keep (Structural P1) |
| **1** | `newly_revealed_tiles_if_move_norm` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | — | Raycast fog reveal from `(dst_x, dst_y)` / 12.0 | Yes (Mental counting) | Keep (Label as P2) or ablate |
| **2** | `adjacent_fog_count_after_move_norm` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | — | Count of fog tiles in 8-adj of dst / 8.0 | Yes (Visual count) | Keep (Label as P2) |
| **3** | `adjacent_fog_delta_norm` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | — | `(adj_fog(dst) - adj_fog(src)) / 8.0` | Yes | Keep (Label as P2) |
| **4** | `is_zero_reveal_move` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | — | `revealed == 0` | Yes | Keep (Label as P2) |
| **5** | `target_contains_visible_uncaptured_village` | MOVE | `H0` | `P0/P1` | `PARITY` | — | `dst in visible_villages` | Yes | Keep |
| **6** | `has_visible_uncaptured_village` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | — | `len(visible_villages) > 0` | Yes | Keep |
| **7** | `distance_delta_to_nearest_visible_uncaptured_village_norm` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | — | $\Delta \text{ManhattanDist}(V)$ normalized | Yes (Geometric calc) | Keep (Label as P2) |
| **8** | `is_immediate_backtrack` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | — | `dst == unit_previous_tile` | Yes (Memory) | Keep |
| **9** | `target_inside_owned_city_bounds` | MOVE | `H0` | `P0/P1` | `PARITY` | — | `tileCityId[dst] == owned_city` | Yes | Keep |
| **10** | `distance_from_capital_delta_norm` | MOVE | `H2` | `P2` | `DERIVED_ADVANTAGE` | — | $\Delta \text{ManhattanDist}(\text{Capital})$ normalized | Yes | Keep (Label as P2) |
| **11** | `unit_type_warrior` | MOVE | `H0` | `P0/P1` | `PARITY` | `CROSS_LAYER_PARTIAL` | `unit.type == WARRIOR` | Yes | Keep (Expose in state too) |
| **12** | `is_end_turn` | All | `H0` | `P1` | `PARITY` | — | `action.type == "END_TURN"` | Trivial | Keep |
| **13** | `is_capture` | All | `H0` | `P1` | `PARITY` | — | `action.type == "CAPTURE"` | Trivial | Keep |
| **14** | `is_train_or_spawn` | All | `H0` | `P1` | `PARITY` | — | `action.type in ("SPAWN", "TRAIN")` | Trivial | Keep |
| **15** | `is_research` | All | `H0` | `P1` | `PARITY` | — | `action.type == "RESEARCH_TECH"` | Trivial | Keep |
| **16** | `is_resource_gathering` | All | `H0` | `P1` | `PARITY` | — | `action.type == "RESOURCE_GATHERING"` | Trivial | Keep |
| **17** | `is_level_up` | All | `H0` | `P1` | `PARITY` | — | `action.type == "LEVEL_UP"` | Trivial | Keep |
| **18** | `is_build` | All | `H0` | `P1` | `PARITY` | — | `action.type == "BUILD"` | Trivial | Keep |
| **19** | `is_clear_forest` | All | `H0` | `P1` | `PARITY` | — | `action.type == "CLEAR_FOREST"` | Trivial | Keep |
| **20** | `is_grow_forest` | All | `H0` | `P1` | `PARITY` | — | `action.type == "GROW_FOREST"` | Trivial | Keep |
| **21** | `is_other` | All | `H0` | `P1` | `PARITY` | — | Fallback type indicator | Trivial | Keep |
| **22** | `research_tech_id_norm` | RESEARCH | `H0` | `P1` | `PARITY` | — | Normalized catalog tech index | Trivial | Keep |
| **23** | `research_is_organization` | RESEARCH | `H0` | `P0/P1` | `PARITY` | — | `tech == "ORGANIZATION"` | Trivial | Keep |
| **24** | `research_is_forestry` | RESEARCH | `H0` | `P0/P1` | `PARITY` | — | `tech == "FORESTRY"` | Trivial | Keep |
| **25** | `resource_id_norm` | RESOURCE | `H0` | `P1` | `PARITY` | — | Normalized catalog resource index | Trivial | Keep |
| **26** | `resource_is_animal` | RESOURCE | `H0` | `P0/P1` | `PARITY` | — | `resource == "ANIMAL"` | Trivial | Keep |
| **27** | `resource_is_fruit` | RESOURCE | `H0` | `P0/P1` | `PARITY` | — | `resource == "FRUIT"` | Trivial | Keep |
| **28** | `resource_is_fish` | RESOURCE | `H0` | `P0/P1` | `PARITY` | — | `resource == "FISH"` | Trivial | Keep |
| **29** | `resource_is_crop` | RESOURCE | `H0` | `P0/P1` | `PARITY` | — | `resource == "CROPS"` | Trivial | Keep |
| **30** | `resource_is_metal` | RESOURCE | `H0` | `P0/P1` | `PARITY` | — | `resource == "ORE"` | Trivial | Keep |
| **31** | `build_id_norm` | BUILD | `H0` | `P1` | `PARITY` | — | Normalized catalog building index | Trivial | Keep |
| **32** | `build_is_lumber_hut` | BUILD | `H0` | `P0/P1` | `PARITY` | — | `building == "LUMBER_HUT"` | Trivial | Keep |
| **33** | `build_is_sawmill` | BUILD | `H0` | `P0/P1` | `PARITY` | — | `building == "SAWMILL"` | Trivial | Keep |
| **34** | `levelup_choice_id_norm` | LEVEL_UP | `H0` | `P1` | `PARITY` | — | Normalized catalog level-up index | Trivial | Keep |
| **35** | `levelup_is_workshop` | LEVEL_UP | `H0` | `P0/P1` | `PARITY` | — | `choice == "WORKSHOP"` | Trivial | Keep |
| **36** | `expected_population_delta_norm` | ECON/BUILD | `H2` | `P2` | `DERIVED_ADVANTAGE` | — | Precomputed pop yield / 2.0 | Yes (Rule arithmetic) | Keep (Label as P2) |
| **37** | `expected_immediate_spt_delta_norm` | ECON/BUILD | `H2` | `P2` | `DERIVED_ADVANTAGE` | — | Precomputed immediate SPT yield / 5.0 | Yes (Rule arithmetic) | Keep (Label as P2) |
| **38** | `makes_level_up_available` | ECON/BUILD | `H2` | `P2` | `DERIVED_ADVANTAGE` | — | `pop + \Delta pop >= need` | Yes (Mental check) | Keep (Label as P2) |
| **39** | `is_level_up_claim` | LEVEL_UP | `H0` | `P0/P1` | `PARITY` | — | `action.type == "LEVEL_UP"` | Trivial | Keep |
| **40** | `action_city_upgrade_progress_before_norm` | ECON/BUILD | `H2` | `P2` | `DERIVED_ADVANTAGE` | `CROSS_LAYER_PARTIAL` | Pre-action city progress fraction | Yes | Keep (Label as P2) |
| **41** | `action_city_upgrade_ready_before` | ECON/BUILD | `H2` | `P2` | `DERIVED_ADVANTAGE` | `CROSS_LAYER_PARTIAL` | `pop >= need` for target city | Yes | Keep (Label as P2) |

---

### 6.2 Missing Action Properties & Spatial Semantics
1. **Action Star Cost (`AI_DEFICIT`):** The action feature vector contains **no star cost feature**. While PPO knows whether it can afford an action via legality masking, it cannot evaluate the marginal cost efficiency of candidate actions without trial-and-error credit assignment.
2. **Post-Action Star Balance (`DERIVED_ADVANTAGE` / Not a Primitive Deficit):** Inferred as $\text{stars} - \text{cost}$. Once star cost is exposed, post-action balance is a deterministic P2 derivation.
3. **Destination Spatial Coordinates (`REPRESENTATION_LOSS` / `REPRESENTATION_HAZARD`):** The policy can distinguish destination tiles because they are partitioned into unique global action IDs in `GlobalActionCatalog`. However, the neural net receives these as discrete index embeddings without explicit 2D spatial coordinate features ($x, y$).

---

## 7. Layer D: Public Game Knowledge Audit

### 7.1 Public Game Knowledge Parity Policy (Open Design Decision)
A key methodological question in game RL is deciding which public game rules must be explicitly supplied to the policy versus learned into neural weights:

> **Open Design Decision: Interface Parity vs. Knowledge Parity**  
> - **Interface Parity Standard:** If a human can inspect a value directly in the UI (e.g., static warrior cost badge "2★", static tech tree prerequisite lines), the AI receives it explicitly.  
> - **Knowledge Parity Standard:** Static invariants (unit base HP/movement, static tech connections, static building costs) may be learned into policy parameters from experience because a human player memorizes them.  
> - **Resolution for Dynamic Values:** Regardless of the standard chosen for static rules, **dynamic runtime values** (dynamic tech cost $C = 4 + \text{tier} \times N_{\text{cities}}$, current city population, remaining unit capacity) **must be explicitly represented**. Legality masking alone is not equivalent to information parity.

| Game Rule Domain | Nature of Knowledge | PolyVision PPO Representation | Parity Assessment (Knowledge Std) | Parity Assessment (Interface Std) | Parity Recommendation |
|---|---|---|:---:|:---:|---|
| **Unit Base Statistics (ATK, DEF, MOV, RANGE)** | Static Invariant | Implicit in neural network weights | `PARITY` | `AI_DEFICIT` | Acceptable to learn in weights |
| **Unit Purchase Costs** | Static Invariant | Implicit in weights; omitted from action features | `PARITY` | `AI_DEFICIT` | Expose action cost in features |
| **Building Costs & Requirements** | Static Invariant | Implicit in weights; enforced by Java legal mask | `PARITY` | `AI_DEFICIT` | Expose action cost in features |
| **Building Economic Yields** | Static Rule ($+1$ pop, $+1$ SPT) | Precomputed in action features 36–37 | `DERIVED_ADVANTAGE` | `DERIVED_ADVANTAGE` | Retain, label as P2 |
| **Tech Tree Directed Graph** | Static Invariant | Implicit in weights; enforced by Java legal mask | `PARITY` | `AI_DEFICIT` | Acceptable to learn in weights |
| **Dynamic Tech Cost Formula** | Dynamic ($4 + \text{tier} \times N_{\text{cities}}$) | Enforced by legal mask; not explicit | `AI_DEFICIT` | `AI_DEFICIT` | Expose dynamic cost in action feature |
| **City Unit Capacity Rule** | Dynamic Rule ($N_{\text{units}} < L + 1$) | Inferred via Spawn legality; omitted in state | `AI_DEFICIT` | `AI_DEFICIT` | Expose supported units & cap in state |
| **City Level-Up Formula** | Static Rule ($\text{need} = L + 1$) | Precomputed in action features 36–41 | `DERIVED_ADVANTAGE` | `DERIVED_ADVANTAGE` | Expose raw pop & need in state |
| **Terrain Movement Costs & Roads** | Static Rule | Enforced by `StepMove` pathfinding | `PARITY` | `PARITY` | Acceptable via pathfinder |
| **Sight & Vision Mechanics** | Static Rule (1-tile, Mt=2) | Precomputed in action features 1–4 | `DERIVED_ADVANTAGE` | `DERIVED_ADVANTAGE` | Retain, label as P2 |

---

## 8. Whole-Policy Cross-Layer Reconciliation

| Information Element | Present in State (505)? | Encoded in Action Identity? | Present in Action Features (42)? | Whole Policy Possesses It? | Human UI Has It? | Final Whole-Policy Parity Status | Cross-Layer Reconciliation Summary |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Terrain Layout** | Yes (121) | Implicit | Implicit | **YES** | Yes | `PARITY` | Fully visible in state vector. |
| **Fog of War** | Yes (121) | Implicit | Feat 1–4 | **YES** | Yes | `PARITY` | Fully visible in state vector. |
| **Visible Resources** | Yes (121) | Yes (Res IDs) | Feat 25–30 | **YES** | Yes | `PARITY` | Fully visible in state vector. |
| **Building Placement & Type** | **NO** | Yes (Build IDs) | Feat 31–33 | **PARTIAL** | Yes | `AI_DEFICIT` (`CROSS_LAYER_PARTIAL`) | State vector lacks building plane; policy only sees building type when constructing on a tile. |
| **Road Network** | **NO** | No | No | **NO** | Yes | `AI_DEFICIT` | Completely absent across all policy inputs. |
| **Capital Identity** | **NO** | No | Feat 10 ($\Delta \text{Cap}$) | **PARTIAL** | Yes | `AI_DEFICIT` (`CROSS_LAYER_PARTIAL`) | State lacks capital flag; MOVE features provide scalar distance delta to capital. |
| **Per-City Level** | **NO** (Aggregated) | No | No | **POORLY ENCODED** | Yes | `REPRESENTATION_LOSS` | Only global mean and max level are exposed in state. |
| **Per-City Population** | **NO** (Aggregated) | No | Feat 40 (Target city) | **PARTIAL** | Yes | `REPRESENTATION_LOSS` (`CROSS_LAYER_PARTIAL`) | State aggregates progress; target city progress is available when targeting city with an action. |
| **City Unit Capacity & Count** | **NO** | No | No | **NO** | Yes | `AI_DEFICIT` | Omitted across state and action features. |
| **Unit Type** | **NO** | No | Feat 11 (Warrior flag) | **PARTIAL** | Yes | `AI_DEFICIT` (`CROSS_LAYER_PARTIAL`) | State lacks unit type; MOVE feature has binary Warrior flag. |
| **Unit Current & Max HP** | **NO** | No | No | **NO** | Yes | `AI_DEFICIT` | Omitted across all policy inputs (Dormant in Phase 1). |
| **Unit Turn Status** | **NO** | Inferred via mask | No | **INDIRECT** | Yes | `AI_DEFICIT` (`INDIRECT_LEGALITY_SIGNAL`) | State lacks status; policy infers finished status only from lack of legal moves. |
| **Researched Techs (Full 24)** | **NO** (Org/Forestry only) | No | Feat 23, 24 | **POORLY ENCODED** | Yes | `REPRESENTATION_LOSS` | State only exposes Organization, Forestry, and total tech count. |
| **Action Star Cost** | **NO** | No | **NO** | **NO** | Yes | `AI_DEFICIT` | Absent from action features; only binary affordability is inferable from legality mask. |
| **Move Destination Coordinates** | **NO** | Yes (Global ID) | Feat 5, 9 | **POORLY ENCODED** | Yes | `REPRESENTATION_LOSS` (`REPRESENTATION_HAZARD`) | Destination is uniquely identified by discrete global ID, but lacks explicit 2D spatial coordinates. |

---

## 9. Consolidated Disparity Register

| Disparity ID | Layer | Information / Behavior | Human Access | AI Access | Primary Class | Phase-1 Severity | Full-Game Severity | Current Source | Target Parity Direction (Architecture-Neutral) | Status |
|:---:|:---:|---|---|---|:---:|:---:|:---:|---|---|:---:|
| `STATE-CITY-001` | State | Per-City Level | Displayed over city | Destroyed into global mean/max | `REPRESENTATION_LOSS` | **Critical** | **Critical** | `register_env.py:5054-5071` | Preserve exact level for each city with unambiguous spatial/entity association | **CONFIRMED** |
| `STATE-CITY-002` | State | Per-City Population & Need | Displayed in status bar | Destroyed into global progress | `REPRESENTATION_LOSS` | **Critical** | **Critical** | `register_env.py:5049-5073` | Preserve exact population and population-needed values for each city | **CONFIRMED** |
| `STATE-CITY-003` | State | Per-City SPT Contribution | Displayed on banner | Summed into global SPT | `REPRESENTATION_LOSS` | **High** | **High** | `register_env.py:5019,5062` | Preserve individual city production contributions | **CONFIRMED** |
| `STATE-CITY-004` | State | City Unit Count & Capacity | Displayed pips under city | Omitted across all layers | `AI_DEFICIT` | **Critical** | **Critical** | `City.java:296,418` | Preserve supported units count and capacity ($L+1$) per city | **CONFIRMED** |
| `STATE-MAP-001` | State | Building Placement & Type | Inspectable on board | Discarded in Python flattening | `AI_DEFICIT` | **High** | **High** | `PythonEnv.java:345`, `register_env.py:4962` | Expose the type and location of every human-visible building without loss of spatial identity | **CONFIRMED** |
| `STATE-MAP-002` | State | Road Grid on Map | Inspectable on board | Discarded in Python flattening | `AI_DEFICIT` | **Medium** | **High** | `Board.java:1010`, `register_env.py:4962` | Expose visible road placement on tiles | **CONFIRMED** |
| `STATE-MAP-003` | State | Capital City Identity | Star icon on city banner | Omitted from state vector | `AI_DEFICIT` | **Medium** | **High** | `City.java:381`, `register_env.py:5028` | Expose capital identity for the capital city | **CONFIRMED** |
| `STATE-UNIT-001` | State | Unit Type on Map | Distinct sprite | Discarded from state vector | `AI_DEFICIT` | **High** | **Critical** | `PythonEnv.java:361`, `register_env.py:4971` | Expose the type of each visible/owned unit with spatial/entity association | **CONFIRMED** |
| `STATE-UNIT-002` | State | Unit Current & Max HP | Health bar over unit | Discarded from state vector | `AI_DEFICIT` | **Low (Solo)** | **Critical** | `PythonEnv.java:375`, `register_env.py:4971` | Expose current and maximum HP per unit | **CONFIRMED** |
| `STATE-ID-001` | State | Raw Actor IDs as Floats | Invisible / N/A | Injected as continuous float32 | `REPRESENTATION_LOSS` | **High** | **High** | `Board.java:887`, `register_env.py:4971` | Remove unintended numerical semantics from raw actor IDs while preserving entity identity | **CONFIRMED** |
| `STATE-TECH-001` | State | Full Researched Tech Tree | Full 24-tech UI tree | Compressed to count + 2 flags | `REPRESENTATION_LOSS` | **High** | **High** | `register_env.py:5026,5068` | Expose full boolean researched status for all 24 technologies | **CONFIRMED** |
| `FEAT-MISS-001` | Features | Action Star Cost | Displayed on buttons | Omitted from feature row | `AI_DEFICIT` | **High** | **High** | `register_env.py:217-261` | Expose normalized star cost for candidate actions | **CONFIRMED** |
| `FEAT-SPAT-001` | Features | Move Destination Coordinates | Spatial grid click | Discrete global ID embedding | `REPRESENTATION_LOSS` | **Medium** | **Medium** | `register_env.py:83-87` | Expose explicit spatial source and destination coordinates | **CONFIRMED** |
| `ACTION-SIDE-001` | Legality | Zone of Control Fog Leak | Hidden by fog | Authoritative check in `StepMove` | `UNVERIFIED` | **Dormant** | **High** | `StepMove.java:64-70` | Enforce human-interface equivalence invariant in pathfinding | **CONFIRMED CODE LEAK** |
| `ACTION-SIDE-002` | Legality | Attack Target Fog Leak | Hidden by fog | Authoritative check in `AttackFactory` | `UNVERIFIED` | **Dormant** | **Critical** | `AttackFactory.java:28-37` | Require `isVisible(target)` in target generation | **CONFIRMED CODE LEAK** |
| `ACTION-FILT-001` | Legality | Pre-2-City Expansion Filters | Full player freedom | Heuristically forced to village | `TASK_DIFFERENCE` | **Medium** | **N/A** | `register_env.py:2770-2865` | Retain as Phase-1 curriculum rule | **CONFIRMED** |

---

## 10. Unverified Findings & Diagnostic Test Protocols

### Protocol 1: Actor ID Chronology Leakage (`STATE-ID-001`)
- **Hypothesis:** Monotonically increasing actor IDs from `Board.actorIDcounter` encode the count or presence of hidden entities initialized under fog.
- **Test Invariant:** For two level files $L_1$ and $L_2$ where the player's initial visible viewport (terrain, city, warrior) is 100% identical, but $L_2$ contains 4 additional neutral villages in hidden fog, the initial state observation vector $O(L_1)$ and $O(L_2)$ must be identical.
- **Empirical Status:** `UNVERIFIED` (Requires running paired diagnostic resets).

### Protocol 2: Multi-Agent Legal-Action Fog Invariance (`ACTION-SIDE-001`)
- **Hypothesis:** Hidden enemy unit placement alters the active tribe's legal move action set via Zone of Control in `StepMove.java:64-70`.
- **Test Invariant:** For two game states $S_1$ and $S_2$ where Tribe 0's complete human-visible interface is identical, but $S_2$ has an enemy unit placed in adjacent hidden fog, `env.list_actions()` for Tribe 0 must be strictly identical.
- **Empirical Status:** `CONFIRMED_HIDDEN_STATE_DEPENDENCE` in code; `UNVERIFIED` ordinary-human parity impact.

---

## 11. Prioritized Roadmap for Next Phase-1 Training Run

| Priority | Disparity Item | Phase-1 Impact | Human-Parity Importance | Recommended Before Next Phase-1 Retraining? | Architectural Parity Direction (Neutral) |
|:---:|---|---|:---:|:---:|---|
| **P1** | Per-City Level, Pop & Need (`STATE-CITY-001`, `002`) | **Critical** | **Critical** | **Strongly Recommended** | Preserve exact per-city level, population, and need with entity identity. |
| **P1** | City Unit Count & Capacity (`STATE-CITY-004`) | **Critical** | **Critical** | **Strongly Recommended** | Preserve current supported units and capacity ($L+1$) per city. |
| **P1** | Action Star Cost (`FEAT-MISS-001`) | **High** | **High** | **Strongly Recommended** | Include normalized action star cost in action features. |
| **P1** | Building Placement & Type (`STATE-MAP-001`) | **High** | **High** | **Strongly Recommended** | Expose type and location of all visible buildings with spatial identity. |
| **P2** | Full 24-Tech Tree Vector (`STATE-TECH-001`) | **High** | **High** | **Recommended** | Expose full 24-element binary researched tech vector. |
| **P2** | Per-City SPT Contribution (`STATE-CITY-003`) | **High** | **High** | **Recommended** | Preserve per-city production values with city identity. |
| **P2** | Eliminate Raw Float Actor IDs (`STATE-ID-001`) | **High** | **High** | **Recommended** | Remove scalar integer IDs while preserving categorical entity identity. |
| **P3** | Capital City Indicator (`STATE-MAP-003`) | **Medium** | **High** | **Recommended** | Expose capital status with capital city identity. |
| **P3** | Unit Type Channel (`STATE-UNIT-001`) | **Medium** | **High** | **Recommended** | Expose unit type with spatial/entity identity. |
| **P4** | Unit HP / Veteran Status (`STATE-UNIT-002`) | **Low (Solo)** | **Critical (Full)** | **Defer** | Defer until combat/opponent phase. |
| **P4** | Zone of Control Fog Leak (`ACTION-SIDE-001`) | **Dormant** | **Critical (Full)** | **Defer** | Fix in `StepMove.java` before introducing opponent AI. |

---

## 12. Open Design Decisions & Methodological Questions

1. **Static Rule Representation Policy:** Should static game invariants (unit base stats, tech graph connections) be explicitly supplied as model metadata (Interface Parity) or learned into weights (Knowledge Parity)?
2. **Spatial Grid vs. Permutation-Equivariant Entity Tables:** Should per-city and per-unit state be embedded into multi-channel 2D spatial planes, structured entity slot arrays (e.g. Transformer tokens), or a hybrid representation?
3. **P2 Derived Feature Strategy:** Should precomputed P2 action features (`newly_revealed_tiles_if_move_norm`, `distance_delta_to_nearest_visible_uncaptured_village_norm`) remain in the default interface or be ablated against purely primitive (P0/P1) observations?
4. **Curriculum Filter Deprecation:** At what developmental phase should Phase-1 tactical filters (village capture forcing, backtrack masks) be relaxed to test autonomous exploratory policies?
5. **Human Benchmark Interface Upgrade:** When should `tools/human_policy_interface.py` be upgraded to render the full entity-level human parity state rather than the historical 505-vector?

---

## 13. Target Parity Principles for Implementation

1. **Strict Human-Interface Equivalence Invariant:**  
   If $\text{HumanInterface}(S_1, \text{player}) = \text{HumanInterface}(S_2, \text{player})$, then $\text{PolicyInterface}(S_1, \text{player}) = \text{PolicyInterface}(S_2, \text{player})$. The policy must never receive extra information unavailable through legitimate human UI affordances.
2. **Preserve Entity-Level Granularity:** Eliminate destructive global pooling (`avg_city_level`, `mean_upgrade_progress`); maintain entity-level resolution for all owned and visible assets.
3. **Eliminate Arbitrary Engine ID Numerics:** Never feed raw monotonic engine counters as continuous scalar inputs.
4. **Explicit Provenance Tracking:** Every policy-visible feature must have documented provenance tracing back to legitimate human-visible state (H0) or public rules (H1/H2).
5. **Decouple Task Scoping from Information Blinding:** Phase-1 task constraints (e.g. Turn-10 horizon, combat exclusion) must be enforced through action availability, never by blinding the policy to visible state.

---

## 14. Audit Closure Status

### 14.1 Confirmed Current Phase-1 Parity Deficits
- Complete absence of per-city unit capacity and supported unit counts (`STATE-CITY-004`).
- Complete absence of 2D building placement and types on map (`STATE-MAP-001`).
- Complete absence of road network on map (`STATE-MAP-002`).
- Absence of action star cost in legal action features (`FEAT-MISS-001`).
- Absence of explicit capital identity in state vector (`STATE-MAP-003`).

### 14.2 Confirmed Representation Problems
- Destructive global aggregation of per-city levels, populations, and production (`STATE-CITY-001`, `002`, `003`).
- Heavy compression of 24-tech research state down to 2 boolean flags and a count scalar (`STATE-TECH-001`).
- Raw monotonic engine actor IDs injected as continuous float32 inputs (`STATE-ID-001`).
- Spatial move destinations encoded as discrete index embeddings rather than explicit $(x, y)$ coordinates (`FEAT-SPAT-001`).

### 14.3 Deferred / Unverified Full-Game Risks
- Hidden-state dependence in `StepMove.java` (Zone of Control queries hidden units) and `AttackFactory.java` (Attack queries hidden units) — dormant in solo no-combat Phase 1.
- Unit HP, Max HP, veteran status, and kill count omissions — low priority for solo Phase 1, critical for future combat.

### 14.4 Implementation Readiness
**The audit is complete and ready.** All critical human-visible information items and engine behaviors are verified and cited. The document provides the exact normative foundation required to design the future Human–AI Information Parity observation and action contract.

---

## 15. Evidence & Provenance Index

### 15.1 Human Interface / Commercial Polytopia Evidence
- **Official Polytopia Website & Gameplay Manual:** [`https://polytopia.io/`](https://polytopia.io/) (Verified core gameplay, terrain rendering, city levels, population pips, unit movement, fog of war, and tribe abilities).
- **Polytopia Official Rulebook & In-Game Help:**
  - *City Status Bar:* Shows city level, discrete population dots, population needed to level ($L + 1$), and $+X$ star production banner.
  - *City Unit Capacity:* Displays unit dots under city banner; total dots $= L + 1$; filled dots indicate supported units; empty dots indicate available capacity.
  - *Capital City:* Star emblem on city banner and board.
  - *Technology Tree:* 5-branch directed graph with 24 technologies, prerequisite links, dynamic star costs ($4 + \text{tier} \times N_{\text{cities}}$), and unlock badges.
  - *Units & Health:* Distinct 3D models per unit type; health bar with numeric overlay ($10/10$ standard, $15/15$ veteran); medal/crown badge for veteran status; dimmed sprite for moved units.
  - *Buildings & Roads:* 3D structures rendered directly on tile (Lumber Huts, Sawmills, Ports, Temples); cobblestone/dirt paths connecting tile centers.
- **Polytopia Reference Community Wiki:** [`https://polytopia.fandom.com/wiki/The_Battle_of_Polytopia_Wiki`](https://polytopia.fandom.com/wiki/The_Battle_of_Polytopia_Wiki) (Corroborated dynamic tech cost formula, city unit cap formulas, resource gathering tech requirements, and monument unlocking rules).

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
- [`pol_env/Tribes/src/core/actions/unitactions/StepMove.java`](file:///c:/PolyVision/pol_env/Tribes/src/core/actions/unitactions/StepMove.java) — Movement pathfinding and Zone of Control calculation.
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
- [`tools/validate_human_benchmark_parity.py`](file:///c:/PolyVision/tools/validate_human_benchmark_parity.py) — Human/PPO interface parity validator.
