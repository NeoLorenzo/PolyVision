# Actions

PolyVision separates the Java engine's state-local action indices from the policy's stable global action IDs. The policy never relies on “action 7” meaning the seventh legal Java action across states.

## Global catalog

`GlobalActionCatalog` deterministically allocates ID ranges from map geometry and Java enum vocabularies. Families include `END_TURN`, `MOVE`, `CAPTURE`, training/spawning, `RESOURCE_GATHERING`, `CLEAR_FOREST`, `GROW_FOREST`, `BUILD`, `RESEARCH_TECH`, `LEVEL_UP`, and `EXAMINE`.

For the current 11×11 corpus, the discrete global action space contains **63,913 IDs**. Most IDs are illegal in any one state. The catalog is versioned as `flat-v1`; structured canonicalization is `flat-v1-structured`. Its fingerprint is saved with checkpoints because geometry or vocabulary changes can change ID meaning.

## Legal action construction

For each state the wrapper:

1. asks Java for the complete raw legal-action list;
2. removes action families outside the Phase 1 interface and applies current tactical filters;
3. canonicalizes each remaining structured action to one global ID;
4. rejects missing mappings or ID collisions;
5. builds the dense debug mask and fixed-size sparse legal tensors;
6. maps a chosen global ID back to its exact raw Java index.

The Phase 1 allow-list currently includes `END_TURN`, `MOVE`, `CAPTURE`, `EXAMINE`, `SPAWN`, `RESOURCE_GATHERING`, `CLEAR_FOREST`, `GROW_FOREST`, `LEVEL_UP`, `RESEARCH_TECH`, and `BUILD`. Java `ATTACK` actions are deliberately excluded. Water-unit and super-unit training entries are not included in the catalog.

The wrapper also contains active curriculum filters: before two cities it prioritizes available village captures, then movement onto or toward visible neutral villages; it blocks one narrow immediate-backtrack pattern on early turns; and it can optionally gate resource gathering on city-upgrade usefulness. These filters mean “Java-legal” and “policy-visible” are not always identical.

## Actor modes

| Mode | Policy scoring path | Intended use |
|---|---|---|
| `legal_only` | Scores only current legal slots using learned embeddings of stable global IDs. | Current default and efficient baseline. |
| `legal_features` | Adds a learned encoding of per-action semantic/economic features to legal-slot scoring. | Current richer legality-aware actor. |
| `dense_debug` | Produces logits for all 63,913 global IDs, then applies a dense legality mask. | Debugging and equivalence checks, not the efficient path. |

The fixed legal-slot capacity defaults to **256** (`POLYVISION_MAX_LEGAL_ACTIONS` / `--max-legal-actions`). Exceeding it is a hard error, not silent truncation.

## Current action features

`legal_features` uses version `v1_3_move_focus_plus_semantic_econ`, a **42-dimensional** vector per legal slot. It combines:

- movement and exploration signals, including predicted reveal, adjacent fog, backtracking, capital distance, and village targeting;
- one-hot action-family indicators;
- normalized research, resource, building, and level-up identities;
- semantic flags for Organization, Forestry, common resources/buildings, and Workshop;
- predicted population/SPT effects and city-upgrade readiness/progress.

Features are derived from policy-visible state plus the legal action's structured metadata. Full-visibility observations are diagnostic-only and are not fed into these features.

## Interface tensors

`reset()` and `step()` return in `info`:

- `legal_global_ids_padded`: length 256 by default;
- `legal_action_valid_mask`: which slots are populated;
- `legal_action_count`: number of populated slots;
- `legal_action_features_padded`: shape `(256, 42)` by default;
- feature/catalog versions and fingerprints.

Checkpoint loading validates geometry, observation/action dimensions, actor mode, catalog fingerprint, canonicalizer/catalog versions, feature version/dimension, and legal-slot capacity before model tensors are used.
