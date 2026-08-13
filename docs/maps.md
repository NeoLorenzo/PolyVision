# Maps

PolyVision Phase 1 uses a frozen, identity-based split of **5,517 genuine 11×11 Bardur Drylands maps**. The authoritative record is `pol_env/Tribes/levels/phase1_pool_bardur_real/split_manifest.json`; paths alone do not define membership.

```text
harvested genuine maps
        ↓
deduplicated / canonical map corpus
        ↓
fixed split (seed 20260813)
        ├── train
        ├── validation
        ├── test
        └── human_benchmark
```

| Pool | Maps | Used for gradient training? | Can influence model development? | Purpose |
|---|---:|---|---|---|
| Train | 5000 | Yes | Yes | PPO learning |
| Validation | 250 | No | Yes | Model/config selection |
| Test | 250 | No | No | Final held-out generalization |
| Human benchmark | 17 | No | Yes | Human-vs-agent challenge |

Held-out means held out by canonical map identity and exact CSV content hash, not merely by filename or directory. The four pools contain no duplicate canonical `map_sha256` values and no duplicate CSV SHA-256 values.

## Pool semantics

### Training

`levels/phase1_pool_bardur_real/train/*.csv` is the only pool permitted for PPO gradient updates. Maps may recur arbitrarily during learning, and training performance is not evidence of generalization. This is the wrapper's safe default when `POLYVISION_LEVEL_POOL_GLOB` is unset.

### Validation

`levels/phase1_pool_bardur_real/validation/*.csv` is excluded from gradients but may be evaluated repeatedly for checkpoint selection, reward and architecture experiments, actor comparisons, training-duration choices, and hyperparameter tuning. Because its results may influence development, validation performance is not an unbiased final test result.

### Test

`levels/phase1_pool_bardur_real/test/*.csv` is the pristine scientific held-out set. Do not train on it, tune against it, select checkpoints with it, or repeatedly inspect it during development. Use it only after the model and configuration have been selected. If test results cause a model or configuration change, the test set is contaminated for that development cycle and the resulting result must not be described as pristine.

### Human benchmark

`levels/phase1_pool_bardur_real/human_benchmark/*.csv` is a permanently non-training challenge set. Humans should play through the Phase 1 wrapper so human and agent receive the same map, scripted opening, fog, Turn-10 horizon, action interface and filters, restrictions, and scoring semantics. Future models may be evaluated repeatedly here and results may influence development, so this pool is conceptually separate from the scientific test set.

Run the human wrapper with its benchmark-safe default:

```powershell
python tools/human_benchmark.py
```

For non-registry ad hoc play, select it explicitly:

```powershell
python tools/play_human_t10_wrapper.py `
    --level-pool-glob 'levels/phase1_pool_bardur_real/human_benchmark/*.csv'
```

Official benchmark attempts are registered permanently, preserve the first completed score, and prohibit privileged render/state paths. See [Human benchmark](human-benchmark.md).

## Frozen split and manifest

The split seed is `20260813`. Initial assignments were created by sorting maps by SHA-256 of `<split_seed>:<canonical_map_sha256>` and applying the fixed counts above. This avoids filename-order assignment and is reproducible across Python versions. The manifest records every filename, assigned pool, relative path, exact CSV SHA-256, canonical map SHA-256, file size, and aggregate pool identity compatible with `compute_level_pool_identity()`.

Verify the cheap static/hash contract at any time:

```powershell
python tools/split_phase1_map_pool.py
```

The command checks exact counts, manifest/filesystem agreement, every file hash, 11×11 CSV structure, one Bardur capital, duplicate canonical identities, duplicate exact CSV content, cross-pool overlap, aggregate pool identities, and loose root CSVs. It is verification-only once the manifest exists and never reshuffles or rebalances an established split.

`--establish` exists only to create the initial contract from an exact flat 5,517-map corpus and conversion manifest. It refuses partial splits, duplicates, hash disagreement, unexpected counts, and existing manifests.

## Ingestion and future maps

The ingestion pipeline is:

```text
compressed .state
  → canonical schema-v1 JSON
  → validated Tribes CSV in data/polytopia_maps/incoming_csv/
  → explicit uniqueness review and manifest assignment
  → one canonical pool
```

From the repository root:

```powershell
go -C tools/polytopia_state_converter run . `
    --input ../../data/polytopia_maps/raw_states `
    --output ../../data/polytopia_maps/parsed

python tools/polytopia_map_converter/convert_maps.py `
    --input data/polytopia_maps/parsed `
    --output data/polytopia_maps/incoming_csv `
    --manifest data/polytopia_maps/conversion_manifest.csv `
    --java-validate
```

Newly harvested maps are incoming/unassigned data. Never point training or evaluation globs at the staging directory, and never write converter output directly into a canonical pool. Harvesting more maps must not reshuffle any established assignment.

When deliberately updating the dataset contract:

- training may grow with newly harvested unique maps;
- validation may grow through an explicit manifest update;
- test may grow only with new maps never used for training or model selection;
- human benchmark may grow only with new never-trained maps;
- an old training map must never be relabeled as held out;
- a validation map already used for development must never be promoted to pristine test;
- every assignment must be recorded in an updated/versioned manifest.

The present utility intentionally does not automate future reallocation. Updating this frozen contract requires an explicit reviewed change.

## Live validation

Static verification is fast and should be routine. Full Java resets are separate because checking thousands of maps is substantially more expensive:

```powershell
python tools/validate_environment_contract.py `
    --level-pool-glob 'levels/phase1_pool_bardur_real/train/*.csv' `
    --expected-width 11 --expected-height 11
```

Use `--max-maps 1` for a small live smoke check. Substitute `validation`, `test`, or `human_benchmark` explicitly when validating those pools. All pools must preserve the same 11×11, 505-observation, 63,913-action Phase 1 environment contract.

The optional [harvester manual](../tools/polytopia_harvester/README.md), [state converter manual](../tools/polytopia_state_converter/README.md), and [map converter manual](../tools/polytopia_map_converter/README.md) describe the preceding ingestion stages.
