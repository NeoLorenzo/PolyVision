# Maps

PolyVision's active Phase 1 corpus consists of **256 genuine 11×11 Bardur Drylands maps** in:

```text
pol_env/Tribes/levels/phase1_pool_bardur_real/
```

These maps originate from generated initial-map data in genuine Polytopia save files. They are not procedurally produced by the retired Tribes map generator.

## Ingestion pipeline

```text
compressed .state
  → tools/polytopia_state_converter
  → canonical schema-v1 JSON
  → tools/polytopia_map_converter
  → validated Tribes CSV
```

The first stage reads only the generated initial state, excluding later moves and account-identifying fields. It retains raw semantic data, validates coordinates, and records a source-file SHA-256 plus a content-derived `map_sha256`.

The second stage maps known Polytopia terrain, resource, improvement, owner, and tribe IDs to Tribes tokens. It preserves coordinates (`row = y`, `column = x`) without rotation, resizing, cropping, or padding. Unsupported or ambiguous semantics fail explicitly. Optional Java validation loads the result through `LevelLoader` and compares every representable coordinate back to the canonical JSON.

## Import maps

See the focused [state converter manual](../tools/polytopia_state_converter/README.md) for parser requirements and canonical schema. A typical batch conversion is:

```powershell
go run ./tools/polytopia_state_converter `
    --input data/polytopia_maps/raw_states `
    --output data/polytopia_maps/parsed
```

Then follow the [map converter manual](../tools/polytopia_map_converter/README.md):

```powershell
python tools/polytopia_map_converter/convert_maps.py `
    --input data/polytopia_maps/parsed `
    --output pol_env/Tribes/levels/phase1_pool_bardur_real `
    --manifest data/polytopia_maps/conversion_manifest.csv `
    --java-validate
```

Keep the conversion manifest outside the level directory so a `*.csv` runtime glob cannot load it as a map. Raw saves, canonical JSON, and the manifest live under ignored `data/polytopia_maps/`; committed runtime CSVs are the distributable corpus.

## Validate the corpus

Run converter tests and then the live environment contract:

```powershell
go test ./tools/polytopia_state_converter/...
python -m unittest discover -s tools/polytopia_map_converter/tests -v
python tools/validate_environment_contract.py --expected-width 11 --expected-height 11
```

The live validator selects the real pool and solo mode itself, resets all pool entries, and verifies a single stable geometry, 505-value observation, 63,913-ID global catalog, and consistent catalog fingerprint.

## Runtime selection

Until the wrapper's legacy fallback constant is corrected in runtime code, explicitly set:

```powershell
$env:POLYVISION_LEVEL_POOL_GLOB = 'levels/phase1_pool_bardur_real/*.csv'
$env:POLYVISION_SOLO_NO_OPPONENT_MODE = '1'
```

Pools must be square and dimension-homogeneous. Map geometry is part of checkpoint compatibility. A model trained on a different geometry or pool contract must not be assumed compatible merely because PyTorch tensor loading succeeds.

The optional [harvester manual](../tools/polytopia_harvester/README.md) documents Windows UI automation for collecting save files. Harvesting controls the external game and is separate from normal environment setup or training.
