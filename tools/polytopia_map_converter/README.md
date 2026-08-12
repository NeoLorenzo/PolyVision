# PolyVision Canonical Map Converter

This tool implements the second, deliberately separate real-map ingestion stage:

```text
compressed Polytopia .state
    -> tools/polytopia_state_converter
    -> canonical schema-v1 JSON
    -> tools/polytopia_map_converter
    -> validated PolyVision/Tribes CSV
```

It does not parse `.state` files, alter map geometry, generate synthetic maps, or change training configuration. The current real output pool is separate from the existing synthetic pool.

## Requirements

- Python 3.10 or newer; the converter uses only the standard library.
- Canonical JSON with exactly `"schema_version": 1`.
- A compiled Tribes engine and JDK when using `--java-validate`.

Compile the existing Java engine from the repository root when needed:

```powershell
Set-Location pol_env/Tribes
$sources = Get-ChildItem -Path src -Recurse -Filter *.java | ForEach-Object FullName
javac -cp "lib/json.jar" -d out -sourcepath src $sources
Set-Location ../..
```

## Semantic mappings

Polytopia save IDs and PolyVision Java enum keys are different namespaces. Numeric Polytopia IDs are converted through named semantic constants; they are never passed directly to Java.

| Raw kind | Polytopia ID | Meaning | Tribes CSV |
|---|---:|---|---|
| Terrain | 1 | Coast | `s` |
| Terrain | 2 | Ocean | `d` |
| Terrain | 3 | Field | `.` |
| Terrain | 4 | Mountain | `m` |
| Terrain | 5 | Forest | `f` |
| Resource | 1 | Game/animal | `a` |
| Resource | 2 | Crop | `c` |
| Resource | 3 | Fish | `h` |
| Resource | 4 | Whale | `w` |
| Resource | 5 | Metal/ore | `o` |
| Resource | 6 | Fruit | `f` |
| Improvement | 1 | City or village | contextual |
| Improvement | 2 | Ruin | underlying terrain plus `r` |
| Improvement | 47 | Lighthouse | validated corner marker; not representable |

Raw Polytopia tribe ID `4` is Bardur in the harvested version-122 saves. PolyVision's `Types.TRIBE.BARDUR` key is `2`, so the owned starting capital becomes `c:2`. Raw tribe ID `1` is the off-map Nature player in this corpus and never becomes a playable CSV tribe.

Improvement ID `1` is interpreted using explicit initial-state metadata:

- `capital=true`, owned by the sole in-bounds starting player, at that player's start coordinate: `c:2`;
- neutral and non-capital: `v:`;
- any other ownership/capital combination: validation error.

A ruin preserves its underlying terrain, for example `f:r` or `m:r`. Version-122 saves sometimes retain an ordinary resource underneath a ruin. Tribes CSV has only one suffix, so `r` takes precedence and that hidden resource remains available only in canonical JSON.

Modern Polytopia places four lighthouses at the map corners. Tribes has no lighthouse terrain, resource, or building token. ID `47` is accepted only when it is neutral, resource-free, non-capital, and on a corner; conversion retains the underlying terrain. This is an explicit format limitation, not an unknown-ID fallback.

## Coordinates and geometry

Canonical coordinates are `(x,y)`. CSV is written as:

```text
row = y
column = x
```

No transpose, rotation, mirroring, crop, padding, or resizing occurs. An 11×11 canonical map produces exactly 11 rows with 11 tokens each.

## Usage

Batch conversion from the repository root:

```powershell
python tools/polytopia_map_converter/convert_maps.py `
    --input data/polytopia_maps/parsed `
    --output pol_env/Tribes/levels/phase1_pool_bardur_real `
    --manifest data/polytopia_maps/conversion_manifest.csv `
    --java-validate
```

Single file:

```powershell
python tools/polytopia_map_converter/convert_maps.py `
    --input data/polytopia_maps/parsed/map_000001.json `
    --output test_map.csv `
    --manifest data/polytopia_maps/test_conversion_manifest.csv
```

Use `--overwrite` to regenerate existing files and `--verbose` to print every successful filename. Without `--overwrite`, an existing CSV is skipped only when its bytes, CSV SHA-256, canonical `map_sha256`, and existing manifest record all agree. Any inconsistency fails clearly.

Batch processing analyzes all JSON files first, reports observed IDs and duplicate canonical hashes, then validates and converts each file independently. One malformed JSON does not prevent the remaining files from being analyzed or converted. The command exits non-zero if anything fails.

## Validation

Canonical validation requires:

- schema version 1;
- positive square dimensions and exactly `width * height` unique in-bounds coordinates;
- at least one player and exactly one in-bounds starter;
- Bardur raw tribe ID 4 for the Phase-1 starter;
- exactly one owned capital matching the starter's player ID and coordinates;
- supported terrain, resource, improvement, player-tribe, and owner IDs;
- strict city/village/ruin/lighthouse combinations;
- no starting roads, water routes, flooding, or non-default tile skins that the CSV cannot encode;
- a valid canonical map hash and source identity.

Generated CSV is independently reparsed before its atomic temporary-file rename. Validation requires exact dimensions, `<terrain>:<optional-suffix>` syntax, Java-supported characters, and exactly one `c:2`.

With `--java-validate`, `CanonicalCsvValidator.java` loads every CSV through the existing `LevelLoader`, confirms dimensions and a single Bardur capital, and compares every Java-loaded coordinate against the canonical JSON for all CSV-representable terrain, resources, villages, ruins, and the capital. Its `(y,x)` board lookup explicitly protects against coordinate transposition.

## Manifest

The default manifest is:

```text
data/polytopia_maps/conversion_manifest.csv
```

It is outside the training-map directory so a `*.csv` level glob cannot load it. Columns are:

```text
index, source_json, source_state_filename, source_state_sha256,
map_sha256, csv_filename, csv_sha256, width, height, game_version,
seed, tribe, capital_x, capital_y, validation_status
```

The local real-map dataset and manifest remain covered by the repository's existing `data/polytopia_maps/` ignore rule. Generated training CSVs are written to:

```text
pol_env/Tribes/levels/phase1_pool_bardur_real/
```

## Tests

```powershell
python -m unittest discover -s tools/polytopia_map_converter/tests -v
```

Tests cover schema rejection, every mapping, precedence, unsupported IDs, capital count, strict lighthouse placement, coordinate orientation, 11×11 dimensions, duplicate hashes, deterministic bytes, resume/overwrite behavior, manifest hashes, and a real canonical fixture when available.

## Current corpus and limitations

The current 256 version-122 maps contain only Field, Mountain, Forest, Game, Metal, Fruit, cities/villages, ruins, and corner lighthouses. Coast/Ocean/Crop/Fish/Whale mappings are implemented and tested but not exercised by this Drylands corpus. Unknown IDs always fail.

The genuine pool is 11×11 while existing synthetic Phase-1 maps and models are 12×12. This converter does not change the default pool or solve observation/action/checkpoint compatibility. Existing 12×12 models must not be assumed compatible with these maps; that belongs to the next stage.
