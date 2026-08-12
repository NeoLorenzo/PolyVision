# PolyVision Polytopia State Converter

This tool converts compressed Polytopia `.state` saves directly into stable, schema-versioned JSON describing the complete generated initial map. It is the first ingestion stage only:

```text
.state -> canonical JSON
```

It does not generate PolyVision CSV, translate Polytopia IDs, or modify any training/runtime code.

## Requirements and pinned parser

- Go 1.23.2 or newer;
- `github.com/samuelyuan/polytopiamapmodelgo` pinned in `go.mod` to commit `d09f4e08b66dadb8b9f1fa12fdad50a4150b6c5e` (pseudo-version `v0.0.0-20250928064600-d09f4e08b66d`);
- its LZ4 dependency, pinned transitively in `go.sum`.

Dependencies are vendored, so normal builds are reproducible and do not require fetching a moving branch. The vendored parser has a narrow documented compatibility adaptation for harvested game-version-122 saves:

- consume the version-122 header byte before map dimensions;
- consume the optional observed `0xff` marker before version-122 player names;
- expose `ReadPolytopiaCompressedInitialFile`, which stops after the initial header, tiles, and players.

The last point is intentional. Upstream `ReadPolytopiaCompressedFile` continues into the current map and actions. Current version-122 saves have a changed current-header layout, but this converter must neither use nor depend on that later state. The adaptation therefore avoids inventing current-state semantics and provides exactly the initial-only parser needed here.

## Initial state, not current state

Polytopia saves contain both generated state and later/current state. This converter reads only:

- the initial map header;
- `InitialTileData`;
- `InitialPlayerData`.

It intentionally ignores `TileData`, `PlayerData`, units/actions, and other current-game mutations. The canonicalization API accepts only initial tiles and players, comments call out the distinction, and regression tests provide deliberately different initial/current values.

Compressed `.state` files are accepted directly. Decompression remains in memory; no `.decomp` files are written.

## PowerShell usage

From the repository root, single-file conversion:

```powershell
go run ./tools/polytopia_state_converter `
    --input data/polytopia_maps/raw_states/map_000001.state `
    --output data/polytopia_maps/parsed/map_000001.json
```

Batch conversion:

```powershell
go run ./tools/polytopia_state_converter `
    --input data/polytopia_maps/raw_states `
    --output data/polytopia_maps/parsed
```

Options:

```text
--input PATH       Required .state file or directory
--output PATH      Required .json file or directory
--overwrite        Explicitly replace existing JSON
--pretty=false     Emit compact JSON (pretty JSON is the default)
--verbose          Show parser-worker diagnostics on failure
```

Batch mode reads only direct `*.state` children, sorts them lexically by filename, and maps each basename to `.json`. Existing outputs are skipped by default for resumability. One malformed file does not abort the rest of a batch; failures are collected, printed, and cause a non-zero final exit code. Single-file failures return immediately.

The recommended ignored dataset layout is:

```text
data/polytopia_maps/raw_states/map_000001.state
data/polytopia_maps/parsed/map_000001.json
```

The repository `.gitignore` already ignores `data/polytopia_maps/`, covering both raw and parsed datasets.

## Canonical JSON schema version 1

Each file has this stable top-level shape:

```json
{
  "schema_version": 1,
  "source": {
    "filename": "map_000001.state",
    "sha256": "...",
    "size_bytes": 1872
  },
  "map_sha256": "...",
  "game": {},
  "players": [],
  "tiles": []
}
```

`game` retains the actual game version, signed seed, map name, width, height, square size, preset, game type, difficulty, opponent count, game-mode bytes, disabled/unlocked tribe IDs, and selected tribe/skin pairs.

Each initial player retains:

- numeric player and tribe IDs;
- starting `x,y`;
- currency and score;
- number of cities;
- starting technology IDs;
- player skin ID.

Player/account names and account IDs are deliberately excluded. Players and technology IDs are deterministically sorted.

Each initial tile explicitly retains:

- `x,y`, sorted in row-major order (`y`, then `x`);
- raw terrain and climate IDs, altitude, and owner ID;
- explicit capital flag and capital coordinates;
- nullable raw resource ID;
- nullable raw improvement ID and all parsed improvement data: level, founded turn, populations, production, base score, border size, upgrade count, capital connection, city name, founded tribe ID, rewards, and rebellion fields;
- road, water-route, skin, flooded flag, and flooded value.

Unknown numeric resource/improvement/terrain/etc. IDs remain ordinary integers. No ID-to-name or ID-to-CSV mapping occurs in this stage.

## Hashes

`source.sha256` is SHA-256 over every byte of the compressed input `.state` file. `source.size_bytes` is the length of those same bytes.

`map_sha256` is SHA-256 over compact deterministic JSON containing exactly:

- hash schema version (`1`);
- map width and height;
- canonical sorted initial players;
- canonical row-major initial tiles, including terrain, climate, altitude, ownership, explicit capital data, resources, improvements, roads/routes, skin, and flooding.

It excludes filename, compressed bytes, seed, map name, timestamps, current state, actions, and profile/account data. Thus byte-different saves with identical generated map content receive the same map hash. Changing a retained player/tile field changes the hash.

## Validation and failures

Before JSON is written, the converter requires:

- readable positive game version;
- positive width and height;
- non-empty `InitialPlayerData`;
- exactly `height` initial tile rows;
- exactly `width` tiles in every row;
- exactly `width * height` tiles;
- every coordinate in bounds and present exactly once;
- improvement data whenever an improvement-present flag is set.

JSON is written through a temporary file and atomically renamed only after parsing, canonicalization, validation, serialization, and hashing succeed. Partial JSON is not emitted.

The upstream parser uses process-terminating errors for some corrupt binary input. To preserve batch resilience, the CLI parses each save in an isolated child process. Such a termination becomes a failure for that filename and the parent continues with the next save.

Common errors include an unsupported/corrupt header, LZ4 decompression failure, unexpected tile coordinates, malformed player data, missing/duplicate/out-of-bounds coordinates, and an existing output without `--overwrite` in single-file mode.

## Tests

Run from the repository root:

```powershell
go test ./tools/polytopia_state_converter/...
```

Tests cover initial-vs-current selection, dimensions, row-major ordering, coordinate validation, nullable/unknown resources, full improvement retention, raw SHA-256, map-hash equivalence and sensitivity, deterministic JSON, batch naming/order, resume/overwrite behavior, and a real parser integration check when `data/polytopia_maps/raw_states/map_000001.state` is locally available.
