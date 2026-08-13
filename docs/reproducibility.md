# Reproducibility

A PolyVision result is identified by more than a model file and seed. Map contents, geometry, action semantics, reward settings, and validation code can all change behavior.

## Record for every run

- Git commit SHA and whether the worktree was dirty
- Python, PyTorch, Gymnasium, Py4J, and JDK versions
- complete trainer arguments and relevant `POLYVISION_*` environment variables
- training seed, validation seed, map-selection mode, and episode-seed policy
- exact ordered map list, per-file hashes, and aggregate pool identity
- actor mode, geometry, observation dimension, global action-space size, legal capacity, feature version/dimension, and catalog fingerprint
- reward/filter toggles, including terminal SPT and resource-upgrade filtering
- checkpoint path, step, SHA-256, and `.action_interface.json` sidecar
- W&B/TensorBoard run identity and raw evaluation outputs

## Seeds and map selection

PPO seeds Python, NumPy, and PyTorch and enables deterministic cuDNN behavior by default. Each spawned environment receives a derived reset seed. The wrapper maintains separate deterministic streams for episode seeds and map selection.

`round_robin` selection uses a seed-derived pool offset and episode index. `seeded_random` samples with its dedicated RNG. Neither mode makes comparisons fair unless policies receive the same ordered maps and episode seeds.

## Pool identity

`compute_level_pool_identity()` hashes each selected CSV and a deterministic list of its relative path, size, and SHA-256. The strict validator includes this identity in its cache fingerprint. For published evaluation, retain the manifest itself rather than relying only on an aggregate hash.

The authoritative `split_manifest.json` freezes 5,000 training, 250 validation, 250 test, and 17 human-benchmark assignments at split seed `20260813`. The split is disjoint by canonical `map_sha256` and exact CSV SHA-256, not just filename. Run `python tools/split_phase1_map_pool.py` before research runs to verify counts, hashes, aggregate pool identities, and filesystem agreement.

Never reshuffle established assignments when new maps arrive. New converter output belongs in `data/polytopia_maps/incoming_csv/` until an explicit dataset-contract update assigns unique, previously unused maps. Training maps cannot later become honestly held-out; validation maps used for development cannot later become pristine test maps.

## Interface and checkpoints

Saved sidecars contain required compatibility fields:

- map width and height;
- observation and action-space dimensions;
- action-catalog fingerprint;
- actor mode;
- legal-action feature version and dimension;
- catalog and canonicalizer versions;
- maximum legal-action slots.

Current evaluators reject missing or different values before loading model tensors. Preserve sidecars with checkpoints. The trainer saves model weights only; it does not save optimizer/RNG state for exact training continuation.

## Validator cache

Successful action-interface checks are cached under `.cache/action_validator/`. The fingerprint includes validator settings, environment contract, pool files and identity, and hashes of key Python/Java interface files. Use `--force-revalidate-action-interface` when auditing a result or when untracked external changes could evade the fingerprint.

## Evaluation protocol

Use multiple independent training seeds. Select models on validation, then evaluate the frozen test pool only after selection. Pair policies on identical map/episode seeds; choose deterministic or stochastic action selection in advance; report uncertainty and raw episode records. Separate visible-information baselines from full-visibility diagnostics and distinguish training summaries, development validation, pristine test, and human-challenge aggregates.

The [evaluation guide](evaluation.md) lists recommended metrics. The [historical benchmark registry](history/model-run-benchmark-log.md) illustrates why interface and protocol metadata are necessary.

Human benchmark attempt files additionally record stable map hashes, episode seed, interface contract, reward/filter settings, Git provenance, shaped return, curated terminal metrics, and every chosen stable global ID. Preserve the first completed attempt separately from later replays. `outputs/human_benchmark/summary.json` is derived evidence; individual attempt files and the append-only event index are the durable source. See [Human benchmark](human-benchmark.md).
