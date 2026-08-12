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

Training and evaluation pools should be disjoint by map hash, not just filename. Canonical `map_sha256` identifies equivalent initial maps even when source save bytes or names differ.

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

Use multiple independent training seeds and a frozen held-out pool. Pair policies on identical map/episode seeds; choose deterministic or stochastic action selection in advance; report uncertainty and raw episode records. Separate visible-information baselines from full-visibility diagnostics and distinguish final training summaries from evaluation aggregates.

The [evaluation guide](evaluation.md) lists recommended metrics. The [historical benchmark registry](history/model-run-benchmark-log.md) illustrates why interface and protocol metadata are necessary.
