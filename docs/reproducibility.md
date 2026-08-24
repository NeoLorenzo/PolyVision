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

The [Phase 1 scripted-opening audit](results/Phase1_Scripted_Opening_Audit.md) is part of the environment fingerprint. Seed-1 used historical `v1_mixed_capital_regression`: training was 55.02% two-unit / 44.98% one-unit and validation 56.40% / 43.60%. Corrected `v2_guaranteed_two_unit` passed 5,517/5,517 maps and is emitted in reset/step info, checkpoint sidecars, evaluation metadata, and human provenance. Checkpoint compatibility requires this field; the historical checkpoint is rejected against v2 by default. New evidence requires training from scratch and a separately identified evaluation.

Use `tools/evaluate_phase1.py` for canonical Phase 1 batch evaluation. Its deterministic schedule is derived from the top-level seed, canonical map identity, replicate, and policy. Environment episode seeds are identical across policies for a given map/replicate; stochastic policies receive separate recorded policy RNG seeds. PPO sampling uses a per-episode PyTorch generator, while random legal uses a per-episode NumPy generator, so unrelated global RNG consumption cannot silently change the action stream.

The full validation suite contains 3,000 episodes: one PPO-argmax and one visible-greedy episode on each of 250 maps, plus five PPO-sampled and five random-legal episodes per map. The evaluator verifies every selected file against `split_manifest.json`, rejects training-identity overlap, verifies runtime map hashes, and fails on missing, repeated, or unexpected schedule entries. Stochastic headline statistics operate on 250 per-map replicate means rather than treating 1,250 episodes as independent maps. Bootstrap and paired-comparison RNG seeds are derived reproducibly from the evaluation seed.

Validation is development evidence and may influence later decisions. Test evaluation requires the explicit `--pool test --confirm-test` guard and should happen only after freezing the checkpoint and protocol. Use multiple independent training seeds before a final claim. Separate policy-visible baselines from full-visibility diagnostics and distinguish training summaries, development validation, pristine test, and human-challenge aggregates.

The [evaluation guide](evaluation.md) lists recommended metrics. The [historical benchmark registry](history/model-run-benchmark-log.md) illustrates why interface and protocol metadata are necessary.

## Frozen Reference Benchmark Provenance (Phase 1 v3 Seed3 16M)

The authoritative frozen reference benchmark for Phase 1 optimization is fully specified by the following provenance records:

| Parameter | Authoritative Value |
|---|---|
| **Run directory** | `runs/Tribes-v0__Phase1-Scientific-Train-V3-Seed3__3__1787526811` |
| **Model file** | `Phase1-Scientific-Train-V3-Seed3.cleanrl_model` |
| **Model SHA-256** | `A6803CB00E08522819FDEB74BEDE8816CF131C1F28244C12D44F3EEFB9D2C234` |
| **Sidecar SHA-256** | `0CE2B01464A47506C19BF5C34B4E39F294C992AEA0607AB3755F15D5E10DB736` |
| **Training seed** | `3` |
| **Total timesteps** | 16,000,000 |
| **Training commit** | `3bdabaa4f1de8236b5bc80f1f0fbadbd1905282e` |
| **Environment version** | `v3_corrected_turn_economy` |
| **Opening version** | `v2_guaranteed_two_unit` |
| **Train split identity (5,000 maps)** | `a99b309a3020704e4a95886d5ab3f8b6f7ed00931ee111e3117702009605e8c4` |
| **Validation split identity (250 maps)** | `a56e74c952ad3b08d4645fc16d66ce0f17e0589fea2fe9042f7cd293669eac72` |
| **Test split identity (250 maps)** | `8a8e0f784535e8169ffb829f2e3058b60bcd7493277912dd995ff17b217f4b94` |
| **Human benchmark split identity (17 maps)** | `adbe5a707392aad15342f3c595a78abeeb85cb0ce77b91a8d9339276a2674a69` |
| **Canonical validation evaluation** | `outputs/evaluations/20260824_phase1_v3_seed3_16m_validation_canonical` |
| **Canonical pristine test evaluation** | `outputs/evaluations/20260824_phase1_v3_seed3_16m_pristine_test` |
| **Human benchmark agent evaluation** | `outputs/evaluations/20260824_phase1_v3_seed3_16m_human_benchmark_argmax` |

See [Phase 1 v3 Seed3 16M Reference Run](results/Phase1_V3_Seed3_16M_Reference_Run.md) for full evaluation metrics and paired comparisons.

Human benchmark attempt files additionally record stable map hashes, episode seed, interface contract, reward/filter settings, Git provenance, shaped return, curated terminal metrics, and every chosen stable global ID. Preserve the first completed attempt separately from later replays. `outputs/human_benchmark/summary.json` is derived evidence; individual attempt files and the append-only event index are the durable source. See [Human benchmark](human-benchmark.md).
