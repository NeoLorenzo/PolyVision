# Human benchmark

The Phase 1 human benchmark is a persistent human-versus-agent challenge on the 17 maps in `levels/phase1_pool_bardur_real/human_benchmark/`. It measures decision quality inside PolyVision's constrained Bardur Turn-10 task. It is not the pristine scientific test set and does not measure unrestricted full-game Polytopia skill.

Run the ordinary workflow from the repository root:

```powershell
python tools/human_benchmark.py
```

The command verifies the split assignment and hashes, finds maps without a completed first human attempt, selects one uniformly at random, records an attempt-start event, launches the exact map, and saves the final result. Use `--selection-seed N` for reproducible map selection and `--episode-seed N` to set the recorded environment reset seed.

## First attempts and replays

The canonical human statistic for each map is its first completed attempt. A quit, interruption, crash, or error does not produce a score and leaves the map eligible, but the started/aborted/error attempt remains visible in the history. Completed attempt files are immutable and are never overwritten.

After a first completion, deliberately replay a map by filename, stem, or unique canonical-hash prefix:

```powershell
python tools/human_benchmark.py --replay map_000354
```

Replays are numbered attempts and retained separately. The summary reports first-attempt, latest-attempt, and best-attempt aggregates; first-attempt performance remains the default benchmark.

When all maps have a first completion, the ordinary command reports that the pool is complete instead of silently choosing a replay.

## Human/model parity

The official UI is a presentation layer over `TribesGymWrapper`, the same authoritative environment used by PPO. Both decision-makers share:

- the identical CSV map and episode seed;
- the deterministic scripted opening through the start of Turn 2;
- Bardur/tribe 0 control and solo/no-opponent mode;
- Java/Py4J mechanics and the wrapper's Turn-10 truncation;
- the same reward/filter configuration;
- the same current geometry, action catalog, canonicalizer, legal-slot capacity, and feature contract;
- the exact legal global IDs in `legal_global_ids_padded[legal_action_valid_mask]`;
- every active Phase 1 action filter, because neither UI nor registry reimplements filtering;
- execution through `env.step(global_id)` and the wrapper's maintained global-ID-to-Java-action mapping.

The human does not need to inspect the 42-dimensional legal-action feature tensor. Stable IDs are decoded into neutral descriptions using public catalog structure: for example, `Move unit (x, y) -> (x, y)` or `Research ORGANIZATION`. The decoder never reads raw Java action dictionaries.

Run the maintained parity validator after interface changes:

```powershell
python tools/validate_human_benchmark_parity.py --maps 3 --states-per-map 5
```

It creates paired wrappers for identical map/seed inputs, compares observations and interface contracts, proves that the human menu equals the legal-slot tensors, checks stable-ID/raw-Java-action resolution and execution, compares rewards and horizon signals, completes one Turn-10 path, and audits the official presentation module for forbidden APIs.

## Information boundary

Official benchmark mode derives its state display only from the 505-value flattened policy observation. It shows fog-respecting terrain, resources, visible unit occupancy, and observation-derived economy scalars. Action descriptions come only from selectable stable global IDs and catalog metadata. It does not display recommendations, rankings, logits, values, oracle distances, predicted outcomes, or hidden-map facts.

The following paths are prohibited in official mode:

- `tribes_env._last_obs` or other direct Java-observation dictionaries;
- raw structured action dictionaries or raw action `repr` details;
- `get_observation(full_visibility=True)` / `observationJsonFull()`;
- ANSI, RGB, or Java/Swing renderers;
- debug info mode and privileged/no-fog/oracle utilities.

`tools/play_human_t10_wrapper.py` remains available for ad hoc play. Its default UI uses the safe shared policy presentation. `--show-ansi-map` and `--render-java` require the explicit `--unsafe-debug-ui` acknowledgement and their output must never be recorded as official benchmark evidence.

## Results and registry

Canonical evidence lives under:

```text
outputs/human_benchmark/
    results.jsonl
    summary.json
    attempts/
        <attempt-id>.json
```

Attempt files contain status/timestamps, map and split identities, episode configuration, Git provenance, environment/interface contract, curated final metrics, shaped return, and the chosen stable-global-ID history. `results.jsonl` is an append-only lifecycle event index. `summary.json` is deterministically rebuilt from attempt files and answers pool completion plus first/latest/best SPT aggregates.

Inspect or regenerate the summary without playing:

```powershell
python tools/human_benchmark.py --summary
```

Automated workflow tests use `--synthetic-smoke` with an explicit non-canonical output directory. Such attempts carry `participant_kind: synthetic_test` and are excluded from every human statistic.

The result format is checkpoint-independent: stable map hashes and episode configuration allow future PPO evaluations to join against human results without changing or replacing the human record.

## Interpretation

Human benchmark maps may be replayed by people and evaluated repeatedly by future models, and results may influence development. They therefore remain permanently separate from the pristine test pool. Report human challenge results, scientific test generalization, and privileged-oracle diagnostics as different evidence categories. Beating this benchmark does not imply general Polytopia or full-game strength.
