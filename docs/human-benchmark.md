# Human benchmark

> **Current Benchmark Progress (1/17 Completed):** 1 of 17 benchmark maps has a completed canonical first attempt (`map_004393.csv`). Human Turn-10 SPT reached **27**, while the frozen reference agent (Phase 1 v3 Seed3 16M PPO argmax) scored **20** on the same map ($+7$ SPT advantage for human). This single result is an illustrative $n=1$ anecdotal comparison and must **not** be interpreted as a statistically generalizable claim about human versus agent capability. 16 benchmark maps remain unplayed.

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
- the same corrected, fail-closed two-unit scripted opening through the start of Turn 2;
- Bardur/tribe 0 control and solo/no-opponent mode;
- Java/Py4J mechanics and the wrapper's Turn-10 truncation;
- the same reward/filter configuration;
- the same current geometry, action catalog, canonicalizer, legal-slot capacity, and feature contract;
- the exact legal global IDs in `legal_global_ids_padded[legal_action_valid_mask]`;
- the corresponding row from `legal_action_features_padded` for each legal action;
- every active Phase 1 action filter, because neither UI nor registry reimplements filtering;
- execution through `env.step(global_id)` and the wrapper's maintained global-ID-to-Java-action mapping.

For `actor_mode=legal_features` models (such as the Phase 1 v5 PARITY002 reference model), the official interface translates both the flattened observation and each legal action's 47-dimensional feature row into concise, structured human-readable annotations.

### Tactical Map and Movement Display

The official terminal interface renders a high-legibility tactical map and source-grouped action menu:
- **ANSI-colored tactical board**: Uses background colors for terrain (forest=green, mountain=gray, water=blue, city=cyan, village=yellow, fog=dark) and high-contrast foreground glyphs for occupants, terrain cues, and resources (`1`, `2`..=unit, `T`=forest, `C`=city, `V`=village, `A`=animal, `F`=fruit, `H`=fish, `W`=whale, `O`=ore, `P`=crops, `R`=ruin). Forests use green background with brown/tan text cues. Colors and glyphs are purely cosmetic presentation transforms of the PPO observation array.
- **Monochrome fallback**: Automatically used when output is redirected, `isatty()` is False, or `NO_COLOR` is set. Renders clear textual cues (`T1`, `C2`, `Ta`, `Tf`, `T`, `F`, `M`, `~`, `?`) without ANSI escape codes, ensuring `T` unambiguously denotes forest and `F` denotes fruit.
- **Visible unit numbering**: Numbered visible unit labels (`1`, `2`, `3`, ...) are assigned in deterministic `(x, y)` spatial tile order from the PPO observation. The same numbers identify source units on the board and in the movement menu.
- **Movement grouping and directional arrows**: Movement actions are grouped under their source unit (`MOVES - UNIT 1 at (6, 4)`), displaying directional arrows (`↑`, `↓`, `←`, `→`, `↖`, `↗`, `↙`, `↘` with ASCII fallback `N`, `S`, `W`, `E`, `NW`, `NE`, `SW`, `SE`) derived deterministically from source/destination coordinates. Original legal-slot order is strictly preserved within each group without desirability ranking or recommendation.
- **Compact annotations**: Model-visible feature rows from `legal_action_features_padded` are compressed into single-line semantic descriptors (e.g. `reveal +4 | adjacent fog 4 | away from capital` or `pop +1 | city progress 50% | level-up ready`).

Informational parity is strictly preserved: no raw Java state, hidden fog information, full-visibility data, or policy logits/recommendations are ever consulted.

Run the maintained parity validator after interface changes:

```powershell
python tools/validate_human_benchmark_parity.py --maps 3 --states-per-map 5
```

It creates paired wrappers for identical map/seed inputs, compares observations and interface contracts, proves that the human menu equals the legal-slot tensors, checks feature tensor shape/metadata and slot associations, checks stable-ID/raw-Java-action resolution and execution, compares rewards and horizon signals, completes one Turn-10 path, and audits the official presentation module for forbidden APIs.

## Information boundary

Official benchmark mode derives its state display solely from the 6,424-value flattened policy observation (under `v5_human_information_parity`) and the policy-visible legal action tensors (`legal_global_ids_padded`, `legal_action_valid_mask`, `legal_action_features_padded`).

It shows:
- fog-respecting terrain, visible resources, unit occupancy, and exact owned city slots reconstructed from the observation array;
- complete model-visible economy/city scalars and exact per-city state (stars, SPT, city levels, population, population need, production, unit capacity, and researched technologies);
- action annotations derived deterministically from that action's exact feature row in `legal_action_features_padded`.

It does not display recommendations, rankings, logits, values, oracle distances, predicted rewards, or hidden-map facts. This is strict informational parity with the PPO actor, not ordinary unrestricted Polytopia play.

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

### Current Benchmark Status

- **Pool Progress:** 1 / 17 completed canonical first attempts (16 unplayed).
- **Completed Map:** `map_004393.csv`
- **Human Turn-10 SPT:** 27
- **PPO Argmax Turn-10 SPT (v3 Seed3 16M on same map):** 20
- **Advantage:** Human $+7$ SPT on `map_004393.csv`

The full 17-map PPO argmax baseline is recorded at `outputs/evaluations/20260824_phase1_v3_seed3_16m_human_benchmark_argmax` (mean 16.71 SPT, median 17.00 SPT, 95% CI [14.94, 18.47]).

Automated workflow tests use `--synthetic-smoke` with an explicit non-canonical output directory. Such attempts carry `participant_kind: synthetic_test` and are excluded from every human statistic.

The result format is checkpoint-independent: stable map hashes and episode configuration allow future PPO evaluations to join against human results without changing or replacing the human record.

## Interpretation

Human benchmark maps may be replayed by people and evaluated repeatedly by future models, and results may influence development. They therefore remain permanently separate from the pristine test pool. Report human challenge results, scientific test generalization, and privileged-oracle diagnostics as different evidence categories.

The current single-map human result ($n=1$) is strictly an illustrative anecdotal comparison and does not constitute generalizable evidence of overall human versus agent capability. Furthermore, beating this benchmark does not imply general Polytopia or full-game strength.
