# Observations

`Tribes-v0` exposes a one-dimensional `float32` observation. Its size is derived from map geometry:

```text
observation_dim = 4 × (width × height) + 21
```

The active 11×11 pool therefore produces **505 values**.

## Layout

The vector preserves a legacy prefix and appends current visible-resource and economy features:

| Block | Size on 11×11 | Contents |
|---|---:|---|
| Terrain | 121 | Java observation terrain IDs, including fog markers. |
| Unit IDs | 121 | Board unit identifiers from the visible observation. |
| City IDs | 121 | Board city identifiers from the visible observation. |
| Legacy scalars | 6 | Bardur stars, score, city count, kills, engine tick, and active tribe ID. |
| Visible resources | 121 | Resource IDs normalized to `[0,1]`; fogged tiles are masked before normalization. |
| Economy/state scalars | 15 | Normalized stars/SPT/turn timing, key technologies, research count, city levels, and upgrade progress/readiness. |

The 15 appended scalars are current stars, current SPT, turn progress, two remaining-turn measures, Organization and Forestry flags, technology count, city count, average and maximum city level, mean and maximum upgrade progress, fraction of cities ready to level up, and any-level-up-available.

## Visibility boundary

Policy observations come from Java's normal `observationJson()` and respect fog of war. The visible resource block explicitly masks fogged resources. `TribesGymEnv.get_observation(full_visibility=True)` exists for diagnostics and audits; it must not be treated as the training observation or used by a fair visible-information baseline.

## Geometry and compatibility

The wrapper supports square, dimension-homogeneous pools. It derives the observation space and global action catalog from the first map and rejects later maps with different dimensions. Changing geometry changes observation dimension and usually the global action space and fingerprint, so checkpoints are geometry-specific.

The observation is a compact engineered vector, not pixels and not the entire Java state. Legal action tensors are carried in `info`, not concatenated into the observation. The critic consumes the 505-value state vector; legality-aware actors additionally consume legal IDs and, in `legal_features`, their feature vectors.

See [Actions](actions.md) for the policy interface and [Reproducibility](reproducibility.md) for checkpoint contract fields.
