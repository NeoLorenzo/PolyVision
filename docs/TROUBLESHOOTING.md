# Troubleshooting Log

This file tracks real issues we hit during training/analysis and the exact fixes that worked.

## 2026-05-14: W&B `global_step` Charts Blank For One Run

### Symptoms

- Charts in W&B dashboard were not working as expected for a synced run.
- After checking axes, charts worked with X-axis = `Step` but were blank with X-axis = `global_step`.

### What Was Tried

1. Synced run from local storage with beta sync.
2. Noticed axis behavior (`Step` worked, `global_step` blank).
3. Re-synced the same run again.
4. Issue persisted.

### Resolution

1. Deleted the problematic run in W&B dashboard.
2. Created/synced as a new run.
3. Charts then worked correctly.

### Takeaway

- If `global_step` charts are blank for one run but `Step` works, re-syncing alone may not fix it.
- Deleting the bad dashboard run and creating/syncing a new run can resolve the issue.

### Useful Command

```powershell
wandb beta sync ".\run-pvh5hss7.wandb"
```

If you delete a run and need to upload the same local file again, use a new run ID.
