# train_vla_weighted_bc.md

Weighted BC fine-tuning of SmolVLA using IQL advantage weights. Runs in `lerobot-env`.

Script: `scripts/train/train_vla_weighted_bc.py`

---

Runs in **lerobot-env** (`act-lerobot`). See [setup](../workflow/00_setup.md).

## Overview

Thin wrapper around `lerobot-train` that monkey-patches `SmolVLAPolicy.forward()` to apply
per-sample IQL advantage weights before loss aggregation. Requires `iql_weight` column in the
dataset parquet (added by `compute_iql_weights.py`).

**How the patch works:**
1. Standard SmolVLA forward computes per-element losses `(B, chunk_size, action_dim)` and masks padding
2. Patch intercepts `losses_after_rm_padding`, multiplies by `iql_weight` from batch `(B,)`
3. Computes weighted mean instead of plain mean

If `iql_weight` is absent from the batch, patch falls back to standard BC — safe to use with
datasets that don't have the column.

VIP ResNet50 (used for IQL) and SigLIP (used by SmolVLA) are separate encoders.
`iql_weight` is just a scalar in parquet; SmolVLA training does not affect Q/V networks.

---

## Pipeline step 4

This is the final step after:
```
1. prepare_reward_dataset.py  → VIP rewards in next.reward
2. train_iql_critics.py       → train Q(s,a) and V(s)
3. compute_iql_weights.py     → iql_weight column in parquet
4. train_vla_weighted_bc.py   → weighted BC (this script)
```

---

## CLI

All `lerobot-train` arguments are supported:

```bash
act-lerobot  # activate lerobot-env
cd /path/to/so101-lab

python scripts/train/train_vla_weighted_bc.py \
    --policy.path=outputs/smolvla_teleop_only/checkpoints/016000/pretrained_model \
    --dataset.repo_id=local/figure_shape_placement_v5_vip \
    --dataset.root=data/recordings/figure_shape_placement_v5_vip \
    --output_dir=outputs/smolvla_weighted_bc_v1 \
    --batch_size=8 --steps=10000 --save_freq=2000 --log_freq=200 --eval_freq=0 \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
```

**Extra logged metrics:**
- `loss_unweighted` — standard BC loss (for comparison)
- `iql_weight_mean` — mean weight in batch

---

## Notes

**β (temperature) for weights** is set at `compute_iql_weights.py` step, not here. Use
`--dry-run` in compute step to inspect weight distribution before writing.

**`--eval_freq=0`** recommended — SmolVLA eval requires sim and is not available in lerobot-env.

**`--rename_map`** — needed if dataset uses `observation.images.top`/`wrist` keys and SmolVLA
checkpoint was trained with `camera1`/`camera2` names.
