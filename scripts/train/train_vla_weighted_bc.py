"""Weighted BC training for SmolVLA using IQL advantage weights.

Monkey-patches SmolVLAPolicy.forward() to multiply per-sample losses by
`iql_weight` from the dataset before aggregation. Everything else is
standard lerobot-train.

Requires `iql_weight` column in dataset parquet (see compute_iql_weights.py).

Usage:
    eval "$(./activate_lerobot.sh)"
    python scripts/train/train_vla_weighted_bc.py \
        --policy.type=smolvla \
        --dataset.repo_id=data/recordings/figure_shape_placement_v5_vip \
        ...

    All lerobot-train arguments are supported.
"""

import logging

import torch
from torch import Tensor

_original_forward = None


def _weighted_forward(self, batch: dict[str, Tensor], noise=None, time=None):
    """SmolVLAPolicy.forward with IQL weight support."""
    loss, loss_dict = _original_forward(self, batch, noise, time)

    if "iql_weight" not in batch:
        return loss, loss_dict

    # Recompute loss with per-sample weights.
    # losses shape after forward: (B, chunk_size, action_dim)
    losses = loss_dict["losses_after_rm_padding"]
    weights = batch["iql_weight"].to(losses.device).float()  # (B,)
    weights = weights[:, None, None]  # (B, 1, 1) for broadcasting

    weighted_loss = (losses * weights).mean()

    loss_dict["loss_unweighted"] = loss.item()
    loss_dict["loss"] = weighted_loss.item()
    loss_dict["iql_weight_mean"] = weights.mean().item()

    return weighted_loss, loss_dict


def patch_smolvla():
    """Monkey-patch SmolVLAPolicy.forward for weighted BC."""
    global _original_forward
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    _original_forward = SmolVLAPolicy.forward
    SmolVLAPolicy.forward = _weighted_forward
    logging.info("Patched SmolVLAPolicy.forward for weighted BC (iql_weight)")


def main():
    from lerobot.scripts.lerobot_train import train
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    patch_smolvla()
    train()


if __name__ == "__main__":
    main()
