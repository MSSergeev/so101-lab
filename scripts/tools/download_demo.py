"""Download demo datasets and checkpoints from Hugging Face.

Downloads pre-recorded datasets and trained policy checkpoints so you can
run evaluation or training without recording your own data first.

Usage:
    # Download everything (datasets + checkpoint)
    python scripts/tools/download_demo.py

    # Download only datasets
    python scripts/tools/download_demo.py --dataset easy
    python scripts/tools/download_demo.py --dataset medium
    python scripts/tools/download_demo.py --dataset all

    # Download only the checkpoint
    python scripts/tools/download_demo.py --checkpoint easy

Requires: lerobot-env (huggingface_hub, lerobot)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Registry: name -> (hf_repo_id, local_dir_relative_to_project_root, type)
DATASETS = {
    "easy": ("MSSerg/so101-easy-task-v1", "data/recordings/easy_task_v1"),
    "medium": ("MSSerg/so101-figure-shape-placement-v1", "data/recordings/figure_shape_placement_v1"),
}

CHECKPOINTS = {
    "easy": ("MSSerg/so101-smolvla-iql-easy-v1", "outputs/smolvla_iql_easy_v1"),
}


def download_dataset(name: str) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id, local_dir = DATASETS[name]
    dest = PROJECT_ROOT / local_dir

    if dest.exists() and any(dest.iterdir()):
        print(f"  skip: {dest} already exists")
        return

    print(f"  downloading dataset '{name}' from {repo_id} ...")
    ds = LeRobotDataset(repo_id=repo_id, root=dest)
    print(f"  done: {ds.meta.total_episodes} episodes, {len(ds)} frames -> {dest}")


def download_checkpoint(name: str) -> None:
    from huggingface_hub import snapshot_download

    repo_id, local_dir = CHECKPOINTS[name]
    dest = PROJECT_ROOT / local_dir

    if dest.exists() and any(dest.iterdir()):
        print(f"  skip: {dest} already exists")
        return

    print(f"  downloading checkpoint '{name}' from {repo_id} ...")
    snapshot_download(repo_id=repo_id, local_dir=dest)
    print(f"  done: -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download demo datasets and checkpoints from Hugging Face"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default=None,
        help="Dataset to download (default: download everything)",
    )
    parser.add_argument(
        "--checkpoint",
        choices=list(CHECKPOINTS.keys()) + ["all"],
        default=None,
        help="Checkpoint to download (default: download everything)",
    )
    args = parser.parse_args()

    # If nothing specified, download everything
    download_all = args.dataset is None and args.checkpoint is None

    # Datasets
    if download_all or args.dataset is not None:
        ds_names = list(DATASETS.keys()) if (download_all or args.dataset == "all") else [args.dataset]
        print("Datasets:")
        for name in ds_names:
            download_dataset(name)

    # Checkpoints
    if download_all or args.checkpoint is not None:
        ckpt_names = list(CHECKPOINTS.keys()) if (download_all or args.checkpoint == "all") else [args.checkpoint]
        print("Checkpoints:")
        for name in ckpt_names:
            download_checkpoint(name)

    # Print next steps
    print()
    print("Next steps:")
    print("  # Visualize dataset (rerun venv)")
    print("  python scripts/visualize_lerobot_rerun.py data/recordings/easy_task_v1 --episodes 0,1,2")
    print()
    print("  # Evaluate checkpoint (two terminals)")
    print("  # Terminal 1 (lerobot-env):")
    print("  python scripts/eval/smolvla_server.py --checkpoint outputs/smolvla_iql_easy_v1")
    print("  # Terminal 2 (isaaclab-env):")
    print("  python scripts/eval/eval_vla_policy.py \\")
    print("      --checkpoint outputs/smolvla_iql_easy_v1 \\")
    print("      --env figure_shape_placement_easy --episodes 10 --no-domain-rand --gui")
    print()
    print("  # Train your own policy on the downloaded dataset (lerobot-env)")
    print("  python scripts/train/train_act.py \\")
    print("      --dataset data/recordings/easy_task_v1 --name my_act_v1")


if __name__ == "__main__":
    main()
