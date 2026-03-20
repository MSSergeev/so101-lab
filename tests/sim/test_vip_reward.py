"""Test VIP reward on demo episodes — check that reward increases toward goal.

Usage (Isaac Lab env):
    # Default: mean mode, final frames as goals
    python tests/sim/test_vip_reward.py

    # Min distance mode
    python tests/sim/test_vip_reward.py --goal-mode min

    # Labels from labeled dataset, images from original (full-res)
    python tests/sim/test_vip_reward.py --use-labeled \
        --goal-dataset data/recordings/figure_shape_placement_v4 \
        --label-dataset data/recordings/figure_shape_placement_v4_labeled

    # Specific episodes, wrist camera
    python tests/sim/test_vip_reward.py --episodes 0 10 50 --camera observation.images.wrist
"""

import argparse
import sys

def _get_lerobot_src() -> str:
    import os
    from pathlib import Path
    if src := os.environ.get("LEROBOT_SRC"):
        return os.path.expanduser(src)
    env_file = Path(__file__).parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("LEROBOT_SRC="):
                return os.path.expanduser(line.split("=", 1)[1].strip())
    raise RuntimeError("LEROBOT_SRC not set. Add it to .env or set the environment variable.")

_lerobot_src = _get_lerobot_src()
if _lerobot_src not in sys.path:
    sys.path.insert(0, _lerobot_src)

import numpy as np
import pandas as pd

from so101_lab.rewards.vip_reward import VIPReward


def main():
    parser = argparse.ArgumentParser(description="Test VIP reward on demo episodes")
    parser.add_argument("--dataset", type=str, default="data/recordings/figure_shape_placement_v4",
                        help="Dataset to test on (iterate frames)")
    parser.add_argument("--goal-dataset", type=str, default=None,
                        help="Dataset for goal images (default: same as --dataset)")
    parser.add_argument("--camera", type=str, default="observation.images.top")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Episode indices to test (default: 5 evenly spaced)")
    parser.add_argument("--sample-every", type=int, default=20, help="Sample every N frames")
    parser.add_argument("--n-goal-frames", type=int, default=5,
                        help="Final frames per episode for goal (ignored with --use-labeled)")
    parser.add_argument("--goal-mode", type=str, default="mean", choices=["mean", "min"],
                        help="mean: single averaged goal; min: closest of all goals")
    parser.add_argument("--use-labeled", action="store_true",
                        help="Use frames with next.reward>0.5 as goals")
    parser.add_argument("--label-dataset", type=str, default=None,
                        help="Dataset with next.reward labels (default: same as --goal-dataset)")
    args = parser.parse_args()

    goal_path = args.goal_dataset or args.dataset

    # 1. Load VIP
    vip = VIPReward(
        goal_path,
        device="cuda",
        image_key=args.camera,
        n_goal_frames=args.n_goal_frames,
        goal_mode=args.goal_mode,
        use_labeled=args.use_labeled,
        label_dataset_path=args.label_dataset,
    )

    # 2. Load test dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="local", root=args.dataset)
    ep_df = pd.read_parquet(f"{args.dataset}/meta/episodes/chunk-000/file-000.parquet")
    n_episodes = len(ep_df)

    # 3. Pick episodes
    if args.episodes is not None:
        ep_indices = args.episodes
    else:
        ep_indices = np.linspace(0, n_episodes - 1, 5, dtype=int).tolist()

    print(f"\nTesting episodes: {ep_indices}")
    print(f"Sampling every {args.sample_every} frames")
    print()

    # 4. Run
    results = []
    for ep_idx in ep_indices:
        row = ep_df.iloc[ep_idx]
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        ep_len = end - start

        rewards = []
        frame_nums = []
        for i in range(start, end, args.sample_every):
            frame = ds[i]
            img = frame[args.camera]  # (C, H, W) float [0, 1]
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            r = vip.compute_reward({args.camera: img_np})
            rewards.append(r)
            frame_nums.append(i - start)

        # Always include last frame
        if frame_nums[-1] != ep_len - 1:
            frame = ds[end - 1]
            img = frame[args.camera]
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            r = vip.compute_reward({args.camera: img_np})
            rewards.append(r)
            frame_nums.append(ep_len - 1)

        # Print per-frame rewards
        print(f"--- Episode {ep_idx} ({ep_len} frames) ---")
        r_min, r_max = min(rewards), max(rewards)
        for fn, r in zip(frame_nums, rewards):
            bar_len = int((r - r_min) / (r_max - r_min + 1e-8) * 30)
            bar = "#" * bar_len
            print(f"  frame {fn:4d}: {r:8.4f}  {bar}")

        # Summary
        n = len(rewards)
        first_q = np.mean(rewards[:n // 4 + 1])
        last_q = np.mean(rewards[-(n // 4 + 1):])
        trend = "OK" if rewards[-1] > rewards[0] else "WEAK"
        results.append(trend)
        print(f"  first={rewards[0]:.2f}  last={rewards[-1]:.2f}  "
              f"first_quarter_avg={first_q:.2f}  last_quarter_avg={last_q:.2f}  [{trend}]")
        print()

    # Overall summary
    ok = results.count("OK")
    print(f"=== Summary: {ok}/{len(results)} episodes with positive trend ===")


if __name__ == "__main__":
    main()
