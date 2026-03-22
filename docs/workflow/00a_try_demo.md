# Try it now (demo datasets + checkpoint)

Download pre-recorded datasets and a trained checkpoint to explore the pipeline without recording your own data.

**Prerequisites:** complete [00_setup.md](00_setup.md) first (Isaac Lab + LeRobot venvs, `.env` file). All HF assets are public — no authentication needed.

## What's available

| Asset | HF repo | Local path | Description |
|-------|---------|------------|-------------|
| Easy dataset | [MSSerg/so101-easy-task-v1](https://huggingface.co/datasets/MSSerg/so101-easy-task-v1) | `data/recordings/easy_task_v1` | 200 episodes, small spawn zone |
| Medium dataset | [MSSerg/so101-figure-shape-placement-v1](https://huggingface.co/datasets/MSSerg/so101-figure-shape-placement-v1) | `data/recordings/figure_shape_placement_v1` | 300 episodes, 4x spawn area |
| IQL checkpoint | [MSSerg/so101-smolvla-iql-easy-v1](https://huggingface.co/MSSerg/so101-smolvla-iql-easy-v1) | `outputs/smolvla_iql_easy_v1` | SmolVLA + IQL weighted BC, 86-88% SR on easy task |

## 1. Download

Requires `huggingface_hub` (already installed in both isaaclab-env and lerobot-env).

**Everything at once:**

```bash
python scripts/tools/download_demo.py
```

**Selective download:**

```bash
# Datasets only
python scripts/tools/download_demo.py --dataset easy
python scripts/tools/download_demo.py --dataset medium
python scripts/tools/download_demo.py --dataset all

# Checkpoint only
python scripts/tools/download_demo.py --checkpoint easy

# Combine
python scripts/tools/download_demo.py --dataset easy --checkpoint easy
```

Skips assets that already exist locally.

**Manual download** (without the script):

```bash
# Dataset
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('MSSerg/so101-easy-task-v1', root='data/recordings/easy_task_v1')
"

# Checkpoint
python -c "
from huggingface_hub import snapshot_download
snapshot_download('MSSerg/so101-smolvla-iql-easy-v1', local_dir='outputs/smolvla_iql_easy_v1')
"
```

## 2. Visualize episodes (rerun)

```bash
source venvs/rerun/bin/activate

python scripts/tools/visualize_episodes.py \
    --dataset data/recordings/easy_task_v1 \
    --episodes 0 1 2
```

## 3. Evaluate the checkpoint in sim

Requires Isaac Lab env + lerobot env (gRPC two-process setup).

```bash
eval "$(./activate_isaaclab.sh)"

python scripts/eval/eval_vla_policy.py \
    --checkpoint outputs/smolvla_iql_easy_v1 \
    --env figure_shape_placement_easy \
    --episodes 10 --max-steps 500 --no-domain-rand \
    --n-action-steps 15 --seed 1988042740 \
    --auto-server --gui --preview
```

Expected: ~86% success rate. Uses gRPC to run the model in a separate lerobot-env process — see [eval_vla_policy.md](../reference/eval_vla_policy.md) for details.

## 4. Train a policy on the demo dataset

```bash
eval "$(./activate_lerobot.sh)"

# ACT
python scripts/train/train_act.py \
    --dataset data/recordings/easy_task_v1 \
    --config configs/policy/act/baseline.yaml \
    --name my_act_test

# SmolVLA (requires smolvla deps in lerobot-env)
python scripts/train/lerobot-train \
    --dataset.repo_id=data/recordings/easy_task_v1 \
    --dataset.local_files_only=true \
    --output_dir=outputs/my_smolvla_test
```

See [03_bc_training.md](03_bc_training.md) for full training options and configs.

## Next steps

- [01_env_design.md](01_env_design.md) — understand the task environments
- [02_recording.md](02_recording.md) — record your own episodes
- [04_offline_rl.md](04_offline_rl.md) — train IQL critics for advantage-weighted BC
- [05_online_rl.md](05_online_rl.md) — fine-tune with PPO
