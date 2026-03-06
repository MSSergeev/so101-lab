# Troubleshooting

Common issues and fixes.

---

## Setup and environment

**`ModuleNotFoundError: No module named 'pxr'`**
Wrong venv. Isaac Lab env is not active.
```bash
act-isaac  # activate Isaac Lab env
```

**`.env not found` when running activate script**
```bash
cp .env.example .env
# Edit .env: set ISAACLAB_ENV and LEROBOT_ENV to your actual venv paths
```

**Isaac Sim takes 1–2 minutes to start**
Normal on first run — assets and physics are compiled and cached. Subsequent starts take 30–60 seconds.

**`RuntimeError: A camera was spawned without --enable_cameras`**
Scripts that use cameras set `args.enable_cameras = True` automatically. If you see this, you are calling a script that doesn't — add `args.enable_cameras = True` before `AppLauncher(args)`.

---

## Real robot

**`PermissionError: [Errno 13] Permission denied: '/dev/ttyACM0'`**
```bash
sudo usermod -a -G dialout $USER
# Log out and back in
```

**`RuntimeError: Failed to open port /dev/ttyACM0`**
- Check USB connection: `lsusb`
- Check port: `ls /dev/ttyACM*` — port number may differ (`ACM1`, `ACM2`)
- Pass the correct port: `--port /dev/ttyACM1`

**`FileNotFoundError: Calibration file not found`**
Run calibration first:
```bash
act-lerobot  # activate lerobot-env
python scripts/calibrate_leader.py --port /dev/ttyACM2
```

---

## Recording

**`[WARNING] Viewer venv not found`**
Viewer venv is not created. Create it:
```bash
uv venv venvs/viewer
source venvs/viewer/bin/activate
uv pip install opencv-python numpy
```

**Recording is slower than real time**
- Use `--headless` (biggest impact)
- Reduce `--render-hz`

---

## Training

**CUDA out of memory**

All tasks fit in 8–16 GB VRAM with default config values. To reduce memory:
- ACT: `--batch-size 16` (default 24)
- SmolVLA: `--batch_size 4` (default 8)
- Parallel eval: reduce `--num-envs`

**ACT: `kld_loss = 0.0000` (posterior collapse)**
Model ignores the latent space. Reduce `kl_weight`:
```bash
python scripts/train/train_act.py \
    --dataset data/recordings/my_task \
    --config configs/policy/act/low_kl.yaml
```

**ACT: loss not decreasing**
- Check dataset quality (visualize with Rerun)
- Try lower learning rate: `--lr 1e-6`
- Add more episodes

**ACT: best checkpoint by loss ≠ best success rate**
Always run a sweep eval. Do not rely on training loss alone:
```bash
python scripts/eval/sweep_act_eval_single.py --checkpoint outputs/act_v1 --all --episodes 50 --max-steps 500
```

**Temporal ensembling hurts ACT results**
Observed with `chunk_size=20`: 15% vs 60% without ensembling. Works better with small chunk sizes (1–5). Disable with `--temporal-ensemble-coeff` omitted.

**`ImportError: No module named 'diffusers'`**
```bash
act-isaac  # activate Isaac Lab env
uv pip install diffusers
```

**Diffusion Policy inference is slow**
Diffusion runs 100 denoising steps by default. Speed up:
```bash
--num-inference-steps 20   # DDIM with fewer steps
```
Parallel eval (`eval_diffusion_policy_parallel.py`) uses the same temporal ensembling fix as ACT parallel eval.

**SmolVLA: `train_expert_only=true` overrides `freeze_vision_encoder=false`**
These are two independent flags but `train_expert_only` freezes the entire VLM including the vision encoder. To unfreeze vision encoder you must explicitly set both:
```bash
--policy.train_expert_only=false --policy.freeze_vision_encoder=false
```

**SmolVLA: package downgrades after `[smolvla]` install**
Not an issue in LeRobot 0.5.1 — `[smolvla]` only adds `transformers`, `num2words`, `accelerate`, `safetensors`. torch is not downgraded.

---

## Evaluation

**Low success rate despite good training loss**
1. Overfitting — try an earlier checkpoint (`--step 25000`)
2. `n_action_steps` too large — try `--n-action-steps 10`
3. Policy-hz mismatch — pass `--policy-hz <training_fps>` (must match recording fps)
4. Insufficient data diversity

**`avg_steps_fail ≈ max_steps`** — policy gets stuck, never reaches goal.
**`avg_steps_fail << max_steps`** — policy fails early (wrong grasp, drops object).
**`avg_steps_success << max_steps`** — policy is efficient (good sign).

**Eval hangs after pressing Isaac Sim Play/Pause button**
Do not use the GUI Play/Pause button — it stops physics but not Python. Use keyboard controls: Space to pause, N to skip episode, Escape to quit. Or use `eval_viewer.py` in a second terminal.
