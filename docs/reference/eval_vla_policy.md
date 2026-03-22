# eval_vla_policy.py / sweep_vla_eval.py / smolvla_server.py

Evaluate VLA policies (SmolVLA, Pi0, GR00T) in Isaac Lab simulation.

---

## Architecture

Policy inference runs in a separate process to avoid the Python 3.11 (isaaclab-env) vs Python 3.12 (lerobot 0.5.1) incompatibility:

```
isaaclab-env (Python 3.11)          lerobot-env (Python 3.12)
──────────────────────────          ─────────────────────────
eval_vla_policy.py                  smolvla_server.py
  GrpcPolicyClient          gRPC      SimPolicyServer
  select_action(obs)  ──────────▶   predict_action_chunk()
  action_np           ◀──────────   preprocessor + policy + postprocessor
```

Transport: `AsyncInference` gRPC service (lerobot's `transport/services.proto`). Obs serialized as pickle, actions as `numpy.save` bytes (cross-Python compatible).

---

## Quickstart

**Single eval** — `eval_vla_policy.py` starts server automatically with `--auto-server`:

```bash
act-isaac

python scripts/eval/eval_vla_policy.py \
    --checkpoint outputs/smolvla_figure_v5/checkpoints/last/pretrained_model \
    --env figure_shape_placement \
    --episodes 20 --max-steps 500 --no-domain-rand \
    --n-action-steps 15 --seed 1988042740 \
    --auto-server --gui --preview
```

**Sweep** — server started/stopped automatically by `sweep_vla_eval.py`:

```bash
act-isaac

python scripts/eval/sweep_vla_eval.py \
    --checkpoint outputs/smolvla_figure_v5/checkpoints \
    --all \
    --env figure_shape_placement \
    --episodes 50 --max-steps 500 --no-domain-rand \
    --n-action-steps 15 --seed 1988042740
```

**Manual server** (two terminals):

```bash
# Terminal 1
act-lerobot
python scripts/eval/smolvla_server.py --port 8080

# Terminal 2
act-isaac
python scripts/eval/eval_vla_policy.py \
    --checkpoint outputs/smolvla_figure_v5/checkpoints/last/pretrained_model \
    --port 8080 --episodes 20
```

---

## CLI — eval_vla_policy.py

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | `pretrained_model` dir or HuggingFace model ID |
| `--task` | `"Place the cube..."` | Language instruction passed to policy |
| `--env` | `figure_shape_placement` | Task name (task registry) |
| `--episodes` | 20 | Number of eval episodes |
| `--max-steps` | 500 | Max steps per episode |
| `--n-action-steps` | 15 | Actions to execute per inference call |
| `--seed` | random | Random seed |
| `--episode-seed` | None | Run single episode with this exact seed |
| `--start-paused` | false | Start paused before first episode (press Space to begin) |
| `--gui` | false | Run with Isaac Sim GUI |
| `--preview` | false | Camera preview window |
| `--no-domain-rand` | false | Disable all DR |
| `--noise-prior` | None | Path to `noise_prior.pt` — passed to server for model patching |
| `--output` | None | Directory to write `summary.json` |
| `--server` | `127.0.0.1` | Policy server host |
| `--port` | 8080 | Policy server port |
| `--auto-server` | false | Auto-start server (do not use with sweep) |

---

## CLI — sweep_vla_eval.py

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Path to `checkpoints/` directory |
| `--all` | false | Evaluate all checkpoints |
| `--last` | false | Evaluate `last/` checkpoint only |
| `--checkpoints` | [] | Specific step numbers (e.g. `5000 10000`) |
| `--episodes` | 20 | Episodes per checkpoint |
| `--max-steps` | 500 | Max steps per episode |
| `--env` | `figure_shape_placement` | Task name |
| `--seed` | random | Shared seed across all checkpoints |
| `--n-action-steps` | None | Override actions per chunk |
| `--no-domain-rand` | false | Disable DR |
| `--noise-prior` | None | Passed to policy server |
| `--port` | 8080 | Policy server port |
| `--output` | None | Output directory |

---

## CLI — smolvla_server.py

```bash
act-lerobot
python scripts/eval/smolvla_server.py --host 127.0.0.1 --port 8080
```

Server waits for `SendPolicyInstructions` (checkpoint path, policy type, task, device) before loading the model. The `task` string is stored on the server and injected into every inference batch — it is **not** sent per-observation from the client. Loading happens once per `connect()` call — sweep reloads the model for each checkpoint.

---

## Data Flow

```
IsaacLabGymEnv obs
  {"observation.state": (6,), "observation.images.top": (H,W,3), ...}
        │  pickle (cross-Python safe for numpy)
        ▼
smolvla_server.py
  torch conversion (CHW float32 / 255)
  preprocessor (tokenize, normalize, batch dim, device)
  policy.predict_action_chunk()  →  (1, chunk_size, 6)
  postprocessor (unnormalize)
  numpy.save → bytes
        │
        ▼
GrpcPolicyClient
  action_chunk: (actions_per_chunk, 6)
  served locally step-by-step, re-queried when exhausted
        │
        ▼
clip [-100,100] / gripper [0,100] → env.step()
```

---

## Success Detection

Uses `milestone_placed_weighted` from `ManagerBasedRLEnv` reward (ground-truth sim state: `dist_xy < 0.04 m`). No reward classifier needed.

---

## Fine-tuning SmolVLA

See [workflow/03_bc_training.md](../workflow/03_bc_training.md) for training commands and variant comparison.

---

## Setup (first time)

```bash
# lerobot-env: install smolvla deps and gRPC
cd /path/to/lerobot
uv pip install -e ".[smolvla]"
uv pip install -e ".[async]"   # grpcio
```

`[smolvla]` in lerobot 0.5.1 does not downgrade torch.
