# eval_act_policy.md

Evaluate trained ACT policy in Isaac Lab simulation. Isaac Lab env.

Covers: `eval_act_policy.py` (single-env, interactive), `eval_act_policy_parallel.py` (multi-env, batch), `sweep_act_eval_single.py` (single-env sweep), `sweep_act_eval.py` (parallel sweep).

---

Runs in **Isaac Lab env** (`act-isaac`). See [setup](../workflow/00_setup.md).

## Quick start

```bash
act-isaac  # activate Isaac Lab env

# Single-env: 10 episodes, headless
python scripts/eval/eval_act_policy.py --checkpoint outputs/act_v1

# Single-env: GUI + camera preview
python scripts/eval/eval_act_policy.py --checkpoint outputs/act_v1 --gui --preview

# Sweep: evaluate all checkpoints (recommended)
python scripts/eval/sweep_act_eval_single.py --checkpoint outputs/act_v1 --all --episodes 50 --max-steps 500
```

---

## eval_act_policy.py (single-env)

### Parameters

**Checkpoint selection:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | **required** | Training output directory |
| `--use-best` | true | Load `best/` checkpoint |
| `--use-latest` | false | Load from root (latest step) |
| `--step N` | — | Load `checkpoint_N/` |

**Evaluation:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--env` | `figure_shape_placement` | Environment name |
| `--episodes` | 10 | Number of episodes |
| `--max-steps` | 1000 | Max steps per episode |
| `--timeout-s` | — | Wall-clock timeout (steps take priority) |
| `--seed` | random | Initial RNG seed (generates per-episode seeds) |
| `--episode-seed` | — | Run one episode with this exact seed (reproduce specific scenario) |

**Recording:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--save-episodes` | `none` | `all`, `success`, `fail`, `none` |
| `--output` | auto | Output directory |
| `--crf` | 23 | Video quality (lower = better) |
| `--gop` | `auto` | GOP size (`auto` for eval recordings; `2` for training data) |

**Display:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--gui` | false | Isaac Sim window |
| `--preview` | false | Camera preview (eval_viewer.py) |

**Frequencies:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--physics-hz` | 120 | Physics frequency |
| `--policy-hz` | 30 | Policy/control frequency |
| `--render-hz` | 30 | Render frequency |
| `--preview-hz` | 30 | Preview update frequency |

**Policy:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-action-steps` | from config | Override actions per chunk |
| `--temporal-ensemble-coeff` | — | Enable temporal ensembling (e.g., `0.01`) |
| `--ensemble-interval` | 1 | Run model every N steps (requires `--temporal-ensemble-coeff`) |

**Domain randomization:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--randomize-light` | false | Randomize sun light intensity/color |

### Keyboard controls

Works in Isaac Sim window and in `eval_viewer.py`:

| Key | Action |
|-----|--------|
| Space | Pause/Resume |
| N | Next — skip current episode |
| R | Restart — same seed, same scene |
| Escape | Quit |

**Do not use the Isaac Sim GUI Play/Pause button** — it stops physics but not Python, causing a hang.

### Output

```
outputs/eval/figure_shape_placement_2025-01-27_12-30-00/
├── config.json      # Run parameters
├── summary.json     # Aggregated statistics
├── registry.json    # Per-episode details (seed, steps, success, initial/final state)
└── episodes/        # Saved episodes (if --save-episodes != none)
```

`summary.json`:
```json
{
  "total_episodes": 100,
  "successes": 73,
  "success_rate": 73.0,
  "avg_steps": 342.5,
  "avg_steps_success": 298.2,
  "avg_steps_fail": 461.8,
  "avg_time_s": 11.42,
  "realtime_factor": 2.1
}
```

### Usage examples

```bash
# Reproduce specific episode from registry.json
python scripts/eval/eval_act_policy.py --checkpoint outputs/act_v1 \
    --episode-seed 1234567890 --gui --preview

# Save failed episodes for Rerun analysis
python scripts/eval/eval_act_policy.py --checkpoint outputs/act_v1 \
    --episodes 50 --save-episodes fail

# Try smaller n_action_steps (more reactive, may shift distribution)
python scripts/eval/eval_act_policy.py --checkpoint outputs/act_v1 \
    --n-action-steps 10 --episodes 20

# Temporal ensembling (smoother motion)
python scripts/eval/eval_act_policy.py --checkpoint outputs/act_v1 \
    --temporal-ensemble-coeff 0.01 --episodes 20
```

---

## eval_act_policy_parallel.py (multi-env)

Runs N environments in one Isaac Lab scene, batches policy inference across envs.

### Parameters

Same as single-env plus:

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-envs` | 4 | Number of parallel environments |

No `--preview`, `--save-episodes`, `--episode-seed`, `--gui` controls. Always headless recommended.

### Performance

Approximate on RTX 4090:

| num_envs | 100 episodes | Speedup |
|----------|-------------|---------|
| 1 | ~300s | 1x |
| 4 | ~100s | 3x |
| 8 | ~60s | 5x |
| 16 | ~40s | 7.5x |

Limit: GPU memory for cameras (2 cameras per env).

### Action queue

At `n_action_steps=10`, model runs once per 10 steps, queue serves the rest — ~10x fewer forward passes:

```
Step 1:  all queues empty → batch inference for all envs
Steps 2-10: pop from queues, no inference
Step 11: queues empty → batch inference
...
```

Output includes: `Inference calls: 342 (11.7 actions/call)`

### Temporal ensembling

Alternative to action queue: overlapping chunk predictions blended with exponential weighting. Available in both single-env and parallel scripts.

| Mode | Inference | Smoothness | Passes |
|------|-----------|------------|--------|
| Action queue | Every n_action_steps | Stepped | 1/n_action_steps |
| Ensemble (interval=1) | Every step | Maximum | 1/step |
| Ensemble + interval=N | Every N steps | Good | 1/N |

```bash
# Single-env
python scripts/eval/eval_act_policy.py --checkpoint outputs/act_v1 \
    --temporal-ensemble-coeff 0.01 --episodes 20

# Single-env + interval (model every 5 steps, buffer serves rest)
python scripts/eval/eval_act_policy.py --checkpoint outputs/act_v1 \
    --temporal-ensemble-coeff 0.01 --ensemble-interval 5 --episodes 20

# Parallel
python scripts/eval/eval_act_policy_parallel.py --checkpoint outputs/act_v1 \
    --temporal-ensemble-coeff 0.01 --ensemble-interval 5
```

**Internals (parallel):** Batched `ACTTemporalEnsembler` with per-env state `(B, T, action_dim)` and per-env count for correct blending with staggered resets. `update()` blends only overlapping positions (buffer shrinks after each `pop()`). On auto-reset: envs with stale cameras are excluded from blending via `skip_env_ids` — they receive zero action (camera update step) and `reset_envs()` marks them for full re-initialization on the next `update()` call with fresh observations.

Do not use Isaac Sim GUI Play/Pause in parallel eval — no pause support, causes hang.

### Seeds

`--seed` sets global `torch.manual_seed()`. Per-env seeds are not supported by Isaac Lab auto-reset — `registry.json` records `"N/A (auto-reset)"`. To reproduce a specific episode, use single-env `--episode-seed`.

### Multi-env scene architecture

Isaac Lab places N envs in a grid (`env_spacing=10.0m`), each with its own table/robot/objects/lighting. PhysicsScene is global (one per simulation). Per-env ground plane via `{ENV_REGEX_NS}/GroundPlane` (white 3×3m cube). Scene USD (`workbench_table_only.usd`) has PhysicsScene and global GroundPlane removed.

---

## sweep_act_eval_single.py (recommended)

Runs `eval_act_policy.py` sequentially for multiple checkpoints, aggregates results. Slower than parallel sweep but produces correct results.

```bash
# Evaluate specific checkpoints + best
python scripts/eval/sweep_act_eval_single.py --checkpoint outputs/act_v1 \
    --best --checkpoints 30000 40000 50000 \
    --env figure_shape_placement --episodes 50 --max-steps 500

# With temporal ensembling
python scripts/eval/sweep_act_eval_single.py --checkpoint outputs/act_v1 \
    --best --checkpoints 30000 40000 50000 \
    --episodes 100 --max-steps 600 --temporal-ensemble-coeff 0.01

# All checkpoints
python scripts/eval/sweep_act_eval_single.py --checkpoint outputs/act_v1 \
    --all --episodes 50 --max-steps 500
```

### Parameters

Same checkpoint selection as `sweep_act_eval.py`. All `eval_act_policy.py` params proxied: `--episodes`, `--max-steps`, `--env`, `--seed`, `--n-action-steps`, `--temporal-ensemble-coeff`, `--ensemble-interval`, `--randomize-light`, `--physics-hz`, `--policy-hz`.

No `--num-envs` (always single-env).

---

## sweep_act_eval.py (parallel)

Runs `eval_act_policy_parallel.py` sequentially for multiple checkpoints, aggregates results.

```bash
python scripts/eval/sweep_act_eval.py --checkpoint outputs/act_v1 \
    --all --num-envs 4 --episodes 100
```

### Checkpoint selection

| Flag | Description |
|------|-------------|
| (no flags) | `best` only (default) |
| `--best` | `best/` subdirectory |
| `--latest` | Root directory (latest step) |
| `--checkpoints N M` | Specific steps |
| `--all` | All `checkpoint_*/` + `best/` |

Combinations allowed: `--best --latest --checkpoints 15000`.

All eval params proxied: `--num-envs`, `--episodes`, `--max-steps`, `--env`, `--seed`, `--n-action-steps`, `--temporal-ensemble-coeff`, `--ensemble-interval`, `--randomize-light`, `--physics-hz`, `--policy-hz`.

### Seeds

Sweep auto-generates a shared seed and passes it to all checkpoint runs — same scenes for all (fair comparison). `--seed random` = independent seeds per checkpoint.

### Output

```
outputs/eval_sweeps/<model_name>_<timestamp>/
├── sweep_results.json     # Aggregated JSON for all checkpoints
├── best/
│   ├── summary.json
│   ├── registry.json
│   └── config.json
└── checkpoint_10000/
    └── ...
```

Console table after each checkpoint:

```
+------------------+----------+----------+------------+
| Checkpoint       | Episodes | Success% | Avg Steps  |
+------------------+----------+----------+------------+
| best             | 100      |     45.0 |      234.5 |
| checkpoint_10000 | 100      |     32.0 |      456.2 |
+------------------+----------+----------+------------+
```

Invalid checkpoints (no `model.safetensors`) are pre-validated and marked SKIPPED. If a checkpoint crashes, sweep continues with the rest.

---

## Success detection

Both scripts set `env_cfg.terminate_on_success = True`. `_get_dones()` calls `is_success()` and returns it as `terminated`. Isaac Lab auto-resets successful envs immediately. Success is read from `terminated` in `env.step()` return value — checked **before** auto-reset, so object positions are still correct.

---

## Interpreting results

- **>80%** — good policy, ready for sim2real
- **50–80%** — needs more data or tuning
- **<50%** — training or data issue

- `avg_steps_fail ≈ max_steps` — policy gets stuck
- `avg_steps_fail << max_steps` — early critical errors
- `avg_steps_success << max_steps` — policy solves task efficiently (early exit)

### Analyzing failed episodes

```bash
# Save
python scripts/eval/eval_act_policy.py --checkpoint outputs/act_v1 \
    --save-episodes fail --episodes 50

# Visualize
source venvs/rerun/bin/activate
python scripts/visualize_lerobot_rerun.py outputs/eval/run_001/episodes
```

---

## Troubleshooting

**"Policy trained at X Hz, but eval running at Y Hz"** — pass `--policy-hz <training_fps>`.

**Low success at good train loss:**
1. Overfitting — try earlier checkpoint (`--step`)
2. n_action_steps too large — try `--n-action-steps 10`
3. Insufficient data diversity

**Preview not opening** — create viewer venv (see `docs/workflow/02_recording.md`).
