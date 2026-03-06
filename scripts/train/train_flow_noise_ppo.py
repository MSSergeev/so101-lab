"""PPO training for SmolVLA with Flow-Noise stochastic denoising.

Fine-tunes action_out_proj + noise_log_std_head (~24k params) and value MLP
(~593k params) using PPO with VIP dense reward in Isaac Lab simulation.

Usage:
    # Smoke test (2 updates)
    python scripts/train/train_flow_noise_ppo.py \
        --checkpoint outputs/smolvla_teleop_only/checkpoints/016000/pretrained_model \
        --iql-checkpoint outputs/iql_critics_v2_100k/final/critics.pt \
        --goal-dataset data/recordings/figure_shape_placement_v5 \
        --total-updates 2 --rollout-steps 16 \
        --output outputs/flow_noise_ppo_smoke --headless

    # With labeled VIP goals
    python scripts/train/train_flow_noise_ppo.py \
        --checkpoint outputs/smolvla_teleop_only/checkpoints/016000/pretrained_model \
        --iql-checkpoint outputs/iql_critics_v2_100k/final/critics.pt \
        --goal-dataset data/recordings/figure_shape_placement_v5 \
        --vip-use-labeled \
        --vip-label-dataset data/recordings/figure_shape_placement_v5_vip \
        --total-updates 100 --rollout-steps 256 \
        --output outputs/flow_noise_ppo_v1 --tracker trackio --headless
"""

# Section 1: argparse + AppLauncher (before Isaac Sim init)
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PPO training for SmolVLA with Flow-Noise")

# Environment
parser.add_argument("--env", type=str, default="figure_shape_placement",
                    help="Task environment name (default: figure_shape_placement)")

# Policy
parser.add_argument(
    "--checkpoint", type=str, required=True,
    help="Path to SmolVLA BC checkpoint (pretrained_model dir)",
)
parser.add_argument(
    "--iql-checkpoint", type=str, default=None,
    help="IQL critics checkpoint for value head warm start",
)
parser.add_argument(
    "--task", type=str,
    default="Place the cube into the matching slot on the platform",
    help="Language instruction for VLA policy",
)
parser.add_argument(
    "--noise-prior", type=str, default=None,
    help="Path to noise_prior.pt (learned sampler or fixed mu shift)",
)

# Reward / VIP goals
parser.add_argument("--goal-dataset", type=str, required=True, help="Dataset path for VIP goal embeddings (images)")
parser.add_argument("--vip-use-labeled", action="store_true", help="Use labeled success frames (next.reward > 0.5) as VIP goals")
parser.add_argument(
    "--vip-label-dataset", type=str, default=None,
    help="Dataset with next.reward labels (default: same as --goal-dataset)",
)
parser.add_argument("--vip-goal-mode", type=str, default="mean", choices=["mean", "min"])
parser.add_argument("--n-goal-frames", type=int, default=5, help="Final frames per episode for goal (if not --vip-use-labeled)")
parser.add_argument("--vip-weight", type=float, default=1.0)
parser.add_argument("--sim-weight", type=float, default=0.0)
parser.add_argument("--success-bonus", type=float, default=0.0)

# PPO
parser.add_argument("--total-updates", type=int, default=1000)
parser.add_argument("--rollout-steps", type=int, default=256)
parser.add_argument("--update-epochs", type=int, default=4)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--clip-ratio", type=float, default=0.2)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--actor-lr", type=float, default=3e-5)
parser.add_argument("--value-lr", type=float, default=1e-4)
parser.add_argument("--max-grad-norm", type=float, default=1.0)
parser.add_argument("--reeval-batch-size", type=int, default=1,
    help="Batch size for suffix re-eval (higher = faster, more VRAM)")
parser.add_argument("--kv-cache-device", type=str, default="cpu",
    choices=["cpu", "cuda"],
    help="Device for KV cache storage (cuda = faster, +2.9 GB VRAM)")
parser.add_argument("--warmup-value", type=int, default=0,
    help="First N PPO updates: train only value head, freeze actor (0 = off)")
parser.add_argument("--normalize-rewards", action="store_true",
    help="Normalize rewards with running mean/std")

# Environment
parser.add_argument("--max-episode-steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)

# Logging & Checkpoints
parser.add_argument("--output", type=str, default="outputs/flow_noise_ppo_v1")
from so101_lab.utils.tracker import add_tracker_args
add_tracker_args(parser, default_project="so101-flow-ppo")
parser.add_argument("--log-freq", type=int, default=1, help="Log every N updates")
parser.add_argument("--save-freq", type=int, default=50, help="Save checkpoint every N updates")
parser.add_argument("--eval-freq", type=int, default=50, help="Evaluate every N updates")
parser.add_argument("--eval-episodes", type=int, default=5)

# Resume
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint dir to resume from")

# Domain randomization
parser.add_argument("--no-domain-rand", action="store_true")
parser.add_argument("--no-randomize-light", action="store_true")
parser.add_argument("--no-randomize-physics", action="store_true")
parser.add_argument("--no-camera-noise", action="store_true")
parser.add_argument("--no-distractors", action="store_true")

# Display
parser.add_argument("--gui", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.gui:
    args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)

# Section 2: sys.path hack for LeRobot
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

from so101_lab.utils import disable_rate_limiting
disable_rate_limiting()

# Section 3: Imports after AppLauncher
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from so101_lab.policies.rl.flow_noise_smolvla import FlowNoiseSmolVLA
from so101_lab.rewards.vip_reward import VIPReward
from so101_lab.rl.isaac_lab_gym_env import IsaacLabGymEnv
from so101_lab.rl.domain_rand import apply_domain_rand_flags
from so101_lab.tasks import get_rl_task

ACTION_DIM = 6


# --- Reward normalization ---

class RunningMeanStd:
    """Welford's online algorithm for running mean/variance."""

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, x: torch.Tensor):
        batch_mean = x.mean().item()
        batch_var = x.var().item() if len(x) > 1 else 0.0
        batch_count = len(x)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / max(total, 1)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / max(total, 1)
        self.var = M2 / max(total, 1)
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var**0.5 + 1e-8)


# --- GAE ---

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# --- Reward ---

def compute_reward(obs: dict, env: IsaacLabGymEnv, vip_reward: VIPReward, args) -> float:
    """Compute combined reward."""
    r = 0.0

    if args.vip_weight > 0:
        r += args.vip_weight * vip_reward.compute_reward(obs)

    if args.sim_weight > 0 or args.success_bonus > 0:
        details = env.get_reward_details()
        if args.sim_weight > 0:
            r += args.sim_weight * details.get("reward/distance_cube_to_slot_weighted", 0.0)
        if args.success_bonus > 0:
            r += args.success_bonus * details.get("reward/milestone_placed_weighted", 0.0)

    return float(r)


# --- Evaluation ---

def evaluate(
    policy: FlowNoiseSmolVLA,
    env: IsaacLabGymEnv,
    episodes: int,
    max_steps: int,
    obs_ref: list | None = None,
    use_learned_sampler: bool = False,
) -> dict[str, float]:
    """Run deterministic evaluation episodes."""
    policy.set_eval_noise_bounds()
    policy.policy.reset()

    successes = 0
    total_steps = 0
    total_reward = 0.0

    for ep in range(episodes):
        obs, _ = env.reset()
        policy.policy.reset()
        ep_reward = 0.0

        for step in range(max_steps):
            if use_learned_sampler and obs_ref is not None:
                obs_ref[0] = obs
            action = policy.sample_actions_deterministic(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            details = env.get_reward_details()
            ep_reward += details.get("reward/milestone_placed_weighted", 0.0)
            total_steps += 1

            if terminated or truncated:
                break

        if ep_reward > 0:
            successes += 1
        total_reward += ep_reward

    policy.set_train_noise_bounds()

    return {
        "success_rate": successes / episodes,
        "avg_reward": total_reward / episodes,
        "avg_steps": total_steps / episodes,
    }


# --- Main ---

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # Save config
    with open(os.path.join(args.output, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # 1. Create environment
    RLEnvCfgClass = get_rl_task(args.env)
    env_cfg = RLEnvCfgClass()
    env_cfg.scene.num_envs = 1
    apply_domain_rand_flags(env_cfg, args)
    env = IsaacLabGymEnv(env_cfg=env_cfg)

    # 2. Load policy
    print(f"\nLoading SmolVLA from {args.checkpoint}")
    policy = FlowNoiseSmolVLA(
        checkpoint_path=args.checkpoint,
        device="cuda",
        iql_checkpoint=args.iql_checkpoint,
        task_string=args.task,
        kv_cache_device=args.kv_cache_device,
    )

    # 2b. Apply noise prior if provided
    use_learned_sampler = False
    obs_ref = [None]
    if args.noise_prior:
        ckpt = torch.load(args.noise_prior, map_location="cuda", weights_only=False)
        np_data = ckpt.get("noise_prior", {})
        if isinstance(np_data, dict) and np_data.get("type") == "learned":
            from so101_lab.policies.rl.learned_noise_sampler import LearnedNoiseSampler
            noise_dims = np_data.get("noise_dims", 6)
            learned_sampler = LearnedNoiseSampler(device="cuda", noise_dims=noise_dims)
            learned_sampler.load(args.noise_prior)
            learned_sampler.eval()
            learned_sampler.patch_model(policy.policy.model, obs_ref)
            use_learned_sampler = True
            print(f"Learned noise sampler loaded (noise_dims={noise_dims})")
        else:
            from so101_lab.policies.rl.noise_prior import NoisePrior
            noise_prior = NoisePrior(device="cuda")
            noise_prior.load_state_dict(np_data)
            noise_prior.patch_model(policy.policy.model)
            mu_str = ", ".join(f"{v:.3f}" for v in noise_prior.mu.cpu().numpy())
            print(f"Noise prior loaded: mu = [{mu_str}]")

    # 3. VIP reward model
    print(f"\nInitializing VIP reward from {args.goal_dataset}")
    vip_reward = VIPReward(
        args.goal_dataset,
        device="cuda",
        use_labeled=args.vip_use_labeled,
        label_dataset_path=args.vip_label_dataset,
        goal_mode=args.vip_goal_mode,
        n_goal_frames=args.n_goal_frames,
    )

    # 4. Optimizers
    actor_optimizer = torch.optim.Adam(
        policy.trainable_actor_params(), lr=args.actor_lr
    )
    value_optimizer = torch.optim.Adam(
        policy.value_mlp.parameters(), lr=args.value_lr
    )
    optimizers = {"actor": actor_optimizer, "value": value_optimizer}

    # 5. Resume
    start_update = 0
    best_sr = -1.0
    ep_count = 0
    if args.resume:
        meta = policy.load_checkpoint(args.resume, optimizers)
        start_update = meta.get("update", 0)
        best_sr = meta.get("best_sr", -1.0)
        ep_count = meta.get("ep_count", 0)
        print(f"Resumed from update {start_update}, best_sr={best_sr:.0%}, ep_count={ep_count}")

    # 6. Tracker
    run_name = os.path.basename(os.path.normpath(args.output))

    from so101_lab.utils.tracker import setup_tracker, cleanup_tracker
    tracker, sys_monitor = setup_tracker(args, run_name)

    # 7. Training info
    total_updates = start_update + args.total_updates
    print(f"\nStarting PPO training: {args.total_updates} updates (total {total_updates})")
    print(f"  Rollout steps: {args.rollout_steps}")
    print(f"  Update epochs: {args.update_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Clip ratio: {args.clip_ratio}")
    print(f"  Actor LR: {args.actor_lr}")
    print(f"  Value LR: {args.value_lr}")
    print(f"  Gamma: {args.gamma}, GAE lambda: {args.gae_lambda}")
    print(f"  VIP weight: {args.vip_weight}, Sim weight: {args.sim_weight}")
    print(f"  Success bonus: {args.success_bonus}")
    print(f"  KV cache device: {args.kv_cache_device}")
    if args.warmup_value > 0:
        print(f"  Warmup: {args.warmup_value} updates (value only)")
    if args.normalize_rewards:
        print(f"  Reward normalization: enabled")
    print()

    # 8. PPO loop
    reward_rms = RunningMeanStd() if args.normalize_rewards else None
    obs, _ = env.reset()
    policy.policy.reset()
    ep_reward_accum = 0.0
    ep_steps = 0
    t_start = time.time()

    pbar = tqdm(
        range(start_update, total_updates),
        desc="PPO", unit="update",
        initial=start_update, total=total_updates,
    )

    for update in pbar:
        update_t0 = time.time()

        # === Rollout ===
        rollout_obs = []           # list of obs dicts (for re-evaluation)
        rollout_trajectories = []  # per-step x_t and v_sampled for PPO reeval
        rollout_vip_embs = []      # cached VIP embeddings (2048,) for value head
        rollout_states = []        # cached states (6,) for value head
        rollout_log_probs = []     # scalar tensors (detached)
        rollout_values = []        # scalar tensors
        rollout_rewards = []       # floats
        rollout_dones = []         # bools

        for step in range(args.rollout_steps):
            # Cache VIP embedding for value re-evaluation
            vip_emb = policy.cache_vip_embedding(obs)
            state_tensor = torch.from_numpy(obs["observation.state"]).float().to("cuda")

            # Get value (no grad)
            with torch.no_grad():
                value = policy.get_value_from_cached(
                    vip_emb.unsqueeze(0), state_tensor.unsqueeze(0)
                )

            # Update obs_ref for learned noise sampler (patches model.sample_noise)
            if use_learned_sampler:
                obs_ref[0] = obs

            # Sample action with log_prob + trajectory data
            action, log_prob, trajectory = policy.sample_actions_with_log_prob(obs)

            # Step environment
            obs_next, _, terminated, truncated, _ = env.step(action)
            reward = compute_reward(obs_next, env, vip_reward, args)
            done = terminated or truncated

            # Store transition
            rollout_obs.append(obs)
            rollout_trajectories.append(trajectory)
            rollout_vip_embs.append(vip_emb)
            rollout_states.append(state_tensor)
            rollout_log_probs.append(log_prob)
            rollout_values.append(value.squeeze())
            rollout_rewards.append(reward)
            rollout_dones.append(float(done))

            ep_reward_accum += reward
            ep_steps += 1

            if done:
                ep_count += 1
                obs, _ = env.reset()
                policy.policy.reset()
                ep_reward_accum = 0.0
                ep_steps = 0
            else:
                obs = obs_next

        # Bootstrap last value
        with torch.no_grad():
            last_vip = policy.cache_vip_embedding(obs)
            last_state = torch.from_numpy(obs["observation.state"]).float().to("cuda")
            last_value = policy.get_value_from_cached(
                last_vip.unsqueeze(0), last_state.unsqueeze(0)
            ).item()

        # Convert to tensors
        raw_rewards_t = torch.tensor(rollout_rewards, dtype=torch.float32)
        if reward_rms is not None:
            reward_rms.update(raw_rewards_t)
            rewards_t = reward_rms.normalize(raw_rewards_t)
        else:
            rewards_t = raw_rewards_t
        values_t = torch.stack(rollout_values).cpu()
        dones_t = torch.tensor(rollout_dones, dtype=torch.float32)
        old_log_probs_t = torch.stack(rollout_log_probs).cpu()

        # Stack VIP embeddings and states for batched value re-evaluation
        vip_embs_t = torch.stack(rollout_vip_embs)   # (T, 2048)
        states_t = torch.stack(rollout_states)         # (T, 6)

        # === GAE ===
        advantages, returns = compute_gae(
            rewards_t, values_t, dones_t, last_value, args.gamma, args.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Move to GPU for update
        advantages = advantages.to("cuda")
        returns = returns.to("cuda")
        old_log_probs_t = old_log_probs_t.to("cuda")
        vip_embs_t = vip_embs_t.to("cuda")
        states_t = states_t.to("cuda")

        # === PPO Update ===
        is_warmup = (update < start_update + args.warmup_value)
        T = args.rollout_steps
        indices = np.arange(T)
        actor_losses = []
        value_losses = []
        ratios_all = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, args.batch_size):
                end = min(start + args.batch_size, T)
                batch_idx = indices[start:end]

                # Re-evaluate values using cached embeddings
                batch_vip = vip_embs_t[batch_idx]
                batch_states = states_t[batch_idx]
                new_values = policy.get_value_from_cached(batch_vip, batch_states)  # (B,)

                # Value loss (always updated)
                value_loss = 0.5 * (new_values - returns[batch_idx]).pow(2).mean()
                value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.value_mlp.parameters(), args.max_grad_norm
                )
                value_optimizer.step()
                value_losses.append(value_loss.item())

                if is_warmup:
                    actor_losses.append(0.0)
                    ratios_all.append(torch.ones(len(batch_idx)))
                    continue

                # Re-evaluate log_prob under current params
                if args.reeval_batch_size > 1:
                    batch_log_probs = []
                    for rb_start in range(0, len(batch_idx), args.reeval_batch_size):
                        rb_end = min(rb_start + args.reeval_batch_size, len(batch_idx))
                        rb_trajs = [rollout_trajectories[batch_idx[j]] for j in range(rb_start, rb_end)]
                        rb_lps = policy.reeval_log_prob_batched(rb_trajs)
                        batch_log_probs.append(rb_lps)
                    new_log_probs = torch.cat(batch_log_probs)  # (B,) with grad
                else:
                    batch_log_probs = []
                    for i in batch_idx:
                        new_lp = policy.reeval_log_prob(
                            rollout_obs[i], rollout_trajectories[i]
                        )
                        batch_log_probs.append(new_lp)
                    new_log_probs = torch.stack(batch_log_probs)  # (B,) with grad

                # PPO ratio
                ratio = torch.exp(new_log_probs - old_log_probs_t[batch_idx])
                adv = advantages[batch_idx]

                # Clipped surrogate
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Update actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(policy.trainable_actor_params()), args.max_grad_norm
                )
                actor_optimizer.step()

                actor_losses.append(actor_loss.item())
                ratios_all.append(ratio.detach().cpu())

        # === Logging ===
        update_time = time.time() - update_t0
        all_ratios = torch.cat(ratios_all)

        log_dict = {
            "rollout/reward_mean": raw_rewards_t.mean().item(),
            "rollout/reward_std": raw_rewards_t.std().item(),
            "rollout/advantage_mean": advantages.mean().item(),
            "rollout/log_prob_mean": old_log_probs_t.mean().item(),
            "rollout/value_mean": values_t.mean().item(),
            "rollout/episodes": ep_count,
            "train/actor_loss": np.mean(actor_losses),
            "train/value_loss": np.mean(value_losses),
            "train/ratio_mean": all_ratios.mean().item(),
            "train/ratio_std": all_ratios.std().item(),
            "train/ratio_max": all_ratios.max().item(),
            "train/ratio_min": all_ratios.min().item(),
            "train/update_time": update_time,
            "train/is_warmup": float(is_warmup),
        }

        pbar.set_postfix({
            "r": f"{raw_rewards_t.mean():.3f}",
            "a_loss": f"{np.mean(actor_losses):.4f}",
            "v_loss": f"{np.mean(value_losses):.4f}",
            "ratio": f"{all_ratios.mean():.3f}",
        })

        if update % args.log_freq == 0 and tracker:
            tracker.log(log_dict, step=update)

        # === Evaluation ===
        if update % args.eval_freq == 0 and update > start_update:
            eval_metrics = evaluate(
                policy, env, args.eval_episodes, args.max_episode_steps,
                obs_ref=obs_ref, use_learned_sampler=use_learned_sampler,
            )
            print(f"\n  Eval (update {update}): SR={eval_metrics['success_rate']:.0%}"
                  f" reward={eval_metrics['avg_reward']:.2f}"
                  f" steps={eval_metrics['avg_steps']:.0f}")

            if tracker:
                tracker.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=update)

            if eval_metrics["success_rate"] > best_sr:
                best_sr = eval_metrics["success_rate"]
                meta = {"update": update + 1, "best_sr": best_sr, "ep_count": ep_count}
                policy.save_checkpoint(
                    os.path.join(args.output, "best"), optimizers, meta
                )
                print(f"  New best! SR={best_sr:.0%}")

            # Reset env after eval
            obs, _ = env.reset()
            policy.policy.reset()
            ep_reward_accum = 0.0
            ep_steps = 0

        # === Checkpoint ===
        if update % args.save_freq == 0 and update > start_update:
            meta = {"update": update + 1, "best_sr": best_sr, "ep_count": ep_count}
            policy.save_checkpoint(
                os.path.join(args.output, f"update_{update}"), optimizers, meta
            )

    pbar.close()

    # Final save
    meta = {"update": total_updates, "best_sr": best_sr, "ep_count": ep_count}
    policy.save_checkpoint(os.path.join(args.output, "final"), optimizers, meta)

    elapsed = time.time() - t_start
    print(f"\nTraining complete. {args.total_updates} updates in {elapsed / 3600:.1f}h")
    print(f"  Best success rate: {best_sr:.0%}")
    print(f"  Output: {args.output}")

    cleanup_tracker(tracker, sys_monitor)

    env.close()


if __name__ == "__main__":
    main(args)
    app_launcher.app.close()
