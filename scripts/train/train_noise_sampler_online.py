"""Online REINFORCE training for learned noise sampler in sim.

Requires Isaac Lab (AppLauncher). For offline training, use
train_noise_sampler_offline.py (runs in lerobot-env, no sim needed).

Usage:
    python scripts/train/train_noise_sampler_online.py \
        --checkpoint outputs/easy_smolvla_vlm_pretrained_v1/checkpoints/030000/pretrained_model \
        --output outputs/easy_learned_sampler_online_v1 \
        --env figure_shape_placement_easy --no-domain-rand
"""

# Section 1: argparse + AppLauncher (before Isaac Sim init)
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Online noise sampler training (REINFORCE)")

# Environment
parser.add_argument("--env", type=str, default="figure_shape_placement",
                    help="Task environment name")

# Policy
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to SmolVLA BC checkpoint (pretrained_model dir)")
parser.add_argument("--task", type=str,
                    default="Place the cube into the matching slot on the platform")
parser.add_argument("--n-action-steps", type=int, default=None,
                    help="Override n_action_steps from model config")

# Sampler
parser.add_argument("--noise-dims", type=int, default=6, choices=[6, 32, 1600])

# Online mode
parser.add_argument("--episodes-per-batch", type=int, default=10)
parser.add_argument("--total-batches", type=int, default=100)

# Shared
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--max-episode-steps", type=int, default=500)
parser.add_argument("--eval-freq", type=int, default=10,
                    help="Evaluate every N batches")
parser.add_argument("--eval-episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, required=True)

# Resume
parser.add_argument("--resume", type=str, default=None,
                    help="Path to noise_prior.pt checkpoint to resume from")

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

import numpy as np
import torch

from lerobot.policies.factory import get_policy_class
from lerobot.processor.pipeline import PolicyProcessorPipeline
import lerobot.policies.smolvla.processor_smolvla  # noqa: F401

from so101_lab.policies.rl.learned_noise_sampler import LearnedNoiseSampler
from so101_lab.rl.isaac_lab_gym_env import IsaacLabGymEnv
from so101_lab.rl.domain_rand import apply_domain_rand_flags
from so101_lab.tasks import get_rl_task

ENV_IMAGE_KEYS = ["observation.images.top", "observation.images.wrist"]


def load_policy(checkpoint_path: str, device: str = "cuda"):
    """Load SmolVLA policy with processors."""
    policy_cls = get_policy_class("smolvla")
    policy = policy_cls.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()

    pre = PolicyProcessorPipeline.from_pretrained(
        checkpoint_path, "policy_preprocessor.json")
    post = PolicyProcessorPipeline.from_pretrained(
        checkpoint_path, "policy_postprocessor.json")

    policy_img_keys = sorted(policy.config.image_features.keys())
    env_keys = list(ENV_IMAGE_KEYS)
    if all(k in policy_img_keys for k in env_keys):
        img_map = {k: k for k in env_keys}
    else:
        img_map = {}
        for i, pk in enumerate(policy_img_keys):
            img_map[pk] = env_keys[i] if i < len(env_keys) else None

    return policy, pre, post, img_map


def obs_to_batch(obs, image_mapping, task_string):
    """Convert env obs to processor-ready batch."""
    batch = {
        "observation.state": torch.from_numpy(obs["observation.state"]).float(),
        "task": task_string,
    }
    for policy_key, env_key in image_mapping.items():
        if env_key is not None:
            img = torch.from_numpy(
                np.ascontiguousarray(obs[env_key])
            ).permute(2, 0, 1).float() / 255.0
        else:
            h, w = obs[ENV_IMAGE_KEYS[0]].shape[:2]
            img = torch.zeros(3, h, w)
        batch[policy_key] = img
    return batch


# ============================================================
# Online mode: REINFORCE
# ============================================================

def run_episode_online(env, policy, pre, post, img_map, task_str,
                       sampler, max_steps):
    """Run one episode collecting (vip_emb, state, x_action) for REINFORCE."""
    obs, _ = env.reset()
    policy.reset()
    ep_reward = 0.0

    obs_ref = [None]
    original_sample_noise = policy.model.sample_noise
    nd = sampler.noise_dims

    sampled_x_actions = []
    sampled_obs_data = []

    def tracking_sample_noise(shape, device):
        if obs_ref[0] is not None:
            mu, log_sigma = sampler.forward_from_obs(obs_ref[0])
            sigma = torch.exp(log_sigma)
            eps = torch.randn(nd, device=sampler.device)
            x_action = (mu.squeeze(0) + sigma.squeeze(0) * eps).detach()

            with torch.no_grad():
                import numpy as np_
                img_top = torch.from_numpy(
                    np_.ascontiguousarray(obs_ref[0]["observation.images.top"])
                ).unsqueeze(0)
                img_wrist = torch.from_numpy(
                    np_.ascontiguousarray(obs_ref[0]["observation.images.wrist"])
                ).unsqueeze(0)
                vip_emb = sampler.vip_backbone(img_top, img_wrist)
            state_t = torch.from_numpy(
                obs_ref[0]["observation.state"]
            ).unsqueeze(0).float().to(sampler.device)
            sampled_obs_data.append((vip_emb.detach(), state_t.detach()))
            sampled_x_actions.append(x_action.detach())

            x_0 = torch.randn(shape, device=device)
            x_0 = x_0.clone()
            if nd <= 32:
                x_0[0, 0, :nd] = x_action
            else:
                x_0[0, :, :] = x_action.reshape(50, 32)
            return x_0
        return original_sample_noise(shape, device)

    policy.model.sample_noise = tracking_sample_noise

    try:
        for step in range(max_steps):
            obs_ref[0] = obs
            batch = pre(obs_to_batch(obs, img_map, task_str))
            with torch.no_grad():
                action = policy.select_action(batch)
            action = post({"action": action})["action"]
            action_np = action.squeeze(0).cpu().numpy()[:6]
            action_np = np.clip(action_np, -100, 100)
            action_np[5] = np.clip(action_np[5], 0, 100)

            obs, _, terminated, truncated, _ = env.step(action_np)

            reward_details = env.get_reward_details()
            milestone = reward_details.get("reward/milestone_placed_weighted", 0.0)
            ep_reward += milestone

            if ep_reward > 0 or terminated or truncated:
                break
    finally:
        policy.model.sample_noise = original_sample_noise

    transitions = [
        (vip_emb, state_t, x_act)
        for (vip_emb, state_t), x_act in zip(sampled_obs_data, sampled_x_actions)
    ]
    return ep_reward > 0, step + 1, transitions


def run_eval_episode(env, policy, pre, post, img_map, task_str,
                     sampler, max_steps):
    """Run deterministic eval episode (sigma → 0, just use mu)."""
    obs, _ = env.reset()
    policy.reset()
    ep_reward = 0.0
    nd = sampler.noise_dims

    obs_ref = [None]
    original_sample_noise = policy.model.sample_noise

    def deterministic_sample_noise(shape, device):
        if obs_ref[0] is not None:
            mu, _ = sampler.forward_from_obs(obs_ref[0])
            x_0 = torch.randn(shape, device=device)
            x_0 = x_0.clone()
            if nd <= 32:
                x_0[0, 0, :nd] = mu.squeeze(0)
            else:
                x_0[0, :, :] = mu.squeeze(0).reshape(50, 32)
            return x_0
        return original_sample_noise(shape, device)

    policy.model.sample_noise = deterministic_sample_noise

    try:
        for step in range(max_steps):
            obs_ref[0] = obs
            batch = pre(obs_to_batch(obs, img_map, task_str))
            with torch.no_grad():
                action = policy.select_action(batch)
            action = post({"action": action})["action"]
            action_np = action.squeeze(0).cpu().numpy()[:6]
            action_np = np.clip(action_np, -100, 100)
            action_np[5] = np.clip(action_np[5], 0, 100)

            obs, _, terminated, truncated, _ = env.step(action_np)

            reward_details = env.get_reward_details()
            milestone = reward_details.get("reward/milestone_placed_weighted", 0.0)
            ep_reward += milestone

            if ep_reward > 0 or terminated or truncated:
                break
    finally:
        policy.model.sample_noise = original_sample_noise

    return ep_reward > 0, step + 1


def train_online(args, env, policy, pre, post, img_map, sampler):
    """REINFORCE training loop."""
    optimizer = torch.optim.Adam(sampler.mlp.parameters(), lr=args.lr)
    baseline = 0.0
    baseline_alpha = 0.1
    best_sr = 0.0
    log_path = os.path.join(args.output, "train_log.jsonl")
    log_file = open(log_path, "a")

    print(f"\n{'='*60}")
    print(f"Online Training (REINFORCE)")
    print(f"{'='*60}")
    print(f"  Episodes/batch: {args.episodes_per_batch}")
    print(f"  Total batches: {args.total_batches}")
    print(f"  Noise dims: {args.noise_dims}")
    print(f"  LR: {args.lr}")
    print(f"{'='*60}\n")

    for batch_idx in range(args.total_batches):
        batch_returns = []
        batch_transitions = []
        batch_successes = 0

        for ep in range(args.episodes_per_batch):
            success, steps, transitions = run_episode_online(
                env, policy, pre, post, img_map, args.task,
                sampler, args.max_episode_steps,
            )
            ret = float(success) + (
                0.1 * (1.0 - steps / args.max_episode_steps) if success else 0.0)
            batch_returns.append(ret)
            batch_transitions.append(transitions)
            batch_successes += int(success)

        # REINFORCE update
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=sampler.device)
        n_transitions = 0

        for transitions, ret in zip(batch_transitions, batch_returns):
            advantage = ret - baseline
            for vip_emb, state_t, x_action in transitions:
                lp = sampler.log_prob(x_action.unsqueeze(0), vip_emb, state_t)
                total_loss = total_loss - lp.squeeze() * advantage
                n_transitions += 1

        if n_transitions > 0:
            total_loss = total_loss / n_transitions
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(sampler.mlp.parameters(), 1.0)
            optimizer.step()

        mean_return = np.mean(batch_returns)
        baseline = baseline * (1 - baseline_alpha) + mean_return * baseline_alpha

        sr = batch_successes / args.episodes_per_batch
        print(f"  Batch {batch_idx+1}/{args.total_batches} | "
              f"SR={sr:.0%} | return={mean_return:.3f} | "
              f"loss={total_loss.item():.4f} | transitions={n_transitions}")

        log_entry = {
            "batch": batch_idx + 1, "sr": sr,
            "mean_return": mean_return, "loss": total_loss.item(),
            "baseline": baseline, "transitions": n_transitions,
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        # Eval
        if (batch_idx + 1) % args.eval_freq == 0:
            eval_successes = 0
            eval_total_steps = 0
            for _ in range(args.eval_episodes):
                success, steps = run_eval_episode(
                    env, policy, pre, post, img_map, args.task,
                    sampler, args.max_episode_steps,
                )
                eval_successes += int(success)
                eval_total_steps += steps
            eval_sr = eval_successes / args.eval_episodes
            eval_avg_steps = eval_total_steps / args.eval_episodes
            print(f"  >> Eval: SR={eval_sr:.0%}, avg_steps={eval_avg_steps:.0f}")

            if eval_sr >= best_sr:
                best_sr = eval_sr
                sampler.save(
                    os.path.join(args.output, "noise_prior.pt"),
                    extra={"eval_sr": eval_sr, "batch": batch_idx + 1},
                )
                print(f"  >> Saved best (SR={eval_sr:.0%})")

            log_entry = {
                "eval_batch": batch_idx + 1,
                "eval_sr": eval_sr, "eval_avg_steps": eval_avg_steps,
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

    log_file.close()
    sampler.save(
        os.path.join(args.output, "noise_prior_final.pt"),
        extra={"total_batches": args.total_batches, "best_sr": best_sr},
    )
    print(f"\nOnline training complete. Best SR: {best_sr:.0%}")


# ============================================================
# Main
# ============================================================

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # 1. Policy
    print(f"\nLoading SmolVLA from {args.checkpoint}")
    policy, pre, post, img_map = load_policy(args.checkpoint)

    if args.n_action_steps is not None:
        old = policy.config.n_action_steps
        policy.config.n_action_steps = args.n_action_steps
        print(f"n_action_steps: {old} → {args.n_action_steps}")

    for p in policy.parameters():
        p.requires_grad = False

    # 2. Sampler
    sampler = LearnedNoiseSampler(device="cuda", noise_dims=args.noise_dims)
    if args.resume:
        sampler.load(args.resume)
        print(f"Resumed sampler from {args.resume}")

    trainable = sum(p.numel() for p in sampler.mlp.parameters())
    print(f"Sampler: noise_dims={args.noise_dims}, trainable params={trainable:,}")

    # 3. Environment
    RLEnvCfgClass = get_rl_task(args.env)
    env_cfg = RLEnvCfgClass()
    env_cfg.scene.num_envs = 1
    apply_domain_rand_flags(env_cfg, args)
    env = IsaacLabGymEnv(env_cfg=env_cfg)

    # Baseline eval
    print("\n--- Baseline evaluation (standard N(0,I) noise) ---")
    baseline_successes = 0
    for i in range(min(args.eval_episodes, 5)):
        obs, _ = env.reset()
        policy.reset()
        ep_reward = 0.0
        for step in range(args.max_episode_steps):
            batch = pre(obs_to_batch(obs, img_map, args.task))
            with torch.no_grad():
                action = policy.select_action(batch)
            action = post({"action": action})["action"]
            action_np = action.squeeze(0).cpu().numpy()[:6]
            action_np = np.clip(action_np, -100, 100)
            action_np[5] = np.clip(action_np[5], 0, 100)
            obs, _, terminated, truncated, _ = env.step(action_np)
            reward_details = env.get_reward_details()
            milestone = reward_details.get("reward/milestone_placed_weighted", 0.0)
            ep_reward += milestone
            if ep_reward > 0 or terminated or truncated:
                break
        success = ep_reward > 0
        baseline_successes += int(success)
        print(f"  Baseline ep {i+1}: {'OK' if success else 'FAIL'} ({step+1} steps)")
    baseline_rate = baseline_successes / min(args.eval_episodes, 5)
    print(f"  Baseline SR: {baseline_rate:.0%}")

    # 4. Train
    train_online(args, env, policy, pre, post, img_map, sampler)
    env.close()


if __name__ == "__main__":
    main(args)
    app_launcher.app.close()
