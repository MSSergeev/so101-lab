"""SmolVLA wrapper with stochastic noise head + value head for PPO.

Adds a learnable noise_log_std_head to the flow matching denoising loop,
making each ODE step sample velocity from N(v_mean, v_std). This yields
a tractable log π(a|s) for PPO training.

Trainable parameters (~639k):
  - action_out_proj (Linear 720→32): 23,072
  - noise_log_std_head (Linear 720→32): 23,072
  - value_mlp (2054→256→256→1): ~593k

Everything else (SmolVLA backbone, VIP backbone) is frozen.

PPO re-evaluation strategy:
  During rollout, we save per-step x_t and v_sampled tensors.
  At re-evaluation, we feed saved x_t into the (updated) denoise_step
  to get new v_mean/v_std, then compute log_prob of saved v_sampled
  under the new distribution. No need to re-run the full ODE.
"""

import numpy as np
import torch
import torch.nn as nn

from so101_lab.policies.rl.critics import VIPBackbone


ACTION_DIM = 6
ENV_IMAGE_KEYS = ["observation.images.top", "observation.images.wrist"]


class FlowNoiseSmolVLA:
    """SmolVLA + noise head + value head for PPO training."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        iql_checkpoint: str | None = None,
        task_string: str = "Place the cube into the matching slot on the platform",
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        kv_cache_device: str = "cpu",
    ):
        self.device = device
        self._checkpoint_path = checkpoint_path
        self.task_string = task_string
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.kv_cache_device = kv_cache_device

        # 1. Load SmolVLA via LeRobot API
        from lerobot.policies.factory import get_policy_class
        from lerobot.processor.pipeline import PolicyProcessorPipeline
        import lerobot.policies.smolvla.processor_smolvla  # noqa: F401

        policy_cls = get_policy_class("smolvla")
        self.policy = policy_cls.from_pretrained(checkpoint_path)
        self.policy.to(device)
        self.policy.eval()

        # Pre/post processors
        self.pre_processor = PolicyProcessorPipeline.from_pretrained(
            checkpoint_path, "policy_preprocessor.json"
        )
        self.post_processor = PolicyProcessorPipeline.from_pretrained(
            checkpoint_path, "policy_postprocessor.json"
        )
        self.image_mapping = self._build_image_key_mapping()

        # 2. Freeze everything
        for p in self.policy.parameters():
            p.requires_grad = False

        # 3. Unfreeze action_out_proj
        for p in self.policy.model.action_out_proj.parameters():
            p.requires_grad = True

        # 4. Add noise_log_std_head (same dims as action_out_proj)
        expert_hidden_size = self.policy.model.vlm_with_expert.expert_hidden_size
        max_action_dim = self.policy.config.max_action_dim
        self.noise_log_std_head = nn.Linear(expert_hidden_size, max_action_dim).to(device)
        nn.init.constant_(self.noise_log_std_head.bias, -2.0)
        nn.init.zeros_(self.noise_log_std_head.weight)

        # 5. Value head: VIP backbone (frozen) + MLP
        self.vip_backbone = VIPBackbone(device)
        self.value_mlp = self._build_value_mlp(device)

        # 6. Warm start value head from IQL checkpoint
        if iql_checkpoint:
            self._load_iql_value_weights(iql_checkpoint)

        # Cache action feature dim for unpadding
        self._action_dim = self.policy.config.action_feature.shape[0]
        self._kv_cache_logged = False

        print(f"FlowNoiseSmolVLA initialized:")
        print(f"  expert_hidden_size={expert_hidden_size}, max_action_dim={max_action_dim}")
        print(f"  Trainable actor params: {self._count_params(self.trainable_actor_params())}")
        print(f"  Trainable value params: {self._count_params(self.value_mlp.parameters())}")

    def _build_image_key_mapping(self) -> dict[str, str | None]:
        """Map policy image keys to env image keys."""
        policy_img_keys = sorted(self.policy.config.image_features.keys())
        env_keys = list(ENV_IMAGE_KEYS)

        if all(k in policy_img_keys for k in env_keys):
            return {k: k for k in env_keys}

        mapping = {}
        for i, policy_key in enumerate(policy_img_keys):
            mapping[policy_key] = env_keys[i] if i < len(env_keys) else None
        return mapping

    def _build_value_mlp(self, device: str) -> nn.Sequential:
        """Build value MLP: VIP(2048) + state(6) → 256 → 256 → 1."""
        mlp = nn.Sequential(
            nn.Linear(2054, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )
        return mlp.to(device)

    def _load_iql_value_weights(self, iql_checkpoint: str):
        """Load V-net weights from IQL critics checkpoint."""
        ckpt = torch.load(iql_checkpoint, map_location=self.device, weights_only=False)
        if "v_net" not in ckpt:
            print(f"Warning: no v_net in {iql_checkpoint}, skipping warm start")
            return

        v_state = ckpt["v_net"]
        new_state = {}
        for k, v in v_state.items():
            new_key = k
            for prefix in ("mlp.", "net."):
                if k.startswith(prefix):
                    new_key = k[len(prefix):]
                    break
            new_state[new_key] = v

        self.value_mlp.load_state_dict(new_state)
        print(f"Loaded IQL V-net weights from {iql_checkpoint}")

    def _log_kv_cache_info(self, prefix_pad_masks, past_key_values):
        """Log KV cache structure and size (called once on first forward pass)."""
        prefix_len = prefix_pad_masks.shape[1]
        num_layers = len(past_key_values)

        total_bytes = 0
        sample_layer = next(iter(past_key_values.values()))
        k_shape = sample_layer["key_states"].shape
        v_shape = sample_layer["value_states"].shape
        dtype = sample_layer["key_states"].dtype

        for layer_kv in past_key_values.values():
            for t in layer_kv.values():
                total_bytes += t.nelement() * t.element_size()

        mb = total_bytes / 1024 / 1024
        print(f"  KV cache: {num_layers} layers, prefix_len={prefix_len}, stored on {self.kv_cache_device}")
        print(f"    key shape={list(k_shape)}, value shape={list(v_shape)}, dtype={dtype}")
        print(f"    Size per obs: {mb:.1f} MB, 500 obs: {500 * mb / 1024:.1f} GB")

    @staticmethod
    def _count_params(params) -> str:
        n = sum(p.numel() for p in params if p.requires_grad)
        return f"{n:,}"

    def trainable_actor_params(self):
        """Yield trainable actor parameters (action_out_proj + noise_log_std_head)."""
        yield from self.policy.model.action_out_proj.parameters()
        yield from self.noise_log_std_head.parameters()

    def _obs_to_raw_batch(self, obs: dict) -> dict:
        """Convert IsaacLabGymEnv obs dict to raw batch for processor pipeline."""
        batch = {
            "observation.state": torch.from_numpy(obs["observation.state"]).float(),
            "task": self.task_string,
        }
        for policy_key, env_key in self.image_mapping.items():
            if env_key is not None:
                img = torch.from_numpy(obs[env_key]).permute(2, 0, 1).float() / 255.0
            else:
                h, w = obs[ENV_IMAGE_KEYS[0]].shape[:2]
                img = torch.zeros(3, h, w)
            batch[policy_key] = img
        return batch

    # --- Prefix computation (shared by sample and re-evaluate) ---

    def _compute_prefix(self, obs: dict):
        """Run SmolVLA prefix (images + language + state) → KV cache.

        Returns (prefix_pad_masks, past_key_values) needed for denoise steps.
        All frozen — no gradients needed.
        """
        raw_batch = self._obs_to_raw_batch(obs)
        batch = self.pre_processor(raw_batch)

        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)

        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        model = self.policy.model

        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            _, past_key_values = model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=model.config.use_cache,
                fill_kv_cache=True,
            )

        return prefix_pad_masks, past_key_values

    # --- Denoise step internals ---

    def _get_suffix_out(self, model, prefix_pad_masks, past_key_values, x_t, timestep):
        """Run frozen expert transformer to get suffix_out (384-dim).

        Everything here is frozen — gradients only flow through
        action_out_proj and noise_log_std_head applied to suffix_out.
        """
        with torch.no_grad():
            suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(
                x_t, timestep
            )

            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len
            )

            from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
            )
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            outputs_embeds, _ = model.vlm_with_expert.forward(
                attention_mask=full_att_2d_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=model.config.use_cache,
                fill_kv_cache=False,
            )

            suffix_out = outputs_embeds[1][:, -model.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=torch.float32)

        # Detach and re-enable grad so action_out_proj/noise_log_std_head
        # can compute gradients w.r.t. their own parameters
        return suffix_out.detach().requires_grad_(True)

    def _compute_gaussian_log_prob(self, v_mean, v_log_std, v_sampled):
        """Gaussian log_prob over first chunk step and real action dims only.

        Only the first step of the chunk (index 0) and first _action_dim dims
        (6 out of 32) are used for env actions. Summing over all 50×32=1600
        padding dims causes ratio explosion.

        Result: sum over _action_dim (6) per ODE step, ~O(1) per step.
        """
        # Slice to first chunk step, real action dims: (B, action_dim)
        v_mean_s = v_mean[:, 0, :self._action_dim]
        v_log_std_s = v_log_std[:, 0, :self._action_dim]
        v_sampled_s = v_sampled[:, 0, :self._action_dim]

        v_std_s = torch.exp(v_log_std_s)
        var_s = v_std_s.pow(2)
        log_prob = -0.5 * ((v_sampled_s - v_mean_s).pow(2) / var_s) - v_log_std_s
        return log_prob.sum(dim=-1)  # (B,)

    # --- Main API ---

    def sample_actions_with_log_prob(self, obs: dict) -> tuple[np.ndarray, torch.Tensor, dict]:
        """Sample stochastic actions from the flow matching ODE with noise injection.

        Returns:
            actions: (ACTION_DIM,) numpy array in normalized motor space [-100, 100]
            log_prob: scalar tensor (detached), accumulated log probability
            trajectory: dict with per-step x_t, v_sampled, and cached prefix KV
        """
        model = self.policy.model
        device = torch.device(self.device)

        # Prefix (frozen, no grad)
        prefix_pad_masks, past_key_values = self._compute_prefix(obs)
        bsize = prefix_pad_masks.shape[0]

        # Log KV cache size once
        if not self._kv_cache_logged:
            self._log_kv_cache_info(prefix_pad_masks, past_key_values)
            self._kv_cache_logged = True

        # Initial noise
        actions_shape = (bsize, model.config.chunk_size, model.config.max_action_dim)
        x_0 = model.sample_noise(actions_shape, device)

        # Denoising loop — collect trajectory for re-evaluation
        num_steps = model.config.num_steps
        dt = -1.0 / num_steps
        x_t = x_0

        traj_x_t = []       # x_t at each step input (for re-eval)
        traj_v_sampled = []  # sampled velocity at each step (for re-eval)
        total_log_prob = torch.zeros(bsize, device=device)

        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(
                time, dtype=torch.float32, device=device
            ).expand(bsize)

            # Save x_t before this step
            traj_x_t.append(x_t.detach().clone())

            # Get suffix_out (frozen transformer, detached)
            suffix_out = self._get_suffix_out(
                model, prefix_pad_masks, past_key_values, x_t.detach(), time_tensor
            )

            # Trainable heads
            v_mean = model.action_out_proj(suffix_out)
            v_log_std = self.noise_log_std_head(suffix_out)
            v_log_std = torch.clamp(v_log_std, self.log_std_min, self.log_std_max)
            v_std = torch.exp(v_log_std)

            # Sample velocity
            eps = torch.randn_like(v_mean)
            v_sampled = v_mean + v_std * eps

            traj_v_sampled.append(v_sampled.detach().clone())

            # Gaussian log_prob (has grad through v_mean, v_log_std)
            step_lp = self._compute_gaussian_log_prob(v_mean, v_log_std, v_sampled.detach())
            total_log_prob = total_log_prob + step_lp

            # Euler step (detach — ODE trajectory doesn't backprop)
            x_t = x_t.detach() + dt * v_sampled.detach()

        # Unpad + post-process
        actions = x_t[:, :, :self._action_dim]
        action_tensor = self.post_processor({"action": actions.detach()})["action"]

        action_np = action_tensor[0, 0].cpu().numpy()
        action_np = np.clip(action_np[:ACTION_DIM], -100, 100)
        action_np[5] = np.clip(action_np[5], 0, 100)

        # Cache prefix KV for re-evaluation (avoids recomputing 450M prefix)
        # past_key_values is dict[int, {"key_states": tensor, "value_states": tensor}]
        cache_dev = self.kv_cache_device
        cached_kv = {
            layer_idx: {k: v.to(cache_dev) for k, v in layer_kv.items()}
            for layer_idx, layer_kv in past_key_values.items()
        }

        trajectory = {
            "x_t": [t.cpu() for t in traj_x_t],             # list of (1, 50, 32)
            "v_sampled": [t.cpu() for t in traj_v_sampled],  # list of (1, 50, 32)
            "prefix_pad_masks": prefix_pad_masks.to(cache_dev),  # (1, prefix_len)
            "prefix_kv": cached_kv,                          # dict of dicts on cache_dev
        }

        return action_np, total_log_prob.squeeze(0).detach(), trajectory

    def run_ode_with_x0(self, obs: dict, x_0: torch.Tensor) -> list[np.ndarray]:
        """Run deterministic ODE from provided x_0. Returns full action chunk.

        Uses frozen action_out_proj only (no noise_log_std_head).
        No gradients, no trajectory caching — pure inference.

        Args:
            obs: observation dict from IsaacLabGymEnv.
            x_0: (1, chunk_size, max_action_dim) initial noise tensor.

        Returns:
            List of n_action_steps (ACTION_DIM,) numpy arrays.
        """
        model = self.policy.model
        device = torch.device(self.device)

        with torch.no_grad():
            prefix_pad_masks, past_key_values = self._compute_prefix(obs)
            bsize = prefix_pad_masks.shape[0]

            num_steps = model.config.num_steps
            dt = -1.0 / num_steps
            x_t = x_0.to(device)

            for step in range(num_steps):
                time = 1.0 + step * dt
                time_tensor = torch.tensor(
                    time, dtype=torch.float32, device=device
                ).expand(bsize)

                suffix_out = self._get_suffix_out(
                    model, prefix_pad_masks, past_key_values, x_t, time_tensor
                )
                v_mean = model.action_out_proj(suffix_out)
                x_t = x_t + dt * v_mean

            actions = x_t[:, :, :self._action_dim]
            action_tensor = self.post_processor({"action": actions})["action"]

        # Return n_action_steps actions from the chunk
        n_steps = self.policy.config.n_action_steps
        chunk = []
        for i in range(n_steps):
            a = action_tensor[0, i].cpu().numpy()
            a = np.clip(a[:ACTION_DIM], -100, 100)
            a[5] = np.clip(a[5], 0, 100)
            chunk.append(a)
        return chunk

    def reeval_log_prob(self, obs: dict, trajectory: dict) -> torch.Tensor:
        """Re-evaluate log_prob of a stored trajectory under current parameters.

        For PPO: feeds saved x_t into (updated) action_out_proj + noise_log_std_head,
        computes log_prob of saved v_sampled under new distribution.
        No ODE re-simulation needed.

        If trajectory contains cached prefix KV (from sample_actions_with_log_prob),
        uses it directly instead of recomputing the expensive 450M prefix forward.
        This is the main speed optimization: 2000 prefix forwards → 0.

        Returns:
            log_prob: scalar tensor WITH grad (for PPO actor loss).
        """
        model = self.policy.model
        device = torch.device(self.device)

        # Use cached prefix KV if available (main speedup)
        if "prefix_kv" in trajectory:
            prefix_pad_masks = trajectory["prefix_pad_masks"].to(device)
            past_key_values = {
                layer_idx: {k: v.to(device) for k, v in layer_kv.items()}
                for layer_idx, layer_kv in trajectory["prefix_kv"].items()
            }
        else:
            prefix_pad_masks, past_key_values = self._compute_prefix(obs)
        bsize = prefix_pad_masks.shape[0]

        num_steps = model.config.num_steps
        dt = -1.0 / num_steps
        total_log_prob = torch.zeros(bsize, device=device)

        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(
                time, dtype=torch.float32, device=device
            ).expand(bsize)

            # Use saved x_t and v_sampled from rollout
            x_t = trajectory["x_t"][step].to(device)
            v_sampled = trajectory["v_sampled"][step].to(device)

            # Frozen transformer → suffix_out
            suffix_out = self._get_suffix_out(
                model, prefix_pad_masks, past_key_values, x_t, time_tensor
            )

            # Trainable heads (current params → new v_mean, v_std)
            v_mean = model.action_out_proj(suffix_out)
            v_log_std = self.noise_log_std_head(suffix_out)
            v_log_std = torch.clamp(v_log_std, self.log_std_min, self.log_std_max)

            # Log_prob of saved v_sampled under new distribution
            step_lp = self._compute_gaussian_log_prob(v_mean, v_log_std, v_sampled)
            total_log_prob = total_log_prob + step_lp

        return total_log_prob.squeeze(0)  # scalar with grad

    def reeval_log_prob_batched(self, trajectories: list[dict]) -> torch.Tensor:
        """Batched re-eval: stack N trajectories, run one suffix forward with batch=N.

        Same as reeval_log_prob but processes multiple obs in parallel for better
        GPU utilization. Requires prefix_kv in all trajectories.

        Returns:
            log_probs: (N,) tensor WITH grad.
        """
        N = len(trajectories)
        device = torch.device(self.device)
        model = self.policy.model

        # Stack prefix KV caches along batch dim
        prefix_pad_masks = torch.cat(
            [t["prefix_pad_masks"].to(device) for t in trajectories], dim=0
        )

        first_kv = trajectories[0]["prefix_kv"]
        past_key_values = {}
        for layer_idx in first_kv:
            past_key_values[layer_idx] = {
                "key_states": torch.cat(
                    [t["prefix_kv"][layer_idx]["key_states"].to(device) for t in trajectories], dim=0
                ),
                "value_states": torch.cat(
                    [t["prefix_kv"][layer_idx]["value_states"].to(device) for t in trajectories], dim=0
                ),
            }

        num_steps = model.config.num_steps
        dt = -1.0 / num_steps
        total_log_prob = torch.zeros(N, device=device)

        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(
                time, dtype=torch.float32, device=device
            ).expand(N)

            x_t = torch.cat([t["x_t"][step].to(device) for t in trajectories], dim=0)
            v_sampled = torch.cat([t["v_sampled"][step].to(device) for t in trajectories], dim=0)

            suffix_out = self._get_suffix_out(
                model, prefix_pad_masks, past_key_values, x_t, time_tensor
            )

            v_mean = model.action_out_proj(suffix_out)
            v_log_std = self.noise_log_std_head(suffix_out)
            v_log_std = torch.clamp(v_log_std, self.log_std_min, self.log_std_max)

            step_lp = self._compute_gaussian_log_prob(v_mean, v_log_std, v_sampled)
            total_log_prob = total_log_prob + step_lp

        return total_log_prob  # (N,) with grad

    def sample_actions_deterministic(self, obs: dict) -> np.ndarray:
        """Deterministic action selection for evaluation (standard SmolVLA inference)."""
        raw_batch = self._obs_to_raw_batch(obs)
        batch = self.pre_processor(raw_batch)

        with torch.no_grad():
            action = self.policy.select_action(batch)

        action = self.post_processor({"action": action})["action"]
        action_np = action.squeeze(0).cpu().numpy()[:ACTION_DIM]
        action_np = np.clip(action_np, -100, 100)
        action_np[5] = np.clip(action_np[5], 0, 100)
        return action_np

    @torch.no_grad()
    def get_value(self, obs: dict) -> torch.Tensor:
        """Compute value estimate V(s) using VIP backbone + MLP."""
        img_top = torch.from_numpy(obs["observation.images.top"]).unsqueeze(0)
        img_wrist = torch.from_numpy(obs["observation.images.wrist"]).unsqueeze(0)
        state = torch.from_numpy(obs["observation.state"]).unsqueeze(0).float().to(self.device)

        vip_emb = self.vip_backbone(img_top, img_wrist)  # (1, 2048)
        x = torch.cat([vip_emb, state], dim=-1)  # (1, 2054)
        return self.value_mlp(x).squeeze(-1)  # (1,)

    def get_value_from_cached(
        self, vip_emb: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Compute value from pre-cached VIP embeddings."""
        x = torch.cat([vip_emb, state], dim=-1)
        return self.value_mlp(x).squeeze(-1)

    @torch.no_grad()
    def cache_vip_embedding(self, obs: dict) -> torch.Tensor:
        """Compute and return VIP embedding for caching. Returns (2048,)."""
        img_top = torch.from_numpy(obs["observation.images.top"]).unsqueeze(0)
        img_wrist = torch.from_numpy(obs["observation.images.wrist"]).unsqueeze(0)
        return self.vip_backbone(img_top, img_wrist).squeeze(0)

    def set_eval_noise_bounds(self):
        """Set near-deterministic noise bounds for evaluation."""
        self.log_std_min = -20.0
        self.log_std_max = -2.0

    def set_train_noise_bounds(self):
        """Restore training noise bounds."""
        self.log_std_min = -5.0
        self.log_std_max = 2.0

    def export_smolvla_checkpoint(self, output_dir: str):
        """Export a standalone SmolVLA checkpoint with updated action_out_proj.

        Copies the original SmolVLA checkpoint and overwrites action_out_proj
        weights in the safetensors file. The result is a standard SmolVLA
        checkpoint usable with eval_vla_policy.py and lerobot tools.

        noise_log_std_head is NOT included — inference is deterministic.
        """
        import json
        import os
        import shutil

        from safetensors.torch import load_file, save_file

        src = self._checkpoint_path
        os.makedirs(output_dir, exist_ok=True)

        # Copy all non-safetensor files (config, processors)
        for fname in os.listdir(src):
            src_file = os.path.join(src, fname)
            dst_file = os.path.join(output_dir, fname)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)

        # Patch safetensors with updated action_out_proj
        st_path = os.path.join(output_dir, "model.safetensors")
        tensors = load_file(st_path)

        # Find action_out_proj keys and replace
        proj_state = self.policy.model.action_out_proj.state_dict()
        for k, v in proj_state.items():
            full_key = f"model.action_out_proj.{k}"
            if full_key in tensors:
                tensors[full_key] = v.cpu()
            else:
                # Try without "model." prefix
                alt_key = f"action_out_proj.{k}"
                if alt_key in tensors:
                    tensors[alt_key] = v.cpu()

        save_file(tensors, st_path)
        print(f"Exported SmolVLA checkpoint: {output_dir}")

    def save_checkpoint(self, path: str, optimizers: dict | None = None, metadata: dict | None = None):
        """Save trainable parameters, optimizer state, and SmolVLA export."""
        import os
        os.makedirs(path, exist_ok=True)

        # 1. Training checkpoint (overlay + optimizers)
        state = {
            "action_out_proj": self.policy.model.action_out_proj.state_dict(),
            "noise_log_std_head": self.noise_log_std_head.state_dict(),
            "value_mlp": self.value_mlp.state_dict(),
        }
        if optimizers:
            state["optimizers"] = {k: v.state_dict() for k, v in optimizers.items()}
        if metadata:
            state["metadata"] = metadata

        torch.save(state, os.path.join(path, "flow_noise_ppo.pt"))

        # 2. Standalone SmolVLA checkpoint for eval_vla_policy.py
        export_dir = os.path.join(path, "pretrained_model")
        self.export_smolvla_checkpoint(export_dir)

        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str, optimizers: dict | None = None) -> dict:
        """Load trainable parameters and optimizer state. Returns metadata dict."""
        import os
        ckpt_path = os.path.join(path, "flow_noise_ppo.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.policy.model.action_out_proj.load_state_dict(ckpt["action_out_proj"])
        self.noise_log_std_head.load_state_dict(ckpt["noise_log_std_head"])
        self.value_mlp.load_state_dict(ckpt["value_mlp"])

        if optimizers and "optimizers" in ckpt:
            for k, opt in optimizers.items():
                if k in ckpt["optimizers"]:
                    opt.load_state_dict(ckpt["optimizers"][k])

        print(f"Loaded FlowNoiseSmolVLA checkpoint: {path}")
        return ckpt.get("metadata", {})
