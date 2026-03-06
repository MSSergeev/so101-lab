# Attribution

Sources, licenses, and inspiration for SO-101 Lab.

---

## LeRobot

**Source:** https://github.com/huggingface/lerobot
**License:** Apache 2.0
**Used for:** Motor bus (`FeetechMotorsBus`), dataset format (LeRobot v3.0, parquet + video), SmolVLA training pipeline, dataset recording infrastructure.

Code adapted from LeRobot carries a `# Adapted from: lerobot` header comment.

---

## leisaac

**Source:** https://github.com/LightwheelAI/leisaac
**License:** Apache 2.0
**Used for:** Isaac Lab environment structure — scene setup, task organization, spawn logic. The `so101_lab/tasks/` structure and `DirectRLEnv` patterns are inspired by leisaac.

---

## VIP

**Paper:** "VIP: Towards Universal Visual Reward and Representation" (Ma et al., 2023)
**Source:** https://github.com/facebookresearch/vip
**License:** CC BY-NC 4.0
**Used for:** Dense reward labeling in the IQL pipeline. ResNet50 encoder pre-trained on Ego4D video; rewards computed as negative distance in feature space. Weights downloaded to `~/.vip/resnet50/` on first use.

The `vip-utils` package is not used directly (incompatible dependencies). A standalone weight loader is implemented in `so101_lab/rewards/vip.py`.

---

## GreenVLA

**Paper:** "GreenVLA" (arxiv 2602.00919)
**Source:** https://github.com/greenvla/GreenVLA
**License:** Apache 2.0
**Used for:** Two ideas:
- IQL advantage-weighted BC fine-tuning of a VLA policy
- Learned Noise Sampler concept: "a small separate actor network that generates noise fed into the base model"

---

## IQL (Implicit Q-Learning)

**Paper:** "Offline Reinforcement Learning with Implicit Q-Learning" (Kostrikov et al., 2022, arxiv 2110.06169)
**Used for:** Offline critic training with VIP rewards. Twin Q-networks + value network trained on demonstration data, then used for advantage-weighted BC fine-tuning of SmolVLA. Implementation in `so101_lab/policies/rl/critics.py`.

---

## SAC (Soft Actor-Critic)

**Paper:** "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (Haarnoja et al., 2018, arxiv 1801.01290)
**Used for:** Online RL training via `train_sac_composite_grpc.py`. Uses LeRobot's `SACPolicy` implementation with RLPD (Reinforcement Learning with Prior Data) pattern — simultaneous learning from online rollouts and offline demonstrations.

See also: "Efficient Online Reinforcement Learning with Offline Data" (Ball et al., 2023, arxiv 2302.02948) for the RLPD approach.

---

## ReinFlow / DPPO — Flow-Noise PPO

**Paper:** "ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning" (Zhang et al., 2025, arxiv 2505.22094)
**Related:** "Diffusion Policy Policy Optimization" (Ren et al., 2024, ICLR 2025, arxiv 2409.00588)
**Used for:** The `FlowNoiseSmolVLA` wrapper (`so101_lab/policies/rl/flow_noise_smolvla.py`) implements the core idea from ReinFlow: injecting learnable Gaussian noise into the flow matching ODE, converting it from a deterministic path into a discrete-time Markov process with tractable log π(a|s) for PPO updates. Only the noise head + value MLP are trained (~639k params); the VLA backbone stays frozen. DPPO introduced the broader concept of applying PPO to denoising-based policies.
