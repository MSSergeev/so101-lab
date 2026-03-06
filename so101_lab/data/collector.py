"""Recording manager for teleoperation demonstrations."""

from pathlib import Path

import numpy as np
import torch

from so101_lab.data.converters import joint_rad_to_motor_normalized
from so101_lab.data.lerobot_dataset import LeRobotDatasetWriter


class RecordingManager:
    """
    Manages episode recording lifecycle during teleoperation.

    Integrates with Isaac Lab environment to collect demonstrations
    directly in LeRobot format.
    """

    def __init__(self, dataset: LeRobotDatasetWriter, env):
        """
        Initialize recording manager.

        Args:
            dataset: LeRobotDatasetWriter instance for writing frames
            env: Isaac Lab environment instance
        """
        self.dataset = dataset
        self.env = env
        self._recording_active = False
        self._episode_start_time = 0.0

        # Sim rewards side-file buffer
        self._sim_rewards_episode: list[dict[str, float]] = []
        self._sim_rewards_all: dict[str, list[float]] = {}

    def set_task(self, task: str) -> None:
        """Set task description for next episode."""
        self.dataset.set_task(task)

    def set_episode_seed(self, seed: int) -> None:
        """Set seed for current episode."""
        self.dataset.set_episode_seed(seed)

    def set_episode_initial_state(self, state: dict) -> None:
        """Set initial prim state for current episode."""
        self.dataset.set_episode_initial_state(state)

    def on_reset(self, obs_dict: dict) -> None:
        """
        Called after env.reset() to start new episode recording.

        Args:
            obs_dict: Observation dictionary from environment
        """
        # Clear any leftover frames from previous aborted recording
        self.dataset.clear_episode()
        self._recording_active = True
        self._episode_start_time = 0.0

    def on_step(self, obs_dict: dict, action: torch.Tensor) -> None:
        """
        Called BEFORE env.step() to record frame.

        Records (observation_before, action) pair following standard
        imitation learning semantics: action is taken in observation state.

        Args:
            obs_dict: Observation dictionary from environment (BEFORE action)
            action: Action tensor to be executed
        """
        if not self._recording_active:
            return

        frame = self._build_frame(obs_dict, action)
        self.dataset.add_frame(frame)

    def on_episode_end(self, success: bool = True) -> None:
        """
        Called when episode ends to save or discard recording.

        Args:
            success: Whether to save episode (True) or discard (False)
        """
        if not self._recording_active:
            return

        if success:
            self.dataset.save_episode()
        else:
            self.dataset.clear_episode()

        self._recording_active = False

    def set_last_reward(self, reward: float) -> None:
        """Attach reward to last recorded frame."""
        if self.dataset._episode_buffer:
            self.dataset._episode_buffer[-1]["next.reward"] = np.float32(reward)

    def add_sim_rewards(self, metrics: dict[str, float]) -> None:
        """Buffer per-frame sim reward metrics for side-file."""
        self._sim_rewards_episode.append(metrics)

    def flush_sim_rewards(self) -> None:
        """Move episode sim rewards to global buffer (called on save)."""
        if not self._sim_rewards_episode:
            return
        ep_idx = self.dataset._current_episode_idx
        n = len(self._sim_rewards_episode)
        if "episode_index" not in self._sim_rewards_all:
            self._sim_rewards_all["episode_index"] = []
        self._sim_rewards_all["episode_index"].extend([ep_idx] * n)
        for key in self._sim_rewards_episode[0]:
            if key not in self._sim_rewards_all:
                self._sim_rewards_all[key] = []
            self._sim_rewards_all[key].extend(
                m[key] for m in self._sim_rewards_episode
            )
        self._sim_rewards_episode.clear()

    def load_sim_rewards(self, path: Path) -> None:
        """Load existing sim rewards for resume."""
        if not path.exists():
            return
        data = torch.load(path, weights_only=True)
        for k, v in data.items():
            self._sim_rewards_all[k] = v.tolist()

    def save_sim_rewards(self, path: Path) -> None:
        """Save accumulated sim rewards to .pt file."""
        if not self._sim_rewards_all:
            return
        data = {}
        for k, v in self._sim_rewards_all.items():
            dtype = torch.int64 if k == "episode_index" else torch.float32
            data[k] = torch.tensor(v, dtype=dtype)
        torch.save(data, path)

    def _build_frame(self, obs_dict: dict, action: torch.Tensor) -> dict:
        """
        Convert Isaac Lab observation to LeRobot frame format.

        Converts joint positions from radians to normalized motor positions
        for LeRobot compatibility.

        Args:
            obs_dict: Observation dictionary with "policy" group
            action: Action tensor [num_envs, action_dim]

        Returns:
            Dictionary with LeRobot-compatible keys and numpy arrays
        """
        obs = obs_dict["policy"]

        # Convert tensors to numpy arrays (radians)
        joint_pos_rad = obs["joint_pos"][0].cpu().numpy()  # [6]
        action_rad = action[0].cpu().numpy()  # [6] or [8]

        # Handle action dimension
        if action_rad.shape[0] == 8:
            # For keyboard/gamepad, use joint_pos_target from environment
            action_rad = obs["joint_pos_target"][0].cpu().numpy()[:6]
        elif action_rad.shape[0] != 6:
            raise ValueError(f"Unexpected action dimension: {action_rad.shape[0]}")

        # Convert radians → normalized motor positions for LeRobot
        joint_pos = joint_rad_to_motor_normalized(joint_pos_rad)
        action_np = joint_rad_to_motor_normalized(action_rad)

        # Extract camera images
        top_img = obs["top"][0].cpu().numpy() if "top" in obs else np.zeros((480, 640, 3), dtype=np.uint8)
        wrist_img = obs["wrist"][0].cpu().numpy() if "wrist" in obs else np.zeros((480, 640, 3), dtype=np.uint8)

        # Convert images from float [0, 1] to uint8 [0, 255] if needed
        if top_img.dtype in (np.float32, np.float64):
            top_img = (top_img * 255).astype(np.uint8)
        if wrist_img.dtype in (np.float32, np.float64):
            wrist_img = (wrist_img * 255).astype(np.uint8)

        # Calculate timestamp
        if hasattr(self.env, "episode_length_buf") and hasattr(self.env, "step_dt"):
            current_time = self.env.episode_length_buf[0].item() * self.env.step_dt
        else:
            current_time = 0.0
        timestamp = current_time - self._episode_start_time

        frame = {
            "observation.state": joint_pos.astype(np.float32),
            "action": action_np.astype(np.float32),
            "observation.images.top": top_img,
            "observation.images.wrist": wrist_img,
            "timestamp": timestamp,
        }

        return frame
