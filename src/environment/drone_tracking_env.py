# src/environment/drone_tracking_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import time

from src.environment.virtual_target import VirtualTarget
from src.environment.drone_wrapper import DroneWrapper


class DroneTrackingEnv(gym.Env):
    """
    Gymnasium environment for drone gaze-locked tracking.

    The drone maintains position-hold mode while the RL agent
    controls yaw rate to track a virtual target.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        drone,
        max_steps: int = 1000,
        fov_deg: float = 60.0,
        max_yaw_rate: float = 1.0,
        control_freq: float = 50.0,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            drone: Existing Drone instance with MAVLink connection
            max_steps: Maximum steps per episode
            fov_deg: Camera field of view in degrees
            max_yaw_rate: Maximum yaw rate in rad/s
            control_freq: Control loop frequency in Hz
            render_mode: 'human' for visualization or None
        """
        super().__init__()

        self.drone_wrapper = DroneWrapper(drone, max_yaw_rate=max_yaw_rate)
        self.virtual_target = VirtualTarget()

        self.max_steps = max_steps
        self.fov_deg = fov_deg
        self.control_dt = 1.0 / control_freq

        # Action space: continuous yaw rate [-1.0, 1.0] (normalized)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space: [yaw, yaw_rate, dx, dy, dz, in_fov]
        # All normalized to [-1, 1] or [0, 1] where appropriate
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        self.render_mode = render_mode

        # Episode state
        self.steps = 0
        self.lock_duration = 0.0
        self.lost_duration = 0.0
        self.last_obs = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset episode without resetting simulation."""
        super().reset(seed=seed)

        # Reset virtual target position
        self.virtual_target.reset(
            r_min=5.0, r_max=50.0, azimuth_range=(0, 360), elevation_range=(-30, 30)
        )

        # Reset episode counters
        self.steps = 0
        self.lock_duration = 0.0
        self.lost_duration = 0.0

        # Get initial observation
        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one timestep."""
        # Scale action to actual yaw rate
        yaw_rate = action[0] * self.drone_wrapper.max_yaw_rate

        # Send command to drone
        self.drone_wrapper.set_yaw_rate(yaw_rate)

        # Advance virtual target
        self.virtual_target.step(self.control_dt)

        # Wait for control loop timing (if needed)
        time.sleep(self.control_dt)

        # Get observation
        obs = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(action[0])

        # Check termination conditions
        terminated = False
        truncated = False

        # Extract target info from obs (dx, dy, dz are indices 2,3,4)
        target_distance = np.linalg.norm(obs[2:5])
        in_fov = bool(obs[5])
        alignment = self.virtual_target.get_alignment(self.drone_wrapper.get_yaw())

        # Check if target is locked (centered and aligned)
        is_locked = in_fov and alignment > 0.95

        if is_locked:
            self.lock_duration += self.control_dt
            if self.lock_duration >= 5.0:
                reward += 50.0  # Success bonus
                terminated = True
        else:
            self.lock_duration = 0.0

        # Check if target is lost
        if not in_fov:
            self.lost_duration += self.control_dt
            if self.lost_duration >= 3.0:
                reward -= 100.0  # Failure penalty
                terminated = True
        else:
            self.lost_duration = 0.0

        # Check max steps
        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True

        info = {
            "steps": self.steps,
            "lock_duration": self.lock_duration,
            "lost_duration": self.lost_duration,
            "alignment": alignment,
            "target_distance": target_distance,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Compute current observation."""
        # Get drone state
        yaw = self.drone_wrapper.get_yaw()
        yaw_rate = self.drone_wrapper.get_yaw_rate()
        drone_pos = self.drone_wrapper.get_position()

        # Get target relative position
        rel_pos = self.virtual_target.get_relative_position(drone_pos, yaw)

        # Check if in FOV
        in_fov = self.virtual_target.is_in_fov(drone_pos, yaw, self.fov_deg)

        # Normalize observations
        yaw_norm = yaw / np.pi  # [-π, π] → [-1, 1]
        yaw_rate_norm = yaw_rate / self.drone_wrapper.max_yaw_rate

        # Normalize relative position (approximate max distance 100m)
        rel_pos_norm = np.clip(rel_pos / 100.0, -1.0, 1.0)

        obs = np.array(
            [
                yaw_norm,
                yaw_rate_norm,
                rel_pos_norm[0],  # dx (forward)
                rel_pos_norm[1],  # dy (right)
                rel_pos_norm[2],  # dz (up)
                float(in_fov),
            ],
            dtype=np.float32,
        )

        return obs

    def _calculate_reward(self, action: float) -> float:
        """Calculate step reward."""
        # Alignment reward
        alignment = self.virtual_target.get_alignment(self.drone_wrapper.get_yaw())
        reward = alignment * 10.0

        # Energy penalty
        reward -= 0.1 * abs(action)

        return reward

    def render(self):
        """Render environment (optional)."""
        pass

    def close(self):
        """Close environment."""
        pass
