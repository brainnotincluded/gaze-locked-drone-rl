# src/environment/virtual_target.py
import numpy as np
from typing import Tuple, Optional


class VirtualTarget:
    """Virtual target object for RL training - pure Python, no camera needed."""

    def __init__(self):
        self.position = np.zeros(3)  # [x, y, z] in meters
        self.velocity = np.zeros(3)  # [vx, vy, vz] in m/s
        self.trajectory_type = "static"  # 'static', 'linear', 'evasive'

    def reset(
        self,
        r_min: float = 5.0,
        r_max: float = 50.0,
        azimuth_range: Tuple[float, float] = (0, 360),
        elevation_range: Tuple[float, float] = (-30, 30),
    ) -> np.ndarray:
        """Reset target to random position in spherical coordinates."""
        # Random radius
        r = np.random.uniform(r_min, r_max)

        # Random azimuth (horizontal angle)
        azimuth = np.random.uniform(*azimuth_range)
        azimuth_rad = np.radians(azimuth)

        # Random elevation (vertical angle)
        elevation = np.random.uniform(*elevation_range)
        elevation_rad = np.radians(elevation)

        # Convert spherical to Cartesian (assuming drone at origin)
        # x: forward, y: right, z: up
        self.position = np.array(
            [
                r * np.cos(elevation_rad) * np.cos(azimuth_rad),
                r * np.cos(elevation_rad) * np.sin(azimuth_rad),
                r * np.sin(elevation_rad),
            ]
        )

        self.velocity = np.zeros(3)
        return self.position

    def set_velocity(self, velocity: np.ndarray):
        """Set target velocity for linear movement."""
        self.velocity = velocity.copy()

    def set_trajectory(self, trajectory_type: str):
        """Set trajectory type: 'static', 'linear', 'evasive'."""
        assert trajectory_type in ["static", "linear", "evasive"]
        self.trajectory_type = trajectory_type

    def step(self, dt: float):
        """Advance target position by one timestep."""
        if self.trajectory_type == "static":
            pass  # No movement
        elif self.trajectory_type == "linear":
            self.position += self.velocity * dt
        elif self.trajectory_type == "evasive":
            # Simple evasive: sinusoidal motion
            time_factor = np.linalg.norm(self.position) * 0.1
            evasive_vel = np.array(
                [
                    self.velocity[0] + 2.0 * np.sin(time_factor * 5),
                    self.velocity[1] + 2.0 * np.cos(time_factor * 3),
                    self.velocity[2],
                ]
            )
            self.position += evasive_vel * dt

    def get_relative_position(
        self, drone_position: np.ndarray, drone_yaw: float
    ) -> np.ndarray:
        """Get target position relative to drone's camera frame."""
        # World frame offset
        world_offset = self.position - drone_position

        # Transform to drone frame (rotate by -yaw around Z axis)
        cos_yaw = np.cos(-drone_yaw)
        sin_yaw = np.sin(-drone_yaw)

        # 2D rotation (ignoring roll/pitch, assuming level flight)
        rel_x = world_offset[0] * cos_yaw - world_offset[1] * sin_yaw
        rel_y = world_offset[0] * sin_yaw + world_offset[1] * cos_yaw
        rel_z = world_offset[2]

        return np.array([rel_x, rel_y, rel_z])

    def is_in_fov(
        self, drone_position: np.ndarray, drone_yaw: float, fov_deg: float = 60.0
    ) -> bool:
        """Check if target is within camera field of view."""
        rel_pos = self.get_relative_position(drone_position, drone_yaw)

        # Target must be in front of camera (positive x)
        if rel_pos[0] <= 0:
            return False

        # Calculate horizontal angle from camera center
        horizontal_angle = np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))

        # Check if within FOV
        half_fov = fov_deg / 2
        return abs(horizontal_angle) <= half_fov

    def get_alignment(self, drone_yaw: float) -> float:
        """Get cosine similarity between camera forward and target direction."""
        # Target direction in world frame
        direction = self.position / (np.linalg.norm(self.position) + 1e-8)

        # Camera forward vector in world frame (x, y components only)
        camera_forward = np.array([np.cos(drone_yaw), np.sin(drone_yaw)])

        # Cosine similarity (1.0 = perfectly aligned, -1.0 = opposite)
        return np.dot(camera_forward, direction[:2])
