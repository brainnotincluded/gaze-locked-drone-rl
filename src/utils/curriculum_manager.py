# src/utils/curriculum_manager.py
import numpy as np
from typing import List
from collections import deque


class CurriculumManager:
    """Manages curriculum learning for target difficulty."""

    def __init__(
        self, success_threshold: float = 0.8, window_size: int = 100, max_level: int = 3
    ):
        """
        Args:
            success_threshold: Success rate required to advance level
            window_size: Number of episodes to track for success rate
            max_level: Maximum curriculum level
        """
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.max_level = max_level

        self.level = 0
        self.episode_history = deque(maxlen=window_size)
        self.alignment_history = deque(maxlen=window_size)
        self.target_speed = 0.0  # Initialize target_speed for level 0

    def record_episode(self, success: bool, alignment: float):
        """Record episode outcome."""
        self.episode_history.append(success)
        self.alignment_history.append(alignment)

        # Check if should level up first
        should_level_up = False
        if len(self.episode_history) >= self.window_size:
            success_rate = sum(self.episode_history) / len(self.episode_history)
            if success_rate >= self.success_threshold and self.level < self.max_level:
                should_level_up = True

        if should_level_up:
            self.level += 1
            self.episode_history.clear()  # Reset for next level
            print(f"[Curriculum] Advanced to level {self.level}")

        # Update target_speed based on current level (after potential level up)
        if self.level == 0:
            self.target_speed = 0.0
        elif self.level == 1:
            self.target_speed = 1.5
        else:
            self.target_speed = 2.5

    def get_trajectory_type(self) -> str:
        """Get trajectory type for current level."""
        if self.level == 0:
            return "static"
        elif self.level == 1:
            return "linear"
        else:
            return "evasive"

    def get_target_velocity(self) -> np.ndarray:
        """Get target velocity for current level."""
        if self.level == 0:
            # Static target
            return np.zeros(3)
        elif self.level == 1:
            # Linear movement (1-2 m/s)
            speed = 1.5
            # Random direction
            angle = np.random.uniform(0, 2 * np.pi)
            return np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0])
        else:
            # Evasive maneuvers (2-3 m/s)
            speed = 2.5
            angle = np.random.uniform(0, 2 * np.pi)
            return np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0])

    def get_target_distance_range(self) -> tuple:
        """Get target distance range for current level."""
        if self.level == 0:
            return (5.0, 30.0)
        elif self.level == 1:
            return (10.0, 40.0)
        else:
            return (15.0, 50.0)

    def get_stats(self) -> dict:
        """Get curriculum statistics."""
        if len(self.episode_history) == 0:
            return {"level": self.level, "success_rate": 0.0, "episodes": 0}

        return {
            "level": self.level,
            "success_rate": sum(self.episode_history) / len(self.episode_history),
            "episodes": len(self.episode_history),
            "avg_alignment": sum(self.alignment_history) / len(self.alignment_history),
        }
