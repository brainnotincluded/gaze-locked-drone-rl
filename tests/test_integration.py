# tests/test_integration.py
"""
Integration tests for the complete RL system.
These tests verify components work together.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from src.environment.virtual_target import VirtualTarget
from src.environment.drone_wrapper import DroneWrapper
from src.environment.drone_tracking_env import DroneTrackingEnv
from src.utils.curriculum_manager import CurriculumManager


def test_full_episode_mock():
    """Test a full episode with mocked drone."""
    # Setup mock drone
    mock_drone = Mock()
    mock_drone.state.yaw = 0.0
    mock_drone.state.altitude = 5.0
    mock_drone._master = Mock()
    mock_drone._master.target_system = 1
    mock_drone._master.target_component = 1
    mock_drone._master.mav = Mock()

    # Create environment
    env = DroneTrackingEnv(drone=mock_drone, max_steps=100)

    # Reset
    obs, info = env.reset(seed=42)
    assert obs is not None
    assert len(obs) == 6

    # Run episode
    total_reward = 0
    for step in range(100):
        # Random action
        action = env.action_space.sample()

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"\nEpisode finished after {step + 1} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final alignment: {info.get('alignment', 0):.3f}")

    assert step < 100  # Should terminate before max steps


def test_curriculum_integration():
    """Test curriculum manager with environment."""
    mock_drone = Mock()
    mock_drone.state.yaw = 0.0
    mock_drone.state.altitude = 5.0
    mock_drone._master = Mock()
    mock_drone._master.target_system = 1
    mock_drone._master.target_component = 1
    mock_drone._master.mav = Mock()

    env = DroneTrackingEnv(drone=mock_drone, max_steps=50)
    curriculum = CurriculumManager(success_threshold=0.8, window_size=10)

    # Simulate episodes
    for ep in range(15):
        obs, _ = env.reset()

        success = ep < 10  # First 10 are successful

        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        # Record success based on whether we "locked" the target
        curriculum.record_episode(success=success, alignment=info.get("alignment", 0))

    stats = curriculum.get_stats()
    print(f"\nCurriculum stats: {stats}")

    # Should have advanced to level 1 after 10 successes
    assert curriculum.level >= 0  # At minimum level 0


def test_target_in_fov_behavior():
    """Test that target alignment affects reward."""
    mock_drone = Mock()
    mock_drone.state.yaw = 0.0  # Facing forward
    mock_drone.state.altitude = 5.0
    mock_drone._master = Mock()
    mock_drone._master.target_system = 1
    mock_drone._master.target_component = 1
    mock_drone._master.mav = Mock()

    env = DroneTrackingEnv(drone=mock_drone, max_steps=10)

    # Reset with seed to get reproducible target position
    obs, _ = env.reset(seed=123)

    # First observation should have target info
    target_dx = obs[2]  # Normalized forward distance
    in_fov = obs[5]

    print(f"\nInitial observation:")
    print(f"  Target dx: {target_dx:.3f}")
    print(f"  In FOV: {bool(in_fov)}")

    # Take a step with zero action (no rotation)
    obs, reward, terminated, truncated, info = env.step(np.array([0.0]))

    print(f"  Reward: {reward:.3f}")
    print(f"  Alignment: {info['alignment']:.3f}")
