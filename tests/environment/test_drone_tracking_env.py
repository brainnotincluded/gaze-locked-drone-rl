# tests/environment/test_drone_tracking_env.py
import pytest
import numpy as np
from unittest.mock import Mock
import gymnasium as gym
from src.environment.drone_tracking_env import DroneTrackingEnv


def test_environment_initialization():
    mock_drone = Mock()
    mock_drone.state.yaw = 0.0
    mock_drone.state.altitude = 5.0

    env = DroneTrackingEnv(drone=mock_drone)

    assert env is not None
    assert env.action_space is not None
    assert env.observation_space is not None


def test_reset_returns_observation():
    mock_drone = Mock()
    mock_drone.state.yaw = 0.0
    mock_drone.state.altitude = 5.0

    env = DroneTrackingEnv(drone=mock_drone)
    obs, info = env.reset(seed=42)

    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert len(obs) == 6  # [yaw, yaw_rate, dx, dy, dz, in_fov]
    assert isinstance(info, dict)


def test_step_returns_tuple():
    mock_drone = Mock()
    mock_drone.state.yaw = 0.0
    mock_drone.state.altitude = 5.0
    mock_drone._master = Mock()
    mock_drone._master.target_system = 1
    mock_drone._master.target_component = 1
    mock_drone._master.mav = Mock()

    env = DroneTrackingEnv(drone=mock_drone)
    obs, _ = env.reset(seed=42)

    action = np.array([0.5])  # Positive yaw rate
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_reward_calculation():
    mock_drone = Mock()
    mock_drone.state.yaw = 0.0
    mock_drone.state.altitude = 5.0
    mock_drone._master = Mock()
    mock_drone._master.target_system = 1
    mock_drone._master.target_component = 1
    mock_drone._master.mav = Mock()

    env = DroneTrackingEnv(drone=mock_drone)
    env.reset(seed=42)

    # Step with zero action
    obs, reward, terminated, truncated, info = env.step(np.array([0.0]))

    # Reward should be calculated based on alignment
    assert isinstance(reward, float)


def test_episode_termination_max_steps():
    mock_drone = Mock()
    mock_drone.state.yaw = 0.0
    mock_drone.state.altitude = 5.0
    mock_drone._master = Mock()
    mock_drone._master.target_system = 1
    mock_drone._master.target_component = 1
    mock_drone._master.mav = Mock()

    env = DroneTrackingEnv(drone=mock_drone, max_steps=10)
    env.reset(seed=42)

    truncated = False
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))

    assert truncated == True
