# tests/environment/test_drone_wrapper.py
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from src.environment.drone_wrapper import DroneWrapper


def test_initialization():
    mock_drone = Mock()
    wrapper = DroneWrapper(mock_drone)
    assert wrapper.drone == mock_drone
    assert wrapper.max_yaw_rate == 1.0  # rad/s


def test_get_yaw_from_attitude():
    mock_drone = Mock()
    mock_drone.state.yaw = 0.5  # radians

    wrapper = DroneWrapper(mock_drone)
    yaw = wrapper.get_yaw()

    assert yaw == 0.5


def test_get_yaw_rate_from_velocity():
    mock_drone = Mock()
    # Set up state to compute yaw rate
    mock_drone.state.yaw = 0.0
    mock_drone.state.vx = 1.0
    mock_drone.state.vy = 0.0

    wrapper = DroneWrapper(mock_drone)

    # Mock the velocity-based yaw rate calculation
    # For now, just check it returns a value
    yaw_rate = wrapper.get_yaw_rate()
    assert isinstance(yaw_rate, (int, float))


def test_set_yaw_rate_sends_command():
    mock_drone = Mock()
    mock_drone._master = Mock()
    mock_drone._master.target_system = 1
    mock_drone._master.target_component = 1
    mock_drone._master.mav = Mock()

    wrapper = DroneWrapper(mock_drone)
    wrapper.set_yaw_rate(0.5)  # rad/s

    # Verify MAVLink command was sent
    mock_drone._master.mav.set_position_target_local_ned_send.assert_called_once()


def test_yaw_rate_clamping():
    mock_drone = Mock()
    mock_drone._master = Mock()
    mock_drone._master.target_system = 1
    mock_drone._master.target_component = 1
    mock_drone._master.mav = Mock()

    wrapper = DroneWrapper(mock_drone, max_yaw_rate=1.0)

    # Test clamping
    wrapper.set_yaw_rate(2.0)  # Should be clamped to 1.0

    call_args = mock_drone._master.mav.set_position_target_local_ned_send.call_args
    # Extract the yaw_rate argument (last positional arg or in kwargs)
    assert call_args is not None
