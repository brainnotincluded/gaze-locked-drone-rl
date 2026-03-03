# tests/environment/test_virtual_target.py
import pytest
import numpy as np
from src.environment.virtual_target import VirtualTarget


def test_initialization():
    target = VirtualTarget()
    assert target.position is not None
    assert len(target.position) == 3


def test_reset_randomizes_position():
    target = VirtualTarget()
    target.reset(r_min=5.0, r_max=50.0)

    # Check spherical coordinates are within bounds
    r = np.linalg.norm(target.position)
    assert 5.0 <= r <= 50.0

    # Check elevation bounds
    x, y, z = target.position
    elevation = np.degrees(np.arcsin(z / r))
    assert -30.0 <= elevation <= 30.0


def test_relative_position_calculation():
    target = VirtualTarget()
    target.position = np.array([10.0, 0.0, 0.0])  # 10m ahead

    # Drone at origin, facing forward (0° yaw)
    drone_pos = np.array([0.0, 0.0, 0.0])
    drone_yaw = 0.0

    rel_pos = target.get_relative_position(drone_pos, drone_yaw)

    # Target should be directly in front
    assert rel_pos[0] > 0  # x positive (forward)
    assert abs(rel_pos[1]) < 0.01  # y ~ 0 (no lateral offset)


def test_in_fov_check():
    target = VirtualTarget()
    target.position = np.array([10.0, 0.0, 0.0])

    drone_pos = np.array([0.0, 0.0, 0.0])

    # Target directly ahead should be in FOV
    assert target.is_in_fov(drone_pos, drone_yaw=0.0, fov_deg=60.0) == True

    # Target 90° to the side should be out of FOV
    assert target.is_in_fov(drone_pos, drone_yaw=np.pi / 2, fov_deg=60.0) == False


def test_step_linear_movement():
    target = VirtualTarget()
    target.reset()
    target.set_trajectory("linear")
    initial_pos = target.position.copy()

    # Move with velocity (1 m/s forward)
    target.set_velocity(np.array([1.0, 0.0, 0.0]))
    target.step(dt=1.0)

    assert target.position[0] > initial_pos[0]
