# tests/utils/test_curriculum_manager.py
import pytest
import numpy as np
from src.utils.curriculum_manager import CurriculumManager


def test_initialization():
    cm = CurriculumManager()
    assert cm.level == 0
    assert cm.target_speed == 0.0


def test_level_progression():
    cm = CurriculumManager(success_threshold=0.8, window_size=10)

    # Simulate successful episodes
    for _ in range(10):
        cm.record_episode(success=True, alignment=0.9)

    assert cm.level == 1
    assert cm.target_speed > 0.0


def test_no_progression_with_failures():
    cm = CurriculumManager(success_threshold=0.8, window_size=10)

    # Mix of success and failure
    for i in range(10):
        cm.record_episode(success=(i % 2 == 0), alignment=0.5)

    assert cm.level == 0  # Should stay at level 0


def test_get_target_velocity():
    cm = CurriculumManager(success_threshold=0.8, window_size=10)

    # Level 0: static target
    vel = cm.get_target_velocity()
    assert np.allclose(vel, np.zeros(3))

    # Advance to level 1
    for _ in range(10):
        cm.record_episode(success=True, alignment=0.9)

    # Level 1: linear movement
    vel = cm.get_target_velocity()
    assert np.linalg.norm(vel) > 0


def test_get_trajectory_type():
    cm = CurriculumManager()

    assert cm.get_trajectory_type() == "static"

    # Advance levels
    for _ in range(20):
        cm.record_episode(success=True, alignment=0.9)

    if cm.level >= 2:
        assert cm.get_trajectory_type() in ["linear", "evasive"]
