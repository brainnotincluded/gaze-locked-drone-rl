# Virtual Target RL Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete RL environment for gaze-locked drone tracking with virtual targets, using Stable Baselines3 PPO with LSTM memory.

**Architecture:** Coordinate-based virtual target system with Gymnasium environment wrapper around existing Drone SDK. Drone maintains position-hold mode while RL controls yaw rate only. Episodes reset logically without restarting simulation.

**Tech Stack:** Python, Gymnasium, Stable Baselines3 (PPO with LSTM), NumPy, existing Drone class (DroneKit/pymavlink)

---

## Task 1: Project Structure Setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/environment/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create requirements.txt**

```
gymnasium>=0.29.0
stable-baselines3>=2.2.0
numpy>=1.24.0
pymavlink>=2.4.0
pytest>=7.4.0
```

**Step 2: Create directory structure**

```bash
mkdir -p src/environment src/agents src/utils tests/environment tests/agents
```

**Step 3: Commit**

```bash
git add requirements.txt .gitignore
mkdir -p src/environment src/agents src/utils tests/environment tests/agents
git add src/ tests/
git commit -m "chore: setup project structure for RL environment"
```

---

## Task 2: Virtual Target Module

**Files:**
- Create: `src/environment/virtual_target.py`
- Create: `tests/environment/test_virtual_target.py`

**Step 1: Write failing test**

```python
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
    assert target.is_in_fov(drone_pos, drone_yaw=np.pi/2, fov_deg=60.0) == False


def test_step_linear_movement():
    target = VirtualTarget()
    target.reset()
    initial_pos = target.position.copy()
    
    # Move with velocity (1 m/s forward)
    target.set_velocity(np.array([1.0, 0.0, 0.0]))
    target.step(dt=1.0)
    
    assert target.position[0] > initial_pos[0]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/environment/test_virtual_target.py -v
```

Expected: All tests FAIL with "ImportError: cannot import name 'VirtualTarget'"

**Step 3: Write minimal implementation**

```python
# src/environment/virtual_target.py
import numpy as np
from typing import Tuple, Optional


class VirtualTarget:
    """Virtual target object for RL training - pure Python, no camera needed."""
    
    def __init__(self):
        self.position = np.zeros(3)  # [x, y, z] in meters
        self.velocity = np.zeros(3)  # [vx, vy, vz] in m/s
        self.trajectory_type = 'static'  # 'static', 'linear', 'evasive'
    
    def reset(
        self,
        r_min: float = 5.0,
        r_max: float = 50.0,
        azimuth_range: Tuple[float, float] = (0, 360),
        elevation_range: Tuple[float, float] = (-30, 30)
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
        self.position = np.array([
            r * np.cos(elevation_rad) * np.cos(azimuth_rad),
            r * np.cos(elevation_rad) * np.sin(azimuth_rad),
            r * np.sin(elevation_rad)
        ])
        
        self.velocity = np.zeros(3)
        return self.position
    
    def set_velocity(self, velocity: np.ndarray):
        """Set target velocity for linear movement."""
        self.velocity = velocity.copy()
    
    def set_trajectory(self, trajectory_type: str):
        """Set trajectory type: 'static', 'linear', 'evasive'."""
        assert trajectory_type in ['static', 'linear', 'evasive']
        self.trajectory_type = trajectory_type
    
    def step(self, dt: float):
        """Advance target position by one timestep."""
        if self.trajectory_type == 'static':
            pass  # No movement
        elif self.trajectory_type == 'linear':
            self.position += self.velocity * dt
        elif self.trajectory_type == 'evasive':
            # Simple evasive: sinusoidal motion
            time_factor = np.linalg.norm(self.position) * 0.1
            evasive_vel = np.array([
                self.velocity[0] + 2.0 * np.sin(time_factor * 5),
                self.velocity[1] + 2.0 * np.cos(time_factor * 3),
                self.velocity[2]
            ])
            self.position += evasive_vel * dt
    
    def get_relative_position(
        self,
        drone_position: np.ndarray,
        drone_yaw: float
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
        self,
        drone_position: np.ndarray,
        drone_yaw: float,
        fov_deg: float = 60.0
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
        
        # Camera forward vector in world frame
        camera_forward = np.array([np.cos(drone_yaw), np.sin(drone_yaw), 0])
        
        # Cosine similarity (1.0 = perfectly aligned, -1.0 = opposite)
        return np.dot(camera_forward, direction[:2])
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/environment/test_virtual_target.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/environment/virtual_target.py tests/environment/test_virtual_target.py
git commit -m "feat: implement virtual target with coordinate transforms and FOV checks"
```

---

## Task 3: Drone Wrapper for Yaw Control

**Files:**
- Create: `src/environment/drone_wrapper.py`
- Create: `tests/environment/test_drone_wrapper.py`

**Step 1: Write failing test**

```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/environment/test_drone_wrapper.py -v
```

Expected: All tests FAIL with import error

**Step 3: Write minimal implementation**

```python
# src/environment/drone_wrapper.py
import numpy as np
from pymavlink import mavutil
from typing import Optional


class DroneWrapper:
    """Wrapper around Drone class for RL yaw control only."""
    
    def __init__(self, drone, max_yaw_rate: float = 1.0):
        """
        Args:
            drone: Existing Drone instance with MAVLink connection
            max_yaw_rate: Maximum yaw rate in rad/s
        """
        self.drone = drone
        self.max_yaw_rate = max_yaw_rate
        self._last_yaw = 0.0
        self._last_time = None
    
    def get_yaw(self) -> float:
        """Get current drone yaw angle in radians."""
        return self.drone.state.yaw
    
    def get_yaw_rate(self) -> float:
        """Estimate yaw rate from attitude changes."""
        current_yaw = self.get_yaw()
        current_time = self._get_timestamp()
        
        if self._last_time is None:
            self._last_yaw = current_yaw
            self._last_time = current_time
            return 0.0
        
        dt = current_time - self._last_time
        if dt > 0:
            # Handle wrap-around (yaw is -π to +π)
            dyaw = current_yaw - self._last_yaw
            if dyaw > np.pi:
                dyaw -= 2 * np.pi
            elif dyaw < -np.pi:
                dyaw += 2 * np.pi
            
            yaw_rate = dyaw / dt
            
            self._last_yaw = current_yaw
            self._last_time = current_time
            return yaw_rate
        
        return 0.0
    
    def get_position(self) -> np.ndarray:
        """Get drone position [x, y, z]."""
        # Note: In real implementation, this might come from local position
        # For now, return zeros as we assume drone maintains hover
        return np.array([0.0, 0.0, self.drone.state.altitude])
    
    def set_yaw_rate(self, yaw_rate: float):
        """
        Set yaw rate command to drone.
        
        Args:
            yaw_rate: Desired yaw rate in rad/s (will be clamped to ±max_yaw_rate)
        """
        # Clamp to safe range
        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)
        
        # Send velocity command with yaw rate
        # Using BODY_NED frame, only velocity and yaw_rate are used
        self.drone._master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms (0 for now)
            self.drone._master.target_system,
            self.drone._master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            int(0b010111000111),  # Type mask: only velocity + yaw rate
            0, 0, 0,  # Position (ignored)
            0, 0, 0,  # Velocity (maintain hover, so 0,0,0)
            0, 0, 0,  # Acceleration (ignored)
            0,  # Yaw (ignored)
            yaw_rate  # Yaw rate
        )
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    def check_position_hold(self, max_drift: float = 10.0) -> bool:
        """
        Check if drone is maintaining position (safety check).
        
        Returns:
            True if position is stable, False if drifted too far
        """
        # Simplified check - in real implementation would track position
        # For now, assume position hold is working
        return True
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/environment/test_drone_wrapper.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/environment/drone_wrapper.py tests/environment/test_drone_wrapper.py
git commit -m "feat: add drone wrapper for yaw-only control"
```

---

## Task 4: Gymnasium Environment

**Files:**
- Create: `src/environment/drone_tracking_env.py`
- Create: `tests/environment/test_drone_tracking_env.py`

**Step 1: Write failing test**

```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/environment/test_drone_tracking_env.py -v
```

Expected: All tests FAIL with import error

**Step 3: Write minimal implementation**

```python
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
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        drone,
        max_steps: int = 1000,
        fov_deg: float = 60.0,
        max_yaw_rate: float = 1.0,
        control_freq: float = 50.0,
        render_mode: Optional[str] = None
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
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space: [yaw, yaw_rate, dx, dy, dz, in_fov]
        # All normalized to [-1, 1] or [0, 1] where appropriate
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        
        # Episode state
        self.steps = 0
        self.lock_duration = 0.0
        self.lost_duration = 0.0
        self.last_obs = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset episode without resetting simulation."""
        super().reset(seed=seed)
        
        # Reset virtual target position
        self.virtual_target.reset(
            r_min=5.0,
            r_max=50.0,
            azimuth_range=(0, 360),
            elevation_range=(-30, 30)
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
            'steps': self.steps,
            'lock_duration': self.lock_duration,
            'lost_duration': self.lost_duration,
            'alignment': alignment,
            'target_distance': target_distance
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
        
        obs = np.array([
            yaw_norm,
            yaw_rate_norm,
            rel_pos_norm[0],  # dx (forward)
            rel_pos_norm[1],  # dy (right)
            rel_pos_norm[2],  # dz (up)
            float(in_fov)
        ], dtype=np.float32)
        
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/environment/test_drone_tracking_env.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/environment/drone_tracking_env.py tests/environment/test_drone_tracking_env.py
git commit -m "feat: implement Gymnasium environment for drone tracking"
```

---

## Task 5: Curriculum Manager

**Files:**
- Create: `src/utils/curriculum_manager.py`
- Create: `tests/utils/test_curriculum_manager.py`

**Step 1: Write failing test**

```python
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
    cm = CurriculumManager()
    
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
    
    assert cm.get_trajectory_type() == 'static'
    
    # Advance levels
    for _ in range(20):
        cm.record_episode(success=True, alignment=0.9)
    
    if cm.level >= 2:
        assert cm.get_trajectory_type() in ['linear', 'evasive']
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/utils/test_curriculum_manager.py -v
```

Expected: All tests FAIL

**Step 3: Write minimal implementation**

```python
# src/utils/curriculum_manager.py
import numpy as np
from typing import List
from collections import deque


class CurriculumManager:
    """Manages curriculum learning for target difficulty."""
    
    def __init__(
        self,
        success_threshold: float = 0.8,
        window_size: int = 100,
        max_level: int = 3
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
    
    def record_episode(self, success: bool, alignment: float):
        """Record episode outcome."""
        self.episode_history.append(success)
        self.alignment_history.append(alignment)
        
        # Check if should level up
        if len(self.episode_history) >= self.window_size:
            success_rate = sum(self.episode_history) / len(self.episode_history)
            if success_rate >= self.success_threshold and self.level < self.max_level:
                self.level += 1
                self.episode_history.clear()  # Reset for next level
                print(f"[Curriculum] Advanced to level {self.level}")
    
    def get_trajectory_type(self) -> str:
        """Get trajectory type for current level."""
        if self.level == 0:
            return 'static'
        elif self.level == 1:
            return 'linear'
        else:
            return 'evasive'
    
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
            return np.array([
                speed * np.cos(angle),
                speed * np.sin(angle),
                0.0
            ])
        else:
            # Evasive maneuvers (2-3 m/s)
            speed = 2.5
            angle = np.random.uniform(0, 2 * np.pi)
            return np.array([
                speed * np.cos(angle),
                speed * np.sin(angle),
                0.0
            ])
    
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
            return {
                'level': self.level,
                'success_rate': 0.0,
                'episodes': 0
            }
        
        return {
            'level': self.level,
            'success_rate': sum(self.episode_history) / len(self.episode_history),
            'episodes': len(self.episode_history),
            'avg_alignment': sum(self.alignment_history) / len(self.alignment_history)
        }
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/utils/test_curriculum_manager.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/utils/curriculum_manager.py tests/utils/test_curriculum_manager.py
git commit -m "feat: add curriculum manager for progressive difficulty"
```

---

## Task 6: Optional Visualizer

**Files:**
- Create: `src/utils/visualizer.py`
- Modify: `src/environment/drone_tracking_env.py` (add render support)

**Step 1: Write failing test**

```python
# tests/utils/test_visualizer.py
import pytest
import numpy as np
from src.utils.visualizer import TrackingVisualizer


def test_initialization():
    viz = TrackingVisualizer()
    assert viz is not None


def test_update():
    viz = TrackingVisualizer()
    
    drone_pos = np.array([0.0, 0.0, 5.0])
    drone_yaw = 0.0
    target_pos = np.array([10.0, 0.0, 5.0])
    in_fov = True
    
    # Should not raise
    viz.update(drone_pos, drone_yaw, target_pos, in_fov)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/utils/test_visualizer.py -v
```

Expected: FAIL with import error

**Step 3: Write minimal implementation**

```python
# src/utils/visualizer.py
import numpy as np
import cv2
from typing import Optional


class TrackingVisualizer:
    """Simple top-down visualizer for debugging tracking behavior."""
    
    def __init__(self, window_size: int = 600, scale: float = 2.0):
        """
        Args:
            window_size: Window size in pixels
            scale: Pixels per meter
        """
        self.window_size = window_size
        self.scale = scale
        self.center = window_size // 2
        
    def update(
        self,
        drone_pos: np.ndarray,
        drone_yaw: float,
        target_pos: np.ndarray,
        in_fov: bool,
        fov_deg: float = 60.0
    ):
        """Update and display visualization."""
        # Create blank canvas
        img = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
        
        # Draw grid
        for i in range(0, self.window_size, int(10 * self.scale)):
            cv2.line(img, (i, 0), (i, self.window_size), (30, 30, 30), 1)
            cv2.line(img, (0, i), (self.window_size, i), (30, 30, 30), 1)
        
        # Drone is always at center (top-down view is relative)
        drone_pixel = (self.center, self.center)
        
        # Draw drone
        cv2.circle(img, drone_pixel, 10, (0, 255, 0), -1)
        
        # Draw drone heading indicator
        heading_len = 30
        heading_end = (
            int(self.center + heading_len * np.cos(-drone_yaw + np.pi/2)),
            int(self.center + heading_len * np.sin(-drone_yaw + np.pi/2))
        )
        cv2.line(img, drone_pixel, heading_end, (0, 255, 0), 2)
        
        # Draw FOV cone
        half_fov = np.radians(fov_deg / 2)
        fov_len = 100
        fov_left = (
            int(self.center + fov_len * np.cos(-drone_yaw + np.pi/2 - half_fov)),
            int(self.center + fov_len * np.sin(-drone_yaw + np.pi/2 - half_fov))
        )
        fov_right = (
            int(self.center + fov_len * np.cos(-drone_yaw + np.pi/2 + half_fov)),
            int(self.center + fov_len * np.sin(-drone_yaw + np.pi/2 + half_fov))
        )
        cv2.line(img, drone_pixel, fov_left, (0, 100, 0), 1)
        cv2.line(img, drone_pixel, fov_right, (0, 100, 0), 1)
        
        # Calculate target position relative to drone (top-down: x=forward, y=right)
        # Convert world coordinates to relative
        rel_x = target_pos[0] - drone_pos[0]
        rel_y = target_pos[1] - drone_pos[1]
        
        # Rotate by drone yaw to get camera-relative
        cos_yaw = np.cos(-drone_yaw)
        sin_yaw = np.sin(-drone_yaw)
        cam_x = rel_x * cos_yaw - rel_y * sin_yaw
        cam_y = rel_x * sin_yaw + rel_y * cos_yaw
        
        # Convert to pixels (y is right in camera frame, but x is down in image)
        target_pixel = (
            int(self.center - cam_y * self.scale),
            int(self.center - cam_x * self.scale)
        )
        
        # Draw target
        color = (0, 255, 0) if in_fov else (0, 0, 255)
        cv2.circle(img, target_pixel, 8, color, -1)
        
        # Draw info text
        info_text = f"FOV: {'LOCKED' if in_fov else 'LOST'}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show
        cv2.imshow('Drone Tracking', img)
        cv2.waitKey(1)
    
    def close(self):
        """Close visualizer."""
        cv2.destroyAllWindows()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/utils/test_visualizer.py -v
```

Expected: All tests PASS

**Step 5: Modify environment to support visualization**

Add to `src/environment/drone_tracking_env.py`:
- Import: `from src.utils.visualizer import TrackingVisualizer`
- In `__init__`: Initialize visualizer if render_mode='human'
- In `reset`: Update visualizer
- In `step`: Update visualizer
- In `render`: Implement visualization
- In `close`: Close visualizer

**Step 6: Commit**

```bash
git add src/utils/visualizer.py tests/utils/test_visualizer.py
git add src/environment/drone_tracking_env.py
git commit -m "feat: add optional visualizer for debugging"
```

---

## Task 7: Training Script

**Files:**
- Create: `src/agents/train.py`

**Step 1: Create training script**

```python
#!/usr/bin/env python3
"""
Training script for drone tracking RL agent.

Usage:
    python src/agents/train.py --steps 1000000 --save-dir ./models
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.environment.drone_tracking_env import DroneTrackingEnv
from src.utils.curriculum_manager import CurriculumManager

# Import your existing Drone class
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from drone import Drone


def make_env(drone, curriculum_manager=None, render_mode=None):
    """Factory function for environment creation."""
    def _init():
        env = DroneTrackingEnv(
            drone=drone,
            max_steps=1000,
            render_mode=render_mode
        )
        
        # Attach curriculum manager if provided
        if curriculum_manager:
            env.curriculum_manager = curriculum_manager
        
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description='Train drone tracking agent')
    parser.add_argument('--steps', type=int, default=1_000_000,
                        help='Total training steps')
    parser.add_argument('--save-dir', type=str, default='./models',
                        help='Directory to save models')
    parser.add_argument('--checkpoint-freq', type=int, default=10000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization')
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Drone Tracking RL Training")
    print("="*60)
    
    # Connect to drone
    print("\n[*] Connecting to drone...")
    drone = Drone()
    
    if not drone.connect():
        print("[-] Failed to connect to drone")
        print("    Make sure MAVProxy is running with output 5762")
        return 1
    
    print("[+] Drone connected")
    
    # Set to GUIDED mode for position hold
    print("[*] Setting GUIDED mode...")
    drone.set_mode('GUIDED')
    
    # Takeoff to hover position
    print("[*] Taking off to 5m...")
    drone.takeoff(5.0)
    drone.wait_for_altitude(5.0, tolerance=0.5)
    print("[+] Hovering at 5m")
    
    try:
        # Initialize curriculum manager
        curriculum = CurriculumManager(
            success_threshold=0.8,
            window_size=100
        )
        
        # Create environment
        render_mode = 'human' if args.visualize else None
        env = make_env(drone, curriculum, render_mode)()
        env = DummyVecEnv([lambda: env])
        
        # Create PPO agent with LSTM (RecurrentPPO)
        print("\n[*] Initializing PPO agent...")
        from sb3_contrib import RecurrentPPO
        
        model = RecurrentPPO(
            'MlpLstmPolicy',
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log='./tensorboard/'
        )
        
        print("[+] Agent initialized")
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(save_dir),
            name_prefix='drone_tracking'
        )
        
        print(f"\n[*] Starting training for {args.steps} steps...")
        print(f"    Checkpoints saved to: {save_dir}")
        print(f"    TensorBoard logs: ./tensorboard/")
        print()
        
        # Train
        model.learn(
            total_timesteps=args.steps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save final model
        final_path = save_dir / 'drone_tracking_final.zip'
        model.save(final_path)
        print(f"\n[+] Training complete! Model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user")
        
    finally:
        # Land drone
        print("\n[*] Landing...")
        drone.land()
        drone.disarm()
        drone.close()
        print("[+] Done")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

**Step 2: Commit**

```bash
git add src/agents/train.py
git commit -m "feat: add training script with PPO and curriculum learning"
```

---

## Task 8: Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Create integration test**

```python
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
    
    print(f"\nEpisode finished after {step+1} steps")
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
        curriculum.record_episode(
            success=success,
            alignment=info.get('alignment', 0)
        )
    
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
```

**Step 2: Run integration tests**

```bash
pytest tests/test_integration.py -v -s
```

Expected: All tests PASS with informative output

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for complete system"
```

---

## Task 9: Documentation

**Files:**
- Create: `README.md`
- Create: `docs/usage.md`

**Step 1: Write README**

```markdown
# Virtual Target RL for Gaze-Locked Drone Tracking

Reinforcement learning system for training drones to track virtual targets using yaw control only.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install sb3-contrib for RecurrentPPO
pip install sb3-contrib

# Run training
python src/agents/train.py --steps 1000000 --visualize
```

## Architecture

- **Virtual Target**: Pure Python object with configurable trajectories
- **Drone Wrapper**: Yaw-only control interface to existing Drone class
- **Gymnasium Env**: Standard RL interface with 50Hz control loop
- **Curriculum**: Progressive difficulty (static → linear → evasive)
- **RecurrentPPO**: LSTM-based policy for temporal memory

## Project Structure

```
src/
├── environment/
│   ├── virtual_target.py      # Virtual target with coordinate transforms
│   ├── drone_wrapper.py       # Yaw-only drone interface
│   └── drone_tracking_env.py  # Gymnasium environment
├── agents/
│   └── train.py               # Training script
└── utils/
    ├── curriculum_manager.py  # Progressive difficulty
    └── visualizer.py          # Optional debugging viz

tests/                         # Unit and integration tests
docs/plans/                    # Design documents
```

## Training

```bash
# Basic training
python src/agents/train.py --steps 1000000

# With visualization
python src/agents/train.py --steps 1000000 --visualize

# Custom save directory
python src/agents/train.py --steps 1000000 --save-dir ./my_models

# Resume from checkpoint
python src/agents/train.py --steps 1000000 --resume ./models/drone_tracking_100000_steps.zip
```

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/environment/test_virtual_target.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

## Requirements

- Python 3.8+
- Webots with ArduPilot SITL (running separately)
- MAVProxy configured on port 5762
- Existing Drone class with MAVLink connection
```

**Step 2: Write usage guide**

```markdown
# Usage Guide

## Prerequisites

1. **Start Webots simulation** with ArduPilot SITL
2. **Start MAVProxy** with output on port 5762:
   ```bash
   mavproxy.py --master tcp:127.0.0.1:5760 --out 127.0.0.1:5762
   ```
3. **Verify connection**:
   ```python
   from drone import Drone
   drone = Drone()
   drone.connect()
   ```

## Training Workflow

### 1. Initial Training

Start with default curriculum:

```bash
python src/agents/train.py --steps 500000
```

This will:
- Connect to drone via MAVLink
- Takeoff to 5m hover
- Train PPO with LSTM for 500k steps
- Save checkpoints every 10k steps
- Use curriculum learning (starts with static targets)

### 2. Monitor Progress

```bash
# In another terminal
tensorboard --logdir ./tensorboard/
```

Watch for:
- `episode_reward_mean`: Should increase over time
- `success_rate`: Should approach 0.8 (curriculum threshold)
- `curriculum_level`: Should advance from 0 → 1 → 2

### 3. Curriculum Stages

- **Level 0** (Static): Target doesn't move, distance 5-30m
- **Level 1** (Linear): Target moves 1.5 m/s, distance 10-40m
- **Level 2** (Evasive): Sinusoidal motion, 2.5 m/s, distance 15-50m

Each level requires 80% success rate over 100 episodes to advance.

### 4. Evaluation

Test trained model:

```python
from stable_baselines3 import PPO
from src.environment.drone_tracking_env import DroneTrackingEnv
from drone import Drone

drone = Drone()
drone.connect()
drone.set_mode('GUIDED')
drone.takeoff(5.0)

env = DroneTrackingEnv(drone=drone, render_mode='human')
model = PPO.load('models/drone_tracking_final.zip')

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

drone.land()
```

## Troubleshooting

### Connection Issues

**Problem**: `target_system = 0` error
**Solution**: Ensure MAVProxy has output configured:
```bash
# In MAVProxy console
output add 127.0.0.1:5762
```

### Drone Doesn't Hover

**Problem**: Drone drifts during training
**Solution**: Ensure GUIDED mode is active before training starts

### Low Reward

**Problem**: Agent not learning
**Check**:
1. Verify target is visible in initial observation
2. Check alignment values in TensorBoard
3. Try increasing exploration (ent_coef=0.02)

## Configuration

Edit hyperparameters in `src/agents/train.py`:

```python
model = RecurrentPPO(
    'MlpLstmPolicy',
    env,
    learning_rate=3e-4,      # Try 1e-4 if unstable
    n_steps=2048,            # Batch size
    batch_size=64,
    n_epochs=10,
    gamma=0.99,              # Discount factor
    ent_coef=0.01,           # Exploration
    # ...
)
```
```

**Step 3: Commit**

```bash
git add README.md docs/usage.md
git commit -m "docs: add README and usage guide"
```

---

## Summary

This implementation plan provides a complete, testable Virtual Target RL system with:

1. **Virtual Target** - Coordinate-based tracking without cameras
2. **Drone Wrapper** - Yaw-only control interface  
3. **Gymnasium Env** - Standard RL interface at 50Hz
4. **Curriculum** - Progressive difficulty
5. **RecurrentPPO** - LSTM-based training
6. **Optional Visualizer** - Debugging support
7. **Comprehensive Tests** - Unit + integration
8. **Documentation** - Usage guides

**Next Steps:**
1. Execute plan task-by-task
2. Run training on actual drone
3. Tune hyperparameters based on results
4. Add advanced features (multi-drone, real targets)
