# src/environment/yolo_human_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
import time
from ultralytics import YOLO

from src.environment.drone_wrapper import DroneWrapper


class YOLOHumanEnv(gym.Env):
    """
    Gymnasium environment for drone tracking real humans using YOLO detection.

    The drone maintains position-hold mode while the RL agent
    controls yaw rate to track detected humans.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        drone,
        max_steps: int = 1000,
        max_yaw_rate: float = 1.0,
        control_freq: float = 50.0,
        render_mode: Optional[str] = None,
        camera_width: int = 640,
        camera_height: int = 640,
        max_target_distance: float = 10.0,  # Max distance in meters to detect
        min_detection_conf: float = 0.5,
    ):
        """
        Args:
            drone: Existing Drone instance with MAVLink connection and camera
            max_steps: Maximum steps per episode
            max_yaw_rate: Maximum yaw rate in rad/s
            control_freq: Control loop frequency in Hz
            render_mode: 'human' for visualization or None
            camera_width: Camera frame width
            camera_height: Camera frame height
            max_target_distance: Max distance to track target (meters)
            min_detection_conf: Minimum YOLO confidence for human detection
        """
        super().__init__()

        self.drone_wrapper = DroneWrapper(drone, max_yaw_rate=max_yaw_rate)

        self.max_steps = max_steps
        self.control_dt = 1.0 / control_freq
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.max_target_distance = max_target_distance
        self.min_detection_conf = min_detection_conf

        # Action space: continuous yaw rate [-1.0, 1.0] (normalized)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space: [angular_error, yaw_rate]
        # Both normalized to [-1, 1]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        self.render_mode = render_mode

        # Episode state
        self.steps = 0
        self.lock_duration = 0.0
        self.lost_duration = 0.0
        self.last_obs = None

        # Target state
        self.target_position = np.array([5.0, 0.0, 0.0])  # Default: 5m ahead
        self.target_detected = False
        self.last_target_position = np.array([5.0, 0.0, 0.0])

        # YOLO model
        self.yolo = None

        # Initialize YOLO
        self._init_yolo()

    def _init_yolo(self):
        """Initialize YOLO model."""
        try:
            print("[*] Loading YOLOv8 model...")
            self.yolo = YOLO("yolov8n.pt")  # Load pretrained nano model
            print("[+] YOLO loaded")
        except Exception as e:
            print(f"[-] Failed to load YOLO: {e}")
            raise

    def _get_target_from_yolo(self) -> Tuple[bool, np.ndarray]:
        """
        Get target position from YOLO detection.

        Returns:
            (detected, position): Tuple of detection status and relative position [dx, dy, dz]
        """
        # Get frame from drone camera
        frame = self.drone_wrapper.drone.get_frame()

        if frame is None:
            # Return last known position if frame unavailable
            return self.target_detected, self.last_target_position

        # Convert grayscale to RGB if needed
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = frame

        # Run YOLO detection
        results = self.yolo(frame_rgb, classes=[0], verbose=False)  # class 0 = person

        best_bbox = None
        best_conf = 0.0

        # Find highest confidence detection
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf and conf >= self.min_detection_conf:
                        best_conf = conf
                        best_bbox = box.xyxy[0].cpu().numpy()

        if best_bbox is None:
            # No human detected
            return False, self.last_target_position

        # Calculate center of bounding box
        x1, y1, x2, y2 = best_bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Calculate relative position from image center
        # cx, cy range from 0 to camera_width/height
        dx_normalized = (cx - self.camera_width / 2) / (
            self.camera_width / 2
        )  # [-1, 1]
        dy_normalized = (cy - self.camera_height / 2) / (
            self.camera_height / 2
        )  # [-1, 1]

        # Estimate distance based on bounding box size
        # Larger bbox = closer target (rough approximation)
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        bbox_area_ratio = (bbox_height * bbox_width) / (
            self.camera_width * self.camera_height
        )

        # Estimate distance inversely proportional to bbox size
        # Assuming person height of ~1.7m, rough distance estimation
        distance = max(
            0.5, self.max_target_distance * (1.0 - min(bbox_area_ratio * 10, 0.9))
        )

        # Convert to 3D relative position
        # dx is forward distance, dy is lateral offset, dz is vertical offset
        dx = distance  # Approximate forward distance
        dy = dx * dx_normalized * np.tan(np.radians(30))  # FOV-dependent lateral offset
        dz = (
            -dx * dy_normalized * np.tan(np.radians(30))
        )  # FOV-dependent vertical offset (negative because positive y is down in images)

        target_pos = np.array([dx, dy, dz])
        self.last_target_position = target_pos

        return True, target_pos

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset episode."""
        super().reset(seed=seed)

        # Reset target detection state
        self.target_position = np.array([5.0, 0.0, 0.0])  # Default 5m ahead
        self.target_detected = False
        self.last_target_position = np.array([5.0, 0.0, 0.0])

        # Reset episode counters
        self.steps = 0
        self.lock_duration = 0.0
        self.lost_duration = 0.0

        # Wait for first detection
        print("[*] Waiting for human detection...")
        timeout = 10.0  # seconds
        start = time.time()
        while time.time() - start < timeout:
            detected, pos = self._get_target_from_yolo()
            if detected:
                self.target_detected = True
                self.target_position = pos
                self.last_target_position = pos
                print(f"[+] Human detected at position: {pos}")
                break
            time.sleep(0.1)
        else:
            print("[!] Timeout waiting for detection, using default position")

        # Get initial observation
        obs = self._get_observation()
        info = {"target_detected": self.target_detected}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one timestep."""
        # Scale action to actual yaw rate
        yaw_rate = action[0] * self.drone_wrapper.max_yaw_rate

        # Send command to drone
        self.drone_wrapper.set_yaw_rate(yaw_rate)

        # Update target from YOLO
        self.target_detected, self.target_position = self._get_target_from_yolo()

        # Wait for control loop timing
        time.sleep(self.control_dt)

        # Get observation
        obs = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(action[0])

        # Check termination conditions
        terminated = False
        truncated = False

        # Extract target info
        target_distance = np.linalg.norm(self.target_position)

        # Check if target is centered (aligned)
        # Target is centered if lateral/vertical offset is small relative to distance
        lateral_offset = abs(self.target_position[1])
        vertical_offset = abs(self.target_position[2])
        is_aligned = (
            lateral_offset < target_distance * 0.1
            and vertical_offset < target_distance * 0.1
        )
        is_locked = self.target_detected and is_aligned

        if is_locked:
            self.lock_duration += self.control_dt
            if self.lock_duration >= 5.0:
                reward += 50.0  # Success bonus
                terminated = True
        else:
            self.lock_duration = 0.0

        # Check if target is lost
        if not self.target_detected:
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
            "target_detected": self.target_detected,
            "target_distance": target_distance,
            "target_position": self.target_position.copy(),
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Compute current observation (2D: angular error + yaw rate)."""
        # Get drone state
        yaw_rate = self.drone_wrapper.get_yaw_rate()

        # Calculate angular error to target
        if self.target_detected:
            rel_pos = self.target_position
            # Angle = atan2(dy, dx) - this is the angle to target
            angle_to_target = np.arctan2(rel_pos[1], rel_pos[0])
            angular_error = angle_to_target
        else:
            # No target - max angular error
            angular_error = np.pi  # Worst case: 180 degrees

        # Normalize to [-1, 1]
        angular_error_norm = np.clip(angular_error / np.pi, -1.0, 1.0)
        yaw_rate_norm = yaw_rate / self.drone_wrapper.max_yaw_rate

        obs = np.array(
            [angular_error_norm, yaw_rate_norm],
            dtype=np.float32,
        )

        return obs

    def _calculate_reward(self, action: float) -> float:
        """Calculate step reward."""
        if not self.target_detected:
            # Penalty for losing target
            return -1.0

        # Calculate alignment based on how centered the target is
        target_distance = np.linalg.norm(self.target_position)
        if target_distance > 0:
            # Alignment = cosine of angle to target (1.0 = perfectly centered)
            # For small angles, this is approximately 1 - (offset_angle)^2/2
            lateral_ratio = abs(self.target_position[1]) / target_distance
            vertical_ratio = abs(self.target_position[2]) / target_distance
            offset_angle = np.sqrt(lateral_ratio**2 + vertical_ratio**2)
            alignment = max(0.0, 1.0 - offset_angle)
        else:
            alignment = 1.0

        # Alignment reward
        reward = alignment * 10.0

        # Energy penalty
        reward -= 0.1 * abs(action)

        # Bonus for maintaining lock
        if alignment > 0.95:
            reward += 2.0

        return reward

    def render(self):
        """Render environment (optional)."""
        pass

    def close(self):
        """Close environment."""
        pass
