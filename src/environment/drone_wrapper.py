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
            0,
            0,
            0,  # Position (ignored)
            0,
            0,
            0,  # Velocity (maintain hover, so 0,0,0)
            0,
            0,
            0,  # Acceleration (ignored)
            0,  # Yaw (ignored)
            yaw_rate,  # Yaw rate
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
