#!/usr/bin/env python3
"""
Pure PID controller for Webots pedestrian tracking (NO neural network).
Uses direct position data from Webots supervisor.

Usage:
    python src/agents/inference_pid_pure.py --duration 60
"""

import argparse
import sys
import time
import socket
import json
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from drone import Drone


class PIDController:
    """High-performance PID for yaw tracking."""

    def __init__(self, Kp=2.0, Ki=0.0, Kd=1.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def update(self, error):
        """Update PID with error (radians)."""
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 0.02

        # Proportional
        P = self.Kp * error

        # Integral with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)
        I = self.Ki * self.integral

        # Derivative
        D = self.Kd * (error - self.prev_error) / dt

        # Combine
        output = P + I + D

        # Update state
        self.prev_error = error
        self.prev_time = current_time

        # Normalize to [-1, 1]
        return np.clip(output, -1.0, 1.0)

    def reset(self):
        """Reset PID state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()


def get_pedestrian_position(host="localhost", port=9999):
    """Connect to Webots supervisor and get pedestrian position."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        sock.connect((host, port))

        # Receive position data
        data = sock.recv(1024).decode().strip()
        sock.close()

        if data:
            pos = json.loads(data)
            return np.array([pos["x"], pos["y"], pos["z"]])
    except Exception as e:
        pass
    return None


def get_observation(drone, pedestrian_pos):
    """Compute angular error and yaw rate."""
    # Drone spawn point in Webots
    drone_pos = np.array([0.084, 1.442, drone.state.altitude])

    # Get drone yaw
    yaw = drone.state.attitude[2] if hasattr(drone.state, "attitude") else 0.0

    # Vector to pedestrian
    dx = pedestrian_pos[0] - drone_pos[0]
    dy = pedestrian_pos[1] - drone_pos[1]

    # Target angle
    target_angle = np.arctan2(dy, dx)

    # Angular error (normalized to [-pi, pi])
    angular_error = target_angle - yaw
    angular_error = (angular_error + np.pi) % (2 * np.pi) - np.pi

    return angular_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    args = parser.parse_args()

    print("=" * 60)
    print("PID Controller - Webots Pedestrian Tracking")
    print("=" * 60)

    # Connect to drone
    print("\n[*] Connecting to drone...")
    drone = Drone()
    if not drone.connect():
        print("[-] Failed to connect")
        return
    print("[+] Drone connected")

    # Connect camera
    print("[*] Connecting camera...")
    if not drone.connect_camera():
        print("[-] Failed to connect camera")
        drone.close()
        return
    print("[+] Camera connected")

    # Connect to supervisor
    print("\n[*] Connecting to Webots supervisor...")
    ped_pos = None
    for _ in range(10):
        ped_pos = get_pedestrian_position()
        if ped_pos is not None:
            print(f"[+] Got pedestrian position: {ped_pos}")
            break
        time.sleep(0.5)

    if ped_pos is None:
        print("[-] Failed to get pedestrian position")
        print("    Make sure Webots is running with supervisor on port 9999")
        drone.close()
        return

    # Setup video
    print("\n[*] Setting up video recording...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = Path(f"~/Desktop/pid_pure_{timestamp}.mp4").expanduser()
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Wait for first frame
    frame = None
    for _ in range(50):
        frame = drone.get_frame()
        if frame is not None:
            break
        time.sleep(0.1)

    if frame is None:
        print("[-] No camera frames")
        drone.close()
        return

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, (width, height))
    print(f"[+] Video: {video_path}")

    # Arm and takeoff
    print("\n[*] Arming and taking off...")
    drone.set_mode("GUIDED")
    drone.arm()
    time.sleep(2)
    drone.takeoff(5.0)

    # Wait for altitude (SITL telemetry unreliable, just wait)
    print("[*] Waiting for takeoff...")
    time.sleep(5)
    print("[+] Takeoff assumed complete")

    # Initialize PID
    pid = PIDController(Kp=2.0, Ki=0.0, Kd=1.0)

    # Inference loop
    print(f"\n[*] Starting PID tracking for {args.duration}s...")
    total_reward = 0
    frames = 0
    start_time = time.time()

    try:
        while time.time() - start_time < args.duration:
            # Get frame
            frame = drone.get_frame()
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)

                # Get pedestrian position
                ped_pos = get_pedestrian_position()

                if ped_pos is not None:
                    # Compute angular error
                    angular_error = get_observation(drone, ped_pos)

                    # PID control
                    action = pid.update(angular_error)

                    # Send yaw command (action is normalized [-1, 1], scale to degrees for rotate)
                    yaw_speed = action * 30.0  # 30 deg/s max
                    drone.rotate(yaw_speed)

                    # Calculate reward
                    reward = -abs(angular_error)
                    total_reward += reward

                    # Draw info
                    cv2.putText(
                        frame_bgr,
                        f"PID Pure | Reward: {total_reward:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame_bgr,
                        f"Error: {np.degrees(angular_error):.1f} deg",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame_bgr,
                        f"Action: {action:+.2f}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame_bgr,
                        "NO TARGET",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                    )

                writer.write(frame_bgr)
                frames += 1

            time.sleep(0.02)  # 50Hz

    except KeyboardInterrupt:
        print("\n[!] Interrupted")

    finally:
        writer.release()
        drone.land()
        drone.disarm()
        drone.close()

        elapsed = time.time() - start_time
        print(f"\n[+] Done!")
        print(f"  Video: {video_path}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Frames: {frames}")
        print(f"  Total reward: {total_reward:.1f}")
        print(f"  Avg reward/frame: {total_reward / max(1, frames):.3f}")


if __name__ == "__main__":
    main()
