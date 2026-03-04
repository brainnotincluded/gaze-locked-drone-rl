#!/usr/bin/env python3
"""
Record one training episode as video.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import cv2
from datetime import datetime
from drone import Drone
from src.environment.drone_tracking_env import DroneTrackingEnv

# Connect to drone
print("[*] Connecting to drone...")
drone = Drone(mavlink_host="127.0.0.1", mavlink_port=5762)

if not drone.connect():
    print("[-] Failed to connect")
    sys.exit(1)

print("[+] Connected")
drone.set_mode("GUIDED")
drone.takeoff(5.0)
drone.wait_for_altitude(5.0, tolerance=0.5)
print("[+] Hovering at 5m")

# Create environment
env = DroneTrackingEnv(drone=drone, max_steps=100, render_mode=None)
obs, info = env.reset(seed=42)

# Setup video writer
video_path = "/home/webmaster/gaze-locked-drone-rl/episode_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = None

frames = []
print("[*] Recording episode...")

try:
    for step in range(100):
        # Get frame from visualizer
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Create visualization frame
        img = np.zeros((600, 600, 3), dtype=np.uint8)

        # Draw grid
        for i in range(0, 600, 50):
            cv2.line(img, (i, 0), (i, 600), (30, 30, 30), 1)
            cv2.line(img, (0, i), (600, i), (30, 30, 30), 1)

        # Drone at center
        drone_pos = (300, 300)
        cv2.circle(img, drone_pos, 15, (0, 255, 0), -1)

        # Get target relative position
        target_dx = obs[2] * 100  # Denormalize
        target_dy = obs[3] * 100
        target_x = int(300 - target_dy * 2)  # Scale for display
        target_y = int(300 - target_dx * 2)

        in_fov = bool(obs[5])
        color = (0, 255, 0) if in_fov else (0, 0, 255)
        cv2.circle(img, (target_x, target_y), 12, color, -1)

        # Draw FOV cone
        yaw = obs[0] * np.pi
        fov_half = np.pi / 6  # 30 degrees
        fov_len = 150

        fov_left = (
            int(300 + fov_len * np.cos(-yaw + np.pi / 2 - fov_half)),
            int(300 + fov_len * np.sin(-yaw + np.pi / 2 - fov_half)),
        )
        fov_right = (
            int(300 + fov_len * np.cos(-yaw + np.pi / 2 + fov_half)),
            int(300 + fov_len * np.sin(-yaw + np.pi / 2 + fov_half)),
        )
        cv2.line(img, drone_pos, fov_left, (0, 100, 0), 2)
        cv2.line(img, drone_pos, fov_right, (0, 100, 0), 2)

        # Status text
        status = "LOCKED" if in_fov else "LOST"
        cv2.putText(
            img,
            f"Step: {step} | {status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            f"Reward: {reward:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        frames.append(img)

        if terminated or truncated:
            break

    print(f"[+] Recorded {len(frames)} frames")

    # Write video
    if frames:
        video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (600, 600))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()
        print(f"[+] Video saved to: {video_path}")

finally:
    drone.land()
    drone.disarm()
    drone.close()
    print("[+] Done")
