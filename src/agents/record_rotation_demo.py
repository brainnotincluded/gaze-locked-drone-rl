#!/usr/bin/env python3
"""
Record drone yaw rotation demonstration.
Shows intentional left/right rotation to prove visualization works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import cv2
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
print("[+] Hovering at 5m")

# Create environment
env = DroneTrackingEnv(drone=drone, max_steps=200, render_mode=None)
obs, info = env.reset(seed=42)

video_path = "/home/webmaster/gaze-locked-drone-rl/rotation_demo.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = None

print("[*] Recording yaw rotation demo...")
print("    Phase 1: Rotate LEFT")
print("    Phase 2: Rotate RIGHT")
print("    Phase 3: Track target")

frames = []

for phase, (start, end, action_val, label) in enumerate(
    [
        (0, 50, -0.8, "ROTATING LEFT"),
        (50, 100, 0.8, "ROTATING RIGHT"),
        (100, 150, None, "TRACKING (Random)"),
    ]
):
    for step in range(start, end):
        if action_val is not None:
            action = np.array([action_val])
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        # Create visualization with CLEAR rotation indicator
        img = np.zeros((600, 600, 3), dtype=np.uint8)

        # Draw grid
        for i in range(0, 600, 50):
            cv2.line(img, (i, 0), (i, 600), (30, 30, 30), 1)
            cv2.line(img, (0, i), (600, i), (30, 30, 30), 1)

        # Drone at center
        drone_pos = (300, 300)

        # Get actual yaw from observation (normalized -1 to 1, representing -pi to pi)
        yaw = obs[0] * np.pi

        # Draw drone body (arrow showing direction)
        arrow_len = 40
        arrow_end = (
            int(300 + arrow_len * np.cos(-yaw + np.pi / 2)),
            int(300 + arrow_len * np.sin(-yaw + np.pi / 2)),
        )
        cv2.circle(img, drone_pos, 20, (0, 255, 0), -1)
        cv2.circle(img, drone_pos, 20, (255, 255, 255), 3)
        cv2.arrowedLine(img, drone_pos, arrow_end, (255, 255, 0), 4, tipLength=0.3)

        # Draw heading text
        heading_deg = np.degrees(yaw) % 360
        cv2.putText(
            img,
            f"Heading: {heading_deg:.1f}°",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Draw FOV cone
        fov_half = np.pi / 6
        fov_len = 120

        fov_left = (
            int(300 + fov_len * np.cos(-yaw + np.pi / 2 - fov_half)),
            int(300 + fov_len * np.sin(-yaw + np.pi / 2 - fov_half)),
        )
        fov_right = (
            int(300 + fov_len * np.cos(-yaw + np.pi / 2 + fov_half)),
            int(300 + fov_len * np.sin(-yaw + np.pi / 2 + fov_half)),
        )
        # FOV cone fill
        triangle = np.array([drone_pos, fov_left, fov_right])
        cv2.fillPoly(img, [triangle], (0, 50, 0))
        cv2.line(img, drone_pos, fov_left, (0, 200, 0), 2)
        cv2.line(img, drone_pos, fov_right, (0, 200, 0), 2)

        # Target
        target_dx = obs[2] * 100
        target_dy = obs[3] * 100
        target_x = int(300 - target_dy * 2)
        target_y = int(300 - target_dx * 2)

        in_fov = bool(obs[5])
        color = (0, 255, 0) if in_fov else (0, 0, 255)
        cv2.circle(img, (target_x, target_y), 15, color, -1)
        cv2.circle(img, (target_x, target_y), 15, (255, 255, 255), 2)

        # Info text
        cv2.putText(
            img,
            f"Step: {step} | {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            f"Reward: {reward:.2f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Phase indicator
        cv2.putText(
            img,
            f"Phase {phase + 1}/3",
            (450, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        frames.append(img)

        if video_writer is None:
            h, w = img.shape[:2]
            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))

        video_writer.write(img)

video_writer.release()
print(f"[+] Saved {len(frames)} frames to: {video_path}")

# Land
drone.land()
drone.close()
print("[+] Done")
