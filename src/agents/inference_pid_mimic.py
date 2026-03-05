#!/usr/bin/env python3
"""
Inference script for simplified PID mimic model with YOLO human tracking.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import time
import torch
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.environment.yolo_human_env import YOLOHumanEnv
from drone import Drone


class SimplePolicy(torch.nn.Module):
    """Simple policy from pid_mimic training."""

    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/pid_mimic.pt")
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()

    print("=" * 60)
    print("PID Mimic Model Inference - Human Tracking")
    print("=" * 60)

    # Load model
    print(f"\n[*] Loading model: {args.model}")
    model = SimplePolicy()
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.eval()
    print("[+] Model loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"    Device: {device}")

    # Connect to drone
    print("\n[*] Connecting to drone...")
    drone = Drone()
    if not drone.connect():
        print("[-] Failed to connect")
        return
    print("[+] Connected")

    # Connect camera
    if not drone.connect_camera():
        print("[-] Failed to connect camera")
        drone.close()
        return
    print("[+] Camera connected")

    # Wait for first frame
    print("[*] Waiting for camera...")
    for _ in range(50):
        if drone.get_frame() is not None:
            break
        time.sleep(0.1)
    print("[+] Camera ready")

    # Set GUIDED mode and takeoff
    print("\n[*] Setting GUIDED mode...")
    drone.set_mode("GUIDED")
    print("[+] Mode set to GUIDED")

    print("[*] Arming...")
    drone.arm()
    time.sleep(2)
    print("[+] Armed")

    print("[*] Taking off to 5m...")
    drone.takeoff(5.0)

    # Wait for takeoff
    print("[*] Waiting for altitude...")
    timeout = 30
    start = time.time()
    while time.time() - start < timeout:
        alt = drone.state.altitude
        print(f"    Altitude: {alt:.1f}m", end="\r")
        if alt > 4.0:
            print(f"\n[+] Takeoff complete: {alt:.1f}m")
            break
        time.sleep(0.5)
    else:
        print(f"\n[!] Takeoff timeout. Current alt: {drone.state.altitude:.1f}m")
        print("    Continuing anyway...")

    # Create environment
    env = YOLOHumanEnv(
        drone=drone,
        max_steps=100000,
        max_yaw_rate=1.0,
        control_freq=50.0,
    )

    # Setup video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"~/Desktop/pid_mimic_{timestamp}.mp4").expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get first frame for video
    first_frame = None
    for _ in range(30):
        first_frame = drone.get_frame()
        if first_frame is not None:
            break
        time.sleep(0.1)

    if first_frame is None:
        print("[-] No frames available")
        drone.land()
        drone.close()
        return

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, 20.0, (width, height))
    print(f"[*] Recording to: {output_path}")

    # Reset and wait for detection
    print("\n[*] Waiting for human detection...")
    obs, info = env.reset()
    detected = False
    for _ in range(10):
        if env.target_detected:
            detected = True
            break
        time.sleep(0.5)

    if not detected:
        print("[-] No human detected!")
        drone.land()
        drone.close()
        return

    print("[+] Human detected, starting inference...")

    # Inference loop
    total_reward = 0
    frames = 0
    detections = 0
    lost_target_frames = 0
    max_lost_frames = 50  # Keep tracking last known position for ~1 second
    last_known_action = 0.0
    last_known_obs = obs.copy()
    start_time = time.time()

    try:
        while time.time() - start_time < args.duration:
            # Get frame
            frame = drone.get_frame()
            if frame is not None:
                frame_copy = frame.copy()

                # Ensure BGR
                if len(frame_copy.shape) == 2:
                    frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)
                elif frame_copy.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame_copy

                # Get action from model
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model(obs_tensor).item()

                # Store last known action when target is detected
                if env.target_detected:
                    last_known_action = action
                    last_known_obs = obs.copy()
                    lost_target_frames = 0
                    detections += 1
                else:
                    # Target lost - use last known action
                    lost_target_frames += 1
                    action = last_known_action
                    # Show warning on frame
                    cv2.putText(
                        frame_bgr,
                        f"TARGET LOST - Using last action",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                # Draw overlay
                cv2.putText(
                    frame_bgr,
                    f"PID Mimic | Reward: {total_reward:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    f"Action: {action:+.2f} | Lost: {lost_target_frames}/{max_lost_frames}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

                video_writer.write(frame_bgr)
                frames += 1

            # Step
            action_array = np.array([action], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action_array)
            done = terminated or truncated
            total_reward += float(reward)

            # Only reset if target lost for too long
            if done and lost_target_frames >= max_lost_frames:
                print(f"[!] Target lost for {max_lost_frames} frames, resetting...")
                obs, info = env.reset()
                lost_target_frames = 0
                if not env.target_detected:
                    print("[-] Target still not found after reset")
                    # Try once more
                    time.sleep(1)
                    obs, info = env.reset()
                    if not env.target_detected:
                        print("[-] Giving up")
                        break

    except KeyboardInterrupt:
        print("\n[!] Interrupted")

    finally:
        video_writer.release()
        drone.land()
        drone.disarm()
        drone.close()

        elapsed = time.time() - start_time
        print(f"\n[+] Inference complete!")
        print(f"    Video: {output_path}")
        print(f"    Duration: {elapsed:.1f}s")
        print(f"    Frames: {frames}")
        print(f"    Detections: {detections}")
        print(f"    Total reward: {total_reward:.1f}")
        print(f"    Avg FPS: {frames / elapsed:.1f}")


if __name__ == "__main__":
    main()
