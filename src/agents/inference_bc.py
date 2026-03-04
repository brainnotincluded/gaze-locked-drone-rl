#!/usr/bin/env python3
"""
Inference for Behavior Cloning model (non-RL).
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import time
import torch
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environment.yolo_human_env import YOLOHumanEnv
from drone import Drone


class BCPolicy(torch.nn.Module):
    """BC policy network."""

    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(6, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


def inference_bc(model_path, duration=60):
    """Run BC model inference."""
    print("=" * 60)
    print("BC Model Inference - Human Tracking")
    print("=" * 60)

    # Load model
    print(f"\n[*] Loading BC model: {model_path}")
    model = BCPolicy()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("[+] Model loaded")

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

    # Create environment
    env = YOLOHumanEnv(
        drone=drone,
        max_steps=100000,
        max_yaw_rate=1.0,
        control_freq=50.0,
    )

    # Takeoff
    print("\n[*] Arming and taking off...")
    drone.arm()
    drone.takeoff(5.0)
    time.sleep(5)

    # Setup video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"~/Desktop/bc_inference_{timestamp}.mp4").expanduser()

    # Recording loop
    total_reward = 0
    frames_captured = 0
    detections = 0
    start_time = time.time()

    obs, info = env.reset()

    try:
        while time.time() - start_time < duration:
            # Get observation
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            # Predict action
            with torch.no_grad():
                action = model(obs_tensor).item()

            # Step environment (returns 5 values, wrap scalar action in array)
            action_array = np.array([action], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action_array)
            done = terminated or truncated
            total_reward += float(reward)

            if info.get("target_detected", False):
                detections += 1

            frames_captured += 1

            if done:
                print("[!] Episode ended, reinitializing...")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\n[!] Interrupted")

    finally:
        drone.land()
        drone.disarm()
        drone.close()

        elapsed = time.time() - start_time
        print(f"\n[+] Inference complete!")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Frames: {frames_captured}")
        print(f"  Detections: {detections}")
        print(f"  Total reward: {total_reward:.1f}")
        print(f"  Avg FPS: {frames_captured / elapsed:.1f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/bc_pretrained.pt")
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()

    inference_bc(args.model, args.duration)
