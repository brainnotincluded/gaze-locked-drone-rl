#!/usr/bin/env python3
"""
PID Controller with Data Collection for Expert Demonstrations.
Records (observation, action, reward, done) tuples from PID control.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import time
import pickle
from datetime import datetime
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO
from drone import Drone
from src.environment.yolo_human_env import YOLOHumanEnv


class PIDController:
    """PID controller for yaw tracking."""

    def __init__(self, Kp=0.8, Ki=0.0, Kd=0.2):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0
        self.prev_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 0.02

        P = self.Kp * error
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)
        I = self.Ki * self.integral
        derivative = (error - self.prev_error) / dt
        D = self.Kd * derivative

        output = P + I + D
        self.prev_error = error
        self.prev_time = current_time

        return np.clip(output, -1.0, 1.0)

    def reset(self):
        self.integral = 0
        self.prev_error = 0
        self.prev_time = time.time()


class PIDDataCollector:
    """Collects expert demonstrations from PID controller."""

    def __init__(self, env, pid_controller, output_dir="./data/pid_demonstrations"):
        self.env = env
        self.pid = pid_controller
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes = []

    def collect_episode(self, max_steps=1000, save_video=True):
        """Collect one episode of PID control data."""
        obs, info = self.env.reset()
        if not self.env.target_detected:
            print("[-] No target detected, skipping episode")
            return None

        self.pid.reset()
        episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "target_positions": [],
            "timestamps": [],
        }

        frames = []
        step = 0

        while step < max_steps:
            # Get current target error from env
            target = (
                self.env.last_target_position
                if not self.env.target_detected
                else self.env.target_position
            )

            if target is not None:
                # Calculate error (target should be in front = dx=0)
                dx = target[0]  # Forward distance
                dy = target[1]  # Lateral distance

                # Normalize error for PID (-1 to 1 range based on camera FOV)
                error = dy / (abs(dx) + 0.1)  # Angle error approximation
                error = np.clip(error, -1.0, 1.0)

                # Get PID action
                action = self.pid.update(error)
            else:
                action = 0.0

            # Step environment (wrap scalar action in array)
            action_array = np.array([action], dtype=np.float32)
            next_obs, reward, terminated, truncated, info = self.env.step(action_array)
            done = terminated or truncated

            # Store data
            episode_data["observations"].append(
                obs.copy() if hasattr(obs, "copy") else obs
            )
            episode_data["actions"].append(float(action))
            episode_data["rewards"].append(float(reward))
            episode_data["dones"].append(bool(done))
            episode_data["target_positions"].append(
                target.copy() if target is not None else [0, 0, 0]
            )
            episode_data["timestamps"].append(time.time())

            # Capture frame for video
            if save_video:
                frame = self.env.drone_wrapper.drone.get_frame()
                if frame is not None:
                    frames.append(frame)

            obs = next_obs
            step += 1

            if done:
                break

        # Convert to numpy arrays
        episode_data["observations"] = np.array(
            episode_data["observations"], dtype=np.float32
        )
        episode_data["actions"] = np.array(episode_data["actions"], dtype=np.float32)
        episode_data["rewards"] = np.array(episode_data["rewards"], dtype=np.float32)
        episode_data["dones"] = np.array(episode_data["dones"], dtype=bool)
        episode_data["target_positions"] = np.array(
            episode_data["target_positions"], dtype=np.float32
        )

        # Calculate metrics
        episode_data["total_reward"] = np.sum(episode_data["rewards"])
        episode_data["length"] = len(episode_data["rewards"])
        episode_data["avg_reward"] = episode_data["total_reward"] / max(
            1, episode_data["length"]
        )

        # Save episode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_path = self.output_dir / f"episode_{timestamp}.pkl"
        with open(episode_path, "wb") as f:
            pickle.dump(episode_data, f)

        print(f"[+] Episode saved: {episode_path}")
        print(f"    Length: {episode_data['length']} steps")
        print(f"    Total reward: {episode_data['total_reward']:.1f}")
        print(f"    Avg reward: {episode_data['avg_reward']:.3f}")

        # Save video if requested
        if save_video and frames:
            video_path = self.output_dir / f"episode_{timestamp}.mp4"
            self._save_video(frames, video_path)
            print(f"    Video: {video_path}")

        return episode_data

    def _save_video(self, frames, output_path):
        """Save frames to video file."""
        if not frames:
            return

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, 20.0, (width, height))

        for frame in frames:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)

        out.release()

    def collect_multiple_episodes(self, num_episodes=10):
        """Collect multiple episodes of PID data."""
        print(f"[*] Collecting {num_episodes} episodes...")

        for i in range(num_episodes):
            print(f"\n=== Episode {i + 1}/{num_episodes} ===")
            episode = self.collect_episode()

            if episode is None:
                print("[-] Failed to collect episode, retrying...")
                time.sleep(2)
                continue

            self.episodes.append(episode)
            time.sleep(1)  # Brief pause between episodes

        print(f"\n[+] Collected {len(self.episodes)} episodes")
        return self.episodes


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect PID demonstration data")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument(
        "--output",
        type=str,
        default="./data/pid_demonstrations",
        help="Output directory",
    )
    args = parser.parse_args()

    # Connect to drone
    print("[*] Connecting to drone...")
    drone = Drone()
    if not drone.connect():
        print("[-] Failed to connect")
        return

    # Connect camera
    if not drone.connect_camera():
        print("[-] Failed to connect camera")
        drone.close()
        return

    # Wait for first frame
    print("[*] Waiting for camera...")
    for _ in range(50):
        if drone.get_frame() is not None:
            break
        time.sleep(0.1)

    # Create environment
    env = YOLOHumanEnv(
        drone=drone,
        max_steps=1000,
        max_yaw_rate=1.0,
        control_freq=50.0,
    )

    # Create PID controller
    pid = PIDController(Kp=0.8, Ki=0.0, Kd=0.2)

    # Create collector
    collector = PIDDataCollector(env, pid, output_dir=args.output)

    # Takeoff
    print("[*] Arming and taking off...")
    drone.arm()
    drone.takeoff(5.0)
    time.sleep(5)

    try:
        # Collect episodes
        collector.collect_multiple_episodes(args.episodes)

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")

    finally:
        # Land
        print("[*] Landing...")
        drone.land()
        drone.disarm()
        drone.close()
        print("[+] Done")


if __name__ == "__main__":
    main()
