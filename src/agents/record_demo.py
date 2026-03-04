#!/usr/bin/env python3
"""
Video recording script for drone tracking demonstration.
Records video from drone's camera showing NN tracking a moving target.

Usage:
    python src/agents/record_demo.py --model models/drone_tracking_final.zip --duration 60
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import time
import numpy as np
import cv2
from datetime import datetime
from sb3_contrib import RecurrentPPO

from src.environment.drone_tracking_env import DroneTrackingEnv
from src.utils.curriculum_manager import CurriculumManager

# Import existing Drone class
from drone import Drone


class VideoRecorder:
    """Records video from drone camera with tracking visualization."""

    def __init__(self, output_path, fps=20, resolution=(640, 480)):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.frames = []
        self.frame_count = 0

    def add_frame(self, frame, target_pos=None, alignment=None, reward=None):
        """Add frame with optional overlay info."""
        # Resize if needed
        if frame.shape[:2] != self.resolution[::-1]:
            frame = cv2.resize(frame, self.resolution)

        # Convert grayscale to RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Add overlay text
        if alignment is not None:
            cv2.putText(
                frame,
                f"Alignment: {alignment:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        if reward is not None:
            cv2.putText(
                frame,
                f"Reward: {reward:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        if target_pos is not None:
            # Draw target indicator on frame (if we know where it is)
            pass  # Would need camera calibration for exact pixel position

        self.frames.append(frame)
        self.frame_count += 1

    def save(self):
        """Save recorded video."""
        if not self.frames:
            print("No frames to save")
            return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.resolution)

        for frame in self.frames:
            writer.write(frame)

        writer.release()
        print(f"[+] Video saved: {self.output_path}")
        print(f"    Duration: {self.frame_count / self.fps:.1f}s")
        print(f"    Frames: {self.frame_count}")


def record_tracking_demo(model_path, duration=60, visualize=False):
    """Record demonstration of drone tracking moving target."""

    print("=" * 60)
    print("Drone Tracking Demo Recording")
    print("=" * 60)

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"tracking_demo_{timestamp}.mp4"
    recorder = VideoRecorder(video_path, fps=20)

    # Connect to drone
    print("\n[*] Connecting to drone...")
    drone = Drone()
    if not drone.connect():
        print("[-] Failed to connect to drone")
        return 1
    print("[+] Drone connected")

    # Connect camera
    print("[*] Connecting camera...")
    camera_connected = drone.connect_camera()
    if not camera_connected:
        print("[!] Warning: Camera not connected, will record without video")
    else:
        print("[+] Camera connected")

    # Setup environment
    print("[*] Setting up environment...")
    env = DroneTrackingEnv(drone=drone, render_mode="human" if visualize else None)

    # Advance curriculum to moving targets (Level 2)
    env.virtual_target.set_trajectory("linear")
    env.virtual_target.set_velocity(np.array([2.0, 0.0, 0.0]))  # 2 m/s forward
    print("[+] Target set to linear movement at 2 m/s")

    # Load model
    print(f"[*] Loading model: {model_path}")
    try:
        model = RecurrentPPO.load(model_path, env=env)
        print("[+] Model loaded")
    except Exception as e:
        print(f"[-] Failed to load model: {e}")
        drone.close()
        return 1

    # Recording loop
    print(f"\n[*] Recording for {duration} seconds...")
    print("    Press Ctrl+C to stop early")
    print()

    obs, _ = env.reset()
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    start_time = time.time()
    total_reward = 0
    frame_time = 1.0 / 20.0  # 20 FPS

    try:
        while time.time() - start_time < duration:
            loop_start = time.time()

            # Get action from model
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True,  # Use deterministic policy for demo
            )
            episode_start = np.zeros((1,), dtype=bool)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Get camera frame if available
            if camera_connected:
                frame = drone.get_frame()
                if frame is not None:
                    recorder.add_frame(
                        frame, alignment=info.get("alignment", 0), reward=reward
                    )

            # Check if episode ended
            if terminated or truncated:
                print(f"    Episode ended. Reward: {total_reward:.1f}")
                obs, _ = env.reset()
                lstm_states = None
                episode_start = np.ones((1,), dtype=bool)
                total_reward = 0

            # Maintain frame rate
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\n[!] Recording stopped by user")

    # Save video
    print("\n[*] Saving video...")
    recorder.save()

    # Cleanup
    print("[*] Landing drone...")
    drone.land()
    drone.disarm()
    drone.close()

    print("\n[+] Demo complete!")
    print(f"    Video: {video_path}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Record drone tracking demo")
    parser.add_argument(
        "--model",
        type=str,
        default="models/drone_tracking_final.zip",
        help="Path to trained model",
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Recording duration in seconds"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Show visualization window"
    )
    args = parser.parse_args()

    return record_tracking_demo(args.model, args.duration, args.visualize)


if __name__ == "__main__":
    sys.exit(main())
