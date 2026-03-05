#!/usr/bin/env python3
"""
PID Mimic inference using Webots pedestrian position directly.
No YOLO needed - gets exact human XYZ from Webots Supervisor.

Usage:
    python src/agents/inference_webots_pos.py --model models/pid_mimic.pt --duration 60
"""

import sys
import time
import socket
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from drone import Drone
from src.environment.drone_wrapper import DroneWrapper


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


class PedestrianPositionClient:
    """Connects to Webots supervisor and reads pedestrian position."""

    def __init__(self, host="localhost", port=9999):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = ""
        self.last_position = None

    def connect(self, timeout=10):
        """Connect to supervisor server."""
        print(f"[*] Connecting to Webots supervisor at {self.host}:{self.port}...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(2.0)
                self.sock.connect((self.host, self.port))
                self.sock.settimeout(0.1)
                print("[+] Connected to supervisor!")
                return True
            except Exception:
                time.sleep(0.5)
        print("[-] Failed to connect to supervisor")
        return False

    def get_position(self):
        """Get latest pedestrian position [x, y, z]."""
        if self.sock is None:
            return self.last_position

        try:
            data = self.sock.recv(4096).decode()
            self.buffer += data

            # Parse last complete JSON line
            lines = self.buffer.split("\n")
            self.buffer = lines[-1]  # Keep incomplete line

            for line in reversed(lines[:-1]):
                line = line.strip()
                if line:
                    parsed = json.loads(line)
                    self.last_position = np.array(
                        [parsed["x"], parsed["y"], parsed["z"]], dtype=np.float32
                    )
                    return self.last_position

        except socket.timeout:
            pass
        except Exception as e:
            print(f"[!] Supervisor connection error: {e}")
            self.sock = None

        return self.last_position

    def close(self):
        if self.sock:
            self.sock.close()


def get_observation(drone_wrapper, pedestrian_pos, max_yaw_rate=1.0):
    """Compute 2D observation [angular_error, yaw_rate] from world positions."""
    # Webots Iris spawn point: translation 0.084 1.442 0.795
    drone_pos = np.array([0.084, 1.442, drone_wrapper.drone.state.altitude])
    yaw = drone_wrapper.get_yaw()  # Current drone yaw in radians
    yaw_rate = drone_wrapper.get_yaw_rate()

    # Vector from drone to pedestrian
    dx = pedestrian_pos[0] - drone_pos[0]
    dy = pedestrian_pos[1] - drone_pos[1]

    # Angle to pedestrian in world frame
    target_angle = np.arctan2(dy, dx)

    # Angular error (target angle relative to drone yaw)
    angular_error = target_angle - yaw
    # Wrap to [-pi, pi]
    angular_error = (angular_error + np.pi) % (2 * np.pi) - np.pi

    # Normalize
    angular_error_norm = np.clip(angular_error / np.pi, -1.0, 1.0)
    yaw_rate_norm = np.clip(yaw_rate / max_yaw_rate, -1.0, 1.0)

    return np.array([angular_error_norm, yaw_rate_norm], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Webots position-based PID mimic inference"
    )
    parser.add_argument("--model", type=str, default="./models/pid_mimic.pt")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--supervisor-host", type=str, default="localhost")
    parser.add_argument("--supervisor-port", type=int, default=9999)
    args = parser.parse_args()

    print("=" * 60)
    print("Webots Position-Based PID Mimic Inference (No YOLO)")
    print("=" * 60)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Loading model: {args.model} on {device}")
    model = PolicyNetwork().to(device)
    model.load_state_dict(
        torch.load(args.model, map_location=device, weights_only=True)
    )
    model.eval()
    print("[+] Model loaded")

    # Connect to Webots supervisor
    supervisor = PedestrianPositionClient(args.supervisor_host, args.supervisor_port)
    if not supervisor.connect():
        print(
            "[-] Could not connect to supervisor. Is the supervisor controller running?"
        )
        return

    # Connect to drone
    print("\n[*] Connecting to drone...")
    drone = Drone()
    if not drone.connect():
        print("[-] Failed to connect to drone")
        return
    print("[+] Drone connected")

    drone_wrapper = DroneWrapper(drone, max_yaw_rate=1.0)

    # Connect camera (for video recording only)
    drone.connect_camera()

    # Arm, GUIDED mode and takeoff
    print("\n[*] Setting GUIDED mode...")
    drone.set_mode("GUIDED")
    time.sleep(1)
    print("[*] Arming...")
    drone.arm()
    time.sleep(2)
    print("[*] Taking off to 5m...")
    drone.takeoff(5.0)

    # Wait for altitude
    print("[*] Waiting for altitude...")
    for _ in range(30):
        alt = drone.state.altitude
        print(f"    Alt: {alt:.1f}m", end="\r")
        if alt > 4.0:
            break
        time.sleep(0.5)
    print(f"\n[+] Airborne")

    # Setup video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"~/Desktop/webots_pos_{timestamp}.mp4").expanduser()
    video_writer = None

    # Inference loop
    total_reward = 0
    frames = 0
    start_time = time.time()

    try:
        while time.time() - start_time < args.duration:
            elapsed = time.time() - start_time

            # Get pedestrian position from Webots
            ped_pos = supervisor.get_position()

            if ped_pos is None:
                print("[!] No pedestrian position available")
                time.sleep(0.02)
                continue

            # Compute observation
            obs = get_observation(drone_wrapper, ped_pos)

            # Model inference
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(obs_tensor).item()

            # Send yaw command
            drone_wrapper.set_yaw_rate(action * 1.0)  # max_yaw_rate = 1.0 rad/s

            # Calculate reward (negative angular error)
            # Angular error is obs[0], scaled back from [-1, 1] to [-pi, pi]
            angular_error_rad = obs[0] * np.pi

            # Align camera to pedestrian correctly
            # In Webots (iris), the camera points +X. If the drone's yaw aligns with target angle,
            # the pedestrian is centered in the image.

            reward = -abs(angular_error_rad)
            total_reward += reward

            # Record video
            frame = drone.get_frame()
            if frame is not None:
                # Setup video writer on first frame
                if video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(output_path), fourcc, 20.0, (w, h)
                    )

                frame_bgr = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)

                # Overlay
                cv2.putText(
                    frame_bgr,
                    f"Webots Pos Mode | t={elapsed:.1f}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    f"Error: {obs[0] * 180 / np.pi:+.1f}deg | Action: {action:+.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    f"Ped: ({ped_pos[0]:.1f}, {ped_pos[1]:.1f})",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                video_writer.write(frame_bgr)
                frames += 1

            # 50Hz control loop
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[!] Interrupted")

    finally:
        if video_writer:
            video_writer.release()
        supervisor.close()
        drone.land()
        drone.disarm()
        drone.close()

        elapsed = time.time() - start_time
        print(f"\n[+] Done!")
        print(f"  Video: {output_path}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Frames: {frames}")
        print(f"  Avg reward: {total_reward / max(1, frames):.3f}")


if __name__ == "__main__":
    main()
