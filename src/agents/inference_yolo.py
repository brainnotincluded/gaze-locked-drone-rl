#!/usr/bin/env python3
"""
Inference script for trained RL agent to track humans using YOLO.

Usage:
    python src/agents/inference_yolo.py --model models/drone_tracking_final.zip
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import cv2
import time
from datetime import datetime
from ultralytics import YOLO

from src.environment.yolo_human_env import YOLOHumanEnv
from drone import Drone


def inference_yolo(model_path, duration=60, visualize=True):
    """
    Run trained RL model inference on real humans detected by YOLO.

    Args:
        model_path: Path to trained RecurrentPPO model
        duration: Runtime duration in seconds
        visualize: Show live camera feed with overlays
    """
    print("=" * 60)
    print("RL Inference - Human Tracking with YOLO")
    print("=" * 60)

    # Load RL model
    print(f"\n[*] Loading RL model: {model_path}")
    from sb3_contrib import RecurrentPPO

    if not Path(model_path).exists():
        print(f"[-] Model not found: {model_path}")
        return

    # Connect to drone
    print("\n[*] Connecting to drone...")
    drone = Drone()

    if not drone.connect():
        print("[-] Failed to connect")
        return

    print("[+] Connected")

    # Connect camera
    print("\n[*] Connecting camera...")
    if not drone.connect_camera():
        print("[-] Failed to connect camera")
        drone.close()
        return
    print("[+] Camera connected")

    # Wait for first frame
    print("[*] Waiting for first frame...")
    frame = None
    for _ in range(50):
        frame = drone.get_frame()
        if frame is not None:
            print(f"[+] Got frame: {frame.shape}")
            break
        time.sleep(0.1)

    if frame is None:
        print("[-] No frames received")
        drone.close()
        return

    # Create environment (for model compatibility)
    print("\n[*] Initializing environment...")
    raw_env = YOLOHumanEnv(
        drone=drone,
        max_steps=100000,  # Long episodes for inference
        max_yaw_rate=1.0,
        control_freq=50.0,
        camera_width=640,
        camera_height=640,
        min_detection_conf=0.5,
    )

    # Load model (this will wrap the env)
    model = RecurrentPPO.load(model_path, env=raw_env)
    print("[+] Model loaded")

    # Get the wrapped env from model for later use
    env = model.get_env()

    # Initialize LSTM states
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    # ARM and TAKEOFF
    print("\n[*] Arming...")
    drone.arm()
    time.sleep(1)

    print("[*] Taking off to 5m...")
    drone.takeoff(5.0)

    # Wait for takeoff (skip altitude check due to SITL telemetry bug)
    print("    Altitude telemetry unreliable, continuing anyway...")
    time.sleep(5)  # Wait for takeoff sequence to complete
    print(f"[+] Takeoff command sent (alt: {drone.state.altitude:.1f}m)")

    # Wait for human detection with retries
    print("\n[*] Waiting for human detection...")
    detected = False
    for _ in range(5):
        obs = env.reset()
        # Use raw_env to check detection status (env is wrapped by model)
        if raw_env.target_detected:
            detected = True
            break
        print("    Retrying...")
        time.sleep(1)

    if not detected:
        print("[-] No human detected after retries!")
        print("[*] Landing...")
        drone.land()
        drone.close()
        return

    print(f"[+] Human detected at: {raw_env.target_position}")

    # Setup video recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = Path("~/Desktop").expanduser()
    video_dir.mkdir(parents=True, exist_ok=True)
    output_path = video_dir / f"inference_yolo_{timestamp}.mp4"

    height, width = frame.shape[:2]
    if len(frame.shape) == 2:
        height, width = frame.shape
    else:
        height, width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 20.0, (width * 2, height))

    print(f"\n[*] Starting inference for {duration}s")
    print(f"    Recording to: {output_path}")
    print("    Press Ctrl+C to stop\n")

    # Inference loop
    frames_captured = 0
    detections_count = 0
    total_reward = 0.0
    start_time = time.time()
    last_time = start_time

    try:
        while time.time() - start_time < duration:
            # Get observation from raw environment (wrapped env doesn't have _get_observation)
            obs = raw_env._get_observation()
            obs = obs.reshape(1, -1)  # Add batch dimension

            # Predict action using LSTM model
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,  # Use deterministic policy for inference
            )
            episode_starts = np.zeros((1,), dtype=bool)

            # Execute action in environment (action is 2D array from model)
            # VecEnv returns (obs, reward, done, info) not (obs, reward, terminated, truncated, info)
            obs, reward, done, info = env.step(action)
            # reward is a numpy array from VecEnv, flatten it to scalar
            total_reward += (
                float(reward[0]) if hasattr(reward, "__len__") else float(reward)
            )

            # Track stats (info is a list in VecEnv)
            if info and info[0].get("target_detected", False):
                detections_count += 1

            # Get frame for visualization
            frame = drone.get_frame()
            if frame is not None:
                # Convert to RGB for YOLO visualization
                if len(frame.shape) == 2:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    frame_rgb = frame

                # Run YOLO to get visualization using raw_env
                results = raw_env.yolo(frame_rgb, classes=[0], verbose=False)

                # Draw detections
                frame_annotated = frame_rgb.copy()
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])

                            cv2.rectangle(
                                frame_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2
                            )
                            cv2.putText(
                                frame_annotated,
                                f"Person {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )

                            # Draw center
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            cv2.circle(frame_annotated, (cx, cy), 5, (0, 0, 255), -1)

                # Add RL info overlay
                elapsed = time.time() - start_time
                action_value = action[0][0] if len(action.shape) > 1 else action[0]

                cv2.putText(
                    frame_annotated,
                    f"Time: {elapsed:.1f}s | Reward: {total_reward:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame_annotated,
                    f"Action: {action_value:+.2f} | Detected: {info[0].get('target_detected', False) if info else False}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame_annotated,
                    f"Target: {info[0].get('target_position', [0, 0, 0]) if info else [0, 0, 0]}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                # Create side-by-side view
                frame_original = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frame_annotated_bgr = cv2.cvtColor(frame_annotated, cv2.COLOR_RGB2BGR)
                combined = np.hstack([frame_original, frame_annotated_bgr])

                # Record
                writer.write(combined)
                frames_captured += 1

                # Show live if requested
                if visualize:
                    cv2.imshow("RL Inference", combined)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # Check for episode termination
            if done:
                print("\n[!] Episode ended (lost target or timeout)")
                print("[*] Reinitializing...")
                obs = env.reset()  # VecEnv reset returns only obs
                episode_starts = np.ones((1,), dtype=bool)
                lstm_states = None

                if not raw_env.target_detected:
                    print("[-] No human detected after reset")
                    break

            # Control timing (50Hz)
            current_time = time.time()
            sleep_time = 0.02 - (current_time - last_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()

    except KeyboardInterrupt:
        print("\n[!] Inference stopped by user")

    finally:
        # Cleanup
        writer.release()
        if visualize:
            cv2.destroyAllWindows()

        drone.hover()
        time.sleep(1)
        drone.land()
        time.sleep(2)
        drone.disarm()
        drone.close()

        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Inference complete!")
        print(f"  Video: {output_path}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Frames: {frames_captured}")
        print(f"  FPS: {frames_captured / elapsed:.1f}")
        print(f"  Detections: {detections_count}")
        print(f"  Total reward: {float(total_reward):.1f}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RL inference on humans detected by YOLO"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/drone_tracking_final.zip",
        help="Path to trained RL model",
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Inference duration in seconds"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable live visualization (still records video)",
    )

    args = parser.parse_args()

    inference_yolo(args.model, args.duration, visualize=not args.no_visualize)
