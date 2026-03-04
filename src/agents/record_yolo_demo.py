#!/usr/bin/env python3
"""
YOLO Human Tracking Demo
Records video from drone's onboard camera with YOLO human detection overlay.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import cv2
import time
from datetime import datetime
from ultralytics import YOLO

from src.environment.drone_wrapper import DroneWrapper
from drone import Drone


def record_yolo_demo(model_path=None, duration=60, output_dir="./"):
    """
    Record drone camera feed with YOLO human detection.

    Args:
        model_path: Path to trained RL model (optional)
        duration: Recording duration in seconds
        output_dir: Directory to save video
    """
    # Load YOLO model
    print("[*] Loading YOLOv8 model...")
    yolo = YOLO("yolov8n.pt")  # Load pretrained nano model
    print("[+] YOLO loaded")

    # Connect to drone
    print("\n[*] Connecting to drone...")
    drone = Drone()

    if not drone.connect():
        print("[-] Failed to connect")
        return

    print("[+] Connected")

    # Setup wrapper
    wrapper = DroneWrapper(drone, max_yaw_rate=1.0)

    # ARM and TAKEOFF
    print("\n[*] Arming...")
    drone.arm()
    time.sleep(1)

    print("[*] Taking off...")
    drone.takeoff(5.0)

    # Wait for takeoff with timeout
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

    # Load RL model if provided
    rl_model = None
    if model_path and Path(model_path).exists():
        print(f"\n[*] Loading RL model: {model_path}")
        from sb3_contrib import RecurrentPPO
        from src.environment.drone_tracking_env import DroneTrackingEnv

        # Create dummy env for model
        dummy_env = DroneTrackingEnv(drone=drone)
        rl_model = RecurrentPPO.load(model_path, env=dummy_env)
        print("[+] RL model loaded")

    # Connect camera
    print("\n[*] Connecting camera...")
    if not drone.connect_camera():
        print("[-] Failed to connect camera")
        drone.land()
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
        drone.land()
        return

    # Setup video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"yolo_human_tracking_{timestamp}.mp4"

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 20.0, (width * 2, height))

    print(f"\n[*] Recording to: {output_path}")
    print(f"    Duration: {duration}s | Resolution: {width}x{height}")
    print("    Press Ctrl+C to stop early\n")

    # Recording loop
    frames_captured = 0
    humans_detected = 0
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            # Get frame from drone camera
            frame = drone.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Convert grayscale to RGB if needed
            if len(frame.shape) == 2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame_rgb = frame

            # Run YOLO detection
            results = yolo(frame_rgb, classes=[0], verbose=False)  # class 0 = person

            # Draw detections
            frame_annotated = frame_rgb.copy()
            target_bbox = None

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])

                        # Draw bounding box
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

                        # Track first detected person
                        if target_bbox is None:
                            target_bbox = (x1, y1, x2, y2)
                            humans_detected += 1

            # Calculate target position (center of bbox)
            if target_bbox:
                cx = (target_bbox[0] + target_bbox[2]) // 2
                cy = (target_bbox[1] + target_bbox[3]) // 2

                # Draw center point
                cv2.circle(frame_annotated, (cx, cy), 5, (0, 0, 255), -1)

                # Calculate offset from center
                offset_x = cx - width // 2
                offset_y = cy - height // 2

                # Display info
                cv2.putText(
                    frame_annotated,
                    f"Target offset: ({offset_x:+d}, {offset_y:+d})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                # If RL model loaded, control drone
                if rl_model:
                    # Simple proportional control based on offset
                    # Normalize to [-1, 1]
                    yaw_action = -offset_x / (
                        width / 2
                    )  # Negative because yaw is opposite to offset
                    yaw_action = np.clip(yaw_action, -1.0, 1.0)

                    # Send command
                    wrapper.set_yaw_rate(yaw_action)

                    cv2.putText(
                        frame_annotated,
                        f"RL Action: {yaw_action:+.2f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )
            else:
                cv2.putText(
                    frame_annotated,
                    "No human detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

                if rl_model:
                    wrapper.set_yaw_rate(0.0)  # Stop if no target

            # Add timestamp and frame info
            elapsed = time.time() - start_time
            cv2.putText(
                frame_annotated,
                f"Time: {elapsed:.1f}s | Frames: {frames_captured}",
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Create side-by-side view
            # Left: original frame, Right: YOLO annotated
            frame_original = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_annotated_bgr = cv2.cvtColor(frame_annotated, cv2.COLOR_RGB2BGR)
            combined = np.hstack([frame_original, frame_annotated_bgr])

            # Write to video
            writer.write(combined)
            frames_captured += 1

            # Small delay for ~20 FPS
            time.sleep(0.04)

    except KeyboardInterrupt:
        print("\n[!] Recording stopped by user")

    finally:
        # Cleanup
        writer.release()
        drone.hover()
        time.sleep(1)
        drone.land()
        time.sleep(2)
        drone.disarm()
        drone.close()

        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Recording complete!")
        print(f"  Video: {output_path}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Frames: {frames_captured}")
        print(f"  FPS: {frames_captured / elapsed:.1f}")
        print(f"  Humans detected: {humans_detected} times")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record YOLO human tracking demo")
    parser.add_argument(
        "--model",
        type=str,
        default="models/drone_tracking_final.zip",
        help="Path to trained RL model",
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Recording duration in seconds"
    )
    parser.add_argument("--output", type=str, default="./", help="Output directory")

    args = parser.parse_args()

    record_yolo_demo(args.model, args.duration, args.output)
