#!/usr/bin/env python3
"""
PID Controller Demo for Human Tracking
Simple proportional controller for yaw tracking.
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


class PIDController:
    """Simple PID controller for yaw control."""

    def __init__(self, Kp=0.5, Ki=0.0, Kd=0.1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0
        self.prev_time = time.time()

    def update(self, error):
        """Update PID controller with new error."""
        current_time = time.time()
        dt = current_time - self.prev_time

        if dt <= 0:
            dt = 0.02  # Default 50Hz

        # Proportional term
        P = self.Kp * error

        # Integral term
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)  # Anti-windup
        I = self.Ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / dt
        D = self.Kd * derivative

        # Total output
        output = P + I + D

        # Update state
        self.prev_error = error
        self.prev_time = current_time

        return np.clip(output, -1.0, 1.0)

    def reset(self):
        """Reset controller state."""
        self.integral = 0
        self.prev_error = 0
        self.prev_time = time.time()


def record_pid_demo(duration=60, output_dir="./"):
    """
    Record drone camera feed with PID human tracking.

    Args:
        duration: Recording duration in seconds
        output_dir: Directory to save video
    """
    # Load YOLO model
    print("[*] Loading YOLOv8 model...")
    yolo = YOLO("yolov8n.pt")
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

    # Initialize PID controller
    pid = PIDController(Kp=0.8, Ki=0.0, Kd=0.2)

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

    # Setup video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"pid_human_tracking_{timestamp}.mp4"

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
    errors = []

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
            results = yolo(frame_rgb, classes=[0], verbose=False)

            # Draw detections and control
            frame_annotated = frame_rgb.copy()
            target_bbox = None
            error = 0

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    # Find the largest person (closest)
                    best_box = None
                    best_area = 0

                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)

                        if area > best_area:
                            best_area = area
                            best_box = (x1, y1, x2, y2)

                    if best_box:
                        x1, y1, x2, y2 = best_box
                        conf = float(boxes[0].conf[0])

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

                        # Calculate target position (center)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # Draw center point
                        cv2.circle(frame_annotated, (cx, cy), 5, (0, 0, 255), -1)

                        # Calculate error from center (normalized -1 to 1)
                        error = (cx - width // 2) / (width / 2)
                        target_bbox = best_box
                        humans_detected += 1

            # PID control
            if target_bbox:
                # Update PID
                action = pid.update(error)

                # Send command to drone
                wrapper.set_yaw_rate(action)

                errors.append(abs(error))

                # Display info
                cv2.putText(
                    frame_annotated,
                    f"PID Error: {error:+.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame_annotated,
                    f"PID Action: {action:+.3f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame_annotated,
                    f"Kp={pid.Kp} Ki={pid.Ki} Kd={pid.Kd}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
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
                pid.reset()
                wrapper.set_yaw_rate(0.0)

            # Add timestamp and frame info
            elapsed = time.time() - start_time
            avg_error = np.mean(errors[-100:]) if errors else 0
            cv2.putText(
                frame_annotated,
                f"Time: {elapsed:.1f}s | Avg Error: {avg_error:.3f}",
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Create side-by-side view
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
        wrapper.set_yaw_rate(0.0)
        time.sleep(0.5)
        drone.close()

        elapsed = time.time() - start_time
        avg_error = np.mean(errors) if errors else 0

        print(f"\n{'=' * 60}")
        print(f"PID Tracking Complete!")
        print(f"  Video: {output_path}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Frames: {frames_captured}")
        print(f"  FPS: {frames_captured / elapsed:.1f}")
        print(f"  Humans detected: {humans_detected} times")
        print(f"  Average tracking error: {avg_error:.3f}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record PID human tracking demo")
    parser.add_argument(
        "--duration", type=int, default=60, help="Recording duration in seconds"
    )
    parser.add_argument("--output", type=str, default="./", help="Output directory")

    args = parser.parse_args()

    record_pid_demo(args.duration, args.output)
