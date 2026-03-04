#!/usr/bin/env python3
"""
Training script for drone tracking RL agent.

Usage:
    python src/agents/train.py --steps 1000000 --save-dir ./models
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.environment.drone_tracking_env import DroneTrackingEnv
from src.utils.curriculum_manager import CurriculumManager
from src.utils.callbacks import MetricsCallback

# Import your existing Drone class
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from drone import Drone


def make_env(drone, curriculum_manager=None, render_mode=None):
    """Factory function for environment creation."""

    def _init():
        env = DroneTrackingEnv(drone=drone, max_steps=1000, render_mode=render_mode)

        # Attach curriculum manager if provided
        if curriculum_manager:
            env.curriculum_manager = curriculum_manager

        return env

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train drone tracking agent")
    parser.add_argument(
        "--steps", type=int, default=1_000_000, help="Total training steps"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./models", help="Directory to save models"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Drone Tracking RL Training")
    print("=" * 60)

    # Connect to drone
    print("\n[*] Connecting to drone...")
    drone = Drone()

    if not drone.connect():
        print("[-] Failed to connect to drone")
        print("    Make sure MAVProxy is running with output 5762")
        return 1

    print("[+] Drone connected")

    # Set to GUIDED mode for position hold
    print("[*] Setting GUIDED mode...")
    drone.set_mode("GUIDED")

    # Takeoff to hover position
    print("[*] Taking off to 5m...")
    drone.takeoff(5.0)
    drone.wait_for_altitude(5.0, tolerance=0.5)
    print("[+] Hovering at 5m")

    try:
        # Initialize curriculum manager
        curriculum = CurriculumManager(success_threshold=0.8, window_size=100)

        # Create environment
        render_mode = "human" if args.visualize else None
        env = make_env(drone, curriculum, render_mode)()
        env = DummyVecEnv([lambda: env])

        # Create PPO agent with LSTM (RecurrentPPO)
        print("\n[*] Initializing PPO agent...")
        from sb3_contrib import RecurrentPPO

        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="./tensorboard/",
        )

        print("[+] Agent initialized")

        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(save_dir),
            name_prefix="drone_tracking",
        )

        metrics_callback = MetricsCallback(verbose=1)

        print(f"\n[*] Starting training for {args.steps} steps...")
        print(f"    Checkpoints saved to: {save_dir}")
        print(f"    TensorBoard logs: ./tensorboard/")
        print()

        # Train
        model.learn(
            total_timesteps=args.steps,
            callback=[checkpoint_callback, metrics_callback],
            progress_bar=True,
        )

        # Save final model
        final_path = save_dir / "drone_tracking_final.zip"
        model.save(final_path)
        print(f"\n[+] Training complete! Model saved to: {final_path}")

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user")

    finally:
        # Land drone
        print("\n[*] Landing...")
        drone.land()
        drone.disarm()
        drone.close()
        print("[+] Done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
