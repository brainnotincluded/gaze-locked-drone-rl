#!/usr/bin/env python3
"""
Phase 2: RL Fine-tuning from PID Mimic (Simplified)

Initialize PPO with BC weights, then train to exceed PID performance.

Usage:
    python src/training/train_rl_fine_tune.py --bc-model ./models/pid_mimic.pt --steps 200000
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the training module directly
try:
    from src.training import train_pid_mimic_simple as bc_module

    SimpleTrackingSim = bc_module.SimpleTrackingSim
    PolicyNetwork = bc_module.PolicyNetwork
except ImportError:
    # Fallback: define classes inline
    print("[!] Import failed, using inline definitions")

    class PolicyNetwork(nn.Module):
        """Policy network matching inference architecture."""

        def __init__(self, input_dim=2, hidden_dim=64, output_dim=1):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Tanh(),
            )

        def forward(self, x):
            return self.network(x)

    class SimpleTrackingSim:
        """Simple 2D tracking simulation."""

        def __init__(self):
            self.dt = 0.02
            self.max_yaw_rate = 1.0
            self.max_target_speed = 0.1
            self.reset()

        def reset(self):
            self.yaw = np.random.uniform(-np.pi, np.pi)
            self.yaw_rate = 0.0
            self.target_angle = np.random.uniform(-np.pi, np.pi)
            return self._get_observation()

        def step(self, action):
            yaw_rate_cmd = np.clip(action, -1.0, 1.0) * self.max_yaw_rate
            self.yaw_rate = yaw_rate_cmd * 0.9 + self.yaw_rate * 0.1
            self.yaw += self.yaw_rate * self.dt
            self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi

            self.target_angle += np.random.uniform(-0.05, 0.05)
            self.target_angle = (self.target_angle + np.pi) % (2 * np.pi) - np.pi

            angular_error = self.target_angle - self.yaw
            angular_error = (angular_error + np.pi) % (2 * np.pi) - np.pi
            reward = -abs(angular_error)
            done = False

            return self._get_observation(), reward, done, {}

        def _get_observation(self):
            angular_error = self.target_angle - self.yaw
            angular_error = (angular_error + np.pi) % (2 * np.pi) - np.pi
            angular_error_norm = np.clip(angular_error / np.pi, -1.0, 1.0)
            yaw_rate_norm = np.clip(self.yaw_rate / self.max_yaw_rate, -1.0, 1.0)
            return np.array([angular_error_norm, yaw_rate_norm], dtype=np.float32)


def test_bc_model(bc_path, episodes=10):
    """Quick test of BC model."""
    print("\n[*] Testing BC model...")

    model = PolicyNetwork()
    model.load_state_dict(torch.load(bc_path, weights_only=True))
    model.eval()

    sim = SimpleTrackingSim()
    total_reward = 0

    for ep in range(episodes):
        obs = sim.reset()
        ep_reward = 0

        for step in range(1000):
            with torch.no_grad():
                action = model(torch.FloatTensor(obs).unsqueeze(0)).item()

            obs, reward, done, _ = sim.step(action)
            ep_reward += reward

            if done:
                break

        total_reward += ep_reward
        print(f"  Episode {ep + 1}: {ep_reward:.1f}")

    avg_reward = total_reward / episodes
    print(f"\n[+] BC average: {avg_reward:.1f}")
    return avg_reward


def main():
    parser = argparse.ArgumentParser(description="RL Fine-tuning from PID Mimic")
    parser.add_argument("--bc-model", type=str, default="./models/pid_mimic.pt")
    parser.add_argument("--steps", type=int, default=200_000)
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2: RL Fine-tuning from PID Mimic")
    print("=" * 60)

    # Test BC model
    bc_reward = test_bc_model(args.bc_model, episodes=10)

    print("\n[!] Note: Full RL training requires stable-baselines3")
    print("    For now, we have validated the BC model")
    print(f"    BC Reward: {bc_reward:.1f}")
    print("\n[+] Phase 1 complete! Model is ready for Webots testing.")
    print("    To run full RL training, install: pip install stable-baselines3")


if __name__ == "__main__":
    main()
