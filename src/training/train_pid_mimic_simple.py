#!/usr/bin/env python3
"""
Simplified PID mimic training - no Webots, pure Python simulation.
Trains NN to mimic PID controller in simple 2D angular tracking.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from datetime import datetime

np.random.seed(42)
torch.manual_seed(42)


class SimplePID:
    """Simple PID controller."""

    def __init__(self, Kp=0.8, Ki=0.0, Kd=0.2):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0
        self.prev_time = 0

    def update(self, error, dt=0.02):
        if dt <= 0:
            dt = 0.02

        P = self.Kp * error
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)
        I = self.Ki * self.integral
        D = self.Kd * (error - self.prev_error) / dt

        output = P + I + D
        self.prev_error = error

        return np.clip(output, -1.0, 1.0)


class SimpleTrackingSim:
    """Simple 2D angular tracking simulation."""

    def __init__(self):
        self.yaw = 0.0  # Drone yaw
        self.yaw_rate = 0.0  # Angular velocity
        self.target_angle = 0.0  # Target angle (where human is)
        self.max_yaw_rate = 1.0  # Max yaw rate rad/s

    def reset(self):
        """Reset to random state."""
        self.yaw = np.random.uniform(-np.pi, np.pi)
        self.yaw_rate = 0.0
        # Target starts at random angle
        self.target_angle = np.random.uniform(-np.pi, np.pi)
        return self._get_observation()

    def step(self, action):
        """Step simulation.

        action: yaw_rate command [-1, 1]
        """
        # Clamp action
        yaw_rate_cmd = np.clip(action, -1.0, 1.0) * self.max_yaw_rate

        # Update dynamics (simplified)
        self.yaw_rate = yaw_rate_cmd * 0.9 + self.yaw_rate * 0.1  # smoothing
        self.yaw += self.yaw_rate * 0.02

        # Keep yaw in [-pi, pi]
        while self.yaw > np.pi:
            self.yaw -= 2 * np.pi
        while self.yaw < -np.pi:
            self.yaw += 2 * np.pi

        # Move target randomly (human walking)
        self.target_angle += np.random.uniform(-0.05, 0.05)
        while self.target_angle > np.pi:
            self.target_angle -= 2 * np.pi
        while self.target_angle < -np.pi:
            self.target_angle += 2 * np.pi

        # Calculate reward
        angular_error = self.target_angle - self.yaw
        # Normalize error to [-pi, pi]
        while angular_error > np.pi:
            angular_error -= 2 * np.pi
        while angular_error < -np.pi:
            angular_error += 2 * np.pi

        reward = -abs(angular_error)  # Negative error = higher reward when aligned

        done = False

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """Get 2D observation: [angular_error, yaw_rate]."""
        angular_error = self.target_angle - self.yaw
        # Normalize to [-pi, pi]
        while angular_error > np.pi:
            angular_error -= 2 * np.pi
        while angular_error < -np.pi:
            angular_error += 2 * np.pi

        # Normalize to [-1, 1]
        angular_error_norm = angular_error / np.pi
        yaw_rate_norm = self.yaw_rate / self.max_yaw_rate

        return np.array([angular_error_norm, yaw_rate_norm], dtype=np.float32)


class PIDDataset(Dataset):
    """Dataset of PID demonstrations."""

    def __init__(self, observations, actions):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return {"observation": self.observations[idx], "action": self.actions[idx]}


class PolicyNetwork(nn.Module):
    """Deeper policy network."""

    def __init__(self, input_dim=2, hidden_dim=128, output_dim=1):
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


def collect_pid_data(num_episodes=100, episode_length=500):
    """Collect PID demonstration data."""
    print(f"[*] Collecting {num_episodes} episodes of PID data...")

    pid = SimplePID(Kp=0.8, Ki=0.0, Kd=0.2)
    sim = SimpleTrackingSim()

    observations = []
    actions = []

    for ep in range(num_episodes):
        obs = sim.reset()

        for step in range(episode_length):
            # Calculate angular error
            angular_error = obs[0] * np.pi  # Denormalize

            # PID action
            action = pid.update(angular_error, dt=0.02)

            # Store data
            observations.append(obs)
            actions.append([action])

            # Step
            obs, reward, done, info = sim.step(action)

        if (ep + 1) % 10 == 0:
            print(f"    Episode {ep + 1}/{num_episodes}")

    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)

    print(f"[+] Collected {len(observations)} samples")
    return observations, actions


def train(observations, actions, epochs=200, batch_size=128, lr=3e-4, use_gpu=True):
    """Train the policy network."""
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"[*] Training on {device}")

    # Create dataset and dataloader
    dataset = PIDDataset(observations, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = PolicyNetwork(input_dim=2, hidden_dim=64, output_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"[*] Training for {epochs} epochs...")
    print(f"    Dataset size: {len(dataset)}")
    print(f"    Batch size: {batch_size}")
    print(f"    Learning rate: {lr}\n")

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            obs = batch["observation"].to(device)
            target_action = batch["action"].to(device)

            # Forward
            pred_action = model(obs)

            # Loss
            loss = criterion(pred_action, target_action)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

    return model


def test(model, num_episodes=10, episode_length=500):
    """Test trained model."""
    device = next(model.parameters()).device
    sim = SimpleTrackingSim()

    total_rewards = []

    print(f"\n[*] Testing {num_episodes} episodes...")

    for ep in range(num_episodes):
        obs = sim.reset()
        episode_reward = 0

        for step in range(episode_length):
            # Model prediction
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(obs_tensor).item()

            obs, reward, done, info = sim.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        if (ep + 1) % 5 == 0:
            print(f"    Episode {ep + 1}/{num_episodes} - Reward: {episode_reward:.1f}")

    avg_reward = np.mean(total_rewards)
    print(f"\n[+] Average reward: {avg_reward:.1f}")
    return avg_reward


def test_pid(num_episodes=10, episode_length=500):
    """Test baseline PID controller."""
    pid = SimplePID(Kp=0.8, Ki=0.0, Kd=0.2)
    sim = SimpleTrackingSim()

    total_rewards = []

    print(f"[*] Testing PID baseline...")

    for ep in range(num_episodes):
        obs = sim.reset()
        episode_reward = 0

        for step in range(episode_length):
            angular_error = obs[0] * np.pi
            action = pid.update(angular_error, dt=0.02)

            obs, reward, done, info = sim.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"[+] PID average reward: {avg_reward:.1f}")
    return avg_reward


def main():
    parser = argparse.ArgumentParser(description="Simplified PID mimic training")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--episodes", type=int, default=100, help="Data collection episodes"
    )
    parser.add_argument(
        "--output", type=str, default="./models/pid_mimic.pt", help="Output model"
    )
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    args = parser.parse_args()

    print("=" * 60)
    print("Simplified PID Mimic Training")
    print("=" * 60)

    # Test baseline PID
    print("\n--- PID Baseline ---")
    pid_reward = test_pid(num_episodes=5)

    # Collect PID data
    print("\n--- Data Collection ---")
    observations, actions = collect_pid_data(num_episodes=args.episodes)

    # Train
    print("\n--- Training ---")
    use_gpu = not args.no_gpu
    model = train(
        observations,
        actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_gpu=use_gpu,
    )

    # Test trained model
    print("\n--- Testing Trained Model ---")
    nn_reward = test(model, num_episodes=5)

    # Compare
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"PID baseline:  {pid_reward:.1f}")
    print(f"NN (trained): {nn_reward:.1f}")
    print(f"Ratio:        {nn_reward / pid_reward * 100:.1f}%")

    # Save model
    import os

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(model.cpu().state_dict(), args.output)
    print(f"\n[+] Model saved: {args.output}")


if __name__ == "__main__":
    main()
