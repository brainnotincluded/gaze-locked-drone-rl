#!/usr/bin/env python3
"""
Train NN from PID demonstrations using Behavior Cloning.
Also supports RL fine-tuning on collected data.
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO


class PIDEpisodeDataset(Dataset):
    """Dataset for PID demonstration episodes."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.episodes = []

        # Load all episodes
        for ep_file in self.data_dir.glob("*.pkl"):
            with open(ep_file, "rb") as f:
                episode = pickle.load(f)
                self.episodes.append(episode)

        # Flatten all timesteps
        self.observations = []
        self.actions = []

        for ep in self.episodes:
            for i in range(len(ep["actions"])):
                self.observations.append(ep["observations"][i])
                self.actions.append(ep["actions"][i])

        self.observations = np.array(self.observations)
        self.actions = np.array(self.actions)

        print(
            f"[+] Loaded {len(self)} demonstration timesteps from {len(self.episodes)} episodes"
        )

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return {
            "observation": torch.FloatTensor(self.observations[idx]),
            "action": torch.FloatTensor([self.actions[idx]]),
        }


def train_behavior_cloning(data_dir, output_model, epochs=50, batch_size=64, lr=3e-4):
    """
    Train neural network using behavior cloning from PID demonstrations.

    Args:
        data_dir: Directory containing .pkl episode files
        output_model: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
    """
    print("=" * 60)
    print("Behavior Cloning from PID Demonstrations")
    print("=" * 60)

    # Load dataset
    dataset = PIDEpisodeDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create a simple policy network
    # Input: 6-dim observation [yaw, yaw_rate, target_dx, target_dy, target_dz, in_fov]
    # Output: 1-dim action [yaw_rate]
    class PolicyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh(),  # Output in [-1, 1]
            )

        def forward(self, x):
            return self.network(x)

    model = PolicyNetwork()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"\n[*] Training for {epochs} epochs...")
    print(f"    Dataset size: {len(dataset)}")
    print(f"    Batch size: {batch_size}")
    print(f"    Learning rate: {lr}\n")

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            obs = batch["observation"]
            target_action = batch["action"]

            # Forward pass
            pred_action = model(obs)

            # Compute loss
            loss = criterion(pred_action, target_action)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

    # Save model
    output_path = Path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\n[+] Model saved: {output_path}")

    return model


def rl_finetune_from_pid(data_dir, base_model, output_model, timesteps=100000):
    """
    Fine-tune existing RL model using PID demonstrations as initial dataset.

    This uses the PID data to pre-train the policy, then continues with RL.
    """
    from src.environment.yolo_human_env import YOLOHumanEnv
    from drone import Drone

    print("=" * 60)
    print("RL Fine-tuning from PID Demonstrations")
    print("=" * 60)

    # First, do behavior cloning to initialize policy
    print("\n[*] Phase 1: Behavior Cloning...")
    bc_model = train_behavior_cloning(data_dir, "./models/bc_pretrained.pt", epochs=30)

    # Connect to environment for RL training
    print("\n[*] Phase 2: RL Fine-tuning...")
    print("[*] Connecting to drone...")

    drone = Drone()
    if not drone.connect():
        print("[-] Failed to connect")
        return

    if not drone.connect_camera():
        print("[-] Failed to connect camera")
        drone.close()
        return

    # Create environment
    env = YOLOHumanEnv(
        drone=drone,
        max_steps=1000,
        max_yaw_rate=1.0,
        control_freq=50.0,
    )
    env = DummyVecEnv([lambda: env])

    # Load base model or create new one
    if Path(base_model).exists():
        print(f"[*] Loading base model: {base_model}")
        model = RecurrentPPO.load(base_model, env=env)
    else:
        print("[*] Creating new RecurrentPPO model...")
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
        )

    # Optional: Initialize policy with BC weights
    # This would require custom code to map BC network to SB3 policy

    # Train
    print(f"[*] Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # Save
    model.save(output_model)
    print(f"[+] Model saved: {output_model}")

    # Cleanup
    drone.close()

    return model


def compare_pid_vs_nn(pid_data_dir, nn_model_path, num_episodes=5):
    """
    Compare PID controller performance vs trained NN.
    """
    from src.agents.inference_yolo import inference_yolo
    import pickle

    print("=" * 60)
    print("PID vs NN Performance Comparison")
    print("=" * 60)

    # Load PID stats
    pid_rewards = []
    for ep_file in Path(pid_data_dir).glob("*.pkl"):
        with open(ep_file, "rb") as f:
            ep = pickle.load(f)
            pid_rewards.append(ep["total_reward"])

    print(f"\nPID Performance (from {len(pid_rewards)} episodes):")
    print(f"  Mean reward: {np.mean(pid_rewards):.2f} ± {np.std(pid_rewards):.2f}")
    print(f"  Min/Max: {np.min(pid_rewards):.2f} / {np.max(pid_rewards):.2f}")

    # Run NN inference and collect stats
    print(f"\n[*] Running NN for {num_episodes} episodes...")
    nn_rewards = []

    for i in range(num_episodes):
        print(f"\n  Episode {i + 1}/{num_episodes}")
        # Run inference and capture reward
        # This would need modification to inference_yolo to return stats
        result = inference_yolo(nn_model_path, duration=60, visualize=False)
        if result:
            nn_rewards.append(result.get("total_reward", 0))

    print(f"\nNN Performance (from {len(nn_rewards)} episodes):")
    print(f"  Mean reward: {np.mean(nn_rewards):.2f} ± {np.std(nn_rewards):.2f}")
    print(f"  Min/Max: {np.min(nn_rewards):.2f} / {np.max(nn_rewards):.2f}")

    # Compare
    improvement = (
        (np.mean(nn_rewards) - np.mean(pid_rewards)) / abs(np.mean(pid_rewards)) * 100
    )
    print(f"\nComparison:")
    print(f"  NN vs PID: {improvement:+.1f}% improvement")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train NN from PID demonstrations")
    parser.add_argument(
        "--mode",
        choices=["bc", "rl", "compare"],
        default="bc",
        help="Training mode: behavior cloning, RL finetune, or compare",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/pid_demonstrations",
        help="Directory with PID episode data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/nn_from_pid.zip",
        help="Output model path",
    )
    parser.add_argument(
        "--base-model", type=str, default="", help="Base model for RL fine-tuning"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Epochs for behavior cloning"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100000, help="Timesteps for RL fine-tuning"
    )

    args = parser.parse_args()

    if args.mode == "bc":
        train_behavior_cloning(args.data_dir, args.output, epochs=args.epochs)
    elif args.mode == "rl":
        rl_finetune_from_pid(
            args.data_dir, args.base_model, args.output, args.timesteps
        )
    elif args.mode == "compare":
        compare_pid_vs_nn(args.data_dir, args.output)


if __name__ == "__main__":
    main()
