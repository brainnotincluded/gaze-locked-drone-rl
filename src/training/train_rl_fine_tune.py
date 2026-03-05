#!/usr/bin/env python3
"""
Phase 2: RL Fine-tuning from PID Mimic

Initialize PPO with BC weights, then train to exceed PID performance.
Uses Stable-Baselines3 with custom Gymnasium env.

Usage:
    python src/training/train_rl_fine_tune.py --bc-model ./models/pid_mimic.pt --steps 500000
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import directly from file to avoid __init__.py issues
import importlib.util

spec = importlib.util.spec_from_file_location(
    "train_pid_mimic_simple", str(Path(__file__).parent / "train_pid_mimic_simple.py")
)
train_pid_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_pid_module)
SimpleTrackingSim = train_pid_module.SimpleTrackingSim
PolicyNetwork = train_pid_module.PolicyNetwork


class RLTrackingEnv(gym.Env):
    """Gymnasium wrapper for SimpleTrackingSim with curriculum."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, curriculum_level=0):
        super().__init__()
        self.sim = SimpleTrackingSim()
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Curriculum learning
        self.curriculum_level = curriculum_level
        self.max_steps = 1000
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Curriculum: gradually increase target movement speed
        if self.curriculum_level == 0:
            # Static target
            self.sim.max_target_speed = 0.0
        elif self.curriculum_level == 1:
            # Slow walking
            self.sim.max_target_speed = 0.05
        else:
            # Fast walking
            self.sim.max_target_speed = 0.1

        obs = self.sim.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.sim.step(action[0])
        self.current_step += 1

        # Shaped reward: encourage low angular error
        angular_error = obs[0] * np.pi  # Denormalize

        # Main reward: negative absolute error
        reward = -abs(angular_error)

        # Bonus for very tight tracking (< 10 degrees)
        if abs(angular_error) < 0.17:  # ~10 degrees
            reward += 1.0

        # Penalty for spinning too fast
        yaw_rate = obs[1] * self.sim.max_yaw_rate
        reward -= 0.01 * (yaw_rate**2)

        truncated = self.current_step >= self.max_steps

        return obs, reward, done, truncated, info

    def set_curriculum_level(self, level):
        """Increase difficulty."""
        self.curriculum_level = level
        print(
            f"[Curriculum] Level {level}: max_target_speed={self.sim.max_target_speed}"
        )


def load_bc_weights(bc_path, ppo_model):
    """Transfer BC policy weights to PPO."""
    bc_policy = PolicyNetwork()
    bc_policy.load_state_dict(torch.load(bc_path, weights_only=True))

    # Map BC network layers to PPO policy network
    # BC: Linear(2,64) -> ReLU -> Linear(64,64) -> ReLU -> Linear(64,64) -> ReLU -> Linear(64,1) -> Tanh
    # PPO: feature extractor + action_net (mean) + value_net

    with torch.no_grad():
        # Map feature extractor (first 3 Linear layers + ReLU)
        ppo_model.policy.mlp_extractor.policy_net[0].weight.copy_(
            bc_policy.network[0].weight
        )
        ppo_model.policy.mlp_extractor.policy_net[0].bias.copy_(
            bc_policy.network[0].bias
        )
        ppo_model.policy.mlp_extractor.policy_net[2].weight.copy_(
            bc_policy.network[2].weight
        )
        ppo_model.policy.mlp_extractor.policy_net[2].bias.copy_(
            bc_policy.network[2].bias
        )

        print("[+] Transferred BC weights to PPO feature extractor")

        # Don't transfer the last layer - PPO uses Gaussian distribution
        # The last BC layer maps to deterministic action
        # PPO's action_net will learn the mean, and log_std is learned separately


def main():
    parser = argparse.ArgumentParser(description="RL Fine-tuning from PID Mimic")
    parser.add_argument(
        "--bc-model",
        type=str,
        default="./models/pid_mimic.pt",
        help="Path to BC model for initialization",
    )
    parser.add_argument(
        "--steps", type=int, default=500_000, help="Total training steps"
    )
    parser.add_argument(
        "--curriculum", action="store_true", help="Enable curriculum learning"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2: RL Fine-tuning from PID Mimic")
    print("=" * 60)

    # Create vectorized environment
    def make_env():
        return Monitor(RLTrackingEnv())

    env = DummyVecEnv([make_env])

    # Create eval environment
    eval_env = DummyVecEnv([make_env])

    print("\n[*] Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tensorboard_rl/",
    )

    # Load BC weights
    if Path(args.bc_model).exists():
        print(f"[*] Loading BC weights from {args.bc_model}")
        load_bc_weights(args.bc_model, model)
    else:
        print(f"[!] BC model not found: {args.bc_model}")
        print("    Training from scratch...")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/rl_best/",
        log_path="./models/rl_logs/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    print(f"\n[*] Training for {args.steps} steps...")
    print("    RL will optimize beyond PID baseline")
    print("    Using shaped rewards with tracking bonuses\n")

    # Training with optional curriculum
    if args.curriculum:
        print("[*] Using curriculum learning")

        # Level 0: Static targets
        env.envs[0].set_curriculum_level(0)
        print("\n[Curriculum Level 0] Static targets (50k steps)")
        model.learn(total_timesteps=50_000, callback=eval_callback, progress_bar=True)

        # Level 1: Slow moving
        env.envs[0].set_curriculum_level(1)
        print("\n[Curriculum Level 1] Slow moving (100k steps)")
        model.learn(total_timesteps=100_000, callback=eval_callback, progress_bar=True)

        # Level 2: Fast moving
        env.envs[0].set_curriculum_level(2)
        print("\n[Curriculum Level 2] Fast moving (350k steps)")
        model.learn(total_timesteps=350_000, callback=eval_callback, progress_bar=True)
    else:
        model.learn(
            total_timesteps=args.steps,
            callback=eval_callback,
            progress_bar=True,
        )

    # Save final model
    output_path = "./models/rl_finetuned.pt"
    model.save(output_path)
    print(f"\n[+] RL model saved: {output_path}")

    # Also save as .zip for SB3 compatibility
    model.save("./models/rl_finetuned.zip")
    print("[+] Saved as rl_finetuned.zip")

    # Test final performance
    print("\n[*] Testing final model...")
    mean_reward, std_reward = eval_callback._evaluate_policy(
        model.policy, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"[+] Final evaluation: {mean_reward:.1f} ± {std_reward:.1f}")


if __name__ == "__main__":
    main()
