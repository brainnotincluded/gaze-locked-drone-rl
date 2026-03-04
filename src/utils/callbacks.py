from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class MetricsCallback(BaseCallback):
    """Custom callback for logging alignment and reward metrics."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_alignments = []
        self.current_episode_reward = 0
        self.current_episode_alignments = []

    def _on_step(self) -> bool:
        # Get info from environment
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [0])

        for i, info in enumerate(infos):
            if i < len(rewards):
                self.current_episode_reward += rewards[i]

                # Log alignment if available
                if "alignment" in info:
                    self.current_episode_alignments.append(info["alignment"])

                # Check if episode ended
                dones = self.locals.get("dones", [False])
                if i < len(dones) and dones[i]:
                    # Episode finished - log metrics
                    self.episode_rewards.append(self.current_episode_reward)

                    if self.current_episode_alignments:
                        avg_alignment = np.mean(self.current_episode_alignments)
                        self.episode_alignments.append(avg_alignment)

                        # Log to tensorboard
                        self.logger.record(
                            "rollout/ep_rew_mean", self.current_episode_reward
                        )
                        self.logger.record("rollout/alignment_mean", avg_alignment)
                        self.logger.record(
                            "rollout/alignment_max",
                            np.max(self.current_episode_alignments),
                        )
                        self.logger.record(
                            "rollout/alignment_min",
                            np.min(self.current_episode_alignments),
                        )

                        # Log step count
                        if "steps" in info:
                            self.logger.record("rollout/ep_len", info["steps"])

                    # Reset for next episode
                    self.current_episode_reward = 0
                    self.current_episode_alignments = []

        return True
