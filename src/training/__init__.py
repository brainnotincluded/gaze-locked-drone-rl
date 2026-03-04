"""Training module for learning from demonstrations."""

from .train_from_pid import (
    PIDEpisodeDataset,
    train_behavior_cloning,
    rl_finetune_from_pid,
    compare_pid_vs_nn,
)

__all__ = [
    "PIDEpisodeDataset",
    "train_behavior_cloning",
    "rl_finetune_from_pid",
    "compare_pid_vs_nn",
]
