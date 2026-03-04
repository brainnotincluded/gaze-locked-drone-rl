#!/usr/bin/env python3
import sys

# Read the file
with open("src/agents/train.py", "r") as f:
    lines = f.readlines()

# Find and modify lines
output = []
for i, line in enumerate(lines):
    # Add resume argument after visualize argument
    if 'parser.add_argument("--visualize"' in line:
        output.append(line)
        output.append(
            '    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")\n'
        )
        continue

    # Modify model creation to support resume
    if (
        "model = RecurrentPPO(" in line
        and i > 0
        and "from sb3_contrib import RecurrentPPO" not in lines[i - 1]
    ):
        # Replace model creation block
        output.append("        # Create or load PPO agent\n")
        output.append("        if args.resume:\n")
        output.append(
            '            print(f"[*] Resuming from checkpoint: {args.resume}")\n'
        )
        output.append("            model = RecurrentPPO.load(args.resume, env=env)\n")
        output.append('            print(f"[+] Resumed from {args.resume}")\n')
        output.append("        else:\n")
        # Indent the original model creation
        output.append("            model = RecurrentPPO(\n")
        continue

    # Fix indentation for model creation block
    if (
        "model = RecurrentPPO(" in line
        or "verbose=1," in line
        or "learning_rate=3e-4," in line
        or "n_steps=2048," in line
        or "batch_size=64," in line
        or "n_epochs=10," in line
        or "gamma=0.99," in line
        or "gae_lambda=0.95," in line
        or "clip_range=0.2," in line
        or "ent_coef=0.01," in line
        or "tensorboard_log=" in line
        or ")" in line
    ):
        if i > 105 and i < 125:  # Within the model creation block
            output.append("            " + line)
            continue

    output.append(line)

# Write back
with open("src/agents/train.py", "w") as f:
    f.writelines(output)

print("Patched train.py for resume support")
