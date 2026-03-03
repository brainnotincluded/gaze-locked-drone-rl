# Drone Virtual Target RL

Reinforcement learning system for training drones to track virtual targets using yaw control only.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install sb3-contrib for RecurrentPPO
pip install sb3-contrib

# Run training
python src/agents/train.py --steps 1000000 --visualize
```

## Architecture

- **Virtual Target**: Pure Python object with configurable trajectories
- **Drone Wrapper**: Yaw-only control interface to existing Drone class
- **Gymnasium Env**: Standard RL interface with 50Hz control loop
- **Curriculum**: Progressive difficulty (static → linear → evasive)
- **RecurrentPPO**: LSTM-based policy for temporal memory

## Project Structure

```
src/
├── environment/
│   ├── virtual_target.py      # Virtual target with coordinate transforms
│   ├── drone_wrapper.py       # Yaw-only drone interface
│   └── drone_tracking_env.py  # Gymnasium environment
├── agents/
│   └── train.py               # Training script
└── utils/
    ├── curriculum_manager.py  # Progressive difficulty
    └── visualizer.py          # Optional debugging viz

tests/                         # Unit and integration tests
docs/plans/                    # Design documents
```

## Training

```bash
# Basic training
python src/agents/train.py --steps 1000000

# With visualization
python src/agents/train.py --steps 1000000 --visualize

# Custom save directory
python src/agents/train.py --steps 1000000 --save-dir ./my_models

# Resume from checkpoint
python src/agents/train.py --steps 1000000 --resume ./models/drone_tracking_100000_steps.zip
```

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/environment/test_virtual_target.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

## Requirements

- Python 3.8+
- Webots with ArduPilot SITL (running separately)
- MAVProxy configured on port 5762
- Existing Drone class with MAVLink connection