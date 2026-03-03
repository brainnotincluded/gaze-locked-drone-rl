# Usage Guide

## Prerequisites

1. **Start Webots simulation** with ArduPilot SITL
2. **Start MAVProxy** with output on port 5762:
   ```bash
   mavproxy.py --master tcp:127.0.0.1:5760 --out 127.0.0.1:5762
   ```
3. **Verify connection**:
   ```python
   from drone import Drone
   drone = Drone()
   drone.connect()
   ```

## Training Workflow

### 1. Initial Training

Start with default curriculum:

```bash
python src/agents/train.py --steps 500000
```

This will:
- Connect to drone via MAVLink
- Takeoff to 5m hover
- Train PPO with LSTM for 500k steps
- Save checkpoints every 10k steps
- Use curriculum learning (starts with static targets)

### 2. Monitor Progress

```bash
# In another terminal
tensorboard --logdir ./tensorboard/
```

Watch for:
- `episode_reward_mean`: Should increase over time
- `success_rate`: Should approach 0.8 (curriculum threshold)
- `curriculum_level`: Should advance from 0 → 1 → 2

### 3. Curriculum Stages

- **Level 0** (Static): Target doesn't move, distance 5-30m
- **Level 1** (Linear): Target moves 1.5 m/s, distance 10-40m
- **Level 2** (Evasive): Sinusoidal motion, 2.5 m/s, distance 15-50m

Each level requires 80% success rate over 100 episodes to advance.

### 4. Evaluation

Test trained model:

```python
from stable_baselines3 import PPO
from src.environment.drone_tracking_env import DroneTrackingEnv
from drone import Drone

drone = Drone()
drone.connect()
drone.set_mode('GUIDED')
drone.takeoff(5.0)

env = DroneTrackingEnv(drone=drone, render_mode='human')
model = PPO.load('models/drone_tracking_final.zip')

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

drone.land()
```

## Troubleshooting

### Connection Issues

**Problem**: `target_system = 0` error
**Solution**: Ensure MAVProxy has output configured:
```bash
# In MAVProxy console
output add 127.0.0.1:5762
```

### Drone Doesn't Hover

**Problem**: Drone drifts during training
**Solution**: Ensure GUIDED mode is active before training starts

### Low Reward

**Problem**: Agent not learning
**Check**:
1. Verify target is visible in initial observation
2. Check alignment values in TensorBoard
3. Try increasing exploration (ent_coef=0.02)

## Configuration

Edit hyperparameters in `src/agents/train.py`:

```python
model = RecurrentPPO(
    'MlpLstmPolicy',
    env,
    learning_rate=3e-4,      # Try 1e-4 if unstable
    n_steps=2048,            # Batch size
    batch_size=64,
    n_epochs=10,
    gamma=0.99,              # Discount factor
    ent_coef=0.01,           # Exploration
    # ...
)
```