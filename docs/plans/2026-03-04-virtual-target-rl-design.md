# Virtual Target RL for Gaze-Locked Drone Tracking - Design Document

**Date**: 2026-03-04
**Approach**: Multi-threaded with Optional Visualization (Approach B)

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Simulation Backend                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Webots     │←──→│  ArduPilot   │←──→│   MAVProxy   │  │
│  │  SITL Drone  │    │     SITL     │    │  (Port 5762) │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↑
                              │ TCP (DroneKit/pymavlink)
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Python RL Environment                       │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Drone SDK   │    │   Virtual    │    │   Stable     │  │
│  │  (Existing)  │←──→│   Target     │←──→│  Baselines3  │  │
│  │              │    │   Object     │    │     PPO      │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                              │
│         ↓                   ↓                              │
│  ┌──────────────┐    ┌──────────────┐                      │
│  │  Telemetry   │    │   Visualizer │ (Optional thread)    │
│  │   (yaw,      │    │  (Top-down   │                      │
│  │  position)   │    │   view)      │                      │
│  └──────────────┘    └──────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

**Key Principle**: Episode boundaries are purely logical - the simulation never resets. Drone maintains position-hold mode while RL controls yaw only.

## 2. Core Components

### 2.1 Drone Interface Layer
- Wraps existing `Drone` class
- Methods: `get_yaw()`, `get_position()`, `set_yaw_rate(rate)`
- Position-hold mode active (drone maintains x,y,z)
- Thread-safe telemetry reading at 50Hz

### 2.2 Virtual Target Manager
```python
- reset_episode(): Randomize target position (r=5-50m, azimuth=0-360°, elevation=-30° to +30°)
- step(dt): Update target position based on trajectory (static → linear → evasive)
- is_in_fov(drone_yaw, fov=60°): Check if target visible
- get_relative_position(): Returns (dx, dy, dz) in camera frame
```

### 2.3 RL Environment (Gymnasium)
- **Observations**: `[normalized_yaw, normalized_yaw_rate, target_dx, target_dy, target_dz, in_fov_flag]` (6 dims)
- **Actions**: Continuous `yaw_rate ∈ [-1.0, 1.0]` rad/s (scaled to drone limits)
- **Reward**: `alignment * 10 - 0.1 * |action| - 100 * lost_target + 50 * locked`
- **Termination**: Target lost >3s OR locked >5s OR max 1000 steps

### 2.4 Curriculum Manager
- Track success rate over last 100 episodes
- Increase target speed/distance every N successes
- 3 curriculum levels: static → linear movement → evasive maneuvers

### 2.5 Optional Visualizer
- Top-down 2D view: drone at center, target position shown
- Color-coded: green (in FOV), red (lost)
- Update rate: 10Hz (decoupled from RL loop)

## 3. Data Flow & Episode Lifecycle

### Episode Reset (Soft Reset)
```python
def reset():
    # 1. Randomize virtual target position
    target.randomize(r=(5, 50), azimuth=(0, 360), elevation=(-30, 30))
    
    # 2. Reset LSTM hidden states (SB3 handles this internally)
    
    # 3. Reset episode counters
    steps = 0
    lock_duration = 0
    lost_duration = 0
    
    # 4. Return initial observation
    obs = get_observation()
    return obs
```

### Step Function (50Hz)
```python
def step(action):
    # 1. Apply yaw rate action to drone
    drone.set_yaw_rate(action * max_rate)
    
    # 2. Advance virtual target
    target.step(dt=0.02)
    
    # 3. Calculate observation
    obs = compute_observation()
    
    # 4. Calculate reward
    alignment = dot(camera_forward, target_direction)
    reward = alignment * 10.0 - 0.1 * abs(action)
    
    # 5. Check termination
    if not target.in_fov():
        lost_duration += 0.02
        if lost_duration > 3.0:
            reward -= 100.0
            terminated = True
    
    if target.centered() and alignment > 0.95:
        lock_duration += 0.02
        if lock_duration > 5.0:
            reward += 50.0
            terminated = True
    
    if steps >= 1000:
        truncated = True
    
    return obs, reward, terminated, truncated, info
```

## 4. Error Handling & Safety

### 4.1 Safety Wrapper
- Position drift check: If drone moves >10m from hover point → auto-land
- Communication timeout: If MAVLink loses heartbeat >2s → pause episode
- Action limits: Clamp yaw_rate to safe range before sending
- Emergency stop: Keyboard interrupt handler lands drone gracefully

### 4.2 Error Recovery
- Episode soft-reset doesn't affect simulation
- If drone loses position lock → re-center before next episode
- Failed MAVLink commands → retry 3x with backoff
- Visualizer disconnect → training continues

## 5. Testing Strategy

### 5.1 Unit Tests
- Virtual target position randomization
- Coordinate transform math (world → camera frame)
- FOV calculation (target at boundary cases)
- Reward function edge cases

### 5.2 Integration Tests
- Drone interface communication
- 50Hz control loop timing
- Episode reset doesn't reset simulation
- LSTM state reset between episodes

### 5.3 Training Verification
- Random policy baseline (should achieve ~0 reward)
- PPO learning curve (should improve over 1M steps)
- Curriculum progression (target speed increases)

## 6. Technical Specifications

- **Communication**: MAVLink via DroneKit (pymavlink)
- **RL Framework**: Stable Baselines3 with PPO
- **Action Space**: Continuous yaw_rate ∈ [-1.0, 1.0] rad/s
- **Control Rate**: 50Hz (20ms per step)
- **Memory**: LSTM hidden state (handled by SB3 RecurrentPPO)
- **Episode Duration**: Max 1000 steps (20 seconds)

## 7. Key Constraints

1. **Simulation never resets** - Webots/SITL runs continuously
2. **Position-hold mode** - Drone maintains (x,y,z), RL controls yaw only
3. **Virtual target** - Pure Python object, no physical camera
4. **Soft episode boundaries** - Logical resets only
