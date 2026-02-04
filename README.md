# RL-Quadruped

**Reinforcement learning-based quadruped robot that learns to walk forward with slow, spider-like locomotion using PPO in PyBullet.**

## Features

- **PPO Training** with 12 parallel environments (Stable Baselines3)
- **Slow Hip Movement** - Emphasizes deliberate hip joint control
- **Custom Reward Function** - Balances forward velocity, stability, and smooth actuation
- **Vision System** - YOLOv8 object detection with 3D localization (cube, sphere, cylinder) for establishing cordinated for path planning.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Haring-Bot/RL-Quadruped.git
cd RL-Quadruped

# Create conda environment
conda create -n quadruped python=3.11
conda activate quadruped

# Install dependencies
pip install -r requirements.txt
```

### Train the Robot

```bash
python src/train.py
```

Models save automatically to `models/` with timestamps. After training, the policy visualizes automatically.

### Object Detection (Optional)

```bash
# Test vision system
cd Detect
python scripts/test_detection.py --scenes 5 --objects 3
```

## How It Works

### Locomotion (RL)
- **Target**: 0.4 m/s forward walking
- **Hip Control**: Extremely slow movement (α=0.99 blending, -2000x penalty for speed)
- **Reward**: Forward velocity (250x), hip slowness (500x), stability penalties
- **Episode**: Up to 1000 steps (~42s at 240Hz)

### Vision (YOLOv8)
- **Trained on**: 2000 synthetic images
- **Detects**: Cube, sphere, cylinder
- **Output**: 3D world coordinates for path planning
- **Accuracy**: F1 ≥80%, distance error <15cm

## Key Parameters

```python
# Locomotion
alpha = 0.99                    # Action blending (ultra-smooth)
action_range = 0.3              # ±0.3 rad joint movement
sim_steps = 24                  # Steps per action
hip_velocity_penalty = -2000.0  # Enforce slow hips
```

## Project Structure

```
RL-Quadruped/
├── src/
│   ├── train.py              # Main RL training
│   ├── simulation.py         # PyBullet wrapper
│   └── IK_solver.py          # Kinematics
├── Detect/                   # Vision system
│   ├── src/vision_system.py  # YOLOv8 detection
│   └── weights/best.pt       # Trained model
├── data/Full_robot_urdf/     # Robot model
└── models/                   # Saved policies
```

## Requirements

Core: PyBullet, Stable-Baselines3, Gymnasium, NumPy, PyTorch  
Vision: YOLOv8/Ultralytics, OpenCV  
Python: 3.11 recommended

## Testing

```bash
# Test locomotion policy
python src/testModel.py --model models/quadruped_ppo_TIMESTAMP.zip

# Test vision system
cd Detect
python scripts/test_detection.py --scenes 10 --objects 5
```

## Notes

- **Hip joints** (Z-axis rotation) are 1st in each leg - heavily constrained for slow movement
- **Thigh/calf** (2nd/3rd joints) work in parallel - allowed faster movement
- Results still lacking. Unusual morphological shape makes learning process less intuitive
- Training uses 12 CPU cores, ~800 FPS training throughput
- Vision system: Camera at [0, -1.8, 0.6] looking at workspace

---

*Developed for robotics research and education by JULIAN HARING; PIERRE GAUDET, THU HTOO ZAW, THIRI TOE TOE ZIN*
