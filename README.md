# 🤖 Dual Franka Arm PPO Training

Reinforcement learning project for dual Franka robot arms manipulation using PPO (Proximal Policy Optimization) in Isaac Gym.

## 🎯 Project Overview

Training dual Franka Emika Panda arms to perform coordinated manipulation tasks using:
- **Environment**: NVIDIA Isaac Gym
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Task**: Dual arm bottle manipulation/coordination

## 🛠️ Tech Stack

- **Simulator**: NVIDIA Isaac Gym
- **Framework**: IsaacGymEnvs
- **RL Algorithm**: PPO
- **Language**: Python 3.8+
- **Physics**: PhysX GPU-accelerated

## 📁 Project Structure
```
dual_franka_ppo/
├── tasks/
│   └── dual_franka_bottle.py      # Custom task implementation
├── configs/
│   ├── DualFrankaBottle.yaml      # Task configuration
│   └── DualFrankaBottlePPO.yaml   # Training hyperparameters
├── scripts/
│   └── train.py                    # Training script
├── assets/
│   └── urdf/                       # Robot models (if custom)
└── results/
    └── runs/                       # Training logs & checkpoints
```

## 🚀 Installation

### Prerequisites
- Ubuntu 20.04/22.04
- NVIDIA GPU (RTX series recommended)
- CUDA 11.3+
- Python 3.8+

### 1. Install Isaac Gym

Download from: https://developer.nvidia.com/isaac-gym
```bash
cd isaacgym/python
pip install -e .
```

### 2. Install IsaacGymEnvs
```bash
git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
cd IsaacGymEnvs
pip install -e .
```

### 3. Setup This Project
```bash
git clone https://github.com/YOUR_USERNAME/dual_franka_ppo.git
cd dual_franka_ppo

# Copy task file to IsaacGymEnvs
cp tasks/dual_franka_bottle.py ../IsaacGymEnvs/isaacgymenvs/tasks/

# Copy config files
cp configs/*.yaml ../IsaacGymEnvs/isaacgymenvs/cfg/task/
cp configs/*PPO.yaml ../IsaacGymEnvs/isaacgymenvs/cfg/train/
```

## 🎮 Usage

### Train from scratch
```bash
cd IsaacGymEnvs
python train.py task=DualFrankaBottle
```

### Train with custom config
```bash
python train.py task=DualFrankaBottle \
    train=DualFrankaBottlePPO \
    num_envs=2048 \
    headless=True
```

### Resume training
```bash
python train.py task=DualFrankaBottle \
    checkpoint=runs/DualFrankaBottle/nn/last_DualFrankaBottle_ep_XXX.pth
```

### Evaluate trained model
```bash
python train.py task=DualFrankaBottle \
    test=True \
    checkpoint=runs/DualFrankaBottle/nn/best_model.pth \
    num_envs=64
```

## 📊 Task Details

### Observation Space
- Joint positions (7 joints × 2 arms = 14)
- Joint velocities (14)
- End-effector positions (3 × 2 = 6)
- Object pose (7: position + quaternion)
- Goal pose (7)

**Total**: ~48 dimensions

### Action Space
- Continuous control: Joint torques for 14 joints
- Range: [-1, 1] (normalized)

### Reward Function
- Distance to goal: -||end_effector - goal||
- Success bonus: +10 when reaching goal
- Collision penalty: -1 per collision
- Action smoothness: -0.01 × ||action||²

## 🔧 Hyperparameters

Key training parameters:
- **Episodes**: 10,000
- **Timesteps per episode**: 500
- **Num environments**: 4096
- **Learning rate**: 3e-4
- **Batch size**: 65536
- **PPO epochs**: 5
- **Clip range**: 0.2

See `configs/DualFrankaBottlePPO.yaml` for full config.

## 📈 Results

### Training Progress
- Episode 1000: Avg reward = -15.2
- Episode 5000: Avg reward = -5.8
- Episode 10000: Avg reward = 8.3 ✅

(Update với kết quả thật của bạn)

### Performance Metrics
- Success rate: 85%
- Avg episode length: 234 steps
- Training time: ~6 hours (RTX 3090)

## 🎥 Demo

[Insert video/gif here]

## 🐛 Troubleshooting

**Issue: GPU out of memory**
```bash
# Reduce num_envs
python train.py task=DualFrankaBottle num_envs=1024
```

**Issue: Simulation unstable**
- Check physics timestep in config
- Reduce control frequency
- Check collision meshes

## 📚 References

- [Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## 🤝 Contributing

Feel free to open issues or PRs!

## 📝 License

MIT License

## 📧 Contact

**Author**: Thanh Bao  
**Email**: thanh-bao.tran@grenoble-inp.org  
**GitHub**: [@Tranthanhbao198](https://github.com/Tranthanhbao198)

---

⭐ Star this repo if you find it helpful!
