# MPBRS: Multi-Potential-Based Reward Shaping for Unified Reward Configuration in Humanoid Locomotion
<p align="center">
  <img src="figures/humanoidtype.png" alt="failure" width="1000">
</p>

# Installation
The training was built on top of IsaacLab and run on Ubuntu 22.04 on a computer equipped with an Intel i7-9700F CPU and an NVIDIA GeForce RTX 2060 GPU (6 GB). The training took approximately 50 minutes for each robot.

**1**. Install IsaabLab and create the virtual environment:
```bash
Following the IsaacLab installation instructions for Ubuntu 22.04: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html```
```

**2**. Install packages. Clone the package or download and unzip it:
```bash
source env_isaaclab/bin/activate
cd humanoid_locomotion && pip install -e .
cp trainhumanoidlocomotion.py playhumanoidlocomotion.py ~/IsaacLab/
cd ~/IsaacLab/
```

**3**. Running the packages (running and training was in the IsaabLab environment)
```bash
./isaaclab.sh -p trainhumanoidlocomotion.py --task Isaac-Humanoid-Locomotion --num_envs 2048 --headless
```
Note that nine humanoid robots are available for training: Fourier N1, Tienkung2, Unitree G1, Unitree H1, Booster T1, Fourier GR3V2, Kuavo S45, Hightorque HI, and Berkeley Humanoid Lite. To switch to a different robot, open env_cfg.py and change the robot name at line 2428.

**3**. Play the trained agent
```bash
./isaaclab.sh -p playhumanoidlocomotion.py   --task=Isaac-Humanoid-Locomotion   --num_envs 10   --checkpoint ~/IsaacLab/logs/rsl_rl/humanoid-locomotion/***/model_***.pt
*** should be replaced with: the training log folder name (the run directory), andthe saved model checkpoint file (e.g., model_1000.pt).
```

