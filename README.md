# MPBRS: Multi-Potential-Based Reward Shaping for Unified Reward Configuration in Humanoid Locomotion
<p align="center">
  <img src="figures/humanoidtype.png" alt="failure" width="1000">
</p>

# Installation
The training was built on top of IsaacLab and run on Ubuntu 22.04 on a computer equipped with an Intel i7-9700F CPU and an NVIDIA GeForce RTX 2060 GPU (6 GB). The training took approximately 50 minutes for each robot.
**1**. Create conda environment:
```bash
conda env remove -n twist
conda create -n twist python=3.8
conda activate twist
```

**2**. Install packages. Download from [official link](https://developer.nvidia.com/isaac-gym) and then install it:
```bash
cd humanoid_locomotion && pip install -e .
```

**3**. Install packages
```bash

cd legged_gym && pip install -e . && cd ..
pip install "numpy==1.23.0" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core imageio[ffmpeg] mujoco mujoco-python-viewer isaacgym-stubs pytorch-kinematics rich termcolor 
pip install redis[hiredis]
pip install pyttsx3 # for voice control
cd pose && pip install -e . && cd ..
```
