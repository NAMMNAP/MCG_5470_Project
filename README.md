# MCG5470 Project: Disturbance Rejection and Optimal Path Following in 3-Dimensions for Generalized UAS Through Adaptive PID Gains

# Demo
The video below shows a demo of the SAC model running in Project AirSim.
[![Watch the video](https://img.youtube.com/vi/40be-exhKdc/maxresdefault.jpg)](https://www.youtube.com/watch?v=40be-exhKdc)

# Project Structure
```text
.
├── sac/
│   ├── point_mass_sim/
│   │   └── __init.py__
│   │   └── Controller.py		      		# PID controller logic
│   │   └── step.py		      				# Point-mass UAS logic
│   │   └── env.py							# Gymnasium environment
│   │   └── rl.py							# SAC training script
│   │   └── sac_pm.zip						# SAC model
│   │   └── sac_pm_vec.pkl					# SAC model vectorization
│   └── project_airsim/
│   │   └── sim_config/						# Project AirSim environment metadata
│   │   └── __init.py__
│   │   └── pid.py		      				# Helper functions
│   │   └── pid_env.py						# Gymnasium environment
│   │   └── pid_rl.py						# SAC training script
│   │   └── sac_airsim.zip					# SAC model
│   │   └── sac_airsim_vec.pkl				# SAC model vectorization
├── q_learning/
│   └── TODO
└── README.md
└── requirements.txt
```
# Installation
Run `pip install -r requirements.txt` (recommended in a virtual environment or package manager).
For Project AirSim, [please see installation instructions.](https://github.com/iamaisim/ProjectAirSim)
# Usage
## SAC
Run `env.py` for baseline, run `rl.py` for SAC model. If Project AirSim is installed, this can be substitued with `pid_env.py` and `pid_rl.py`, respectively, in the `project_airsim/` folder.
## Q-Learning
TODO
