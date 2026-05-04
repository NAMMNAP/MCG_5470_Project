
import gymnasium as gym
import numpy as np
import os
from pid_env import MCG5740_Project_Env_AirSim
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

def create_env(log_dir = "./logs/"):
    # Time variables
    dt = 0.01
    T = 60
    t_array = np.arange(0, T + dt, dt)
    # Reference path
    x = (50 * np.sin(t_array / 5)) + 300
    y = 20 * np.cos(t_array / 5)
    z = (t_array * 10) + 4.0
    path = np.vstack((x, y, z)).T
    # Creating env
    env = MCG5740_Project_Env_AirSim(path, dt)
    return Monitor(env, log_dir)

def train():
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    max_time_steps = 600000
    # Create envrionment and invoke wrappers
    env = DummyVecEnv([create_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    # Create stable-baselines3 model
    model = SAC("MlpPolicy", env, verbose=1) # device="cpu"
    # Begin training
    model.learn(total_timesteps=max_time_steps, log_interval=1, progress_bar=True)
    # Saving data
    model.save("sac_airsim_new")
    env.save("sac_airsim_vec_new.pkl")

    del model

def evaluate():
    # Create environment
    env = DummyVecEnv([create_env])
    # Load normalization
    env = VecNormalize.load("sac_airsim_vec.pkl", env)
    env.training = False
    env.norm_reward = False

    # Load model
    model = SAC.load("sac_airsim", env=env) # device="cpu"

    # Perform evaluation
    #obs, _ = env.reset()
    obs = env.reset()
    t = 0
    T = len(env.get_attr("gamma")[0])
    while True:
        action, _states = model.predict(obs, deterministic=True)
        #obs, reward, terminated, truncated, _ = env.step(action)       # TODO: Uncomment this if using different version of Gymnasium
        obs, reward, terminated, _ = env.step(action)
        print("#############################")
        print("Action:", action)
        print("Error:", env.get_attr("err")[0])
        print("Reward:", reward)
        print("Control:", env.get_attr("u")[0])
        print("Observation:", obs)
        if terminated or t >= T-2:
            # TODO: Error within plot, starting and ending positions become the same.
            env.env_method("render", indices=[0])
            break
        t = t + 1

def main():
    #train()
    evaluate()

if __name__ == "__main__":
    main()

