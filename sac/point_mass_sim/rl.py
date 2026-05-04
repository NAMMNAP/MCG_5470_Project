import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from env import MCG5740_Project_Env
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy

def create_env():
    # Parameters
    p = {
        "frame": "inertial",
        "max_hor_vel": 20.0,              # [m/s]
        "max_vert_vel": 10.0,             # [m/s]
        "max_hor_acc": 1 * 9.81,          # [m/s^2]
        "max_vert_acc_up": 0.8 * 9.81,    # [m/s^2]
        "max_vert_acc_down": 0.8 * 9.81,  # [m/s^2]
        "max_yaw_rate": np.deg2rad(150)   # [rad/s]
    }
    # Time variables
    dt = 0.01
    T = 120
    t_array = np.arange(0, T + dt, dt)
    # Reference path
    x = 50 * np.sin(t_array / 5)
    y = 20 * np.cos(t_array / 5)
    z = t_array / 2
    path = np.vstack((x, y, z)).T
    # Creating env
    return MCG5740_Project_Env(p, path, dt)

def moving_average(values, window=50):
    return np.convolve(values, np.ones(window)/window, mode='valid')

def plot_rewards(log_dir = "./logs/"):
    # Load results
    results = load_results(log_dir)

    # Extract timesteps and rewards
    x, y = ts2xy(results, 'timesteps')

    # Plot
    y_smooth = moving_average(y, window=50)

    plt.plot(x[len(x)-len(y_smooth):], y_smooth)
    plt.xlabel("Timesteps")
    plt.ylabel("Smoothed Reward")
    plt.title("Training Rewards (Smoothed)")
    plt.grid()
    plt.show()

def train():
    max_time_steps = 2000000
    # Create envrionment and invoke wrappers
    env = SubprocVecEnv([create_env for _ in range(8)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    # Create stable-baselines3 model
    model = SAC("MlpPolicy", env, device="cpu", verbose=1)
    # Begin training
    model.learn(total_timesteps=max_time_steps, log_interval=1, progress_bar=True)
    # Saving data
    model.save("sac_pm_new")
    env.save("sac_pm_vec_new.pkl")

    del model

def evaluate():
    # Create environment
    env = DummyVecEnv([create_env])
    # Load normalization
    env = VecNormalize.load("sac_pm_vec.pkl", env)
    env.training = False
    env.norm_reward = False

    # Load model
    model = SAC.load("sac_pm", env=env, device="cpu")

    # Perform evaluation
    #obs, _ = env.reset()
    obs = env.reset()
    t = 0
    T = len(env.get_attr("gamma")[0])
    while True:
        action, _ = model.predict(obs, deterministic=True)
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
    #plot_rewards()
    evaluate()

if __name__ == "__main__":
    main()