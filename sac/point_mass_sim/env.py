import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from step import step
from Controller import Controller


class MCG5740_Project_Env(gym.Env):
    def __init__(self, parameters, ref_path, dt):
        self.p = parameters         # Parameters
        self.gamma = ref_path       # Reference path [(x_1, y_1, z_1), ..., (x_N, y_N, z_N)]
        self.dt = dt
        self.render_mode = None

        # Hyperparameters
        self.action_scalar = 10
        self.max_err = 100
        self.max_u = 100
        self.lookahead = 5

        # Assign initial variables (in case reset() is not called)
        self.reset()

        # Observation space definition
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18+self.lookahead*3,),
            dtype=np.float64
        )

        # Action space definition
        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(self.gains),),
            dtype=np.float32
        )
       
    def reset(self, seed=42, options=None):
        # Reset time step counter to 0
        self.time_step = 0
        # Reset initial state
        self.state = {
            "pos": np.array([0.0, 0.0, 0.0]),
            "vel": np.array([0.0, 0.0, 0.0]),
            "vel_air": np.array([0.0, 0.0, 0.0]),
            "yaw": 0.0
        }
        # Reset gains
        self.gains = {
            "kp_pos": 2,
            "kd_pos": 1,
            "ki_pos": 0.01,
            "kp_vel": 2,
            "kd_vel": 1,
            "ki_vel": 0.01
        }
        self.original_gains = self.gains.copy()
        self.log_gains = np.array([np.log(self.gains["kp_pos"]), np.log(self.gains["kd_pos"]), np.log(self.gains["ki_pos"]), 
                                   np.log(self.gains["kp_vel"]), np.log(self.gains["kd_vel"]), np.log(self.gains["ki_vel"])])
        # Reset log
        N = len(self.gamma)
        self.log = {
            "t": np.arange(N) * self.dt,
            "pos": np.zeros((N, 3)),
            "vel": np.zeros((N, 3)),
            "acc": np.zeros((N, 3)),
            "yaw": np.zeros(N),
            "err": np.zeros((N, 3))
        }
        # Reseting error (e_t) and derivative/integral
        self.err = np.zeros(3)
        self.err_dot = np.zeros(3)
        self.err_int = np.zeros(3)
        # Resetting control action (u_t)
        self.u = np.zeros(3)
        # Defining time-varying wind function
        self.wind = lambda t: np.array([
            2*(5 + 2.0*np.sin(0.1*t)),
            1*(2 + 2.0*np.sin(0.1*t)),
            0
        ])

        # Getting observation
        obs = self.get_obs()
        info = {}
        return (obs, info)


    def step(self, action):
        # Transforming normalized [0, 1] action to proper values
        action = action * self.action_scalar
        action = action + 1e-6                    # Prevents log(0)
        # Adjust gains based on action
        self.gains = {
            "kp_pos": self.original_gains["kp_pos"] * action[0],
            "kd_pos": self.original_gains["kd_pos"] * action[1],
            "ki_pos": self.original_gains["ki_pos"] * action[2],
            "kp_vel": self.original_gains["kp_vel"] * action[3],
            "kd_vel": self.original_gains["kd_vel"] * action[4],
            "ki_vel": self.original_gains["ki_vel"] * action[5],
        }
        self.log_gains = np.array([np.log(self.gains["kp_pos"]), np.log(self.gains["kd_pos"]), np.log(self.gains["ki_pos"]), 
                                   np.log(self.gains["kp_vel"]), np.log(self.gains["kd_vel"]), np.log(self.gains["ki_vel"])])

        # Getting reference path position at the current time step
        cmd_pos = self.gamma[self.time_step, :]
        u, e, e_dot, e_int = Controller(cmd_pos, self.state, self.p, self.gains, self.dt)
        # Taking a step in the environment
        self.state, _ = step(self.state, u, self.p, self.dt, wind_vec=self.wind(self.time_step))

        # Logging
        self.log["pos"][self.time_step, :] = self.state["pos"]
        self.log["vel"][self.time_step, :] = self.state["vel"]
        self.log["yaw"][self.time_step]    = self.state["yaw"]
        self.log["err"][self.time_step, :] = e
        if self.time_step > 0:
            self.log["acc"][self.time_step, :] = (self.log["vel"][self.time_step, :] - self.log["vel"][self.time_step - 1, :]) / self.dt
        else:
            self.log["acc"][self.time_step, :] = np.array([0, 0, 0])

        # Reshaping e and u into vectors
        e_vec = e.reshape(-1, 1)
        u_vec = np.vstack((np.array(u["acc_hor_cmd"]).reshape(-1, 1), np.array(u["acc_z_cmd"]).reshape(-1, 1)))
        # Normalizing error and control
        e_vec = e_vec / self.max_err
        u_vec = u_vec / self.max_u
        # Q and R matrices
        Q = (1 / (self.max_err**2)) * np.eye(3)
        R = (1 / (self.max_u**2)) * np.eye(3)
        # Calculating reward term
        reward = -(0.5 * e_vec.T @ Q @ e_vec + 0.5 * u_vec.T @ R @ u_vec)
        # Convert to scalar
        reward = reward[0][0]

        # Updating internal variables
        self.err = e
        self.u = u_vec.reshape(-1)
        self.err_dot = e_dot
        self.err_int = e_int

        # Checking if done
        done = False
        # No more reference path to follow, we're done
        if self.time_step == len(self.gamma) - 1:
            #self.render()
            done = True
        
        # Getting observation
        obs = self.get_obs()
        # Advancing time step
        self.time_step = self.time_step + 1

        truncated = False
        info = {}
        return (obs, reward, done, truncated, info)
    

    def get_obs(self):
        # Observation vector:
        # Error at time t
        # Derivative of error at time t
        # Integral of error at time t
        # Control action
        # P gain (position)
        # I gain (position)
        # D gain (position)
        # P gain (velocity)
        # I gain (velocity)
        # D gain (velocity)

        # Array of future reference points (lookahead reference)
        lookahead_arr = []
        for k in range(self.lookahead):
            idx = min(self.time_step + k, len(self.gamma) - 1)
            # Relative position
            rel = self.gamma[idx] - self.state["pos"]
            lookahead_arr.append(rel)
        # Convert to numpy array
        lookahead_arr = np.asarray(lookahead_arr).flatten()

        return np.concatenate([self.err, self.err_dot, self.err_int, self.u, lookahead_arr, self.log_gains]).astype(np.float32)

  

    def render(self):
        # Plotting
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.gamma[:, 0], self.gamma[:, 1], self.gamma[:, 2], label="Reference Path")
        ax.plot(self.log["pos"][:, 0], self.log["pos"][:, 1], self.log["pos"][:, 2], label="Actual Path")
        ax.scatter(self.log["pos"][0, 0], self.log["pos"][0, 1], self.log["pos"][0, 2], c='g', label="Start")
        ax.scatter(self.log["pos"][-1, 0], self.log["pos"][-1, 1], self.log["pos"][-1, 2], c='r', label="End")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        ax.grid(True)

        # --- 2D Metrics ---
        _, axs = plt.subplots(3, 1, figsize=(9, 7))

        axs[0].plot(self.log["t"], self.log["pos"][:, 2])
        axs[0].set_ylabel("Altitude (m)")


        axs[1].plot(self.log["t"], np.linalg.norm(self.log["vel"][:, :2], axis=1))
        axs[1].axhline(self.p["max_hor_vel"], linestyle='--', color='r')
        axs[1].set_ylabel("Horizontal Velocity (m/s)")
        axs[1].set_title("Ground Speed")


        axs[2].plot(self.log["pos"][:, 0], self.log["pos"][:, 1])
        axs[2].set_xlabel("X (m)")
        axs[2].set_ylabel("Y (m)")
        axs[2].set_title("XY Track")

        # --- Velocity & Acceleration ---
        _, axs = plt.subplots(2, 2, figsize=(9, 7))

        # Vertical velocity
        axs[0, 0].plot(self.log["t"], self.log["vel"][:, 2])
        axs[0, 0].axhline(self.p["max_vert_vel"], linestyle='--', color='r')
        axs[0, 0].axhline(-self.p["max_vert_vel"], linestyle='--', color='r')
        axs[0, 0].set_title("Vertical Velocity")

        # Horizontal velocity
        axs[0, 1].plot(self.log["t"], np.linalg.norm(self.log["vel"][:, :2], axis=1))
        axs[0, 1].axhline(self.p["max_hor_vel"], linestyle='--', color='r')
        axs[0, 1].set_title("Horizontal Velocity")

        # Vertical acceleration
        axs[1, 0].plot(self.log["t"], self.log["acc"][:, 2])
        axs[1, 0].axhline(self.p["max_vert_acc_up"], linestyle='--', color='r')
        axs[1, 0].axhline(-self.p["max_vert_acc_down"], linestyle='--', color='r')
        axs[1, 0].set_title("Vertical Acceleration")

        # Horizontal acceleration
        axs[1, 1].plot(self.log["t"], np.linalg.norm(self.log["acc"][:, :2], axis=1))
        axs[1, 1].axhline(self.p["max_hor_acc"], linestyle='--', color='r')
        axs[1, 1].set_title("Horizontal Acceleration")

        # --- Error plot ---
        plt.figure()
        plt.plot(self.log["err"])
        plt.legend(["x", "y", "z"])
        plt.title("Tracking Error")
        plt.show()


def main():
    # Testing environment (no RL)
    # Parameters
    p = {
        "frame": "inertial",
        "max_hor_vel": 20.0,              # [m/s]
        "max_vert_vel": 10.0,             # [m/s]
        "max_hor_acc": 1 * 9.81,          # [m/s^2]
        "max_vert_acc_up": 0.8 * 9.81,    # [m/s^2]
        "max_vert_acc_down": 0.8 * 9.81,  # [m/s^2]
        "max_yaw_rate": np.deg2rad(150),  # [rad/s]
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
    env = MCG5740_Project_Env(p, path, dt)

    # Executing env
    env.reset()
    while True:
        #action = np.array([np.log(2)/5, np.log(1)/5, np.log(0.01)/5,
        #                      np.log(2)/5, np.log(1)/5, np.log(0.01)/5])
        #action = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        action = np.ones(6) * (1/10)
        obs, reward, terminated, truncated, _ = env.step(action)
        print("#############################")
        print("Error:", env.err)
        print("Reward:", reward)
        print("Control:", env.u)
        print("Observation:", obs)

        if terminated or truncated:
            break
    # Results
    env.render()


if __name__ == "__main__":
    main()