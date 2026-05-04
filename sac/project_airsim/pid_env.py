import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from pid import *


class MCG5740_Project_Env_AirSim(gym.Env):
    def __init__(self, ref_path, dt):
        self.gamma = ref_path       # Reference path [(x_1, y_1, z_1), ..., (x_N, y_N, z_N)]
        self.dt = dt
        self.render_mode = None

        # Hyperparameters
        # TODO
        self.action_scalar = 10
        self.max_err = 100
        self.max_u = 500
        self.lookahead = 5

        # Project AirSim connection
        self.client = ProjectAirSimClient()
        self.client.connect()
        self.world = World(self.client, "scene_drone_classic_pid.jsonc")
        # Loading Project AirSim environment
        self.drone = Drone(self.client, self.world, "Drone1")
        self.simple_flight_rc = projectairsim.rc.SimpleFlightRC(self.client, "Drone1")
        # Drone API logic
        self.drone.enable_api_control()
        self.drone.arm()
        self.drone.disable_api_control()
        self.spawn_pose = self.drone.get_ground_truth_pose()
        # Rremote control logic
        rc_config = projectairsim.rc.RCConfig()
        rc_config_filename = "sim_config/xbox_rc_config.jsonc"
        projectairsim_log().info(f'Loading RC config file "{rc_config_filename}"')
        self.simple_flight_rc.rc_config = rc_config
        rc_config.load(rc_config_filename)

        # Setting constant wind
        self.world.set_wind_velocity(10, 0, 0)

        # CLient subscriptions
        self.client.subscribe(
            self.drone.robot_info["collision_info"],
            lambda _, collision_info: self.callback_collision(collision_info),
        )

        # Gains
        self.original_gains = np.asarray([0.5, 0.5, 0.8, 0.6, 0.6, 0.6, 0.0, 0.0, 0.02])

        # Assign initial variables (in case reset() is not called)
        self.reset()

        # Observation space definition
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(21+self.lookahead*3,),
            dtype=np.float64
        )

        # Action space definition
        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.original_gains.shape[0],),
            dtype=np.float32
        )
       
    def reset(self, seed=42, options=None):
        # Reset drone position
        self.drone.set_pose(self.spawn_pose)
        self.collision = False
    
        # Reset time step counter to 0
        self.time_step = 0
        # Reset gains
        self.gains = self.original_gains.copy()
        # Resetting PID controller
        self.pid = PIDController3D(self.gains[0:3], self.gains[3:6], self.gains[6:9], self.dt)
        # Reset log
        N = len(self.gamma)
        self.log = {
            "t": np.arange(N) * self.dt,
            "pos": np.zeros((N, 3)),
            "vel": np.zeros((N, 3)),
            "acc": np.zeros((N, 3)),
            "err": np.zeros((N, 3))
        }
        # Reseting error (e_t) and derivative/integral
        self.err = np.zeros(3)
        self.err_dot = np.zeros(3)
        self.err_int = np.zeros(3)
        # Resetting control action (u_t)
        self.u = np.zeros(3)

        # Getting observation
        obs = self.get_obs()
        info = {}
        return (obs, info)
    

    def callback_collision(self, collision_info):
        """Collision callback function."""
        self.collision = True
    
    
    def get_kinematics(self):
        kinematics = self.drone.get_ground_truth_kinematics()

        pos_x = kinematics["pose"]["position"]["x"]
        pos_y = kinematics["pose"]["position"]["y"]
        pos_z = -kinematics["pose"]["position"]["z"]        # NED to NEU

        v_x = kinematics["twist"]["linear"]["x"]
        v_y = kinematics["twist"]["linear"]["y"]
        v_z = -kinematics["twist"]["linear"]["z"]           # NED to NEU

        # Returning position and velocity
        return np.array([pos_x, pos_y, pos_z]), np.array([v_x, v_y, v_z])
    

    def move_drone(self, u):
        # Convert to RC commands
        throttle = thrust_to_throttle(u[2])

        controller_cmds = [
            roll_controller_cmd(u[0]),
            pitch_controller_cmd(u[1]),
            throttle_controller_cmd(throttle)
        ]

        channels = {
            "xLeft": controller_cmds[0],   # Roll
            "xRight": 0,
            "yLeft": controller_cmds[2],   # Throttle
            "yRight": controller_cmds[1],  # Pitch
            "switchLevelRate": 0,
            "switchEnableAPIControl": 0,
            "btnStart": 0,
            "btnBack": 0
        }

        # Assigning commands to flight controller
        self.simple_flight_rc.set(channels)
        # Resume simulator
        self.world.continue_for_sim_time(self.dt * 1e9)


    def step(self, action):
        # Transforming normalized [0, 1] action to proper values
        action = action * self.action_scalar
        # Adjust gains based on action
        self.gains = self.original_gains * action
        # Update PID gains
        self.pid.update_gains(self.gains[0:3], self.gains[3:6], self.gains[6:9])

        # Getting drone kinematics
        pos, vel = self.get_kinematics()

        # Getting reference path position at the current time step
        pos_ref = self.gamma[self.time_step, :]
        u, e, e_dot, e_int = self.pid.compute_control(pos, vel, pos_ref)
        # Taking a step in the environment
        self.move_drone(u)

        # Logging
        self.log["pos"][self.time_step, :] = pos
        self.log["vel"][self.time_step, :] = vel
        self.log["err"][self.time_step, :] = e
        if self.time_step > 0:
            self.log["acc"][self.time_step, :] = (self.log["vel"][self.time_step, :] - self.log["vel"][self.time_step - 1, :]) / self.dt
        else:
            self.log["acc"][self.time_step, :] = np.array([0, 0, 0])

        # Reshaping e and u into vectors
        e_vec = e.reshape(-1, 1)
        u = np.sign(u) * np.log(np.abs(u) + 1e-6)
        u = np.clip(u, -self.max_u, self.max_u)
        u_vec = u.reshape(-1, 1)
        # Q and R matrices
        Q = (1 / (self.max_err**2)) * np.eye(3)
        R = (1 / (self.max_u**2)) * np.eye(3)
        # Calculating reward term
        # TODO: normalize to prevent exploding ???
        reward = -(0.5 * e_vec.T @ Q @ e_vec + 0.5 * u_vec.T @ R @ u_vec)
        # Convert to scalar
        reward = reward[0][0]
        #print("Reward")
        #print(reward)

        # Updating internal variables
        self.err = e_vec.reshape(-1) / self.max_err
        self.u = u_vec.reshape(-1)
        self.err_dot = e_dot / self.max_err
        self.err_int = e_int / self.max_err

        # Checking if done
        done = False
        # No more reference path to follow, we're done
        if self.time_step == len(self.gamma) - 1:
            done = True
            #self.render()
        # Check if collision
        if self.collision:
            done = True
            reward = reward - 100
        
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
        # Lookahead points
        # Log PID gains (pos/vel)

        pos, _ = self.get_kinematics()

        # Array of future reference points (lookahead reference)
        lookahead_arr = []
        for k in range(self.lookahead):
            idx = min(self.time_step + k, len(self.gamma) - 1)
            # Relative position
            rel = self.gamma[idx] - pos
            lookahead_arr.append(rel)
        # Convert to numpy array
        lookahead_arr = np.asarray(lookahead_arr).flatten()

        # Log gains array
        log_gains = np.log(self.gains + 1e-6)           # + 1e-6 to prevent log(0)

        return np.concatenate([self.err, self.err_dot, self.err_int, self.u, lookahead_arr, log_gains]).astype(np.float32)

  

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

        # --- Error plot ---
        plt.figure()
        plt.plot(self.log["err"])
        plt.legend(["x", "y", "z"])
        plt.title("Tracking Error")
        plt.show()


    def close(self):
        self.client.disconnect()


def main():
    # Testing environment (no RL)
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

    # Executing env
    env.reset()
    while True:
        action = np.ones(9) * (1/10)
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

