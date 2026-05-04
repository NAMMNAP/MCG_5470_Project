import numpy as np
import matplotlib.pyplot as plt
import projectairsim.rc
from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log


class Path3D:
    def __init__(self, waypoints, ds=1.0):
        self.waypoints = np.asarray(waypoints)
        self.ds = ds
        self.s_grid = np.arange(len(waypoints)) * ds
        self.length = self.s_grid[-1]

    def position_numeric(self, s):
        s = np.clip(s, 0.0, self.length)
        idx = int(np.floor(s / self.ds))
        idx = min(idx, len(self.waypoints) - 2)
        alpha = (s - idx * self.ds) / self.ds
        return (1 - alpha) * self.waypoints[idx] + alpha * self.waypoints[idx + 1]


def update_progress(path, position, s_prev, window=1.0, samples=20):
    s_candidates = np.linspace(
        max(0.0, s_prev - window),
        min(path.length, s_prev + window),
        samples
    )

    dists = [
        np.linalg.norm(path.position_numeric(s) - position)
        for s in s_candidates
    ]

    return s_candidates[np.argmin(dists)]


# ================= PID CONTROLLER =================
class PIDController3D:
    def __init__(self, kp, kd, ki, dt, mass=1.22, g=9.81):
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.ki = np.array(ki)

        self.dt = dt
        self.mass = mass
        self.g = g

        self.integral = np.zeros(3)

    def update_gains(self, kp, kd, ki):
        self.kp = kp
        self.kd = kd
        self.ki = ki

    def compute_control(self, pos, vel, pos_ref):
        error = pos_ref - pos
        d_error = -vel

        self.integral += error * self.dt

        acc_cmd = (
            self.kp * error +
            self.kd * d_error +
            self.ki * self.integral
        )

        ax, ay, az = acc_cmd

        # Convert to drone commands (small-angle approx)
        roll  =  -ay / self.g
        pitch = ax / self.g
        thrust = self.mass * (self.g + az)
        thrust = self.mass * (self.g + az) - 0.5

        return np.array([roll, pitch, thrust]), error, d_error, self.integral


# ================= ACTUATOR MAPPING =================
def thrust_to_throttle(thrust, throttle_hover=0.5939, thrust_slope=19.9104):
    return (thrust / thrust_slope) + throttle_hover


def throttle_controller_cmd(throttle):
    if throttle >= 0.5:
        return (throttle - 0.5) * 2
    return -((throttle - 0.5) * 2)


def roll_controller_cmd(roll, max_roll=0.571):
    return -(roll / max_roll)


def pitch_controller_cmd(pitch, max_pitch=0.571):
    return (pitch / max_pitch)


# ================= MAIN =================
def main():
    # Time
    dt = 0.01
    T = 120
    t_array = np.arange(0, T + dt, dt)
    # Reference path
    x = (50 * np.sin(t_array / 5)) + 300
    y = 20 * np.cos(t_array / 5)
    z = (t_array * 10) + 4.0
    waypoints = np.vstack((x, y, z)).T

    # PID gains
    pid = PIDController3D(
        kp = [0.5, 0.5, 0.8],
        kd = [0.6, 0.6, 0.6],
        ki = [0.0, 0.0, 0.02],
        dt=dt
    )

    trajectory = []
    progress = []

    client = ProjectAirSimClient()
    client.connect()
    world = World(client, "scene_drone_classic_pid.jsonc")
    drone = Drone(client, world, "Drone1")
    simple_flight_rc = projectairsim.rc.SimpleFlightRC(client, "Drone1")

    drone.enable_api_control()
    drone.arm()
    drone.disable_api_control()

    rc_config = projectairsim.rc.RCConfig()
    rc_config_filename = "sim_config/xbox_rc_config.jsonc"
    projectairsim_log().info(f'Loading RC config file "{rc_config_filename}"')
    simple_flight_rc.rc_config = rc_config

    try:
        rc_config.load(rc_config_filename)
    except FileNotFoundError:
        print("RC config missing")
        return

    s_prev = 0
    goal_tol = 2
    k = 0

    # Setting constant wind
    world.set_wind_velocity(10, 0, 0)

    while True:
        kinematics = drone.get_ground_truth_kinematics()

        pos_x = kinematics["pose"]["position"]["x"]
        pos_y = kinematics["pose"]["position"]["y"]
        pos_z = -kinematics["pose"]["position"]["z"]        # NED to NEU

        v_x = kinematics["twist"]["linear"]["x"]
        v_y = kinematics["twist"]["linear"]["y"]
        v_z = -kinematics["twist"]["linear"]["z"]           # NED to NEU

        pos = np.array([pos_x, pos_y, pos_z])
        vel = np.array([v_x, v_y, v_z])
        pos_ref = waypoints[k]

        # PID control
        u = pid.compute_control(pos, vel, pos_ref)

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

        simple_flight_rc.set(channels)

        trajectory.append(pos.copy())

        world.continue_for_sim_time(dt * 1e9)
        k += 1
        if k >= x.shape[0]:
            break

    simple_flight_rc.stop()
    client.disconnect()

    trajectory = np.array(trajectory)

    # ================= PLOTTING =================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], "k--")
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], "b")

    plt.show()


if __name__ == "__main__":
    main()