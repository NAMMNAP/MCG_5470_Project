import numpy as np

def Controller(pos_cmd, state, p, gains, dt):
    """
    Compute acceleration commands from position tracking.

    Returns:
        u (dict): control commands
        pos_err (np.array): position error
    """

    # --- Default gains ---
    if "kp_vel" not in gains:
        gains["kp_vel"] = 1.5
    if "kd_vel" not in gains:
        gains["kd_vel"] = 0.0
    if "ki_vel" not in gains:
        gains["ki_vel"] = 0.0

    # --- Initialize persistent variables ---
    if not hasattr(Controller, "pos_err_previous"):
        Controller.pos_err_previous = np.zeros(3)
        Controller.pos_int_previous = np.zeros(3)
        Controller.vel_err_previous = np.zeros(3)
        Controller.vel_int_previous = np.zeros(3)

    # --- POSITION LOOP ---
    pos_err = pos_cmd - state["pos"]

    pos_err_derivative = (pos_err - Controller.pos_err_previous) / dt
    pos_err_integral = Controller.pos_int_previous + pos_err * dt

    Controller.pos_err_previous = pos_err
    Controller.pos_int_previous = pos_err_integral

    # Desired velocity
    vel_cmd = (
        gains["kp_pos"] * pos_err +
        gains["kd_pos"] * pos_err_derivative +
        gains["ki_pos"] * pos_err_integral
    )

    # --- VELOCITY LOOP ---
    vel_err = vel_cmd - state["vel"]

    vel_err_derivative = (vel_err - Controller.vel_err_previous) / dt
    vel_err_integral = Controller.vel_int_previous + vel_err * dt

    Controller.vel_err_previous = vel_err
    Controller.vel_int_previous = vel_err_integral

    # Desired acceleration
    a_des = (
        gains["kp_vel"] * vel_err +
        gains["kd_vel"] * vel_err_derivative +
        gains["ki_vel"] * vel_err_integral
    )

    # --- Split commands ---
    acc_hor_cmd = a_des[:2]
    acc_z_cmd = a_des[2]

    # --- Assemble control ---
    u = {
        "acc_hor_cmd": acc_hor_cmd,
        "acc_z_cmd": acc_z_cmd,
        "yaw_rate_cmd": 0.0  # same as MATLAB (unused)
    }

    return u, pos_err, pos_err_derivative, pos_err_integral