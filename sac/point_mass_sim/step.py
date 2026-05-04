import numpy as np

def step(state, u, p, dt, wind_vec=[0,0,0]):
    """
    One integration step for a multirotor intruder model with wind.

    state: dict
        pos      - np.array([x, y, z])
        vel_air  - np.array([vx, vy, vz])   # velocity relative to air
        yaw      - float (rad)

    u: dict
        acc_hor_cmd  - np.array([ax, ay])
        acc_z_cmd    - float
        yaw_rate_cmd - float (unused)

    dt: float

    Returns:
        state (updated), applied (dict)
    """

    # --- Defaults ---
    if "frame" not in p:
        p["frame"] = "inertial"

    # --- Extract state ---
    pos = state["pos"].copy()
    v_air = state["vel_air"].copy()
    yaw = state["yaw"]

    # --- Wind ---
    v_wind = np.asarray(wind_vec)

    # --- Horizontal acceleration ---
    a_hor_cmd = np.array(u["acc_hor_cmd"])

    if p["frame"].lower() == "body":
        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])
        a_hor_cmd = R @ a_hor_cmd

    # Limit horizontal acceleration
    a_hor_mag = np.linalg.norm(a_hor_cmd)
    if a_hor_mag > p["max_hor_acc"]:
        a_hor = (a_hor_cmd / a_hor_mag) * p["max_hor_acc"]
    else:
        a_hor = a_hor_cmd

    # --- Vertical acceleration ---
    a_z_cmd = u["acc_z_cmd"]

    if a_z_cmd > p["max_vert_acc_up"]:
        a_z = p["max_vert_acc_up"]
    elif a_z_cmd < -p["max_vert_acc_down"]:
        a_z = -p["max_vert_acc_down"]
    else:
        a_z = a_z_cmd

    # --- Integrate AIR velocity ---
    v_air_hor = v_air[:2] + a_hor * dt
    vz_air = v_air[2] + a_z * dt

    # --- Limit AIR velocity ---
    v_air_hor_speed = np.linalg.norm(v_air_hor)
    if v_air_hor_speed > p["max_hor_vel"]:
        v_air_hor = (v_air_hor / v_air_hor_speed) * p["max_hor_vel"]

    if vz_air > p["max_vert_vel"]:
        vz_air = p["max_vert_vel"]
    elif vz_air < -p["max_vert_vel"]:
        vz_air = -p["max_vert_vel"]

    # --- Compute ground velocity (wind added AFTER limiting) ---
    v_hor = v_air_hor + v_wind[:2]
    vz = vz_air + v_wind[2]

    # --- Yaw logic ---
    # Choose alignment reference:
    if np.linalg.norm(v_air_hor) > 1e-6:
        desired_yaw = np.arctan2(v_air_hor[1], v_air_hor[0])
    else:
        desired_yaw = yaw

    dyaw = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
    max_dyaw = p["max_yaw_rate"] * dt

    if abs(dyaw) > max_dyaw:
        yaw = yaw + np.sign(dyaw) * max_dyaw
        dyaw = np.sign(dyaw) * max_dyaw
    else:
        yaw = desired_yaw

    # --- Update position using GROUND velocity ---
    pos[:2] += v_hor * dt
    pos[2] += vz * dt

    # --- Write back ---
    state["pos"] = pos
    state["vel_air"] = np.array([v_air_hor[0], v_air_hor[1], vz_air])
    state["vel"] = np.array([v_hor[0], v_hor[1], vz])  # optional (ground velocity)
    state["yaw"] = yaw

    # --- Applied outputs ---
    applied = {
        "a_hor": a_hor,
        "a_z": a_z,
        "yaw_rate": dyaw,
        "wind": v_wind
    }

    return state, applied