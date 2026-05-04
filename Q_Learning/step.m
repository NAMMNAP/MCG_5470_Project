function [state, applied] = step(state, u, p, dt)
% INTRUDER_STEP One integration step for a multirotor intruder model.
%
% state: struct with fields
%   pos  - 3x1 [x;y;z] (m)   (z positive = up)
%   vel  - 3x1 [vx;vy;vz] (m/s)
%   yaw  - scalar (rad)  (heading, 0 = x axis)
%
% u: control struct with fields (commanded)
%   acc_hor_cmd - 2x1 [ax; ay] (m/s^2) in body or inertial frame (see p.frame)
%   acc_z_cmd   - scalar (m/s^2) vertical accel (positive = up)
%   yaw_rate_cmd- scalar (rad/s) desired yaw rate (positive CCW)
%
% p: parameters struct with fields
%   frame          - 'inertial' or 'body' (if 'body', acc_hor_cmd is in body frame)
%   max_hor_vel    - scalar (m/s)
%   max_vert_vel   - scalar (m/s) (positive up)
%   max_hor_acc    - scalar (m/s^2) (limits magnitude of horizontal accel)
%   max_vert_acc_up   - scalar (m/s^2) (positive up)
%   max_vert_acc_down - scalar (m/s^2) (positive down magnitude)
%   max_yaw_rate   - scalar (rad/s)
%   mass           - scalar (kg) (optional, not used unless you want forces)
%   gravity        - scalar (m/s^2) positive (default 9.81)
%
% dt: timestep (s)
%
% RETURNS:
%  state: updated state struct
%  applied: struct with applied accelerations and yaw rate
%
% Notes:
%  - horizontal accel is limited by max_hor_acc (magnitude)
%  - vertical accel is limited between [-max_vert_acc_down, max_vert_acc_up]
%  - velocities are saturated after integration
%  - yaw rate command is clipped to +/- max_yaw_rate

if ~isfield(p,'frame'), p.frame = 'inertial'; end

% --- Extract current horizontal state
pos = state.pos;
vel = state.vel;   % [vx; vy; vz]
yaw = state.yaw;

% horizontal acceleration command (2x1)
a_hor_cmd = u.acc_hor_cmd;
if strcmpi(p.frame,'body')
    % rotate body->inertial by yaw
    R = [cos(yaw) -sin(yaw); sin(yaw) cos(yaw)];
    a_hor_cmd = R * a_hor_cmd;
end

% a_hor = a_hor_cmd;

% limit horizontal accel magnitude
a_hor_mag = norm(a_hor_cmd);
if a_hor_mag > p.max_hor_acc
    a_hor = (a_hor_cmd / a_hor_mag) * p.max_hor_acc;
else
    a_hor = a_hor_cmd;
end

% vertical accel (scalar)
a_z_cmd = u.acc_z_cmd;
a_z = a_z_cmd;
% limit vertical accel: allow asymmetric up/down limits
if a_z_cmd > p.max_vert_acc_up
    a_z = p.max_vert_acc_up;
elseif a_z_cmd < -p.max_vert_acc_down
    a_z = -p.max_vert_acc_down;
else
    a_z = a_z_cmd;
end

% integrate velocities (Euler)
v_hor = vel(1:2) + a_hor * dt;
vz = vel(3) + a_z * dt;

% enforce horizontal speed limit (preserve direction)
% v_hor_speed = norm(v_hor);
% if v_hor_speed > p.max_hor_vel
%     v_hor = (v_hor / v_hor_speed) * p.max_hor_vel;
% end


% 
% % enforce vertical speed limit
% if vz > p.max_vert_vel
%     vz = p.max_vert_vel;
% elseif vz < -p.max_vert_vel
%     vz = -p.max_vert_vel;
% end


desired_yaw = atan2(v_hor(2), v_hor(1));
yaw = desired_yaw;
dyaw = mod(desired_yaw-state.yaw+pi,2*pi)-pi; %ensures yaw is between pi and -pi
% max_dyaw = p.max_yaw_rate*dt; %yaw rate from euler integration
% % if yaw rate is above the max, recaclulate the cartesian velocity
% % components to give the maximum yaw
% if abs(dyaw)>max_dyaw
%     limited_yaw = state.yaw + sign(dyaw)*max_dyaw;
%     v_hor = norm(v_hor)*[cos(limited_yaw), sin(limited_yaw)];
%     yaw = limited_yaw;
%     dyaw = max_dyaw;
% end

% update position
pos(1:2) = pos(1:2) + v_hor * dt;
pos(3) = pos(3) + vz * dt;

% write back
state.pos = pos;
state.vel = [v_hor vz];
state.yaw = yaw;

% return applied values for logging
applied.a_hor = a_hor;
applied.a_z = a_z;
applied.yaw_rate = dyaw;

end
