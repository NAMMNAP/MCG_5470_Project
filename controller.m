function [u, pos_err, W1, W2, W3] = RLController(pos_cmd, state)
% Compute accel commands (horizontal + vertical) from a desired velocity command.
% Simple PD-like controller in velocity space with maximum accel saturations applied later
%
% state: current state struct (as above)
% vel_cmd: desired velocity 3x1 [vx;vy;vz] in inertial frame (or body if gains.frame='body')
% p: same parameters struct (for limits)
% gains: struct with kp_vel (scalar or 3x1), kd (optional)
%
% RETURNS u: control struct with fields:
%  acc_hor_cmd (2x1), acc_z_cmd (scalar), yaw_rate_cmd (0 by default)

gamma = 0.9;
alpha = 0.05;
R = 0.05;
q = [0.05, 0; 0, 0];

psi = @(s, u) [s(1); s(2); s(1)^2; s(1)*s(2); u*s(1); u*s(2); s(2)^2; u^2];
hj = @(s,W) (-W(5) * s(1) - W(6) * s(2)) / (2 * W(8));
r = @(s, u) (1/2) * s.' * q * s + (1/2) * R*u.^2;

persistent W_tot
if isempty(W_tot)
    W_tot = 1 * rand(8, 3)+0.5;
    % W_tot(8,:) = 1;
end

persistent sprev_tot
if isempty(sprev_tot)
    sprev_tot = [0, 0, 0 ; 0, 0, 0];
end

persistent uprev_tot
if isempty(uprev_tot)
    uprev_tot = [0, 0, 0];
end

% position error
pos_err = state.pos - pos_cmd;

s_tot = [pos_err; state.vel];

W1 = W_tot(:,1);
W2 = W_tot(:,2);
W3 = W_tot(:,3);

for k = 1:3
    s = s_tot(:,k);
    sprev = sprev_tot(:,k);
    uprev = uprev_tot(k);
    W = W_tot(:,k);

    up = max(min(hj(s,W),10),-10);

    omega = psi(sprev, uprev) - gamma * psi(s, up);

    W = W - alpha * omega * (W.' * omega - r(sprev, uprev));
    W = max(min(W, 100), -100);

    W_tot(:,k) = W;

    u_cmd = max(min(hj(s,W),10),-10);

    u_cmd_tot(k) = u_cmd;

end




sprev_tot = s_tot;
uprev_tot = u_cmd_tot;

u_hor = u_cmd_tot(1:2);
u_vert = u_cmd_tot(3);

% assemble u; 
u.acc_hor_cmd = u_hor;
u.acc_z_cmd = u_vert;