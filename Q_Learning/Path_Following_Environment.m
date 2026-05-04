% DEMO_INTRUDER_FLIGHT_MODES.m
% Demonstrates intruder flight modes with constraints
clc;clear all;close all;
clear RLController
clear step

%% === PARAMETERS ===
p.frame = 'inertial';
p.max_hor_vel = 20;                 % [m/s]
p.max_vert_vel = 10;                % [m/s]
p.max_hor_acc = 1 * 9.81;         % [m/s^2]
p.max_vert_acc_up = 0.8 * 9.81;       % [m/s^2]
p.max_vert_acc_down = 0.8 * 9.81;     % [m/s^2]
p.max_yaw_rate = deg2rad(150);      % [rad/s]

% Time
dt = 0.001;
T = 500; % total time (s)

% timestep
N = round(T/dt);

t_array = 0:dt:T;

% % create path
% x = 5*sin(t_array/50);
% y = 5*cos(t_array/50);
% z = 0.1*(t_array);
% dx = 0.5*cos(t_array/2);
% dy = -0.5*sin(t_array/2);
% dz = cos(t_array/2);

ddx = 15 * rand(1,length(t_array)) - 7.5;
ddy = 15 * rand(1,length(t_array)) - 7.5;
ddz = 5 * rand(1,length(t_array)) - 2.4;


ddx = smoothdata(ddx, 'movmean', 50);
ddy = smoothdata(ddy, 'movmean', 50);
ddz = smoothdata(ddz, 'movmean', 50);

x0 = 5;
y0 = 0;
z0 = 0;

dx0 = 0;
dy0 = 0;
dz0 = 0;


dx = zeros(1,N);
dy = zeros(1,N);
dz = zeros(1,N);

dx(1) = dx0;
dy(1) = dy0;
dz(1) = dz0;


dx(2:end) = dx0 + cumsum(ddx(1:end-2)) * dt;
dy(2:end) = dy0 + cumsum(ddy(1:end-2)) * dt;
dz(2:end) = dz0 + cumsum(ddz(1:end-2)) * dt;

x = zeros(1,N);
y = zeros(1,N);
z = zeros(1,N);

x(1) = x0;
y(1) = y0;
z(1) = z0;

x(2:end) = x0 + cumsum(dx(1:end-1)) * dt;
y(2:end) = y0 + cumsum(dy(1:end-1)) * dt;
z(2:end) = z0 + cumsum(dz(1:end-1)) * dt;


path = [x; y; z].';
dpath = [dx; dy; dz].';

% plot3(path(:,1),path(:,2),path(:,3))


%% === INITIAL STATE ===
start_pos = [0 0 0];
start_vel = [0 0 0];
start_yaw = 0;

state.pos = start_pos;   % start at initialized position
state.vel = start_vel;
state.yaw = start_yaw;

% gains for controller
gains.kp_pos =2;
gains.kd_pos = 1;
gains.ki_pos = 0.01;

gains.kp_vel = 2;
gains.kd_vel = 1;
gains.ki_vel = 0.01;


% create function for path
% preallocate logs
log.t = (0:N-1)'*dt;    % time log
log.pos = zeros(N,3);   % position log
log.vel = zeros(N,3);   % velocity log
log.acc = zeros(N,3);   % acceleration log
log.yaw = zeros(N,1);   % yaw log
log.err = zeros(N,3);   % yaw log
log.Wx = zeros(N,8);
log.Wy = zeros(N,8);
log.Wz = zeros(N,8);


%% === SIMULATION ===

% === Absolute altitude reference ===

for k = 1:N
    t = (k-1)*dt; 

    cmd_pos = path(k,:);
    cmd_vel = dpath(k,:);

    % --- Controller + Dynamics ---
    [u, e, Wx, Wy, Wz] = RLController(cmd_pos, state);
    [state, applied] = step(state, u, p, dt);

    % --- Log ---
    log.pos(k,:) = state.pos;
    log.vel(k,:) = state.vel;
    log.yaw(k) = state.yaw;
    log.err(k,:) = e;
    log.Wx(k,:) = Wx';
    log.Wy(k,:) = Wy';
    log.Wz(k,:) = Wz';
    hor_dist_trav = state.pos(1);
    if k > 1
        log.acc(k,:) = (log.vel(k,:) - log.vel(k-1,:)) / dt;
    else
        log.acc(k,:) = [0 0 0];  % initialize
    end
end

%% === PLOTS ===
figure('Name','3D Trajectory','NumberTitle','off','Position',[50 50 900 700]);
ax = axes;
hold(ax,'on');
plot3(ax, path(:,1), path(:,2), path(:,3), 'LineWidth', 1.5);
plot3(ax, log.pos(:,1), log.pos(:,2), log.pos(:,3), 'LineWidth', 1.5);
plot3(ax, log.pos(1,1), log.pos(1,2), log.pos(1,3), 'go','MarkerFaceColor','g','MarkerSize',8);
plot3(ax, log.pos(end,1), log.pos(end,2), log.pos(end,3), 'ro','MarkerFaceColor','r','MarkerSize',8);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
%title(sprintf('Intruder Trajectory - %s Mode', Mode));
grid on; axis equal;
% zlim([0 100])
%     xlim([0 800])
% ylim([-100 100])
%view([0 0]);
rotate3d on;
legend( 'Set Path','Actual Path', 'Start', 'End')


figure('Name','UAV 2D Metrics','NumberTitle','off','Position',[1150 150 850 650]);
subplot(3,1,1)
plot(log.t, log.pos(:,3)); 
ylabel('Altitude (m)');
xlabel('Time (s)');

subplot(3,1,2)
plot(log.t, vecnorm(log.vel(:,1:2),2,2)); 
ylabel('Horizontal Velocity (m/s)');
xlabel('Time (s)');
yline(p.max_hor_vel,'r--','Max');
title('Ground Speed');
% ylim([0 30])

subplot(3,1,3)
plot(log.pos(:,1), log.pos(:,2));
xlabel('X (m)'); 
ylabel('Y (m)'); 
title('XY track');
% ylim([0 80])

figure('Name','Velocity and Acceleration','NumberTitle','off','Position',[200 200 850 650]);

% --- Vertical velocity ---
subplot(2,2,1)
plot(log.t, log.vel(:,3));
yline(p.max_vert_vel, 'r--', 'Max climb');
yline(-p.max_vert_vel, 'r--', 'Max descent');
ylabel('Vertical Velocity (m/s)');
title('Vertical Velocity');
xlabel('Time (s)');
% ylim([-10 15])

% --- Horizontal velocity ---
subplot(2,2,2)
plot(log.t, vecnorm(log.vel(:,1:2),2,2));
ylabel('Horizontal Speed (m/s)');
yline(p.max_hor_vel,'r--','Max');
title('Horizontal Velocity');
xlabel('Time (s)');
% ylim([0 30])

% --- Vertical acceleration ---
subplot(2,2,3)
plot(log.t, log.acc(:,3));
yline(p.max_vert_acc_up, 'r--', 'Max upward acc');
yline(-p.max_vert_acc_down, 'r--', 'Max downward acc');
ylabel('Vertical Acceleration (m/s²)');
xlabel('Time (s)');
title('Vertical Acceleration');
% ylim([-10 15])

% --- Horizontal acceleration ---
subplot(2,2,4)
plot(log.t, vecnorm(log.acc(:,1:2),2,2));
ylabel('Horizontal Acceleration (m/s²)');
yline(p.max_hor_acc,'r--','Max');
title('Horizontal Acceleration');
xlabel('Time (s)');
% ylim([-5 20])

figure()
plot(log.err)
legend('x','y','z')


figure()
plot(log.Wx, LineWidth=1.5)
xlabel("TIme Step (k)")
ylabel("Parameter Value")
legend('1','2','3','4','5','6','7','8')

figure()
plot(log.Wy, LineWidth=1.5)
xlabel("TIme Step (k)")
ylabel("Parameter Value")
legend('1','2','3','4','5','6','7','8')

figure()
plot(log.Wz, LineWidth=1.5)
xlabel("TIme Step (k)")
ylabel("Parameter Value")
legend('1','2','3','4','5','6','7','8')