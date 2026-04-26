clc;
clear;
close all;

%% Load Robot
[kuka_robot, arm_info] = loadrobot('kukaIiwa14', 'DataFormat', 'row'); 
q = homeConfiguration(kuka_robot); % 1x7 row vector
ee_body = 'iiwa_link_ee';

%% APF Params
goal = [0.5, 0.3, 0.6]; % Goal in Cartesian space [x, y, z] (meter)
k_att = 2.0;            % Attractive gain
k_rep = 0.01;           % Repulsive gain 
dt = 0.05;              % Timestep
threshold = 0.05;       % Convergence threshold (1cm)
rho_0 = 0.2;            % Distance of influence (20cm)
iterations = 500;       % Limit for iterations

%% 3. Obstacle Configuration 1
% Matrix format: [x, y, z, radius]
obs_config_1 = [
    0.3, 0.2, 0.6, 0.05;
    0.2, 0.4, 0.7, 0.05
];

%% Visualize Robot
figure;
hold on; grid on; axis equal;
view(140, 20);
title('kukaIiwa14 - APF Collision Avoidance');

% draw robot's axes
trplot(eye(4), 'frame', 'W', 'color', 'k');

[sx, sy, sz] = sphere(20);
lightBlue = [0.678, 0.847, 0.902];

% expand visual space manually
axis([-0.5 1 -0.5 1 0 1.2]);
ax = gca; % capture current axes

%% APF
fprintf('Starting movement toward goal...\n');

% Pre-allocate trace storage and create persistent trace handle
trace_x = nan(1, iterations);
trace_y = nan(1, iterations);
trace_z = nan(1, iterations);

goalReached = false;

for i = 1:iterations
    T = getTransform(kuka_robot, q, ee_body);
    x = T(1:3, 4)'; 
    
    % Error vector pointing from EE to Goal (Attractive Force)
    dx = x - goal;
    dist_to_goal = norm(dx); % Calculate current distance
    
    if dist_to_goal < threshold
        fprintf('Goal reached in %d steps! Final Distance: %.4f m\n', i, dist_to_goal);
        goalReached = true;
        break; 
    end

    F_att = -k_att * dx;

    % B. Repulsive Force (Summed from all obstacles)
    F_rep_total = [0, 0, 0];
    for j = 1:size(obs_config_1, 1)
        obs_p = obs_config_1(j, 1:3);
        obs_r = obs_config_1(j, 4);
        
        dist_vec = x - obs_p;
        dist_to_center = norm(dist_vec);
        rho = dist_to_center - obs_r; % Distance to obstacle surface
        
        if rho <= rho_0 && rho > 0.001
            % Formula: k_rep * (1/rho - 1/rho_0) * (1/rho^2) * unit_direction
            rep_mag = k_rep * (1/rho - 1/rho_0) * (1/rho^2);
            unit_dir = dist_vec / dist_to_center;
            F_rep_total = F_rep_total + (rep_mag * unit_dir);
        end
    end

    F_total = F_att + F_rep_total;

    % returns [angular; linear] velocities (6x7)
    J = geometricJacobian(kuka_robot, q, ee_body); 
    J_v = J(4:6, :); % extract 3x7 linear part
    
    q_dot = pinv(J_v) * F_total';

    % update robot state over time
    q = q + (q_dot' * dt);
    
    % update the visualization
    show(kuka_robot, q, 'Parent', ax, 'Visuals', 'on', 'PreservePlot', false, 'Frames', 'off');
    % Re-draw persistent elements after show() clears them
    plot3(ax, goal(1), goal(2), goal(3), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
    for j = 1:size(obs_config_1, 1)
        o_x = obs_config_1(j,1); o_y = obs_config_1(j,2); o_z = obs_config_1(j,3); o_r = obs_config_1(j,4);
        surf(ax, sx*o_r + o_x, sy*o_r + o_y, sz*o_r + o_z, ...
            'FaceColor', lightBlue, 'EdgeColor', 'none', 'FaceAlpha', 0.6);
    end
    trace_x(i) = x(1); trace_y(i) = x(2); trace_z(i) = x(3); % Trace path
    plot3(ax, trace_x, trace_y, trace_z, 'b.', 'MarkerSize', 5);
    axis(ax, [-0.5 1 -0.5 1 0 1.2]);
    drawnow limitrate; % Force configure update
end

if goalReached
    title(ax, sprintf('Goal reached in %d steps! Final Distance: %.4f m', i, dist_to_goal), 'Color', [0 0.5 0]);
else
    title(ax, sprintf('Max iterations reached. Dist: %.3f m', norm(x - goal)), 'Color', 'r');
end