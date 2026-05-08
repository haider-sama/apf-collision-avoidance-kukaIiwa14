clc;
clear;
close all;

%% Load Robot
[kuka_robot, arm_info] = loadrobot('kukaIiwa14', 'DataFormat', 'row'); 
ee_body = 'iiwa_link_ee';

% The links which will be avoiding the obstacles
collision_links = {'iiwa_link_4', 'iiwa_link_5', 'iiwa_link_6', 'iiwa_link_ee'};

%% APF Params
goal = [0.6, 0.0, 0.5]; % Goal in Cartesian space [x, y, z] (meter)
k_att = 4.0;            % Attractive gain
k_rep = 0.04;           % Repulsive gain 
dt = 0.05;              % Timestep
threshold = 0.02;       % Convergence threshold (2cm)
rho_0 = 0.05;           % Distance of influence (5cm)
iterations = 500;       % Limit for iterations
d_star = 0.3;           % Conic/quadratic switch distance
q_dot_max = 5.0;        % Max joint velocity (rad/s)


%% 3. Obstacle Configurations
% Matrix format: [x, y, z, radius]

obs_config_1 = [
    0.42,  0.02,  0.8, 0.06;  
    0.45, -0.10,  0.4, 0.06;
    0.35,  0.15,  0.7, 0.06;
    0.55,  0.30,  0.4, 0.06;
    0.25, -0.35,  0.5, 0.06
];

obs_config_2 = [
    0.45,  0.00, 0.6, 0.08;
    0.35,  0.15, 0.6, 0.08;
    0.45,  0.15, 0.6, 0.08;
    0.35, -0.15, 0.6, 0.08;
    0.45, -0.15, 0.6, 0.08;
];

%% Screenshot folder — created if missing, overwritten on every run
if ~exist('screenshots', 'dir')
    mkdir('screenshots');
end

%% Sphere mesh for obstacle rendering
[sx, sy, sz] = sphere(16);
lightBlue = [0.678, 0.847, 0.902];

%% Storage for both configs
all_trace_x = cell(1,2);
all_trace_y = cell(1,2);
all_trace_z = cell(1,2);
all_qdot_history = cell(1,2);
all_obs = {obs_config_1, obs_config_2};
config_names = {'Config 1', 'Config 2'};

%% APF

for cfg = 1:2
    
    obs = all_obs{cfg};
    q = homeConfiguration(kuka_robot); % 1x7 row vector

    trace_x = nan(1, iterations);
    trace_y = nan(1, iterations);
    trace_z = nan(1, iterations);
    q_dot_history = nan(7, iterations);
    q_history = nan(7, iterations);

    pos_buffer = zeros(3, 15); % For Case 3: Limit Cycles/Oscillation
    goalReached = false;

    fprintf('%s \n', config_names{cfg});
    fprintf('Starting APF...\n');

    for i = 1:iterations
        T_ee = getTransform(kuka_robot, q, ee_body);
        x_ee = T_ee(1:3, 4)';

        % Error vector pointing from EE to Goal (Attractive Force)
        dx = x_ee - goal;
        dist_to_goal = norm(dx); % Calculate current distance

        % Record trace before the break check so the final point is saved
        trace_x(i) = x_ee(1);
        trace_y(i) = x_ee(2);
        trace_z(i) = x_ee(3);

        if dist_to_goal < threshold
            fprintf('Goal reached in %d steps! Final Distance: %.4f m\n', i, dist_to_goal);
            goalReached = true;
            break; 
        end

        % A. Attractive Force (conic/quadratic blend)
        if dist_to_goal <= d_star
            % quadratic region
            F_att = -k_att * dx;
        else
            % conic region
            F_att = -d_star * k_att * (dx / dist_to_goal);
        end

        % returns [angular; linear] velocities (6x7)
        J_ee = geometricJacobian(kuka_robot, q, ee_body); 
        J_v_ee = J_ee(4:6, :); % extract 3x7 linear part
        
        % Project EE attractive force to joint space: q_dot = J' * F
        q_dot_total = (pinv(J_v_ee) * F_att')';
    
        % B. Repulsive Force (summed over all collision links and obstacles)
        F_rep_total = [0 0 0];

        for L = 1:length(collision_links)
            link_name = collision_links{L};
            T_link = getTransform(kuka_robot, q, link_name);
            x_link = T_link(1:3, 4)';
            
            for j = 1:size(obs, 1)
                obs_p = obs(j, 1:3);
                obs_r = obs(j, 4);
                
                dist_vec = x_link - obs_p;
                dist_to_center = norm(dist_vec);
                rho = dist_to_center - obs_r; % Distance to obstacle surface
                
                if rho <= rho_0 && rho > 0.001
                    % Formula: k_rep * (1/rho - 1/rho_0) * (1/rho^2) * unit_direction
                    rep_mag = k_rep * (1/rho - 1/rho_0) * (1/rho^2);
                    F_rep_total = F_rep_total + (rep_mag * (dist_vec / dist_to_center));
                end
            end

            % Project this link's repulsive force into joint space
            J_link = geometricJacobian(kuka_robot, q, link_name);
            q_dot_link = (pinv(J_link(4:6, :)) * F_rep_total')';
            q_dot_total = q_dot_total + q_dot_link;
        end

        F_total = F_att + F_rep_total;
        stuck = false;
        
        % CASE 1: Local Minimum (gradient is 0)
        if norm(F_total) < 1e-3 && dist_to_goal > threshold
            fprintf('Absolute local minima at iter %d\n', i);
            stuck = true;
        end
        
        % CASE 2: Limit Cycle / Oscillation
        pos_buffer(:, mod(i,15)+1) = x_ee';
        if i > 15
            net_displacement = norm(x_ee' - pos_buffer(:, mod(i-14,15)+1));
            if net_displacement < 0.01 % Trapped if moved < 1cm in 15 steps
                fprintf('Limit cycle at iter %d\n', i);
                stuck = true;
            end
        end
    
        % Random Perturbation
        if stuck
            fprintf('Triggering random perturbation escape...\n');
            q = q + (rand(1,7) - 0.5) * 5.0; 
            continue;
        end
    
        % Velocity clamp
        max_qdot = max(abs(q_dot_total));
        if max_qdot > q_dot_max
            q_dot_total = q_dot_total * (q_dot_max / max_qdot);
        end
    
        % update robot state over time
        q_dot_history(:, i) = q_dot_total';
        q_history(:, i) = q';
        q = q + (q_dot_total * dt);
    end % APF loop

    if ~goalReached
        fprintf('Max iterations reached for %s\n', config_names{cfg});
    end

    %  PLOT 0 - Each unique config
    fig_robot = figure('Visible', 'on');
    ax_robot  = gca;
    hold on; grid on; axis equal;
    view(140, 20);
    axis([-0.5 1 -0.5 1 0 1.2]);

    % Draw final robot pose
    show(kuka_robot, q, 'Parent', ax_robot, 'Visuals', 'on', ...
        'PreservePlot', false, 'Frames', 'off');

    % Draw EE traced path
    valid_r = ~isnan(trace_x);
    plot3(ax_robot, trace_x(valid_r), trace_y(valid_r), trace_z(valid_r), ...
        'b-', 'LineWidth', 2);

    % Draw goal marker
    plot3(ax_robot, goal(1), goal(2), goal(3), ...
        'gs', 'MarkerSize', 10, 'LineWidth', 2);

    % Draw obstacles
    for j = 1:size(obs, 1)
        o = obs(j,:);
        surf(ax_robot, sx*o(4)+o(1), sy*o(4)+o(2), sz*o(4)+o(3), ...
            'FaceColor', lightBlue, 'EdgeColor', 'none', 'FaceAlpha', 0.6);
    end

    title(ax_robot, sprintf('%s | Steps: %d | Dist: %.4f m', ...
        config_names{cfg}, i, dist_to_goal));
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    ax_robot.Toolbar.Visible = 'off';
    exportgraphics(fig_robot, sprintf('screenshots/robot_config%d.png', cfg), 'Resolution', 150);

    % Store for comparison plot later
    all_trace_x{cfg}      = trace_x;
    all_trace_y{cfg}      = trace_y;
    all_trace_z{cfg}      = trace_z;
    all_qdot_history{cfg} = q_dot_history;

    %  SAVE valid Trajectory for Simulink
    valid_idx = any(~isnan(q_history), 1);
    waypoints = q_history(:, valid_idx);   % [7 x N]
    dt = 0.05;
    N  = size(waypoints, 2);
    time = (0:N-1) * dt; % [1 x N]
    t_end = time(end);
    traj = timeseries(waypoints', time); % [N x 7]
  
    save_name = sprintf('sim_waypoints_%d.mat', cfg);
    save(save_name, 'waypoints', 'traj', 'time', 't_end');
    fprintf('Saved Config %d: %s\n', cfg, save_name);
    fprintf('Waypoints size: [%d x %d]\n', size(waypoints,1), size(waypoints,2));
    fprintf('End Time: %.2f s\n', t_end);

    %  PLOT 1 — 3D EE Trajectory
    fig_traj = figure('Visible', 'on');
    valid = ~isnan(trace_x);
    plot3(trace_x(valid), trace_y(valid), trace_z(valid), 'b-', 'LineWidth', 2);
    hold on;
    plot3(trace_x(find(valid,1)), trace_y(find(valid,1)), trace_z(find(valid,1)), ...
        'go', 'MarkerSize', 10, 'LineWidth', 2);
    plot3(goal(1), goal(2), goal(3), 'rs', 'MarkerSize', 10, 'LineWidth', 2);
    for j = 1:size(obs, 1)
        o = obs(j,:);
        surf(sx*o(4)+o(1), sy*o(4)+o(2), sz*o(4)+o(3), ...
            'FaceColor', lightBlue, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    end
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title(sprintf('EE 3D Trajectory — %s', config_names{cfg}));
    legend('EE Path', 'Start', 'Goal');
    grid on; axis equal; view(140, 20);
    exportgraphics(fig_traj, sprintf('screenshots/traj_config%d.png', cfg), 'Resolution', 150);

    %  PLOT 2 — Joint Velocity Profile
    fig_vel = figure('Visible', 'on');
    steps_q = find(any(~isnan(q_dot_history), 1), 1, 'last');
    time    = (1:steps_q) * dt;
    plot(time, q_dot_history(:, 1:steps_q)');
    xlabel('Time (s)'); ylabel('Joint Velocity (rad/s)');
    title(sprintf('Joint Velocity Profiles — %s', config_names{cfg}));
    legend('J1','J2','J3','J4','J5','J6','J7');
    grid on;
    exportgraphics(fig_vel, sprintf('screenshots/jointvel_config%d.png', cfg), 'Resolution', 150);

    %  PLOT 3 — Potential Field Landscape (2D slice at z = goal_z)
    res     = 100;
    x_range = linspace(-0.5, 1.0, res);
    y_range = linspace(-0.5, 1.0, res);
    z_slice = goal(3);
    [X, Y]  = meshgrid(x_range, y_range);
    U_total = zeros(res, res);

    for xi = 1:res
        for yi = 1:res
            p       = [X(yi,xi), Y(yi,xi), z_slice];
            rho_att = norm(p - goal);
            if rho_att <= d_star
                U_att = 0.5 * k_att * rho_att^2;
            else
                U_att = d_star * k_att * rho_att - 0.5 * k_att * d_star^2;
            end
            
            U_rep = 0;
            for j = 1:size(obs, 1)
                obs_p = obs(j, 1:3);
                obs_r = obs(j, 4);
                rho   = norm(p - obs_p) - obs_r;
                if rho <= rho_0 && rho > 0.001
                    U_rep = U_rep + 0.5 * k_rep * (1/rho - 1/rho_0)^2;
                end
            end

            U_total(yi,xi) = U_att + U_rep;
        end
    end

    U_plot  = min(U_total, prctile(U_total(:), 95));
    fig_pot = figure('Visible', 'on');
    surf(X, Y, U_plot, 'EdgeColor', 'none');
    colormap turbo; colorbar; hold on;
    plot3(goal(1), goal(2), 0, 'gp', 'MarkerSize', 15, 'MarkerFaceColor', 'g');
    for j = 1:size(obs, 1)
        obs_p = obs(j, 1:3);
        obs_r = obs(j, 4);
        dz    = abs(obs_p(3) - z_slice);
        if dz < obs_r
            r_slice = sqrt(obs_r^2 - dz^2);
            theta   = linspace(0, 2*pi, 60);
            cx = obs_p(1) + r_slice * cos(theta);
            cy = obs_p(2) + r_slice * sin(theta);
            plot3(cx, cy, zeros(size(cx)), 'r-', 'LineWidth', 2);
        end
    end
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('U_{total}');
    title(sprintf('APF Potential Landscape (z=%.2f m) — %s', z_slice, config_names{cfg}));
    view(45, 35); grid on;
    exportgraphics(fig_pot, sprintf('screenshots/potential_config%d.png', cfg), 'Resolution', 150);

end % config loop

% Joint Velocity Comparison (both configs)
fig_cmp = figure('Visible', 'on');

subplot(2,1,1);
steps1 = find(any(~isnan(all_qdot_history{1}), 1), 1, 'last');
time1  = (1:steps1) * dt;
plot(time1, all_qdot_history{1}(:,1:steps1)');
title(sprintf('Joint Velocities — %s', config_names{1}));
xlabel('Time (s)'); ylabel('rad/s');
legend('J1','J2','J3','J4','J5','J6','J7');
grid on;

subplot(2,1,2);
steps2 = find(any(~isnan(all_qdot_history{2}), 1), 1, 'last');
time2  = (1:steps2) * dt;
plot(time2, all_qdot_history{2}(:,1:steps2)');
title(sprintf('Joint Velocities — %s', config_names{2}));
xlabel('Time (s)'); ylabel('rad/s');
legend('J1','J2','J3','J4','J5','J6','J7');
grid on;

ax = gca;
ax.Toolbar.Visible = 'off';
exportgraphics(fig_cmp, 'screenshots/velocity_comparison.png', 'Resolution', 150);
fprintf('\nplots saved to screenshots/ folder.\n');