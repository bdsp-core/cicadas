function [trajectories,t] = fcn_generateTrajectory(params, N, T, dt)
% GENERATEMETHOD2TRAJECTORIES Generate disease trajectories using Method 2 (time-varying volatility)
%
% Inputs:
%   params - Structure or vector containing parameters:
%            If vector: [growth_rate, peak_height, alpha, decay_rate, sigma_early, sigma_late, sigma_transition]
%            If struct: fields with same names
%   N      - Number of trajectories to generate (default: 100)
%   T      - Total simulation time in hours (default: 170)
%   dt     - Time step in hours (default: 0.1)
%
% Output:
%   trajectories - N x Nt matrix where each row is a trajectory
%
% Example usage:
%   % Using vector input
%   params = [0.25, 1.0, 0.15, 0.02, 0.15, 0.03, 40];
%   trajectories = generateMethod2Trajectories(params, 50, 170, 0.1);
%   
%   % Using struct input
%   params.growth_rate = 0.25;
%   params.peak_height = 1.0;
%   params.alpha = 0.15;
%   params.decay_rate = 0.02;
%   params.sigma_early = 0.15;
%   params.sigma_late = 0.03;
%   params.sigma_transition = 40;
%   trajectories = generateMethod2Trajectories(params);

    % Set default values if not provided
    if nargin < 2 || isempty(N)
        N = 100;
    end
    if nargin < 3 || isempty(T)
        T = 170;
    end
    if nargin < 4 || isempty(dt)
        dt = 0.1;
    end
    
    % Extract parameters
    if isstruct(params)
        growth_rate = params.growth_rate;
        peak_height = params.peak_height;
        alpha = params.alpha;
        decay_rate = params.decay_rate;
        sigma_early = params.sigma_early;
        sigma_late = params.sigma_late;
        sigma_transition = params.sigma_transition;
    else
        % Assume vector input in specified order
        if length(params) ~= 7
            error('Parameter vector must have 7 elements');
        end
        growth_rate = params(1);
        peak_height = params(2);
        alpha = params(3);
        decay_rate = params(4);
        sigma_early = params(5);
        sigma_late = params(6);
        sigma_transition = params(7);
    end
    
    % Create time vector
    t = 0:dt:T;
    Nt = length(t);
    
    % Initialize output matrix
    trajectories = zeros(N, Nt);
    
    % Generate each trajectory
    for i = 1:N
        X = zeros(1, Nt);
        X(1) = 0;  % Initial condition
        
        for j = 2:Nt
            current_time = t(j);
            
            % Growth dynamics
            if current_time < 30
                % Growth phase - logistic growth
                growth_term = growth_rate * X(j-1) * (1 - X(j-1)/peak_height);
            else
                % Decay phase with exponential decay
                time_since_peak = current_time - 30;
                decay_factor = exp(-decay_rate * time_since_peak);
                growth_term = -alpha * (X(j-1) - peak_height * decay_factor * 0.2);
            end
            
            % Mean reversion term to prevent overshooting peak
            mean_reversion = -alpha * max(0, X(j-1) - peak_height);
            
            % Total drift
            drift = growth_term + mean_reversion;
            
            % Time-varying diffusion coefficient
            sigma_t = sigma_early * exp(-(current_time/sigma_transition)) + sigma_late;
            diffusion = sigma_t * sqrt(max(0.001, X(j-1)));  % Minimum variance to avoid numerical issues
            
            % Euler-Maruyama update
            dW = sqrt(dt) * randn;  % Brownian increment
            X(j) = max(0, X(j-1) + drift * dt + diffusion * dW);
        end
        
        trajectories(i, :) = X;
    end
    
    % Optional: Display summary statistics
    if nargout == 0
        figure;
        subplot(2,1,1);
        hold on;
        for i = 1:min(N, 50)  % Plot up to 50 trajectories
            plot(t, trajectories(i, :), 'LineWidth', 0.5, 'Color', [0.3, 0.3, 0.8, 0.3]);
        end
        xlabel('Time (hours)');
        ylabel('Disease Burden');
        title('Individual Trajectories');
        grid on;
        
        subplot(2,1,2);
        mean_traj = mean(trajectories, 1);
        prc_5 = prctile(trajectories, 5, 1);
        prc_25 = prctile(trajectories, 25, 1);
        prc_50 = prctile(trajectories, 50, 1);
        prc_75 = prctile(trajectories, 75, 1);
        prc_95 = prctile(trajectories, 95, 1);
        
        hold on;
        fill([t, fliplr(t)], [prc_5, fliplr(prc_95)], [0.8, 0.8, 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
        fill([t, fliplr(t)], [prc_25, fliplr(prc_75)], [0.6, 0.6, 1], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        plot(t, mean_traj, 'b-', 'LineWidth', 3);
        plot(t, prc_50, 'r--', 'LineWidth', 2);
        xlabel('Time (hours)');
        ylabel('Disease Burden');
        title('Mean and Percentiles (5%, 25%, 50%, 75%, 95%)');
        legend({'5-95%', '25-75%', 'Mean', 'Median'}, 'Location', 'northeast');
        grid on;
    end
end