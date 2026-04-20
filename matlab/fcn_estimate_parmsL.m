function [theta_est, diagnostics] = fcn_estimate_parmsL(trajectories, t)
%% estimateParameters - Estimate parameters from observed trajectories
%
% Inputs:
%   trajectories - N x T matrix of observed disease trajectories
%   t            - Time vector (hours)
%   true_params  - (Optional) True parameter values for comparison
%
% Outputs:
%   theta_est    - Estimated parameter vector [gamma, H, alpha, delta, sigma_early, sigma_late, tau]
%   diagnostics  - Structure containing diagnostic information

dt = t(2) - t(1);
[N, Nt] = size(trajectories);


% Estimate growth rate from early exponential phase
early_growth_rates = zeros(N, 1);
for i = 1:N
    peak_val = max(trajectories(i, :));
    idx_5pct = find(trajectories(i, :) > 0.05*peak_val, 1);
    idx_50pct = find(trajectories(i, :) > 0.5*peak_val, 1);
    
    if ~isempty(idx_5pct) && ~isempty(idx_50pct) && idx_50pct > idx_5pct
        t_growth = t(idx_5pct:idx_50pct);
        X_growth = trajectories(i, idx_5pct:idx_50pct);
        X_norm = X_growth / peak_val;
        valid_idx = X_norm < 0.8 & X_norm > 0;
        
        if sum(valid_idx) > 3
            y = log(X_norm(valid_idx) ./ (1 - X_norm(valid_idx)));
            p = polyfit(t_growth(valid_idx), y, 1);
            early_growth_rates(i) = p(1);
        end
    end
end
growth_rate_init = median(early_growth_rates(early_growth_rates > 0));

% Estimate decay rate from late phase
decay_rates = zeros(N, 1);
for i = 1:N
    [peak_val, peak_idx] = max(trajectories(i, :));
    if peak_idx < length(t) - 20
        t_decay = t(peak_idx:end) - t(peak_idx);
        X_decay = trajectories(i, peak_idx:end);
        valid_idx = X_decay > 0.1*peak_val;
        
        if sum(valid_idx) > 10
            p = polyfit(t_decay(valid_idx), log(X_decay(valid_idx)), 1);
            decay_rates(i) = -p(1);
        end
    end
end
decay_rate_init = median(decay_rates(decay_rates > 0));

% Initial parameter vector
theta_init = [
    growth_rate_init;    % γ (growth rate)
    1.0;                 % H (peak height)
    0.15;                % α (mean reversion)
    decay_rate_init;     % δ (decay rate)
    0.15;                % σ_early
    0.03;                % σ_late
    40                   % τ (volatility decay constant)
];


%% Step 2: Maximum Likelihood Estimation

% Define negative log-likelihood function
nll_func = @(theta) compute_nll(theta, trajectories, t, dt);

% Set bounds for parameters
lb = [0.05; 0.5; 0.01; 0.001; 0.01; 0.001; 10];   % Lower bounds
ub = [1.0;  2.0; 0.5;  0.1;   0.3;  0.1;   100];   % Upper bounds

% Optimization options
opts = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 100, ...
                    'OptimalityTolerance', 1e-6, 'StepTolerance', 1e-6);

% Run optimization
theta_est = fmincon(nll_func, theta_init, [], [], [], [], lb, ub, [], opts);

%% Step 3: Generate trajectories with estimated parameters for validation
N_sim = 100;
simulated_trajectories = fcnGenerateStochasticTrajectories(t, theta_est, N_sim);



%% Compute goodness of fit metrics

obs_peaks = max(trajectories, [], 2);
sim_peaks = max(simulated_trajectories, [], 2);

obs_onset = computeOnsetTimes(trajectories, t);
sim_onset = computeOnsetTimes(simulated_trajectories, t);

obs_mean = mean(trajectories, 1);
obs_prc = prctile(trajectories, [5, 25, 50, 75, 95], 1);
sim_mean = mean(simulated_trajectories, 1);
sim_prc = prctile(simulated_trajectories, [5, 25, 50, 75, 95], 1);

obs_var = var(trajectories, 0, 1);
sim_var = var(simulated_trajectories, 0, 1);

% Kolmogorov-Smirnov tests
[~, p_peaks] = kstest2(obs_peaks, sim_peaks);
[~, p_onset] = kstest2(obs_onset, sim_onset);

% Mean squared error
mse_diff = (obs_mean - sim_mean).^2;
mse_mean = mean(mse_diff(isfinite(mse_diff)));
if ~isfinite(mse_mean)
    mse_mean = NaN;
end

% Relative error in variance
var_diff = abs(obs_var - sim_var) ./ (obs_var + 1e-6);
var_error = mean(var_diff(isfinite(var_diff)));
if ~isfinite(var_error)
    var_error = NaN;
end


% Store diagnostics
diagnostics.initial_estimates = theta_init;
diagnostics.p_peaks = p_peaks;
diagnostics.p_onset = p_onset;
diagnostics.mse_mean = mse_mean;
diagnostics.var_error = var_error;
if nargin > 2 && ~isempty(true_params)
    diagnostics.param_errors = param_errors;
end


end

%% Helper function for negative log-likelihood
function nll = compute_nll(theta, trajectories, t, dt)
    % Extract parameters
    gamma = theta(1);
    H = theta(2);
    alpha = theta(3);
    delta = theta(4);
    sigma_early = theta(5);
    sigma_late = theta(6);
    tau = theta(7);
    
    [N, Nt] = size(trajectories);
    nll = 0;
    
    % For each trajectory
    for i = 1:N
        X = trajectories(i, :);
        
        % For each time step (skip first)
        for j = 2:Nt
            if X(j-1) > 0  % Only count if previous state was positive
                current_time = t(j);
                
                % Compute drift
                if current_time < 30
                    growth_term = gamma * X(j-1) * (1 - X(j-1)/H);
                else
                    time_since_peak = current_time - 30;
                    decay_factor = exp(-delta * time_since_peak);
                    growth_term = -alpha * (X(j-1) - H * decay_factor * 0.2);
                end
                
                mean_reversion = -alpha * max(0, X(j-1) - H);
                drift = growth_term + mean_reversion;
                
                % Compute diffusion
                sigma_t = sigma_early * exp(-current_time/tau) + sigma_late;
                diffusion_sq = sigma_t^2 * max(0.001, X(j-1));
                
                % Expected value and variance
                mu_t = X(j-1) + drift * dt;
                var_t = diffusion_sq * dt;
                
                % Add log-likelihood (normal approximation)
                if var_t > 0 && ~isnan(X(j))
                    nll = nll - log(normpdf(X(j), mu_t, sqrt(var_t)));
                end
            end
        end
    end
    
    % Regularization to prevent extreme values
    nll = nll + 0.1 * sum(theta.^2);
end

%% Helper function to compute onset times
function onset_times = computeOnsetTimes(trajectories, t)
    [N, ~] = size(trajectories);
    onset_times = zeros(N, 1);
    
    for i = 1:N
        peak_val = max(trajectories(i, :));
        threshold_val = 0.1 * peak_val;
        idx = find(trajectories(i, :) > threshold_val, 1);
        if ~isempty(idx)
            onset_times(i) = t(idx);
        else
            onset_times(i) = NaN;
        end
    end
    
    onset_times = onset_times(~isnan(onset_times));
end