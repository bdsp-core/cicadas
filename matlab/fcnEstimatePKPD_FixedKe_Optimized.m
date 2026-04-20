function [theta_est, patient_params, L0_est, results] = fcnEstimatePKPD_FixedKe_Optimized(L_obs, A_obs, age, sofa, t, parmsL, ke_fixed, varargin)
%% State-Space Mixed Effects Model with FIXED ke
% Optimized implementation that properly handles fixed ke
%
% INPUTS:
%   L_obs, A_obs: observed data
%   age, sofa: patient covariates
%   t: time vector
%   parmsL: natural disease parameters
%   ke_fixed: FIXED value of ke (not estimated)
%
% OUTPUTS:
%   theta_est: [b0_C, b1_C, b2_C, b0_g, b1_g, b2_g, sigma_C, sigma_g]
%   Note: ke is NOT in theta_est since it's fixed

    % Parse inputs
    p = inputParser;
    addParameter(p, 'MaxIterEM', 50, @isnumeric);
    addParameter(p, 'TolEM', 1e-4, @isnumeric);
    addParameter(p, 'Verbose', true, @islogical);
    addParameter(p, 'RegularizationStrength', 2.0, @isnumeric);
    addParameter(p, 'UsePriors', true, @islogical);
    addParameter(p, 'PriorMeans', [3, 0.1, 0.15, 4, 0.08, 0.12], @isnumeric);
    addParameter(p, 'PriorStds', [0.5, 0.05, 0.05, 0.5, 0.05, 0.05], @isnumeric);
    parse(p, varargin{:});
    
    max_iter = p.Results.MaxIterEM;
    tol = p.Results.TolEM;
    verbose = p.Results.Verbose;
    reg_strength = p.Results.RegularizationStrength;
    use_priors = p.Results.UsePriors;
    prior_means = p.Results.PriorMeans;
    prior_stds = p.Results.PriorStds;
    
    % Get dimensions
    [N, T] = size(L_obs);
    dt = t(2) - t(1);
    
    % Normalize covariates
    age_norm = (age - mean(age)) / std(age);
    sofa_norm = (sofa - mean(sofa)) / std(sofa);
    
    if verbose
        fprintf('\n========================================\n');
        fprintf('FIXED ke ESTIMATION (ke = %.3f)\n', ke_fixed);
        fprintf('========================================\n');
        fprintf('Patients: %d, Time points: %d\n', N, T);
        fprintf('Parameters to estimate: 6 (b0_C, b1_C, b2_C, b0_g, b1_g, b2_g)\n');
        fprintf('Regularization: %.2f\n', reg_strength);
    end
    
    %% Initialize parameters (6 parameters + 2 variance terms)
    if use_priors
        theta = [prior_means, 0.5, 0.5]';  % 8 parameters total
    else
        theta = [3; 0.1; 0.1; 4; 0.1; 0.1; 0.5; 0.5];
    end
    
    % Initialize individual parameters
    C_indiv = theta(1) + theta(2)*age_norm + theta(3)*sofa_norm + 0.1*randn(N,1);
    g_indiv = theta(4) + theta(5)*age_norm + theta(6)*sofa_norm + 0.1*randn(N,1);
    C_indiv = max(0.5, min(10, C_indiv));
    g_indiv = max(0.5, min(10, g_indiv));
    
    % Initialize L0 using fixed ke
    L0_est = initializeL0_withFixedKe(L_obs, A_obs, C_indiv, g_indiv, ke_fixed, t);
    
    % Store history
    log_likelihood_history = [];
    theta_history = [];
    
    %% EM Algorithm with FIXED ke
    for iter = 1:max_iter
        
        if verbose && mod(iter, 5) == 0
            fprintf('\n--- EM Iteration %d (Fixed ke) ---\n', iter);
        end
        
        theta_prev = theta;
        L0_prev = L0_est;
        
        % ============ E-STEP: Update L0 with FIXED ke ============
        log_lik = 0;
        
        for i = 1:N
            % Run EKF with FIXED ke
            [L0_est(i,:), ~, lik_i] = runEKF_fixedKe(...
                L_obs(i,:), A_obs(i,:), C_indiv(i), g_indiv(i), ke_fixed, dt);
            
            log_lik = log_lik + lik_i;
        end
        
        % Add prior likelihood
        if use_priors
            prior_lik = computePriorLikelihood(theta(1:6), prior_means, prior_stds);
            log_lik = log_lik + reg_strength * prior_lik;
        end
        
        log_likelihood_history(iter) = log_lik;
        
        % ============ M-STEP: Update C and g parameters only ============
        
        % Step 1: Update individual C and g with FIXED ke
        for i = 1:N
            if max(A_obs(i,:)) > 0
                % Expected values from population
                C_expected = theta(1) + theta(2)*age_norm(i) + theta(3)*sofa_norm(i);
                g_expected = theta(4) + theta(5)*age_norm(i) + theta(6)*sofa_norm(i);
                
                % Optimize C and g for patient i with FIXED ke
                params_i = optimizeIndividual_fixedKe(...
                    L_obs(i,:), A_obs(i,:), L0_est(i,:), ke_fixed, dt, ...
                    [C_expected, g_expected], reg_strength);
                
                C_indiv(i) = params_i(1);
                g_indiv(i) = params_i(2);
            end
        end
        
        % Step 2: Update population parameters
        valid_idx = C_indiv > 0 & g_indiv > 0 & ~isnan(C_indiv) & ~isnan(g_indiv);
        
        if sum(valid_idx) > 10
            X = [ones(sum(valid_idx), 1), age_norm(valid_idx), sofa_norm(valid_idx)];
            
            % Ridge regression for C
            lambda_C = reg_strength * 10;
            beta_C = (X'*X + lambda_C*eye(3)) \ (X'*C_indiv(valid_idx) + lambda_C*prior_means(1:3)');
            
            % Smooth update
            momentum = 0.7;
            theta(1:3) = momentum * theta(1:3) + (1-momentum) * beta_C;
            
            % Variance
            C_pred = X * theta(1:3);
            theta(7) = sqrt(mean((C_indiv(valid_idx) - C_pred).^2));
            
            % Ridge regression for g
            lambda_g = reg_strength * 10;
            beta_g = (X'*X + lambda_g*eye(3)) \ (X'*g_indiv(valid_idx) + lambda_g*prior_means(4:6)');
            
            theta(4:6) = momentum * theta(4:6) + (1-momentum) * beta_g;
            
            g_pred = X * theta(4:6);
            theta(8) = sqrt(mean((g_indiv(valid_idx) - g_pred).^2));
        end
        
        % Update individual parameters with shrinkage
        C_pop = theta(1) + theta(2)*age_norm + theta(3)*sofa_norm;
        g_pop = theta(4) + theta(5)*age_norm + theta(6)*sofa_norm;
        
        shrinkage = 0.2;
        C_indiv = (1-shrinkage)*C_indiv + shrinkage*C_pop;
        g_indiv = (1-shrinkage)*g_indiv + shrinkage*g_pop;
        
        % Ensure bounds
        C_indiv = max(0.5, min(10, C_indiv));
        g_indiv = max(0.5, min(10, g_indiv));
        
        theta_history(:, iter) = theta;
        
        % ============ Check Convergence ============
        if iter > 1
            param_change = norm(theta - theta_prev) / (norm(theta) + 1e-10);
            L0_change = norm(L0_est - L0_prev, 'fro') / (norm(L0_est, 'fro') + 1e-10);
            
            if verbose && mod(iter, 5) == 0
                fprintf('  Parameter change: %.6f\n', param_change);
                fprintf('  L0 change: %.6f\n', L0_change);
                fprintf('  Log-likelihood: %.2f\n', log_lik);
            end
            
            if param_change < tol && L0_change < tol
                if verbose
                    fprintf('\nConverged after %d iterations\n', iter);
                end
                break;
            end
        end
    end
    
    %% Compile results (Note: theta_est does NOT include ke)
    theta_est = theta;  % 8 parameters: 6 regression + 2 variances
    
    patient_params = struct();
    patient_params.C_indiv = C_indiv;
    patient_params.g_indiv = g_indiv;
    patient_params.C_pred = theta(1) + theta(2)*age_norm + theta(3)*sofa_norm;
    patient_params.g_pred = theta(4) + theta(5)*age_norm + theta(6)*sofa_norm;
    patient_params.ke_fixed = ke_fixed;  % Store the fixed ke value
    
    results = struct();
    results.iterations = iter;
    results.log_likelihood_history = log_likelihood_history;
    results.theta_history = theta_history;
    results.converged = iter < max_iter;
    results.ke_was_fixed = true;
    results.ke_value = ke_fixed;
    
    % Calculate R²
    valid_C = C_indiv > 0 & ~isnan(C_indiv);
    valid_g = g_indiv > 0 & ~isnan(g_indiv);
    
    if sum(valid_C) > 10
        results.R2_C = 1 - sum((C_indiv(valid_C) - patient_params.C_pred(valid_C)).^2) / ...
                           sum((C_indiv(valid_C) - mean(C_indiv(valid_C))).^2);
    else
        results.R2_C = NaN;
    end
    
    if sum(valid_g) > 10
        results.R2_g = 1 - sum((g_indiv(valid_g) - patient_params.g_pred(valid_g)).^2) / ...
                           sum((g_indiv(valid_g) - mean(g_indiv(valid_g))).^2);
    else
        results.R2_g = NaN;
    end
    
    if verbose
        fprintf('\n========================================\n');
        fprintf('FINAL ESTIMATES (with FIXED ke = %.3f):\n', ke_fixed);
        fprintf('C = %.3f + %.3f*age + %.3f*sofa (σ=%.3f)\n', ...
            theta(1), theta(2), theta(3), theta(7));
        fprintf('g = %.3f + %.3f*age + %.3f*sofa (σ=%.3f)\n', ...
            theta(4), theta(5), theta(6), theta(8));
        fprintf('R² for C: %.3f, R² for g: %.3f\n', results.R2_C, results.R2_g);
        fprintf('========================================\n');
    end
end

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function L0_init = initializeL0_withFixedKe(L_obs, A_obs, C, g, ke, t)
    % Initialize L0 using inverse treatment effect with FIXED ke
    [N, T] = size(L_obs);
    dt = t(2) - t(1);
    L0_init = L_obs;
    
    for i = 1:N
        X = zeros(1, T);
        
        % Calculate drug concentration with FIXED ke
        for j = 2:T
            X(j) = ke * X(j-1) + A_obs(i, j);
        end
        
        % Invert treatment effect
        for j = 1:T
            if X(j) > 0 && ~isnan(L_obs(i, j))
                sX = 1 - 1/((C(i)/X(j))^g(i) + 1);
                if sX > 0.1 && sX < 0.9  % Avoid extreme values
                    L0_init(i, j) = L_obs(i, j) / sX;
                else
                    L0_init(i, j) = L_obs(i, j);
                end
            end
        end
        
        % Smooth and interpolate
        valid_idx = ~isnan(L0_init(i, :)) & L0_init(i, :) > 0;
        if sum(valid_idx) > 5
            L0_init(i, :) = smooth(L0_init(i, :), 5, 'rloess')';
        end
    end
    
    % Ensure reasonable bounds
    L0_init = max(0, min(5, L0_init));
end

function [L0_smooth, P_smooth, log_lik] = runEKF_fixedKe(L_i, A_i, C, g, ke, dt)
    % EKF with FIXED ke
    T = length(L_i);
    
    % Initialize
    L0_smooth = zeros(1, T);
    L0_smooth(1) = L_i(1);
    
    % Process and measurement noise
    Q = 0.01;
    R = 0.05;
    
    % Drug concentration with FIXED ke
    X = zeros(1, T);
    for t = 2:T
        X(t) = ke * X(t-1) + A_i(t);
    end
    
    log_lik = 0;
    
    % Forward pass
    L0_filt = L0_smooth;
    P_filt = 0.1 * ones(1, T);
    
    for t = 2:T
        % Predict
        L0_pred = L0_filt(t-1);
        P_pred = P_filt(t-1) + Q;
        
        if ~isnan(L_i(t))
            % Observation model
            if X(t-1) > 0
                sX = 1 - 1/((C/X(t-1))^g + 1);
                H = sX;
            else
                sX = 1;
                H = 1;
            end
            
            % Innovation
            y_pred = L0_pred * sX;
            innovation = L_i(t) - y_pred;
            
            % Kalman update
            S = H * P_pred * H + R;
            K = P_pred * H / S;
            
            L0_filt(t) = L0_pred + K * innovation;
            P_filt(t) = (1 - K * H) * P_pred;
            
            % Log-likelihood
            if S > 0
                log_lik = log_lik - 0.5 * (log(2*pi*S) + innovation^2/S);
            end
        else
            L0_filt(t) = L0_pred;
            P_filt(t) = P_pred;
        end
    end
    
    % Smoothing
    L0_smooth = smooth(L0_filt, 5)';
    L0_smooth = max(0, L0_smooth);
    P_smooth = P_filt;
end

function params = optimizeIndividual_fixedKe(L_i, A_i, L0_i, ke, dt, expected_params, reg_weight)
    % Optimize C and g with FIXED ke
    
    % Objective function
    obj = @(p) individualNLL_fixedKe(p, L_i, A_i, L0_i, ke, expected_params, reg_weight);
    
    % Initial guess (start from expected)
    params_init = expected_params;
    
    % Bounds
    lb = [0.5; 0.5];
    ub = [10; 10];
    
    % Optimize
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
    params = fmincon(obj, params_init, [], [], [], [], lb, ub, [], options);
end

function nll = individualNLL_fixedKe(params, L_i, A_i, L0_i, ke, expected_params, reg_weight)
    % NLL with FIXED ke
    C = params(1);
    g = params(2);
    
    T = length(L_i);
    
    % Calculate drug concentration with FIXED ke
    X = zeros(1, T);
    for t = 2:T
        X(t) = ke * X(t-1) + A_i(t);
    end
    
    nll = 0;
    n_obs = 0;
    
    for t = 2:T
        if ~isnan(L_i(t)) && L0_i(t) > 0 && X(t-1) >= 0
            if X(t-1) > 0
                sX = 1 - 1/((C/X(t-1))^g + 1);
                L_pred = L0_i(t) * sX;
            else
                L_pred = L0_i(t);
            end
            
            % Variance model
            var_L = 0.01 + 0.05 * abs(L_pred);
            
            if var_L > 0 && L_pred >= 0
                nll = nll + 0.5 * log(2*pi*var_L) + 0.5 * (L_i(t) - L_pred)^2 / var_L;
                n_obs = n_obs + 1;
            end
        end
    end
    
    % Normalize by number of observations
    if n_obs > 0
        nll = nll / n_obs;
    end
    
    % L2 regularization
    nll = nll + reg_weight * sum((params - expected_params).^2);
end

function prior_lik = computePriorLikelihood(theta, prior_means, prior_stds)
    % Gaussian prior likelihood
    z = (theta - prior_means') ./ prior_stds';
    prior_lik = -0.5 * sum(z.^2);
end