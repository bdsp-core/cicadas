function [theta_est, patient_params, L0_est, results] = fcnEstimatePKPD_StateSpaceMixedEffects_v2(L_obs, A_obs, age, sofa, t, parmsL, varargin)
%% State-Space Mixed Effects Model with Enhanced Regularization
%
% IMPROVEMENTS:
% 1. Stronger priors on population parameters
% 2. Adaptive regularization based on data quality
% 3. Improved EKF with better process/measurement noise models
% 4. Hierarchical Bayesian-inspired regularization
% 5. Constraint on parameter changes between iterations

    % Parse inputs
    p = inputParser;
    addParameter(p, 'MaxIterEM', 50, @isnumeric);
    addParameter(p, 'TolEM', 1e-4, @isnumeric);
    addParameter(p, 'Verbose', true, @islogical);
    addParameter(p, 'InitMethod', 'informed', @ischar);
    addParameter(p, 'RegularizationStrength', 1.0, @isnumeric);
    addParameter(p, 'UsePriors', true, @islogical);
    addParameter(p, 'PriorMeans', [3, 0.1, 0.15, 4, 0.08, 0.12, 0.5], @isnumeric);
    addParameter(p, 'PriorStds', [1, 0.05, 0.05, 1, 0.05, 0.05, 0.2], @isnumeric);
    parse(p, varargin{:});
    
    max_iter = p.Results.MaxIterEM;
    tol = p.Results.TolEM;
    verbose = p.Results.Verbose;
    init_method = p.Results.InitMethod;
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
        fprintf('ENHANCED STATE-SPACE MIXED EFFECTS\n');
        fprintf('========================================\n');
        fprintf('Patients: %d, Time points: %d\n', N, T);
        fprintf('Regularization strength: %.2f\n', reg_strength);
        fprintf('Using priors: %s\n', string(use_priors));
    end
    
    %% Initialize with informed priors
    if use_priors
        theta = [prior_means, 0.5, 0.5]';  % Add variance terms
        if verbose
            fprintf('Initialized with prior means\n');
        end
    else
        theta = [3; 0.1; 0.1; 4; 0.1; 0.1; 0.5; 0.5; 0.5];
    end
    
    % Initialize individual parameters close to population
    C_indiv = theta(1) + theta(2)*age_norm + theta(3)*sofa_norm + 0.1*randn(N,1);
    g_indiv = theta(4) + theta(5)*age_norm + theta(6)*sofa_norm + 0.1*randn(N,1);
    C_indiv = max(0.5, min(10, C_indiv));  % Bounded
    g_indiv = max(0.5, min(10, g_indiv));
    
    % Initialize L0 with smoothed observed L
    L0_est = initializeL0_improved(L_obs, A_obs, C_indiv, g_indiv, theta(7), parmsL, t);
    
    % Store history
    log_likelihood_history = [];
    theta_history = [];
    
    % Adaptive regularization weights
    data_quality = assessDataQuality(L_obs, A_obs);
    reg_weights = computeRegularizationWeights(data_quality, reg_strength);
    
    %% EM Algorithm with regularization
    for iter = 1:max_iter
        
        if verbose && mod(iter, 5) == 0
            fprintf('\n--- EM Iteration %d ---\n', iter);
        end
        
        theta_prev = theta;
        L0_prev = L0_est;
        
        % ============ E-STEP: Enhanced EKF ============
        log_lik = 0;
        
        for i = 1:N
            % Adaptive noise parameters based on data quality
            process_noise = 0.01 * (1 + 2*(1 - data_quality(i)));
            measurement_noise = 0.05 * (1 + 3*(1 - data_quality(i)));
            
            % Run improved EKF
            [L0_est(i,:), P_smooth, lik_i] = runEKF_improved(...
                L_obs(i,:), A_obs(i,:), C_indiv(i), g_indiv(i), ...
                theta(7), parmsL, dt, process_noise, measurement_noise);
            
            log_lik = log_lik + lik_i;
        end
        
        % Add prior likelihood if using priors
        if use_priors
            prior_lik = computePriorLikelihood(theta(1:7), prior_means, prior_stds);
            log_lik = log_lik + reg_strength * prior_lik;
        end
        
        log_likelihood_history(iter) = log_lik;
        
        % ============ M-STEP: Regularized parameter update ============
        
        % Step 1: Update individual parameters with L2 regularization
        for i = 1:N
            if max(A_obs(i,:)) > 0
                % Expected values from population model
                C_expected = theta(1) + theta(2)*age_norm(i) + theta(3)*sofa_norm(i);
                g_expected = theta(4) + theta(5)*age_norm(i) + theta(6)*sofa_norm(i);
                
                % Optimize with regularization toward population
                params_i = optimizeIndividualParams_regularized(...
                    L_obs(i,:), A_obs(i,:), L0_est(i,:), theta(7), dt, ...
                    [C_expected, g_expected], reg_weights(i));
                
                C_indiv(i) = params_i(1);
                g_indiv(i) = params_i(2);
            end
        end
        
        % Step 2: Update population parameters with ridge regression
        valid_idx = ~isnan(C_indiv) & C_indiv > 0;
        if sum(valid_idx) > 10
            % Ridge regression for C
            X = [ones(sum(valid_idx), 1), age_norm(valid_idx), sofa_norm(valid_idx)];
            lambda_C = reg_strength * 10;  % Ridge parameter
            beta_C = (X'*X + lambda_C*eye(3)) \ (X'*C_indiv(valid_idx) + lambda_C*prior_means(1:3)');
            
            % Smooth update (momentum)
            momentum = 0.7;
            theta(1:3) = momentum * theta(1:3) + (1-momentum) * beta_C;
            
            % Variance
            C_pred = X * theta(1:3);
            theta(8) = sqrt(mean((C_indiv(valid_idx) - C_pred).^2));
        end
        
        % Ridge regression for g
        valid_idx = ~isnan(g_indiv) & g_indiv > 0;
        if sum(valid_idx) > 10
            X = [ones(sum(valid_idx), 1), age_norm(valid_idx), sofa_norm(valid_idx)];
            lambda_g = reg_strength * 10;
            beta_g = (X'*X + lambda_g*eye(3)) \ (X'*g_indiv(valid_idx) + lambda_g*prior_means(4:6)');
            
            theta(4:6) = momentum * theta(4:6) + (1-momentum) * beta_g;
            
            g_pred = X * theta(4:6);
            theta(9) = sqrt(mean((g_indiv(valid_idx) - g_pred).^2));
        end
        
        % Step 3: Update ke with regularization
        ke_new = optimizeKe_regularized(L_obs, A_obs, L0_est, C_indiv, g_indiv, dt, ...
            prior_means(7), reg_strength);
        theta(7) = momentum * theta(7) + (1-momentum) * ke_new;
        
        % Update individual parameters based on new population model
        C_indiv_new = theta(1) + theta(2)*age_norm + theta(3)*sofa_norm;
        g_indiv_new = theta(4) + theta(5)*age_norm + theta(6)*sofa_norm;
        
        % Shrinkage toward population mean
        shrinkage = 0.3;
        C_indiv = (1-shrinkage)*C_indiv + shrinkage*C_indiv_new;
        g_indiv = (1-shrinkage)*g_indiv + shrinkage*g_indiv_new;
        
        % Ensure bounds
        C_indiv = max(0.5, min(10, C_indiv));
        g_indiv = max(0.5, min(10, g_indiv));
        
        theta_history(:, iter) = theta;
        
        % ============ Check Convergence ============
        if iter > 1
            param_change = norm(theta - theta_prev) / norm(theta);
            L0_change = norm(L0_est - L0_prev, 'fro') / norm(L0_est, 'fro');
            
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
    
    %% Compile results
    theta_est = theta;
    
    patient_params = struct();
    patient_params.C_indiv = C_indiv;
    patient_params.g_indiv = g_indiv;
    patient_params.C_pred = theta(1) + theta(2)*age_norm + theta(3)*sofa_norm;
    patient_params.g_pred = theta(4) + theta(5)*age_norm + theta(6)*sofa_norm;
    
    results = struct();
    results.iterations = iter;
    results.log_likelihood_history = log_likelihood_history;
    results.theta_history = theta_history;
    results.converged = iter < max_iter;
    results.data_quality = data_quality;
    
    % Calculate R²
    valid_C = ~isnan(C_indiv) & C_indiv > 0;
    valid_g = ~isnan(g_indiv) & g_indiv > 0;
    
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
        fprintf('FINAL ESTIMATES:\n');
        fprintf('C = %.3f + %.3f*age + %.3f*sofa (σ=%.3f)\n', ...
            theta(1), theta(2), theta(3), theta(8));
        fprintf('g = %.3f + %.3f*age + %.3f*sofa (σ=%.3f)\n', ...
            theta(4), theta(5), theta(6), theta(9));
        fprintf('ke = %.3f\n', theta(7));
        fprintf('R² for C: %.3f, R² for g: %.3f\n', results.R2_C, results.R2_g);
        fprintf('========================================\n');
    end
end

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function data_quality = assessDataQuality(L_obs, A_obs)
    % Assess data quality for each patient (0 to 1, higher is better)
    [N, T] = size(L_obs);
    data_quality = zeros(N, 1);
    
    for i = 1:N
        % Factors: data completeness, dose variation, signal-to-noise
        completeness = sum(~isnan(L_obs(i,:))) / T;
        
        % Dose variation (coefficient of variation)
        A_nonzero = A_obs(i, A_obs(i,:) > 0);
        if ~isempty(A_nonzero)
            dose_cv = std(A_nonzero) / mean(A_nonzero);
        else
            dose_cv = 0;
        end
        
        % Signal quality (autocorrelation)
        valid_L = L_obs(i, ~isnan(L_obs(i,:)));
        if length(valid_L) > 10
            autocorr_val = abs(corr(valid_L(1:end-1)', valid_L(2:end)'));
        else
            autocorr_val = 0;
        end
        
        data_quality(i) = completeness * (0.5 + 0.5*min(dose_cv, 1)) * autocorr_val;
    end
    
    % Normalize to [0.1, 1]
    data_quality = 0.1 + 0.9 * (data_quality - min(data_quality)) / (max(data_quality) - min(data_quality) + eps);
end

function reg_weights = computeRegularizationWeights(data_quality, base_strength)
    % Higher regularization for lower quality data
    reg_weights = base_strength ./ (data_quality + 0.1);
    reg_weights = min(10, reg_weights);  % Cap maximum regularization
end

function prior_lik = computePriorLikelihood(theta, prior_means, prior_stds)
    % Gaussian prior likelihood
    z = (theta - prior_means') ./ prior_stds';
    prior_lik = -0.5 * sum(z.^2);
end

function L0_init = initializeL0_improved(L_obs, A_obs, C, g, ke, parmsL, t)
    % Improved L0 initialization using smoothing and interpolation
    [N, T] = size(L_obs);
    L0_init = L_obs;
    
    for i = 1:N
        % First pass: inverse treatment effect
        X = zeros(1, T);
        for j = 2:T
            X(j) = ke * X(j-1) + A_obs(i, j);
            
            if X(j-1) > 0 && ~isnan(L_obs(i, j))
                sX = 1 - 1/((C(i)/X(j-1))^g(i) + 1);
                if sX > 0.1
                    L0_init(i, j) = L_obs(i, j) / sX;
                end
            end
        end
        
        % Fill NaN with interpolation
        valid_idx = ~isnan(L0_init(i, :));
        if sum(valid_idx) > 3
            t_valid = find(valid_idx);
            L0_init(i, ~valid_idx) = interp1(t_valid, L0_init(i, valid_idx), ...
                find(~valid_idx), 'linear', 'extrap');
        end
        
        % Smooth with robust smoothing
        L0_init(i, :) = smooth(L0_init(i, :), 7, 'rloess')';
        
        % Ensure reasonable bounds based on parmsL
        peak_height = parmsL(2);
        L0_init(i, :) = max(0, min(2*peak_height, L0_init(i, :)));
    end
end

function [L0_smooth, P_smooth, log_lik] = runEKF_improved(L_i, A_i, C, g, ke, parmsL, dt, Q, R)
    % Improved Extended Kalman Filter with adaptive noise
    
    T = length(L_i);
    
    % Initialize
    L0_smooth = zeros(1, T);
    P_smooth = zeros(1, T);
    
    % Initial state (use first observation if no treatment)
    if A_i(1) == 0
        L0_smooth(1) = L_i(1);
    else
        L0_smooth(1) = parmsL(2) * 0.5;  % Half of peak height
    end
    P_smooth(1) = 0.1;
    
    % Drug concentration
    X = zeros(1, T);
    
    log_lik = 0;
    
    % Forward pass
    L0_filt = zeros(1, T);
    P_filt = zeros(1, T);
    L0_filt(1) = L0_smooth(1);
    P_filt(1) = P_smooth(1);
    
    for t = 2:T
        % Update drug concentration
        X(t) = ke * X(t-1) + A_i(t);
        
        % Predict step with simple dynamics model
        % Assume slow evolution of L0
        L0_pred = L0_filt(t-1) * (1 - 0.01*dt) + 0.01*parmsL(2)*dt;
        P_pred = P_filt(t-1) + Q;
        
        if ~isnan(L_i(t))
            % Observation model
            if X(t-1) > 0
                sX = 1 - 1/((C/X(t-1))^g + 1);
                H = sX;  % Jacobian
            else
                sX = 1;
                H = 1;
            end
            
            % Innovation
            y_pred = L0_pred * sX;
            innovation = L_i(t) - y_pred;
            
            % Adaptive measurement noise based on innovation
            R_adaptive = R * (1 + abs(innovation)/(abs(y_pred) + 0.1));
            
            % Innovation covariance
            S = H * P_pred * H + R_adaptive;
            
            % Kalman gain
            K = P_pred * H / S;
            
            % Update
            L0_filt(t) = L0_pred + K * innovation;
            P_filt(t) = (1 - K * H) * P_pred;
            
            % Log-likelihood
            log_lik = log_lik - 0.5 * (log(2*pi*S) + innovation^2/S);
        else
            L0_filt(t) = L0_pred;
            P_filt(t) = P_pred;
        end
    end
    
    % Backward smoothing pass
    L0_smooth = L0_filt;
    P_smooth = P_filt;
    
    for t = T-1:-1:2
        if P_filt(t) > 0 && P_filt(t+1) > 0
            % Smoothing gain
            A_smooth = P_filt(t) / (P_filt(t) + Q);
            
            % Smooth estimates
            L0_smooth(t) = L0_filt(t) + A_smooth * (L0_smooth(t+1) - L0_filt(t));
            P_smooth(t) = P_filt(t) + A_smooth^2 * (P_smooth(t+1) - P_filt(t) - Q);
        end
    end
    
    % Final bounds check
    L0_smooth = max(0, L0_smooth);
end

function params = optimizeIndividualParams_regularized(L_i, A_i, L0_i, ke, dt, expected_params, reg_weight)
    % Optimize with regularization toward expected values
    
    T = length(L_i);
    
    % Objective with L2 regularization
    obj = @(p) individualNLL_regularized(p, L_i, A_i, L0_i, ke, expected_params, reg_weight);
    
    % Start from expected values
    params_init = expected_params;
    
    % Tighter bounds
    lb = [0.5; 0.5];
    ub = [10; 10];
    
    % Optimize
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
        'MaxIterations', 100);
    params = fmincon(obj, params_init, [], [], [], [], lb, ub, [], options);
end

function nll = individualNLL_regularized(params, L_i, A_i, L0_i, ke, expected_params, reg_weight)
    % NLL with regularization
    
    C = params(1);
    g = params(2);
    
    T = length(L_i);
    X = zeros(1, T);
    nll = 0;
    
    for t = 2:T
        X(t) = ke * X(t-1) + A_i(t);
        
        if ~isnan(L_i(t)) && L0_i(t) > 0
            if X(t-1) > 0
                sX = 1 - 1/((C/X(t-1))^g + 1);
                L_pred = L0_i(t) * sX;
            else
                L_pred = L0_i(t);
            end
            
            var_L = 0.01 + 0.05 * L_pred;
            
            if var_L > 0 && L_pred >= 0
                nll = nll + 0.5 * log(2*pi*var_L) + 0.5 * (L_i(t) - L_pred)^2 / var_L;
            end
        end
    end
    
    % L2 regularization toward expected values
    nll = nll + reg_weight * sum((params - expected_params).^2);
end

function ke_opt = optimizeKe_regularized(L_obs, A_obs, L0_est, C_indiv, g_indiv, dt, prior_ke, reg_strength)
    % Optimize ke with regularization toward prior
    
    [N, T] = size(L_obs);
    
    % Objective with regularization
    obj = @(ke) totalNLL_ke_regularized(ke, L_obs, A_obs, L0_est, C_indiv, g_indiv, prior_ke, reg_strength);
    
    % Start from prior
    ke_init = prior_ke;
    lb = 0.1;
    ub = 1.0;
    
    options = optimoptions('fmincon', 'Display', 'off');
    ke_opt = fmincon(obj, ke_init, [], [], [], [], lb, ub, [], options);
end

function nll = totalNLL_ke_regularized(ke, L_obs, A_obs, L0_est, C_indiv, g_indiv, prior_ke, reg_strength)
    % Total NLL with regularization
    
    [N, T] = size(L_obs);
    nll = 0;
    
    for i = 1:N
        if max(A_obs(i,:)) > 0
            X = zeros(1, T);
            for t = 2:T
                X(t) = ke * X(t-1) + A_obs(i, t);
                
                if ~isnan(L_obs(i, t)) && L0_est(i, t) > 0
                    if X(t-1) > 0
                        sX = 1 - 1/((C_indiv(i)/X(t-1))^g_indiv(i) + 1);
                        L_pred = L0_est(i, t) * sX;
                    else
                        L_pred = L0_est(i, t);
                    end
                    
                    var_L = 0.01 + 0.05 * L_pred;
                    
                    if var_L > 0 && L_pred >= 0
                        nll = nll + 0.5 * log(2*pi*var_L) + 0.5 * (L_obs(i, t) - L_pred)^2 / var_L;
                    end
                end
            end
        end
    end
    
    % Regularization toward prior
    nll = nll + reg_strength * 100 * (ke - prior_ke)^2;
end