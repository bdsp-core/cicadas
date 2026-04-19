function [ke_est, results] = fcnEstimateKe_Standalone(L_obs, A_obs, varargin)
%% Standalone ke Estimation from Dose Change Dynamics
%
% Key insight: ke is most identifiable immediately after dose changes
% where the exponential decay/rise dynamics are clearest.
%
% Strategy:
% 1. Detect significant dose changes
% 2. Extract windows around dose changes
% 3. Estimate ke from the response dynamics
% 4. Aggregate estimates across multiple windows
%
% INPUTS:
%   L_obs: [N x T] observed disease intensity
%   A_obs: [N x T] drug infusion rates
%
% OUTPUTS:
%   ke_est: estimated elimination constant
%   results: detailed results structure

    % Parse inputs
    p = inputParser;
    addParameter(p, 'KeRange', [0.3, 0.9], @(x) isnumeric(x) && length(x)==2);
    addParameter(p, 'WindowSize', 10, @isnumeric);  % Time points after dose change
    addParameter(p, 'MinDoseChange', 1.0, @isnumeric);  % Minimum change to consider
    addParameter(p, 'EstimationMethod', 'robust', @ischar);  % 'robust' or 'mle'
    addParameter(p, 'AssumeC', 3.0, @isnumeric);  % Approximate C for initial estimation
    addParameter(p, 'AssumeG', 4.0, @isnumeric);  % Approximate g for initial estimation
    addParameter(p, 'Verbose', true, @islogical);
    addParameter(p, 'dt', 2, @isnumeric);  % Time step
    parse(p, varargin{:});
    
    ke_range = p.Results.KeRange;
    window_size = p.Results.WindowSize;
    min_dose_change = p.Results.MinDoseChange;
    method = p.Results.EstimationMethod;
    C_approx = p.Results.AssumeC;
    g_approx = p.Results.AssumeG;
    verbose = p.Results.Verbose;
    dt = p.Results.dt;
    
    [N, T] = size(L_obs);
    
    if verbose
        fprintf('\n========================================\n');
        fprintf('STANDALONE ke ESTIMATION\n');
        fprintf('========================================\n');
        fprintf('Method: Dose change response analysis\n');
        fprintf('Patients: %d, Time points: %d\n', N, T);
        fprintf('ke search range: [%.2f, %.2f]\n', ke_range(1), ke_range(2));
    end
    
    %% Step 1: Detect dose changes
    dose_changes = detectDoseChanges(A_obs, min_dose_change);
    
    if verbose
        total_changes = sum(dose_changes(:));
        fprintf('Detected %d significant dose changes\n', total_changes);
    end
    
    %% Step 2: Extract response windows
    windows = extractResponseWindows(L_obs, A_obs, dose_changes, window_size);
    
    if verbose
        fprintf('Extracted %d valid response windows\n', length(windows));
    end
    
    %% Step 3: Estimate ke from each window
    ke_estimates = [];
    window_weights = [];
    
    for w = 1:length(windows)
        win = windows{w};
        
        % Estimate ke for this window
        [ke_w, weight_w] = estimateKeFromWindow(win, C_approx, g_approx, ke_range, dt);
        
        if ~isnan(ke_w) && ke_w > 0
            ke_estimates(end+1) = ke_w;
            window_weights(end+1) = weight_w;
        end
    end
    
    if verbose
        fprintf('Successfully estimated ke from %d windows\n', length(ke_estimates));
    end
    
    %% Step 4: Aggregate estimates
    if strcmpi(method, 'robust')
        % Robust estimation using weighted median
        ke_est = weightedMedian(ke_estimates, window_weights);
        ke_std = mad(ke_estimates, 1) * 1.4826;  % Robust std estimate
    else
        % Maximum likelihood using weighted mean
        ke_est = sum(ke_estimates .* window_weights) / sum(window_weights);
        ke_std = std(ke_estimates);
    end
    
    %% Step 5: Confidence interval using bootstrap
    if length(ke_estimates) > 10
        % Bootstrap confidence interval
        n_boot = 1000;
        ke_boot = zeros(n_boot, 1);
        for b = 1:n_boot
            idx = randsample(length(ke_estimates), length(ke_estimates), true);
            ke_boot(b) = weightedMedian(ke_estimates(idx), window_weights(idx));
        end
        ci_95 = prctile(ke_boot, [2.5, 97.5]);
    else
        ci_95 = ke_est + [-1.96, 1.96] * ke_std;
    end
    
    %% Compile results
    results = struct();
    results.ke_est = ke_est;
    results.ke_std = ke_std;
    results.ke_ci95 = ci_95;
    results.all_estimates = ke_estimates;
    results.weights = window_weights;
    results.n_windows = length(windows);
    results.method = method;
    
    % Quality metrics
    results.cv = ke_std / ke_est;  % Coefficient of variation
    results.convergence = length(ke_estimates) / sum(dose_changes(:));  % Success rate
    
    if verbose
        fprintf('\n========================================\n');
        fprintf('RESULTS:\n');
        fprintf('========================================\n');
        fprintf('Estimated ke: %.3f ± %.3f\n', ke_est, ke_std);
        fprintf('95%% CI: [%.3f, %.3f]\n', ci_95(1), ci_95(2));
        fprintf('CV: %.2f%%\n', results.cv * 100);
        fprintf('Success rate: %.1f%%\n', results.convergence * 100);
        
        % Show distribution of estimates
        fprintf('\nDistribution of ke estimates:\n');
        fprintf('  Min: %.3f\n', min(ke_estimates));
        fprintf('  25%%: %.3f\n', prctile(ke_estimates, 25));
        fprintf('  50%%: %.3f (median)\n', median(ke_estimates));
        fprintf('  75%%: %.3f\n', prctile(ke_estimates, 75));
        fprintf('  Max: %.3f\n', max(ke_estimates));
    end
    
    % Optional: Visualize estimates
    if verbose && length(ke_estimates) > 5
        figure('Name', 'ke Estimation Results');
        
        subplot(2,2,1);
        histogram(ke_estimates, 20);
        xline(ke_est, 'r-', 'LineWidth', 2);
        xlabel('ke estimate');
        ylabel('Count');
        title('Distribution of ke Estimates');
        
        subplot(2,2,2);
        plot(ke_estimates, 'o-');
        yline(ke_est, 'r-', 'LineWidth', 2);
        xlabel('Window index');
        ylabel('ke estimate');
        title('ke Estimates by Window');
        
        subplot(2,2,3);
        scatter(window_weights, ke_estimates);
        xlabel('Window weight');
        ylabel('ke estimate');
        title('Estimates vs Weights');
        
        subplot(2,2,4);
        text(0.1, 0.9, sprintf('Final ke: %.3f', ke_est), 'FontSize', 14);
        text(0.1, 0.7, sprintf('Std: %.3f', ke_std), 'FontSize', 12);
        text(0.1, 0.5, sprintf('95%% CI: [%.3f, %.3f]', ci_95(1), ci_95(2)), 'FontSize', 12);
        text(0.1, 0.3, sprintf('N windows: %d', length(ke_estimates)), 'FontSize', 12);
        axis off;
        title('Summary');
        
        sgtitle('Standalone ke Estimation');
    end
end

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function dose_changes = detectDoseChanges(A_obs, min_change)
    % Detect significant dose changes
    [N, T] = size(A_obs);
    dose_changes = false(N, T);
    
    for i = 1:N
        for t = 2:T
            if abs(A_obs(i,t) - A_obs(i,t-1)) >= min_change
                dose_changes(i,t) = true;
            end
        end
    end
end

function windows = extractResponseWindows(L_obs, A_obs, dose_changes, window_size)
    % Extract windows around dose changes
    windows = {};
    [N, T] = size(L_obs);
    
    for i = 1:N
        change_times = find(dose_changes(i,:));
        
        for ct = change_times
            % Ensure we have enough data after the change
            if ct + window_size <= T
                % Extract window
                t_win = ct:(ct + window_size - 1);
                
                % Check for valid data
                L_win = L_obs(i, t_win);
                A_win = A_obs(i, t_win);
                
                if sum(~isnan(L_win)) >= window_size * 0.8  % At least 80% valid
                    win = struct();
                    win.L = L_win;
                    win.A = A_win;
                    win.t = t_win;
                    win.patient = i;
                    win.change_time = ct;
                    win.dose_jump = A_obs(i,ct) - A_obs(i,ct-1);
                    
                    windows{end+1} = win;
                end
            end
        end
    end
end

function [ke_est, weight] = estimateKeFromWindow(win, C, g, ke_range, dt)
    % Estimate ke from a single response window
    
    % Objective function: minimize prediction error
    obj = @(ke) computeWindowError(ke, win, C, g, dt);
    
    % Grid search for robustness
    ke_grid = linspace(ke_range(1), ke_range(2), 20);
    errors = zeros(size(ke_grid));
    
    for i = 1:length(ke_grid)
        errors(i) = obj(ke_grid(i));
    end
    
    % Find minimum
    [min_error, min_idx] = min(errors);
    ke_coarse = ke_grid(min_idx);
    
    % Refine with fminbnd
    options = optimset('Display', 'off');
    ke_est = fminbnd(obj, max(ke_range(1), ke_coarse-0.1), ...
                           min(ke_range(2), ke_coarse+0.1), options);
    
    % Weight based on fit quality and dose jump magnitude
    signal_strength = abs(win.dose_jump);
    fit_quality = 1 / (1 + min_error);
    weight = signal_strength * fit_quality;
end

function error = computeWindowError(ke, win, C, g, dt)
    % Compute prediction error for given ke
    
    L = win.L;
    A = win.A;
    T = length(L);
    
    % Simulate drug concentration with this ke
    X = zeros(1, T);
    for t = 2:T
        X(t) = ke * X(t-1) + A(t);
    end
    
    % Estimate L0 assuming this ke (simple inverse)
    L0_est = zeros(1, T);
    for t = 1:T
        if X(t) > 0
            sX = 1 - 1/((C/X(t))^g + 1);
            if sX > 0.1 && sX < 0.9
                L0_est(t) = L(t) / sX;
            else
                L0_est(t) = L(t);
            end
        else
            L0_est(t) = L(t);
        end
    end
    
    % Smooth L0 estimate
    if sum(~isnan(L0_est)) > 3
        L0_est = smooth(L0_est, 3)';
    end
    
    % Predict L using estimated L0
    L_pred = zeros(1, T);
    for t = 1:T
        if X(t) > 0
            sX = 1 - 1/((C/X(t))^g + 1);
            L_pred(t) = L0_est(t) * sX;
        else
            L_pred(t) = L0_est(t);
        end
    end
    
    % Compute error (robust MAE)
    valid = ~isnan(L) & ~isnan(L_pred) & L > 0;
    if sum(valid) > 3
        error = median(abs(L(valid) - L_pred(valid)) ./ L(valid));
    else
        error = inf;
    end
end

function wm = weightedMedian(values, weights)
    % Compute weighted median
    if isempty(values)
        wm = NaN;
        return;
    end
    
    % Sort by values
    [sorted_vals, idx] = sort(values);
    sorted_weights = weights(idx);
    
    % Normalize weights
    sorted_weights = sorted_weights / sum(sorted_weights);
    
    % Find weighted median
    cumsum_weights = cumsum(sorted_weights);
    median_idx = find(cumsum_weights >= 0.5, 1);
    
    if isempty(median_idx)
        median_idx = length(sorted_vals);
    end
    
    wm = sorted_vals(median_idx);
end