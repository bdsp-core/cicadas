%% Comprehensive Test: Fixed vs Unfixed ke
% Proper comparison with optimized implementations

clear; clc; close all;

fprintf('==========================================================\n');
fprintf('COMPREHENSIVE FIXED vs UNFIXED ke COMPARISON\n');
fprintf('==========================================================\n\n');

%% Load data
T = readtable('trialDataDoseChanging.csv');
load('parmsTrue_DoseChanging.mat', 'parmsPD', 'C', 'g', 'ke', 'age', 'sofa', 'parmsL');
load('L0data.mat', 'L0');
L0_true = L0;

% Prepare data
unique_patients = unique(T.sid);
N = length(unique_patients);
dt = 2;
t = 0:dt:168;
T_len = length(t);

L_obs = NaN(N, T_len);
A_obs = zeros(N, T_len);
patient_age = zeros(N, 1);
patient_sofa = zeros(N, 1);

for i = 1:N
    patient_id = unique_patients(i);
    patient_data = T(T.sid == patient_id, :);
    patient_data = sortrows(patient_data, 't');
    time_idx = patient_data.t / dt + 1;
    L_obs(i, time_idx) = patient_data.L;
    A_obs(i, time_idx) = patient_data.A;
    patient_age(i) = patient_data.age(1);
    patient_sofa(i) = patient_data.sofa(1);
end

true_params = [parmsPD, ke];
param_names = {'b0_C', 'b1_C', 'b2_C', 'b0_g', 'b1_g', 'b2_g', 'ke'};

fprintf('Data: %d patients with random dose changes\n', N);
fprintf('True ke = %.3f\n\n', ke);

%% Test different scenarios
scenarios = {
    'Unfixed ke',
    'Fixed ke (true value)',
    'Fixed ke (0.9*true)',
    'Fixed ke (1.1*true)'
};

ke_values = [NaN, ke, 0.9*ke, 1.1*ke];
results_all = cell(length(scenarios), 1);
theta_all = cell(length(scenarios), 1);
L0_all = cell(length(scenarios), 1);
times = zeros(length(scenarios), 1);

for s = 1:length(scenarios)
    fprintf('----------------------------------------------------------\n');
    fprintf('SCENARIO %d: %s\n', s, scenarios{s});
    fprintf('----------------------------------------------------------\n');
    
    tic;
    
    if s == 1  % Unfixed ke
        fprintf('Estimating all 7 parameters...\n');
        
        [theta_est, patient_params, L0_est, results] = ...
            fcnEstimatePKPD_StateSpaceMixedEffects_v2(...
            L_obs, A_obs, patient_age, patient_sofa, t, parmsL, ...
            'RegularizationStrength', 5.0, ...
            'UsePriors', true, ...
            'PriorMeans', true_params, ...
            'PriorStds', [0.5, 0.05, 0.05, 0.5, 0.05, 0.05, 0.1], ...
            'MaxIterEM', 30, ...
            'Verbose', false);
        
        % Store all 7 parameters
        theta_full = theta_est(1:7);
        
    else  % Fixed ke
        ke_fixed = ke_values(s);
        fprintf('Using FIXED ke = %.3f\n', ke_fixed);
        fprintf('Estimating 6 parameters (C and g coefficients)...\n');
        
        % Prior means WITHOUT ke
        prior_means_no_ke = true_params(1:6);
        
        [theta_est, patient_params, L0_est, results] = ...
            fcnEstimatePKPD_FixedKe_Optimized(...
            L_obs, A_obs, patient_age, patient_sofa, t, parmsL, ke_fixed, ...
            'RegularizationStrength', 5.0, ...
            'UsePriors', true, ...
            'PriorMeans', prior_means_no_ke, ...
            'PriorStds', [0.5, 0.05, 0.05, 0.5, 0.05, 0.05], ...
            'MaxIterEM', 30, ...
            'Verbose', false);
        
        % Add fixed ke to make 7 parameters for comparison
        theta_full = [theta_est(1:6); ke_fixed];
    end
    
    times(s) = toc;
    
    % Store results
    results_all{s} = results;
    theta_all{s} = theta_full;
    L0_all{s} = L0_est;
    
    % Calculate metrics
    errors = 100 * (theta_full' - true_params) ./ abs(true_params);
    [corr_val, rmse_val, mae_val] = computeL0Metrics(L0_est, L0_true);
    
    % Display results
    fprintf('\nResults:\n');
    fprintf('  Time: %.1f seconds\n', times(s));
    fprintf('  Converged: %s\n', string(results.converged));
    if isfield(results, 'R2_C')
        fprintf('  R² for C: %.3f, R² for g: %.3f\n', results.R2_C, results.R2_g);
    end
    
    fprintf('\nParameter estimates:\n');
    fprintf('  %-10s %8s %8s %8s\n', 'Param', 'True', 'Est', 'Error(%)');
    for p = 1:7
        if p == 7 && s > 1  % ke was fixed
            fprintf('  %-10s %8.3f %8.3f %8s\n', param_names{p}, ...
                true_params(p), theta_full(p), '(fixed)');
        else
            fprintf('  %-10s %8.3f %8.3f %8.1f\n', param_names{p}, ...
                true_params(p), theta_full(p), errors(p));
        end
    end
    
    fprintf('\nPerformance metrics:\n');
    if s == 1
        fprintf('  MAPE (all params): %.1f%%\n', mean(abs(errors)));
    else
        fprintf('  MAPE (C,g params only): %.1f%%\n', mean(abs(errors(1:6))));
    end
    fprintf('  L0 correlation: %.3f\n', corr_val);
    fprintf('  L0 RMSE: %.3f\n', rmse_val);
end

%% Visualization
figure('Position', [100, 100, 1400, 900]);

% Subplot 1: Parameter estimates
subplot(2, 4, 1);
bar_data = [true_params(1:6)];
for s = 1:length(scenarios)
    bar_data = [bar_data; theta_all{s}(1:6)'];
end
b = bar(bar_data');
set(gca, 'XTickLabel', param_names(1:6));
ylabel('Parameter Value');
title('C and g Parameter Estimates');
legend(b, ["True"; string(scenarios)], 'Location','best','FontSize',8);
grid on;

% Subplot 2: ke estimates
subplot(2, 4, 2);
ke_estimates = zeros(length(scenarios), 1);
for s = 1:length(scenarios)
    ke_estimates(s) = theta_all{s}(7);
end
bar(ke_estimates);
hold on;
yline(ke, 'r--', 'LineWidth', 2);
set(gca, 'XTickLabel', scenarios);
xtickangle(45);
ylabel('ke Value');
title('ke Estimates/Values');
grid on;

% Subplot 3: MAPE comparison
subplot(2, 4, 3);
mape_values = zeros(length(scenarios), 2);  % All params, C&g only
for s = 1:length(scenarios)
    errors_all = 100 * abs(theta_all{s}' - true_params) ./ abs(true_params);
    mape_values(s, 1) = mean(errors_all);
    mape_values(s, 2) = mean(errors_all(1:6));
end
bar(mape_values);
set(gca, 'XTickLabel', scenarios);
xtickangle(45);
ylabel('MAPE (%)');
title('Estimation Errors');
legend({'All params', 'C&g only'}, 'Location', 'best');
grid on;

% Subplot 4: L0 correlation
subplot(2, 4, 4);
corr_values = zeros(length(scenarios), 1);
for s = 1:length(scenarios)
    [corr_val, ~, ~] = computeL0Metrics(L0_all{s}, L0_true);
    corr_values(s) = corr_val;
end
bar(corr_values);
set(gca, 'XTickLabel', scenarios);
xtickangle(45);
ylabel('L0 Correlation');
title('L0 Recovery Quality');
ylim([0, 1]);
grid on;

% Subplots 5-8: L0 recovery examples
for s = 1:min(4, length(scenarios))
    subplot(2, 4, 4+s);
    patient_idx = 1;
    plot(t, L0_true(patient_idx,:), 'g-', 'LineWidth', 2);
    hold on;
    plot(t, L0_all{s}(patient_idx,:), 'b--', 'LineWidth', 1.5);
    xlabel('Time (h)');
    ylabel('L0(t)');
    title(sprintf('%s', scenarios{s}));
    if s == 1
        legend({'True L0', 'Est L0'}, 'Location', 'best');
    end
    grid on;
end

sgtitle('Fixed vs Unfixed ke: Comprehensive Comparison');

%% Summary comparison table
fprintf('\n==========================================================\n');
fprintf('SUMMARY COMPARISON\n');
fprintf('==========================================================\n\n');

fprintf('%-25s %10s %10s %10s %10s\n', 'Scenario', 'MAPE-All(%)', 'MAPE-C&g(%)', 'L0-Corr', 'Time(s)');
fprintf('%-25s %10s %10s %10s %10s\n', '------------------------', '----------', '----------', '--------', '-------');

for s = 1:length(scenarios)
    errors_all = 100 * abs(theta_all{s}' - true_params) ./ abs(true_params);
    [corr_val, ~, ~] = computeL0Metrics(L0_all{s}, L0_true);
    
    fprintf('%-25s %10.1f %10.1f %10.3f %10.1f\n', ...
        scenarios{s}, mean(errors_all), mean(errors_all(1:6)), corr_val, times(s));
end

%% Statistical analysis
fprintf('\n----------------------------------------------------------\n');
fprintf('STATISTICAL ANALYSIS\n');
fprintf('----------------------------------------------------------\n');

% Find best scenario
mape_cg = zeros(length(scenarios), 1);
for s = 1:length(scenarios)
    errors = 100 * abs(theta_all{s}(1:6)' - true_params(1:6)) ./ abs(true_params(1:6));
    mape_cg(s) = mean(errors);
end

[best_mape, best_idx] = min(mape_cg);
fprintf('\nBest scenario for C&g estimation: %s\n', scenarios{best_idx});
fprintf('  MAPE for C&g: %.1f%%\n', best_mape);

% Analyze effect of fixing ke
if length(scenarios) >= 2
    improvement = mape_cg(1) - mape_cg(2);
    fprintf('\nEffect of fixing ke to true value:\n');
    if improvement > 0
        fprintf('  ✓ Improves C&g estimation by %.1f%% points\n', improvement);
    else
        fprintf('  ✗ Worsens C&g estimation by %.1f%% points\n', -improvement);
    end
    
    % L0 correlation comparison
    [corr_unfixed, ~, ~] = computeL0Metrics(L0_all{1}, L0_true);
    [corr_fixed, ~, ~] = computeL0Metrics(L0_all{2}, L0_true);
    
    fprintf('\nL0 recovery:\n');
    fprintf('  Unfixed ke: correlation = %.3f\n', corr_unfixed);
    fprintf('  Fixed ke:   correlation = %.3f\n', corr_fixed);
    
    if corr_fixed > corr_unfixed
        fprintf('  ✓ Fixing ke improves L0 recovery\n');
    else
        fprintf('  ✗ Fixing ke worsens L0 recovery\n');
    end
end

%% Recommendations
fprintf('\n==========================================================\n');
fprintf('RECOMMENDATIONS\n');
fprintf('==========================================================\n');

if best_idx == 2  % Fixed ke (true value) is best
    fprintf('\n✓ FIX ke to its true value (%.3f) when known\n', ke);
    fprintf('  - Reduces parameter space from 7 to 6 dimensions\n');
    fprintf('  - Improves C and g estimation\n');
    fprintf('  - Better L0 recovery\n');
elseif best_idx == 1  % Unfixed is best
    fprintf('\n✓ ESTIMATE ke jointly with other parameters\n');
    fprintf('  - More flexible model\n');
    fprintf('  - Avoids bias from incorrect ke value\n');
else
    fprintf('\n⚠ Results suggest sensitivity to ke value\n');
    fprintf('  - Consider estimating ke from separate PK data\n');
    fprintf('  - Or use strong prior on ke\n');
end

fprintf('\nGeneral recommendations:\n');
fprintf('  1. If ke is well-known from PK studies → FIX it\n');
fprintf('  2. If ke is uncertain → ESTIMATE it\n');
fprintf('  3. Use strong regularization regardless\n');
fprintf('  4. Consider sensitivity analysis with ke ± 10%%\n');

%% Export results to text file for paper
filename = sprintf('simulation_results_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));
fid = fopen(filename, 'w');

fprintf(fid, '==========================================================\n');
fprintf(fid, 'DOSE-SWITCHING SIMULATION RESULTS FOR PAPER\n');
fprintf(fid, 'Generated on: %s\n', datestr(now));
fprintf(fid, '==========================================================\n\n');

% Simulation parameters
fprintf(fid, 'SIMULATION PARAMETERS:\n');
fprintf(fid, '- Sample size: %d patients\n', N);
fprintf(fid, '- True ke: %.3f\n', ke);
fprintf(fid, '- Target threshold: %.3f\n', 0.1);
fprintf(fid, '- Simulation period: 168 hours\n');
fprintf(fid, '- Time step: 2 hours\n\n');

% Results summary table
fprintf(fid, 'ESTIMATION RESULTS SUMMARY:\n');
fprintf(fid, '%-25s %10s %10s %10s %10s\n', 'Method', 'MAPE-All(%)', 'MAPE-C&g(%)', 'L0-Corr', 'Time(s)');
fprintf(fid, '%-25s %10s %10s %10s %10s\n', repmat('-', 1, 25), repmat('-', 1, 10), repmat('-', 1, 10), repmat('-', 1, 8), repmat('-', 1, 7));

for s = 1:length(scenarios)
    errors_all = 100 * abs(theta_all{s}' - true_params) ./ abs(true_params);
    [corr_val, ~, ~] = computeL0Metrics(L0_all{s}, L0_true);
    fprintf(fid, '%-25s %10.1f %10.1f %10.3f %10.1f\n', ...
        scenarios{s}, mean(errors_all), mean(errors_all(1:6)), corr_val, times(s));
end

% Best method identification
[best_mape, best_idx] = min(mape_cg);
fprintf(fid, '\nBEST METHOD: %s (MAPE C&g: %.1f%%)\n\n', scenarios{best_idx}, best_mape);

% Parameter estimates table
fprintf(fid, 'DETAILED PARAMETER ESTIMATES:\n');
fprintf(fid, '%-12s', 'Parameter');
for s = 1:length(scenarios)
    fprintf(fid, ' %12s', scenarios{s});
end
fprintf(fid, ' %12s\n', 'True Value');

for p = 1:7
    fprintf(fid, '%-12s', param_names{p});
    for s = 1:length(scenarios)
        fprintf(fid, ' %12.4f', theta_all{s}(p));
    end
    fprintf(fid, ' %12.4f\n', true_params(p));
end

% Effect of fixing ke
if length(scenarios) >= 2
    improvement = mape_cg(1) - mape_cg(2);
    fprintf(fid, '\nEFFECT OF FIXING KE:\n');
    fprintf(fid, '- Change in MAPE (C&g): %.1f percentage points\n', improvement);
    [corr_unfixed, ~, ~] = computeL0Metrics(L0_all{1}, L0_true);
    [corr_fixed, ~, ~] = computeL0Metrics(L0_all{2}, L0_true);
    fprintf(fid, '- L0 correlation (unfixed): %.3f\n', corr_unfixed);
    fprintf(fid, '- L0 correlation (fixed): %.3f\n', corr_fixed);
    fprintf(fid, '- L0 correlation change: %.3f\n', corr_fixed - corr_unfixed);
end

% Conclusions
fprintf(fid, '\nCONCLUSIONS:\n');
if best_idx == 2
    fprintf(fid, '- Fixing ke to true value improves estimation performance\n');
    fprintf(fid, '- Reduces parameter space complexity (7 to 6 parameters)\n');
    fprintf(fid, '- Recommended when ke is well-characterized from PK studies\n');
else
    fprintf(fid, '- Joint estimation of ke with other parameters is preferred\n');
    fprintf(fid, '- Provides more robust results when ke is uncertain\n');
end

fclose(fid);
fprintf('Results exported to: %s\n', filename);

fprintf('\n==========================================================\n');

%% Helper function
function [mean_corr, mean_rmse, mean_mae] = computeL0Metrics(L0_est, L0_true)
    N = size(L0_est, 1);
    corr_vals = zeros(N, 1);
    rmse_vals = zeros(N, 1);
    mae_vals = zeros(N, 1);
    
    for i = 1:N
        valid = ~isnan(L0_est(i,:)) & ~isnan(L0_true(i,:));
        if sum(valid) > 10
            corr_vals(i) = corr(L0_est(i,valid)', L0_true(i,valid)');
            rmse_vals(i) = sqrt(mean((L0_est(i,valid) - L0_true(i,valid)).^2));
            mae_vals(i) = mean(abs(L0_est(i,valid) - L0_true(i,valid)));
        end
    end
    
    valid_idx = corr_vals > 0;
    mean_corr = mean(corr_vals(valid_idx));
    mean_rmse = mean(rmse_vals(valid_idx));
    mean_mae = mean(mae_vals(valid_idx));
end