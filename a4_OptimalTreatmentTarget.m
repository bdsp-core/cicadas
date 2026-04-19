clear all; clc; format compact;
dt = 2;
t = 0:dt:168;
Nt = length(t);
T0 = readtable('trialData0.csv');
N = length(unique(T0.sid));
load parmsTrue

Nboot = 1000;
th = linspace(0,1,50);

% Pre-allocate for parallel results
A0 = zeros(Nboot, length(th));
A1 = zeros(Nboot, length(th));

L0 = fcnGenerateStochasticTrajectories(t, parmsL, N);

% Start parallel pool if not already running
if isempty(gcp('nocreate'))
    parpool('local', feature('numcores')); % Use all available cores on Mac
end

% Parallel bootstrap loop
parfor n = 1:Nboot
    T0_boot = fcn_bootstrapBySID(T0, N);
    
    % Estimate parameters
    parmsY_est = fcnEstimateDeathParms(T0_boot);
    
    % Setup for RCT simulation
    RCT = 1;
    treatProb = 0.5*ones(1,N);
    parmsV_est = [0 0 0 0 0 0];
    
    % Local arrays for this iteration
    ATEest = zeros(1, length(th));
    ATEref = zeros(1, length(th));
    
    for i = 1:length(th)
        T_ref = fcnSimulate_N_Patients(N,RCT,treatProb,th(i), C, g, ke, L0, parmsControl, parmsY, parmsV_est, age, sofa);
        T_est = fcnSimulate_N_Patients(N,RCT,treatProb,th(i), C, g, ke, L0, parmsControl, parmsY_est, parmsV_est, age, sofa);
        
        [s0, s1, ~, ~] = fcnPlotKM(T_ref);
        [s0_est, s1_est, ~, ~] = fcnPlotKM(T_est);
        
        ATEest(i) = s1_est(end) - s0_est(end);
        ATEref(i) = s1(end) - s0(end);
    end
    
    A0(n,:) = ATEest;
    A1(n,:) = ATEref;
    
    fprintf('Completed bootstrap %d\n', n);
    
end

%% Export optimal treatment target analysis results to text file for paper
filename = sprintf('optimal_treatment_target_results_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));
fid = fopen(filename, 'w');

fprintf(fid, '==========================================================\n');
fprintf(fid, 'OPTIMAL TREATMENT TARGET ANALYSIS RESULTS FOR PAPER\n');
fprintf(fid, 'Generated on: %s\n', datestr(now));
fprintf(fid, '==========================================================\n\n');

% Study parameters
fprintf(fid, 'STUDY PARAMETERS:\n');
fprintf(fid, '- Sample size: %d patients\n', N);
fprintf(fid, '- Bootstrap iterations: %d\n', Nboot);
fprintf(fid, '- Study period: 168 hours\n');
fprintf(fid, '- Time step: %d hours\n', dt);
fprintf(fid, '- Treatment target range: %.2f to %.2f (50 levels)\n', min(th), max(th));
fprintf(fid, '- Target resolution: %.3f\n', th(2) - th(1));
if isempty(gcp('nocreate'))
    fprintf(fid, '- Parallel processing: No\n\n');
else
    fprintf(fid, '- Parallel processing: Yes\n\n');
end

% Calculate summary statistics
A0_median = median(A0, 1);  % Estimated parameters
A1_median = median(A1, 1);  % True parameters
A0_lower = prctile(A0, 2.5, 1);  % 95% CI lower
A0_upper = prctile(A0, 97.5, 1); % 95% CI upper
A1_lower = prctile(A1, 2.5, 1);  % 95% CI lower
A1_upper = prctile(A1, 97.5, 1); % 95% CI upper

% Find optimal targets
[max_ate_est, opt_idx_est] = max(A0_median);
[max_ate_true, opt_idx_true] = max(A1_median);

fprintf(fid, 'OPTIMAL TREATMENT TARGETS:\n');
fprintf(fid, 'Based on TRUE parameters:\n');
fprintf(fid, '  Optimal threshold: %.3f\n', th(opt_idx_true));
fprintf(fid, '  Maximum ATE: %.3f (%.1f%% survival benefit)\n', max_ate_true, max_ate_true*100);
fprintf(fid, '  95%% CI: [%.3f, %.3f]\n', A1_lower(opt_idx_true), A1_upper(opt_idx_true));

fprintf(fid, '\nBased on ESTIMATED parameters:\n');
fprintf(fid, '  Optimal threshold: %.3f\n', th(opt_idx_est));
fprintf(fid, '  Maximum ATE: %.3f (%.1f%% survival benefit)\n', max_ate_est, max_ate_est*100);
fprintf(fid, '  95%% CI: [%.3f, %.3f]\n', A0_lower(opt_idx_est), A0_upper(opt_idx_est));

% Agreement analysis
target_difference = abs(th(opt_idx_est) - th(opt_idx_true));
ate_difference = max_ate_est - max_ate_true;

fprintf(fid, '\nOPTIMAL TARGET AGREEMENT:\n');
fprintf(fid, 'Target difference: %.3f\n', target_difference);
fprintf(fid, 'ATE difference: %.3f (%.1f%% points)\n', ate_difference, ate_difference*100);

if target_difference < 0.05
    fprintf(fid, '✓ EXCELLENT agreement - targets differ by < 0.05\n');
elseif target_difference < 0.1
    fprintf(fid, '✓ GOOD agreement - targets differ by < 0.10\n');
elseif target_difference < 0.2
    fprintf(fid, '⚠️  MODERATE agreement - targets differ by < 0.20\n');
else
    fprintf(fid, '✗ POOR agreement - targets differ by ≥ 0.20\n');
end

% Performance across target range
fprintf(fid, '\nPERFORMANCE ACROSS TARGET RANGE:\n');

% Calculate bias and correlation across all targets
bias_median = A0_median - A1_median;
[correlation, p_value] = corr(A0_median', A1_median');

fprintf(fid, 'Overall performance:\n');
fprintf(fid, '  Correlation (est vs true): %.3f (p = %.3f)\n', correlation, p_value);
fprintf(fid, '  Mean bias: %.3f (%.1f%% points)\n', mean(bias_median), mean(bias_median)*100);
fprintf(fid, '  RMS bias: %.3f\n', sqrt(mean(bias_median.^2)));
fprintf(fid, '  Max absolute bias: %.3f at th = %.3f\n', max(abs(bias_median)), th(find(abs(bias_median) == max(abs(bias_median)), 1)));

% Key target thresholds analysis
key_targets = [0.05, 0.1, 0.2, 0.5];
fprintf(fid, '\nKEY TARGET THRESHOLDS ANALYSIS:\n');
fprintf(fid, '%-8s %-12s %-12s %-12s %-15s\n', 'Target', 'True ATE', 'Est ATE', 'Bias', 'Est 95% CI');
fprintf(fid, '%-8s %-12s %-12s %-12s %-15s\n', '------', '--------', '-------', '----', '-----------');

for i = 1:length(key_targets)
    target_val = key_targets(i);
    [~, target_idx] = min(abs(th - target_val));
    
    true_ate = A1_median(target_idx);
    est_ate = A0_median(target_idx);
    bias = est_ate - true_ate;
    ci_str = sprintf('[%.3f,%.3f]', A0_lower(target_idx), A0_upper(target_idx));
    
    fprintf(fid, '%-8.2f %-12.3f %-12.3f %-12.3f %-15s\n', ...
        target_val, true_ate, est_ate, bias, ci_str);
end

% Bootstrap uncertainty analysis
fprintf(fid, '\nBOOTSTRAP UNCERTAINTY ANALYSIS:\n');

% Calculate coefficient of variation for optimal targets
cv_est_optimal = std(A0(:, opt_idx_est)) / mean(A0(:, opt_idx_est)) * 100;
cv_true_optimal = std(A1(:, opt_idx_true)) / mean(A1(:, opt_idx_true)) * 100;

fprintf(fid, 'Uncertainty at optimal targets:\n');
fprintf(fid, '  Estimated optimal (th=%.3f): CV = %.1f%%\n', th(opt_idx_est), cv_est_optimal);
fprintf(fid, '  True optimal (th=%.3f): CV = %.1f%%\n', th(opt_idx_true), cv_true_optimal);

% Calculate proportion of bootstrap iterations where estimated > true
proportion_est_better = mean(A0(:, opt_idx_est) > A1(:, opt_idx_true));
fprintf(fid, '  Proportion where est > true at optimal: %.1f%%\n', proportion_est_better*100);

% Target selection robustness
% How often does each method select the "correct" optimal target?
est_optimal_selections = zeros(1, length(th));
true_optimal_selections = zeros(1, length(th));

for n = 1:Nboot
    [~, est_opt_idx] = max(A0(n, :));
    [~, true_opt_idx] = max(A1(n, :));
    est_optimal_selections(est_opt_idx) = est_optimal_selections(est_opt_idx) + 1;
    true_optimal_selections(true_opt_idx) = true_optimal_selections(true_opt_idx) + 1;
end

est_optimal_selections = est_optimal_selections / Nboot * 100;
true_optimal_selections = true_optimal_selections / Nboot * 100;

% Find most frequently selected targets
[max_est_freq, most_freq_est_idx] = max(est_optimal_selections);
[max_true_freq, most_freq_true_idx] = max(true_optimal_selections);

fprintf(fid, '\nTARGET SELECTION ROBUSTNESS:\n');
fprintf(fid, 'Most frequently selected targets across bootstrap iterations:\n');
fprintf(fid, '  Estimated parameters: th = %.3f (%.1f%% of iterations)\n', ...
    th(most_freq_est_idx), max_est_freq);
fprintf(fid, '  True parameters: th = %.3f (%.1f%% of iterations)\n', ...
    th(most_freq_true_idx), max_true_freq);

% Calculate target selection agreement rate
agreement_tolerance = 0.05;  % Within 0.05 of each other
agreement_count = 0;
for n = 1:Nboot
    [~, est_opt_idx] = max(A0(n, :));
    [~, true_opt_idx] = max(A1(n, :));
    if abs(th(est_opt_idx) - th(true_opt_idx)) <= agreement_tolerance
        agreement_count = agreement_count + 1;
    end
end
agreement_rate = agreement_count / Nboot * 100;

fprintf(fid, '  Target agreement rate (within %.2f): %.1f%%\n', agreement_tolerance, agreement_rate);

% Clinical decision zones
fprintf(fid, '\nCLINICAL DECISION ZONES:\n');

% Beneficial treatment zone (ATE > 0.02)
beneficial_threshold = 0.02;
beneficial_est = sum(A0_median > beneficial_threshold);
beneficial_true = sum(A1_median > beneficial_threshold);

fprintf(fid, 'Beneficial treatment zones (ATE > %.1f%%):\n', beneficial_threshold*100);
fprintf(fid, '  True parameters: %d/%d targets (%.1f%%)\n', ...
    beneficial_true, length(th), beneficial_true/length(th)*100);
fprintf(fid, '  Estimated parameters: %d/%d targets (%.1f%%)\n', ...
    beneficial_est, length(th), beneficial_est/length(th)*100);

% Harmful treatment zone (ATE < -0.01)
harmful_threshold = -0.01;
harmful_est = sum(A0_median < harmful_threshold);
harmful_true = sum(A1_median < harmful_threshold);

fprintf(fid, 'Potentially harmful zones (ATE < %.1f%%):\n', harmful_threshold*100);
fprintf(fid, '  True parameters: %d/%d targets (%.1f%%)\n', ...
    harmful_true, length(th), harmful_true/length(th)*100);
fprintf(fid, '  Estimated parameters: %d/%d targets (%.1f%%)\n', ...
    harmful_est, length(th), harmful_est/length(th)*100);

% Safe operating zone (consistent benefit)
safe_zone_est = A0_lower > 0.01;  % Lower CI > 1%
safe_zone_true = A1_lower > 0.01;

safe_targets_est = sum(safe_zone_est);
safe_targets_true = sum(safe_zone_true);

fprintf(fid, 'Safe operating zones (95%% CI lower bound > 1%%):\n');
fprintf(fid, '  True parameters: %d targets\n', safe_targets_true);
fprintf(fid, '  Estimated parameters: %d targets\n', safe_targets_est);

% Clinical recommendations
fprintf(fid, '\nCLINICAL RECOMMENDATIONS:\n');

if target_difference < 0.1 && agreement_rate > 70
    fprintf(fid, '✓ PROCEED with target optimization using estimated parameters\n');
    fprintf(fid, '  - Target agreement is excellent (%.1f%% bootstrap agreement)\n', agreement_rate);
    fprintf(fid, '  - Recommended target: th = %.3f\n', th(opt_idx_est));
    if safe_targets_est > 10
        fprintf(fid, '  - Multiple safe targets available for flexibility\n');
    end
elseif target_difference < 0.2
    fprintf(fid, '⚠️  CAUTION advised for target optimization\n');
    fprintf(fid, '  - Moderate target agreement (%.1f%% bootstrap agreement)\n', agreement_rate);
    fprintf(fid, '  - Consider sensitivity analysis around th = %.3f ± %.2f\n', ...
        th(opt_idx_est), target_difference);
else
    fprintf(fid, '✗ HIGH RISK for target optimization\n');
    fprintf(fid, '  - Poor target agreement (%.1f%% bootstrap agreement)\n', agreement_rate);
    fprintf(fid, '  - Improve parameter estimation before optimization\n');
    fprintf(fid, '  - Consider conservative target selection\n');
end

% Parameter estimation priorities
fprintf(fid, '\nPARAMETER ESTIMATION PRIORITIES:\n');
if mean(abs(bias_median)) < 0.02
    fprintf(fid, '- Current parameter estimation accuracy is EXCELLENT\n');
elseif mean(abs(bias_median)) < 0.05
    fprintf(fid, '- Current parameter estimation accuracy is GOOD\n');
else
    fprintf(fid, '- Parameter estimation accuracy needs IMPROVEMENT\n');
end

fprintf(fid, '- Focus on mortality hazard parameter estimation\n');
fprintf(fid, '- Bootstrap confidence intervals provide uncertainty quantification\n');
fprintf(fid, '- Validate target selection with independent data\n');

% Technical details
fprintf(fid, '\nTECHNICAL DETAILS:\n');
fprintf(fid, '- Optimization method: Grid search with bootstrap resampling\n');
fprintf(fid, '- Target range: [%.2f, %.2f] with %.3f resolution\n', min(th), max(th), th(2)-th(1));
fprintf(fid, '- Bootstrap method: Resampling by patient ID\n');
fprintf(fid, '- Confidence intervals: 2.5th and 97.5th percentiles\n');
fprintf(fid, '- Endpoint: Survival probability at 168 hours\n');
fprintf(fid, '- Simulation: RCT emulation with 50%% treatment probability\n');

% Data availability
fprintf(fid, '\nDATA AVAILABILITY:\n');
fprintf(fid, '- Complete bootstrap results saved to: A01Data.mat\n');
fprintf(fid, '- A0: Bootstrap ATE estimates using estimated parameters\n');
fprintf(fid, '- A1: Bootstrap ATE estimates using true parameters\n');
fprintf(fid, '- th: Treatment target grid (50 levels)\n');
fprintf(fid, '- Confidence bands available for all targets\n');

fclose(fid);
fprintf('Optimal treatment target analysis results exported to: %s\n', filename);

% Plot final results with enhanced visualization
figure(1); clf
set(gcf, 'Position', [100, 100, 1200, 600]);

% Main plot
subplot(1,2,1);
plot(th, A0', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5); hold on;
plot(th, A1', 'Color', [1 0.8 0.8], 'LineWidth', 0.5);
plot(th, A0_median, 'k-', 'LineWidth', 2);
plot(th, A1_median, 'r-', 'LineWidth', 2);
plot(th, A0_lower, 'k--', 'LineWidth', 1);
plot(th, A0_upper, 'k--', 'LineWidth', 1);
plot(th, A1_lower, 'r--', 'LineWidth', 1);
plot(th, A1_upper, 'r--', 'LineWidth', 1);

% Mark optimal points
plot(th(opt_idx_est), max_ate_est, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
plot(th(opt_idx_true), max_ate_true, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

xlabel('Treatment Target Threshold');
ylabel('Average Treatment Effect (ATE)');
title('Optimal Treatment Target Analysis');
legend({'Est Bootstrap', 'True Bootstrap', 'Est Median', 'True Median', ...
    'Est 95% CI', '', 'True 95% CI', '', 'Est Optimal', 'True Optimal'}, ...
    'Location', 'best');
grid on;

% Bias plot
subplot(1,2,2);
plot(th, bias_median, 'b-', 'LineWidth', 2); hold on;
plot(th, zeros(size(th)), 'k--', 'LineWidth', 1);
xlabel('Treatment Target Threshold');
ylabel('Bias (Estimated - True ATE)');
title('Estimation Bias Across Targets');
grid on;

save A01Data

