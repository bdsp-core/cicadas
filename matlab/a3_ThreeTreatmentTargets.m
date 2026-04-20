%% get 3 curves
clear all; clc; format compact;
dt = 2;
t = 0:dt:168;
Nt = length(t);
T0 = readtable('trialData0.csv');
N = length(unique(T0.sid));
load parmsTrue

th = [0 0.02 .1 0.8]; 
    
% Estimate parameters
parmsY_est = fcnEstimateDeathParms(T0);
[parmsL_est, LL, AA, patient_age, patient_sofa, t_local] = fcnEstimateParmsL(T0);

% Setup for RCT simulation
RCT = 1;
treatProb = 0.5*ones(1,N);
parmsV_est = [0 0 0 0 0 0];
L0 = fcnGenerateStochasticTrajectories(t, parmsL, N);

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

    S1est(i,:) = s1_est; 
    S0est(i,:) = s0_est; 

    S1ref(i,:) = s1; 
    S0ref(i,:) = s0; 

end
    
save ThreeCurves

%% Export treatment targets comparison results to text file for paper
filename = sprintf('treatment_targets_results_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));
fid = fopen(filename, 'w');

fprintf(fid, '==========================================================\n');
fprintf(fid, 'TREATMENT TARGETS COMPARISON RESULTS FOR PAPER\n');
fprintf(fid, 'Generated on: %s\n', datestr(now));
fprintf(fid, '==========================================================\n\n');

% Study parameters
fprintf(fid, 'STUDY PARAMETERS:\n');
fprintf(fid, '- Sample size: %d patients\n', N);
fprintf(fid, '- Study period: 168 hours\n');
fprintf(fid, '- Time step: %d hours\n', dt);
fprintf(fid, '- Number of treatment targets tested: %d\n', length(th));
fprintf(fid, '- Target thresholds: [%s]\n\n', sprintf('%.2f ', th));

% Treatment target descriptions
fprintf(fid, 'TREATMENT TARGET DESCRIPTIONS:\n');
target_descriptions = {
    'No treatment (th = 0.00)', 'No active disease suppression';
    'Mild treatment (th = 0.02)', 'Minimal disease suppression';
    'Standard treatment (th = 0.10)', 'Moderate disease suppression';
    'Aggressive treatment (th = 0.80)', 'Intensive disease suppression'
};

for i = 1:length(th)
    fprintf(fid, '  Target %d: %s - %s\n', i, target_descriptions{i,1}, target_descriptions{i,2});
end
fprintf(fid, '\n');

% Parameter estimation accuracy
param_errors_L = 100 * abs(parmsL_est - parmsL(:)) ./ abs(parmsL(:));
param_errors_Y = 100 * abs(parmsY_est - parmsY(:)) ./ abs(parmsY(:));

fprintf(fid, 'PARAMETER ESTIMATION ACCURACY:\n');
fprintf(fid, 'Disease progression (L) parameters:\n');
param_names_L = {'growth_rate', 'peak_height', 'alpha', 'decay_rate', 'sigma_early', 'sigma_late', 'sigma_transition'};
for i = 1:min(length(parmsL_est), length(param_names_L))
    fprintf(fid, '  %s: %.4f (true: %.4f, error: %.1f%%)\n', ...
        param_names_L{i}, parmsL_est(i), parmsL(i), param_errors_L(i));
end

fprintf(fid, '\nMortality hazard (Y) parameters:\n');
for i = 1:length(parmsY_est)
    fprintf(fid, '  a%d: %.4f (true: %.4f, error: %.1f%%)\n', ...
        i-1, parmsY_est(i), parmsY(i), param_errors_Y(i));
end
fprintf(fid, 'Overall MAPE - L parameters: %.1f%%\n', mean(param_errors_L));
fprintf(fid, 'Overall MAPE - Y parameters: %.1f%%\n\n', mean(param_errors_Y));

% Treatment effects comparison
fprintf(fid, 'TREATMENT EFFECTS COMPARISON (ATE at 168 hours):\n');
fprintf(fid, '%-25s %-12s %-12s %-12s %-12s\n', 'Target', 'True ATE', 'Est ATE', 'Bias', 'Bias(%)');
fprintf(fid, '%-25s %-12s %-12s %-12s %-12s\n', repmat('-', 1, 25), repmat('-', 1, 12), repmat('-', 1, 12), repmat('-', 1, 12), repmat('-', 1, 12));

for i = 1:length(th)
    bias_abs = ATEest(i) - ATEref(i);
    bias_pct = 100 * bias_abs / abs(ATEref(i));
    
    fprintf(fid, '%-25s %+11.3f %+11.3f %+11.3f %+11.1f\n', ...
        sprintf('th = %.2f', th(i)), ATEref(i), ATEest(i), bias_abs, bias_pct);
end

% Survival outcomes at 168 hours
fprintf(fid, '\nSURVIVAL OUTCOMES AT 168 HOURS:\n');
fprintf(fid, '%-15s %-15s %-15s %-15s %-15s\n', 'Target', 'Untrt(True)', 'Treat(True)', 'Untrt(Est)', 'Treat(Est)');
fprintf(fid, '%-15s %-15s %-15s %-15s %-15s\n', repmat('-', 1, 15), repmat('-', 1, 15), repmat('-', 1, 15), repmat('-', 1, 15), repmat('-', 1, 15));

for i = 1:length(th)
    fprintf(fid, '%-15s %-15.1f%% %-15.1f%% %-15.1f%% %-15.1f%%\n', ...
        sprintf('th = %.2f', th(i)), ...
        S0ref(i,end)*100, S1ref(i,end)*100, ...
        S0est(i,end)*100, S1est(i,end)*100);
end

% Optimal treatment target analysis
[max_ate_true, optimal_idx_true] = max(ATEref);
[max_ate_est, optimal_idx_est] = max(ATEest);

fprintf(fid, '\nOPTIMAL TREATMENT TARGET ANALYSIS:\n');
fprintf(fid, 'Based on TRUE parameters:\n');
fprintf(fid, '  Optimal target: th = %.2f\n', th(optimal_idx_true));
fprintf(fid, '  Maximum ATE: %.3f (%.1f%% survival benefit)\n', max_ate_true, max_ate_true*100);
fprintf(fid, '  Survival rates: Untreated %.1f%%, Treated %.1f%%\n', ...
    S0ref(optimal_idx_true,end)*100, S1ref(optimal_idx_true,end)*100);

fprintf(fid, '\nBased on ESTIMATED parameters:\n');
fprintf(fid, '  Optimal target: th = %.2f\n', th(optimal_idx_est));
fprintf(fid, '  Maximum ATE: %.3f (%.1f%% survival benefit)\n', max_ate_est, max_ate_est*100);
fprintf(fid, '  Survival rates: Untreated %.1f%%, Treated %.1f%%\n', ...
    S0est(optimal_idx_est,end)*100, S1est(optimal_idx_est,end)*100);

% Agreement analysis
target_agreement = (optimal_idx_true == optimal_idx_est);
fprintf(fid, '\nTARGET SELECTION AGREEMENT:\n');
if target_agreement
    fprintf(fid, '✓ TRUE and ESTIMATED parameters identify the same optimal target\n');
    fprintf(fid, '  → Parameter estimation is sufficient for treatment optimization\n');
else
    fprintf(fid, '✗ TRUE and ESTIMATED parameters identify different optimal targets\n');
    fprintf(fid, '  → Parameter estimation bias affects treatment optimization\n');
    fprintf(fid, '  True optimal: th = %.2f, Estimated optimal: th = %.2f\n', ...
        th(optimal_idx_true), th(optimal_idx_est));
end

% Bias analysis across targets
fprintf(fid, '\nBIAS ANALYSIS ACROSS TARGETS:\n');
mean_bias = mean(abs(ATEest - ATEref));
max_bias = max(abs(ATEest - ATEref));
[~, max_bias_idx] = max(abs(ATEest - ATEref));

fprintf(fid, 'Average absolute bias: %.3f (%.1f%% survival difference)\n', mean_bias, mean_bias*100);
fprintf(fid, 'Maximum absolute bias: %.3f at th = %.2f\n', max_bias, th(max_bias_idx));

% Trend analysis
if length(th) > 2
    % Check if bias increases with target aggressiveness
    bias_trend = corr((1:length(th))', abs(ATEest - ATEref)');
    fprintf(fid, 'Bias vs target aggressiveness correlation: %.3f\n', bias_trend);
    if bias_trend > 0.5
        fprintf(fid, '  → Bias INCREASES with more aggressive targets\n');
    elseif bias_trend < -0.5
        fprintf(fid, '  → Bias DECREASES with more aggressive targets\n');
    else
        fprintf(fid, '  → Bias is relatively CONSTANT across targets\n');
    end
end

% Clinical implications
fprintf(fid, '\nCLINICAL IMPLICATIONS:\n');

% Treatment benefit analysis
beneficial_targets_true = sum(ATEref > 0);
beneficial_targets_est = sum(ATEest > 0);
harmful_targets_true = sum(ATEref < 0);
harmful_targets_est = sum(ATEest < 0);

fprintf(fid, 'Treatment benefit assessment:\n');
fprintf(fid, '  TRUE parameters: %d beneficial, %d harmful targets\n', ...
    beneficial_targets_true, harmful_targets_true);
fprintf(fid, '  ESTIMATED parameters: %d beneficial, %d harmful targets\n', ...
    beneficial_targets_est, harmful_targets_est);

if beneficial_targets_true == beneficial_targets_est
    fprintf(fid, '  ✓ Parameter estimation correctly identifies treatment benefit\n');
else
    fprintf(fid, '  ✗ Parameter estimation misclassifies treatment benefit\n');
end

% Safety considerations
if any(ATEref < 0)
    harmful_idx = find(ATEref < 0);
    fprintf(fid, '\nSAFETY CONSIDERATIONS:\n');
    for i = harmful_idx
        fprintf(fid, '  ⚠️  th = %.2f appears HARMFUL (ATE = %.3f)\n', th(i), ATEref(i));
        if ATEest(i) > 0
            fprintf(fid, '     But estimated as beneficial (ATE = %.3f) - DANGEROUS!\n', ATEest(i));
        end
    end
end

% Recommendations
fprintf(fid, '\nRECOMMENDATIONS:\n');
if target_agreement && mean_bias < 0.05
    fprintf(fid, '✓ PROCEED with treatment optimization using estimated parameters\n');
    fprintf(fid, '  - Parameter estimation is sufficiently accurate\n');
    fprintf(fid, '  - Optimal target correctly identified\n');
    fprintf(fid, '  - Recommended target: th = %.2f\n', th(optimal_idx_est));
else
    fprintf(fid, '⚠️  CAUTION required for treatment optimization\n');
    if ~target_agreement
        fprintf(fid, '  - Parameter estimation affects optimal target selection\n');
        fprintf(fid, '  - Consider sensitivity analysis around th = %.2f\n', th(optimal_idx_true));
    end
    if mean_bias >= 0.05
        fprintf(fid, '  - High bias in treatment effect estimation\n');
        fprintf(fid, '  - Improve parameter estimation before optimization\n');
    end
end

fprintf(fid, '\nGeneral recommendations:\n');
fprintf(fid, '- Validate parameter estimates with independent data\n');
fprintf(fid, '- Consider dose-response relationships\n');
fprintf(fid, '- Monitor for unintended consequences\n');
fprintf(fid, '- Use adaptive treatment strategies\n');

% Technical details
fprintf(fid, '\nTECHNICAL DETAILS:\n');
fprintf(fid, '- Simulation: RCT emulation with 50%% treatment probability\n');
fprintf(fid, '- Endpoint: Survival probability at 168 hours\n');
fprintf(fid, '- Target mechanism: PI controller with variable threshold\n');
fprintf(fid, '- Parameter source: Estimated from observational data\n');
fprintf(fid, '- Comparison: True vs estimated parameter performance\n');

fclose(fid);
fprintf('Treatment targets comparison results exported to: %s\n', filename);

%%
figure(1); clf; 
for i = 1:4; 
    subplot(4,1,i); 
    plot(t,S1ref(i,:),'r--',t,S0ref(i,:),'b--');
    hold on
    plot(t,S1est(i,:),'r',t,S0est(i,:),'b');
    title(sprintf('Target th = %.2f', th(i)));
    ylabel('Survival Probability');
    if i == 4
        xlabel('Time (hours)');
    end
    legend({'Treated (True)', 'Untreated (True)', 'Treated (Est)', 'Untreated (Est)'}, 'Location', 'best');
    grid on;
end
sgtitle('Treatment Targets Comparison: True vs Estimated Parameters');