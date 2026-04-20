clear all; clc; format compact; 
figure(3); clf;

dt = 2; 
t = 0:dt:168;  % Create regular time grid
Nt = length(t); % 85 

T0 = readtable('trialData0.csv');
N = length(unique(T0.sid));
load parmsTrue % gets true values: 'parmsControl', 'parmsPD', 'C', 'g', 'parmsY', 'parmsV', 'parmsL'
th = 0.1; % target level for L(t) - controller tries to keep L(t) <= th

%===================================================
%% ESTIMATE L and Y models 
%===================================================
parmsY_est = fcnEstimateDeathParms(T0);
[parmsL_est, LL, AA, patient_age, patient_sofa, t] = fcnEstimateParmsL(T0);

%% ================================================================================
%% Load estimated coefficients for C, g, and estimated ke
%% ================================================================================
[C_est, g_est, ke_est, parmsPD_est] = fcnGetPKPD_parms_est(patient_age, patient_sofa);

% Ensure parmsL_est is a column vector and has exactly 7 elements
parmsL_est = parmsL_est(:);
if length(parmsL_est) > 7
    parmsL_est = parmsL_est(1:7);
end

% Ensure parmsPD_est is a column vector
parmsPD_est = parmsPD_est(:);

% Create theta_est with exactly 14 parameters
theta_est = [parmsL_est; parmsPD_est; ke_est];

save EstimatedParameters theta_est parmsL_est parmsPD_est ke_est

%% ================================================================================
%% Estimate causal survival curves - contrasting always vs never treat
%% ================================================================================
RCT = 1; % simulated RCT using estimated models
treatProb = 0.5*ones(1,N); % for RCT simulation
parmsV_est = [0 0 0 0 0 0]; % No censoring in RCT simulation

L0_est = fcnGenerateStochasticTrajectories(t, parmsL_est, N);
T1_est = fcnSimulate_N_Patients(N,RCT,treatProb,th, C_est, g_est, ke_est, L0_est, parmsControl, parmsY_est, parmsV_est, age, sofa);

%==========================================
%%=== EVALUATION PLOTS ======
%==========================================

% Combine true parameters for comparison
b0_C_true = parmsPD(1); b1_C_true = parmsPD(2); b2_C_true = parmsPD(3); 
b0_g_true = parmsPD(4); b1_g_true = parmsPD(5); b2_g_true = parmsPD(6); 

% there are 14 parameters to estimate - whoah. 
true_params_all = [parmsL, b0_C_true, b1_C_true, b2_C_true, b0_g_true, b1_g_true, b2_g_true, ke];

fprintf('Mortality model coefficients:\n');
fprintf('  a0 (intercept): %.3f\n', parmsY_est(1));
fprintf('  a1 (time effect): %.3f\n', parmsY_est(2));
fprintf('  a2 (cumL effect): %.3f\n', parmsY_est(3));
fprintf('  a3 (cumA effect): %.3f\n', parmsY_est(4));

% Debug: Check sizes right before calling the function
fprintf('\nDEBUG - Parameter sizes:\n');
fprintf('  theta_est size = %dx%d (should be 14x1)\n', size(theta_est,1), size(theta_est,2));
fprintf('  true_params_all size = %dx%d (should be 14x1)\n', size(true_params_all,1), size(true_params_all,2));
fprintf('  theta_est length = %d\n', length(theta_est));
fprintf('  true_params_all length = %d\n', length(true_params_all));

% make some plots to show how well the estimation worked
fcnDiseaseModelDiagnostics(T1_est,true_params_all, C_est, g_est, theta_est, LL, AA, age, sofa, t)

%% get CI intervals
RCT = 1; % simulated RCT using estimated models
treatProb = 0.5*ones(1,N); % for RCT simulation
parmsV_est = [0 0 0 0 0 0]; % No censoring in RCT simulation

Nboot = 1000; % Number of bootstrap iterations (increased from 10 for better CIs)

S0h = []; 
S1h = [];

fprintf('Running bootstrap iterations...\n');
for i = 1:Nboot
    if mod(i, 10) == 0
        fprintf('  Iteration %d/%d\n', i, Nboot);
    end
    
    T0_boot = fcn_bootstrapBySID(T0, N);
    parmsY_est = fcnEstimateDeathParms(T0_boot);
    L0_est = fcnGenerateStochasticTrajectories(t, parmsL_est, N);
    T1_est = fcnSimulate_N_Patients(N,RCT,treatProb,th, C_est, g_est, ke_est, L0_est, parmsControl, parmsY_est, parmsV_est, age, sofa);
    [s0h,s1h,t0,t1] = fcnPlotKM(T1_est);
    
    S0h = [S0h; s0h'];
    S1h = [S1h; s1h'];
    
    figure(3); clf;  
    plot(t0,S0h,'b',t1,S1h,'r');
    drawnow

end

save bootstrap_confidence_bands_v2

%% Calculate 95% confidence bands
% S0h and S1h are Nboot x 85 matrices (each row is one bootstrap curve)
% S0h and S1h are Nboot x 85 matrices (each row is one bootstrap curve)
load bootstrap_confidence_bands_v2

addpath('CICADA_FIGURES')

alpha = 0.05;  % For 95% confidence intervals

% Calculate percentiles for each time point
s0_lower = prctile(S0h, 100*alpha/2, 1);    % 2.5th percentile
s0_upper = prctile(S0h, 100*(1-alpha/2), 1); % 97.5th percentile
s0_median = prctile(S0h, 50, 1);             % Median

s1_lower = prctile(S1h, 100*alpha/2, 1);    % 2.5th percentile
s1_upper = prctile(S1h, 100*(1-alpha/2), 1); % 97.5th percentile
s1_median = prctile(S1h, 50, 1);             % Median

% Calculate mean curves (alternative to median)
s0_mean = mean(S0h, 1);
s1_mean = mean(S1h, 1);

% Get reference curves
T1 = readtable('trialData1.csv');
[s0_true, s1_true, t0_true, t1_true] = fcnPlotKM(T1);

% Get estimated curves from full dataset (not bootstrap)
% [s0_est, s1_est, t0_est, t1_est] = fcnPlotKM(T1_est);

% Create plot with confidence bands
figure('Position', [100, 100, 800, 600]);
hold on;

% Time vector (should be same for all curves now)
t_grid = 0:2:168;  % This gives 85 points

% Convert to column vectors for consistency
t_grid = t_grid(:);
s0_lower = s0_lower(:);
s0_upper = s0_upper(:);
s1_lower = s1_lower(:);
s1_upper = s1_upper(:);

% Plot confidence bands first (so they appear behind the lines)
% Untreated confidence band (blue)
fill([t_grid; flipud(t_grid)], ...
     [s0_lower; flipud(s0_upper)], ...
     [0.2, 0.4, 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
     'DisplayName', 'Untreated 95% CI');

% Treated confidence band (red)
fill([t_grid; flipud(t_grid)], ...
     [s1_lower; flipud(s1_upper)], ...
     [0.8, 0.2, 0.2], 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
     'DisplayName', 'Treated 95% CI');

% Plot main curves
% True curves (dashed)
plot(t0_true, s0_true, 'b--', 'LineWidth', 2.5, 'DisplayName', 'Untreated (True)');
plot(t1_true, s1_true, 'r--', 'LineWidth', 2.5, 'DisplayName', 'Treated (True)');

% Estimated curves (solid) - from full dataset
plot(t0_true, s0_median, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Untreated (Estimated)');
plot(t0_true, s1_median, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Treated (Estimated)');

% Formatting
xlabel('Time (hours)', 'FontSize', 12);
ylabel('Survival Probability', 'FontSize', 12);
title(sprintf('Survival Curves with 95%% Bootstrap Confidence Bands (n=%d)', Nboot), 'FontSize', 14);
legend('Location', 'southwest', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 11);
xlim([0, 168]);
ylim([0, 1]);
set(gcf, 'Color', 'w');

% Display confidence interval widths at key time points
fprintf('\n95%% Confidence Interval Widths at Key Time Points:\n');
fprintf('Time (h) | Untreated CI Width | Treated CI Width\n');
fprintf('---------|-------------------|------------------\n');

key_times = [24, 48, 72, 96, 120, 144, 168];
for kt = key_times
    [~, idx] = min(abs(t_grid - kt));
    ci_width_0 = s0_upper(idx) - s0_lower(idx);
    ci_width_1 = s1_upper(idx) - s1_lower(idx);
    fprintf('%8d | %17.3f | %16.3f\n', kt, ci_width_0, ci_width_1);
end

% Convert back to row vectors for display (if needed)
s0_lower = s0_lower';
s0_upper = s0_upper';
s1_lower = s1_lower';
s1_upper = s1_upper';
s0_median = s0_median(:)';
s1_median = s1_median(:)';

% Display summary statistics
fprintf('\nBootstrap Summary Statistics:\n');
fprintf('Number of bootstrap samples: %d\n', Nboot);
fprintf('Survival at 168 hours:\n');
fprintf('  Untreated: %.1f%% [%.1f%%, %.1f%%]\n', ...
    s0_median(end)*100, s0_lower(end)*100, s0_upper(end)*100);
fprintf('  Treated: %.1f%% [%.1f%%, %.1f%%]\n', ...
    s1_median(end)*100, s1_lower(end)*100, s1_upper(end)*100);

% Treatment effect at end of study
treatment_effect_median = s1_median(end) - s0_median(end);
treatment_effect_lower = s1_lower(end) - s0_upper(end);  % Conservative
treatment_effect_upper = s1_upper(end) - s0_lower(end);  % Conservative

fprintf('\nTreatment effect at 168 hours:\n');
fprintf('  Median difference: %.3f\n', treatment_effect_median);
fprintf('  95%% CI: [%.3f, %.3f]\n', treatment_effect_lower, treatment_effect_upper);

% Optional: Save the confidence band data
save('bootstrap_confidence_bands.mat', 'S0h', 'S1h', 's0_lower', 's0_upper', ...
     's1_lower', 's1_upper', 's0_median', 's1_median', 't_grid', 'Nboot');

fprintf('\nBootstrap analysis complete!\n');

%% Export causal survival analysis results to text file for paper
filename = sprintf('causal_survival_results_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));
fid = fopen(filename, 'w');

fprintf(fid, '==========================================================\n');
fprintf(fid, 'CAUSAL SURVIVAL ANALYSIS RESULTS FOR PAPER\n');
fprintf(fid, 'Generated on: %s\n', datestr(now));
fprintf(fid, '==========================================================\n\n');

% Study parameters
fprintf(fid, 'STUDY PARAMETERS:\n');
fprintf(fid, '- Sample size: %d patients\n', N);
fprintf(fid, '- Study period: 168 hours\n');
fprintf(fid, '- Bootstrap samples: %d\n', Nboot);
fprintf(fid, '- Target threshold: %.3f\n', th);
fprintf(fid, '- Time step: 2 hours\n\n');

% Parameter estimation results
fprintf(fid, 'ESTIMATED PARAMETERS:\n');
fprintf(fid, 'Disease progression (L) parameters:\n');
for i = 1:length(parmsL_est)
    fprintf(fid, '  parmsL[%d]: %.4f (true: %.4f)\n', i, parmsL_est(i), parmsL(i));
end

fprintf(fid, '\nPKPD parameters:\n');
for i = 1:length(parmsPD_est)
    fprintf(fid, '  parmsPD[%d]: %.4f (true: %.4f)\n', i, parmsPD_est(i), parmsPD(i));
end
fprintf(fid, 'ke: %.4f (true: %.4f)\n\n', ke_est, ke);

% Mortality hazard parameters  
fprintf(fid, 'Mortality hazard parameters:\n');
for i = 1:length(parmsY_est)
    fprintf(fid, '  parmsY[%d]: %.4f (true: %.4f)\n', i, parmsY_est(i), parmsY(i));
end
fprintf(fid, '\n');

% Survival outcomes at key time points
key_times = [24, 48, 72, 96, 120, 144, 168];
fprintf(fid, 'SURVIVAL OUTCOMES (with 95%% Bootstrap CIs):\n');
fprintf(fid, '%-8s %-15s %-15s %-15s\n', 'Time(h)', 'Untreated(%)', 'Treated(%)', 'Difference(%)');
fprintf(fid, '%-8s %-15s %-15s %-15s\n', '-------', '-------------', '-----------', '-------------');

for i = 1:length(key_times)
    t_idx = find(t_grid >= key_times(i), 1);
    if ~isempty(t_idx)
        untreated_pct = s0_median(t_idx) * 100;
        treated_pct = s1_median(t_idx) * 100;
        diff_pct = (s1_median(t_idx) - s0_median(t_idx)) * 100;
        
        fprintf(fid, '%-8d %-15.1f %-15.1f %-15.1f\n', ...
            key_times(i), untreated_pct, treated_pct, diff_pct);
    end
end

% Primary endpoint - survival at 168 hours
fprintf(fid, '\nPRIMARY ENDPOINT (168 hours):\n');
fprintf(fid, 'Untreated survival: %.1f%% [%.1f%%, %.1f%%]\n', ...
    s0_median(end)*100, s0_lower(end)*100, s0_upper(end)*100);
fprintf(fid, 'Treated survival: %.1f%% [%.1f%%, %.1f%%]\n', ...
    s1_median(end)*100, s1_lower(end)*100, s1_upper(end)*100);

% Treatment effect
fprintf(fid, '\nTREATMENT EFFECT:\n');
fprintf(fid, 'Absolute risk difference: %.1f%% [%.1f%%, %.1f%%]\n', ...
    treatment_effect_median*100, treatment_effect_lower*100, treatment_effect_upper*100);

% Risk ratio and NNT if beneficial
if treatment_effect_median > 0
    risk_ratio = s1_median(end) / s0_median(end);
    nnt = 1 / treatment_effect_median;
    fprintf(fid, 'Risk ratio: %.2f\n', risk_ratio);
    fprintf(fid, 'Number needed to treat: %.1f\n', nnt);
    fprintf(fid, 'Interpretation: Treatment is BENEFICIAL\n');
elseif treatment_effect_median < 0
    fprintf(fid, 'Number needed to harm: %.1f\n', 1/abs(treatment_effect_median));
    fprintf(fid, 'Interpretation: Treatment is HARMFUL\n');
else
    fprintf(fid, 'Interpretation: No treatment effect\n');
end

% Statistical significance
ci_excludes_zero = (treatment_effect_lower > 0 && treatment_effect_upper > 0) || ...
                   (treatment_effect_lower < 0 && treatment_effect_upper < 0);
fprintf(fid, '\nSTATISTICAL SIGNIFICANCE:\n');
if ci_excludes_zero
    fprintf(fid, 'The 95%% confidence interval excludes zero → STATISTICALLY SIGNIFICANT\n');
else
    fprintf(fid, 'The 95%% confidence interval includes zero → NOT STATISTICALLY SIGNIFICANT\n');
end

% Methods summary
fprintf(fid, '\nMETHODS SUMMARY:\n');
fprintf(fid, '- Data generation: Logit-based discrete-time hazard models\n');
fprintf(fid, '- Causal inference: G-formula/parametric g-computation\n');
fprintf(fid, '- Parameter estimation: Mixed-effects PKPD modeling\n');
fprintf(fid, '- Uncertainty quantification: %d bootstrap resamples\n', Nboot);
fprintf(fid, '- Simulation: RCT emulation with estimated parameters\n');

% Model performance
fprintf(fid, '\nMODEL PERFORMANCE:\n');
param_errors_L = 100 * abs(parmsL_est - parmsL(:)) ./ abs(parmsL(:));
param_errors_PD = 100 * abs(parmsPD_est - parmsPD(:)) ./ abs(parmsPD(:));
param_errors_Y = 100 * abs(parmsY_est - parmsY(:)) ./ abs(parmsY(:));
ke_error = 100 * abs(ke_est - ke) / abs(ke);

fprintf(fid, 'Mean absolute percentage error (MAPE):\n');
fprintf(fid, '- Disease parameters (L): %.1f%%\n', mean(param_errors_L));
fprintf(fid, '- PKPD parameters: %.1f%%\n', mean(param_errors_PD));
fprintf(fid, '- Mortality parameters (Y): %.1f%%\n', mean(param_errors_Y));
fprintf(fid, '- Elimination constant (ke): %.1f%%\n', ke_error);

fclose(fid);
fprintf('Causal survival analysis results exported to: %s\n', filename);