%% a1_SurvivalCurves_RCT.m - Survival and Hazard Analysis for RCT Data
%
% DESCRIPTION:
%   Analyzes synthetic clinical trial data to calculate and visualize 
%   survival and hazard curves. Compares empirical Kaplan-Meier estimates
%   with smoothed theoretical curves for treatment effectiveness assessment.
%
% INPUTS:
%   - trial_data.csv: Longitudinal trial data from a0_GenerateData.m
%
% OUTPUTS:
%   - Figure 1: Kaplan-Meier vs Fitted Survival Curves
%   - Figure 2: Empirical vs Fitted Hazard Functions
%   - Console output: Summary statistics and curve comparisons
%
% WORKFLOW:
%   1. Load trial data and extract model parameters
%   2. Calculate empirical survival curves (Kaplan-Meier)
%   3. Calculate empirical hazard functions from longitudinal data
%   4. Fit smooth theoretical curves to empirical data
%   5. Generate comparative visualizations
%   6. Report summary statistics
%
% METHODS:
%   - Kaplan-Meier estimation for survival curves
%   - PCHIP interpolation for smooth survival curves
%   - Moving average smoothing for hazard functions
%   - Patient-specific hazard calculation using PKPD model
%
% Author: Mitchell McCauley, Brandon Westover
% Created: 2024
% Last Modified: 2025-06-15

clear; clc; format compact;

%% 1. DATA IMPORT AND VALIDATION =====================================
% Import trial data
filename = 'trial_data.csv';
fprintf('Reading trial data from %s...\n', filename);
T = readtable(filename);

% Validate data import
N = length(unique(T.sid));
fprintf('Loaded %d rows of data for %d patients\n', height(T), N);

% Display trial characteristics
n_treated = sum(T.Rx == 1 & T.t == 0);
n_untreated = sum(T.Rx == 0 & T.t == 0);
n_deaths = sum(T.Y > 0);
max_followup = max(T.t);

fprintf('\n=== TRIAL CHARACTERISTICS ===\n');
fprintf('Treatment groups: %d treated (%.1f%%), %d untreated (%.1f%%)\n', ...
    n_treated, 100*n_treated/N, n_untreated, 100*n_untreated/N);
fprintf('Total deaths: %d (%.1f%%)\n', n_deaths, 100*n_deaths/height(T));
fprintf('Maximum follow-up: %.1f hours\n', max_followup);
fprintf('=============================\n\n');

%% 2. MODEL PARAMETERS ===============================================

% Extract parameters from data (consistent with a0_GenerateData.m)
b0 = 0.1;                       % Baseline mortality risk
hazard_scale_t1 = 0.001/2;      % Disease/treatment harm scaling
hazard_scale_t2 = 0.005/2;      % Time-dependent risk scaling

% Note: Patient-specific parameters (harmE, harmA) are now hidden from analysis
fprintf('=== DATA CHARACTERISTICS ===\n');
fprintf('Variables available: sid, t, Rx, L, A, V, Y\n');
fprintf('Note: Rx reflects time-varying treatment adherence\n');
fprintf('Note: Patient harm parameters are hidden for causal inference\n');
fprintf('=============================\n\n');

% Calculate Kaplan-Meier survival curves
fprintf('Calculating Kaplan-Meier survival curves...\n');
[h_rx1, h_rx0, S_rx1, S_rx0, t_rx1, t_rx0] = fcnEmpiricalSurvivalCurves(T);
fprintf('Kaplan-Meier estimation complete\n');

%% 4. EMPIRICAL HAZARD CALCULATION ==================================

% Note: Detailed hazard calculation requires hidden parameters (harmE, harmA)
% For causal inference, we focus on survival curves from observed data
fprintf('Skipping detailed hazard calculation (requires hidden parameters)\n');
fprintf('Analysis will focus on Kaplan-Meier survival curves from observed data\n');

% Fit smooth theoretical curves to empirical data for cleaner visualization
fprintf('Fitting smooth theoretical curves to empirical data...\n');

% Create fine time grid for smooth interpolation
t_smooth = linspace(0, max([max(t_rx1), max(t_rx0)]), 1000);

% Prepare valid data points for curve fitting
valid_idx_rx1 = ~isnan(S_rx1) & S_rx1 > 0 & S_rx1 <= 1;
t_rx1_valid = t_rx1(valid_idx_rx1);
S_rx1_valid = S_rx1(valid_idx_rx1);

valid_idx_rx0 = ~isnan(S_rx0) & S_rx0 > 0 & S_rx0 <= 1;
t_rx0_valid = t_rx0(valid_idx_rx0);
S_rx0_valid = S_rx0(valid_idx_rx0);

% Fit smooth survival curves using PCHIP interpolation
S_smooth_treated = interp1(t_rx1_valid, S_rx1_valid, t_smooth, 'pchip', 'extrap');
S_smooth_control = interp1(t_rx0_valid, S_rx0_valid, t_smooth, 'pchip', 'extrap');

% Enforce survival curve constraints (monotonic decrease, bounded [0,1])
S_smooth_treated = max(0, min(1, S_smooth_treated));
S_smooth_control = max(0, min(1, S_smooth_control));

% Ensure monotonic decrease
for i = 2:length(S_smooth_treated)
    if S_smooth_treated(i) > S_smooth_treated(i-1)
        S_smooth_treated(i) = S_smooth_treated(i-1);
    end
end
for i = 2:length(S_smooth_control)
    if S_smooth_control(i) > S_smooth_control(i-1)
        S_smooth_control(i) = S_smooth_control(i-1);
    end
end

% Note: Hazard function fitting skipped (requires hidden parameters)
% Analysis focuses on survival curves only

%% 6. SUMMARY STATISTICS ============================================

% Report survival analysis results
fprintf('\n=== SURVIVAL CURVE ANALYSIS SUMMARY ===\n');
fprintf('Survival probability at end of follow-up (t=%.1fh):\n', max_followup);
fprintf('  Empirical Treated (KM): %.3f\n', S_rx1(end));
fprintf('  Fitted Treated: %.3f\n', S_smooth_treated(end));
fprintf('  Empirical Control (KM): %.3f\n', S_rx0(end));
fprintf('  Fitted Control: %.3f\n', S_smooth_control(end));

% Calculate treatment effect from survival curves
survival_difference = S_smooth_treated(end) - S_smooth_control(end);
fprintf('\nTreatment effect estimate:\n');
fprintf('  Survival difference at end: %.3f\n', survival_difference);

% Additional summary statistics
n_treated_obs = sum(T.Rx == 1);
n_control_obs = sum(T.Rx == 0);
fprintf('\nObservation counts (including time-varying treatment):\n');
fprintf('  Treated observations: %d\n', n_treated_obs);
fprintf('  Control observations: %d\n', n_control_obs);
fprintf('=======================================\n');

%% 7. VISUALIZATION ==================================================

%% Figure 1: Survival Curves
fprintf('Creating survival curve visualization...\n');
figure(1); clf; 

% Plot empirical Kaplan-Meier curves
stairs(t_rx1, S_rx1, 'LineWidth', 2, 'Color', [0.8 0.2 0.2], 'DisplayName', 'Empirical Treated');
hold on;
stairs(t_rx0, S_rx0, 'LineWidth', 2, 'Color', [0.2 0.2 0.8], 'DisplayName', 'Empirical Control');

% Plot fitted smooth curves
plot(t_smooth, S_smooth_treated, '--', 'LineWidth', 3, 'Color', [0.6 0.1 0.1], 'DisplayName', 'Fitted Treated');
plot(t_smooth, S_smooth_control, '--', 'LineWidth', 3, 'Color', [0.1 0.1 0.6], 'DisplayName', 'Fitted Control');

% Formatting
xlabel('Time [hours]');
ylabel('Survival Probability');
title('Kaplan-Meier vs Fitted Survival Curves');
legend('Location', 'southwest');
grid on;
axis([0 max([max(t_rx1), max(t_rx0)]) 0 1]);
set(gcf, 'Color', 'white');
box off;

fprintf('Survival curve analysis and visualization complete.\n');
fprintf('Note: Hazard function analysis skipped (requires hidden patient parameters)\n');