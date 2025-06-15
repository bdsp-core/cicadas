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

% Display parameter distributions from empirical data
fprintf('=== PATIENT PARAMETER DISTRIBUTIONS ===\n');
harmE_stats = [mean(T.harmE(T.t==0)), std(T.harmE(T.t==0)), min(T.harmE(T.t==0)), max(T.harmE(T.t==0))];
harmA_stats = [mean(T.harmA(T.t==0)), std(T.harmA(T.t==0)), min(T.harmA(T.t==0)), max(T.harmA(T.t==0))];
fprintf('Disease harm (harmE): mean=%.1f, std=%.1f, range=[%.1f, %.1f]\n', harmE_stats);
fprintf('Treatment harm (harmA): mean=%.1f, std=%.1f, range=[%.1f, %.1f]\n', harmA_stats);
fprintf('======================================\n\n');

% Calculate Kaplan-Meier survival curves
fprintf('Calculating Kaplan-Meier survival curves...\n');
[h_rx1, h_rx0, S_rx1, S_rx0, t_rx1, t_rx0] = fcnEmpiricalSurvivalCurves(T);
fprintf('Kaplan-Meier estimation complete\n');

%% 4. EMPIRICAL HAZARD CALCULATION ==================================

% Calculate empirical hazards directly from longitudinal PKPD data
fprintf('Calculating empirical hazards from longitudinal trajectories...\n');
[h_empirical_rx1, h_empirical_rx0, t_hazard] = calculateEmpiricalHazardsFromData(T, hazard_scale_t1, hazard_scale_t2, b0);
fprintf('Empirical hazard calculation complete\n');

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

% Smooth hazard functions using empirical data with heavy smoothing
h_smooth_treated = interp1(t_hazard, h_empirical_rx1, t_smooth, 'linear', 'extrap');
h_smooth_control = interp1(t_hazard, h_empirical_rx0, t_smooth, 'linear', 'extrap');

% Apply heavy smoothing for clean theoretical curves
window_size_hazard = 201; % Large smoothing window
h_smooth_treated = movmean(h_smooth_treated, window_size_hazard, 'Endpoints', 'shrink');
h_smooth_control = movmean(h_smooth_control, window_size_hazard, 'Endpoints', 'shrink');

% Ensure non-negative hazards
h_smooth_treated = max(0, h_smooth_treated);
h_smooth_control = max(0, h_smooth_control);

% Apply additional smoothing pass
window_size_final = 101;
h_smooth_treated = movmean(h_smooth_treated, window_size_final, 'Endpoints', 'shrink');
h_smooth_control = movmean(h_smooth_control, window_size_final, 'Endpoints', 'shrink');

%% 6. SUMMARY STATISTICS ============================================

% Report curve fitting results
fprintf('\n=== SURVIVAL CURVE ANALYSIS SUMMARY ===\n');
fprintf('Survival probability at end of follow-up (t=%.1fh):\n', max_followup);
fprintf('  Empirical Treated (KM): %.3f\n', S_rx1(end));
fprintf('  Fitted Treated: %.3f\n', S_smooth_treated(end));
fprintf('  Empirical Control (KM): %.3f\n', S_rx0(end));
fprintf('  Fitted Control: %.3f\n', S_smooth_control(end));

fprintf('\nMean hazard rates over follow-up period:\n');
fprintf('  Empirical Treated: %.6f per hour\n', mean(h_empirical_rx1));
fprintf('  Fitted Treated: %.6f per hour\n', mean(h_smooth_treated));
fprintf('  Empirical Control: %.6f per hour\n', mean(h_empirical_rx0));
fprintf('  Fitted Control: %.6f per hour\n', mean(h_smooth_control));

% Calculate treatment effect
hazard_ratio = mean(h_smooth_treated) / mean(h_smooth_control);
survival_difference = S_smooth_treated(end) - S_smooth_control(end);
fprintf('\nTreatment effect estimates:\n');
fprintf('  Hazard ratio (treated/control): %.3f\n', hazard_ratio);
fprintf('  Survival difference at end: %.3f\n', survival_difference);
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

%% Figure 2: Hazard Functions
fprintf('Creating hazard function visualization...\n');
figure(2); clf; 

% Apply light smoothing to empirical hazards for visualization
window_size_empirical = 21;
h_empirical_rx1_smooth = movmean(h_empirical_rx1, window_size_empirical, 'Endpoints', 'shrink');
h_empirical_rx0_smooth = movmean(h_empirical_rx0, window_size_empirical, 'Endpoints', 'shrink');

% Plot empirical hazard functions
plot(t_hazard, h_empirical_rx1_smooth, 'LineWidth', 2, 'Color', [0.8 0.2 0.2], 'DisplayName', 'Empirical Treated');
hold on;
plot(t_hazard, h_empirical_rx0_smooth, 'LineWidth', 2, 'Color', [0.2 0.2 0.8], 'DisplayName', 'Empirical Control');

% Plot fitted smooth hazard functions
plot(t_smooth, h_smooth_treated, '--', 'LineWidth', 3, 'Color', [0.6 0.1 0.1], 'DisplayName', 'Fitted Treated');
plot(t_smooth, h_smooth_control, '--', 'LineWidth', 3, 'Color', [0.1 0.1 0.6], 'DisplayName', 'Fitted Control');

% Formatting
xlabel('Time [hours]');
ylabel('Hazard Rate [per hour]');
title('Empirical vs Fitted Hazard Functions');
legend('Location', 'northwest');
grid on;

% Set appropriate y-axis limits
max_h = max([max(h_empirical_rx1_smooth), max(h_empirical_rx0_smooth), max(h_smooth_treated), max(h_smooth_control)]);
ylim([0 max_h * 1.1]);

set(gcf, 'Color', 'white');
box off;

fprintf('Survival analysis and visualization complete.\n');


%% Helper function to calculate empirical hazards from data
function [h_rx1, h_rx0, t_grid] = calculateEmpiricalHazardsFromData(T, hazard_scale_t1, hazard_scale_t2, b0)
    % Calculate empirical hazards directly from longitudinal data using the model formula
    
    % Get unique time points and patients
    unique_times = sort(unique(T.t));
    t_grid = unique_times(:)';
    
    % Get treated and untreated patients
    treated_patients = unique(T.sid(T.Rx == 1));
    untreated_patients = unique(T.sid(T.Rx == 0));
    
    % Initialize hazard matrices
    h_treated_matrix = zeros(length(treated_patients), length(t_grid));
    h_untreated_matrix = zeros(length(untreated_patients), length(t_grid));
    
    % Calculate hazards for treated patients
    for i = 1:length(treated_patients)
        patient_id = treated_patients(i);
        patient_data = T(T.sid == patient_id, :);
        patient_data = sortrows(patient_data, 't');
        
        % Get patient-specific parameters
        harmE = patient_data.harmE(1);
        harmA = patient_data.harmA(1);
        
        % Calculate hazard at each time point for this patient
        for j = 1:length(t_grid)
            t_current = t_grid(j);
            
            % Find data up to current time
            idx_up_to_t = patient_data.t <= t_current;
            if any(idx_up_to_t)
                L_sum = sum(patient_data.L(idx_up_to_t));
                A_sum = sum(patient_data.A(idx_up_to_t));
                
                % Use exact formula from simulation
                raw_t1 = b0 + harmE*L_sum/18.4 + harmA*A_sum/360;
                t1 = hazard_scale_t1 * raw_t1;
                t2 = hazard_scale_t2 * (t_current/168)^2;
                
                h_treated_matrix(i, j) = t1 + t2;
            end
        end
    end
    
    % Calculate hazards for untreated patients
    for i = 1:length(untreated_patients)
        patient_id = untreated_patients(i);
        patient_data = T(T.sid == patient_id, :);
        patient_data = sortrows(patient_data, 't');
        
        % Get patient-specific parameters
        harmE = patient_data.harmE(1);
        harmA = patient_data.harmA(1);
        
        % Calculate hazard at each time point for this patient
        for j = 1:length(t_grid)
            t_current = t_grid(j);
            
            % Find data up to current time
            idx_up_to_t = patient_data.t <= t_current;
            if any(idx_up_to_t)
                L_sum = sum(patient_data.L(idx_up_to_t));
                A_sum = sum(patient_data.A(idx_up_to_t));
                
                % Use exact formula from simulation
                raw_t1 = b0 + harmE*L_sum/18.4 + harmA*A_sum/360;
                t1 = hazard_scale_t1 * raw_t1;
                t2 = hazard_scale_t2 * (t_current/168)^2;
                
                h_untreated_matrix(i, j) = t1 + t2;
            end
        end
    end
    
    % Pre-compute death times for efficiency
    treated_death_times = inf(length(treated_patients), 1);
    untreated_death_times = inf(length(untreated_patients), 1);
    
    for i = 1:length(treated_patients)
        patient_data = T(T.sid == treated_patients(i), :);
        if any(patient_data.Y > 0)
            death_idx = find(patient_data.Y > 0, 1, 'first');
            treated_death_times(i) = patient_data.t(death_idx);
        end
    end
    
    for i = 1:length(untreated_patients)
        patient_data = T(T.sid == untreated_patients(i), :);
        if any(patient_data.Y > 0)
            death_idx = find(patient_data.Y > 0, 1, 'first');
            untreated_death_times(i) = patient_data.t(death_idx);
        end
    end
    
    % Average across patients in each group, but only include patients still alive at each time
    h_rx1 = zeros(1, length(t_grid));
    h_rx0 = zeros(1, length(t_grid));
    
    for j = 1:length(t_grid)
        t_current = t_grid(j);
        
        % For treated group - only include patients still alive at t_current
        alive_treated_mask = t_current < treated_death_times;
        if any(alive_treated_mask)
            alive_hazards = h_treated_matrix(alive_treated_mask, j);
            alive_hazards = alive_hazards(alive_hazards > 0);
            if ~isempty(alive_hazards)
                h_rx1(j) = mean(alive_hazards);
            end
        end
        
        % For untreated group - only include patients still alive at t_current  
        alive_untreated_mask = t_current < untreated_death_times;
        if any(alive_untreated_mask)
            alive_hazards = h_untreated_matrix(alive_untreated_mask, j);
            alive_hazards = alive_hazards(alive_hazards > 0);
            if ~isempty(alive_hazards)
                h_rx0(j) = mean(alive_hazards);
            end
        end
    end
end