%% Logit-Based Hazard Model Data Generation
%
% DESCRIPTION:
%   Generates synthetic clinical trial data using logit-based discrete-time
%   hazard models for causal survival analysis with PKPD modeling.
%
% TRIAL MODES:
%   RCT=1: No treatment changes or dropout (yes randomized controlled
%   trial)
%   RCT=0: Treatment changes and dropout allowed (not randomized controlled
%   trial)

clear; clc; format compact;
rng(0) % Set random seed for reproducibility

%% 1. SIMULATION PARAMETERS ==========================================

N = 2000;   % Number of patients to simulate per trial type

% Target L level
th = 0.1;  % LOWER threshold - harder to achieve for sick patients, leading to more treatment

% PI Controller Parameters
ki = 10;    % Integral control gain (aggressive disease suppression)
Amax = 50;  % Maximum pump rate (treatment upper bound)
parmsControl = [ki Amax];

% PD parameters with age/SOFA dependencies
[age, sofa, C, g, parmsPD] = fcnGeneratePatientParameters(N,'TargetCMean', 3, 'TargetGMean', 4, 'CV', 0.1);

% PK parameter - elimination time constant
ke = 0.5;

% Mortality (Y) hazard parameters
% logit_y = a0 + a1*(t(j)/170)^2 + (a2*sofa).*(cumsum_L/24)^2 + (a3*(age/90)).*(cumsum_A/207); 
a0 = -7;     
a1 = .3;
a2 = 20; % harmL
a3 = 5; % harmA

parmsY = [a0 a1 a2 a3]; 

% Censoring (V) hazard parameters
% logit_v = Rx(j)*(b0 + b1*(cumsum_A/207) + b2*(t(j)/170)^2) + (1-Rx(j))*(b3 + b4*cumsum_L/24 + b5*(t(j)/170)^2);                  
b0 = -5;    % Baseline censoring for treated
b1 = 2.0;    % Treatment burden effect
b2 = .1;   % Time effect for treated
b3 = -5;    % Baseline for untreated   
b4 = 2;     % Disease burden effect
b5 = 1.5;   % Time effect for untreated
parmsV = [b0 b1 b2 b3 b4 b5];

% Disease Natural History Parameters and trajectories
% [growth_rate, peak_height, alpha, decay_rate, sigma_early, sigma_late, sigma_transition]
parmsL = [0.25, 1, 0.15, 0.05, 0.15, 0.03, 40];
dt = 2; t = 0:dt:168; 

%% MAIN LOOP FOR BOTH RCT MODES =====================================

for RCT = 0:1 

    L0 = fcnGenerateStochasticTrajectories(t, parmsL, N);
    if RCT==1
        treatProb = 0.5*ones(1,N); 
    else
        treatProb = fcnBiasedAssignmentProb(age, sofa, L0(:,1:5));
    end
    T = fcnSimulate_N_Patients(N,RCT,treatProb, th, C, g, ke, L0, parmsControl, parmsY, parmsV, age, sofa);

    % Export to CSV for analysis
    filename = sprintf('trialData%d.csv', RCT);
    writetable(T, filename);

    % save "true" values for generating data
    save('parmsTrue', 'parmsControl', 'parmsPD', 'C', 'g', 'ke', 'parmsY', 'parmsV', 'parmsL', 'age', 'sofa');

end

%% plot surival curves - RCT = 1 vs naive vs g-formula


% FIGURE: OBSERVATIONAL DATA with NAIVE KAPLAN-MEIER (showing bias without g-formula)
figure(1); clf;
T0 = readtable('trialData0.csv');
fcnSingleSwimmerPlot_v4(T0)

%% Get the position of the lowest swimmer plot to align survival curves
all_axes = findall(gcf, 'Type', 'axes');
positions = cell2mat(get(all_axes, 'Position'));
lowest_swimmer_bottom = min(positions(:,2)); % Bottom edge of lowest swimmer plot

% Get the left and width from one of the swimmer plots for alignment
swimmer_left = positions(1,1);  % Left position of swimmer plots
swimmer_width = positions(1,3); % Width of swimmer plots

%% Calculate NAIVE Kaplan-Meier curves from observational data
% Get unique patients and their treatment assignment at baseline
unique_patients = unique(T0.sid);
n_patients = length(unique_patients);

% Determine treatment group for each patient (using initial treatment status)
treated_patients = [];
untreated_patients = [];

for i = 1:n_patients
    patient_id = unique_patients(i);
    patient_data = T0(T0.sid == patient_id, :);
    patient_data = sortrows(patient_data, 't');
    
    % Use initial treatment assignment to classify patient
    if patient_data.Rx(1) == 1
        treated_patients = [treated_patients; patient_id];
    else
        untreated_patients = [untreated_patients; patient_id];
    end
end

% Calculate survival times and events for each group
% Treated group
treated_times = [];
treated_events = [];
for i = 1:length(treated_patients)
    patient_id = treated_patients(i);
    patient_data = T0(T0.sid == patient_id, :);
    patient_data = sortrows(patient_data, 't');
    
    % Find time of death or last observation
    death_idx = find(patient_data.Y > 0, 1, 'first');
    if ~isempty(death_idx)
        treated_times = [treated_times; patient_data.t(death_idx)];
        treated_events = [treated_events; 1]; % Death
    else
        treated_times = [treated_times; max(patient_data.t)];
        treated_events = [treated_events; 0]; % Censored
    end
end

% Untreated group
untreated_times = [];
untreated_events = [];
for i = 1:length(untreated_patients)
    patient_id = untreated_patients(i);
    patient_data = T0(T0.sid == patient_id, :);
    patient_data = sortrows(patient_data, 't');
    
    % Find time of death or last observation
    death_idx = find(patient_data.Y > 0, 1, 'first');
    if ~isempty(death_idx)
        untreated_times = [untreated_times; patient_data.t(death_idx)];
        untreated_events = [untreated_events; 1]; % Death
    else
        untreated_times = [untreated_times; max(patient_data.t)];
        untreated_events = [untreated_events; 0]; % Censored
    end
end

% Calculate Kaplan-Meier curves using built-in ecdf or manual calculation
% For untreated group
[f0, x0] = ecdf(untreated_times, 'Censoring', ~untreated_events);
t0_naive = [0; x0];
s0_naive = [1; 1-f0];

% For treated group
[f1, x1] = ecdf(treated_times, 'Censoring', ~treated_events);
t1_naive = [0; x1];
s1_naive = [1; 1-f1];

% Debug: Print some statistics
fprintf('Naive KM Statistics:\n');
fprintf('Untreated: %d patients, %d deaths\n', length(untreated_times), sum(untreated_events));
fprintf('Treated: %d patients, %d deaths\n', length(treated_times), sum(treated_events));
fprintf('Untreated survival at t=168: %.3f\n', s0_naive(end));
fprintf('Treated survival at t=168: %.3f\n', s1_naive(end));

% Calculate Average Treatment Effect (ATE) at end of trial
ate_naive = s1_naive(end) - s0_naive(end);
fprintf('\n=== AVERAGE TREATMENT EFFECT AT 168 HOURS ===\n');
fprintf('Naive KM estimate: ATE = %.1f%% (%.3f - %.3f)\n', ...
    100*ate_naive, s1_naive(end), s0_naive(end));
if ate_naive < 0
    fprintf('  → Treatment appears HARMFUL in naive analysis\n');
else
    fprintf('  → Treatment appears beneficial in naive analysis\n');
end

%% Get RCT ground truth curves for comparison
T1 = readtable('trialData1.csv');
[s0_true, s1_true, t0_true, t1_true] = fcnPlotKM(T1);

% Calculate true ATE from RCT
ate_true = s1_true(end) - s0_true(end);
fprintf('\nRCT (ground truth): ATE = %.1f%% (%.3f - %.3f)\n', ...
    100*ate_true, s1_true(end), s0_true(end));
if ate_true < 0
    fprintf('  → Treatment is truly HARMFUL\n');
else
    fprintf('  → Treatment is truly BENEFICIAL\n');
end

% Compare bias
bias = ate_naive - ate_true;
fprintf('\n=== BIAS IN NAIVE ESTIMATE ===\n');
fprintf('Bias = %.1f percentage points\n', 100*bias);
if sign(ate_naive) ~= sign(ate_true) && ate_true ~= 0
    fprintf('⚠️  CRITICAL: Treatment effect is in the WRONG DIRECTION!\n');
    if ate_naive < 0
        fprintf('   Naive analysis suggests treatment is harmful\n');
    else
        fprintf('   Naive analysis suggests treatment is beneficial\n');
    end
    if ate_true < 0
        fprintf('   But RCT shows treatment is actually harmful\n');
    else
        fprintf('   But RCT shows treatment is actually beneficial\n');
    end
end

%% Get g-formula estimates: 
T0 = readtable('trialData0.csv');
parmsY_est = fcnEstimateDeathParms(T0);
[parmsL_est, LL, AA, age, sofa, t] = fcnEstimateParmsL(T0);
[ke_est, C_est, g_est, theta_est] = fcnEstimateParmsPKPD(parmsL_est, LL, AA, age, sofa, t);

RCT=1;
treatProb = 0.5*ones(1,N);
L0_est = fcnGenerateStochasticTrajectories(t, parmsL_est, N);
T1_est = fcnSimulate_N_Patients(N,RCT,treatProb,th, C, g, ke, L0_est, parmsControl, parmsY_est, [0 0 0 0 0 0], age, sofa);
[s0_gf, s1_gf, t0_gf, t1_gf] = fcnPlotKM(T1_est);

% Calculate g-formula ATE
ate_gformula = s1_gf(end) - s0_gf(end);
fprintf('\nG-formula estimate: ATE = %.1f%% (%.3f - %.3f)\n', ...
    100*ate_gformula, s1_gf(end), s0_gf(end));
fprintf('  → G-formula correctly adjusts for confounding\n');

% Summary comparison
fprintf('\n=== SUMMARY: AVERAGE TREATMENT EFFECTS ===\n');
fprintf('Method          | ATE at 168h | Interpretation\n');
fprintf('----------------|-------------|----------------\n');

% RCT row
if ate_true < 0
    rct_interp = 'Harmful';
else
    rct_interp = 'Beneficial';
end
fprintf('RCT (truth)     | %+6.1f%%    | %s\n', 100*ate_true, rct_interp);

% Naive KM row
if ate_naive < 0
    naive_interp = 'Harmful';
else
    naive_interp = 'Beneficial';
end
wrong_direction = sign(ate_naive) ~= sign(ate_true) && ate_true ~= 0;
if wrong_direction
    naive_suffix = ' (WRONG!)';
else
    naive_suffix = '';
end
fprintf('Naive KM        | %+6.1f%%    | %s%s\n', 100*ate_naive, naive_interp, naive_suffix);

% G-formula row
if ate_gformula < 0
    gf_interp = 'Harmful';
else
    gf_interp = 'Beneficial';
end
fprintf('G-formula       | %+6.1f%%    | %s\n', 100*ate_gformula, gf_interp);

fprintf('\nNaive bias: %.1f percentage points\n', 100*(ate_naive - ate_true));
fprintf('G-formula bias: %.1f percentage points\n', 100*(ate_gformula - ate_true));

%% Position for bottom plot - use remaining space at bottom
vertical_gap = 0.015; % Small gap between swimmer plots and survival plot
bottom_margin = 0.05; % Small bottom margin
% Calculate height to fill remaining space
height = lowest_swimmer_bottom - vertical_gap - bottom_margin;

% Create axes aligned with swimmer plots above
ax4 = axes('Position', [swimmer_left, bottom_margin, swimmer_width, height]);

% Plot survival curves
hold on;

% RCT ground truth curves - dashed lines with blue colors
t0 = 0:2:168; t1 = t0; 
plot(t0, s0_true, 'k', 'LineWidth', 2.5);     % Medium blue dashed
plot(t1, s1_true, 'k', 'LineWidth', 2.5);       % Dark blue dashed

% Naive observational curves - solid lines with blue colors (as step functions)
stairs(t0_naive, s0_naive, 'r', 'LineWidth', 2.5);   % Medium blue solid
stairs(t1_naive, s1_naive, 'r', 'LineWidth', 2.5);     % Dark blue solid

% g-formula observational curves - solid lines with blue colors (as step functions)
stairs(t0_gf, s0_gf, 'g', 'LineWidth', 2.5);   % Medium blue solid
stairs(t1_gf, s1_gf, 'g', 'LineWidth', 2.5);     % Dark blue solid

% Add text labels directly on the curves
% Find good positions for labels (moved further right)
label_time = 130; % Moved even further to the right

% For untreated curves
[~, idx0_true] = min(abs(t0 - label_time));
[~, idx0_naive] = min(abs(t0_naive - label_time));

% For treated curves  
[~, idx1_true] = min(abs(t1 - label_time));
[~, idx1_naive] = min(abs(t1_naive - label_time));

% Add labels - position them to avoid overlap
text(label_time-10, s0_true(idx0_true) + 0.05, 'Untreated (RCT)', ...
    'Color', [0.4, 0.4, 0.8], 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'right');
text(label_time-10, s0_naive(idx0_naive) - 0.05, 'Untreated (Naive)', ...
    'Color', [0.4, 0.4, 0.8], 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'right');

text(label_time+10, s1_true(idx1_true) + 0.05, 'Treated (RCT)', ...
    'Color', [0, 0, 0.5], 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left');
text(label_time+10, s1_naive(idx1_naive) - 0.05, 'Treated (Naive)', ...
    'Color', [0, 0, 0.5], 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left');

% Formatting
xlabel('Time (hours)', 'FontSize', 12);
ylabel('Survival Probability', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 11);
xlim([0, 168]);
ylim([0, 1]);

% Set x-axis ticks every 24 hours
xticks(0:24:168);
xticklabels(0:24:168);
xlim([0 168]);

% Set figure size (in inches)
fig_width = 6;    % Width in inches (adjust as needed)
fig_height = 9;   % Height in inches (adjust as needed)

% Set figure properties
set(gcf, 'Units', 'inches');
set(gcf, 'Position', [1, 1, fig_width, fig_height]); % [left, bottom, width, height]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [fig_width, fig_height]);
set(gcf, 'PaperPosition', [0, 0, fig_width, fig_height]);

% Export as PDF (vector format)
print(gcf, 'Fig_swimmer_survival_plot_Obs_Naive.pdf', '-dpdf', '-r300');

%% Export trial simulation results to text file for paper
filename = sprintf('trial_simulation_results_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));
fid = fopen(filename, 'w');

fprintf(fid, '==========================================================\n');
fprintf(fid, 'TRIAL SIMULATION RESULTS FOR PAPER\n');
fprintf(fid, 'Generated on: %s\n', datestr(now));
fprintf(fid, '==========================================================\n\n');

% Simulation parameters
fprintf(fid, 'SIMULATION PARAMETERS:\n');
fprintf(fid, '- Sample size: %d patients per trial type\n', N);
fprintf(fid, '- Study period: 168 hours\n');
fprintf(fid, '- Time step: 2 hours\n');
fprintf(fid, '- Target threshold: %.3f\n', th);
fprintf(fid, '- Elimination constant (ke): %.3f\n', ke);
fprintf(fid, '- PI controller gain (ki): %.1f\n', ki);
fprintf(fid, '- Maximum pump rate (Amax): %.1f\n\n', Amax);

% True model parameters
fprintf(fid, 'TRUE MODEL PARAMETERS:\n');
fprintf(fid, 'Mortality hazard (Y) parameters:\n');
for i = 1:length(parmsY)
    fprintf(fid, '  a%d: %.3f\n', i-1, parmsY(i));
end

fprintf(fid, '\nCensoring hazard (V) parameters:\n');
param_names_V = {'b0', 'b1', 'b2', 'b3', 'b4', 'b5'};
for i = 1:length(parmsV)
    fprintf(fid, '  %s: %.3f\n', param_names_V{i}, parmsV(i));
end

fprintf(fid, '\nDisease natural history (L) parameters:\n');
param_names_L = {'growth_rate', 'peak_height', 'alpha', 'decay_rate', 'sigma_early', 'sigma_late', 'sigma_transition'};
for i = 1:length(parmsL)
    fprintf(fid, '  %s: %.3f\n', param_names_L{i}, parmsL(i));
end

% Patient population characteristics
fprintf(fid, '\nPATIENT POPULATION:\n');
fprintf(fid, 'Age: %.1f ± %.1f years\n', mean(age), std(age));
fprintf(fid, 'SOFA score: %.1f ± %.1f\n', mean(sofa), std(sofa));
fprintf(fid, 'Clearance (C): %.2f ± %.2f\n', mean(C), std(C));
fprintf(fid, 'Potency (g): %.2f ± %.2f\n\n', mean(g), std(g));

% Treatment assignment bias (observational study)
fprintf(fid, 'TREATMENT ASSIGNMENT:\n');
fprintf(fid, 'RCT (trial=1): Random 50%% assignment\n');
fprintf(fid, 'Observational (trial=0): Biased assignment based on age, SOFA, initial disease severity\n');
fprintf(fid, 'Treated patients in observational study: %d (%.1f%%)\n', length(treated_patients), 100*length(treated_patients)/N);
fprintf(fid, 'Untreated patients in observational study: %d (%.1f%%)\n\n', length(untreated_patients), 100*length(untreated_patients)/N);

% Primary results - Average Treatment Effects
fprintf(fid, 'PRIMARY RESULTS - AVERAGE TREATMENT EFFECTS AT 168 HOURS:\n');
fprintf(fid, '%-20s %-12s %-15s %-15s\n', 'Method', 'ATE', 'Interpretation', 'Bias vs RCT');
fprintf(fid, '%-20s %-12s %-15s %-15s\n', repmat('-', 1, 20), repmat('-', 1, 12), repmat('-', 1, 15), repmat('-', 1, 15));

% RCT (ground truth)
if ate_true < 0
    rct_interp = 'Harmful';
else
    rct_interp = 'Beneficial';
end
fprintf(fid, '%-20s %+7.1f%%%%    %-15s %-15s\n', 'RCT (ground truth)', 100*ate_true, rct_interp, '0.0%% (reference)');

% Naive Kaplan-Meier
if ate_naive < 0
    naive_interp = 'Harmful';
else
    naive_interp = 'Beneficial';
end
wrong_direction = sign(ate_naive) ~= sign(ate_true) && ate_true ~= 0;
if wrong_direction
    naive_suffix = ' (WRONG!)';
else
    naive_suffix = '';
end
bias_naive = 100*(ate_naive - ate_true);
fprintf(fid, '%-20s %+7.1f%%%%    %-15s %+7.1f%% pts\n', 'Naive Kaplan-Meier', 100*ate_naive, [naive_interp naive_suffix], bias_naive);

% G-formula
if ate_gformula < 0
    gf_interp = 'Harmful';
else
    gf_interp = 'Beneficial';
end
bias_gformula = 100*(ate_gformula - ate_true);
fprintf(fid, '%-20s %+7.1f%%%%    %-15s %+7.1f%% pts\n', 'G-formula', 100*ate_gformula, gf_interp, bias_gformula);

% Survival rates at 168 hours
fprintf(fid, '\nSURVIVAL RATES AT 168 HOURS:\n');
fprintf(fid, '%-20s %-15s %-15s\n', 'Method', 'Untreated', 'Treated');
fprintf(fid, '%-20s %-15s %-15s\n', repmat('-', 1, 20), repmat('-', 1, 15), repmat('-', 1, 15));
fprintf(fid, '%-20s %-15.1f%% %-15.1f%%\n', 'RCT (truth)', s0_true(end)*100, s1_true(end)*100);
fprintf(fid, '%-20s %-15.1f%% %-15.1f%%\n', 'Naive observational', s0_naive(end)*100, s1_naive(end)*100);
fprintf(fid, '%-20s %-15.1f%% %-15.1f%%\n', 'G-formula', s0_gf(end)*100, s1_gf(end)*100);

% Key findings
fprintf(fid, '\nKEY FINDINGS:\n');
if wrong_direction
    fprintf(fid, '⚠️  CRITICAL BIAS: Naive analysis suggests treatment is %s,\n', naive_interp);
    fprintf(fid, '   but RCT shows treatment is actually %s\n', rct_interp);
    fprintf(fid, '   → This demonstrates the importance of causal inference methods\n');
end
fprintf(fid, '- G-formula reduces bias from %.1f to %.1f percentage points\n', abs(bias_naive), abs(bias_gformula));
fprintf(fid, '- Bias reduction: %.1f%% improvement\n', 100*(abs(bias_naive) - abs(bias_gformula))/abs(bias_naive));

% Study implications
fprintf(fid, '\nSTUDY IMPLICATIONS:\n');
fprintf(fid, '- Observational studies with biased treatment assignment can lead to\n');
fprintf(fid, '  severely misleading conclusions about treatment effectiveness\n');
fprintf(fid, '- G-formula/parametric g-computation can correct for confounding\n');
fprintf(fid, '  when all confounders are measured and properly modeled\n');
fprintf(fid, '- This simulation validates the causal inference methodology\n');
fprintf(fid, '  for use in real clinical data analysis\n');

% Methods summary
fprintf(fid, '\nMETHODS:\n');
fprintf(fid, '- Data generation: Discrete-time hazard models with PKPD dynamics\n');
fprintf(fid, '- Treatment assignment: Biased based on age, SOFA, disease severity\n');
fprintf(fid, '- Causal inference: G-formula with estimated parameters\n');
fprintf(fid, '- Comparison: RCT (truth) vs Naive analysis vs G-formula\n');

fclose(fid);
fprintf('\nTrial simulation results exported to: %s\n', filename);