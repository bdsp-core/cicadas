clear all; clc; format compact

% FIGURE: OBSERVATIONAL DATA with NAIVE KAPLAN-MEIER (showing bias without g-formula)
figure(3); clf;
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

%% Calculate disease severity confounding statistics
% Calculate mean disease severity for each group
treated_mean_L = [];
untreated_mean_L = [];
treated_max_L = [];
untreated_max_L = [];
treated_progression_rates = [];
untreated_progression_rates = [];

for i = 1:length(treated_patients)
    patient_id = treated_patients(i);
    patient_data = T0(T0.sid == patient_id, :);
    patient_data = sortrows(patient_data, 't');
    
    % Calculate mean L over first 20 hours (10 time points)
    early_data = patient_data(patient_data.t <= 20, :);
    if ~isempty(early_data)
        treated_mean_L = [treated_mean_L; mean(early_data.L)];
        treated_max_L = [treated_max_L; max(early_data.L)];
        
        % Calculate progression rate (slope of L over first 20 hours)
        if length(early_data.L) > 1
            p = polyfit(early_data.t, early_data.L, 1);
            treated_progression_rates = [treated_progression_rates; p(1)];
        end
    end
end

for i = 1:length(untreated_patients)
    patient_id = untreated_patients(i);
    patient_data = T0(T0.sid == patient_id, :);
    patient_data = sortrows(patient_data, 't');
    
    % Calculate mean L over first 20 hours (10 time points)
    early_data = patient_data(patient_data.t <= 20, :);
    if ~isempty(early_data)
        untreated_mean_L = [untreated_mean_L; mean(early_data.L)];
        untreated_max_L = [untreated_max_L; max(early_data.L)];
        
        % Calculate progression rate (slope of L over first 20 hours)
        if length(early_data.L) > 1
            p = polyfit(early_data.t, early_data.L, 1);
            untreated_progression_rates = [untreated_progression_rates; p(1)];
        end
    end
end

% Calculate fold differences
mean_severity_fold = mean(treated_mean_L) / mean(untreated_mean_L);
max_severity_fold = mean(treated_max_L) / mean(untreated_max_L);
progression_rate_fold = mean(abs(treated_progression_rates)) / mean(abs(untreated_progression_rates));

fprintf('\nCONFOUNDING BY INDICATION STATISTICS:\n');
fprintf('Mean disease severity:\n');
fprintf('  Treated patients: %.4f ± %.4f\n', mean(treated_mean_L), std(treated_mean_L));
fprintf('  Untreated patients: %.4f ± %.4f\n', mean(untreated_mean_L), std(untreated_mean_L));
fprintf('  Fold difference: %.1fx higher in treated patients\n', mean_severity_fold);

fprintf('Maximum disease severity:\n');
fprintf('  Treated patients: %.4f ± %.4f\n', mean(treated_max_L), std(treated_max_L));
fprintf('  Untreated patients: %.4f ± %.4f\n', mean(untreated_max_L), std(untreated_max_L));
fprintf('  Fold difference: %.1fx higher in treated patients\n', max_severity_fold);

fprintf('Disease progression rates:\n');
fprintf('  Treated patients: %.5f ± %.5f\n', mean(abs(treated_progression_rates)), std(abs(treated_progression_rates)));
fprintf('  Untreated patients: %.5f ± %.5f\n', mean(abs(untreated_progression_rates)), std(abs(untreated_progression_rates)));
fprintf('  Fold difference: %.1fx faster in treated patients\n', progression_rate_fold);

% Calculate treatment probability statistics
total_patients = length(treated_patients) + length(untreated_patients);
treatment_prob_mean = length(treated_patients) / total_patients;
fprintf('\nTreatment assignment statistics:\n');
fprintf('  Treatment probability mean: %.3f (%.1f%%)\n', treatment_prob_mean, treatment_prob_mean*100);
fprintf('  Treated: %d patients (%.1f%%)\n', length(treated_patients), length(treated_patients)/total_patients*100);
fprintf('  Untreated: %d patients (%.1f%%)\n', length(untreated_patients), length(untreated_patients)/total_patients*100);

%% Get RCT ground truth curves for comparison
T1_logit = readtable('trialData1.csv');
[s0_true, s1_true, t0_true, t1_true] = fcnPlotKM(T1_logit);

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
plot(t0, s0_true, '--', 'Color', [0.4, 0.4, 0.8], 'LineWidth', 2.5);     % Medium blue dashed
plot(t1, s1_true, '--', 'Color', [0, 0, 0.5], 'LineWidth', 2.5);       % Dark blue dashed

% Naive observational curves - solid lines with blue colors (as step functions)
stairs(t0_naive, s0_naive, '-', 'Color', [0.4, 0.4, 0.8], 'LineWidth', 2.5);   % Medium blue solid
stairs(t1_naive, s1_naive, '-', 'Color', [0, 0, 0.5], 'LineWidth', 2.5);     % Dark blue solid

% Add text labels directly on the curves with UPDATED POSITIONS from manual adjustment
% Treated (Naive)
text(131.21, 0.4020, 'Treated (Naive)', ...
    'Color', [0, 0, 0.5], 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left');

% Treated (RCT)
text(91.87, 0.7253, 'Treated (RCT)', ...
    'Color', [0, 0, 0.5], 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left');

% Untreated (Naive)
text(80.66, 0.6789, 'Untreated (Naive)', ...
    'Color', [0.4, 0.4, 0.8], 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'right');

% Untreated (RCT)
text(61.69, 0.5247, 'Untreated (RCT)', ...
    'Color', [0.4, 0.4, 0.8], 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'right');

% Formatting
xlabel('Time (hours)', 'FontSize', 12);
% Move y-axis label inside the plot
text(5, 0.3, 'Survival Probability', 'FontSize', 12, 'FontWeight', 'bold', 'Rotation', 90);
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
print(gcf, 'Fig4_swimmer_survival_plot_Obs_Naive.pdf', '-dpdf', '-r300');

%% Export observational naive analysis results to text file for paper
filename = sprintf('observational_naive_analysis_results_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));
fid = fopen(filename, 'w');

fprintf(fid, '==========================================================\n');
fprintf(fid, 'OBSERVATIONAL NAIVE ANALYSIS RESULTS FOR PAPER\n');
fprintf(fid, 'Generated on: %s\n', datestr(now));
fprintf(fid, '==========================================================\n\n');

% Study characteristics
fprintf(fid, 'STUDY CHARACTERISTICS:\n');
fprintf(fid, '- Study type: Observational (RCT=0)\n');
fprintf(fid, '- Analysis method: Naive Kaplan-Meier\n');
fprintf(fid, '- Sample size: %d patients\n', n_patients);
fprintf(fid, '- Study period: 168 hours\n');
fprintf(fid, '- Data file: trialData0.csv\n\n');

% Treatment assignment analysis (showing confounding)
fprintf(fid, 'TREATMENT ASSIGNMENT (Biased):\n');
fprintf(fid, '- Treated patients: %d (%.1f%%)\n', length(treated_patients), length(treated_patients)/n_patients*100);
fprintf(fid, '- Untreated patients: %d (%.1f%%)\n', length(untreated_patients), length(untreated_patients)/n_patients*100);
fprintf(fid, '- Assignment method: Biased based on age, SOFA, disease severity\n');
fprintf(fid, '- Note: This creates confounding bias in naive analysis\n\n');

% Add confounding statistics to file
fprintf(fid, 'CONFOUNDING BY INDICATION STATISTICS:\n');
fprintf(fid, 'Mean disease severity (first 20 hours):\n');
fprintf(fid, '- Treated patients: %.4f ± %.4f\n', mean(treated_mean_L), std(treated_mean_L));
fprintf(fid, '- Untreated patients: %.4f ± %.4f\n', mean(untreated_mean_L), std(untreated_mean_L));
fprintf(fid, '- Fold difference: %.1fx higher in treated patients\n', mean_severity_fold);
fprintf(fid, 'Maximum disease severity (first 20 hours):\n');
fprintf(fid, '- Treated patients: %.4f ± %.4f\n', mean(treated_max_L), std(treated_max_L));
fprintf(fid, '- Untreated patients: %.4f ± %.4f\n', mean(untreated_max_L), std(untreated_max_L));
fprintf(fid, '- Fold difference: %.1fx higher in treated patients\n', max_severity_fold);
fprintf(fid, 'Disease progression rates (first 20 hours):\n');
fprintf(fid, '- Treated patients: %.5f ± %.5f per hour\n', mean(abs(treated_progression_rates)), std(abs(treated_progression_rates)));
fprintf(fid, '- Untreated patients: %.5f ± %.5f per hour\n', mean(abs(untreated_progression_rates)), std(abs(untreated_progression_rates)));
fprintf(fid, '- Fold difference: %.1fx faster in treated patients\n', progression_rate_fold);
fprintf(fid, 'Treatment probability statistics:\n');
fprintf(fid, '- Mean assignment probability: %.3f (%.1f%%)\n', treatment_prob_mean, treatment_prob_mean*100);
fprintf(fid, '- This represents substantial deviation from 50%% randomized assignment\n\n');

% Compare to RCT assignment for bias assessment
RCT_expected_treated = n_patients * 0.5;
assignment_bias = length(treated_patients) - RCT_expected_treated;
fprintf(fid, 'TREATMENT ASSIGNMENT BIAS:\n');
fprintf(fid, '- Expected treated (50%% RCT): %.0f patients\n', RCT_expected_treated);
fprintf(fid, '- Actual treated (biased): %d patients\n', length(treated_patients));
fprintf(fid, '- Assignment bias: %+.0f patients (%.1f%% deviation)\n', assignment_bias, assignment_bias/RCT_expected_treated*100);

% Survival outcomes - Naive vs RCT comparison
fprintf(fid, '\nSURVIVAL OUTCOMES COMPARISON:\n');

% Calculate primary endpoints
naive_untreated_survival = s0_naive(end) * 100;
naive_treated_survival = s1_naive(end) * 100;
naive_ate = (s1_naive(end) - s0_naive(end)) * 100;

rct_untreated_survival = s0_true(end) * 100;
rct_treated_survival = s1_true(end) * 100;
rct_ate = (s1_true(end) - s0_true(end)) * 100;

fprintf(fid, '%-20s %-15s %-15s %-15s\n', 'Analysis Method', 'Untreated(%)', 'Treated(%)', 'ATE(%)');
fprintf(fid, '%-20s %-15s %-15s %-15s\n', repmat('-', 1, 20), repmat('-', 1, 15), repmat('-', 1, 15), repmat('-', 1, 15));
fprintf(fid, '%-20s %-15.1f %-15.1f %-15.1f\n', 'RCT (Ground Truth)', rct_untreated_survival, rct_treated_survival, rct_ate);
fprintf(fid, '%-20s %-15.1f %-15.1f %-15.1f\n', 'Naive Observational', naive_untreated_survival, naive_treated_survival, naive_ate);

% Bias quantification
bias_untreated = naive_untreated_survival - rct_untreated_survival;
bias_treated = naive_treated_survival - rct_treated_survival;
bias_ate = naive_ate - rct_ate;

fprintf(fid, '\nBIAS QUANTIFICATION:\n');
fprintf(fid, '- Untreated survival bias: %+.1f%% points\n', bias_untreated);
fprintf(fid, '- Treated survival bias: %+.1f%% points\n', bias_treated);
fprintf(fid, '- Treatment effect bias: %+.1f%% points\n', bias_ate);

% Relative bias
if rct_ate ~= 0
    relative_bias_ate = (bias_ate / rct_ate) * 100;
    fprintf(fid, '- Relative bias in ATE: %+.1f%%\n', relative_bias_ate);
end

% Clinical significance of bias
fprintf(fid, '\nCLINICAL SIGNIFICANCE OF BIAS:\n');
if abs(bias_ate) > 10
    fprintf(fid, '⚠️  SEVERE bias (>10%% points) - naive analysis highly misleading\n');
elseif abs(bias_ate) > 5
    fprintf(fid, '⚠️  MODERATE bias (5-10%% points) - caution required\n');
elseif abs(bias_ate) > 2
    fprintf(fid, '⚠️  MILD bias (2-5%% points) - some concern\n');
else
    fprintf(fid, '✓ MINIMAL bias (<2%% points) - relatively reliable\n');
end

% Direction of bias analysis
if sign(naive_ate) ~= sign(rct_ate) && rct_ate ~= 0
    fprintf(fid, '🚨 CRITICAL: Bias changes treatment effect DIRECTION!\n');
    if naive_ate > 0 && rct_ate < 0
        fprintf(fid, '   Naive analysis suggests benefit, but treatment is actually harmful\n');
    elseif naive_ate < 0 && rct_ate > 0
        fprintf(fid, '   Naive analysis suggests harm, but treatment is actually beneficial\n');
    end
    fprintf(fid, '   → This could lead to dangerous clinical decisions\n');
end

% Detailed mortality analysis
fprintf(fid, '\nMORTALITY ANALYSIS:\n');
untreated_deaths = sum(untreated_events);
treated_deaths = sum(treated_events);
untreated_mortality_rate = untreated_deaths / length(untreated_patients) * 100;
treated_mortality_rate = treated_deaths / length(treated_patients) * 100;

fprintf(fid, 'Naive observational data:\n');
fprintf(fid, '- Untreated deaths: %d/%d (%.1f%%)\n', untreated_deaths, length(untreated_patients), untreated_mortality_rate);
fprintf(fid, '- Treated deaths: %d/%d (%.1f%%)\n', treated_deaths, length(treated_patients), treated_mortality_rate);

if treated_mortality_rate > 0 && untreated_mortality_rate > 0
    naive_relative_risk = treated_mortality_rate / untreated_mortality_rate;
    fprintf(fid, '- Naive relative risk: %.2f\n', naive_relative_risk);
end

% Key time points comparison
key_times = [24, 48, 72, 96, 120, 144, 168];
fprintf(fid, '\nSURVIVAL COMPARISON AT KEY TIME POINTS:\n');
fprintf(fid, '%-8s %-12s %-12s %-12s %-12s %-12s\n', 'Time(h)', 'RCT Untrt', 'Naive Untrt', 'RCT Trt', 'Naive Trt', 'Bias ATE');
fprintf(fid, '%-8s %-12s %-12s %-12s %-12s %-12s\n', '-------', '----------', '-----------', '--------', '---------', '--------');

for i = 1:length(key_times)
    time_point = key_times(i);
    
    % Find closest time indices for RCT
    [~, rct_idx0] = min(abs(t0_true - time_point));
    [~, rct_idx1] = min(abs(t1_true - time_point));
    
    % Find closest time indices for naive (may not exist for all time points)
    naive_untrt_surv = NaN;
    naive_trt_surv = NaN;
    
    % For naive curves, find survival at time point
    if time_point <= max(t0_naive)
        naive_idx0 = find(t0_naive <= time_point, 1, 'last');
        if ~isempty(naive_idx0)
            naive_untrt_surv = s0_naive(naive_idx0) * 100;
        end
    end
    
    if time_point <= max(t1_naive)
        naive_idx1 = find(t1_naive <= time_point, 1, 'last');
        if ~isempty(naive_idx1)
            naive_trt_surv = s1_naive(naive_idx1) * 100;
        end
    end
    
    rct_untrt_surv = s0_true(rct_idx0) * 100;
    rct_trt_surv = s1_true(rct_idx1) * 100;
    
    % Calculate bias in ATE at this time point
    if ~isnan(naive_untrt_surv) && ~isnan(naive_trt_surv)
        rct_ate_timepoint = rct_trt_surv - rct_untrt_surv;
        naive_ate_timepoint = naive_trt_surv - naive_untrt_surv;
        bias_ate_timepoint = naive_ate_timepoint - rct_ate_timepoint;
        
        fprintf(fid, '%-8d %-12.1f %-12.1f %-12.1f %-12.1f %-12.1f\n', ...
            time_point, rct_untrt_surv, naive_untrt_surv, rct_trt_surv, naive_trt_surv, bias_ate_timepoint);
    else
        fprintf(fid, '%-8d %-12.1f %-12s %-12.1f %-12s %-12s\n', ...
            time_point, rct_untrt_surv, 'N/A', rct_trt_surv, 'N/A', 'N/A');
    end
end

% Confounding analysis
fprintf(fid, '\nCONFOUNDING ANALYSIS:\n');
fprintf(fid, 'Sources of confounding in observational data:\n');
fprintf(fid, '- Age: Older, sicker patients more likely to receive treatment\n');
fprintf(fid, '- SOFA score: Higher severity patients preferentially treated\n');
fprintf(fid, '- Disease severity: Initial L0 levels influence treatment decisions\n');
fprintf(fid, '- Indication bias: Treatment given to patients most likely to benefit\n');
fprintf(fid, '- Contraindication bias: Treatment withheld from high-risk patients\n');

% Statistical considerations
fprintf(fid, '\nSTATISTICAL CONSIDERATIONS:\n');
fprintf(fid, '- Naive Kaplan-Meier assumes random treatment assignment\n');
fprintf(fid, '- Violation of this assumption leads to biased estimates\n');
fprintf(fid, '- Confounders are not adjusted for in naive analysis\n');
fprintf(fid, '- Results cannot be interpreted as causal effects\n');

% Comparison with ideal scenarios
fprintf(fid, '\nCOMPARISON WITH IDEAL SCENARIOS:\n');

% Number needed to treat comparison
if rct_ate ~= 0 && naive_ate ~= 0
    rct_nnt = 100 / abs(rct_ate);
    naive_nnt = 100 / abs(naive_ate);
    nnt_bias = naive_nnt - rct_nnt;
    
    fprintf(fid, 'Number needed to treat:\n');
    fprintf(fid, '- RCT (true): %.1f patients\n', rct_nnt);
    fprintf(fid, '- Naive estimate: %.1f patients\n', naive_nnt);
    fprintf(fid, '- Bias in NNT: %+.1f patients\n', nnt_bias);
end

% Clinical decision making implications
fprintf(fid, '\nCLINICAL DECISION MAKING IMPLICATIONS:\n');

if abs(bias_ate) > 5
    fprintf(fid, '❌ DANGEROUS: Naive analysis would lead to wrong clinical decisions\n');
    fprintf(fid, '   - High risk of patient harm from biased treatment recommendations\n');
    fprintf(fid, '   - Causal inference methods are essential\n');
elseif abs(bias_ate) > 2
    fprintf(fid, '⚠️  CAUTION: Naive analysis may mislead clinical decisions\n');
    fprintf(fid, '   - Consider causal inference methods for better estimates\n');
    fprintf(fid, '   - Additional validation recommended\n');
else
    fprintf(fid, '✓ ACCEPTABLE: Bias is minimal, naive analysis may be adequate\n');
    fprintf(fid, '   - Low risk from confounding in this scenario\n');
end

% Methodological lessons
fprintf(fid, '\nMETHODOLOGICAL LESSONS:\n');
fprintf(fid, '1. Observational data can be severely biased without proper adjustment\n');
fprintf(fid, '2. Treatment assignment patterns create systematic differences between groups\n');
fprintf(fid, '3. Naive survival analysis ignores these systematic differences\n');
fprintf(fid, '4. RCT provides unbiased comparison for evaluating observational methods\n');
fprintf(fid, '5. G-formula and other causal methods are needed to correct bias\n');

% Figure generation details
fprintf(fid, '\nFIGURE GENERATION:\n');
fprintf(fid, '- Figure type: Observational swimmer plots with naive survival curves\n');
fprintf(fid, '- Output file: Fig4_swimmer_survival_plot_Obs_Naive.pdf\n');
fprintf(fid, '- Format: PDF vector graphics (300 DPI)\n');
fprintf(fid, '- Dimensions: 6 × 9 inches\n');
fprintf(fid, '- Curve styles: Solid lines (naive), dashed lines (RCT reference)\n');
fprintf(fid, '- Colors: Blue tones matching RCT figure for comparison\n');

% Data quality and validity
fprintf(fid, '\nDATA QUALITY AND VALIDITY:\n');
total_observations = size(T0,1);
fprintf(fid, '- Total observations: %d\n', total_observations);
fprintf(fid, '- Patients analyzed: %d\n', n_patients);
fprintf(fid, '- Average observations per patient: %.1f\n', total_observations/n_patients);
fprintf(fid, '- Data completeness: Assumed 100%% (simulation data)\n');

% Treatment switching analysis
fprintf(fid, '\nTREATMENT PATTERNS:\n');
% Count patients who switch treatments during study
switchers = 0;
for i = 1:n_patients
    patient_id = unique_patients(i);
    patient_data = T0(T0.sid == patient_id, :);
    if length(unique(patient_data.Rx)) > 1
        switchers = switchers + 1;
    end
end

fprintf(fid, '- Patients switching treatment: %d (%.1f%%)\n', switchers, switchers/n_patients*100);
fprintf(fid, '- Analysis method: Initial treatment assignment (as-randomized)\n');
fprintf(fid, '- Note: Treatment switches add complexity not captured in naive analysis\n');

% Research implications
fprintf(fid, '\nRESEARCH IMPLICATIONS:\n');
fprintf(fid, '- Demonstrates the need for causal inference in observational studies\n');
fprintf(fid, '- Quantifies the magnitude of confounding bias\n');
fprintf(fid, '- Validates the importance of RCTs as gold standard\n');
fprintf(fid, '- Motivates development of bias correction methods\n');

% Technical specifications
fprintf(fid, '\nTECHNICAL SPECIFICATIONS:\n');
fprintf(fid, '- Analysis software: MATLAB\n');
fprintf(fid, '- Survival analysis: ecdf function with censoring\n');
fprintf(fid, '- Visualization: fcnSingleSwimmerPlot_v4 + custom curves\n');
fprintf(fid, '- Statistical method: Kaplan-Meier estimator\n');
fprintf(fid, '- Censoring: Right censoring at end of observation\n');

fclose(fid);
fprintf('Observational naive analysis results exported to: %s\n', filename);