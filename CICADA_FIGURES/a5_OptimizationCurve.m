clear all; clc; format compact

addpath('..'); % needs access to files one folder up

% FIGURE: OPTIMIZATION CURVES WITH SURVIVAL SUBPLOTS
figure(1); clf;

%% Load data
load('A01Data.mat');
% Contains: A0, A1, th

load('ThreeCurves.mat');
% Contains: S0ref, S1ref (RCT data), S0est, S1est (observational data)
% Each matrix has rows corresponding to th=0, 0.02, 0.1, 0.8

%% Calculate 95% confidence bands for ATE
alpha = 0.05;

% Check if A0 and A1 are matrices (multiple bootstrap samples)
% If rows are bootstrap samples, calculate percentiles across rows (dimension 1)
if size(A0, 1) > 1
    A0_lower = prctile(A0, 100*alpha/2, 1);     % 2.5th percentile across rows
    A0_upper = prctile(A0, 100*(1-alpha/2), 1); % 97.5th percentile across rows
    A0_median = prctile(A0, 50, 1);             % Median across rows
    A1_median = prctile(A1, 50, 1);             % Median across rows
else
    % If single row, just use the values directly
    A0_lower = A0;
    A0_upper = A0;
    A0_median = A0;
    A1_median = A1;
end

%% Set up figure with multiple subplots
fig_width = 10;    % Wider to accommodate 3 subplots
fig_height = 8;    % Taller to fit subplots above main plot

set(gcf, 'Units', 'inches');
set(gcf, 'Position', [1, 1, fig_width, fig_height]);
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [fig_width, fig_height]);
set(gcf, 'PaperPosition', [0, 0, fig_width, fig_height]);
set(gcf, 'Color', 'w');  % Set white background

%% Create subplots for survival curves as insets
% Four survival curve subplots in top row - smaller to emphasize they're insets
subplot_height = 0.23;  % Smaller height for inset appearance
subplot_width = 0.17;   % Smaller width to fit 4 plots within margins
subplot_y = 0.50;       % Lower position to reduce white space
subplot_spacing = 0.055;  % Reduced spacing between insets

% Subplot 1: th = 0
ax1 = axes('Position', [0.1, subplot_y, subplot_width, subplot_height]);
hold on;
% Define time vector
t = 0:2:168; 

% Add only left and bottom box lines for inset appearance
box off;
set(gca, 'LineWidth', 1.5);
set(gca, 'XAxisLocation', 'bottom');
set(gca, 'YAxisLocation', 'left');

% Plot observational curves (row 1 for th=0)
plot(t, S0est(1,:), '-', 'Color', [0.4, 0.4, 0.8], 'LineWidth', 1.5);  % Untreated
plot(t, S1est(1,:), '-', 'Color', [0, 0, 0.5], 'LineWidth', 1.5);      % Treated
% Plot RCT curves (dashed, row 1 for th=0)
plot(t, S0ref(1,:), '--', 'Color', [0.4, 0.4, 0.8], 'LineWidth', 1);
plot(t, S1ref(1,:), '--', 'Color', [0, 0, 0.5], 'LineWidth', 1);
xlabel('Time (h)', 'FontSize', 9);
ylabel('Survival', 'FontSize', 9);
title('\theta = 0', 'FontSize', 10, 'FontWeight', 'bold');
xlim([0, 168]);
ylim([0, 1]);
grid on;
set(gca, 'FontSize', 8);
xticks(0:84:168);

% Subplot 2: th = 0.02
ax2 = axes('Position', [0.1+subplot_width+subplot_spacing, subplot_y, subplot_width, subplot_height]);
hold on;

% Add only left and bottom box lines for inset appearance
box off;
set(gca, 'LineWidth', 1.5);
set(gca, 'XAxisLocation', 'bottom');
set(gca, 'YAxisLocation', 'left');

% Plot observational curves (row 2 for th=0.02)
plot(t, S0est(2,:), '-', 'Color', [0.4, 0.4, 0.8], 'LineWidth', 1.5);
plot(t, S1est(2,:), '-', 'Color', [0, 0, 0.5], 'LineWidth', 1.5);
% Plot RCT curves (dashed, row 2 for th=0.02)
plot(t, S0ref(2,:), '--', 'Color', [0.4, 0.4, 0.8], 'LineWidth', 1);
plot(t, S1ref(2,:), '--', 'Color', [0, 0, 0.5], 'LineWidth', 1);
xlabel('Time (h)', 'FontSize', 9);
title('\theta = 0.02', 'FontSize', 10, 'FontWeight', 'bold');
xlim([0, 168]);
ylim([0, 1]);
grid on;
set(gca, 'FontSize', 8);
set(gca, 'YTickLabel', []);  % Remove y-axis labels for middle plots
xticks(0:84:168);

% Subplot 3: th = 0.1
ax3 = axes('Position', [0.1+2*(subplot_width+subplot_spacing), subplot_y, subplot_width, subplot_height]);
hold on;

% Add only left and bottom box lines for inset appearance
box off;
set(gca, 'LineWidth', 1.5);
set(gca, 'XAxisLocation', 'bottom');
set(gca, 'YAxisLocation', 'left');

% Plot observational curves (row 3 for th=0.1)
plot(t, S0est(3,:), '-', 'Color', [0.4, 0.4, 0.8], 'LineWidth', 1.5);
plot(t, S1est(3,:), '-', 'Color', [0, 0, 0.5], 'LineWidth', 1.5);
% Plot RCT curves (dashed, row 3 for th=0.1)
plot(t, S0ref(3,:), '--', 'Color', [0.4, 0.4, 0.8], 'LineWidth', 1);
plot(t, S1ref(3,:), '--', 'Color', [0, 0, 0.5], 'LineWidth', 1);
xlabel('Time (h)', 'FontSize', 9);
title('\theta = 0.1', 'FontSize', 10, 'FontWeight', 'bold');
xlim([0, 168]);
ylim([0, 1]);
grid on;
set(gca, 'FontSize', 8);
set(gca, 'YTickLabel', []);  % Remove y-axis labels for middle plots
xticks(0:84:168);

% Subplot 4: th = 0.8
ax4 = axes('Position', [0.1+3*(subplot_width+subplot_spacing), subplot_y, subplot_width, subplot_height]);
hold on;

% Add only left and bottom box lines for inset appearance
box off;
set(gca, 'LineWidth', 1.5);
set(gca, 'XAxisLocation', 'bottom');
set(gca, 'YAxisLocation', 'left');

% Plot observational curves (row 4 for th=0.8)
plot(t, S0est(4,:), '-', 'Color', [0.4, 0.4, 0.8], 'LineWidth', 1.5);
plot(t, S1est(4,:), '-', 'Color', [0, 0, 0.5], 'LineWidth', 1.5);
% Plot RCT curves (dashed, row 4 for th=0.8)
plot(t, S0ref(4,:), '--', 'Color', [0.4, 0.4, 0.8], 'LineWidth', 1);
plot(t, S1ref(4,:), '--', 'Color', [0, 0, 0.5], 'LineWidth', 1);
xlabel('Time (h)', 'FontSize', 9);
title('\theta = 0.8', 'FontSize', 10, 'FontWeight', 'bold');
xlim([0, 168]);
ylim([0, 1]);
grid on;
set(gca, 'FontSize', 8);
set(gca, 'YTickLabel', []);  % Remove y-axis labels for right plot
xticks(0:84:168);

% Remove legend text - no labels needed

%% Main ATE optimization plot
main_height = 0.35;
main_width = 0.85;
main_y = 0.1;

ax5 = axes('Position', [0.1, main_y, main_width, main_height]);
hold on;

% Convert to column vectors
th = th(:);
A0_lower = A0_lower(:);
A0_upper = A0_upper(:);
A0_median = A0_median(:);
A1_median = A1_median(:);

% Debug: Check the dimensions and values
fprintf('\n=== Debug Information ===\n');
fprintf('th: %d elements, range [%.3f, %.3f]\n', length(th), min(th), max(th));
fprintf('A0_median: %d elements, range [%.3f, %.3f]\n', length(A0_median), min(A0_median), max(A0_median));
fprintf('A1_median: %d elements, range [%.3f, %.3f]\n', length(A1_median), min(A1_median), max(A1_median));
fprintf('A0 matrix dimensions: %d x %d\n', size(A0,1), size(A0,2));
fprintf('========================\n\n');

% Plot confidence band for observational
load('A01Data.mat');
th=th';

fill([th; flipud(th)], ...
     [A0_lower; flipud(A0_upper)], ...
     [0.4, 0.4, 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none');

% Plot main curves
plot(th, A0_median, '-', 'Color', [0.4, 0.4, 0.8], 'LineWidth', 2.5);
plot(th, A1_median, '-', 'Color', [0, 0, 0.5], 'LineWidth', 2.5);

% Find and mark optimal threshold
[max_A0, idx_A0] = max(A0_median);
th_opt_A0 = th(idx_A0);
plot(th_opt_A0, max_A0, 'o', 'Color', [0.4, 0.4, 0.8], ...
     'MarkerSize', 10, 'MarkerFaceColor', [0.4, 0.4, 0.8]);

% Get current y-axis limits for positioning
yl = ylim;

% Add vertical lines and markers at the four threshold values
% th = 0
plot([0 0], yl, '-', 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1);
idx_0 = find(abs(th - 0) < 0.01, 1);
if ~isempty(idx_0)
    plot(0, A0_median(idx_0), 'o', 'Color', [0.4, 0.4, 0.8], ...
         'MarkerSize', 8, 'MarkerFaceColor', 'w', 'LineWidth', 2);
end

% th = 0.02
plot([0.02 0.02], yl, '-', 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1);
idx_002 = find(abs(th - 0.02) < 0.01, 1);
if ~isempty(idx_002)
    plot(0.02, A0_median(idx_002), 'o', 'Color', [0.4, 0.4, 0.8], ...
         'MarkerSize', 8, 'MarkerFaceColor', 'w', 'LineWidth', 2);
end

% th = 0.1
plot([0.1 0.1], yl, '-', 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1);
idx_01 = find(abs(th - 0.1) < 0.01, 1);
if ~isempty(idx_01)
    plot(0.1, A0_median(idx_01), 'o', 'Color', [0.4, 0.4, 0.8], ...
         'MarkerSize', 8, 'MarkerFaceColor', 'w', 'LineWidth', 2);
end

% th = 0.8
plot([0.8 0.8], yl, '-', 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1);
idx_08 = find(abs(th - 0.8) < 0.01, 1);
if ~isempty(idx_08)
    plot(0.8, A0_median(idx_08), 'o', 'Color', [0.4, 0.4, 0.8], ...
         'MarkerSize', 8, 'MarkerFaceColor', 'w', 'LineWidth', 2);
end

% Add vertical dashed line at optimal threshold
plot([th_opt_A0 th_opt_A0], ylim, '--', 'Color', [0.4, 0.4, 0.8], ...
     'LineWidth', 1.5);

% Formatting
xlabel('Treatment Threshold (\theta)', 'FontSize', 12);
ylabel('Average Treatment Effect (ATE)', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 11);
xlim([-0.05, max(th)]);  % Start at -0.05 to give space for theta=0 marker

% Set y-axis limits with padding (allow negative values)
y_min = min([A0_lower; A1_median; A0_median]) - 0.02;
y_max = max([A0_upper; A1_median; A0_median]) + 0.02;
ylim([y_min, y_max]);

% No overall title

%% Add connecting lines from insets to main plot points
% These are drawn as annotations with right-angle turns for cleaner appearance

% Main plot parameters
main_left = 0.1;
main_width = 0.85;
main_top = 0.45;

% Calculate x-positions on main plot for each theta value
% Account for the new x-axis range from -0.05 to max(th)
x_range = max(th) - (-0.05);
theta_0_x = main_left + main_width * (0 - (-0.05)) / x_range;
theta_002_x = main_left + main_width * (0.02 - (-0.05)) / x_range;
theta_01_x = main_left + main_width * (0.1 - (-0.05)) / x_range;
theta_08_x = main_left + main_width * (0.8 - (-0.05)) / x_range;

% Calculate x-position for theta=30 within each inset (for line starting point)
% Each inset shows 0-168 hours, so theta=30 is at 30/168 of the width
inset_theta30_offset = subplot_width * (30/168);
inset1_theta30 = 0.1 + inset_theta30_offset;
inset2_theta30 = 0.1 + subplot_width + subplot_spacing + inset_theta30_offset;
inset3_theta30 = 0.1 + 2*(subplot_width + subplot_spacing) + inset_theta30_offset;
inset4_theta30 = 0.1 + 3*(subplot_width + subplot_spacing) + inset_theta30_offset;

% Staggered y-positions for horizontal segments (higher left, lower right)
mid_y1 = 0.485;  % Highest for leftmost
mid_y2 = 0.462;  % Lowered more to go under text but above 3rd line
mid_y3 = 0.455;  % Lowered even more to go under the text
mid_y4 = 0.470;  % Between 2nd and 3rd

% Connection from theta=0 inset to main plot (3 segments)
% Vertical down from inset at theta=30
annotation('line', [inset1_theta30, inset1_theta30], [subplot_y, mid_y1], ...
    'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
% Horizontal to target x position
annotation('line', [inset1_theta30, theta_0_x], [mid_y1, mid_y1], ...
    'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
% Vertical down to main plot
annotation('line', [theta_0_x, theta_0_x], [mid_y1, main_top], ...
    'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);

% Connection from theta=0.02 inset to main plot (3 segments)
annotation('line', [inset2_theta30, inset2_theta30], [subplot_y, mid_y2], ...
    'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
annotation('line', [inset2_theta30, theta_002_x], [mid_y2, mid_y2], ...
    'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
annotation('line', [theta_002_x, theta_002_x], [mid_y2, main_top], ...
    'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);

% Connection from theta=0.1 inset to main plot (3 segments)
annotation('line', [inset3_theta30, inset3_theta30], [subplot_y, mid_y3], ...
    'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
annotation('line', [inset3_theta30, theta_01_x], [mid_y3, mid_y3], ...
    'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
annotation('line', [theta_01_x, theta_01_x], [mid_y3, main_top], ...
    'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);

% Connection from theta=0.8 inset to main plot (3 segments)
annotation('line', [inset4_theta30, inset4_theta30], [subplot_y, mid_y4], ...
    'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
annotation('line', [inset4_theta30, theta_08_x], [mid_y4, mid_y4], ...
    'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
annotation('line', [theta_08_x, theta_08_x], [mid_y4, main_top], ...
    'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);

%% Display summary statistics
fprintf('\n=== Optimization Results ===\n');
fprintf('Observational (G-formula):\n');
fprintf('  Optimal threshold: %.3f\n', th_opt_A0);
fprintf('  Maximum ATE: %.3f [%.3f, %.3f]\n', ...
    max_A0, A0_lower(idx_A0), A0_upper(idx_A0));

% Get ATE values at the four specific thresholds
if ~isempty(idx_0)
    ate_0 = A0_median(idx_0);
    ate_0_ci_lower = A0_lower(idx_0);
    ate_0_ci_upper = A0_upper(idx_0);
else
    ate_0 = NaN; ate_0_ci_lower = NaN; ate_0_ci_upper = NaN;
end

if ~isempty(idx_002)
    ate_002 = A0_median(idx_002);
    ate_002_ci_lower = A0_lower(idx_002);
    ate_002_ci_upper = A0_upper(idx_002);
else
    ate_002 = NaN; ate_002_ci_lower = NaN; ate_002_ci_upper = NaN;
end

if ~isempty(idx_01)
    ate_01 = A0_median(idx_01);
    ate_01_ci_lower = A0_lower(idx_01);
    ate_01_ci_upper = A0_upper(idx_01);
else
    ate_01 = NaN; ate_01_ci_lower = NaN; ate_01_ci_upper = NaN;
end

if ~isempty(idx_08)
    ate_08 = A0_median(idx_08);
    ate_08_ci_lower = A0_lower(idx_08);
    ate_08_ci_upper = A0_upper(idx_08);
else
    ate_08 = NaN; ate_08_ci_lower = NaN; ate_08_ci_upper = NaN;
end

fprintf('\n=== ATE Values at Displayed Thresholds (for Figure Caption) ===\n');
fprintf('At θ = 0.00 (treat all):     ATE = %.3f [95%% CI: %.3f, %.3f]\n', ate_0, ate_0_ci_lower, ate_0_ci_upper);
fprintf('At θ = 0.02 (near optimal):  ATE = %.3f [95%% CI: %.3f, %.3f]\n', ate_002, ate_002_ci_lower, ate_002_ci_upper);
fprintf('At θ = 0.10:                 ATE = %.3f [95%% CI: %.3f, %.3f]\n', ate_01, ate_01_ci_lower, ate_01_ci_upper);
fprintf('At θ = 0.80 (restrictive):   ATE = %.3f [95%% CI: %.3f, %.3f]\n', ate_08, ate_08_ci_lower, ate_08_ci_upper);

fprintf('\nSurvival at 168h for displayed thresholds:\n');
fprintf('th=0.00: Obs Untreated=%.3f, Treated=%.3f | RCT Untreated=%.3f, Treated=%.3f\n', ...
    S0est(1,end), S1est(1,end), S0ref(1,end), S1ref(1,end));
fprintf('th=0.02: Obs Untreated=%.3f, Treated=%.3f | RCT Untreated=%.3f, Treated=%.3f\n', ...
    S0est(2,end), S1est(2,end), S0ref(2,end), S1ref(2,end));
fprintf('th=0.10: Obs Untreated=%.3f, Treated=%.3f | RCT Untreated=%.3f, Treated=%.3f\n', ...
    S0est(3,end), S1est(3,end), S0ref(3,end), S1ref(3,end));
fprintf('th=0.80: Obs Untreated=%.3f, Treated=%.3f | RCT Untreated=%.3f, Treated=%.3f\n', ...
    S0est(4,end), S1est(4,end), S0ref(4,end), S1ref(4,end));

% LaTeX-ready summary for figure caption
fprintf('\n=== LaTeX Figure Caption Values ===\n');
fprintf('Copy these values into your figure caption:\n');
fprintf('- At θ = 0 (treat all patients): ATE = %.3f\n', ate_0);
fprintf('- At θ = 0.02 (near optimal): ATE = %.3f (maximum)\n', ate_002);
fprintf('- At θ = 0.10: ATE = %.3f\n', ate_01);
fprintf('- At θ = 0.80 (highly restrictive): ATE = %.3f\n', ate_08);
fprintf('- Optimal threshold: θ = %.3f with ATE = %.3f\n', th_opt_A0, max_A0);

%% Export as PDF
print(gcf, 'Fig_optimization_curves_with_survival.pdf', '-dpdf', '-r300');
fprintf('\nFigure saved as: Fig_optimization_curves_with_survival.pdf\n');

%% Export optimization curve analysis results to text file for paper
filename = sprintf('optimization_curve_analysis_results_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));
fid = fopen(filename, 'w');

fprintf(fid, '==========================================================\n');
fprintf(fid, 'OPTIMIZATION CURVE ANALYSIS RESULTS FOR PAPER\n');
fprintf(fid, 'Generated on: %s\n', datestr(now));
fprintf(fid, '==========================================================\n\n');

% Load bootstrap information
load('A01Data.mat');
Nboot = size(A0, 1);

% Study characteristics
fprintf(fid, 'STUDY CHARACTERISTICS:\n');
fprintf(fid, '- Analysis: Treatment threshold optimization with bootstrap\n');
fprintf(fid, '- Data sources: A01Data.mat, ThreeCurves.mat\n');
fprintf(fid, '- Bootstrap samples: %d\n', Nboot);
fprintf(fid, '- Threshold range: %.3f to %.3f (50 levels)\n', min(th), max(th));
fprintf(fid, '- Key thresholds analyzed: 0.00, 0.02, 0.10, 0.80\n\n');

% Optimal threshold analysis
fprintf(fid, 'OPTIMAL THRESHOLD ANALYSIS:\n');
fprintf(fid, '- Optimal threshold (G-formula): %.3f\n', th_opt_A0);
fprintf(fid, '- Maximum ATE: %.3f [95%% CI: %.3f, %.3f]\n', ...
    max_A0, A0_lower(idx_A0), A0_upper(idx_A0));
fprintf(fid, '- Optimization method: Bootstrap-based grid search\n\n');

% Detailed results at key thresholds
fprintf(fid, 'ATE RESULTS AT KEY THRESHOLDS:\n');
fprintf(fid, '%-12s %-12s %-20s %-15s\n', 'Threshold', 'ATE', '95%% CI', 'Interpretation');
fprintf(fid, '%-12s %-12s %-20s %-15s\n', '---------', '---', '-----', '--------------');

% Threshold 0.00 (treat all)
if ~isnan(ate_0)
    if ate_0 > 0
        interp_0 = 'Beneficial';
    else
        interp_0 = 'Harmful';
    end
    fprintf(fid, '%-12.2f %-12.3f [%-8.3f,%-8.3f] %-15s\n', 0.00, ate_0, ate_0_ci_lower, ate_0_ci_upper, interp_0);
end

% Threshold 0.02 (near optimal)
if ~isnan(ate_002)
    if ate_002 > 0
        interp_002 = 'Beneficial';
    else
        interp_002 = 'Harmful';
    end
    fprintf(fid, '%-12.2f %-12.3f [%-8.3f,%-8.3f] %-15s\n', 0.02, ate_002, ate_002_ci_lower, ate_002_ci_upper, interp_002);
end

% Threshold 0.10
if ~isnan(ate_01)
    if ate_01 > 0
        interp_01 = 'Beneficial';
    else
        interp_01 = 'Harmful';
    end
    fprintf(fid, '%-12.2f %-12.3f [%-8.3f,%-8.3f] %-15s\n', 0.10, ate_01, ate_01_ci_lower, ate_01_ci_upper, interp_01);
end

% Threshold 0.80 (restrictive)
if ~isnan(ate_08)
    if ate_08 > 0
        interp_08 = 'Beneficial';
    else
        interp_08 = 'Harmful';
    end
    fprintf(fid, '%-12.2f %-12.3f [%-8.3f,%-8.3f] %-15s\n', 0.80, ate_08, ate_08_ci_lower, ate_08_ci_upper, interp_08);
end

% Survival outcomes at key thresholds
fprintf(fid, '\nSURVIVAL OUTCOMES AT KEY THRESHOLDS (168 hours):\n');
fprintf(fid, '%-12s %-12s %-12s %-12s %-12s\n', 'Threshold', 'Obs Untrt', 'Obs Treat', 'RCT Untrt', 'RCT Treat');
fprintf(fid, '%-12s %-12s %-12s %-12s %-12s\n', '---------', '----------', '----------', '----------', '----------');

fprintf(fid, '%-12.2f %-12.3f %-12.3f %-12.3f %-12.3f\n', 0.00, S0est(1,end), S1est(1,end), S0ref(1,end), S1ref(1,end));
fprintf(fid, '%-12.2f %-12.3f %-12.3f %-12.3f %-12.3f\n', 0.02, S0est(2,end), S1est(2,end), S0ref(2,end), S1ref(2,end));
fprintf(fid, '%-12.2f %-12.3f %-12.3f %-12.3f %-12.3f\n', 0.10, S0est(3,end), S1est(3,end), S0ref(3,end), S1ref(3,end));
fprintf(fid, '%-12.2f %-12.3f %-12.3f %-12.3f %-12.3f\n', 0.80, S0est(4,end), S1est(4,end), S0ref(4,end), S1ref(4,end));

% Threshold strategy analysis
fprintf(fid, '\nTHRESHOLD STRATEGY ANALYSIS:\n');

% Analyze different threshold strategies
fprintf(fid, 'Treatment strategy implications:\n');
fprintf(fid, '- θ = 0.00 (treat all): Universal treatment regardless of disease level\n');
if ~isnan(ate_0)
    if ate_0 > 0
        fprintf(fid, '  → Results in %.1f%% survival benefit\n', ate_0*100);
    else
        fprintf(fid, '  → Results in %.1f%% survival harm\n', abs(ate_0)*100);
    end
end

fprintf(fid, '- θ = 0.02 (aggressive): Treat patients with minimal disease burden\n');
if ~isnan(ate_002)
    if ate_002 > 0
        fprintf(fid, '  → Results in %.1f%% survival benefit\n', ate_002*100);
    else
        fprintf(fid, '  → Results in %.1f%% survival harm\n', abs(ate_002)*100);
    end
end

fprintf(fid, '- θ = 0.10 (moderate): Standard treatment threshold\n');
if ~isnan(ate_01)
    if ate_01 > 0
        fprintf(fid, '  → Results in %.1f%% survival benefit\n', ate_01*100);
    else
        fprintf(fid, '  → Results in %.1f%% survival harm\n', abs(ate_01)*100);
    end
end

fprintf(fid, '- θ = 0.80 (restrictive): Treat only severely ill patients\n');
if ~isnan(ate_08)
    if ate_08 > 0
        fprintf(fid, '  → Results in %.1f%% survival benefit\n', ate_08*100);
    else
        fprintf(fid, '  → Results in %.1f%% survival harm\n', abs(ate_08)*100);
    end
end

% Optimization curve characteristics
ate_range = max(A0_median) - min(A0_median);
ate_variation = std(A0_median);

fprintf(fid, '\nOPTIMIZATION CURVE CHARACTERISTICS:\n');
fprintf(fid, '- ATE range across thresholds: %.3f (%.1f%% points)\n', ate_range, ate_range*100);
fprintf(fid, '- ATE standard deviation: %.3f\n', ate_variation);
if ate_range > 0.05
    fprintf(fid, '- Curve shape: Strong threshold dependence\n');
else
    fprintf(fid, '- Curve shape: Relatively flat\n');
end

% Find plateau regions (where ATE changes slowly)
if length(th) > 10
    ate_gradient = gradient(A0_median);
    low_gradient_threshold = 0.01;  % Small change threshold
    plateau_regions = sum(abs(ate_gradient) < low_gradient_threshold);
    plateau_percentage = plateau_regions / length(th) * 100;
    
    fprintf(fid, '- Plateau regions (low gradient): %.0f%% of threshold range\n', plateau_percentage);
    
    if plateau_percentage > 50
        fprintf(fid, '  → Treatment benefit is relatively STABLE across thresholds\n');
    else
        fprintf(fid, '  → Treatment benefit is SENSITIVE to threshold selection\n');
    end
end

% Bootstrap uncertainty analysis
fprintf(fid, '\nBOOTSTRAP UNCERTAINTY ANALYSIS:\n');

% Calculate coefficient of variation at optimal threshold
cv_optimal = std(A0(:, idx_A0)) / mean(A0(:, idx_A0)) * 100;
ci_width_optimal = A0_upper(idx_A0) - A0_lower(idx_A0);

fprintf(fid, 'Uncertainty at optimal threshold:\n');
fprintf(fid, '- Coefficient of variation: %.1f%%\n', cv_optimal);
fprintf(fid, '- 95%% CI width: %.3f (%.1f%% points)\n', ci_width_optimal, ci_width_optimal*100);

if ci_width_optimal < 0.05
    fprintf(fid, '- Precision: HIGH (CI width < 5%% points)\n');
elseif ci_width_optimal < 0.1
    fprintf(fid, '- Precision: MODERATE (CI width 5-10%% points)\n');
else
    fprintf(fid, '- Precision: LOW (CI width > 10%% points)\n');
end

% Clinical recommendations
fprintf(fid, '\nCLINICAL RECOMMENDATIONS:\n');

% Safety assessment
safe_thresholds = sum(A0_lower > 0);  % Lower CI > 0
fprintf(fid, 'Safety assessment:\n');
fprintf(fid, '- Thresholds with CI lower bound > 0: %d/%d (%.1f%%)\n', ...
    safe_thresholds, length(th), safe_thresholds/length(th)*100);

if safe_thresholds/length(th) > 0.8
    fprintf(fid, '  → Treatment is broadly SAFE across threshold range\n');
elseif safe_thresholds/length(th) > 0.5
    fprintf(fid, '  → Treatment is MODERATELY safe - threshold selection matters\n');
else
    fprintf(fid, '  → Treatment safety is THRESHOLD-DEPENDENT - careful selection required\n');
end

% Optimal threshold recommendation
if cv_optimal < 20 && ci_width_optimal < 0.1
    fprintf(fid, '\nRecommended strategy:\n');
    fprintf(fid, '✓ USE optimal threshold θ = %.3f\n', th_opt_A0);
    fprintf(fid, '  - Well-defined optimum with good precision\n');
    fprintf(fid, '  - Expected ATE: %.1f%% survival benefit\n', max_A0*100);
else
    fprintf(fid, '\nRecommended strategy:\n');
    fprintf(fid, '⚠️  CAUTIOUS threshold selection around θ = %.3f\n', th_opt_A0);
    fprintf(fid, '  - High uncertainty in optimal threshold\n');
    fprintf(fid, '  - Consider robust threshold range: %.3f to %.3f\n', ...
        th(max(1, idx_A0-2)), th(min(length(th), idx_A0+2)));
end

% Figure interpretation guide
fprintf(fid, '\nFIGURE INTERPRETATION GUIDE:\n');
fprintf(fid, 'Main optimization curve:\n');
fprintf(fid, '- X-axis: Treatment threshold (θ) from 0 to %.2f\n', max(th));
fprintf(fid, '- Y-axis: Average Treatment Effect (survival difference)\n');
fprintf(fid, '- Blue curve: G-formula estimates with 95%% confidence band\n');
fprintf(fid, '- Peak: Optimal threshold at θ = %.3f\n', th_opt_A0);

fprintf(fid, '\nSurvival inset subplots:\n');
fprintf(fid, '- Show individual survival curves for 4 key thresholds\n');
fprintf(fid, '- Solid lines: Observational estimates (G-formula corrected)\n');
fprintf(fid, '- Dashed lines: RCT ground truth for comparison\n');
fprintf(fid, '- Connected to main plot via gray lines\n');

% Technical details
fprintf(fid, '\nTECHNICAL DETAILS:\n');
fprintf(fid, '- Figure output: Fig_optimization_curves_with_survival.pdf\n');
fprintf(fid, '- Format: PDF vector graphics (300 DPI)\n');
fprintf(fid, '- Dimensions: 10 × 8 inches\n');
fprintf(fid, '- Layout: 4 survival insets above main optimization curve\n');
fprintf(fid, '- Colors: Blue tones consistent with other figures\n');
fprintf(fid, '- Statistical method: Bootstrap percentiles for confidence intervals\n');

fclose(fid);
fprintf('Optimization curve analysis results exported to: %s\n', filename);