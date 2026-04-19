%% Plot Survival Curves with Confidence Intervals (No Swimmer Plots)
% This script loads precomputed bootstrap confidence bands and plots
% survival curves with 95% confidence intervals and RCT reference curves

clear all; close all; clc;

%% Load precomputed bootstrap confidence bands
load('bootstrap_confidence_bands.mat');
% This loads: S0h, S1h, s0_lower, s0_upper, s1_lower, s1_upper, 
%             s0_median, s1_median, t_grid, Nboot

%% Get reference curves from RCT data
T1 = readtable('trialData1.csv');
[s0_true, s1_true, t0_true, t1_true] = fcnPlotKM(T1);

%% Create figure
figure('Position', [100, 100, 800, 600]);
hold on;

% Time vector
t_grid = 0:2:168;  % 85 points

% Convert to column vectors for consistency
t_grid = t_grid(:);
s0_lower = s0_lower(:);
s0_upper = s0_upper(:);
s1_lower = s1_lower(:);
s1_upper = s1_upper(:);
s0_median = s0_median(:);
s1_median = s1_median(:);

%% Plot confidence bands (behind the lines)
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

%% Plot main curves
% True/RCT reference curves (dashed)
plot(t0_true, s0_true, 'b--', 'LineWidth', 2.5, 'DisplayName', 'Untreated (RCT)');
plot(t1_true, s1_true, 'r--', 'LineWidth', 2.5, 'DisplayName', 'Treated (RCT)');

% Estimated curves from bootstrap median (solid)
plot(t_grid, s0_median, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Untreated (Estimated)');
plot(t_grid, s1_median, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Treated (Estimated)');

%% Formatting
xlabel('Time (hours)', 'FontSize', 14);
ylabel('Survival Probability', 'FontSize', 14);
title('Survival Curves with 95% Bootstrap Confidence Intervals', 'FontSize', 16);
legend('Location', 'southwest', 'FontSize', 11);
grid on;
set(gca, 'FontSize', 12);
xlim([0, 168]);
ylim([0, 1]);
set(gcf, 'Color', 'w');

% Set x-axis ticks every 24 hours
xticks(0:24:168);
xticklabels(0:24:168);

%% Display summary statistics
fprintf('\n=== Survival Analysis Summary ===\n');
fprintf('Bootstrap samples: %d\n\n', Nboot);

% Convert back to row vectors for analysis
s0_lower = s0_lower';
s0_upper = s0_upper';
s1_lower = s1_lower';
s1_upper = s1_upper';
s0_median = s0_median';
s1_median = s1_median';

% Survival at key time points
fprintf('Survival Probabilities:\n');
fprintf('-------------------------------------------\n');
fprintf('Time (h) | Untreated         | Treated\n');
fprintf('         | Median [95%% CI]   | Median [95%% CI]\n');
fprintf('-------------------------------------------\n');

key_times = [24, 48, 72, 96, 120, 144, 168];
for kt = key_times
    [~, idx] = min(abs(t_grid' - kt));
    fprintf('%8d | %.1f%% [%.1f-%.1f] | %.1f%% [%.1f-%.1f]\n', ...
        kt, ...
        s0_median(idx)*100, s0_lower(idx)*100, s0_upper(idx)*100, ...
        s1_median(idx)*100, s1_lower(idx)*100, s1_upper(idx)*100);
end

% Treatment effect at end of study
fprintf('\n-------------------------------------------\n');
treatment_effect_median = s1_median(end) - s0_median(end);
fprintf('Treatment Effect at 168 hours:\n');
fprintf('  Absolute Risk Reduction: %.1f%%\n', treatment_effect_median*100);
fprintf('  Number Needed to Treat: %.1f\n', 1/treatment_effect_median);

%% Save figure
print('Fig_survival_curves_only.pdf', '-dpdf', '-r300');
fprintf('\nFigure saved as: Fig_survival_curves_only.pdf\n');