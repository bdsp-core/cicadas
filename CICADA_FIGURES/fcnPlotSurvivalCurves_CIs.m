function fcnPlotSurvivalCurves_CIs(S0h, S1h, s0_lower, s0_upper,s1_lower, s1_upper, s0_median, s1_median, t_grid, Nboot)


%% Calculate 95% confidence bands
% S0h and S1h are Nboot x 85 matrices (each row is one bootstrap curve)
% S0h and S1h are Nboot x 85 matrices (each row is one bootstrap curve)
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
T1 = readtable('trialData1_logit.csv');
[s0_true, s1_true, t0_true, t1_true] = fcnPlotKM(T1);

% Create plot with confidence bands
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
t0 = 0:2:168; t1 = t0; 

plot(t0, s0_true, 'b--', 'LineWidth', 2.5, 'DisplayName', 'Untreated (True)');
plot(t1, s1_true, 'r--', 'LineWidth', 2.5, 'DisplayName', 'Treated (True)');

% Estimated curves (solid) - from full dataset
plot(t0, s0_median, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Untreated (Estimated)');
plot(t1, s1_median, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Treated (Estimated)');

% Formatting
xlabel('Time (hours)', 'FontSize', 12);
ylabel('Survival Probability', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 11);
xlim([0, 168]);
ylim([0, 1]);
set(gcf, 'Color', 'w');

% Convert back to row vectors for display (if needed)
s0_lower = s0_lower';
s0_upper = s0_upper';
s1_lower = s1_lower';
s1_upper = s1_upper';
s0_median = s0_median(:)';
s1_median = s1_median(:)';

xticks(0:24:168);
xticklabels(0:24:168);
xlim([0 168]);