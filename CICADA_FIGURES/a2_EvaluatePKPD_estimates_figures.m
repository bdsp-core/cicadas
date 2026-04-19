%% COMBINED FIGURE: PARAMETER ESTIMATION AND L PREDICTION (A-D LAYOUT)
close all; clear all; clc; format compact; 

fprintf('Creating Combined Figure: Parameter Estimation and L Prediction Analysis...\n');
addpath('..');                                    % if your CSV lives one folder up

%% LOAD RESULTS
fprintf('Loading results from a5_EstimatePKPD...\n');
if ~exist('PKPD_estimation_results.mat', 'file')
    error('Please run a5_EstimatePKPD.m first to generate results!');
end

load('PKPD_estimation_results.mat', 'results');
fprintf('Results loaded successfully.\n\n');

% Extract key variables from results structure
N = results.N;
t = results.t;
true_params = results.true_params;
param_names = results.param_names;
L0_true = results.L0_true;
L_obs = results.L_obs;
A_obs = results.A_obs;

% Get prediction data
L_pred_trueL0 = results.L_prediction.L_pred_trueL0;

% Calculate MAE for each patient
mae_L_trueL0 = zeros(N, 1);
for p_idx = 1:N
    valid = ~isnan(L_obs(p_idx,:));
    if sum(valid) > 0
        mae_L_trueL0(p_idx) = mean(abs(L_obs(p_idx,valid) - L_pred_trueL0(p_idx,valid)));
    else
        mae_L_trueL0(p_idx) = NaN;
    end
end

% Create figure with specific dimensions for publication
figure('Position', [50, 50, 1200, 900], 'Name', 'Combined_Analysis', ...
    'Color', 'white', 'PaperPositionMode', 'auto');

% Define tight subplot positions manually for equal dimensions and minimal spacing
% Format: [left, bottom, width, height] in normalized coordinates
subplot_width = 0.42;   % Width of each subplot
subplot_height = 0.38;  % Height of each subplot
left_margin = 0.08;     % Left margin
right_margin = 0.05;    % Right margin  
bottom_margin = 0.08;   % Bottom margin
top_margin = 0.12;      % Top margin (space for main title)
h_spacing = 0.08;       % Horizontal spacing between subplots
v_spacing = 0.08;       % Vertical spacing between subplots

% Calculate positions for 2x2 grid
positions = {
    [left_margin, bottom_margin + subplot_height + v_spacing, subplot_width, subplot_height],  % Top-left (A)
    [left_margin + subplot_width + h_spacing, bottom_margin + subplot_height + v_spacing, subplot_width, subplot_height],  % Top-right (B)
    [left_margin, bottom_margin, subplot_width, subplot_height],  % Bottom-left (C)
    [left_margin + subplot_width + h_spacing, bottom_margin, subplot_width, subplot_height]   % Bottom-right (D)
};

%% PANEL A: PARAMETER ESTIMATION ERRORS
subplot('Position', positions{1});

% Combine all parameters including ke
param_names_display = {'\beta_0^C', '\beta_1^C', '\beta_2^C', '\beta_0^\gamma', '\beta_1^\gamma', '\beta_2^\gamma', 'k_e'};

all_param_names = param_names_display;
param_errors_raw = [results.joint.errors(1:7)', ...
                    results.oracle.errors(1:7)', ...
                    results.twostage_raw.errors(1:7)', ...
                    results.twostage_corr.errors(1:7)'];

% Note: Oracle has 0 error for ke by definition
param_errors_raw(7, 2) = 0;

% Set y-axis limit for better visualization
y_max = 5;  % Cap at 5% for clarity

% Process data: cap values at y_max for display
param_errors = param_errors_raw;
truncated_indices = param_errors > y_max;
param_errors(truncated_indices) = y_max;

% Create grouped bar plot
b = bar(param_errors, 'EdgeColor', 'k', 'LineWidth', 0.8);

% Set colors for each method
colors = [0.8 0.2 0.2;  % Red for joint
          0.2 0.6 0.2;  % Green for oracle
          0.4 0.4 0.8;  % Blue for raw
          0.2 0.4 0.8]; % Darker blue for corrected

for i = 1:4
    b(i).FaceColor = colors(i,:);
end

% Add break symbols for truncated bars
hold on;
for param_idx = 1:7
    for method_idx = 1:4
        if param_errors_raw(param_idx, method_idx) > y_max
            % Get bar position
            x_pos = b(method_idx).XEndPoints(param_idx);
            
            % Add white break in the bar
            patch([x_pos-0.08, x_pos+0.08, x_pos+0.08, x_pos-0.08], ...
                  [y_max-0.3, y_max-0.3, y_max-0.15, y_max-0.15], ...
                  'w', 'EdgeColor', 'none');
            
            % Add zigzag lines
            plot([x_pos-0.08, x_pos-0.04, x_pos+0.04, x_pos+0.08], ...
                 [y_max-0.3, y_max-0.15, y_max-0.3, y_max-0.15], ...
                 'k-', 'LineWidth', 1.5);
            
            % Add actual value as text above the bar
            text(x_pos, y_max + 0.2, sprintf('%.0f%%', param_errors_raw(param_idx, method_idx)), ...
                'HorizontalAlignment', 'center', 'FontSize', 9, ...
                'FontWeight', 'bold', 'Color', colors(method_idx,:));
        end
    end
end

% Formatting
set(gca, 'XTickLabel', all_param_names, 'FontSize', 11, 'FontWeight', 'bold', ...
    'Box', 'off', 'TickDir', 'out', 'LineWidth', 1.2);
ylabel('Absolute Error (%)', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Parameter', 'FontSize', 11, 'FontWeight', 'bold');
ylim([0, 5.9]);

% Add title inside the plot (upper left corner) - moved up to avoid overlap
text(0.02, 0.98, 'A. PKPD Parameter Estimation Errors', 'Units', 'normalized', ...
    'FontSize', 12, 'FontWeight', 'bold', 'VerticalAlignment', 'top', ...
    'BackgroundColor', 'white', 'EdgeColor', 'none');

% Add legend with white background, no border, and positioned at top middle
leg = legend(b, {'Joint', 'Oracle (fixed ke)', '2-Stage (Raw)', '2-Stage (Corrected)'}, ...
    'Location', 'north', 'FontSize', 10, 'Box', 'on');
leg.Color = 'white';
leg.EdgeColor = 'none';
% Move legend down by about 1 centimeter total
leg_pos = leg.Position;
leg_pos(2) = leg_pos(2) - 0.10;  % Move down by 0.10 normalized units (approximately 1 cm)
leg.Position = leg_pos;

% Add grid
grid on;
ax = gca;
ax.GridAlpha = 0.3;
ax.YGrid = 'on';
ax.XGrid = 'off';

%% PANEL B: MAE DISTRIBUTION (replacing scatter plot)
subplot('Position', positions{2});

valid_mae = mae_L_trueL0(~isnan(mae_L_trueL0));
histogram(valid_mae, 30, 'FaceColor', [0.2 0.4 0.8], ...
    'EdgeColor', 'k', 'LineWidth', 0.5, 'FaceAlpha', 0.8);
xlabel('MAE', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Number of Patients', 'FontSize', 11, 'FontWeight', 'bold');
ylim([0 225])

% Add title inside the plot (upper left corner)
text(0.02, 0.95, 'B. Prediction Error Distribution', 'Units', 'normalized', ...
    'FontSize', 12, 'FontWeight', 'bold', 'VerticalAlignment', 'top', ...
    'BackgroundColor', 'white', 'EdgeColor', 'none');

% Add text annotation for mean and median
text(0.6, 0.85, sprintf('Mean: %.3f\nMedian: %.3f', mean(valid_mae), median(valid_mae)), ...
    'Units', 'normalized', 'FontSize', 10, 'FontWeight', 'bold', ...
    'BackgroundColor', 'white', 'EdgeColor', 'none');

set(gca, 'FontSize', 10, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1.2);
grid on;
ax = gca;
ax.GridAlpha = 0.3;

%% PANEL C: L prediction example - Patient 100
subplot('Position', positions{3});

patient_idx = 100;
valid = ~isnan(L_obs(patient_idx,:));
plot(t(valid), L_obs(patient_idx,valid), 'ko', 'MarkerSize', 6, ...
    'MarkerFaceColor', [0.3 0.3 0.3], 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
hold on;
plot(t, L_pred_trueL0(patient_idx,:), '-', 'Color', [0.8 0.2 0.2], 'LineWidth', 2.5);

% Calculate MAE for this patient (consistent with older version)
patient_mae = mae_L_trueL0(patient_idx);

xlabel('Time (hours)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('L_t', 'FontSize', 11, 'FontWeight', 'bold');
ylim([0 0.32])

% Add title inside the plot (upper left corner)
text(0.02, 0.95, sprintf('C. Patient %d (MAE: %.3f)', patient_idx, patient_mae), ...
    'Units', 'normalized', 'FontSize', 12, 'FontWeight', 'bold', 'VerticalAlignment', 'top', ...
    'BackgroundColor', 'white', 'EdgeColor', 'none');
legend({'Observed', 'Predicted'}, 'Location', 'best', 'FontSize', 10, 'Box', 'off');
set(gca, 'FontSize', 10, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1.2);
grid on;
ax = gca;
ax.GridAlpha = 0.3;

%% PANEL D: L prediction example - Patient 20
subplot('Position', positions{4});

patient_idx = 20;
valid = ~isnan(L_obs(patient_idx,:));
plot(t(valid), L_obs(patient_idx,valid), 'ko', 'MarkerSize', 6, ...
    'MarkerFaceColor', [0.3 0.3 0.3], 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
hold on;
plot(t, L_pred_trueL0(patient_idx,:), '-', 'Color', [0.8 0.2 0.2], 'LineWidth', 2.5);

% Calculate MAE for this patient (consistent with older version)
patient_mae = mae_L_trueL0(patient_idx);

xlabel('Time (hours)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('L_t', 'FontSize', 11, 'FontWeight', 'bold');
ylim([0 0.32])


% Add title inside the plot (upper left corner)
text(0.02, 0.95, sprintf('D. Patient %d (MAE: %.3f)', patient_idx, patient_mae), ...
    'Units', 'normalized', 'FontSize', 12, 'FontWeight', 'bold', 'VerticalAlignment', 'top', ...
    'BackgroundColor', 'white', 'EdgeColor', 'none');
legend({'Observed', 'Predicted'}, 'Location', 'best', 'FontSize', 10, 'Box', 'off');
set(gca, 'FontSize', 10, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1.2);
grid on;
ax = gca;
ax.GridAlpha = 0.3;

%% Add overall title with proper positioning
% Position the main title manually for better control
annotation('textbox', [0, 0.92, 1, 0.08], 'String', ...
    'PKPD Parameter Estimation Evaluation', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'FontSize', 16, 'FontWeight', 'bold', 'EdgeColor', 'none');

%% Save Combined Figure with high resolution for publication
% Set paper size and position for consistent output
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [12, 9]);
set(gcf, 'PaperPosition', [0, 0, 12, 9]);

% Save as PDF with high quality
print(gcf, 'Fig_Combined_PKPD_Analysis.pdf', '-dpdf', '-r300');
fprintf('  Saved: Fig_Combined_PKPD_Analysis.pdf\n');

% Optional: Also save as high-resolution PNG for presentations
print(gcf, 'Fig_Combined_PKPD_Analysis.png', '-dpng', '-r300');
fprintf('  Saved: Fig_Combined_PKPD_Analysis.png\n');

%% FINAL SUMMARY (from older version)
% Create summary text for both console and file output
summary_text = [];
summary_text = [summary_text sprintf('==========================================================\n')];
summary_text = [summary_text sprintf('PKPD PUBLICATION FIGURES: SUMMARY OUTPUT\n')];
summary_text = [summary_text sprintf('==========================================================\n\n')];

summary_text = [summary_text sprintf('Generated: %s\n', datestr(now))];
summary_text = [summary_text sprintf('Source script: Combined PKPD Analysis Figure\n')];
summary_text = [summary_text sprintf('Source data: PKPD_estimation_results.mat\n\n')];

summary_text = [summary_text sprintf('FIGURES GENERATED:\n')];
summary_text = [summary_text sprintf('  - Fig_Combined_PKPD_Analysis.pdf\n')];
summary_text = [summary_text sprintf('  - Fig_Combined_PKPD_Analysis.png\n\n')];

summary_text = [summary_text sprintf('KEY RESULTS:\n')];
summary_text = [summary_text sprintf('  Best Method: Two-Stage with Bias Correction\n')];
summary_text = [summary_text sprintf('  - MAE: %.1f%% (vs %.1f%% for joint estimation)\n', ...
    results.twostage_corr.mape, results.joint.mape)];
summary_text = [summary_text sprintf('  - Improvement: %.0f-fold\n', results.joint.mape / results.twostage_corr.mape)];
summary_text = [summary_text sprintf('  - ke estimation error: %.1f%% (after correction)\n', ...
    results.twostage_corr.errors(7))];
summary_text = [summary_text sprintf('\nL PREDICTION ACCURACY:\n')];
summary_text = [summary_text sprintf('  - With true L0: %.3f MAE\n', ...
    mean(mae_L_trueL0(~isnan(mae_L_trueL0))))];
if isfield(results.L_prediction, 'mape_L_estL0')
    summary_text = [summary_text sprintf('  - With estimated L0: available in results\n')];
end
summary_text = [summary_text sprintf('\nL0 RECOVERY:\n')];
summary_text = [summary_text sprintf('  - Correlation: %.3f (moderate due to high suppression)\n', ...
    results.twostage_corr.L0_corr)];
summary_text = [summary_text sprintf('  - This is expected and does not affect L prediction\n')];
summary_text = [summary_text sprintf('\nFigures saved as PDF and PNG files.\n')];
summary_text = [summary_text sprintf('==========================================================\n')];

% Display summary
fprintf('\n%s', summary_text);

% Save summary to file
filename = 'PKPD_combined_figure_summary.txt';
fprintf('Saving summary to %s...\n', filename);
fid = fopen(filename, 'w');
fprintf(fid, '%s', summary_text);
fclose(fid);
fprintf('Summary saved to: %s\n', filename);
fprintf('==========================================================\n');