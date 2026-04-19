clear all; clc; format compact;
addpath('..'); % needs access to files one folder up
figure(1); clf;
load HeatMapAggressive % Th_opt ate0 ate1 A2 A3
% Smooth the data using 2D Gaussian filter
sigma = 1.5; % Smoothing parameter - adjust as needed
s0_smooth = imgaussfilt(s0, sigma);
s1_smooth = imgaussfilt(s1, sigma);
diff_smooth = s1_smooth - s0_smooth;
% Find the global min and max across all smoothed data
all_data = [s0_smooth(:); s1_smooth(:); diff_smooth(:)];
global_min = min(all_data);
global_max = max(all_data);
figure(1); clf;
% Create custom positions for single row layout
% [left bottom width height]
gap = 0.02; % Small gap between plots
plot_height = 0.35;
plot_width = 0.25;
y_pos = 0.3; % Vertical position (centered)
% Positions for three plots in a row
pos1 = [0.05 y_pos plot_width plot_height]; % Left
pos2 = [0.05+plot_width+gap y_pos plot_width plot_height]; % Center
pos3 = [0.05+2*(plot_width+gap) y_pos plot_width plot_height]; % Right
% Number of contour levels
n_contours = 10; % You can adjust this
% First plot (left)
subplot('Position', pos1);
imagesc(A2, A3, s1_smooth'); axis xy
hold on;
% Add contour lines on smoothed data
contour(A2, A3, s1_smooth', n_contours, 'k', 'LineWidth', 0.5);
hold off;
set(gca,'XTickLabel',[]); % Remove x-axis labels
ylabel('Harm from A')
clim([global_min global_max]);
axis equal
axis([1 50 1 50])
title('S1*');
% Second plot (center)
subplot('Position', pos2);
imagesc(A2, A3, s0_smooth'); axis xy
hold on;
% Add contour lines on smoothed data
contour(A2, A3, s0_smooth', n_contours, 'k', 'LineWidth', 0.5);
hold off;
set(gca,'XTickLabel',[]); % Remove x-axis labels
set(gca,'YTickLabel',[]); % Remove y-axis labels
clim([global_min global_max]);
axis equal
axis([1 50 1 50])
title('S0');
% Third plot (right)
subplot('Position', pos3);
imagesc(A2, A3, diff_smooth'); axis xy
hold on;
% Add contour lines on smoothed data
contour(A2, A3, diff_smooth', n_contours, 'k', 'LineWidth', 0.5);
hold off;
xlabel('Harm from L')
set(gca,'YTickLabel',[]); % Remove y-axis labels
clim([global_min global_max]);
axis equal
axis([1 50 1 50])
title('S1* - S0');
% Add colorbar to the right of the last plot
colorbar('Position', [pos3(1)+pos3(3)+gap y_pos 0.02 plot_height]);
% Add x-label to all plots at the bottom
text(0.4, 0.2, 'Harm from L', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'normal');
set(gcf,'color','white')
colormap hot
drawnow