%% Ultra-tight 2×3 heatmaps with 50% reduced gaps
clear all; clc; format compact;
addpath('..');  % ensure your .mat files are on the path

%% Load & smooth Aggressive data
load('HeatMapAggressive.mat','s0','s1','A2','A3');
sigma = 1.5;
s0_smooth_agg   = imgaussfilt(s0, sigma);
s1_smooth_agg   = imgaussfilt(s1, sigma);
diff_smooth_agg = s1_smooth_agg - s0_smooth_agg;

%% Load & smooth Normal data
load('HeatMapData.mat','s0','s1','A2','A3');
s0_smooth   = imgaussfilt(s0, sigma);
s1_smooth   = imgaussfilt(s1, sigma);
diff_smooth = s1_smooth - s0_smooth;

%% Compute global color limits
all_data    = [s0_smooth_agg(:); s1_smooth_agg(:); diff_smooth_agg(:); ...
               s0_smooth(:);       s1_smooth(:);       diff_smooth(:)];
global_min  = min(all_data);
global_max  = max(all_data);
n_contours  = 10;

%% Define margins & 50% reduced gaps
left_margin   = 0.02;   % unchanged
right_margin  = 0.15;   % unchanged
bottom_margin = 0.08;   % unchanged
top_margin    = 0.05;   % unchanged
gap_h         = 0.005;  % was 0.01 → now half
gap_v         = 0.02;   % was 0.04 → now half

% Compute plot width & height
plot_w = (1 - left_margin - right_margin - 2*gap_h) / 3;
plot_h = (1 - bottom_margin - top_margin - gap_v) / 2;

% X-positions of each column
x1 = left_margin;
x2 = x1 + plot_w + gap_h;
x3 = x2 + plot_w + gap_h;
% Y-positions of each row
y_bottom = bottom_margin;
y_top    = y_bottom + plot_h + gap_v;

% Pack positions
positions = [ ...
    x1, y_top,    plot_w, plot_h;  % top-left
    x2, y_top,    plot_w, plot_h;  % top-center
    x3, y_top,    plot_w, plot_h;  % top-right
    x1, y_bottom, plot_w, plot_h;  % bottom-left
    x2, y_bottom, plot_w, plot_h;  % bottom-center
    x3, y_bottom, plot_w, plot_h   % bottom-right
];

%% Titles & data
titles = { ...
    'S_1', 'S_0',  'S_1 - S_0', ...
    'S_1^*',             'S_0',           'S_1^* - S_0' ...
};
dataArr = { ...
    s1_smooth_agg',   s0_smooth_agg',   diff_smooth_agg', ...
    s1_smooth',       s0_smooth',       diff_smooth' ...
};

%% Plot all heatmaps
figure(1); clf; colormap hot;
for idx = 1:6
    ax = subplot('Position', positions(idx,:));
    imagesc(A2, A3, dataArr{idx}); 
    axis xy; hold on;
      contour(A2, A3, dataArr{idx}, n_contours, 'k', 'LineWidth', 0.5);
      text(25,45,titles{idx}, ...
           'HorizontalAlignment','center', ...
           'FontSize',10,'FontWeight','bold', ...
           'BackgroundColor',[1 1 1 0.7],'EdgeColor','none');
    hold off;
    
    axis equal tight;
    xlim([1 50]); ylim([1 50]);
    clim([global_min global_max]);
    
    if idx ~= 4
        ax.XTick = []; 
        ax.YTick = [];
    else
        xlabel('Harm from L','FontSize',11);
        ylabel('Harm from A','FontSize',11);
        ax.FontSize = 9;
    end
end

%% Single colorbar on the right
cb_width = 0.02;
cb_height = 2*plot_h + gap_v;
cb_x = 1 - right_margin + 0.01;
cb_y = bottom_margin;
colorbar('Position', [cb_x, cb_y, cb_width, cb_height]);

%% Final figure sizing & PDF export
fig_w = 7; fig_h = 4;
set(gcf, ...
    'Units','inches', ...
    'Position',[1 1 fig_w fig_h], ...
    'PaperUnits','inches', ...
    'PaperSize',[fig_w fig_h], ...
    'PaperPosition',[0 0 fig_w fig_h], ...
    'Color','white' ...
);
print(gcf, 'Fig_heatmap_figure', '-dpdf', '-r300');
fprintf('Figure saved as heatmap_figure.pdf\n');
drawnow;
