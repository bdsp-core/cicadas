
% =========================================================================
% Publication-quality Figure 1 with consistent axes, line-widths, full A(t),
% and the black "death" star on the untreated panel.
% Now includes third panel with dose-changing data
% =========================================================================
clear; clc; format compact;
warning off
addpath('..');                                    % if your CSV lives one folder up
T = readtable('trialData1.csv');
T_dose = readtable('trialDataDoseChanging.csv');  % Load dose-changing data

% --- Patient selection (your exemplar IDs) -------------------------------
sid1 = unique(T.sid(T.Rx==1));   % initially treated
sid0 = unique(T.sid(T.Rx==0));   % initially untreated
sid_dose = unique(T_dose.sid);   % patients with dose changes
idx1 = 1;   
idx0 = 9;
%6, 7;  % Select patient 5 from dose-changing data (you can change this)

idx_dose = 58;


% --- Styling parameters --------------------------------------------------
axesFS   = 11;     % tick-label font size
lblFS    = 11;     % axis-label font size
ttlFS    = 12;     % title font size
lgdFS    = 12;     % legend font size
LW_L     = 3;     % line width for L(t)
LW_A     = 1.5;   % line width for A(t)
YL_LIM   = [0    1.25];
YR_LIM   = [-0.1   5];
XLIM     = [0    168];

% --- Build the figure ----------------------------------------------------
figure(1); clf;
set(gcf,'Units','inches','Position',[1 1 6.5 7.5]);  % Increased height for 3 panels
tlo = tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

for p = 1:3
    ax = nexttile;
    
    % Choose data & colors per panel
    if p == 1  % No treatment on top
        pd  = T(T.sid==sid0(idx0), :);
        ttl = 'Untreated';
        C   = [0.2 0.2 0.8];
    elseif p == 2  % Dose changing in middle
        pd  = T_dose(T_dose.sid==sid_dose(idx_dose), :);
        ttl = 'Random Dose Switching';
        C   = [0.2 0.6 0.2];  % Green color for dose-changing panel
    else  % p == 3, closed-loop control on bottom
        pd  = T(T.sid==sid1(idx1), :);
        ttl = 'Closed-Loop Control';
        C   = [0.8 0.2 0.2];
    end

    % -- Left axis: L(t) -----------------------------------------------
    yyaxis(ax,'left')
    plot(ax, pd.t, pd.L, '-', 'Color', C, 'LineWidth', LW_L)
    ax.YColor = C;
    ylim(ax, YL_LIM)

    % -- Black star on untreated panel (now panel 1) -----------------
    if p == 1
        hold(ax,'on')
        plot(ax, pd.t(end), pd.L(end), 'kp', ...
             'MarkerSize', 12, 'MarkerFaceColor','k', ...
             'HandleVisibility','off');
        hold(ax,'off')
    end

    % -- Right axis: A(t), extended to full 0–168h --------------------
    yyaxis(ax,'right')
    t_full = XLIM(1):1:XLIM(2);
    A_full = interp1(pd.t, pd.A, t_full, 'previous', 0);
    plot(ax, t_full, A_full, 'k--', 'LineWidth', LW_A)
    ax.YColor = [0 0 0];
    ylim(ax, YR_LIM)

    % -- Force exact point‐sized fonts ---------------------------------
    ax.FontUnits               = 'points';
    ax.FontSize                = axesFS;
    ax.LabelFontSizeMultiplier = 1;
    ax.TitleFontSizeMultiplier = 1;

    % -- Titles (inside plot), labels, legend, grid ------------------
    % Place title inside the plot area at upper left
    yyaxis(ax,'left')  % Make sure we're on the left axis for positioning
   
    text(ax, 0.02, 0.95, ttl, 'Units', 'normalized', ...
     'FontUnits','points', 'FontSize', ttlFS, ...
     'FontWeight','bold', 'VerticalAlignment','top', ...
     'HorizontalAlignment','left', 'Color','k', ...
     'BackgroundColor','white', 'EdgeColor','none')
    
    % Only add axis labels on the bottom panel
    if p == 3
        xlabel(ax, 'Time [hours]', 'FontUnits','points', 'FontSize', lblFS)
        yyaxis(ax,'left')
          ylabel(ax, 'L_t', 'FontUnits','points', 'FontSize', lblFS)
        yyaxis(ax,'right')
          ylabel(ax, 'A_t', 'FontUnits','points', 'FontSize', lblFS)
    end

    legend(ax, {'L_t','A_t'}, ...
           'FontSize', lgdFS, 'Box','off', 'Location','best')
    grid(ax,'on')
    box(ax,'off')  % Turn off the box
    xlim(ax, XLIM)
end

% --- Export -------------------------------------------------------------
set(gcf,'Color','w')
print(gcf, 'Fig1_singleTrajectories_3panels.pdf', '-dpdf', '-r300');
disp('Fig_singleTrajectories_3panels.pdf saved')
