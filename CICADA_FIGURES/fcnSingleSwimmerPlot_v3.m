function fcnSingleSwimmerPlot_v3(T)
% Create swimmer plots for the epilepsy simulation data with L(t) and A(t) only
% Using heatmap-style coloring for disease and treatment intensity
%
% T: Table with columns sid, t, Rx, harmE, harmA, b0, L, A, V, Y
%
% Creates 2 subplots in a single column taking full figure width

% Extract unique subject IDs
subject_ids = unique(T.sid);
n_subjects = length(subject_ids);

% Parameters
max_time = 168; % Maximum time horizon

% Initialize arrays to track patient characteristics
max_times = zeros(n_subjects, 1);       % Maximum observation time for each patient
initial_treatment = false(n_subjects, 1); % Initial treatment status
died = false(n_subjects, 1);            % Whether the patient died
censored = false(n_subjects, 1);        % Whether the patient was censored
event_times = zeros(n_subjects, 1);     % Time of death or censoring

% Check if V variable exists
has_V = ismember('V', T.Properties.VariableNames);

% Initialize status matrix with background color (white)
time_resolution = 500;
time_grid = linspace(0, max_time, time_resolution);

% First pass through data: determine patient characteristics
for i = 1:n_subjects
    % Get patient ID
    patient_id = subject_ids(i);
    
    % Get data for this patient
    patient_data = T(T.sid == patient_id, :);
    
    % Sort by time
    patient_data = sortrows(patient_data, 't');
    
    % Get maximum observation time
    max_times(i) = max(patient_data.t);
    
    % Check initial treatment status (at first observation)
    initial_treatment(i) = patient_data.Rx(1) == 1;
    
    % Check if patient died
    died(i) = any(patient_data.Y > 0);
    
    % Check censoring
    if has_V
        censored(i) = any(patient_data.V > 0);
    end
    
    % Find time of death or censoring
    event_time = max_times(i); % Default to last observation
    if has_V
        event_idx = find(patient_data.Y > 0 | patient_data.V > 0, 1, 'first');
    else
        event_idx = find(patient_data.Y > 0, 1, 'first');
    end
    if ~isempty(event_idx)
        event_times(i) = patient_data.t(event_idx);
    else
        event_times(i) = event_time;
    end
end

% Define the groups
censored_any = censored;
completed = ~died & ~censored_any;  % Patients who completed the study

group1 = initial_treatment & died;
group2 = initial_treatment & censored_any;
group3 = initial_treatment & completed;
group4 = ~initial_treatment & died;
group5 = ~initial_treatment & censored_any;
group6 = ~initial_treatment & completed;

% Create arrays for each group of patients
group1_ids = subject_ids(group1);
group2_ids = subject_ids(group2);
group3_ids = subject_ids(group3);
group4_ids = subject_ids(group4);
group5_ids = subject_ids(group5);
group6_ids = subject_ids(group6);

% Get event times for each group (use max_times for completed patients)
group1_times = event_times(group1);
group2_times = event_times(group2);
group3_times = max_times(group3);  % Use max observation time for completed patients
group4_times = event_times(group4);
group5_times = event_times(group5);
group6_times = max_times(group6);  % Use max observation time for completed patients

% Sort each group by event/completion time
[~, sort_idx1] = sort(group1_times);
[~, sort_idx2] = sort(group2_times);
[~, sort_idx3] = sort(group3_times);
[~, sort_idx4] = sort(group4_times);
[~, sort_idx5] = sort(group5_times);
[~, sort_idx6] = sort(group6_times);

% Create the final sorted order
sorted_subject_ids = [
    group1_ids(sort_idx1);  % Group 1: Initially on treatment, died
    group2_ids(sort_idx2);  % Group 2: Initially on treatment, censored
    group3_ids(sort_idx3);  % Group 3: Initially on treatment, completed
    group4_ids(sort_idx4);  % Group 4: Initially not on treatment, died
    group5_ids(sort_idx5);  % Group 5: Initially not on treatment, censored
    group6_ids(sort_idx6)   % Group 6: Initially not on treatment, completed
];

% Calculate group divider (only one line between treated and untreated)
group_dividers = [
    length(group1_ids) + length(group2_ids) + length(group3_ids)  % End of all initially treated groups
];

% ===== FIRST SUBPLOT: L Values (Disease Intensity) =====
ax1 = subplot(2, 1, 1);
hold on;

% Create matrices for L values
L_matrix = NaN(length(sorted_subject_ids), time_resolution);

% Find maximum L value for color scaling
max_L = max(T.L);
min_L = 0;  % Minimum is 0

% Process each subject in sorted order
for i = 1:length(sorted_subject_ids)
    % Get patient ID from the sorted list
    patient_id = sorted_subject_ids(i);
    
    % Get data for this patient
    patient_data = T(T.sid == patient_id, :);
    
    % Sort by time
    patient_data = sortrows(patient_data, 't');
    
    % Get time points and L values
    times = patient_data.t;
    L_values = patient_data.L;
    
    % Get event time
    event_time = max(times); % Default to last observation
    event_type = 0; % 0 = no event, 1 = death, 2 = censoring
    if has_V
        event_idx = find(patient_data.Y > 0 | patient_data.V > 0, 1, 'first');
        if ~isempty(event_idx)
            event_time = times(event_idx);
            if patient_data.Y(event_idx) > 0
                event_type = 1; % Death
            else
                event_type = 2; % Censoring
            end
        end
    else
        event_idx = find(patient_data.Y > 0, 1, 'first');
        if ~isempty(event_idx)
            event_time = times(event_idx);
            event_type = 1; % Death
        end
    end
    
    % Interpolate L values to regular grid
    for j = 1:time_resolution
        t = time_grid(j);
        
        % Skip if beyond max time
        if t > max_time
            continue;
        end
        
        % Check if patient is censored/dead at this time
        if t > event_time
            if event_type == 1
                L_matrix(i, j) = NaN; % Will be black
            elseif event_type == 2
                L_matrix(i, j) = -1; % Special value for gray
            end
        else
            % Find closest time point that's not beyond the current time
            valid_indices = find(times <= t);
            if ~isempty(valid_indices)
                closest_idx = valid_indices(end);
                L_matrix(i, j) = L_values(closest_idx);
            end
        end
    end
end

% Create custom colormap for L (disease intensity) - RED scale
n_colors = 256;
% Create the data colormap (white to dark red)
cmap_L = zeros(n_colors, 3);
for i = 1:n_colors
    % Use a square root function to show more of the lower values
    intensity = sqrt((i-1) / (n_colors-1));
    cmap_L(i, :) = [1, 1-intensity, 1-intensity];  % White to dark red
end

% Apply colormap
colormap(ax1, cmap_L);

% Display the data
imagesc(time_grid, 1:n_subjects, L_matrix);
set(gca, 'YDir', 'reverse');

% Set color limits to actual data range
clim([min_L max_L]);

% Overlay special values using patch for better performance
% Find all NaN (death) and -1 (censoring) values
[death_i, death_j] = find(isnan(L_matrix));
[censor_i, censor_j] = find(L_matrix == -1);

% Plot death regions as black
if ~isempty(death_i)
    for k = 1:length(death_i)
        if death_j(k) < time_resolution
            patch([time_grid(death_j(k)) time_grid(death_j(k)+1) time_grid(death_j(k)+1) time_grid(death_j(k))], ...
                  [death_i(k)-0.5 death_i(k)-0.5 death_i(k)+0.5 death_i(k)+0.5], ...
                  'k', 'EdgeColor', 'none');
        end
    end
end

% Plot censoring regions as gray
if ~isempty(censor_i)
    for k = 1:length(censor_i)
        if censor_j(k) < time_resolution
            patch([time_grid(censor_j(k)) time_grid(censor_j(k)+1) time_grid(censor_j(k)+1) time_grid(censor_j(k))], ...
                  [censor_i(k)-0.5 censor_i(k)-0.5 censor_i(k)+0.5 censor_i(k)+0.5], ...
                  [0.7 0.7 0.7], 'EdgeColor', 'none');
        end
    end
end

% Add colorbar for L
c1 = colorbar;
ylabel(c1, 'Disease Intensity L(t)', 'FontSize', 11);

% Draw group dividers
for i = 1:length(group_dividers)
    if group_dividers(i) > 0 && group_dividers(i) < n_subjects
        line([0 max_time], [group_dividers(i) + 0.5 group_dividers(i) + 0.5], 'Color', 'k', 'LineWidth', 2);
    end
end

% Customize plot
title('Disease Intensity L(t)', 'FontSize', 14);
set(gca, 'YTick', []); % Remove y-axis ticks
set(gca, 'YTickLabel', []); % Remove y-axis labels
set(gca, 'XTickLabel', []); % Remove x-axis labels for top plot
xlim([0, max_time]);
ylim([0.5, length(sorted_subject_ids) + 0.5]);
grid on;

% ===== SECOND SUBPLOT: Treatment Intensity (A values) =====
ax2 = subplot(2, 1, 2);
hold on;

% Create matrices for A values
A_matrix = NaN(length(sorted_subject_ids), time_resolution);

% Find maximum A value for color scaling
max_A = max(T.A);
min_A = 0;  % Minimum is 0

% Process each subject in sorted order
for i = 1:length(sorted_subject_ids)
    % Get patient ID from the sorted list
    patient_id = sorted_subject_ids(i);
    
    % Get data for this patient
    patient_data = T(T.sid == patient_id, :);
    
    % Sort by time
    patient_data = sortrows(patient_data, 't');
    
    % Get time points and A values
    times = patient_data.t;
    A_values = patient_data.A;
    
    % Get event time
    event_time = max(times); % Default to last observation
    event_type = 0; % 0 = no event, 1 = death, 2 = censoring
    if has_V
        event_idx = find(patient_data.Y > 0 | patient_data.V > 0, 1, 'first');
        if ~isempty(event_idx)
            event_time = times(event_idx);
            if patient_data.Y(event_idx) > 0
                event_type = 1; % Death
            else
                event_type = 2; % Censoring
            end
        end
    else
        event_idx = find(patient_data.Y > 0, 1, 'first');
        if ~isempty(event_idx)
            event_time = times(event_idx);
            event_type = 1; % Death
        end
    end
    
    % Interpolate A values to regular grid
    for j = 1:time_resolution
        t = time_grid(j);
        
        % Skip if beyond max time
        if t > max_time
            continue;
        end
        
        % Check if patient is censored/dead at this time
        if t > event_time
            if event_type == 1
                A_matrix(i, j) = NaN; % Will be black
            elseif event_type == 2
                A_matrix(i, j) = -1; % Special value for gray
            end
        else
            % Find closest time point that's not beyond the current time
            valid_indices = find(times <= t);
            if ~isempty(valid_indices)
                closest_idx = valid_indices(end);
                A_matrix(i, j) = A_values(closest_idx);
            end
        end
    end
end

% Create custom colormap for A (treatment intensity) - BLUE scale
% Create the data colormap (white to dark blue)
cmap_A = zeros(n_colors, 3);
for i = 1:n_colors
    % Use a square root function to show more of the lower values
    intensity = sqrt((i-1) / (n_colors-1));
    cmap_A(i, :) = [1-intensity, 1-intensity, 1];  % White to dark blue
end

% Apply colormap
colormap(ax2, cmap_A);

% Display the data
imagesc(time_grid, 1:n_subjects, A_matrix);
set(gca, 'YDir', 'reverse');

% Set color limits to actual data range
clim([min_A max_A]);

% Overlay special values using patch for better performance
% Find all NaN (death) and -1 (censoring) values
[death_i_A, death_j_A] = find(isnan(A_matrix));
[censor_i_A, censor_j_A] = find(A_matrix == -1);

% Plot death regions as black
if ~isempty(death_i_A)
    for k = 1:length(death_i_A)
        if death_j_A(k) < time_resolution
            patch([time_grid(death_j_A(k)) time_grid(death_j_A(k)+1) time_grid(death_j_A(k)+1) time_grid(death_j_A(k))], ...
                  [death_i_A(k)-0.5 death_i_A(k)-0.5 death_i_A(k)+0.5 death_i_A(k)+0.5], ...
                  'k', 'EdgeColor', 'none');
        end
    end
end

% Plot censoring regions as gray
if ~isempty(censor_i_A)
    for k = 1:length(censor_i_A)
        if censor_j_A(k) < time_resolution
            patch([time_grid(censor_j_A(k)) time_grid(censor_j_A(k)+1) time_grid(censor_j_A(k)+1) time_grid(censor_j_A(k))], ...
                  [censor_i_A(k)-0.5 censor_i_A(k)-0.5 censor_i_A(k)+0.5 censor_i_A(k)+0.5], ...
                  [0.7 0.7 0.7], 'EdgeColor', 'none');
        end
    end
end

% Add colorbar for A
c2 = colorbar;
ylabel(c2, 'Treatment Intensity A(t)', 'FontSize', 11);

% Draw group dividers
for i = 1:length(group_dividers)
    if group_dividers(i) > 0 && group_dividers(i) < n_subjects
        line([0 max_time], [group_dividers(i) + 0.5 group_dividers(i) + 0.5], 'Color', 'k', 'LineWidth', 2);
    end
end

% Customize plot
title('Treatment Intensity A(t)', 'FontSize', 14);
set(gca, 'YTick', []); % Remove y-axis ticks
set(gca, 'YTickLabel', []); % Remove y-axis labels
xlabel('Time (hours)', 'FontSize', 12);
xlim([0, max_time]);
ylim([0.5, length(sorted_subject_ids) + 0.5]);
grid on;

% Overall figure settings
set(gcf, 'Color', 'white');

% Manually adjust subplot positions to minimize white space and use full width
% Define spacing parameters
vertical_gap = 0.02;     % Gap between subplots vertically
left_margin = 0.06;      % Left margin
right_margin = 0.10;     % Right margin (larger for colorbar)
top_margin = 0.05;       % Top margin
bottom_margin = 0.08;    % Bottom margin

% Calculate width using full figure width
width = 1 - left_margin - right_margin;

% Calculate heights for 2 plots plus space for survival plot
total_height = 1 - top_margin - bottom_margin;
% Reserve space for third plot (survival curves) even though it's not in this function
subplot_height = (total_height - 2*vertical_gap) * 2/3 / 2; % 2/3 of space for 2 plots

% Set new positions
% First subplot (L values)
set(ax1, 'Position', [left_margin, ...
                      1 - top_margin - subplot_height, ...
                      width, ...
                      subplot_height]);

% Second subplot (A values)
set(ax2, 'Position', [left_margin, ...
                      1 - top_margin - 2*subplot_height - vertical_gap, ...
                      width, ...
                      subplot_height]);

end