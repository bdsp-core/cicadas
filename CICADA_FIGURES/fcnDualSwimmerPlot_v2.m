function fcnDualSwimmerPlot_v2(T, side)
% Create swimmer plots for the epilepsy simulation data with properly displayed
% treatment status, death, and censoring
%
% T: Table with columns sid, t, Rx, harmE, harmA, b0, L, A, V, Y
% side: 'L' for left side subplots, 'R' for right side subplots
%
% subplot positions depend on side argument:
% 'L': subplot(4,2,1), subplot(4,2,3), subplot(4,2,5)
% 'R': subplot(4,2,2), subplot(4,2,4), subplot(4,2,6)

% Validate side argument
if ~exist('side', 'var') || ~ismember(upper(side), {'L', 'R'})
    error('Side argument must be ''L'' or ''R''');
end

% Determine subplot positions based on side
if upper(side) == 'L'
    subplot_positions = [1, 3, 5];
else
    subplot_positions = [2, 4, 6];
end

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
status_matrix = ones(n_subjects, time_resolution, 3); % White background

% Define colors
treatment_color = [0.6350 0.0780 0.1840];  % Red for on treatment
no_treatment_color = [0 0.4470 0.7410];    % Blue for off treatment 
death_color = [0, 0, 0];                   % Black for death
censored_color = [0.7, 0.7, 0.7];          % Light gray for censoring

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

% Define the groups (now including non-adherent as a separate category):
% Note: censored now includes V1 (treatment stop), V2 (non-adherent), and V3 (dropout)
% Group 1: Initially on treatment, who died
% Group 2: Initially on treatment, who were censored
% Group 3: Initially on treatment, who completed
% Group 4: Initially not on treatment, who died
% Group 5: Initially not on treatment, who were censored
% Group 6: Initially not on treatment, who completed

% Determine who was censored
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

% ===== FIRST SUBPLOT: Treatment Status and Outcome =====
ax1 = subplot(4, 2, subplot_positions(1));
hold on;

% Reset the status matrix to ensure clean state
status_matrix = ones(n_subjects, time_resolution, 3); % White background

% Fill in the status matrix
for i = 1:length(sorted_subject_ids)
    % Get patient ID from the sorted list
    patient_id = sorted_subject_ids(i);
    
    % Get data for this patient
    patient_data = T(T.sid == patient_id, :);
    
    % Sort by time
    patient_data = sortrows(patient_data, 't');
    
    % Get times, treatment status, and death
    patient_times = patient_data.t;
    patient_rx = patient_data.Rx;
    patient_died = patient_data.Y;
    
    % Handle censoring
    if has_V
        patient_censored = patient_data.V;
    else
        patient_censored = zeros(size(patient_times));
    end
    
    % Find time of event (death or censoring)
    event_time = max(patient_times); % Default to last observation
    if has_V
        event_idx = find(patient_died > 0 | patient_censored > 0, 1, 'first');
    else
        event_idx = find(patient_died > 0, 1, 'first');
    end
    if ~isempty(event_idx)
        event_time = patient_times(event_idx);
    end
    
    % Determine event type
    event_is_death = false;
    event_is_censored = false;
    if ~isempty(event_idx)
        event_is_death = patient_died(event_idx) > 0;
        event_is_censored = patient_censored(event_idx) > 0;
    end
    
    % Fill in status for each time point
    for j = 1:time_resolution
        t = time_grid(j);
        
        % Skip if beyond max time
        if t > max_time
            continue;
        end
        
        % Determine color based on status
        if t > event_time
            % After event (death or censoring)
            if event_is_death
                status_matrix(i, j, :) = reshape(death_color, [1, 1, 3]);
            elseif event_is_censored
                status_matrix(i, j, :) = reshape(censored_color, [1, 1, 3]);
            end
        else
            % Before event - show treatment status
            % Find closest time index that's not beyond the current time
            valid_indices = find(patient_times <= t);
            if ~isempty(valid_indices)
                closest_idx = valid_indices(end);
                % Assign color based on treatment status at or before this time
                if patient_rx(closest_idx) == 1
                    status_matrix(i, j, :) = reshape(treatment_color, [1, 1, 3]);
                else
                    status_matrix(i, j, :) = reshape(no_treatment_color, [1, 1, 3]);
                end
            end
        end
    end
end

% Display the status image
image(time_grid, 1:n_subjects, status_matrix);
set(gca, 'YDir', 'reverse'); % To have first subject at the top

% Draw horizontal lines between groups if there are patients in each group
for i = 1:length(group_dividers)
    if group_dividers(i) > 0 && group_dividers(i) < length(sorted_subject_ids)
        line([0 max_time], [group_dividers(i) + 0.5 group_dividers(i) + 0.5], 'Color', 'k', 'LineWidth', 2);
    end
end

% Customize plot - no title, no x-label for top plot
set(gca, 'YTick', []); % Remove y-axis ticks
set(gca, 'YTickLabel', []); % Remove y-axis labels
set(gca, 'XTickLabel', []); % Remove x-axis labels for top plot
xlim([0, max_time]);
ylim([0.5, length(sorted_subject_ids) + 0.5]);
grid on;

% ===== SECOND SUBPLOT: L Values (Time-varying Covariate) =====
ax2 = subplot(4, 2, subplot_positions(2));
hold on;

% Create matrices for L values
L_matrix = NaN(length(sorted_subject_ids), time_resolution);

% Find maximum L value for color scaling
max_L = max(T.L);

% Define special values for post-event periods
% We'll use values outside the normal data range
death_value = -2;     % Will map to black
censor_value = -1;    % Will map to gray

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
    if has_V
        event_idx = find(patient_data.Y > 0 | patient_data.V > 0, 1, 'first');
    else
        event_idx = find(patient_data.Y > 0, 1, 'first');
    end
    if ~isempty(event_idx)
        event_time = times(event_idx);
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
            % Determine if death or censoring occurred
            if ~isempty(event_idx)
                if patient_data.Y(event_idx) > 0
                    % Death - use special value
                    L_matrix(i, j) = death_value;
                else
                    % Censoring - use special value
                    L_matrix(i, j) = censor_value;
                end
            end
            % Otherwise leave as NaN
        else
            % Find closest time point that's not beyond the current time
            valid_indices = find(times <= t);
            if ~isempty(valid_indices)
                closest_idx = valid_indices(end);
                
                % Store L value
                L_matrix(i, j) = L_values(closest_idx);
            end
        end
    end
end

% Plot L values as image
imagesc(time_grid, 1:n_subjects, L_matrix);
set(gca, 'YDir', 'reverse'); % To match the top plot orientation

% Create custom colormap with special values for death/censoring
n_colors = 256;
custom_cmap = zeros(n_colors, 3);

% Map the color indices based on data range
% death_value = -2, censor_value = -1, data values = [0, max_L]
total_range = max_L - death_value;  % From -2 to max_L

% Allocate colormap sections
death_end = round((0 - death_value) / total_range * n_colors * 0.4);  % First 40% for special values
censor_end = round((0 - death_value) / total_range * n_colors * 0.8); % Next 40% for special values

% Black for death (first section)
custom_cmap(1:death_end, :) = 0;  % Black

% Gray for censoring (second section)
custom_cmap(death_end+1:censor_end, :) = 0.7;  % Light gray

% Grayscale for actual data (remaining section)
data_colors = n_colors - censor_end;
gray_gradient = flipud(gray(data_colors));
custom_cmap(censor_end+1:end, :) = gray_gradient;

colormap(gca, custom_cmap);
clim([death_value max_L]);

% Draw group dividers
for i = 1:length(group_dividers)
    if group_dividers(i) > 0 && group_dividers(i) < n_subjects
        line([0 max_time], [group_dividers(i) + 0.5 group_dividers(i) + 0.5], 'Color', 'k', 'LineWidth', 2);
    end
end

% Customize plot - no title, no x-label for middle plot
set(gca, 'YTick', []); % Remove y-axis ticks
set(gca, 'YTickLabel', []); % Remove y-axis labels
set(gca, 'XTickLabel', []); % Remove x-axis labels for middle plot
xlim([0, max_time]);
ylim([0.5, length(sorted_subject_ids) + 0.5]);
grid on;

% ===== THIRD SUBPLOT: Treatment Intensity (A values) =====
ax3 = subplot(4, 2, subplot_positions(3));
hold on;

% Create matrices for A values
A_matrix = NaN(length(sorted_subject_ids), time_resolution);

% Find maximum A value for color scaling
max_A = max(T.A);

% Use same special values as L subplot (already defined above)

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
    if has_V
        event_idx = find(patient_data.Y > 0 | patient_data.V > 0, 1, 'first');
    else
        event_idx = find(patient_data.Y > 0, 1, 'first');
    end
    if ~isempty(event_idx)
        event_time = times(event_idx);
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
            % Determine if death or censoring occurred
            if ~isempty(event_idx)
                if patient_data.Y(event_idx) > 0
                    % Death - use special value
                    A_matrix(i, j) = death_value;
                else
                    % Censoring - use special value
                    A_matrix(i, j) = censor_value;
                end
            end
            % Otherwise leave as NaN
        else
            % Find closest time point that's not beyond the current time
            valid_indices = find(times <= t);
            if ~isempty(valid_indices)
                closest_idx = valid_indices(end);
                
                % Store A value
                A_matrix(i, j) = A_values(closest_idx);
            end
        end
    end
end

% Plot A values as image
imagesc(time_grid, 1:n_subjects, A_matrix);
set(gca, 'YDir', 'reverse'); % To match the top plot orientation

% Use same custom colormap approach as L subplot
% Recalculate for A value range
total_range_A = max_A - death_value;  % From -2 to max_A

% Allocate colormap sections
death_end_A = round((0 - death_value) / total_range_A * n_colors * 0.3);  % Adjust proportions for A range
censor_end_A = round((0 - death_value) / total_range_A * n_colors * 0.6);

% Black for death (first section)
custom_cmap_A = zeros(n_colors, 3);
custom_cmap_A(1:death_end_A, :) = 0;  % Black

% Gray for censoring (second section)
custom_cmap_A(death_end_A+1:censor_end_A, :) = 0.7;  % Light gray

% Grayscale for actual data (remaining section)
data_colors_A = n_colors - censor_end_A;
gray_gradient_A = flipud(gray(data_colors_A));
custom_cmap_A(censor_end_A+1:end, :) = gray_gradient_A;

colormap(gca, custom_cmap_A);
clim([death_value max_A]);

% Draw group dividers
for i = 1:length(group_dividers)
    if group_dividers(i) > 0 && group_dividers(i) < n_subjects
        line([0 max_time], [group_dividers(i) + 0.5 group_dividers(i) + 0.5], 'Color', 'k', 'LineWidth', 2);
    end
end

% Customize plot - no title, but include x-label for bottom plot
set(gca, 'YTick', []); % Remove y-axis ticks
set(gca, 'YTickLabel', []); % Remove y-axis labels
set(gca, 'XTickLabel', []); % Remove x-axis labels
xlim([0, max_time]);
ylim([0.5, length(sorted_subject_ids) + 0.5]);
grid on;

% Overall figure settings
set(gcf, 'Color', 'white');

% Manually adjust subplot positions to minimize white space
% Get current positions
pos1 = get(ax1, 'Position');
pos2 = get(ax2, 'Position');
pos3 = get(ax3, 'Position');

% Define spacing parameters
vertical_gap = 0.005;    % Very small gap between subplots vertically
horizontal_margin = 0.06; % Left/right margins
top_margin = 0.02;      % Top margin
bottom_margin = 0.05;   % Bottom margin (larger to accommodate xlabel)
middle_gap = 0.01;      % Gap between left and right columns

% Calculate width based on side
if upper(side) == 'L'
    left_margin = horizontal_margin;
    width = 0.5 - horizontal_margin - middle_gap/2; % Half the middle gap
else
    left_margin = 0.5 + middle_gap/2; % Start after left column with half the gap
    width = 0.5 - horizontal_margin - middle_gap/2;
end

% Calculate heights for 4 rows (3 used, 1 reserved)
total_height = 1 - top_margin - bottom_margin;
subplot_height = (total_height - 3*vertical_gap) * 3/4 / 3; % 3/4 of space for 3 plots

% Set new positions
% First subplot (row 1)
set(ax1, 'Position', [left_margin, ...
                      1 - top_margin - subplot_height, ...
                      width, ...
                      subplot_height]);

% Second subplot (row 2)
set(ax2, 'Position', [left_margin, ...
                      1 - top_margin - 2*subplot_height - vertical_gap, ...
                      width, ...
                      subplot_height]);

% Third subplot (row 3)
set(ax3, 'Position', [left_margin, ...
                      1 - top_margin - 3*subplot_height - 2*vertical_gap, ...
                      width, ...
                      subplot_height]);

end