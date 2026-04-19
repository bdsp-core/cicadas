function fcnSingleSwimmerPlot_v4(T)
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

% Create matrices for L values - use RGB for direct color control
L_image = ones(length(sorted_subject_ids), time_resolution, 3); % Initialize as white

% Find maximum L value for color scaling
max_L = max(T.L);

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
    
    % Get event time and type
    event_time = max(times); % Default to last observation
    event_is_death = false;
    event_is_censored = false;
    
    if has_V
        event_idx = find(patient_data.Y > 0 | patient_data.V > 0, 1, 'first');
        if ~isempty(event_idx)
            event_time = times(event_idx);
            event_is_death = patient_data.Y(event_idx) > 0;
            event_is_censored = patient_data.V(event_idx) > 0;
        end
    else
        event_idx = find(patient_data.Y > 0, 1, 'first');
        if ~isempty(event_idx)
            event_time = times(event_idx);
            event_is_death = true;
        end
    end
    
    % Fill color values for each time point
    for j = 1:time_resolution
        t = time_grid(j);
        
        % Skip if beyond max time
        if t > max_time
            continue;
        end
        
        % Check if patient is censored/dead at this time
        if t > event_time
            if event_is_death
                % Black for death
                L_image(i, j, :) = [0, 0, 0];
            elseif event_is_censored
                % Light gray for censoring
                L_image(i, j, :) = [0.7, 0.7, 0.7];
            end
        else
            % Find closest time point that's not beyond the current time
            valid_indices = find(times <= t);
            if ~isempty(valid_indices)
                closest_idx = valid_indices(end);
                L_value = L_values(closest_idx);
                
                % Map L value to color (light red to dark red)
                % Use square root to show more of lower values
                intensity = sqrt(L_value / max_L);
                % Light red [1, 0.8, 0.8] to Dark red [0.5, 0, 0]
                L_image(i, j, :) = [1 - 0.5*intensity, 0.8*(1-intensity), 0.8*(1-intensity)];
            end
        end
    end
end

% Display the image
image(time_grid, 1:n_subjects, L_image);
set(gca, 'YDir', 'reverse');


% Draw group dividers
for i = 1:length(group_dividers)
    if group_dividers(i) > 0 && group_dividers(i) < n_subjects
        line([0 max_time], [group_dividers(i) + 0.5 group_dividers(i) + 0.5], 'Color', 'w', 'LineWidth', 3);
    end
end

% Customize plot
set(gca, 'YTick', []); % Remove y-axis ticks
set(gca, 'YTickLabel', []); % Remove y-axis labels
set(gca, 'XTickLabel', []); % Remove x-axis labels for top plot
xlim([0, max_time]);
ylim([0.5, length(sorted_subject_ids) + 0.5]);
grid on;

% Add L_t label on the left side
ylabel('L_t', 'FontSize', 14, 'FontWeight', 'bold', 'Rotation', 0, 'HorizontalAlignment', 'right');

% Add "Treated" and "Untreated" labels for L_t plot
treated_end = length(group1_ids) + length(group2_ids) + length(group3_ids);
untreated_start = treated_end + 1;
untreated_end = n_subjects;

% Position labels near the bottom of each group
treated_label_y = treated_end - 100;  % Near bottom of treated group
untreated_label_y = untreated_end - 100;  % Near bottom of untreated group

% Ensure labels are within bounds
if treated_label_y < 1, treated_label_y = 1; end
if untreated_label_y < untreated_start, untreated_label_y = untreated_start; end

text(max_time * 0.02, treated_label_y, 'Treated', 'FontSize', 12, 'FontWeight', 'bold', ...
     'Color', 'black', 'HorizontalAlignment', 'left');
text(max_time * 0.02, untreated_label_y, 'Untreated', 'FontSize', 12, 'FontWeight', 'bold', ...
     'Color', 'black', 'HorizontalAlignment', 'left');

% ===== SECOND SUBPLOT: Treatment Intensity (A values) =====
ax2 = subplot(2, 1, 2);
hold on;

% Create matrices for A values - use RGB for direct color control
A_image = ones(length(sorted_subject_ids), time_resolution, 3); % Initialize as white

% Find maximum A value for color scaling
max_A = max(T.A);

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
    
    % Get event time and type
    event_time = max(times); % Default to last observation
    event_is_death = false;
    event_is_censored = false;
    
    if has_V
        event_idx = find(patient_data.Y > 0 | patient_data.V > 0, 1, 'first');
        if ~isempty(event_idx)
            event_time = times(event_idx);
            event_is_death = patient_data.Y(event_idx) > 0;
            event_is_censored = patient_data.V(event_idx) > 0;
        end
    else
        event_idx = find(patient_data.Y > 0, 1, 'first');
        if ~isempty(event_idx)
            event_time = times(event_idx);
            event_is_death = true;
        end
    end
    
    % Fill color values for each time point
    for j = 1:time_resolution
        t = time_grid(j);
        
        % Skip if beyond max time
        if t > max_time
            continue;
        end
        
        % Check if patient is censored/dead at this time
        if t > event_time
            if event_is_death
                % Black for death
                A_image(i, j, :) = [0, 0, 0];
            elseif event_is_censored
                % Light gray for censoring
                A_image(i, j, :) = [0.7, 0.7, 0.7];
            end
        else
            % Find closest time point that's not beyond the current time
            valid_indices = find(times <= t);
            if ~isempty(valid_indices)
                closest_idx = valid_indices(end);
                A_value = A_values(closest_idx);
                
                % Map A value to color (light blue to dark blue)
                % Use square root to show more of lower values
                intensity = sqrt(A_value / max_A);
                % Light blue [0.8, 0.8, 1] to Dark blue [0, 0, 0.5]
                A_image(i, j, :) = [0.8*(1-intensity), 0.8*(1-intensity), 1 - 0.5*intensity];
            end
        end
    end
end

% Display the image
image(time_grid, 1:n_subjects, A_image);
set(gca, 'YDir', 'reverse');


% Draw group dividers
for i = 1:length(group_dividers)
    if group_dividers(i) > 0 && group_dividers(i) < n_subjects
        line([0 max_time], [group_dividers(i) + 0.5 group_dividers(i) + 0.5], 'Color', 'w', 'LineWidth', 3);
    end
end

% Customize plot
set(gca, 'YTick', []); % Remove y-axis ticks
set(gca, 'YTickLabel', []); % Remove y-axis labels
set(gca, 'XTickLabel', []); % Remove x-axis labels for middle plot
xlim([0, max_time]);
ylim([0.5, length(sorted_subject_ids) + 0.5]);
grid on;

% Add A_t label on the left side
ylabel('A_t', 'FontSize', 14, 'FontWeight', 'bold', 'Rotation', 0, 'HorizontalAlignment', 'right');

% Add "Treated" and "Untreated" labels for A_t plot
text(max_time * 0.02, treated_label_y, 'Treated', 'FontSize', 12, 'FontWeight', 'bold', ...
     'Color', 'black', 'HorizontalAlignment', 'left');
text(max_time * 0.02, untreated_label_y, 'Untreated', 'FontSize', 12, 'FontWeight', 'bold', ...
     'Color', 'black', 'HorizontalAlignment', 'left');

% Overall figure settings
set(gcf, 'Color', 'white');

% Manually adjust subplot positions to minimize white space and use full width
% Define spacing parameters
vertical_gap = 0.015;    % Gap between subplots vertically
left_margin = 0.06;      % Left margin
right_margin = 0.10;     % Right margin (larger for colorbar)
top_margin = 0.02;       % Top margin
bottom_margin = 0.20;    % Bottom margin for survival plot

% Calculate width using full figure width
width = 1 - left_margin - right_margin;

% Calculate heights for 2 plots plus space for survival plot
total_height = 1 - top_margin - bottom_margin;
% Use 2/3 of available height for the two swimmer plots
subplot_height = (total_height * 2/3 - vertical_gap) / 2;

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