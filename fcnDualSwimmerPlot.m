function fcnDualSwimmerPlot(T)
% Create swimmer plots for the epilepsy simulation data with properly displayed
% treatment status, death, and censoring
%
% T: Table with columns sid, t, Rx, harmE, harmA, b0, L, A, V, Y
%
% subplot(311): Use color of "lane" to indicate treatment status and outcome
% subplot(312): Use color of "lane" to show the value of L (time-varying covariate)
% subplot(313): Use color of "lane" to show treatment intensity (A)

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

% Initialize status matrix with background color (white)
time_resolution = 500;
time_grid = linspace(0, max_time, time_resolution);
status_matrix = ones(n_subjects, time_resolution, 3); % White background

% Define colors
treatment_color = [0.6350 0.0780 0.1840];  % Red for on treatment
no_treatment_color = [0 0.4470 0.7410];    % Blue for off treatment 
death_color = [0, 0, 0];                   % Black for death
censored_color = [0.7, 0.7, 0.7];          % Gray for censored

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
    
    % Check if patient was censored
    censored(i) = any(patient_data.V > 0);
    
    % Find time of death or censoring
    event_time = max_times(i); % Default to last observation
    event_idx = find(patient_data.Y > 0 | patient_data.V > 0, 1, 'first');
    if ~isempty(event_idx)
        event_times(i) = patient_data.t(event_idx);
    else
        event_times(i) = event_time;
    end
end

% Define the six groups:
% Group 1: Initially on treatment, who died
% Group 2: Initially on treatment, who were censored
% Group 3: Initially on treatment, who completed (neither died nor censored)
% Group 4: Initially not on treatment, who died
% Group 5: Initially not on treatment, who were censored
% Group 6: Initially not on treatment, who completed (neither died nor censored)
completed = ~died & ~censored;  % Patients who completed the study

group1 = initial_treatment & died;
group2 = initial_treatment & censored;
group3 = initial_treatment & completed;
group4 = ~initial_treatment & died;
group5 = ~initial_treatment & censored;
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

% Create figure with three subplots
figure(2); clf; 

% ===== FIRST SUBPLOT: Treatment Status and Outcome =====
subplot(3, 1, 1);
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
    
    % Get times, treatment status, censoring and death
    patient_times = patient_data.t;
    patient_rx = patient_data.Rx;
    patient_censored = patient_data.V;
    patient_died = patient_data.Y;
    
    % Find time of event (death or censoring)
    event_time = max(patient_times); % Default to last observation
    event_idx = find(patient_died > 0 | patient_censored > 0, 1, 'first');
    if ~isempty(event_idx)
        event_time = patient_times(event_idx);
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
            if any(patient_died > 0)
                status_matrix(i, j, :) = reshape(death_color, [1, 1, 3]);
            elseif any(patient_censored > 0)
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

% Customize plot
title('Patient Status Over Time', 'FontSize', 14);
xlabel('Time [hours]');
set(gca, 'YTick', []); % Remove y-axis ticks
set(gca, 'YTickLabel', []); % Remove y-axis labels
xlim([0, max_time]);
ylim([0.5, length(sorted_subject_ids) + 0.5]);
grid on;

% ===== SECOND SUBPLOT: L Values (Time-varying Covariate) =====
subplot(3, 1, 2);
hold on;

% Create matrix for L values
L_matrix = zeros(length(sorted_subject_ids), time_resolution);

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
    
    % Get event time
    event_time = max(times); % Default to last observation
    event_idx = find(patient_data.Y > 0 | patient_data.V > 0, 1, 'first');
    if ~isempty(event_idx)
        event_time = times(event_idx);
    end
    
    % Interpolate L values to regular grid
    for j = 1:time_resolution
        t = time_grid(j);
        
        % Skip if beyond max time (not just event time)
        if t > max_time
            continue;
        end
        
        % Find closest time point that's not beyond the current time
        valid_indices = find(times <= t);
        if ~isempty(valid_indices)
            closest_idx = valid_indices(end);
            
            % Store L value
            L_matrix(i, j) = L_values(closest_idx);
        end
    end
end

% Plot L values as image
imagesc(time_grid, 1:n_subjects, L_matrix);
set(gca, 'YDir', 'reverse'); % To match the top plot orientation
colormap(flipud(gray));

% Draw group dividers
for i = 1:length(group_dividers)
    if group_dividers(i) > 0 && group_dividers(i) < n_subjects
        line([0 max_time], [group_dividers(i) + 0.5 group_dividers(i) + 0.5], 'Color', 'k', 'LineWidth', 2);
    end
end

% Customize plot
title('L Values Over Time (Time-varying Covariate)', 'FontSize', 14);
xlabel('Time [hours]');
set(gca, 'YTick', []); % Remove y-axis ticks
set(gca, 'YTickLabel', []); % Remove y-axis labels
xlim([0, max_time]);
ylim([0.5, length(sorted_subject_ids) + 0.5]);
grid on;

% ===== THIRD SUBPLOT: Treatment Intensity (A values) =====
subplot(3, 1, 3);
hold on;

% Create matrix for A values
A_matrix = zeros(length(sorted_subject_ids), time_resolution);

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
    
    % Get event time
    event_time = max(times); % Default to last observation
    event_idx = find(patient_data.Y > 0 | patient_data.V > 0, 1, 'first');
    if ~isempty(event_idx)
        event_time = times(event_idx);
    end
    
    % Interpolate A values to regular grid
    for j = 1:time_resolution
        t = time_grid(j);
        
        % Skip if beyond max time (not just event time)
        if t > max_time
            continue;
        end
        
        % Find closest time point that's not beyond the current time
        valid_indices = find(times <= t);
        if ~isempty(valid_indices)
            closest_idx = valid_indices(end);
            
            % Store A value
            A_matrix(i, j) = A_values(closest_idx);
        end
    end
end

% Plot A values as image
imagesc(time_grid, 1:n_subjects, A_matrix);
set(gca, 'YDir', 'reverse'); % To match the top plot orientation
colormap(flipud(gray));

% Draw group dividers
for i = 1:length(group_dividers)
    if group_dividers(i) > 0 && group_dividers(i) < n_subjects
        line([0 max_time], [group_dividers(i) + 0.5 group_dividers(i) + 0.5], 'Color', 'k', 'LineWidth', 2);
    end
end

% Customize plot
title('Treatment Intensity (A) Over Time', 'FontSize', 14);
xlabel('Time [hours]');
set(gca, 'YTick', []); % Remove y-axis ticks
set(gca, 'YTickLabel', []); % Remove y-axis labels
xlim([0, max_time]);
ylim([0.5, length(sorted_subject_ids) + 0.5]);
grid on;

% Overall figure settings
set(gcf, 'Color', 'white');
set(gcf, 'Position', [100, 100, 900, 800]);
end