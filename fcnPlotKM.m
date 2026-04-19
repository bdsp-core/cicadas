function [s0,s1,t0,t1] = fcnPlotKM(T1)
%% KM curve - empirical with aligned time grid
% Returns curves aligned to t = 0, 2, 4, ..., 168 (85 values)

N = max(T1.sid);

% Extract survival data for each patient
survival_data = [];
for i = 1:N
    patient_data = T1(T1.sid == i, :);
    initial_treatment = patient_data.Rx(1);
    
    % Time to event (death or censoring)
    if any(patient_data.Y == 1)
        event_time = patient_data.t(find(patient_data.Y == 1, 1));
        event_indicator = 1;
    else
        event_time = patient_data.t(end);
        event_indicator = 0;
    end
    
    survival_data = [survival_data; struct('time', event_time, ...
                                          'event', event_indicator, ...
                                          'treatment', initial_treatment)];
end

% Convert to table for easier manipulation
surv_table = struct2table(survival_data);

% Separate by treatment group
treated_data = surv_table(surv_table.treatment == 1, :);
control_data = surv_table(surv_table.treatment == 0, :);

% Calculate Kaplan-Meier estimates
[t_treat_raw, s_treat_raw, ~, ~] = fcn_kaplanMeier(treated_data.time, treated_data.event);
[t_control_raw, s_control_raw, ~, ~] = fcn_kaplanMeier(control_data.time, control_data.event);

% Add initial point (t=0, s=1) to raw curves
s1_raw = [1; s_treat_raw];
t1_raw = [0; t_treat_raw];
s0_raw = [1; s_control_raw];
t0_raw = [0; t_control_raw];

% Define the regular time grid
t_grid = 0:2:168;  % This gives exactly 85 points
n_points = length(t_grid);  % Should be 85

% Initialize output arrays
s0 = zeros(n_points, 1);
s1 = zeros(n_points, 1);
t0 = t_grid(:);  % Column vector
t1 = t_grid(:);  % Column vector

% Interpolate KM curves to regular grid using step function interpolation
% For KM curves, we use 'previous' interpolation to maintain the step function nature

% For control group (untreated)
for i = 1:n_points
    % Find the last event time before or at current grid time
    idx = find(t0_raw <= t_grid(i), 1, 'last');
    if isempty(idx)
        s0(i) = 1;  % No events before this time
    else
        s0(i) = s0_raw(idx);
    end
end

% For treated group
for i = 1:n_points
    % Find the last event time before or at current grid time
    idx = find(t1_raw <= t_grid(i), 1, 'last');
    if isempty(idx)
        s1(i) = 1;  % No events before this time
    else
        s1(i) = s1_raw(idx);
    end
end

% Handle edge case: if last observed time is before 168, extend with last value
if max(t0_raw) < 168
    last_idx = find(t_grid > max(t0_raw), 1);
    if ~isempty(last_idx)
        s0(last_idx:end) = s0_raw(end);
    end
end

if max(t1_raw) < 168
    last_idx = find(t_grid > max(t1_raw), 1);
    if ~isempty(last_idx)
        s1(last_idx:end) = s1_raw(end);
    end
end

% Verify output dimensions
assert(length(s0) == 85, 'Control survival curve should have 85 points');
assert(length(s1) == 85, 'Treated survival curve should have 85 points');
assert(length(t0) == 85, 'Control time vector should have 85 points');
assert(length(t1) == 85, 'Treated time vector should have 85 points');

end

% function [s0,s1,t0,t1] = fcnPlotKM(T1)
% 
% %% KM curve - empirical
% N= max(T1.sid); 
% 
% % Extract survival data for each patient
% survival_data = [];
% for i = 1:N
%     patient_data = T1(T1.sid == i, :);
%     initial_treatment = patient_data.Rx(1);
% 
%     % Time to event (death or censoring)
%     if any(patient_data.Y == 1)
%         event_time = patient_data.t(find(patient_data.Y == 1, 1));
%         event_indicator = 1;
%     else
%         event_time = patient_data.t(end);
%         event_indicator = 0;
%     end
% 
%     survival_data = [survival_data; struct('time', event_time, ...
%                                            'event', event_indicator, ...
%                                            'treatment', initial_treatment)];
% end
% 
% % Convert to table for easier manipulation
% surv_table = struct2table(survival_data);
% 
% % Separate by treatment group
% treated_data = surv_table(surv_table.treatment == 1, :);
% control_data = surv_table(surv_table.treatment == 0, :);
% 
% % Calculate Kaplan-Meier estimates
% [t_treat, s_treat, se_treat, n_risk_treat] = fcn_kaplanMeier(treated_data.time, treated_data.event);
% [t_control, s_control, se_control, n_risk_control] = fcn_kaplanMeier(control_data.time, control_data.event);
% 
% s1 = [1; s_treat];
% t1 = [0; t_treat];
% 
% s0 = [1; s_control];
% t0 = [0; t_control];