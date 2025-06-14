%% a4_survivalCurves.m - Read trial data and plot survival/hazard curves

clear; clc; format compact;

%% 1. Read the trial data from CSV
filename = 'trial_data.csv';
fprintf('Reading trial data from %s...\n', filename);
T = readtable(filename);
fprintf('Loaded %d rows of data for %d patients\n', height(T), length(unique(T.sid)));

%% 2. Set model parameters (same as in a3_simulate_empirical.m)
C = 3;      % Increased for higher drug potency
g = 4;      % Steeper dose-response curve
kp = 0;     % Pure integral control
ki = 20;    % Very aggressive control for tight disease suppression
Amax = 50;  % pump upper bound
th = 0.05;  % Very low target for tight control
b0 = 0.1;   % Baseline risk
pulseAmp = 1; pulseMu = 20; pulseWidth = 1.8; pulseC = 15;
N = length(unique(T.sid));

%% 3. Calculate empirical survival curves and hazards
[h_rx1, h_rx0, S_rx1, S_rx0, t_rx1, t_rx0] = fcnEmpiricalSurvivalCurves(T);

%% 4. Calculate empirical hazards directly from the data
fprintf('Calculating empirical hazards directly from longitudinal data...\n');

% Get hazard scaling factors from simulation
hazard_scale_t1 = 0.001/2;
hazard_scale_t2 = 0.005/2;

% Calculate empirical hazards for each group
[h_empirical_rx1, h_empirical_rx0, t_hazard] = calculateEmpiricalHazardsFromData(T, hazard_scale_t1, hazard_scale_t2, b0);

fprintf('Empirical hazards calculated from actual L and A trajectories\n');

% Check group sizes
n_treated = sum(T.Rx == 1 & T.t == 0);
n_untreated = sum(T.Rx == 0 & T.t == 0);
fprintf('Group sizes: Treated = %d (%.1f%%), Untreated = %d (%.1f%%)\n', ...
    n_treated, 100*n_treated/N, n_untreated, 100*n_untreated/N);

%% 5. Calculate theoretical curves
fprintf('Calculating theoretical curves for population...\n');

% For survival curves, we use population-level averages
sim_mode = 0;
n_theory_sims = 500;
S_theory_treated_matrix = zeros(n_theory_sims, 337);
S_theory_control_matrix = zeros(n_theory_sims, 337);

for i = 1:n_theory_sims
    hE_num = rand;
    hA_num = rand;
    harmE_pop = 50*hE_num;
    harmA_pop = 1*hA_num;
    
    [t_sim, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, SY_treated] = ...
        fcnRunSimulation_v2(1, harmE_pop, harmA_pop, C, g, th, kp, ki, Amax, b0, pulseAmp, pulseMu, pulseWidth, pulseC, sim_mode);
    [~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, SY_control] = ...
        fcnRunSimulation_v2(0, harmE_pop, harmA_pop, C, g, th, kp, ki, Amax, b0, pulseAmp, pulseMu, pulseWidth, pulseC, sim_mode);
    
    S_theory_treated_matrix(i, 1:length(SY_treated)) = SY_treated;
    S_theory_control_matrix(i, 1:length(SY_control)) = SY_control;
end

S_theoretical_treated = mean(S_theory_treated_matrix, 1);
S_theoretical_control = mean(S_theory_control_matrix, 1);

% Calculate theoretical hazards from theoretical survival curves
t_exact = t_sim;
dt = t_exact(2) - t_exact(1);

% Theoretical hazard for treated group
log_S_treated = log(max(S_theoretical_treated, 1e-10));
h_theoretical_treated = zeros(size(t_exact));
for i = 2:length(t_exact)-1
    h_theoretical_treated(i) = -(log_S_treated(i+1) - log_S_treated(i-1)) / (2*dt);
    h_theoretical_treated(i) = max(h_theoretical_treated(i), 0);
end
h_theoretical_treated(1) = h_theoretical_treated(2);
h_theoretical_treated(end) = h_theoretical_treated(end-1);

% Theoretical hazard for control group  
log_S_control = log(max(S_theoretical_control, 1e-10));
h_theoretical_control = zeros(size(t_exact));
for i = 2:length(t_exact)-1
    h_theoretical_control(i) = -(log_S_control(i+1) - log_S_control(i-1)) / (2*dt);
    h_theoretical_control(i) = max(h_theoretical_control(i), 0);
end
h_theoretical_control(1) = h_theoretical_control(2);
h_theoretical_control(end) = h_theoretical_control(end-1);

fprintf('Theoretical curves calculated\n');

%% 6. Plot survival curves (Figure 1)
figure(1); clf; 

% Empirical curves (stairs)
stairs(t_rx1, S_rx1, 'LineWidth', 2, 'Color',[0.6350 0.0780 0.1840], 'DisplayName', 'Empirical Treated');
hold on;
stairs(t_rx0, S_rx0, 'LineWidth', 2, 'Color',[0 0.4470 0.7410], 'DisplayName', 'Empirical Control');

% Theoretical curves (smooth lines)
if ~isempty(S_theoretical_treated)
    plot(t_exact, S_theoretical_treated, '--', 'LineWidth', 2, 'Color',[0.6350 0.0780 0.1840], 'DisplayName', 'Theoretical Treated');
end
if ~isempty(S_theoretical_control)
    plot(t_exact, S_theoretical_control, '--', 'LineWidth', 2, 'Color',[0 0.4470 0.7410], 'DisplayName', 'Theoretical Control');
end

xlabel('Time [hours]');
ylabel('Survival Probability');
title('Kaplan-Meier vs Theoretical Survival Curves');
legend('Location', 'southwest');
grid on;
axis([0 max(t_rx1(end), t_rx0(end)) 0 1])
box off
set(gcf,'color','white')

%% 7. Plot hazard functions (Figure 2)
figure(2); clf; 

% Empirical hazards
plot(t_hazard, h_empirical_rx1, 'LineWidth', 2, 'Color',[0.6350 0.0780 0.1840], 'DisplayName', 'Empirical Treated');
hold on;
plot(t_hazard, h_empirical_rx0, 'LineWidth', 2, 'Color',[0 0.4470 0.7410], 'DisplayName', 'Empirical Control');

% Theoretical hazards
if ~isempty(h_theoretical_treated)
    plot(t_exact, h_theoretical_treated, '--', 'LineWidth', 2, 'Color',[0.6350 0.0780 0.1840], 'DisplayName', 'Theoretical Treated');
end
if ~isempty(h_theoretical_control)
    plot(t_exact, h_theoretical_control, '--', 'LineWidth', 2, 'Color',[0 0.4470 0.7410], 'DisplayName', 'Theoretical Control');
end

xlabel('Time [hours]');
ylabel('Hazard Rate');
title('Empirical vs Theoretical Hazard Functions');
legend('Location', 'northwest');
grid on;

% Calculate ylim for hazard plots
max_h = max([max(h_empirical_rx1), max(h_empirical_rx0)]);
if ~isempty(h_theoretical_treated)
    max_h = max(max_h, max(h_theoretical_treated));
end
if ~isempty(h_theoretical_control)
    max_h = max(max_h, max(h_theoretical_control));
end
ylim([0 max_h*1.1]);

set(gcf,'color','white')
box off

%% Helper function to calculate empirical hazards from data
function [h_rx1, h_rx0, t_grid] = calculateEmpiricalHazardsFromData(T, hazard_scale_t1, hazard_scale_t2, b0)
    % Calculate empirical hazards directly from longitudinal data using the model formula
    
    % Get unique time points and patients
    unique_times = sort(unique(T.t));
    t_grid = unique_times(:)';
    
    % Get treated and untreated patients
    treated_patients = unique(T.sid(T.Rx == 1));
    untreated_patients = unique(T.sid(T.Rx == 0));
    
    % Initialize hazard matrices
    h_treated_matrix = zeros(length(treated_patients), length(t_grid));
    h_untreated_matrix = zeros(length(untreated_patients), length(t_grid));
    
    % Calculate hazards for treated patients
    for i = 1:length(treated_patients)
        patient_id = treated_patients(i);
        patient_data = T(T.sid == patient_id, :);
        patient_data = sortrows(patient_data, 't');
        
        % Get patient-specific parameters
        harmE = patient_data.harmE(1);
        harmA = patient_data.harmA(1);
        
        % Calculate hazard at each time point for this patient
        for j = 1:length(t_grid)
            t_current = t_grid(j);
            
            % Find data up to current time
            idx_up_to_t = patient_data.t <= t_current;
            if any(idx_up_to_t)
                L_sum = sum(patient_data.L(idx_up_to_t));
                A_sum = sum(patient_data.A(idx_up_to_t));
                
                % Use exact formula from simulation
                raw_t1 = b0 + harmE*L_sum/18.4 + harmA*A_sum/360;
                t1 = hazard_scale_t1 * raw_t1;
                t2 = hazard_scale_t2 * (t_current/168)^2;
                
                h_treated_matrix(i, j) = t1 + t2;
            end
        end
    end
    
    % Calculate hazards for untreated patients
    for i = 1:length(untreated_patients)
        patient_id = untreated_patients(i);
        patient_data = T(T.sid == patient_id, :);
        patient_data = sortrows(patient_data, 't');
        
        % Get patient-specific parameters
        harmE = patient_data.harmE(1);
        harmA = patient_data.harmA(1);
        
        % Calculate hazard at each time point for this patient
        for j = 1:length(t_grid)
            t_current = t_grid(j);
            
            % Find data up to current time
            idx_up_to_t = patient_data.t <= t_current;
            if any(idx_up_to_t)
                L_sum = sum(patient_data.L(idx_up_to_t));
                A_sum = sum(patient_data.A(idx_up_to_t));
                
                % Use exact formula from simulation
                raw_t1 = b0 + harmE*L_sum/18.4 + harmA*A_sum/360;
                t1 = hazard_scale_t1 * raw_t1;
                t2 = hazard_scale_t2 * (t_current/168)^2;
                
                h_untreated_matrix(i, j) = t1 + t2;
            end
        end
    end
    
    % Pre-compute death times for efficiency
    treated_death_times = inf(length(treated_patients), 1);
    untreated_death_times = inf(length(untreated_patients), 1);
    
    for i = 1:length(treated_patients)
        patient_data = T(T.sid == treated_patients(i), :);
        if any(patient_data.Y > 0)
            death_idx = find(patient_data.Y > 0, 1, 'first');
            treated_death_times(i) = patient_data.t(death_idx);
        end
    end
    
    for i = 1:length(untreated_patients)
        patient_data = T(T.sid == untreated_patients(i), :);
        if any(patient_data.Y > 0)
            death_idx = find(patient_data.Y > 0, 1, 'first');
            untreated_death_times(i) = patient_data.t(death_idx);
        end
    end
    
    % Average across patients in each group, but only include patients still alive at each time
    h_rx1 = zeros(1, length(t_grid));
    h_rx0 = zeros(1, length(t_grid));
    
    for j = 1:length(t_grid)
        t_current = t_grid(j);
        
        % For treated group - only include patients still alive at t_current
        alive_treated_mask = t_current < treated_death_times;
        if any(alive_treated_mask)
            alive_hazards = h_treated_matrix(alive_treated_mask, j);
            alive_hazards = alive_hazards(alive_hazards > 0);
            if ~isempty(alive_hazards)
                h_rx1(j) = mean(alive_hazards);
            end
        end
        
        % For untreated group - only include patients still alive at t_current  
        alive_untreated_mask = t_current < untreated_death_times;
        if any(alive_untreated_mask)
            alive_hazards = h_untreated_matrix(alive_untreated_mask, j);
            alive_hazards = alive_hazards(alive_hazards > 0);
            if ~isempty(alive_hazards)
                h_rx0(j) = mean(alive_hazards);
            end
        end
    end
end