%% a0_GenerateData.m - ICU EEG PKPD Emulated Trial Data Generation
%
% DESCRIPTION:
%   Generates synthetic clinical trial data for causal survival analysis
%   with PKPD (pharmacokinetic-pharmacodynamic) modeling in an ICU setting.
%   Simulates disease progression under PI-controller based treatment.
%
% OUTPUTS:
%   - trial_data.csv: Long-format trial data with patient trajectories
%   - Figure 1: Disease burden trajectories for all patients
%   - Figure 2: Dual swimmer plot showing survival outcomes
%
% WORKFLOW:
%   1. Set model and simulation parameters
%   2. Generate patient-specific harm parameters
%   3. Assign treatment (RCT or observational design)
%   4. Simulate disease progression and treatment response
%   5. Collect longitudinal data and export to CSV
%   6. Create visualization plots
%
% Author: Mitchell McCauley, Brandon Westover
% Created: 2024
% Last Modified: 2025-06-15

clear; clc; format compact;
rng(0) % Set random seed for reproducibility

%% 1. SIMULATION PARAMETERS ==========================================

% Trial Design Parameters
N = 2000;   % Number of patients to simulate
RCT = 0;    % Trial type: 1=Randomized Controlled Trial, 0=Observational Study

% PI Controller Parameters
C = 3;      % Drug potency (higher = more effective treatment)
g = 4;      % Dose-response curve steepness
kp = 0;     % Proportional control gain (pure integral control)
ki = 20;    % Integral control gain (aggressive disease suppression)
Amax = 50;  % Maximum pump rate (treatment upper bound)
th = 0.05;  % Control target threshold (very low for tight control)

% Hazard Model Parameters
b0 = 0.1;   % Baseline mortality risk

% Disease Natural History Parameters (log-normal shaped trajectory)
pulseAmp = 1;       % Pulse amplitude
pulseMu = 20;       % Pulse timing parameter
pulseWidth = 1.8;   % Pulse width
pulseC = 15;        % Pulse center parameter

%% 2. DATA STORAGE INITIALIZATION ====================================

% Pre-allocate arrays for efficient data collection
maxRows = N * 337; % Maximum possible rows (N patients × 337 time points)

% Core data arrays
sid = zeros(maxRows, 1);           % Subject ID
time = zeros(maxRows, 1);          % Time points
Rx_vals = zeros(maxRows, 1);       % Treatment assignment (0/1)
harmE_vals = zeros(maxRows, 1);    % Disease harm parameter
harmA_vals = zeros(maxRows, 1);    % Treatment harm parameter
b0_vals = zeros(maxRows, 1);       % Baseline risk
L_vals = zeros(maxRows, 1);        % Disease burden
A_vals = zeros(maxRows, 1);        % Treatment amount
V_vals = zeros(maxRows, 1);        % PI controller variable
Y_vals = zeros(maxRows, 1);        % Death indicator

% Patient-specific pulse parameters (for exact reproducibility)
pulseAmpR_vals = zeros(maxRows, 1);    % Realized pulse amplitude
pulseMuR_vals = zeros(maxRows, 1);     % Realized pulse timing
pulseWidthR_vals = zeros(maxRows, 1);  % Realized pulse width
pulseCR_vals = zeros(maxRows, 1);      % Realized pulse center

% Initialize counters and storage
ct = 0;              % Row counter for data arrays
LL = nan(N, 337);    % Matrix to store L trajectories for visualization

fprintf('Starting simulation of %d patients...\n', N);

%% 3. PATIENT SIMULATION LOOP ========================================

for i = 1:N
    if mod(i, 100) == 0
        fprintf('Simulating patient %d/%d\n', i, N);
    end
    
    % Generate patient-specific harm parameters
    harmE = max(30 + randn*10, 0);  % Disease harm (mean=30, std=10, min=0)
    harmA = max(10 + randn*5, 0);   % Treatment harm (mean=10, std=5, min=0)

    % Assign treatment based on trial design
    if RCT == 1
        % Randomized Controlled Trial: 50/50 assignment
        Rx = rand < 0.5;
    else
        % Observational Study: treatment selection biased by patient characteristics
        baseProb = 0.5;
        relativeFactor = (harmE - harmA) / (harmE + harmA);
        treatProb = baseProb + 0.2 * relativeFactor;
        Rx = rand < treatProb;
    end

    % Generate patient-specific disease natural history parameters
    pulseAmpR = max(0, pulseAmp + randn*(0.1*pulseAmp));      % ±10% variation
    pulseMuR = max(0, pulseMu + randn*(0.2*pulseMu));         % ±20% variation
    pulseWidthR = max(0, pulseWidth + randn*(0.2*pulseWidth)); % ±20% variation
    pulseCR = max(0, pulseC + randn*(0.2*pulseC));            % ±20% variation
    
    % Run individual patient simulation
    [t, L, A, V, Y, Rx_actual] = fcnRunSimulation_GetDataOnly(Rx, harmE, harmA, C, g, th, kp, ki, ...
                                                              Amax, b0, pulseAmpR, pulseMuR, ...
                                                              pulseWidthR, pulseCR, RCT);
    
    % Store longitudinal data for this patient
    numTimePoints = length(t);
    indices = ct + (1:numTimePoints);
    
    % Populate data arrays
    sid(indices) = i;
    time(indices) = t;
    Rx_vals(indices) = Rx_actual;  % Use actual treatment status (time-varying)
    harmE_vals(indices) = harmE;
    harmA_vals(indices) = harmA;
    b0_vals(indices) = b0;
    L_vals(indices) = L;
    A_vals(indices) = A;
    V_vals(indices) = V;
    Y_vals(indices) = Y;
    pulseAmpR_vals(indices) = pulseAmpR;
    pulseMuR_vals(indices) = pulseMuR;
    pulseWidthR_vals(indices) = pulseWidthR;
    pulseCR_vals(indices) = pulseCR;
    
    % Update counters and trajectory storage
    ct = ct + numTimePoints;
    LL(i, 1:length(L)) = L;  % Store trajectory for visualization
end

fprintf('Simulation complete. Processed %d total observations.\n', ct);

%% 4. DATA EXPORT AND STORAGE ========================================

% Create data table with only essential variables for causal inference
T = table(sid(1:ct), time(1:ct), Rx_vals(1:ct), L_vals(1:ct), A_vals(1:ct), V_vals(1:ct), Y_vals(1:ct), ...
          'VariableNames', {'sid', 't', 'Rx', 'L', 'A', 'V', 'Y'});

% Also create comprehensive table for internal analysis (not exported)
T_full = table(sid(1:ct), time(1:ct), Rx_vals(1:ct), harmE_vals(1:ct), harmA_vals(1:ct), ...
               b0_vals(1:ct), L_vals(1:ct), A_vals(1:ct), V_vals(1:ct), Y_vals(1:ct), ...
               pulseAmpR_vals(1:ct), pulseMuR_vals(1:ct), pulseWidthR_vals(1:ct), pulseCR_vals(1:ct), ...
               'VariableNames', {'sid', 't', 'Rx', 'harmE', 'harmA', 'b0', 'L', 'A', 'V', 'Y', ...
               'pulseAmpR', 'pulseMuR', 'pulseWidthR', 'pulseCR'});

% Export to CSV for analysis
filename = 'trial_data.csv';
writetable(T, filename);
fprintf('Trial data exported to %s (%d rows, %d variables)\n', filename, height(T), width(T));

% Display summary statistics
n_treated_initial = sum(T.Rx == 1 & T.t == 0);  % Initial treatment assignment
n_untreated_initial = sum(T.Rx == 0 & T.t == 0);
n_deaths = sum(T.Y > 0);
treatment_switches = sum(diff(T.Rx(T.sid == T.sid(1))) ~= 0);  % Count switches for first patient as example

fprintf('Summary:\n');
fprintf('  Initial treatment assignment: %d treated (%.1f%%), %d untreated (%.1f%%)\n', ...
    n_treated_initial, 100*n_treated_initial/N, n_untreated_initial, 100*n_untreated_initial/N);
fprintf('  Total deaths: %d (%.1f%% of observations)\n', n_deaths, 100*n_deaths/height(T));
if RCT == 0
    fprintf('  Note: Rx variable reflects time-varying treatment adherence (includes switches/stops)\n');
else
    fprintf('  Note: Rx variable reflects randomized assignment (no switches in RCT mode)\n');
end

%% 5. VISUALIZATION ===================================================

%% Figure 1: Disease Burden Trajectories
fprintf('Creating disease burden trajectory plot...\n');
figure(1); clf;
hold on;

% Plot trajectories by initial treatment group (use full dataset for visualization)
patient_ids = unique(T_full.sid);
for i = 1:length(patient_ids)
    patient_data = T_full(T_full.sid == patient_ids(i), :);
    if patient_data.Rx(1) == 1  % Color by initial treatment assignment
        plot(patient_data.t, patient_data.L, 'r-', 'LineWidth', 0.5, 'Color', [0.8 0.2 0.2 0.3]);
    else
        plot(patient_data.t, patient_data.L, 'b-', 'LineWidth', 0.5, 'Color', [0.2 0.2 0.8 0.3]);
    end
end

% Formatting
xlabel('Time [hours]');
ylabel('Disease Burden (L)');
title(sprintf('Disease Trajectories for All %d Patients', N));
legend({'Treated', 'Untreated'}, 'Location', 'northeast');
grid on;
set(gcf, 'Color', 'white');
box off;

%% Figure 2: Dual Swimmer Plot
fprintf('Creating swimmer plot...\n');
fcnDualSwimmerPlot(T);

fprintf('Data generation and visualization complete.\n');