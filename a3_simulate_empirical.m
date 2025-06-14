%% epilepsy_control.m – PI (only) closed loop + hazard model

clear; clc; format compact;
rng(0)

%% 2. Model & PI‑controller parameters -----------------------------
C = 3;      % Increased for higher drug potency
g = 4;      % Steeper dose-response curve
kp = 0;     % Pure integral control
ki = 20;    % Very aggressive control for tight disease suppression
Amax = 50;  % pump upper bound
th = 0.05;  % Very low target for tight control

%% 3. Hazard‑model constants - components of L0 (covariates)
b0 = 0.1;   % Baseline risk

%% Log‑normebal‑shaped trajectory for L
N = 2000;
pulseAmp = 1; pulseMu = 20; pulseWidth = 1.8; pulseC = 15;
RCT = 1; % Set to 1 to enable simulation mode for mortality/dropout

% Pre-allocate data for the table
maxRows = N * 337; % Maximum possible rows (N patients * 337 time points)
sid = zeros(maxRows, 1);
time = zeros(maxRows, 1);
Rx_vals = zeros(maxRows, 1);
harmE_vals = zeros(maxRows, 1);
harmA_vals = zeros(maxRows, 1);
b0_vals = zeros(maxRows, 1);
L_vals = zeros(maxRows, 1);
A_vals = zeros(maxRows, 1);
V_vals = zeros(maxRows, 1);
Y_vals = zeros(maxRows, 1);

ct = 0;

for i = 1:N
    % Get parameters for patient i
    % relative harm of E vs A
    hE_num = rand;
    hA_num = rand;
    % harmE = 5*hE_num/(hE_num + hA_num);
    % harmA = 5*hA_num/(hE_num + hA_num);

    harmE = 50*hE_num;  % Massively increased - disease is MUCH more harmful
    harmA = 1*hA_num;   % Minimal drug harm

    % treatment assignment
    if RCT == 1
        % Randomized trial: 50/50 assignment regardless of patient characteristics
        Rx = rand < 0.5;
    else
        % Observational study: biased selection based on patient characteristics
        baseProb = 0.5;
        relativeFactor = (harmE - harmA) / (harmE + harmA);
        treatProb = baseProb + 0.2 * relativeFactor;  % Reduced from 0.5 to 0.2
        Rx = rand < treatProb;
    end

    % natural history
    pulseAmpR = max(0,pulseAmp + randn*(0.1*pulseAmp));
    pulseMuR = max(0,pulseMu + randn*(0.2*pulseMu));
    pulseWidthR = max(0,pulseWidth + randn*(0.2*pulseWidth));
    pulseCR = max(0,pulseC + randn*(0.2*pulseC));
    
    % Run simulation with mortality support
    [t, L, A, V, Y] = fcnRunSimulation_GetDataOnly(Rx, harmE, harmA, C, g, th, kp, ki, Amax, b0, pulseAmp, pulseMu, pulseWidth, pulseC,RCT);
    
    % Collect data for this patient
    numTimePoints = length(t);
    indices = ct + (1:numTimePoints);
    
    % Add data to arrays
    sid(indices) = i;
    time(indices) = t;
    Rx_vals(indices) = Rx;
    harmE_vals(indices) = harmE;
    harmA_vals(indices) = harmA;
    b0_vals(indices) = b0;
    L_vals(indices) = L;
    A_vals(indices) = A;
    V_vals(indices) = V;
    Y_vals(indices) = Y;
    
    ct = ct + numTimePoints;
end

% Create table with only the used rows
T = table(sid(1:ct), time(1:ct), Rx_vals(1:ct), harmE_vals(1:ct), harmA_vals(1:ct), ...
          b0_vals(1:ct), L_vals(1:ct), A_vals(1:ct), V_vals(1:ct), Y_vals(1:ct), ...
          'VariableNames', {'sid', 't', 'Rx', 'harmE', 'harmA', 'b0', 'L', 'A', 'V', 'Y'});

% 6. Save data to CSV
filename = 'trial_data.csv';
writetable(T, filename);
fprintf('Trial data saved to %s\n', filename);

% 7. Plots ---------------------------------------------------------

%======================
% create swimmer plot
fcnDualSwimmerPlot(T);

