%% epilepsy_control.m – PI (only) closed loop + hazard model
clear; clc; format compact;
%% 2. Model & PI‑controller parameters -----------------------------
C = 2;
g = 3;
kp = 3; % ↓ lower kp to tame overshoot
ki = 10; % integral gain (steady‑state accuracy)
Amax = 50; % pump upper bound
th = 0.1; % target level of L
%% 3. Hazard‑model constants - components of L0 (covariates)
b0 = 0.1;
%% Log‑normal‑shaped trajectory for L
N = 100;
pulseAmp = 1; pulseMu = 20; pulseWidth = 1.8; pulseC = 15;
sim_mode = 0; % Set to 1 to enable simulation mode for mortality/dropout

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
    hE_num = 10 + 8*rand;
    hA_num = 10 + 8*rand;
    harmE = 5*hE_num/(hE_num + hA_num);
    harmA = 5*hA_num/(hE_num + hA_num);

    % treatment (biased selection)
    baseProb = 0.5;
    relativeFactor = (harmE - harmA) / (harmE + harmA);
    treatProb = baseProb + 0.5 * relativeFactor;
    Rx = rand<treatProb;

    % natural history
    pulseAmpR = max(0,pulseAmp + randn*(0.1*pulseAmp));
    pulseMuR = max(0,pulseMu + randn*(0.2*pulseMu));
    pulseWidthR = max(0,pulseWidth + randn*(0.2*pulseWidth));
    pulseCR = max(0,pulseC + randn*(0.2*pulseC));
    
    % Run simulation with mortality support
    [t, z, t1, t2, p, L, A, V, Y, hY, SY, hA, SA, hV, SV] = fcnRunSimulation_v2(Rx, harmE, harmA, C, g, th, kp, ki, Amax, b0, pulseAmpR, pulseMuR, pulseWidthR, pulseCR, sim_mode);
    
    
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

% 6. Plots ---------------------------------------------------------
figure(1); clf
plot(t, L, 'k','LineWidth',1); hold on
plot(t,th*ones(size(t)),'--','linewidth',1.4);
xlabel('time [h]'); ylabel('activity L');
title('Epileptiform activity'); grid on
xlim([0 168])
box off

%======================

fcnDualSwimmerPlot(T)
