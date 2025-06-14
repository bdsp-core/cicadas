%% epilepsy_control.m – PI (only) closed loop + hazard model with exact function comparison
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
N = 1000;
pulseAmp = 1; pulseMu = 20; pulseWidth = 1.8; pulseC = 15;
RCT = 0; % Set to 1 to enable simulation mode for mortality/dropout

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

% Arrays to store exact hazard and survival functions
exact_hY_rx1 = []; exact_SY_rx1 = []; t_exact_rx1 = [];
exact_hY_rx0 = []; exact_SY_rx0 = []; t_exact_rx0 = [];
exact_hV_rx1 = []; exact_SV_rx1 = [];
exact_hV_rx0 = []; exact_SV_rx0 = [];
exact_hA_rx1 = []; exact_SA_rx1 = [];
exact_hA_rx0 = []; exact_SA_rx0 = [];

ct = 0;

for i = 1:N
    % Get parameters for patient i
    % relative harm of E vs A
    hE_num = 50 + 8*rand;
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
    
    % Run simulation with mortality support - now using our new function
    % [t, L, A, V, Y, exact_hY, exact_SY, exact_hV, exact_SV, exact_hA, exact_SA, t_exact] = fcnRunSimulationWithExact(Rx, harmE, harmA, C, g, th, kp, ki, Amax, b0, pulseAmp, pulseMu, pulseWidth, pulseC, RCT);
    
    [t_exact, exact_hY, exact_SY, exact_hV, exact_SV, exact_hA, exact_SA] = fcnCalculateExactFunctions(Rx, harmE, harmA, C, g, th, kp, ki, Amax, b0, pulseAmp, pulseMu, pulseWidth, pulseC);
    [t, L, A, V, Y, hY, SY, hV, SV, hA, SA] = fcnRunSimulation_alone(Rx, harmE, harmA, C, g, th, kp, ki, Amax, b0, pulseAmp, pulseMu, pulseWidth, pulseC, RCT);

    % Store exact hazard and survival functions
   % Store exact hazard and survival functions
    if Rx == 1  
        % First patient of each type, save the exact functions
        if isempty(exact_hY_rx1)  
            exact_hY_rx1 = exact_hY;
            exact_SY_rx1 = exact_SY;
            t_exact_rx1 = t_exact;
            exact_hV_rx1 = exact_hV;
            exact_SV_rx1 = exact_SV;
            exact_hA_rx1 = exact_hA;
            exact_SA_rx1 = exact_SA;
        end
    else
        % First patient of each type, save the exact functions
        if isempty(exact_hY_rx0)  
            exact_hY_rx0 = exact_hY;
            exact_SY_rx0 = exact_SY;
            t_exact_rx0 = t_exact;
            exact_hV_rx0 = exact_hV;
            exact_SV_rx0 = exact_SV;
            exact_hA_rx0 = exact_hA;
            exact_SA_rx0 = exact_SA;
        end
    end
    
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
% create swimmer plot
fcnDualSwimmerPlot(T);

% Calculate empirical survival curves
[h_rx1, h_rx0, S_rx1, S_rx0, t_rx1, t_rx0] = fcnEmpiricalSurvivalCurves(T);

%% Plot survival curves with exact comparisons
figure(4); clf; 

% Plot mortality hazard functions
subplot(221); 
stairs(t_rx1, h_rx1,'LineWidth', 2, 'Color',[0.6350 0.0780 0.1840]);
hold on;
stairs(t_rx0, h_rx0, 'LineWidth', 2, 'Color',[0 0.4470 0.7410]);
% Add exact hazard functions
plot(t_exact_rx1, exact_hY_rx1, '--', 'LineWidth', 1.5, 'Color', [0.6350 0.0780 0.1840]);
plot(t_exact_rx0, exact_hY_rx0, '--', 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);
xlabel('Time [hours]');
ylabel('Hazard Rate');
title('Mortality Hazard Functions');
grid on;
box off
axis([0 max(T.t) 0 .02]);

% Plot mortality survival curves
subplot(222); 
stairs(t_rx1, S_rx1, 'LineWidth', 2, 'Color',[0.6350 0.0780 0.1840]);
hold on;
stairs(t_rx0, S_rx0, 'LineWidth', 2, 'Color',[0 0.4470 0.7410]);
% Add exact survival curves
plot(t_exact_rx1, exact_SY_rx1, '--', 'LineWidth', 1.5, 'Color', [0.6350 0.0780 0.1840]);
plot(t_exact_rx0, exact_SY_rx0, '--', 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);
xlabel('Time [hours]');
ylabel('Survival Probability');
title('Mortality Survival Curves');
grid on;
box off
axis([0 max(T.t) 0 1]);

% Plot dropout hazard functions
subplot(223); 
stairs(t_rx1, zeros(size(t_rx1)), 'LineWidth', 2, 'Color',[0.6350 0.0780 0.1840]); % Placeholder for empirical
hold on;
stairs(t_rx0, zeros(size(t_rx0)), 'LineWidth', 2, 'Color',[0 0.4470 0.7410]); % Placeholder for empirical
% Add exact hazard functions for dropout
plot(t_exact_rx1, exact_hV_rx1, '--', 'LineWidth', 1.5, 'Color', [0.6350 0.0780 0.1840]);
plot(t_exact_rx0, exact_hV_rx0, '--', 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);
xlabel('Time [hours]');
ylabel('Hazard Rate');
title('Dropout Hazard Functions');
grid on;
box off
axis([0 max(T.t) 0 .02]);


% Plot dropout survival curves
subplot(224); 
stairs(t_rx1, ones(size(t_rx1)), 'LineWidth', 2, 'Color',[0.6350 0.0780 0.1840]); % Placeholder for empirical
hold on;
stairs(t_rx0, ones(size(t_rx0)), 'LineWidth', 2, 'Color',[0 0.4470 0.7410]); % Placeholder for empirical
% Add exact survival curves for dropout
plot(t_exact_rx1, exact_SV_rx1, '--', 'LineWidth', 1.5, 'Color', [0.6350 0.0780 0.1840]);
plot(t_exact_rx0, exact_SV_rx0, '--', 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);
xlabel('Time [hours]');
ylabel('Survival Probability');
title('Dropout Survival Curves');
grid on;
box off
axis([0 max(T.t) 0 1]);

set(gcf,'color','white')

%% Add a new figure for treatment switch hazard and survival
figure(5); clf;

% Plot switch hazard functions
subplot(211); 
% No empirical switch hazard available from original code
plot(t_exact_rx1, exact_hA_rx1, '--', 'LineWidth', 1.5, 'Color', [0.6350 0.0780 0.1840]);
hold on;
plot(t_exact_rx0, exact_hA_rx0, '--', 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);
xlabel('Time [hours]');
ylabel('Hazard Rate');
title('Treatment Switch Hazard Functions');
grid on;
box off

% Plot switch survival curves
subplot(212); 
% No empirical switch survival available
plot(t_exact_rx1, exact_SA_rx1, '--', 'LineWidth', 1.5, 'Color', [0.6350 0.0780 0.1840]);
hold on;
plot(t_exact_rx0, exact_SA_rx0, '--', 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);
xlabel('Time [hours]');
ylabel('Survival Probability');
title('Treatment Switch Survival Curves');
grid on;
axis([0 max(t) 0 1]);
box off

set(gcf,'color','white')

%% save data
writetable(T,'TrialData.csv')