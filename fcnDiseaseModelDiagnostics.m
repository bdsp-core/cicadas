function fcnDiseaseModelDiagnostics(T1_est,true_params, C_est_all, g_est_all, theta_est, L_trajectories, A_trajectories, age, sofa, t)
% FCNDISEASEMODELDIAGNOSTICS - Diagnostic plots and metrics for disease model parameter estimation
%
% Inputs:
%   true_params     - True parameter values (14x1 vector)
%   theta_est       - Estimated parameters (14x1 vector)
%   L_trajectories  - Disease burden trajectories (N x Nt matrix)
%   A_trajectories  - Treatment pump rate trajectories (N x Nt matrix)
%   age             - Patient ages (Nx1 vector)
%   sofa            - Patient SOFA scores (Nx1 vector)
%   t               - Time vector (1xNt vector)
%   is_treated      - Logical vector indicating treated patients (Nx1)
%
% Parameter order: [γ, H, α, δ, σ_early, σ_late, τ, b0_C, b1_C, b2_C, b0_g, b1_g, b2_g, ke]

dt = 2; 
t = 0:dt:168;  % Create regular time grid
Nt = length(t); % 85 

is_treated = max(A_trajectories, [], 2) > 0;

% Compute counts
n_treated = sum(is_treated);
n_untreated = sum(~is_treated);

% Parameter names for display
param_names = {'Growth rate (γ)', 'Peak height (H)', 'Mean reversion (α)', ...
               'Decay rate (δ)', 'Early volatility (σ_early)', ...
               'Late volatility (σ_late)', 'Volatility decay (τ)', ...
               'C intercept (b0_C)', 'C age coef (b1_C)', 'C SOFA coef (b2_C)', ...
               'g intercept (b0_g)', 'g age coef (b1_g)', 'g SOFA coef (b2_g)',...
               'PK value (ke)'};

% Display estimated parameters
fprintf('\n\nFinal estimated parameters:\n');
for i = 1:14
    fprintf('  %s: %.4f\n', param_names{i}, theta_est(i));
end

% Calculate and display errors if true parameters provided
if ~isempty(true_params) && length(true_params) >= 14
    param_errors = abs(theta_est(:) - true_params(:)) ./ abs(true_params(:)) * 100;
    fprintf('\nParameter estimation errors:\n');
    for i = 1:14
        fprintf('  %s: %.1f%%\n', param_names{i}, param_errors(i));
    end
    fprintf('\nMean absolute percentage error: %.1f%%\n', mean(param_errors));
end

% Compute goodness of fit metrics
compute_goodness_of_fit_metrics(L_trajectories, A_trajectories, t, theta_est, is_treated);

%% Plot estimated vs true survival curves
% Load reference KM curves - ground truth
T1 = readtable('trialData1.csv');
[s0,s1,t0,t1] = fcnPlotKM(T1);

figure(3); clf 
plot(t0,s0,'b--',t1,s1,'r--','LineWidth',2); 
hold on

[s0_est,s1_est,t0_est,t1_est] = fcnPlotKM(T1_est);
plot(t0_est,s0_est,'b','LineWidth',2,'DisplayName','Untreated (Est)')
plot(t1_est,s1_est,'r','LineWidth',2,'DisplayName','Treated (Est)')

set(gcf,'color','w')
xlabel('Hours')
ylabel('% Alive')
title('Survival Curves: True vs Estimated')
legend({'Untreated (True)', 'Treated (True)', 'Untreated (Est)', 'Treated (Est)'}, ...
    'Location', 'southwest')
box off
grid on

%% Additional analysis
% Calculate true C and g values using the loaded true parameters
load parmsTrue % gets true values: 'parmsControl', 'parmsPD', 'C', 'g', 'parmsY', 'parmsV', 'L_parms'
C_true_all = C;
g_true_all = g;

T0 = readtable('trialData0.csv');

unique_patients = unique(T0.sid);
n_patients = length(unique_patients);

% Extract patient-level data
patient_age = zeros(n_patients, 1);
patient_sofa = zeros(n_patients, 1);
patient_treated = zeros(n_patients, 1);

for i = 1:n_patients
    idx = find(T0.sid == unique_patients(i), 1, 'first');
    patient_age(i) = T0.age(idx);
    patient_sofa(i) = T0.sofa(idx);
    patient_treated(i) = T0.Rx(idx);
end

% Create patient-level trajectory matrices
LL = nan(n_patients, Nt);
AA = nan(n_patients, Nt);

for i = 1:n_patients
    patient_data = T0(T0.sid == unique_patients(i), :);
    n_obs = height(patient_data);
    LL(i, 1:n_obs) = patient_data.L';
    AA(i, 1:n_obs) = patient_data.A';
end

% Compare with true values
C_patient_errors = abs(C_est_all - C_true_all) ./ C_true_all * 100;
g_patient_errors = abs(g_est_all - g_true_all) ./ g_true_all * 100;

fprintf('\n*** PATIENT-LEVEL PARAMETER ACCURACY ***\n');
fprintf('C values:\n');
fprintf('  Mean absolute error: %.3f (%.1f%%)\n', mean(abs(C_est_all - C_true_all)), mean(C_patient_errors));
fprintf('  Median absolute error: %.3f (%.1f%%)\n', median(abs(C_est_all - C_true_all)), median(C_patient_errors));

fprintf('\ng values:\n');
fprintf('  Mean absolute error: %.3f (%.1f%%)\n', mean(abs(g_est_all - g_true_all)), mean(g_patient_errors));
fprintf('  Median absolute error: %.3f (%.1f%%)\n', median(abs(g_est_all - g_true_all)), median(g_patient_errors));

fprintf('\nFor TREATED patients only:\n');
C_treated_errors = C_patient_errors(patient_treated == 1);
g_treated_errors = g_patient_errors(patient_treated == 1);
fprintf('  C mean error: %.1f%%, median: %.1f%%\n', mean(C_treated_errors), median(C_treated_errors));
fprintf('  g mean error: %.1f%%, median: %.1f%%\n', mean(g_treated_errors), median(g_treated_errors));

%% Summary statistics
fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('SUMMARY STATISTICS\n');
fprintf('%s\n', repmat('=', 1, 70));

% Calculate survival rates at key time points
time_points = [24, 48, 72, 96, 120, 144, 168];
fprintf('\nSurvival rates at key time points:\n');
fprintf('Time (h) | True Untreated | True Treated | Est Untreated | Est Treated\n');
fprintf('%s\n', repmat('-', 1, 70));

for tp = time_points
    % Find closest time points in the KM curves
    [~, idx0_true] = min(abs(t0 - tp));
    [~, idx1_true] = min(abs(t1 - tp));
    [~, idx0_est] = min(abs(t0_est - tp));
    [~, idx1_est] = min(abs(t1_est - tp));
    
    fprintf('%8d | %14.1f%% | %12.1f%% | %13.1f%% | %11.1f%%\n', ...
        tp, s0(idx0_true)*100, s1(idx1_true)*100, s0_est(idx0_est)*100, s1_est(idx1_est)*100);
end

% Calculate treatment effect
treatment_effect_true = mean(s1) - mean(s0);
treatment_effect_est = mean(s1_est) - mean(s0_est);

fprintf('\nTreatment effect (mean survival difference):\n');
fprintf('  True: %.3f\n', treatment_effect_true);
fprintf('  Estimated: %.3f\n', treatment_effect_est);
fprintf('  Error: %.3f (%.1f%%)\n', ...
    abs(treatment_effect_est - treatment_effect_true), ...
    abs(treatment_effect_est - treatment_effect_true)/abs(treatment_effect_true)*100);

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('Analysis complete!\n');

end

%% Compute goodness of fit metrics without plots
function compute_goodness_of_fit_metrics(L_trajectories, A_trajectories, t, theta_est, is_treated)
    
    % Extract parameters
    L_params_est = theta_est(1:7);
    dt = t(2) - t(1);
    
    % Generate simulated trajectories for comparison
    N_sim = 100;
    L_sim = fcn_generateTrajectory(L_params_est, N_sim, t(end), dt);
    
    % Separate observed trajectories
    L_obs_untreated = L_trajectories(~is_treated, :);
    
    % Get peak values
    obs_peaks = max(L_obs_untreated, [], 2);
    sim_peaks = max(L_sim, [], 2);
    
    % Get onset times
    obs_onset = computeOnsetTimes(L_obs_untreated, t);
    sim_onset = computeOnsetTimes(L_sim, t);
    
    % Calculate means
    obs_mean = mean(L_obs_untreated, 1);
    sim_mean = mean(L_sim, 1);
    
    % Calculate variances
    obs_var = var(L_obs_untreated, 0, 1);
    sim_var = var(L_sim, 0, 1);
    
    % Compute and display goodness of fit metrics
    fprintf('\n\nGoodness of Fit Metrics:\n');
    
    % Kolmogorov-Smirnov tests
    [~, p_peaks] = kstest2(obs_peaks, sim_peaks);
    [~, p_onset] = kstest2(obs_onset, sim_onset);
    
    % Mean squared error
    mse_mean = mean((obs_mean - sim_mean).^2, 'omitnan');
    
    % Relative error in variance
    var_error = mean(abs(obs_var - sim_var) ./ (obs_var + 1e-6), 'omitnan');
    
    fprintf('Peak values K-S test p-value: %.3f\n', p_peaks);
    fprintf('Onset times K-S test p-value: %.3f\n', p_onset);
    fprintf('MSE of mean trajectories: %.4f\n', mse_mean);
    fprintf('Mean relative error in variance: %.3f\n', var_error);
end

%% Helper function: Compute onset times
function onset_times = computeOnsetTimes(trajectories, t)
    [N, ~] = size(trajectories);
    onset_times = zeros(N, 1);
    
    for i = 1:N
        peak_val = max(trajectories(i, :));
        threshold = 0.1 * peak_val;
        idx = find(trajectories(i, :) >= threshold, 1, 'first');
        if ~isempty(idx)
            onset_times(i) = t(idx);
        else
            onset_times(i) = NaN;
        end
    end
    
    % Remove NaN values
    onset_times = onset_times(~isnan(onset_times));
end