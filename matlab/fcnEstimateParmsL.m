function [parmsL_est, LL, AA, patient_age, patient_sofa, t] = fcnEstimateParmsL(T0)
%% ================================================================================
%% Estimate disease model L parameters from untreated patients
%% ================================================================================

dt = 2; 
t = 0:dt:168;  % Create regular time grid
Nt = length(t); % 85 

% Get unique patient data (one row per patient at time 0)
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

% Identify untreated patients
is_treated = max(AA, [], 2) > 0;
n_untreated = sum(~is_treated);

if n_untreated < 10
    error('Need at least 10 untreated patients for L parameter estimation');
end

% Estimate L parameters from untreated patients only
L_untreated = LL(~is_treated, :);
[parmsL_est, diagnostics_L] = fcn_estimate_parmsL(L_untreated, t);

parmsL_est = parmsL_est';
end