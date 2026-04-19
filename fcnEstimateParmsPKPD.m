function [ke_est, C_est_all, g_est_all, theta_est] = fcnEstimateParmsPKPD(parmsL_est, LL, AA, patient_age, patient_sofa, t)
%% ================================================================================
%% Estimate PD parameters given fixed L parameters
%% ================================================================================

dt = t(2) - t(1);
[n_patients, Nt] = size(LL);

% Identify treated patients
is_treated = max(AA, [], 2) > 0;
n_treated = sum(is_treated);

if n_treated == 0
    error('No treated patients - cannot estimate PD parameters');
end

% Normalize age and SOFA for all patients (for consistency)
age_norm_all = (patient_age - mean(patient_age)) / std(patient_age);
sofa_norm_all = (patient_sofa - mean(patient_sofa)) / std(patient_sofa);

% Extract treated patient data
age_treated = patient_age(is_treated);
sofa_treated = patient_sofa(is_treated);
age_norm_treated = (age_treated - mean(patient_age)) / std(patient_age);
sofa_norm_treated = (sofa_treated - mean(patient_sofa)) / std(patient_sofa);

% Initial guess for regression coefficients
% [b0_C, b1_C, b2_C, b0_g, b1_g, b2_g, ke]
theta_PD_init = [3; 0.1; 0.15; 4; 0.08; 0.12; 0.75];

% Define objective function for PKPD regression coefficients only
nll_PD = @(theta_PD) compute_PD_regression_nll(theta_PD, parmsL_est, ...
    LL(is_treated, :), AA(is_treated, :), ...
    age_norm_treated, sofa_norm_treated, t, dt);

% Bounds for regression coefficients
lb_PD = [0.5,  0,  0, 1,  0,  0, 0.1]';  % All coefficients must be positive
ub_PD = [10,  0.5, 0.5, 8, 0.5, 0.5, 0.9]';

% Optimization
opts = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 200, ...
                   'OptimalityTolerance', 1e-6, 'StepTolerance', 1e-6);
theta_PD_est = fmincon(nll_PD, theta_PD_init, [], [], [], [], lb_PD, ub_PD, [], opts);

% Extract estimated parameters
b0_C_est = theta_PD_est(1);
b1_C_est = theta_PD_est(2);
b2_C_est = theta_PD_est(3);
b0_g_est = theta_PD_est(4);
b1_g_est = theta_PD_est(5);
b2_g_est = theta_PD_est(6);
ke_est = theta_PD_est(7);

% Calculate estimated C and g values for all patients
C_est_all = b0_C_est + b1_C_est * age_norm_all + b2_C_est * sofa_norm_all;
g_est_all = b0_g_est + b1_g_est * age_norm_all + b2_g_est * sofa_norm_all;

% Combine all parameters
theta_est = [parmsL_est(:); b0_C_est; b1_C_est; b2_C_est; b0_g_est; b1_g_est; b2_g_est; ke_est];

end

%%===============================================
%% HELPER FUNCTION
%%===============================================

function nll = compute_PD_regression_nll(theta_PD, L_params, L_treated, A_treated, age_norm, sofa_norm, t, dt)
    % Extract regression coefficients
    b0_C = theta_PD(1);
    b1_C = theta_PD(2);
    b2_C = theta_PD(3);
    b0_g = theta_PD(4);
    b1_g = theta_PD(5);
    b2_g = theta_PD(6);
    ke = theta_PD(7); 
    
    [N_treated, Nt] = size(L_treated);
    x = zeros(1,Nt); 
    nll = 0;
    
    for i = 1:N_treated
        L = L_treated(i, :);
        A = A_treated(i, :);
        
        % Calculate patient-specific C and g values
        C_i = b0_C + b1_C * age_norm(i) + b2_C * sofa_norm(i);
        g_i = b0_g + b1_g * age_norm(i) + b2_g * sofa_norm(i);
        
        % Ensure positive values
        C_i = max(0.1, C_i);
        g_i = max(0.1, g_i);
        
        % Generate natural progression with fixed L parameters for this patient
        L0 = fcn_generateTrajectory(L_params, 1, t(end), dt);
        
        for j = 2:Nt
            if ~isnan(L(j)) && ~isnan(A(j-1)) && L(j) >= 0
                
                % Expected value with treatment based on patient-specific PD
                if x(j-1) > 0
                    sX = 1 - 1./((C_i./x(j-1)).^g_i + 1);
                    L_expected = L0(j) * sX;
                else
                    L_expected = L0(j);
                end
                
                % dynamics
                x(j) = ke*x(j-1) + A(j); 

                % Variance - increase variance for treated patients due to control dynamics
                if A(j-1) > 0
                    var_L = 0.01 + 0.1 * L_expected + 0.05 * L_expected^2;
                else
                    var_L = 0.001 + 0.05 * L_expected + 0.01 * L_expected^2;
                end
                
                % Log-likelihood
                if var_L > 0 && L_expected >= 0 && isfinite(L(j))
                    ll_contrib = -0.5 * log(2*pi*var_L) - 0.5 * (L(j) - L_expected)^2 / var_L;
                    if isfinite(ll_contrib)
                        nll = nll - ll_contrib;
                    end
                end
            end
        end
    end
    
    % Light regularization
    nll = nll + 0.01 * sum(theta_PD.^2);
    
    % Handle numerical issues
    if isnan(nll) || isinf(nll)
        nll = 1e10;
    end
end