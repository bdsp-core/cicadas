% ToDo: 
% - make bias more severe for selection; make it based on L0(t) first few
% hours; generate L0(t) outside of sim function
% - implement other biases

%% Logit-Based Hazard Model Data Generation
%
% DESCRIPTION:
%   Generates synthetic clinical trial data using logit-based discrete-time
%   hazard models for causal survival analysis with PKPD modeling.
%
% TRIAL MODES:
%   RCT=1: No treatment changes or dropout (yes randomized controlled
%   trial)
%   RCT=0: Treatment changes and dropout allowed (not randomized controlled
%   trial)

clear; clc; format compact;
% rng(0) % Set random seed for reproducibility

%% 1. SIMULATION PARAMETERS ==========================================

N = 1000;   % Number of patients to simulate per trial type

% Target L level
th = 0.1;  % LOWER threshold - harder to achieve for sick patients, leading to more treatment

% PI Controller Parameters
ki = 10;    % Integral control gain (aggressive disease suppression)
Amax = 50;  % Maximum pump rate (treatment upper bound)
parmsControl = [ki Amax];

% PD parameters with age/SOFA dependencies
[age, sofa, C, g, parmsPD] = fcnGeneratePatientParameters(N,'TargetCMean', 3, 'TargetGMean', 4, 'CV', 0.1);

% PK parameter - elimination time constant
ke = 0.5;

% Mortality (Y) hazard parameters
% logit_y = a0 + a1*(t(j)/170)^2 + (a2*sofa).*(cumsum_L/24)^2 + (a3*(age/90)).*(cumsum_A/207); 
a0 = -7;     
a1 = .3;
a2 = 20; % harmL
a3 = 5; % harmA

parmsY = [a0 a1 a2 a3]; 

% Censoring (V) hazard parameters
% logit_v = Rx(j)*(b0 + b1*(cumsum_A/207) + b2*(t(j)/170)^2) + (1-Rx(j))*(b3 + b4*cumsum_L/24 + b5*(t(j)/170)^2);                  
b0 = -5;    % Baseline censoring for treated
b1 = 2.0;    % Treatment burden effect
b2 = .1;   % Time effect for treated
b3 = -5;    % Baseline for untreated   
b4 = 2;     % Disease burden effect
b5 = 1.5;   % Time effect for untreated
parmsV = [b0 b1 b2 b3 b4 b5];

% Disease Natural History Parameters and trajectories
% [growth_rate, peak_height, alpha, decay_rate, sigma_early, sigma_late, sigma_transition]
parmsL = [0.25, 1, 0.15, 0.05, 0.15, 0.03, 40];
dt = 2; t = 0:dt:168; 

%% MAIN LOOP FOR BOTH RCT MODES =====================================

L0 = fcnGenerateStochasticTrajectories(t, parmsL, N);
% T = fcnSimulate_N_Patients(N,RCT,treatProb, th, C, g, ke, L0, parmsControl, parmsY, parmsV, age, sofa);
% T =  (N,th, C, g, ke, L0, parmsY, age, sofa)
T = fcnSimulate_DoseChanging(N,th, C, g, ke, L0, parmsY, age, sofa)


% Export to CSV for analysis
filename = sprintf('trialDataDoseChanging.csv');
writetable(T, filename);
save L0data L0

% save "true" values for generating data
save('parmsTrue_DoseChanging', 'parmsControl', 'parmsPD', 'C', 'g', 'ke', 'parmsY', 'parmsV', 'parmsL', 'age', 'sofa');

%% extract data and plot it
for i = 1:N
    ind = find(T.sid==i); 
    A(i,:) = T.A(ind); 
    L(i,:) = T.L(ind); 
end

figure(1); clf; 
idx = 1; 
subplot(211)
plot(t,A(idx,:)','k');
subplot(212); 
plot(t,L(idx,:),'r')

%% estimation of C, g, ke

% run_PKPD_estimation_pipeline
% test_Enhanced_StateSpace
test_FixedKe_Comprehensive