function trajectories = fcn_generateStochasticTrajectories(t, params, N)
%% generateStochasticTrajectories - Generate disease trajectories with time-varying volatility
%
% Inputs:
%   t      - Time vector (hours)
%   params - Parameter vector [gamma, H, alpha, delta, sigma_early, sigma_late, tau]
%            gamma       - Growth rate
%            H           - Peak height
%            alpha       - Mean reversion rate
%            delta       - Decay rate
%            sigma_early - Early volatility
%            sigma_late  - Late volatility
%            tau         - Volatility decay time constant
%   N      - Number of trajectories to generate (default: 100)
%
% Output:
%   trajectories - N x length(t) matrix of disease trajectories

if nargin < 3
    N = 100;  % Default number of trajectories
end

% Extract parameters
gamma = params(1);
H = params(2);
alpha = params(3);
delta = params(4);
sigma_early = params(5);
sigma_late = params(6);
tau = params(7);

% Setup
dt = t(2) - t(1);  % Assume uniform time step
Nt = length(t);
trajectories = zeros(N, Nt);

% Generate each trajectory
for i = 1:N
    X = zeros(1, Nt);
    X(1) = 0;  % Initial condition
    
    for j = 2:Nt
        current_time = t(j);
        
        % Growth dynamics
        if current_time < 30
            % Logistic growth phase
            growth_term = gamma * X(j-1) * (1 - X(j-1)/H);
        else
            % Decay phase
            time_since_peak = current_time - 30;
            decay_factor = exp(-delta * time_since_peak);
            growth_term = -alpha * (X(j-1) - H * decay_factor * 0.2);
        end
        
        % Mean reversion term
        mean_reversion = -alpha * max(0, X(j-1) - H);
        drift = growth_term + mean_reversion;
        
        % Time-varying diffusion
        sigma_t = sigma_early * exp(-current_time/tau) + sigma_late;
        diffusion = sigma_t * sqrt(max(0.001, X(j-1)));  % Minimum variance to avoid numerical issues
        
        % Euler-Maruyama update
        dW = sqrt(dt) * randn;
        X(j) = max(0, X(j-1) + drift * dt + diffusion * dW);
    end
    
    trajectories(i, :) = X;
end

end