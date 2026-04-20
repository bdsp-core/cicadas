function T = fcnSimulate_StructuredDoses(N, th, C, g, ke, L0, parmsY, age, sofa)
% Generate data with STRUCTURED dose changes that provide better identifiability
% Instead of random changes, use:
% 1. Dose escalation/de-escalation protocols
% 2. On/off periods
% 3. Step changes at fixed intervals

% Initialize storage arrays
tt = [];
Lt = [];
At = [];
Vt = [];
Yt = [];
Rxt = [];
sid = [];
Age = [];
Sofa = [];

%% PATIENT SIMULATION LOOP
for i = 1:N
    % Assign treatment (all treated)
    Rx = 1;
    
    % Choose a dose protocol for this patient
    protocol = mod(i-1, 4) + 1;  % Cycle through 4 protocols
    
    % Run simulation with structured doses
    [t, L, A, V, Y, Rx] = fcnRunSimulation_Structured(Rx, C(i), g(i), ke, L0(i,:), parmsY, protocol);
    
    sid = [sid; i*ones(length(t),1)];
    tt = [tt; t];
    Lt = [Lt; L];
    At = [At; A];
    Vt = [Vt; V];
    Yt = [Yt; Y];
    Rxt = [Rxt; Rx];
    Age = [Age; age(i)*ones(size(t))];
    Sofa = [Sofa; sofa(i)*ones(size(t))];
end

%% DATA EXPORT
T = table(Age, Sofa, sid, tt, Lt, At, Vt, Yt, Rxt,...
    'VariableNames', {'age','sofa', 'sid', 't', 'L', 'A', 'V', 'Y', 'Rx'});

%% SIMULATION FUNCTION
function [t, L, A, V, Y, Rx] = fcnRunSimulation_Structured(Rx, C, g, ke, L0, parmsY, protocol)
    
    %% Setup
    dt = 2; % sample period
    t = (0:dt:168); % 7 days
    Nt = numel(t);
    expit = @(x) 1./(1 + exp(-x));
    
    % Initialize
    A = zeros(1, Nt);
    L = zeros(1, Nt);
    V = zeros(1, Nt);
    Y = zeros(1, Nt);
    X = zeros(1, Nt);
    Rx = ones(1, Nt);
    
    L(1) = L0(1);
    
    %% Define structured dose protocols
    switch protocol
        case 1  % Escalation protocol
            dose_schedule = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5];  % Escalate every ~17 hours
            
        case 2  % De-escalation protocol
            dose_schedule = [5, 5, 4, 4, 3, 3, 2, 2, 1, 1];
            
        case 3  % On-off protocol
            dose_schedule = [3, 3, 0, 0, 3, 3, 0, 0, 3, 3];
            
        case 4  % Step changes
            dose_schedule = [2, 2, 4, 4, 2, 2, 4, 4, 2, 2];
    end
    
    % Extend schedule to cover all time points
    schedule_length = length(dose_schedule);
    dose_idx = @(j) dose_schedule(min(ceil(j/8.5), schedule_length));
    
    %% Simulation loop
    for j = 2:Nt
        % Structured dose assignment
        A(j) = dose_idx(j);
        
        % Drug concentration
        X(j) = ke*X(j-1) + A(j);
        
        % Disease dynamics
        if X(j-1) > 0
            sX = 1 - 1./((C./X(j-1)).^g + 1);
        else
            sX = 1;
        end
        L(j) = L0(j)*sX;
        
        % No mortality or censoring in this version (simplify for identifiability)
        V(j) = 0;
        Y(j) = 0;
    end
    
    % Convert to column vectors
    t = t';
    Y = Y';
    L = L';
    A = A';
    V = V';
    Rx = Rx';
end

end