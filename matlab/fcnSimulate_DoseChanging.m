function T = fcnSimulate_DoseChanging(N,th, C, g, ke, L0, parmsY, age, sofa)

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
%% 3. PATIENT SIMULATION LOOP ========================================
for i = 1:N
    % Assign treatment based on trial design
    Rx = 1;
    % Run individual patient simulation with logit-based hazards
    [t, L, A, V, Y, Rx] = fcnRunSimulationDoseChange(Rx, C(i), g(i), ke, L0(i,:), parmsY);
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

%% 4. DATA EXPORT AND STORAGE ========================================
% Create data table with essential variables for causal inference
T = table(Age, Sofa, sid, tt, Lt, At, Vt, Yt, Rxt,...
'VariableNames', {'age','sofa', 'sid', 't', 'L', 'A', 'V', 'Y', 'Rx'});

%% ================================================================
%% FUNCTIONS
%% ================================================================

function [t, L, A, V, Y, Rx] = fcnRunSimulationDoseChange(Rx, C, g, ke, L0, parmsY);

   
    %% Pre-allocate arrays
    dt = 2; % [h] sample period
    t = (0:dt:168); % 7 days
    Nt = numel(t);
    % Expit function (inverse logit)
    expit = @(x) 1./(1 + exp(-x));
   
    % Initialize variables
     A(1) = Rx*2; % Initial pump value
     L(1) = 0;
     V(1) = 0;
     Y(1) = 0;
     X(1) = 0; % drug concentration
     % Cumulative sums for hazard calculations
     cumsum_L = 0;
     cumsum_A = 0;

    % Initialize dose-changing variables
    current_dose = Rx*0; % Start with initial dose
    time_since_last_change = 0; % Track time since last dose change
 
    %% Simulation loop
    for j = 2:Nt
       % ---- Disease dynamics with previous pump rate -------------------
        Rx(j) = Rx(j-1); % continue treatment status
        if Rx(j) ==1; sX = 1 - 1./((C./X(j-1)).^g + 1); else; sX = 1; end
        L(j) = L0(j)*sX;

        % ---- Dose changing mechanism ----------------------------
        time_since_last_change = time_since_last_change + 1;
        % Check if dose should change (on average every 10 time steps)
        % Using exponential distribution with rate parameter lambda = 1/10
        if rand < (1/5)  % Probability of change at each time step
            current_dose = rand * 5; % New random dose between 0 and 3
            time_since_last_change = 0; % Reset counter
        end
 
        A(j) = current_dose; % Set dose to current level
        X(j) = ke*X(j-1) + A(j);
        
        % ---- Event probabilities (logit scale) --------------------------
        V(j)=0; Y(j) = 0; % default values -- can be changed by events below
    end
 
    t=t(1:length(Y));
    Y=Y'; L=L';
    A=A';
    t=t';
    V=V';
    Rx=Rx';