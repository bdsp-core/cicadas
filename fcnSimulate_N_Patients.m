function T = fcnSimulate_N_Patients(N,RCT,treatProb,th, C, g, ke, L0, parmsControl, parmsY, parmsV, age, sofa)

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
    Rx = rand < treatProb(i);
     
    % Run individual patient simulation with logit-based hazards   
    [t, L, A, V, Y, Rx] = fcnRunSimulation(RCT, Rx, th, C(i), g(i), age(i), sofa(i), ke, L0(i,:), parmsControl, parmsY, parmsV);              

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
T =              table(Age,  Sofa,   sid,  tt,  Lt,  At,  Vt,  Yt,  Rxt,...
    'VariableNames', {'age','sofa', 'sid', 't', 'L', 'A', 'V', 'Y', 'Rx'});


%% ================================================================
%% FUNCTIONS  
%% ================================================================

function  [t, L, A, V, Y, Rx] = fcnRunSimulation(RCT, Rx, th, C, g, age, sofa, ke, L0, parmsControl, parmsY, parmsV);              
                      
    %% unpack parameters
    ki = parmsControl(1); Amax = parmsControl(2); 
    a0 = parmsY(1); a1 = parmsY(2); a2 = parmsY(3); a3 = parmsY(4); 
    if length(parmsY) >= 5
        a4 = parmsY(5);  % Interaction term if provided
    else
        a4 = 0;  % No interaction if not provided
    end
    b0 = parmsV(1); b1 = parmsV(2); b2 = parmsV(3); b3 = parmsV(4); b4 = parmsV(5); b5 = parmsV(6); 


    %% Pre-allocate arrays
    dt = 2; % [h] sample period
    t = (0:dt:168); % 7 days
    Nt = numel(t);

    % Expit function (inverse logit)
    expit = @(x) 1./(1 + exp(-x));

    % PI controller state
    eInt = 0;

    % Initialize variables
    A(1) = Rx*2;  % Initial pump value
    L(1) = 0; 
    V(1) = 0; 
    Y(1) = 0; 
    X(1) = 0; % drug concentration
    % Cumulative sums for hazard calculations
    cumsum_L = 0;
    cumsum_A = 0;

    %% Simulation loop
    for j = 2:Nt
        
        % ---- Disease dynamics with previous pump rate -------------------
        Rx(j) = Rx(j-1); % continue treatment status
        if Rx(j) ==1; sX = 1 - 1./((C./X(j-1)).^g + 1); else; sX = 1; end
        L(j) = L0(j)*sX;
        
        % ---- PI control (if on treatment) ----------------------------
        if  Rx(j) == 1 & j>5
            e = L(j) - th;
            eInt = eInt + e*dt;
            Aunsat = ki*eInt;
            
            % Saturation & anti-wind-up
            A(j) = min(max(Aunsat,0), Amax);
            if A(j) ~= Aunsat
                eInt = eInt - (Aunsat - A(j))/ki;
            end
        else
            A(j) = 0;
        end
        X(j) = ke*X(j-1) + A(j);

        % Update cumulative sums
        cumsum_L = cumsum_L + L(j); 
        cumsum_A = cumsum_A + A(j);

         
        % ---- Event probabilities (logit scale) --------------------------
        V(j)=0; Y(j) = 0; % default values -- can be changed by events below

        % Only happens in observational mode
        if RCT == 0
            % V: Censoring (due to switching treatment or actually dropping out)
            logit_v = Rx(j)*(b0 + b1*(cumsum_A/207) + b2*(t(j)/170)^2) + (1-Rx(j))*(b3 + b4*cumsum_L/24 + b5*(t(j)/170)^2);                  
            p_v = expit(logit_v);
            if rand < p_v; V(j) = 1; break; end; % Stop simulation
        end
        
        % Death hazard - simple model where treatment is harmful
        logit_y = a0 + a1*(t(j)/170)^2 + (a2*sofa).*(cumsum_L/24)^2 + (a3*(age/90)).*(cumsum_A/207); 
        p_y = expit(logit_y);
        if rand < p_y; Y(j) = 1; break; end; % Stop simulation
           
    end
    t=t(1:length(Y));
    Y=Y';
    L=L';
    A=A';
    t=t';
    V=V';
    Rx=Rx';
