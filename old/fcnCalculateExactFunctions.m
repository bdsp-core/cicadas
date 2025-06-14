%% Function to calculate exact hazard and survival functions
function [t_exact, exact_hY, exact_SY, exact_hV, exact_SV, exact_hA, exact_SA] = fcnCalculateExactFunctions(Rx, harmE, harmA, C, g, th, kp, ki, Amax, b0, pulseAmp, pulseMu, pulseWidth, pulseC)
    % Time parameters
    dt = 0.5; % [h] sample period
    t_exact = 0:dt:168; % 7 days
    Nt = numel(t_exact);
    
    % Hazard scaling parameters
    hazard_scale_t1 = 0.001/2;
    hazard_scale_t2 = 0.005/2;
    
    % Pulse parameters for L
    f = fcnSmoothPulse(t_exact, pulseAmp, pulseMu, pulseWidth, pulseC);
    
    % Pre-allocate 
    exact_A = Rx*2*ones(1,Nt); % Theoretical infusion rate
    exact_L = zeros(1,Nt);     % Theoretical activity
    eInt = 0;                  % PI integrator state
    
    % Parameters for hazards
    dropout_hazard_base = 0.001;
    switch_hazard_max = 0.015;
    switch_hazard_decay = 0.03;
    
    % Pre-allocate exact hazard and survival functions
    exact_hV = zeros(1,Nt); % Exact dropout hazard
    exact_SV = ones(1,Nt);  % Exact dropout survival probability
    exact_hY = zeros(1,Nt); % Exact mortality hazard
    exact_SY = ones(1,Nt);  % Exact mortality survival probability
    exact_hA = zeros(1,Nt); % Exact switch hazard
    exact_SA = ones(1,Nt);  % Exact switch survival probability
    
    % First, simulate the full trajectory of L and A without any events
    for j = 2:Nt
        % System update with previous pump rate
        sA = 1 - 1./((C./exact_A(j-1)).^g + 1);
        exact_L(j) = (b0 + f(j))*sA;
        
        % PI control
        e = exact_L(j) - th;
        eInt = eInt + e*dt;
        Aunsat = kp*e + ki*eInt;
        
        % Saturation & anti-wind-up
        exact_A(j) = Rx*min(max(Aunsat,0),Amax);
        if exact_A(j) ~= Aunsat
            eInt = eInt - (Aunsat - exact_A(j))/ki;
        end
    end
    
    % Now calculate exact hazard and survival curves using the full L and A
    for j = 2:Nt
        % 1. Dropout hazard
        exact_hV(j) = dropout_hazard_base * (1 + 2*exact_L(j)^2 + 0.5*(harmE + harmA) + 10*((j/Nt)^2));
        exact_SV(j) = exact_SV(j-1) * exp(-exact_hV(j) * dt);
        
        % 2. Treatment switching hazard
        exact_hA(j) = switch_hazard_max * exp(-switch_hazard_decay * t_exact(j));
        exact_SA(j) = exact_SA(j-1) * exp(-exact_hA(j) * dt);
        
        % 3. Mortality hazard
        raw_t1 = b0 + harmE*sum(exact_L(1:j))/18.4 + harmA*sum(exact_A(1:j))/360;
        exact_t1 = hazard_scale_t1 * raw_t1;
        exact_t2 = hazard_scale_t2 * (j/(Nt-1))^2;
        exact_hY(j) = exact_t1 + exact_t2;
        exact_SY(j) = exact_SY(j-1) * exp(-exact_hY(j) * dt);
    end
end
