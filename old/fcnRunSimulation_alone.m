%% Function to run simulation with events (deaths, dropouts, treatment switches)
function [t, L, A, V, Y, hY, SY, hV, SV, hA, SA] = fcnRunSimulation_alone(Rx, harmE, harmA, C, g, th, kp, ki, Amax, b0, pulseAmp, pulseMu, pulseWidth, pulseC, RCT)
    % Pre-allocate
    dt = 0.5; % [h] sample period
    t = 0:dt:168; % 7 days
    Nt = numel(t);
    hazard_scale_t1 = 0.001/2;
    hazard_scale_t2 = 0.005/2;
    f = fcnSmoothPulse(t, pulseAmp, pulseMu, pulseWidth, pulseC);
    A = Rx*2*ones(1,Nt); % infusion rate
    L = zeros(1,Nt); % observed activity
    t1 = zeros(1,Nt); t2 = zeros(1,Nt);
    z = zeros(1,Nt); p = zeros(1,Nt);
    eInt = 0; % PI integrator state
    
    % Variables for treatment switching, stopping and dropout
    V = zeros(1,Nt); % Dropout indicator (0 = still in trial, 1 = dropped out)
    Y = zeros(1,Nt); % Death indicator (0 = alive, 1 = dead)
    switchTime = 0; % Time when treatment was switched (0 = no switch)
    stopped_treatment = 0; % Time when treatment was stopped (0 = not stopped)
    
    % Initialize treatment status
    if RCT == 0
        % Treatment probability based on harmE and harmA
        baseProb = 0.5;
        relativeFactor = (harmE - harmA) / (harmE + harmA);
        treatProb = baseProb + 0.5 * relativeFactor;
        
        % Assign treatment based on probability
        if rand < treatProb
            currentRx = 1; % Assign treatment
        else
            currentRx = 0; % Don't assign treatment
        end
        
        % Update initial treatment
        A(1) = currentRx*2;
    else
        % In non-simulation mode, use provided Rx
        currentRx = Rx;
    end
    
    % Parameters for hazards
    stop_treat_hazard_max = 0.02;
    dropout_hazard_base = 0.001;
    switch_hazard_max = 0.015;
    switch_hazard_decay = 0.03;
    
    % Hazard and survival tracking
    hV = zeros(1,Nt); % Dropout hazard
    SV = ones(1,Nt);  % Dropout survival probability
    hY = zeros(1,Nt); % Mortality hazard
    SY = ones(1,Nt);  % Mortality survival probability
    hA = zeros(1,Nt); % Switch hazard
    SA = ones(1,Nt);  % Switch survival probability
    
    % Simulation loop
    for j = 2:Nt
        % System update with previous pump rate
        sA = 1 - 1./((C./A(j-1)).^g + 1);
        L(j) = (b0 + f(j))*sA;
        
        % PI control
        e = L(j) - th;
        eInt = eInt + e*dt;
        Aunsat = kp*e + ki*eInt;
        
        % Saturation & anti-wind-up
        if stopped_treatment == 0
            A(j) = currentRx*min(max(Aunsat,0),Amax);
        else
            A(j) = 0; % Treatment is stopped
        end
        
        if A(j) ~= Aunsat
            eInt = eInt - (Aunsat - A(j))/ki;
        end
        
        % Hazard calculation for mortality
        raw_t1 = b0 + harmE*sum(L(1:j))/18.4 + harmA*sum(A(1:j))/360;
        t1(j) = hazard_scale_t1 * raw_t1;
        t2(j) = hazard_scale_t2 * (j/(Nt-1))^2;
        z(j) = t1(j) + t2(j);
        p(j) = 1 - exp(-z(j));
        
        % Treatment stopping logic (SIMULATION MODE ONLY)
        if RCT == 0 && stopped_treatment == 0 && currentRx == 1
            stop_treat_hazard = stop_treat_hazard_max * exp(-5 * L(j));
            if rand < stop_treat_hazard * dt
                stopped_treatment = t(j); % Record when treatment was stopped
                A(j) = 0; % Stop treatment
            end
        end
        
        % Treatment switching logic and hazard
        switch_hazard = switch_hazard_max * exp(-switch_hazard_decay * t(j));
        hA(j) = switch_hazard;
        SA(j) = SA(j-1) * exp(-hA(j) * dt);
        
        % Determine if treatment switches (SIMULATION MODE ONLY)
        if RCT == 0 && switchTime == 0 && stopped_treatment == 0 && rand < switch_hazard * dt
            currentRx = 1 - currentRx; % Switch treatment (0->1 or 1->0)
            switchTime = t(j); % Record switch time
            % Update treatment
            A(j) = currentRx*min(max(Aunsat,0),Amax);
        end
        
        % Dropout hazard and logic
        dropout_hazard = dropout_hazard_base * (1 + 2*L(j)^2 + 0.5*(harmE + harmA) + 10*((j/Nt)^2));
        hV(j) = dropout_hazard;
        SV(j) = SV(j-1) * exp(-hV(j) * dt);
        
        % Determine if dropout occurs (SIMULATION MODE ONLY)
        if RCT == 0 && rand < dropout_hazard * dt
            V(j:end) = 1; % Mark as dropped out
            break; % Exit simulation loop
        else
            V(j) = 0; % Still in trial
        end
        
        % Mortality hazard and event
        hY(j) = t1(j) + t2(j);
        SY(j) = SY(j-1) * exp(-hY(j) * dt);
        
        % Determine if death occurs
        if rand < hY(j) * dt
            Y(j:end) = 1; % Mark as dead
            break; % Exit simulation loop
        else
            Y(j) = 0; % Still alive
        end
    end
    
    % In non-simulation mode, reset dropout and treatment changes but keep death
    if RCT == 1
        V = zeros(1,Nt); % No dropouts
        A = Rx*2*ones(1,Nt); % Reset infusion rate
        switchTime = 0;
        stopped_treatment = 0;
        currentRx = Rx;
        % Y is kept as is to preserve death events
    end
    
    % Truncate data if death or dropout occurred
    if sum(Y)>0
        ind = find(Y==1);
        ii = 1:min(ind);
        t=t(ii); z=z(ii); t1=t1(ii); t2=t2(ii); p=p(ii);
        L=L(ii); A=A(ii); V=V(ii); Y=Y(ii);
        hY=hY(ii); SY=SY(ii); hA=hA(ii); SA=SA(ii); hV=hV(ii); SV=SV(ii);
    end
    if sum(V)>0
        ind = find(V==1);
        ii = 1:min(ind);
        t=t(ii); z=z(ii); t1=t1(ii); t2=t2(ii); p=p(ii);
        L=L(ii); A=A(ii); V=V(ii); Y=Y(ii);
        hY=hY(ii); SY=SY(ii); hA=hA(ii); SA=SA(ii); hV=hV(ii); SV=SV(ii);
    end
end