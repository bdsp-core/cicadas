%% Modified fcnRunSimulation - Now with realistic treatment assignment and censoring
function [t, L, A, V, Y, Rx_actual] = fcnRunSimulation_GetDataOnly(Rx, harmE, harmA, C, g, th, kp, ki, Amax, b0, pulseAmp, pulseMu, pulseWidth, pulseC,RCT) 

 %% Pre‑allocate --------------------------------------------------
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
 Rx_actual = zeros(1,Nt); % Actual treatment status at each time point
 switchTime = 0; % Time when treatment was switched (0 = no switch)
 stopped_treatment = 0; % Time when treatment was stopped (0 = not stopped)
 
 % Initialize treatment status using assignment from main script
 % (Treatment assignment logic is handled in the main script for both RCT modes)
 currentRx = Rx;
 A(1) = currentRx*2;
 Rx_actual(1) = currentRx;
 
 % Parameters for hazards (reduced by 50% for more realistic adherence)
 % Treatment stopping hazard (higher when L is low)
 stop_treat_hazard_max = 0.01;  % Reduced from 0.02
 
 % Dropout hazard (depends on L, harmE, harmA)
 dropout_hazard_base = 0.0005;  % Reduced from 0.001
 
 % Treatment switching hazard 
 switch_hazard_max = 0.0075;    % Reduced from 0.015
 switch_hazard_decay = 0.03;
 
 % Hazard functions
 hV = zeros(1,Nt); % Dropout hazard
 hY = zeros(1,Nt); % Mortality hazard
 hA = zeros(1,Nt); % Switch hazard
 
 % Cumulative survival probabilities
 SV = ones(1,Nt); % Probability of not having dropped out
 SY = ones(1,Nt); % Probability of not having died
 SA = ones(1,Nt); % Probability of not having switched treatment
 
 %% Simulation loop
 for j = 2:Nt
     % % Exit early if already dropped out or died (only in simulation mode)
     
     % ---- system update with previous pump rate -------------------
     sA = 1 - 1./((C./A(j-1)).^g + 1);
     L(j) = (b0 + f(j))*sA;
     
     % ---- PI (with optional D) control ----------------------------
     e = L(j) - th;
     eInt = eInt + e*dt;
     Aunsat = kp*e + ki*eInt;
     
     % ---- saturation & anti‑wind‑up -------------------------------
     if stopped_treatment == 0
         A(j) = currentRx*min(max(Aunsat,0),Amax);
         Rx_actual(j) = currentRx;
     else
         A(j) = 0; % Treatment is stopped
         Rx_actual(j) = 0; % Not receiving treatment
     end
     
     if A(j) ~= Aunsat
         eInt = eInt - (Aunsat - A(j))/ki;
     end
     
     % ---- hazard calculation ---------------------------------------
     raw_t1 = b0 + harmE*sum(L(1:j))/18.4 + harmA*sum(A(1:j))/360;
     t1(j) = hazard_scale_t1 * raw_t1;
     t2(j) = hazard_scale_t2 * (j/(Nt-1))^2;
     z(j) = t1(j) + t2(j);
     p(j) = 1 - exp(-z(j));
     
     % ---- Treatment stopping logic (SIMULATION MODE ONLY) ---------------------------
     % Probability of stopping depends on L: low L -> higher chance of stopping
     if RCT == 0 && stopped_treatment == 0 && currentRx == 1
         stop_treat_hazard = stop_treat_hazard_max * exp(-5 * L(j));
         if rand < stop_treat_hazard * dt
             stopped_treatment = t(j); % Record when treatment was stopped
             A(j) = 0; % Stop treatment
             Rx_actual(j) = 0; % Update actual treatment status
         end
     end
     
     % ---- Treatment switching logic (SIMULATION MODE ONLY) -------------------------------
     % Exponentially decreasing switch hazard (higher early, lower later)
     switch_hazard = switch_hazard_max * exp(-switch_hazard_decay * t(j));
     hA(j) = switch_hazard;
     
     % Calculate cumulative probability of not having switched treatment
     SA(j) = SA(j-1) * exp(-hA(j) * dt);
     
     % Determine if treatment switches at this time step (only in simulation mode)
     if RCT == 0 && switchTime == 0 && stopped_treatment == 0 && rand < switch_hazard * dt
         currentRx = 1 - currentRx; % Switch treatment (0->1 or 1->0)
         switchTime = t(j); % Record switch time
         % Update treatment after switch
         A(j) = currentRx*min(max(Aunsat,0),Amax);
         Rx_actual(j) = currentRx; % Update actual treatment status
     end
     
     % ---- Dropout logic (SIMULATION MODE ONLY) -------------------------------------------
     % Dropout hazard depends on L, harmE, harmA
     dropout_hazard = dropout_hazard_base * (1 + 2*L(j)^2 + 0.5*(harmE + harmA)+10*((j/Nt)^2));
     hV(j) = dropout_hazard;
     
     % Calculate cumulative probability of not having dropped out
     SV(j) = SV(j-1) * exp(-hV(j) * dt);
     
     % Determine if participant drops out at this time step (only in simulation mode)
     if RCT == 0 && rand < dropout_hazard * dt
         V(j:end) = 1; % Mark as dropped out for the remainder of the trial
         break; % Exit the simulation loop
     else
         V(j) = 0; % Still in the trial
     end
     
     % ---- Mortality hazard and calculate survival function ----
     hY(j) = t1(j) + t2(j);
     
     % Calculate cumulative probability for mortality
     SY(j) = SY(j-1) * exp(-hY(j) * dt);
     
     % Determine if death occurs (in BOTH simulation modes)
     if rand < hY(j) * dt
         Y(j:end) = 1; % Mark as dead for the remainder of the trial
         break; % Exit the simulation loop
     else
         Y(j) = 0; % Still alive
     end
 end
 
 % In non-simulation mode, prevent dropout and treatment changes but keep death
 if RCT == 1
     V = zeros(1,Nt); % No dropouts in non-simulation mode
     A = Rx*2*ones(1,Nt); % Reset infusion rate to original
     Rx_actual = Rx*ones(1,Nt); % Reset treatment status to original assignment
     switchTime = 0;
     stopped_treatment = 0;
     currentRx = Rx;
     % Y is kept as is to preserve death events
 end
 
 if sum(Y)>0; 
     ind = find(Y==1); 
     ii = 1:min(ind); 
     t=t(ii);
     z=z(ii); 
     t1=t1(ii); 
     t2=t2(ii); 
     p=p(ii); 
     L=L(ii); 
     A=A(ii); 
     V=V(ii); 
     Y=Y(ii); 
     Rx_actual=Rx_actual(ii);
     hY=hY(ii); 
     SY=SY(ii); 
     hA=hA(ii); 
     SA=SA(ii); 
     hV=hV(ii); 
     SV=SV(ii); 
 end
 if sum(V)>0; 
     ind = find(V==1); 
     ii = 1:min(ind); 
     t=t(ii);
     z=z(ii); 
     t1=t1(ii); 
     t2=t2(ii); 
     p=p(ii); 
     L=L(ii); 
     A=A(ii); 
     V=V(ii); 
     Y=Y(ii); 
     Rx_actual=Rx_actual(ii);
     hY=hY(ii); 
     SY=SY(ii); 
     hA=hA(ii); 
     SA=SA(ii); 
     hV=hV(ii); 
     SV=SV(ii); 
 end