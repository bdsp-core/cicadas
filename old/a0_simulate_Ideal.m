%% epilepsy_control.m  –  PI (only) closed loop + hazard model
clear; clc; format compact;

% Log‑normal‑shaped trajectory for L
pulseAmp = 1; pulseMu = 20; pulseWidth = 1.8; pulseC = 15; 

%% 2. Model & PI‑controller parameters -----------------------------
C = 2;      
g = 3;

kp   = 0;      % ↓ lower kp to tame overshoot
ki   = 10;     % integral gain (steady‑state accuracy)
Amax = 50;     % pump upper bound

th = .1;      % target level of L

%% loop over cases: 
figure(1); clf;


% case 1: harm of drug >> harm of epileptiform activity
% case 2: harm of drug << harm of epileptiform activity
% case 3: harm of drug ~ harm of epileptiform activity

hE_num = [0 1 0.5];  
hA_num = [0 0 .5]; 

sp = 0; 

for i = 1:3

    harmE = 5*hE_num(i); % /(hE_num(i) + hA_num(i)); 
    harmA = 5*hA_num(i); % /(hE_num(i) + hA_num(i)); 
   
    %% 3. Hazard‑model constants ---------------------------------------
    b0 = 0.001;      
      
    %% 5. Main simulation block -----------------------------------------
    sim_mode = 0; 
    [t, z, t10, t20, p0, L0, A0, V0, Y0, hY0,SY0,hA0,SA0,hV0,SV0] = fcnRunSimulation_v2(0, harmE, harmA, C, g, th, kp, ki, Amax, b0, pulseAmp, pulseMu, pulseWidth, pulseC, sim_mode);
    [t, z, t11, t21, p1, L1, A1, V1, Y1 ,hY1,SY1,hA1,SA1,hV1,SV1] = fcnRunSimulation_v2(1, harmE, harmA, C, g, th, kp, ki, Amax, b0, pulseAmp, pulseMu, pulseWidth, pulseC, sim_mode);
    
    %% 6. Plots ---------------------------------------------------------
    sp=sp+1; 
    subplot(3,4,sp)
    plot(t, L0, 'LineWidth',1.4); hold on
    plot(t, L1, 'LineWidth',1.4); 
    plot(t,th*ones(size(t)),'--','linewidth',1.4); 
    
    xlabel('time [h]'); ylabel('activity L');
    title('EA'); grid on
    box off
    
    sp=sp+1; 
    subplot(3,4,sp)
    plot(t, A0, 'LineWidth',1.4); hold on
    plot(t, A1, 'LineWidth',1.4); 
    plot(t, Amax*ones(size(t)),'--','LineWidth',1.2)
    xlabel('time [h]'); ylabel('infusion rate A');
    title('Drug rate'); grid on
    ylim([0 5])
    box off
    set(gcf,'color','w')
    
    sp=sp+1; 
    subplot(3,4,sp); 
    col = [0 0.4470 0.7410];
    plot(t, t10, '--','LineWidth',1.2,'color',col); hold on;  % Add '-' for solid line
    plot(t, t20, '--', 'LineWidth',1.2,'color',col);           % Add '-' for solid line
    plot(t, t10+t20, 'LineWidth',2,'color',col)           % Add '-' for solid line
    
    col = [0.6350 0.0780 0.1840];
    plot(t, t11, '--', 'LineWidth',1.2,'color',col); hold on;  % Add '-' for solid line
    plot(t, t21, '--', 'LineWidth',1.2,'color',col);           % Add '-' for solid line
    plot(t, t11+t21, 'LineWidth',2,'color',col);           % Add '-' for solid line
    
    ylabel('hazard')
    xlabel('time [h]')
    title('Hazards'); grid on
    set(gcf,'color','w')
    box off
    xlim([0 max(t)])

    sp=sp+1; 
    subplot(3,4,sp); 
    plot(t, 1-SY0, 'LineWidth',1.4); hold on
    plot(t, 1-SY1, 'LineWidth',1.4); ylabel('Death S(t)')
    box off
    xlim([0 max(t)])
    title('Survival'); grid on
    xlabel('time [h]')

end