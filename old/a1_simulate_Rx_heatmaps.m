%% epilepsy_control.m – PI (only) closed loop + hazard model
clear; clc; format compact;
% Log‑normal‑shaped trajectory for L
pulseAmp = 1; pulseMu = 20; pulseWidth = 1.8; pulseC = 15;
%% 2. Model & PI‑controller parameters -----------------------------
C = 2;
g = 3;
kp = 0; % ↓ lower kp to tame overshoot
ki = 10; % integral gain (steady‑state accuracy)
Amax = 50; % pump upper bound

%% loop over cases:
figure(1); clf;
% case 1: harm of drug >> harm of epileptiform activity
% case 2: harm of drug << harm of epileptiform activity
% case 3: harm of drug ~ harm of epileptiform activity

hE_num = 0:.01:1;
hA_num = 0:.01:1;
Th = linspace(0,1,50); % target level of L
sp = 0;

for i = 1:length(hE_num)
    for j = 1:length(hA_num)
        disp([i j])
     harmE = 5*hE_num(i); % /(hE_num(i) + hA_num(j)+1);
     harmA = 5*hA_num(j); % /(hE_num(i) + hA_num(j)+1);
     mxDiff = -inf;
        for k = 1:length(Th);
             th = Th(k);
             %% 3. Hazard‑model constants ---------------------------------------
             b0 = 0.1;
             %% 5. Main simulation block -----------------------------------------
             sim_mode = 0;
             [t, z, t10, t20, p0, L0, A0, V0, Y0, hY0,SY0,hA0,SA0,hV0,SV0] = fcnRunSimulation_v2(0, harmE, harmA, C, g, th, kp, ki, Amax, b0, pulseAmp, pulseMu, pulseWidth, pulseC, sim_mode);
             [t, z, t11, t21, p1, L1, A1, V1, Y1 ,hY1,SY1,hA1,SA1,hV1,SV1] = fcnRunSimulation_v2(1, harmE, harmA, C, g, th, kp, ki, Amax, b0, pulseAmp, pulseMu, pulseWidth, pulseC, sim_mode);
             d = SY1(end);
            if d>mxDiff;
                 mxDiff = d;
                 d_survival = SY1(end);
                 d_difference = SY1(end)-SY0(end);
                 bestTh = th;
            end
        end
     M_survival(i,j) = d_survival;
     M_difference(i,j) = d_difference;
     M_target(i,j) = bestTh;
    end
end

%% Figures
figure(1); clf;
subplot(131); 
imagesc(hA_num/max(hA_num), hE_num/max(hE_num), M_target); 
axis xy; axis([0 1 0 1]);
xlabel('Harm A (normalized)');
ylabel('Harm E (normalized)');
title('Target E under optimal Rx');
colorbar;
axis square

subplot(132); 
imagesc(hA_num/max(hA_num), hE_num/max(hE_num), M_survival); 
axis xy; axis([0 1 0 1]);
xlabel('Harm A (normalized)');
ylabel('Harm E (normalized)');
title('Survival on optimal Rx');
colorbar;
axis square

subplot(133); 
imagesc(hA_num/max(hA_num), hE_num/max(hE_num), M_difference); 
axis xy; axis([0 1 0 1]);
xlabel('Harm A (normalized)');
ylabel('Harm E (normalized)');
title('Survival-opt vs no Rx');
colorbar;
axis square

set(gcf,'color','white')