function T = fcnSimulate_N_Patients_withU(N, RCT, treatProb, th, C, g, ke, L0, ...
                                           parmsControl, parmsY, parmsV, age, sofa, ...
                                           u_effect_on_Y)
% FCNSIMULATE_N_PATIENTS_WITHU  Simulate N patients with an optional
%   per-patient additive shift to the mortality logit.
%
%   u_effect_on_Y  N-vector of per-patient additive shifts to logit_y.
%                  In the NUC experiment this is U(i) * delta_Y where
%                  U ~ Bernoulli(0.5) is an unmeasured baseline confounder.
%                  Pass zeros(1,N) to recover the baseline DGP.
%
% Identical to fcnSimulate_N_Patients.m except for the extra additive
% mortality term. U is never written to the output table -- it is an
% unobserved confounder by construction, so downstream g-formula estimators
% cannot adjust for it.

if nargin < 14 || isempty(u_effect_on_Y)
    u_effect_on_Y = zeros(1, N);
end

tt = []; Lt = []; At = []; Vt = []; Yt = []; Rxt = [];
sid = []; Age = []; Sofa = [];

for i = 1:N
    Rx = rand < treatProb(i);
    [t, L, A, V, Y, Rx] = runPatient(RCT, Rx, th, C(i), g(i), age(i), sofa(i), ...
                                     ke, L0(i,:), parmsControl, parmsY, parmsV, ...
                                     u_effect_on_Y(i));
    sid = [sid; i*ones(length(t),1)];
    tt  = [tt; t]; Lt  = [Lt; L]; At  = [At; A];
    Vt  = [Vt; V]; Yt  = [Yt; Y]; Rxt = [Rxt; Rx];
    Age = [Age; age(i)*ones(size(t))];
    Sofa = [Sofa; sofa(i)*ones(size(t))];
end

T = table(Age, Sofa, sid, tt, Lt, At, Vt, Yt, Rxt, ...
          'VariableNames', {'age','sofa','sid','t','L','A','V','Y','Rx'});
end

function [t, L, A, V, Y, Rx] = runPatient(RCT, Rx, th, C, g, age, sofa, ke, L0, ...
                                          parmsControl, parmsY, parmsV, u_shift_y)
ki = parmsControl(1); Amax = parmsControl(2);
a0 = parmsY(1); a1 = parmsY(2); a2 = parmsY(3); a3 = parmsY(4);
if length(parmsY) >= 5, a4 = parmsY(5); else, a4 = 0; end %#ok<NASGU>
b0 = parmsV(1); b1 = parmsV(2); b2 = parmsV(3);
b3 = parmsV(4); b4 = parmsV(5); b5 = parmsV(6);

dt = 2; t = (0:dt:168); Nt = numel(t);
expit = @(x) 1./(1 + exp(-x));
eInt = 0;

A(1) = Rx*2; L(1) = 0; V(1) = 0; Y(1) = 0; X(1) = 0;
cumsum_L = 0; cumsum_A = 0;

for j = 2:Nt
    Rx(j) = Rx(j-1);
    if Rx(j) == 1
        sX = 1 - 1./((C./X(j-1)).^g + 1);
    else
        sX = 1;
    end
    L(j) = L0(j) * sX;
    if Rx(j) == 1 && j > 5
        e = L(j) - th;
        eInt = eInt + e*dt;
        Aunsat = ki * eInt;
        A(j) = min(max(Aunsat, 0), Amax);
        if A(j) ~= Aunsat
            eInt = eInt - (Aunsat - A(j))/ki;
        end
    else
        A(j) = 0;
    end
    X(j) = ke * X(j-1) + A(j);
    cumsum_L = cumsum_L + L(j);
    cumsum_A = cumsum_A + A(j);

    V(j) = 0; Y(j) = 0;
    if RCT == 0
        logit_v = Rx(j)*(b0 + b1*(cumsum_A/207) + b2*(t(j)/170)^2) ...
                + (1-Rx(j))*(b3 + b4*cumsum_L/24 + b5*(t(j)/170)^2);
        p_v = expit(logit_v);
        if rand < p_v, V(j) = 1; break; end
    end
    % Mortality hazard with optional U injection
    logit_y = a0 + a1*(t(j)/170)^2 + (a2*sofa)*(cumsum_L/24)^2 ...
              + (a3*(age/90))*(cumsum_A/207) + u_shift_y;
    p_y = expit(logit_y);
    if rand < p_y, Y(j) = 1; break; end
end

t = t(1:length(Y)).';
Y = Y.'; L = L.'; A = A.'; V = V.'; Rx = Rx.';
end
