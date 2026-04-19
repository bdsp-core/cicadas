function parmsY_est = fcnEstimateDeathParms(T0)

% Calculate cumulative sums PER PATIENT
unique_sids = unique(T0.sid);
x1 = []; x2 = []; x3 = []; 
Y_all = [];

% logit_y = a0 + a1*(t(j)/170)^2 + (a2*sofa).*(cumsum_L/24)^2 + (a3*(age/90)).*(cumsum_A/207); 

for i = 1:length(unique_sids)
    patient_data = T0(T0.sid == unique_sids(i), :);
    tt = patient_data.t; 
    sofa = patient_data.sofa; 
    age = patient_data.age; 
    csL = cumsum(patient_data.L); 
    csA = cumsum(patient_data.A); 
    YY = patient_data.Y; 

    x1 = [x1; (tt/170).^2];
    x2 = [x2; sofa.*(csL/24).^2];
    x3 = [x3; (age/90).*(csA/207)];
    Y_all = [Y_all; YY];
end

X = [x1 x2 x3];
Y = Y_all; 
[bY, dev, stats] = glmfit(X, Y, 'binomial', 'link', 'logit');
% bY = [a0 a1 hE1 hE2 hA1 hA2]
parmsY_est = bY; 
