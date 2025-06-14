%% a3_simulate_v3.m - ICU Treatment Trial Simulation with improved realism
% This is the main file to run the ICU treatment trial simulation with
% realistic treatment assignments and censoring mechanisms

%% Main Simulation Script with Data Collection
 %% Model & Controller parameters -----------------------------
 C = 2;      
 g = 3;
 
 kp   = 3;      % ↓ lower kp to tame overshoot
 ki   = 10;     % integral gain (steady‑state accuracy)
 Amax = 50;     % pump upper bound
 
 th = 0.1;      % target level of L
 
 %% Number of subjects
 N = 100; 
 
 %% Set up data collection
 % Structure to collect all simulation data
 trialData = struct();
 trialData.subjectID = zeros(N, 1);
 trialData.b0 = zeros(N, 1);
 trialData.initialRx = zeros(N, 1);
 trialData.harmE = zeros(N, 1);
 trialData.harmA = zeros(N, 1);
 trialData.switchTime = zeros(N, 1);
 trialData.stoppedTreatment = zeros(N, 1);
 
 % Time series data for each subject
 timeData = cell(N, 1);
 
 %% Run simulations for each subject
 for i = 1:N
     % Set subject ID
     trialData.subjectID(i) = i;
     
     % Set baseline covariates with some random variation
     b0 = 0.1 + 0.02*randn;
     trialData.b0(i) = b0;
     
     % Vary the harm parameters between subjects
     hE_num = 10 + 2*randn;  
     hA_num = 3 + 1*randn; 
     harmE = 5*hE_num/(hE_num + hA_num); 
     harmA = 5*hA_num/(hE_num + hA_num);
     
     trialData.harmE(i) = harmE;
     trialData.harmA(i) = harmA;
     
     % Pulse parameters with variation
     pulseAmp = 1 + 0.2*randn;
     pulseMu = 20 + 4*randn;
     pulseWidth = 1.8 + 0.3*randn;
     pulseC = 15 + 3*randn;
     
     % Use simulation mode = 1 for realistic treatment assignment
     sim_mode = 1;
     
     % Initial treatment assignment is handled within fcnRunSimulation
     % using harmE and harmA
     initialRx = 0; % This will be overridden in the simulation
     
     % Run the simulation
     [t, z, t1, t2, p, L, A, V, Y, switchTime, hY, SY, hA, SA, hV, SV, stoppedTreatment] = ...
         fcnRunSimulation_v2(initialRx, harmE, harmA, C, g, th, kp, ki, Amax, b0, ...
         pulseAmp, pulseMu, pulseWidth, pulseC, sim_mode);
     
     % Store actual treatment assignment (check first time point)
     trialData.initialRx(i) = A(1)/2;
     
     % Store times when treatment changes occurred
     trialData.switchTime(i) = switchTime;
     trialData.stoppedTreatment(i) = stoppedTreatment;
     
     % Create a table for time-dependent data
     Nt = length(t);
     timeTable = table();
     timeTable.subjectID = i * ones(Nt, 1);
     timeTable.timePoint = (1:Nt)';
     timeTable.time = t';
     timeTable.L = L';
     timeTable.A = A';
     timeTable.V = V';
     timeTable.Y = Y';
     timeTable.hazard = z';
     timeTable.survivalProb = (1 - p)';
     
     % Store the time table
     timeData{i} = timeTable;
 end
 
 % Combine all time-dependent data into one big table
 allTimeData = vertcat(timeData{:});
 
 % Create a subject-level table
 subjectData = struct2table(trialData);
 
 % Display summary statistics
 disp('Summary of Trial:');
 disp(['Number of subjects: ' num2str(N)]);
 disp(['Number with initial treatment: ' num2str(sum(trialData.initialRx))]);
 disp(['Number with treatment switch: ' num2str(sum(trialData.switchTime > 0))]);
 disp(['Number with stopped treatment: ' num2str(sum(trialData.stoppedTreatment > 0))]);
 
 % Save the data
 save('trial_data.mat', 'subjectData', 'allTimeData');
 
 % Example Visualization
 figure(1);
 subplot(2,1,1);
 plot(t, L);
 title('Epileptiform Activity (L) for All Subjects');
 xlabel('Time [h]');
 ylabel('Activity');
 grid on;
 
 subplot(2,1,2);
 plot(t, A);
 title('Treatment Level (A) for All Subjects');
 xlabel('Time [h]');
 ylabel('Treatment');
 grid on;
 
 % Additional analysis can be performed here
 % ...

