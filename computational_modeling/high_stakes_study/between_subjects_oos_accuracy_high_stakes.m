%-----------------Zbozinek*, Charpentier*, Qi*, & Mobbs---------------------
% *co-first authors

%Calculates out-of-sample predictive accuracy across subjects

clear all
close all

data_dir = pwd
cd(data_dir)

% init the randomization screen
RandStream.setGlobalStream(RandStream('mt19937ar','Seed','shuffle'));

%load data
fname = 'high_stakes_data.xlsx';
[numd,txtd,rawd] = xlsread(fname);

%Setup solver opt 
npar = 9; %number of parameters
opts = optimoptions(@fminunc, ...
        'Algorithm', 'quasi-newton', ...
        'Display','off', ...
        'MaxFunEvals', 50 * npar, ...
        'MaxIter', 50 * npar,...
        'TolFun', 0.01, ...
        'TolX', 0.01);

nsub = 210; %number of subjects  
nsim = 100; %number of simulations
grp_size = 30;
nmod = 6; %number of models

%load model results
load('model_results_high_stakes.mat','fitResult')
for sub = 1:nsub
    isub = numd(:,1) == sub;
    P = numd(isub,[9 2:8]);
    missed = isnan(P(:,8));
    P(missed,:) = [];
    tr_nb = length(P(:,1));
    fitResult.Data(sub,1).P = P;
end
save('model_results_high_stakes.mat','fitResult')
data = fitResult.Data;

params_mod1 = fitResult.Model1.Params;
params_mod2 = fitResult.Model2.Params;
params_mod3 = fitResult.Model3.Params;
params_mod4 = fitResult.Model4.Params;
params_mod5 = fitResult.Model5.Params;
params_mod6 = fitResult.Model6.Params;

%% Now estimate out-of-sample accuracy 
ngrp=7;
model1_accuracy = zeros(nsim,ngrp);
model2_accuracy = zeros(nsim,ngrp);
model3_accuracy = zeros(nsim,ngrp);
model4_accuracy = zeros(nsim,ngrp);
model5_accuracy = zeros(nsim,ngrp);
model6_accuracy = zeros(nsim,ngrp);

ll_mod1 = zeros(nsim,ngrp);
ll_mod2 = zeros(nsim,ngrp);
ll_mod3 = zeros(nsim,ngrp);
ll_mod4 = zeros(nsim,ngrp);
ll_mod5 = zeros(nsim,ngrp);
ll_mod6 = zeros(nsim,ngrp);

parfor (s=1:nsim,8) %change to actual number of cores
    
    disp(['Simulation # ' num2str(s)])
    perm = [randperm(nsub)' repmat((1:ngrp)',grp_size,1)];
    perm_sorted = sortrows(perm,1);
    
    perm_mod1 = [perm_sorted params_mod1];  
    perm_mod2 = [perm_sorted params_mod2];   
    perm_mod3 = [perm_sorted params_mod3];
    perm_mod4 = [perm_sorted params_mod4];
    perm_mod5 = [perm_sorted params_mod5];
    perm_mod6 = [perm_sorted params_mod6];
    
    for g=1:ngrp
        
        %Model 1: Prospect theory model using all data
        id_train = perm_mod1(:,2)~=g;
        param_train = mean(perm_mod1(id_train,3:end),1); %mean params from training subjects
        %now look at behavior of test subjects
        id_test = find(perm_mod1(:,2)==g);
        model_accuracy_test_subs = zeros(length(id_test),1);
        ll_test_subs = zeros(length(id_test),1);
        for i=1:length(id_test)
            subj = id_test(i);
            P = data(subj).P;
            model_accuracy_tmp = zeros(100,1);
            for a=1:100
                P_pred = generate_choice_Model1(param_train,P);
                model_accuracy_tmp(a) = nanmean(P_pred(:,6));
            end
            model_accuracy_test_subs(i) = mean(model_accuracy_tmp);
            ll_test_subs(i) = nanmean(P_pred(:,7));
        end
        model1_accuracy(s,g) = mean(model_accuracy_test_subs);
        ll_mod1(s,g) = mean(ll_test_subs);
    
        %Model 2: One ambiguity parameter common for C2-3-5-6-7-8
        id_train = perm_mod2(:,2)~=g;
        param_train = mean(perm_mod2(id_train,3:end),1); %mean params from training subjects
        %now look at behavior of test subjects
        id_test = find(perm_mod2(:,2)==g);
        model_accuracy_test_subs = zeros(length(id_test),1);
        ll_test_subs = zeros(length(id_test),1);
        for i=1:length(id_test)
            subj = id_test(i);
            P = data(subj).P;
            model_accuracy_tmp = zeros(100,1);
            for a=1:100
                P_pred = generate_choice_Model2(param_train,P);
                model_accuracy_tmp(a) = nanmean(P_pred(:,6));
            end
            model_accuracy_test_subs(i) = mean(model_accuracy_tmp);
            ll_test_subs(i) = nanmean(P_pred(:,7));
        end
        model2_accuracy(s,g) = mean(model_accuracy_test_subs);
        ll_mod2(s,g) = mean(ll_test_subs);

        %Model 3: Separate ambiguity parameters for gains (C3-5-6-7) and losses (C2-8)
        id_train = perm_mod3(:,2)~=g;
        param_train = mean(perm_mod3(id_train,3:end),1); %mean params from training subjects
        %now look at behavior of test subjects
        id_test = find(perm_mod3(:,2)==g);
        model_accuracy_test_subs = zeros(length(id_test),1);
        ll_test_subs = zeros(length(id_test),1);
        for i=1:length(id_test)
            subj = id_test(i);
            P = data(subj).P;
            model_accuracy_tmp = zeros(100,1);
            for a=1:100
                P_pred = generate_choice_Model3(param_train,P);
                model_accuracy_tmp(a) = nanmean(P_pred(:,6));
            end
            model_accuracy_test_subs(i) = mean(model_accuracy_tmp);
            ll_test_subs(i) = nanmean(P_pred(:,7));
        end
        model3_accuracy(s,g) = mean(model_accuracy_test_subs);
        ll_mod3(s,g) = mean(ll_test_subs);
        
        %Model 4: Separate ambiguity parameters depending on context: 
        % a loss is present (C2-3-8) vs no loss present (C5-6-7)
        id_train = perm_mod4(:,2)~=g;
        param_train = mean(perm_mod4(id_train,3:end),1); %mean params from training subjects
        %now look at behavior of test subjects
        id_test = find(perm_mod4(:,2)==g);
        model_accuracy_test_subs = zeros(length(id_test),1);
        ll_test_subs = zeros(length(id_test),1);
        for i=1:length(id_test)
            subj = id_test(i);
            P = data(subj).P;
            model_accuracy_tmp = zeros(100,1);
            for a=1:100
                P_pred = generate_choice_Model4(param_train,P);
                model_accuracy_tmp(a) = nanmean(P_pred(:,6));
            end
            model_accuracy_test_subs(i) = mean(model_accuracy_tmp);
            ll_test_subs(i) = nanmean(P_pred(:,7));
        end
        model4_accuracy(s,g) = mean(model_accuracy_test_subs);
        ll_mod4(s,g) = mean(ll_test_subs);
        
        %Model 5: Separate ambiguity parameters for risky gain (C3-5), risky loss (C2), sure gain (C6-7) and sure loss (C8)
        id_train = perm_mod5(:,2)~=g;
        param_train = mean(perm_mod5(id_train,3:end),1); %mean params from training subjects
        %now look at behavior of test subjects
        id_test = find(perm_mod5(:,2)==g);
        model_accuracy_test_subs = zeros(length(id_test),1);
        ll_test_subs = zeros(length(id_test),1);
        for i=1:length(id_test)
            subj = id_test(i);
            P = data(subj).P;
            model_accuracy_tmp = zeros(100,1);
            for a=1:100
                P_pred = generate_choice_Model5(param_train,P);
                model_accuracy_tmp(a) = nanmean(P_pred(:,6));
            end
            model_accuracy_test_subs(i) = mean(model_accuracy_tmp);
            ll_test_subs(i) = nanmean(P_pred(:,7));
        end
        model5_accuracy(s,g) = mean(model_accuracy_test_subs);
        ll_mod5(s,g) = mean(ll_test_subs);
                
        %Model 6: Full model: ambiguity parameter for C2, 3, 5, 6, 7, 8 separate
        id_train = perm_mod6(:,2)~=g;
        param_train = mean(perm_mod6(id_train,3:end),1); %mean params from training subjects
        %now look at behavior of test subjects
        id_test = find(perm_mod6(:,2)==g);
        model_accuracy_test_subs = zeros(length(id_test),1);
        ll_test_subs = zeros(length(id_test),1);
        for i=1:length(id_test)
            subj = id_test(i);
            P = data(subj).P;
            model_accuracy_tmp = zeros(100,1);
            for a=1:100
                P_pred = generate_choice_Model6(param_train,P);
                model_accuracy_tmp(a) = nanmean(P_pred(:,6));
            end
            model_accuracy_test_subs(i) = mean(model_accuracy_tmp);
            ll_test_subs(i) = nanmean(P_pred(:,7));
        end
        model6_accuracy(s,g) = mean(model_accuracy_test_subs);
        ll_mod6(s,g) = mean(ll_test_subs);

    end     
end

fitResult.Accuracy.Model1.PerGroup = model1_accuracy;
fitResult.Accuracy.Model2.PerGroup = model2_accuracy;
fitResult.Accuracy.Model3.PerGroup = model3_accuracy;
fitResult.Accuracy.Model4.PerGroup = model4_accuracy;
fitResult.Accuracy.Model5.PerGroup = model5_accuracy;
fitResult.Accuracy.Model6.PerGroup = model6_accuracy;

fitResult.Accuracy.MeanAllModels = [mean(model1_accuracy,2) mean(model2_accuracy,2) mean(model3_accuracy,2) ...
    mean(model4_accuracy,2) mean(model5_accuracy,2) mean(model6_accuracy,2)];

save('model_results_high_stakes.mat','fitResult')