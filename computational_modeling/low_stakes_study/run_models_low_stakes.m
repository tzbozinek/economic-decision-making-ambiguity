%-----------------Zbozinek*, Charpentier*, Qi*, & Mobbs---------------------
% *co-first authors

clear all
close all

fs = filesep;

data_dir = pwd; %change with directory where data (subject folders) are saved
cd(data_dir)

numcores = feature('numcores');

fname = 'low_stakes_data.xlsx';
[numd,txtd,rawd] = xlsread(fname);

% init the randomization screen
RandStream.setGlobalStream(RandStream('mt19937ar','Seed','shuffle'));

%Setup solver opt 
npar = 9;
opts = optimoptions(@fminunc, ...
        'Algorithm', 'quasi-newton', ...
        'Display','off', ...
        'MaxFunEvals', 50 * npar, ...
        'MaxIter', 50 * npar,...
        'TolFun', 0.01, ...
        'TolX', 0.01);

nsub = 367; %number of subjects  
sim_nb = 50; %number of simulations for recovery

fitResult = struct();

% Model 1: No Ambiguity
% x(1) = inverse temperature mu (>0)
% x(2) = loss aversion lambda (>0)
% x(3) = risk preference rho (>0)
Model1_Params        = zeros(nsub,3);
Model1_Loglikelihood = zeros(nsub,1);
Model1_ExitFlag      = zeros(nsub,1);
Model1_AIC           = zeros(nsub,1);
Model1_BIC           = zeros(nsub,1);
Model1_PseudoR2      = zeros(nsub,1);
Model1_RecovParams   = zeros(nsub,3);
Model1_RecovStd      = zeros(nsub,3);
Model1_Accuracy      = zeros(nsub,6);
% 
% Model 2: One ambiguity parameter common for C2-3-5-6-7-8
% x(1) = inverse temperature mu (>0)
% x(2) = loss aversion lambda (>0)
% x(3) = risk preference rho (>0)
% x(4) = ambiguity parameter (>0) - amb. preference for gains & aversion for losses
Model2_Params        = zeros(nsub,4);
Model2_Loglikelihood = zeros(nsub,1);
Model2_ExitFlag      = zeros(nsub,1);
Model2_AIC           = zeros(nsub,1);
Model2_BIC           = zeros(nsub,1);
Model2_PseudoR2      = zeros(nsub,1);
Model2_RecovParams   = zeros(nsub,4);
Model2_RecovStd      = zeros(nsub,4);
Model2_Accuracy      = zeros(nsub,6);
%
% Model 3: Separate ambiguity parameters for gains (C3-5-6-7) and losses (C2-8)
% x(1) = inverse temperature mu (>0)
% x(2) = loss aversion lambda (>0)
% x(3) = risk preference rho (>0)
% x(4) = ambiguity preference for gains (>0)
% x(5) = ambiguity aversion for losses (>0)
Model3_Params        = zeros(nsub,5);
Model3_Loglikelihood = zeros(nsub,1);
Model3_ExitFlag      = zeros(nsub,1);
Model3_AIC           = zeros(nsub,1);
Model3_BIC           = zeros(nsub,1);
Model3_PseudoR2      = zeros(nsub,1);
Model3_RecovParams   = zeros(nsub,5);
Model3_RecovStd      = zeros(nsub,5);
Model3_Accuracy      = zeros(nsub,6);
% 
% Model 4: Separate ambiguity parameters depending on context: a loss is
% present (C2-3-8) vs no loss present (C5-6-7)
% x(1) = inverse temperature mu (>0)
% x(2) = loss aversion lambda (>0)
% x(3) = risk preference rho (>0)
% x(4) = ambiguity parameter loss context C2-3-8 (>0)
% x(5) = ambiguity parameter no-loss context C5-6-7 (>0)
Model4_Params        = zeros(nsub,5);
Model4_Loglikelihood = zeros(nsub,1);
Model4_ExitFlag      = zeros(nsub,1);
Model4_AIC           = zeros(nsub,1);
Model4_BIC           = zeros(nsub,1);
Model4_PseudoR2      = zeros(nsub,1);
Model4_RecovParams   = zeros(nsub,5);
Model4_RecovStd      = zeros(nsub,5);
Model4_Accuracy      = zeros(nsub,6);
% 
% Model 5: Separate ambiguity parameters for risky gain (C3-5), risky loss
% (C2), sure gain (C6-7) and sure loss (C8)
% x(1) = inverse temperature mu (>0)
% x(2) = loss aversion lambda (>0)
% x(3) = risk preference rho (>0)
% x(4) = ambiguity preference risky gains C3-5 (>0)
% x(5) = ambiguity aversion risky loss C2 (>0)
% x(6) = ambiguity preference sure gain C6-7 (>0)
% x(7) = ambiguity aversion sure loss C8 (>0)
Model5_Params        = zeros(nsub,7);
Model5_Loglikelihood = zeros(nsub,1);
Model5_ExitFlag      = zeros(nsub,1);
Model5_AIC           = zeros(nsub,1);
Model5_BIC           = zeros(nsub,1);
Model5_PseudoR2      = zeros(nsub,1);
Model5_RecovParams   = zeros(nsub,7);
Model5_RecovStd      = zeros(nsub,7);
Model5_Accuracy      = zeros(nsub,6);
%


parfor (i = 1:nsub,numcores)
    
    isub = numd(:,1) == i;
    P = numd(isub,[9 2:8]);
    missed = isnan(P(:,8));
    P(missed,:) = [];
    tr_nb = length(P(:,1));
    P(:,4:7) = P(:,4:7)*100;
    
      
   %% Model 1: Prospect theory model using all data
    npar = 3;
    
    %first fit params to subject data
    disp(['Sub' num2str(i) '- Model 1 fit'])
    params_rand=[]; 
    for i_rand=1:100*npar
        start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
        [paramtracker, lltracker, ex] = fminunc(@LL_function_Model1, start, opts, P);
        params_rand=[params_rand; paramtracker lltracker ex];
    end
    bad_fit = params_rand(:,npar+2)<=0;
    if sum(bad_fit)<100*npar
        params_rand(bad_fit,:)=[];
    end
    [~,ids] = sort(params_rand(:,npar+1));
    best_params = params_rand(ids(1),:);
    best_params(1) = 1/(1+exp(-best_params(1))); % mu [0 1]
    best_params(2) = 10/(1+exp(-best_params(2))); % lambda [0 10]
    best_params(3) = 3/(1+exp(-best_params(3)));  % rho [0 3]
    BIC = 2*best_params(npar+1) + npar*log(tr_nb);
    AIC = 2*best_params(npar+1) + 2*npar;
    pseudoR2 = 1 + best_params(npar+1)/(log(0.5)*tr_nb);
    
    Model1_Params(i,:)        = best_params(1:npar); %save parameters used to generate data
    Model1_Loglikelihood(i,:) = best_params(npar+1);
    Model1_ExitFlag(i,:)      = best_params(npar+2);
    Model1_AIC(i,:)           = AIC;
    Model1_BIC(i,:)           = BIC;
    Model1_PseudoR2(i,:)      = pseudoR2;
    
    %parameter recovery analysis
    disp(['Sub' num2str(i) '- Model 1 Recovery'])
    params = best_params(1:npar);
    param_r = zeros(sim_nb,npar); %initiate list of recovered parameter
    for s=1:sim_nb
        P_pred = generate_choice_Model1(params,P); %generate choice set
        gen_ch = P_pred(:,4); %set of generated choices, column nb may need to be changed!
        P_new = P;
        P_new(:,8) = gen_ch;
        %fit model to generated choice set
        params_rand=[]; 
        for i_rand=1:20*npar
            start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
            [paramtracker, lltracker, ex] = fminunc(@LL_function_Model1, start, opts, P_new);
            params_rand=[params_rand; paramtracker lltracker ex];
        end
        bad_fit = params_rand(:,npar+2)<=0;
        if sum(bad_fit)<20*npar
            params_rand(bad_fit,:)=[];
        end
        [~,ids] = sort(params_rand(:,npar+1));
        best_params = params_rand(ids(1),:);
        best_params(1) = 1/(1+exp(-best_params(1))); % mu [0 1]
        best_params(2) = 10/(1+exp(-best_params(2))); % lambda [0 10]
        best_params(3) = 3/(1+exp(-best_params(3)));  % rho [0 3]
        param_r(s,:) = best_params(1:npar);
    end
    Model1_RecovParams(i,:)   = mean(param_r,1);
    Model1_RecovStd(i,:)      = std(param_r,1);
    
    %test out of sample accuracy (6-fold leave-one-out)
    disp(['Sub' num2str(i) '- Model 1 Accuracy'])
    model_accuracy_summary = [];  
    grp = repmat((1:6)',56,1);
    P(:,9) = grp(1:tr_nb);
    for b = 1:6 %run leave-one out cross-validation for predictive accuracy
        tr_train_id = P(:,9)~=b;
        tr_test_id  = P(:,9)==b;
        P_train = P(tr_train_id,:); 
        P_test = P(tr_test_id,:); 
              
        params_rand=[]; 
        for i_rand=1:20*npar
            start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
            [paramtracker, lltracker, ex] = fminunc(@LL_function_Model1, start, opts, P_train);
            params_rand=[params_rand; paramtracker lltracker ex];
        end
        bad_fit = params_rand(:,npar+2)<=0;
        if sum(bad_fit)<20*npar
            params_rand(bad_fit,:)=[];
        end
        [~,ids] = sort(params_rand(:,npar+1));
        best_params = params_rand(ids(1),:);
        best_params(1) = 1/(1+exp(-best_params(1))); % mu [0 1]
        best_params(2) = 10/(1+exp(-best_params(2))); % lambda [0 10]
        best_params(3) = 3/(1+exp(-best_params(3)));  % rho [0 3]
        %calculate model accuracy on test data using these params
        param = best_params(1:npar);
        model_accuracy_tmp = zeros(100,1);
        for s=1:100
            P_pred = generate_choice_Model1(param,P_test);
            model_accuracy_tmp(s) = nanmean(P_pred(:,6));
        end
        model_accuracy = mean(model_accuracy_tmp);
        model_accuracy_summary = [model_accuracy_summary model_accuracy];  
    end
    Model1_Accuracy(i,:) = model_accuracy_summary;
    
    %% Model 2: One ambiguity parameter common for C2-3-5-6-7-8
    npar = 4;
    
    %first fit params to subject data
    disp(['Sub' num2str(i) '- Model 2 fit'])
    params_rand=[]; 
    for i_rand=1:100*npar
        start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
        [paramtracker, lltracker, ex] = fminunc(@LL_function_Model2, start, opts, P);
        params_rand=[params_rand; paramtracker lltracker ex];
    end
    bad_fit = params_rand(:,npar+2)<=0;
    if sum(bad_fit)<100*npar
        params_rand(bad_fit,:)=[];
    end
    [~,ids] = sort(params_rand(:,npar+1));
    best_params = params_rand(ids(1),:);
    best_params(1) = 1/(1+exp(-best_params(1))); % mu [0 1]
    best_params(2) = 10/(1+exp(-best_params(2))); % lambda [0 10]
    best_params(3) = 3/(1+exp(-best_params(3)));  % rho [0 3]
    best_params(4) = 3/(1+exp(-best_params(4)));  % ambiguity parameter [0 3]
    BIC = 2*best_params(npar+1) + npar*log(tr_nb);
    AIC = 2*best_params(npar+1) + 2*npar;
    pseudoR2 = 1 + best_params(npar+1)/(log(0.5)*tr_nb);
    
    Model2_Params(i,:)        = best_params(1:npar); %save parameters used to generate data
    Model2_Loglikelihood(i,:) = best_params(npar+1);
    Model2_ExitFlag(i,:)      = best_params(npar+2);
    Model2_AIC(i,:)           = AIC;
    Model2_BIC(i,:)           = BIC;
    Model2_PseudoR2(i,:)      = pseudoR2;
    
    %parameter recovery analysis
    disp(['Sub' num2str(i) '- Model 2 Recovery'])
    params = best_params(1:npar);
    param_r = zeros(sim_nb,npar); %initiate list of recovered parameter
    for s=1:sim_nb
        P_pred = generate_choice_Model2(params,P); %generate choice set
        gen_ch = P_pred(:,4); %set of generated choices, column nb may need to be changed!
        P_new = P;
        P_new(:,8) = gen_ch;
        %fit model to generated choice set
        params_rand=[]; 
        for i_rand=1:20*npar
            start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
            [paramtracker, lltracker, ex] = fminunc(@LL_function_Model2, start, opts, P_new);
            params_rand=[params_rand; paramtracker lltracker ex];
        end
        bad_fit = params_rand(:,npar+2)<=0;
        if sum(bad_fit)<20*npar
            params_rand(bad_fit,:)=[];
        end
        [~,ids] = sort(params_rand(:,npar+1));
        best_params = params_rand(ids(1),:);
        best_params(1) = 1/(1+exp(-best_params(1))); % mu [0 1]
        best_params(2) = 10/(1+exp(-best_params(2))); % lambda [0 10]
        best_params(3) = 3/(1+exp(-best_params(3)));  % rho [0 3]
        best_params(4) = 3/(1+exp(-best_params(4)));  % ambiguity parameter [0 3]
        param_r(s,:) = best_params(1:npar);
    end
    Model2_RecovParams(i,:)   = mean(param_r,1);
    Model2_RecovStd(i,:)      = std(param_r,1);
    
    %test out of sample accuracy (6-fold leave-one-out)
    disp(['Sub' num2str(i) '- Model 2 Accuracy'])
    model_accuracy_summary = [];  
    grp = repmat((1:6)',56,1);
    P(:,9) = grp(1:tr_nb);
    for b = 1:6 %run leave-one out cross-validation for predictive accuracy
        tr_train_id = P(:,9)~=b;
        tr_test_id  = P(:,9)==b;
        P_train = P(tr_train_id,:); 
        P_test = P(tr_test_id,:); 
              
        params_rand=[]; 
        for i_rand=1:20*npar
            start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
            [paramtracker, lltracker, ex] = fminunc(@LL_function_Model2, start, opts, P_train);
            params_rand=[params_rand; paramtracker lltracker ex];
        end
        bad_fit = params_rand(:,npar+2)<=0;
        if sum(bad_fit)<20*npar
            params_rand(bad_fit,:)=[];
        end
        [~,ids] = sort(params_rand(:,npar+1));
        best_params = params_rand(ids(1),:);
        best_params(1) = 1/(1+exp(-best_params(1))); % mu [0 1]
        best_params(2) = 10/(1+exp(-best_params(2))); % lambda [0 10]
        best_params(3) = 3/(1+exp(-best_params(3)));  % rho [0 3]
        best_params(4) = 3/(1+exp(-best_params(4)));  % ambiguity parameter [0 3]
        %calculate model accuracy on test data using these params
        param = best_params(1:npar);
        model_accuracy_tmp = zeros(100,1);
        for s=1:100
            P_pred = generate_choice_Model2(param,P_test);
            model_accuracy_tmp(s) = nanmean(P_pred(:,6));
        end
        model_accuracy = mean(model_accuracy_tmp);
        model_accuracy_summary = [model_accuracy_summary model_accuracy];  
    end
    Model2_Accuracy(i,:) = model_accuracy_summary;

    %% Model 3: Separate ambiguity parameters for gains (C3-5-6-7) and losses (C2-8)
    npar = 5;
    
    %first fit params to subject data
    disp(['Sub' num2str(i) '- Model 3 fit'])
    params_rand=[]; 
    for i_rand=1:100*npar
        start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
        [paramtracker, lltracker, ex] = fminunc(@LL_function_Model3, start, opts, P);
        params_rand=[params_rand; paramtracker lltracker ex];
    end
    bad_fit = params_rand(:,npar+2)<=0;
    if sum(bad_fit)<100*npar
        params_rand(bad_fit,:)=[];
    end
    [~,ids] = sort(params_rand(:,npar+1));
    best_params = params_rand(ids(1),:);
    best_params(1) = 1/(1+exp(-best_params(1)));  % mu [0 1]
    best_params(2) = 10/(1+exp(-best_params(2))); % lambda [0 10]
    best_params(3) = 3/(1+exp(-best_params(3)));  % rho [0 3]
    best_params(4) = 3/(1+exp(-best_params(4)));  % ambiguity parameter gains [0 3]
    best_params(5) = 3/(1+exp(-best_params(5)));  % ambiguity parameter losses [0 3]
    BIC = 2*best_params(npar+1) + npar*log(tr_nb);
    AIC = 2*best_params(npar+1) + 2*npar;
    pseudoR2 = 1 + best_params(npar+1)/(log(0.5)*tr_nb);
    
    Model3_Params(i,:)        = best_params(1:npar); %save parameters used to generate data
    Model3_Loglikelihood(i,:) = best_params(npar+1);
    Model3_ExitFlag(i,:)      = best_params(npar+2);
    Model3_AIC(i,:)           = AIC;
    Model3_BIC(i,:)           = BIC;
    Model3_PseudoR2(i,:)      = pseudoR2;
    
    %parameter recovery analysis
    disp(['Sub' num2str(i) '- Model 3 Recovery'])
    params = best_params(1:npar);
    param_r = zeros(sim_nb,npar); %initiate list of recovered parameter
    for s=1:sim_nb
        P_pred = generate_choice_Model3(params,P); %generate choice set
        gen_ch = P_pred(:,4); %set of generated choices, column nb may need to be changed!
        P_new = P;
        P_new(:,8) = gen_ch;
        %fit model to generated choice set
        params_rand=[]; 
        for i_rand=1:20*npar
            start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
            [paramtracker, lltracker, ex] = fminunc(@LL_function_Model3, start, opts, P_new);
            params_rand=[params_rand; paramtracker lltracker ex];
        end
        bad_fit = params_rand(:,npar+2)<=0;
        if sum(bad_fit)<20*npar
            params_rand(bad_fit,:)=[];
        end
        [~,ids] = sort(params_rand(:,npar+1));
        best_params = params_rand(ids(1),:);
        best_params(1) = 1/(1+exp(-best_params(1))); % mu [0 1]
        best_params(2) = 10/(1+exp(-best_params(2))); % lambda [0 10]
        best_params(3) = 3/(1+exp(-best_params(3)));  % rho [0 3]
        best_params(4) = 3/(1+exp(-best_params(4)));  % ambiguity parameter gains [0 3]
        best_params(5) = 3/(1+exp(-best_params(5)));  % ambiguity parameter losses [0 3]
        param_r(s,:) = best_params(1:npar);
    end
    Model3_RecovParams(i,:)   = mean(param_r,1);
    Model3_RecovStd(i,:)      = std(param_r,1);
    
    %test out of sample accuracy (6-fold leave-one-out)
    disp(['Sub' num2str(i) '- Model 3 Accuracy'])
    model_accuracy_summary = [];  
    grp = repmat((1:6)',56,1);
    P(:,9) = grp(1:tr_nb);
    for b = 1:6 %run leave-one out cross-validation for predictive accuracy
        tr_train_id = P(:,9)~=b;
        tr_test_id  = P(:,9)==b;
        P_train = P(tr_train_id,:); 
        P_test = P(tr_test_id,:); 
              
        params_rand=[]; 
        for i_rand=1:20*npar
            start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
            [paramtracker, lltracker, ex] = fminunc(@LL_function_Model3, start, opts, P_train);
            params_rand=[params_rand; paramtracker lltracker ex];
        end
        bad_fit = params_rand(:,npar+2)<=0;
        if sum(bad_fit)<20*npar
            params_rand(bad_fit,:)=[];
        end
        [~,ids] = sort(params_rand(:,npar+1));
        best_params = params_rand(ids(1),:);
        best_params(1) = 1/(1+exp(-best_params(1))); % mu [0 1]
        best_params(2) = 10/(1+exp(-best_params(2))); % lambda [0 10]
        best_params(3) = 3/(1+exp(-best_params(3)));  % rho [0 3]
        best_params(4) = 3/(1+exp(-best_params(4)));  % ambiguity parameter gains [0 3]
        best_params(5) = 3/(1+exp(-best_params(5)));  % ambiguity parameter losses [0 3]
        %calculate model accuracy on test data using these params
        param = best_params(1:npar);
        model_accuracy_tmp = zeros(100,1);
        for s=1:100
            P_pred = generate_choice_Model3(param,P_test);
            model_accuracy_tmp(s) = nanmean(P_pred(:,6));
        end
        model_accuracy = mean(model_accuracy_tmp);
        model_accuracy_summary = [model_accuracy_summary model_accuracy];  
    end
    Model3_Accuracy(i,:) = model_accuracy_summary;
    
    %% Model 4: Separate ambiguity parameters depending on context: 
    % a loss is present (C2-3-8) vs no loss present (C5-6-7)
    npar = 5;
    
    %first fit params to subject data
    disp(['Sub' num2str(i) '- Model 4 fit'])
    params_rand=[]; 
    for i_rand=1:100*npar
        start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
        [paramtracker, lltracker, ex] = fminunc(@LL_function_Model4, start, opts, P);
        params_rand=[params_rand; paramtracker lltracker ex];
    end
    bad_fit = params_rand(:,npar+2)<=0;
    if sum(bad_fit)<100*npar
        params_rand(bad_fit,:)=[];
    end
    [~,ids] = sort(params_rand(:,npar+1));
    best_params = params_rand(ids(1),:);
    best_params(1) = 1/(1+exp(-best_params(1)));  % mu [0 1]
    best_params(2) = 10/(1+exp(-best_params(2))); % lambda [0 10]
    best_params(3) = 3/(1+exp(-best_params(3)));  % rho [0 3]
    best_params(4) = 3/(1+exp(-best_params(4)));  % ambiguity parameter gains [0 3]
    best_params(5) = 3/(1+exp(-best_params(5)));  % ambiguity parameter losses [0 3]
    BIC = 2*best_params(npar+1) + npar*log(tr_nb);
    AIC = 2*best_params(npar+1) + 2*npar;
    pseudoR2 = 1 + best_params(npar+1)/(log(0.5)*tr_nb);
    
    Model4_Params(i,:)        = best_params(1:npar); %save parameters used to generate data
    Model4_Loglikelihood(i,:) = best_params(npar+1);
    Model4_ExitFlag(i,:)      = best_params(npar+2);
    Model4_AIC(i,:)           = AIC;
    Model4_BIC(i,:)           = BIC;
    Model4_PseudoR2(i,:)      = pseudoR2;
    
    %parameter recovery analysis
    disp(['Sub' num2str(i) '- Model 4 Recovery'])
    params = best_params(1:npar);
    param_r = zeros(sim_nb,npar); %initiate list of recovered parameter
    for s=1:sim_nb
        P_pred = generate_choice_Model4(params,P); %generate choice set
        gen_ch = P_pred(:,4); %set of generated choices, column nb may need to be changed!
        P_new = P;
        P_new(:,8) = gen_ch;
        %fit model to generated choice set
        params_rand=[]; 
        for i_rand=1:20*npar
            start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
            [paramtracker, lltracker, ex] = fminunc(@LL_function_Model4, start, opts, P_new);
            params_rand=[params_rand; paramtracker lltracker ex];
        end
        bad_fit = params_rand(:,npar+2)<=0;
        if sum(bad_fit)<20*npar
            params_rand(bad_fit,:)=[];
        end
        [~,ids] = sort(params_rand(:,npar+1));
        best_params = params_rand(ids(1),:);
        best_params(1) = 1/(1+exp(-best_params(1))); % mu [0 1]
        best_params(2) = 10/(1+exp(-best_params(2))); % lambda [0 10]
        best_params(3) = 3/(1+exp(-best_params(3)));  % rho [0 3]
        best_params(4) = 3/(1+exp(-best_params(4)));  % ambiguity parameter gains [0 3]
        best_params(5) = 3/(1+exp(-best_params(5)));  % ambiguity parameter losses [0 3]
        param_r(s,:) = best_params(1:npar);
    end
    Model4_RecovParams(i,:)   = mean(param_r,1);
    Model4_RecovStd(i,:)      = std(param_r,1);
    
    %test out of sample accuracy (6-fold leave-one-out)
    disp(['Sub' num2str(i) '- Model 4 Accuracy'])
    model_accuracy_summary = [];  
    grp = repmat((1:6)',56,1);
    P(:,9) = grp(1:tr_nb);
    for b = 1:6 %run leave-one out cross-validation for predictive accuracy
        tr_train_id = P(:,9)~=b;
        tr_test_id  = P(:,9)==b;
        P_train = P(tr_train_id,:); 
        P_test = P(tr_test_id,:); 
              
        params_rand=[]; 
        for i_rand=1:20*npar
            start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
            [paramtracker, lltracker, ex] = fminunc(@LL_function_Model4, start, opts, P_train);
            params_rand=[params_rand; paramtracker lltracker ex];
        end
        bad_fit = params_rand(:,npar+2)<=0;
        if sum(bad_fit)<20*npar
            params_rand(bad_fit,:)=[];
        end
        [~,ids] = sort(params_rand(:,npar+1));
        best_params = params_rand(ids(1),:);
        best_params(1) = 1/(1+exp(-best_params(1))); % mu [0 1]
        best_params(2) = 10/(1+exp(-best_params(2))); % lambda [0 10]
        best_params(3) = 3/(1+exp(-best_params(3)));  % rho [0 3]
        best_params(4) = 3/(1+exp(-best_params(4)));  % ambiguity parameter gains [0 3]
        best_params(5) = 3/(1+exp(-best_params(5)));  % ambiguity parameter losses [0 3]
        %calculate model accuracy on test data using these params
        param = best_params(1:npar);
        model_accuracy_tmp = zeros(100,1);
        for s=1:100
            P_pred = generate_choice_Model4(param,P_test);
            model_accuracy_tmp(s) = nanmean(P_pred(:,6));
        end
        model_accuracy = mean(model_accuracy_tmp);
        model_accuracy_summary = [model_accuracy_summary model_accuracy];  
    end
    Model4_Accuracy(i,:) = model_accuracy_summary;
    
    %% Model 5: Separate ambiguity parameters for risky gain (C3-5), risky loss (C2), sure gain (C6-7) and sure loss (C8)
    npar = 7;
    
    %first fit params to subject data
    disp(['Sub' num2str(i) '- Model 5 fit'])
    params_rand=[]; 
    for i_rand=1:100*npar
        start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
        [paramtracker, lltracker, ex] = fminunc(@LL_function_Model5, start, opts, P);
        params_rand=[params_rand; paramtracker lltracker ex];
    end
    bad_fit = params_rand(:,npar+2)<=0;
    if sum(bad_fit)<100*npar
        params_rand(bad_fit,:)=[];
    end
    [~,ids] = sort(params_rand(:,npar+1));
    best_params = params_rand(ids(1),:);
    best_params(1) = 1/(1+exp(-best_params(1)));  % mu [0 1]
    best_params(2) = 10/(1+exp(-best_params(2))); % lambda [0 10]
    best_params(3) = 3/(1+exp(-best_params(3)));  % rho [0 3]
    best_params(4) = 3/(1+exp(-best_params(4)));  % ambiguity parameter risky gains [0 3]
    best_params(5) = 3/(1+exp(-best_params(5)));  % ambiguity parameter risky losses [0 3]
    best_params(6) = 3/(1+exp(-best_params(6)));  % ambiguity parameter sure gains [0 3]
    best_params(7) = 3/(1+exp(-best_params(7)));  % ambiguity parameter sure losses [0 3]
    BIC = 2*best_params(npar+1) + npar*log(tr_nb);
    AIC = 2*best_params(npar+1) + 2*npar;
    pseudoR2 = 1 + best_params(npar+1)/(log(0.5)*tr_nb);
    
    Model5_Params(i,:)        = best_params(1:npar); %save parameters used to generate data
    Model5_Loglikelihood(i,:) = best_params(npar+1);
    Model5_ExitFlag(i,:)      = best_params(npar+2);
    Model5_AIC(i,:)           = AIC;
    Model5_BIC(i,:)           = BIC;
    Model5_PseudoR2(i,:)      = pseudoR2;
    
    %parameter recovery analysis
    disp(['Sub' num2str(i) '- Model 5 Recovery'])
    params = best_params(1:npar);
    param_r = zeros(sim_nb,npar); %initiate list of recovered parameter
    for s=1:sim_nb
        P_pred = generate_choice_Model5(params,P); %generate choice set
        gen_ch = P_pred(:,4); %set of generated choices, column nb may need to be changed!
        P_new = P;
        P_new(:,8) = gen_ch;
        %fit model to generated choice set
        params_rand=[]; 
        for i_rand=1:20*npar
            start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
            [paramtracker, lltracker, ex] = fminunc(@LL_function_Model5, start, opts, P_new);
            params_rand=[params_rand; paramtracker lltracker ex];
        end
        bad_fit = params_rand(:,npar+2)<=0;
        if sum(bad_fit)<20*npar
            params_rand(bad_fit,:)=[];
        end
        [~,ids] = sort(params_rand(:,npar+1));
        best_params = params_rand(ids(1),:);
        best_params(1) = 1/(1+exp(-best_params(1))); % mu [0 1]
        best_params(2) = 10/(1+exp(-best_params(2))); % lambda [0 10]
        best_params(3) = 3/(1+exp(-best_params(3)));  % rho [0 3]
        best_params(4) = 3/(1+exp(-best_params(4)));  % ambiguity parameter risky gains [0 3]
        best_params(5) = 3/(1+exp(-best_params(5)));  % ambiguity parameter risky losses [0 3]
        best_params(6) = 3/(1+exp(-best_params(6)));  % ambiguity parameter sure gains [0 3]
        best_params(7) = 3/(1+exp(-best_params(7)));  % ambiguity parameter sure losses [0 3]
        param_r(s,:) = best_params(1:npar);
    end
    Model5_RecovParams(i,:)   = mean(param_r,1);
    Model5_RecovStd(i,:)      = std(param_r,1);
    
    %test out of sample accuracy (6-fold leave-one-out)
    disp(['Sub' num2str(i) '- Model 5 Accuracy'])
    model_accuracy_summary = [];  
    grp = repmat((1:6)',56,1);
    P(:,9) = grp(1:tr_nb);
    for b = 1:6 %run leave-one out cross-validation for predictive accuracy
        tr_train_id = P(:,9)~=b;
        tr_test_id  = P(:,9)==b;
        P_train = P(tr_train_id,:); 
        P_test = P(tr_test_id,:); 
              
        params_rand=[]; 
        for i_rand=1:20*npar
            start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
            [paramtracker, lltracker, ex] = fminunc(@LL_function_Model5, start, opts, P_train);
            params_rand=[params_rand; paramtracker lltracker ex];
        end
        bad_fit = params_rand(:,npar+2)<=0;
        if sum(bad_fit)<20*npar
            params_rand(bad_fit,:)=[];
        end
        [~,ids] = sort(params_rand(:,npar+1));
        best_params = params_rand(ids(1),:);
        best_params(1) = 1/(1+exp(-best_params(1))); % mu [0 1]
        best_params(2) = 10/(1+exp(-best_params(2))); % lambda [0 10]
        best_params(3) = 3/(1+exp(-best_params(3)));  % rho [0 3]
        best_params(4) = 3/(1+exp(-best_params(4)));  % ambiguity parameter risky gains [0 3]
        best_params(5) = 3/(1+exp(-best_params(5)));  % ambiguity parameter risky losses [0 3]
        best_params(6) = 3/(1+exp(-best_params(6)));  % ambiguity parameter sure gains [0 3]
        best_params(7) = 3/(1+exp(-best_params(7)));  % ambiguity parameter sure losses [0 3]
        %calculate model accuracy on test data using these params
        param = best_params(1:npar);
        model_accuracy_tmp = zeros(100,1);
        for s=1:100
            P_pred = generate_choice_Model5(param,P_test);
            model_accuracy_tmp(s) = nanmean(P_pred(:,6));
        end
        model_accuracy = mean(model_accuracy_tmp);
        model_accuracy_summary = [model_accuracy_summary model_accuracy];  
    end
    Model5_Accuracy(i,:) = model_accuracy_summary;
    
   

end

fitResult.Model1.Params        = Model1_Params;
fitResult.Model1.Loglikelihood = Model1_Loglikelihood;
fitResult.Model1.ExitFlag      = Model1_ExitFlag;
fitResult.Model1.AIC           = Model1_AIC;
fitResult.Model1.BIC           = Model1_BIC;
fitResult.Model1.PseudoR2      = Model1_PseudoR2;
fitResult.Model1.RecovParams   = Model1_RecovParams;
fitResult.Model1.RecovStd      = Model1_RecovStd;
fitResult.Model1.Accuracy      = Model1_Accuracy;
fitResult.Model1.MeanAccuracy  = mean(fitResult.Model1.Accuracy,2);

fitResult.Model2.Params        = Model2_Params;
fitResult.Model2.Loglikelihood = Model2_Loglikelihood;
fitResult.Model2.ExitFlag      = Model2_ExitFlag;
fitResult.Model2.AIC           = Model2_AIC;
fitResult.Model2.BIC           = Model2_BIC;
fitResult.Model2.PseudoR2      = Model2_PseudoR2;
fitResult.Model2.RecovParams   = Model2_RecovParams;
fitResult.Model2.RecovStd      = Model2_RecovStd;
fitResult.Model2.Accuracy      = Model2_Accuracy;
fitResult.Model2.MeanAccuracy  = mean(fitResult.Model2.Accuracy,2);

fitResult.Model3.Params        = Model3_Params;
fitResult.Model3.Loglikelihood = Model3_Loglikelihood;
fitResult.Model3.ExitFlag      = Model3_ExitFlag;
fitResult.Model3.AIC           = Model3_AIC;
fitResult.Model3.BIC           = Model3_BIC;
fitResult.Model3.PseudoR2      = Model3_PseudoR2;
fitResult.Model3.RecovParams   = Model3_RecovParams;
fitResult.Model3.RecovStd      = Model3_RecovStd;
fitResult.Model3.Accuracy      = Model3_Accuracy;
fitResult.Model3.MeanAccuracy  = mean(fitResult.Model3.Accuracy,2);

fitResult.Model4.Params        = Model4_Params;
fitResult.Model4.Loglikelihood = Model4_Loglikelihood;
fitResult.Model4.ExitFlag      = Model4_ExitFlag;
fitResult.Model4.AIC           = Model4_AIC;
fitResult.Model4.BIC           = Model4_BIC;
fitResult.Model4.PseudoR2      = Model4_PseudoR2;
fitResult.Model4.RecovParams   = Model4_RecovParams;
fitResult.Model4.RecovStd      = Model4_RecovStd;
fitResult.Model4.Accuracy      = Model4_Accuracy;
fitResult.Model4.MeanAccuracy  = mean(fitResult.Model4.Accuracy,2);

fitResult.Model5.Params        = Model5_Params;
fitResult.Model5.Loglikelihood = Model5_Loglikelihood;
fitResult.Model5.ExitFlag      = Model5_ExitFlag;
fitResult.Model5.AIC           = Model5_AIC;
fitResult.Model5.BIC           = Model5_BIC;
fitResult.Model5.PseudoR2      = Model5_PseudoR2;
fitResult.Model5.RecovParams   = Model5_RecovParams;
fitResult.Model5.RecovStd      = Model5_RecovStd;
fitResult.Model5.Accuracy      = Model5_Accuracy;
fitResult.Model5.MeanAccuracy  = mean(fitResult.Model5.Accuracy,2);

save('model_results_low_stakes.mat','fitResult')
