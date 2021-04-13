%-----------------Zbozinek*, Charpentier*, Qi*, & Mobbs---------------------
% *co-first authors

clear all
close all

fs = filesep;

data_dir = pwd
cd(data_dir)

fname = 'high_stakes_data.xlsx';
[numd,txtd,rawd] = xlsread(fname);

% init the randomization screen
RandStream.setGlobalStream(RandStream('mt19937ar','Seed','shuffle'));

nsub = 210; %number of subjects  

%load data and models to add to
load('model_results_high_stakes.mat')

for s=1:nsub
    
    isub = numd(:,1) == s;
    P = numd(isub,[9 2:8]);
    missed = isnan(P(:,8));
    P(missed,:) = [];
    tr_nb = length(P(:,1));
    
    % first, test a model assuming P(gamble)=50%
    LL_null = -log(0.5)*tr_nb;
    fitResult.NullModel1.Loglikelihood(s,1) = LL_null;
    fitResult.NullModel1.AIC(s,1) = 2*LL_null;
    fitResult.NullModel1.BIC(s,1) = 2*LL_null;
    
    %second, test a model assuming P(gamble) = mean(P(gamble)) for each subject
    mean_Pg = mean(P(:,8));
    fitResult.NullModel2.MeanPg(s,1) = mean_Pg;
    lk = NaN(tr_nb,1);
    for t=1:tr_nb
        if P(t,8) == 1
            lk(t) = mean_Pg;
        elseif P(t,8) == 0
            lk(t) = 1 - mean_Pg;
        end
    end
    LL_null=-sum(log(lk));
    fitResult.NullModel2.Loglikelihood(s,1) = LL_null;
    fitResult.NullModel2.AIC(s,1) = 2*LL_null;
    fitResult.NullModel2.BIC(s,1) = 2*LL_null;
    fitResult.NullModel2.PseudoR2(s,1) = 1 + LL_null/(log(0.5)*tr_nb);
    if mean_Pg >= 0.5
        fitResult.NullModel2.Accuracy(s,1) = mean_Pg;
    else
        fitResult.NullModel2.Accuracy(s,1) = 1-mean_Pg;
    end
    
end

save('model_results_high_stakes.mat', 'fitResult')