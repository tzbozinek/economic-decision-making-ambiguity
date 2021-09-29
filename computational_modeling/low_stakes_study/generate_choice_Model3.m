%-----------------Zbozinek*, Charpentier*, Qi, & Mobbs---------------------
% *co-first authors

% P is a "parameter" matrix, representing experimental design (gambles).  
% rows are trials, column 1 is trial number
% column 2 is block number, 
% column 3 is condition (trial type),
% column 4 is value of risky win, 
% column 5 is value of risky loss,
% column 6 is value of sure known option,
% column 8 is choice of gamble over sure option (for trial types 1 to 6) or
% choice of ambiguous option over known option (for trial types 7 and 8)

%list of parameters:
% x(1) = inverse temperature mu (>0)
% x(2) = loss aversion lambda (>0)
% x(3) = risk preference rho (>0)
% x(4) = ambiguity preference for gains (>0)
% x(5) = ambiguity aversion for losses (>0)

function Pgen = generate_choice_Model3(x,P)

Pch     = NaN(length(P),1); %probability of choice
U1      = NaN(length(P),1); %utility of option 1 (gamble for cond 1 to 6; known option for cond 7 and 8)
U2      = NaN(length(P),1); %utility of option 2 (sure option for cond 1 to 6; unknown option for cond 7 and 8)
pred_ch = NaN(length(P),1); %predicted choice
ll      = NaN(length(P),1); %log-likelihood
corr    = NaN(length(P),1); %choice predicted by model = subject's choice
ll_S    = NaN(length(P),1); %likelihood of predicting subject choice

%sum log likelihood for each observation (trial)
for i=1:length(P)
    
    if P(i,3) == 1 %trial type 1 = mixed gamble vs sure 0
        U1(i) = 0.5*P(i,4)^x(3) - 0.5*x(2)*(abs(P(i,5)))^x(3);
        U2(i) = 0;
    elseif P(i,3) == 2 %trial type 2 = mixed gamble with ambiguous risky loss vs sure 0
        U1(i) = 0.5*P(i,4)^x(3) - 0.5*x(2)*x(5)*41^x(3); %41 = mean loss value for conditions 1 and 3
        U2(i) = 0;
    elseif P(i,3) == 3 %trial type 3 = mixed gamble with ambiguous risky gain vs sure 0
        U1(i) = 0.5*x(4)*71^x(3) - 0.5*x(2)*(abs(P(i,5)))^x(3); %71 = mean gain value for conditions 1 and 2
        U2(i) = 0;
    elseif P(i,3) == 4 %trial type 4 = gain-only gamble vs sure gain
        U1(i) = 0.5*P(i,4)^x(3);
        U2(i) = P(i,6)^x(3);
    elseif P(i,3) == 5 %trial type 5 = gain-only gamble with ambiguous risky gain vs sure gain
        U1(i) = 0.5*x(4)*75^x(3); %75 = mean gain value for conditions 4 and 6
        U2(i) = P(i,6)^x(3);
    elseif P(i,3) == 6 %trial type 6 = gain-only gamble vs sure ambiguous gain
        U1(i) = 0.5*P(i,4)^x(3);
        U2(i) = x(4)*28^x(3); %28 = mean loss value for conditions 4 and 5
    elseif P(i,3) == 7 %trial type 7 = sure ambiguous gain vs sure known gain
        U1(i) = x(4)*35^x(3); %35 = mean gain value for condition 7 
        U2(i) = P(i,6)^x(3);
    elseif P(i,3) == 8 %trial type 8 = sure ambiguous loss vs sure known loss
        U1(i) = -x(2)*x(5)*35^x(3); %35 = mean loss value for condition 8
        U2(i) = -x(2)*(abs(P(i,6)))^x(3); 
    end

    xb = x(1)*(U1(i) - U2(i));

    if xb<-709
      xb=-709;
    end
    if xb>36
      xb=36;
    end
    
    Pch(i) = (1+exp(-1*xb))^-1;
    
    n=rand();
    if n<Pch(i)
        pred_ch(i) = 1; %predicted choice is gamble (over sure option) or ambiguous option (over known option)
        ll(i) = Pch(i); %loglikelihood
    else
        pred_ch(i) = 0; %predicted choice is sure option (over gamble) or known option (over ambiguous option)
        ll(i) = 1-Pch(i); %loglikelihood
    end 
    if pred_ch(i) == P(i,8) %if predicted choice is subject's choice
        corr(i) = 1;
    else
        corr(i) = 0;
    end
    if P(i,8)==1 %calculate likelihood of predicting subject choice
        ll_S(i) = Pch(i);
    elseif P(i,8)==0
        ll_S(i) = 1-Pch(i);
    end

end
Pgen = [U1 U2 Pch pred_ch ll corr ll_S];
