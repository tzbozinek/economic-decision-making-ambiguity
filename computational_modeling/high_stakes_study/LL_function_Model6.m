%-----------------Zbozinek*, Charpentier*, Qi*, & Mobbs---------------------
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
% x(4) = ambiguity aversion COND 2 (>0)
% x(5) = ambiguity preference COND 3 (>0)
% x(6) = ambiguity preference COND 5 (>0)
% x(7) = ambiguity preference COND 6 (>0)
% x(8) = ambiguity preference COND 7 (>0)
% x(9) = ambiguity aversion COND 8 (>0)


      
function f = LL_function_Model6(x,P)

%transform parameters to make sure they are constrained between values that
%make sense, for ex
x(1) = 5/(1+exp(-x(1))); % mu [0 5]
x(2) = 10/(1+exp(-x(2))); % lambda [0 10]
x(3) = 3/(1+exp(-x(3)));  % rho [0 3]
x(4) = 3/(1+exp(-x(4)));  % ambiguity aversion COND 2 [0 3]
x(5) = 3/(1+exp(-x(5)));  % ambiguity preference COND 3 [0 3]
x(6) = 3/(1+exp(-x(6)));  % ambiguity preference COND 5 [0 3]
x(7) = 3/(1+exp(-x(7)));  % ambiguity preference COND 6 [0 3]
x(8) = 3/(1+exp(-x(8)));  % ambiguity preference COND 7 [0 3]
x(9) = 3/(1+exp(-x(9)));  % ambiguity aversion COND 8 [0 3]

Pch = zeros(length(P),1);

%sum log likelihood for each observation (trial)
for i=1:length(P)
    
    if P(i,3) == 1 %trial type 1 = mixed gamble vs sure 0
        U1 = 0.5*P(i,4)^x(3) - 0.5*x(2)*(abs(P(i,5)))^x(3);
        U2 = 0;
    elseif P(i,3) == 2 %trial type 2 = mixed gamble with ambiguous risky loss vs sure 0
        U1 = 0.5*P(i,4)^x(3) - 0.5*x(2)*x(4)*8.15^x(3); %8.15 = mean loss value for conditions 1 and 3
        U2 = 0;
    elseif P(i,3) == 3 %trial type 3 = mixed gamble with ambiguous risky gain vs sure 0
        U1 = 0.5*x(5)*14.15^x(3) - 0.5*x(2)*(abs(P(i,5)))^x(3); %14.15 = mean gain value for conditions 1 and 2
        U2 = 0;
    elseif P(i,3) == 4 %trial type 4 = gain-only gamble vs sure gain
        U1 = 0.5*P(i,4)^x(3);
        U2 = P(i,6)^x(3);
    elseif P(i,3) == 5 %trial type 5 = gain-only gamble with ambiguous risky gain vs sure gain
        U1 = 0.5*x(6)*15^x(3); %15 = mean risky gain value for conditions 4 and 6
        U2 = P(i,6)^x(3);
    elseif P(i,3) == 6 %trial type 6 = gain-only gamble vs sure ambiguous gain
        U1 = 0.5*P(i,4)^x(3);
        U2 = x(7)*5.50^x(3); %5.50 = mean sure gain value for conditions 4 and 5
    elseif P(i,3) == 7 %trial type 7 = sure ambiguous gain vs sure known gain
        U1 = x(8)*7.00^x(3); %7.00 = mean gain value for condition 7 
        U2 = P(i,6)^x(3);
    elseif P(i,3) == 8 %trial type 8 = sure ambiguous loss vs sure known loss
        U1 = -x(2)*x(9)*7.00^x(3); %7.00 = mean loss value for condition 8
        U2 = -x(2)*(abs(P(i,6)))^x(3); 
    end


    xb = x(1)*(U1 - U2);

    if xb<-709
      xb=-709;
    end
    if xb>36
      xb=36;
    end
    
    %if choice value is 1, use one part of likelihood contribution.
    if(P(i,8)==1)
        Pch(i) = (1+exp(-1*xb))^-1;                  
    %if choice value is 0, use other part of likelihood contribution    
    elseif(P(i,8)==0)
        Pch(i) = 1-(1+exp(-1*xb))^-1;          
    end
end

f=-sum(log(Pch)); %negative value of loglikelihood
