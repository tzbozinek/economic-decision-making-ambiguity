***Gambling Propensity

use "data_wide.dta"

ttest Study1_GamblePercentCond1 == Study2_GamblePercentCond1
ttest Study1_GamblePercentCond2 == Study2_GamblePercentCond2
ttest Study1_GamblePercentCond3 == Study2_GamblePercentCond3
ttest Study1_GamblePercentCond4 == Study2_GamblePercentCond4
ttest Study1_GamblePercentCond5 == Study2_GamblePercentCond5
ttest Study1_GamblePercentCond6 == Study2_GamblePercentCond6
ttest Study1_AmbiguousPercentCond7 == Study2_AmbiguousPercentCond7
ttest Study1_AmbiguousPercentCond8 == Study2_AmbiguousPercentCond8

***Model 5 Real vs Recovered Parameters

use "low_stakes_model5_real_vs_recovered.dta"
use "high_stakes_model5_real_vs_recovered.dta"

pwcorr Real_C1 Recovered_C1, sig
pwcorr Real_C2 Recovered_C2, sig
pwcorr Real_C3 Recovered_C3, sig
pwcorr Real_C4 Recovered_C4, sig
pwcorr Real_C5 Recovered_C5, sig
pwcorr Real_C6 Recovered_C6, sig
pwcorr Real_C7 Recovered_C7, sig
pwcorr Real_C8 Recovered_C8, sig


***Model 5 Parameters and Gambling Propensity

use "data_wide.dta"

	**Loss Aversion
	pwcorr Study1_LossAversion Study1_GamblePercentCond1, sig
	pwcorr Study1_LossAversion Study1_GamblePercentCond2, sig
	pwcorr Study1_LossAversion Study1_GamblePercentCond3, sig
	pwcorr Study2_LossAversion Study2_GamblePercentCond1, sig
	pwcorr Study2_LossAversion Study2_GamblePercentCond2, sig
	pwcorr Study2_LossAversion Study2_GamblePercentCond3, sig
	
	pwcorr Study1_LossAversion Study1_AmbiguousPercentCond8, sig
	pwcorr Study2_LossAversion Study2_AmbiguousPercentCond8, sig
	
	**Ambiguous Sure Loss Aversion
	pwcorr Study1_SureLossAversion Study1_AmbiguousPercentCond8, sig
	pwcorr Study2_SureLossAversion Study2_AmbiguousPercentCond8, sig

	**Ambiguous Risky Loss Aversion
	pwcorr Study1_RiskyLossAversion Study1_GamblePercentCond2, sig
	pwcorr Study2_RiskyLossAversion Study2_GamblePercentCond2, sig
	
	**Risk Preference
	pwcorr Study1_RiskPreference Study1_GamblePercentCond1, sig
	pwcorr Study1_RiskPreference Study1_GamblePercentCond2, sig
	pwcorr Study1_RiskPreference Study1_GamblePercentCond3, sig
	pwcorr Study1_RiskPreference Study1_GamblePercentCond4, sig
	pwcorr Study1_RiskPreference Study1_GamblePercentCond5, sig
	pwcorr Study1_RiskPreference Study1_GamblePercentCond6, sig
	
	pwcorr Study2_RiskPreference Study2_GamblePercentCond1, sig
	pwcorr Study2_RiskPreference Study2_GamblePercentCond2, sig
	pwcorr Study2_RiskPreference Study2_GamblePercentCond3, sig
	pwcorr Study2_RiskPreference Study2_GamblePercentCond4, sig
	pwcorr Study2_RiskPreference Study2_GamblePercentCond5, sig
	pwcorr Study2_RiskPreference Study2_GamblePercentCond6, sig
	
	**Ambiguous Risky Gain Preference
	pwcorr Study1_RiskyGainPreference Study1_GamblePercentCond3, sig
	pwcorr Study1_RiskyGainPreference Study1_GamblePercentCond5, sig
	pwcorr Study2_RiskyGainPreference Study2_GamblePercentCond3, sig
	pwcorr Study2_RiskyGainPreference Study2_GamblePercentCond5, sig
	
	**Ambiguous Sure Gain Preference
	pwcorr Study1_SureGainPreference Study1_AmbiguousPercentCond7, sig
	pwcorr Study1_SureGainPreference Study1_GamblePercentCond6, sig
	pwcorr Study2_SureGainPreference Study2_AmbiguousPercentCond7, sig
	pwcorr Study2_SureGainPreference Study2_GamblePercentCond6, sig
	


***Model 5 - Low vs High Stakes

use "data_wide.dta"
	
	**Loss Aversion
	ttest Study1_LossAversion == 1
	ttest Study2_LossAversion == 1
	ttest Study2_LossAversion == Study1_LossAversion
	
	**Ambiguous Risky Loss Aversion
	ttest Study1_RiskyLossAversion == 1
	ttest Study2_RiskyLossAversion == 1
	ttest Study2_RiskyLossAversion == Study1_RiskyLossAversion
	
	**Ambiguous Sure Loss Aversion
	ttest Study1_SureLossAversion == 1
	ttest Study2_SureLossAversion == 1
	ttest Study2_SureLossAversion == Study1_SureLossAversion
	
	**Risk Preference
	ttest Study1_RiskPreference == 1
	ttest Study2_RiskPreference == 1
	ttest Study2_RiskPreference == Study1_RiskPreference
	
	**Ambiguous Risky Gain Preference
	ttest Study1_RiskyGainPreference == 1
	ttest Study2_RiskyGainPreference == 1
	ttest Study2_RiskyGainPreference == Study1_RiskyGainPreference
	
	**Ambiguous Sure Gain Preference
	ttest Study1_SureGainPreference == 1
	ttest Study2_SureGainPreference == 1
	ttest Study2_SureGainPreference == Study1_SureGainPreference
	
	**Condition6 Gambling Percentage
	ttest Study1_GamblePercentCond6 == 50
	ttest Study2_GamblePercentCond6 == 50
	ttest Study2_GamblePercentCond6 == Study1_GamblePercentCond6


*** Anxiety and Depression
use "data_wide.dta"

ttest Study1_AnxietyComposite == Study1_DepressionComposite
ttest Study2_AnxietyComposite == Study2_DepressionComposite
	
***Model 5 and Trait Anxiety/Depression

use "data_wide.dta"
use "data_wide_study1_careless_dropped.dta"
use "data_wide_study2_careless_dropped.dta"
	
	*Filter out participants who responded less consistently to anxiety/depression questionnaires. This will drop them from the dataset.
	
		*Study 1
		use "data_wide.dta"
		keep if Study1_CarelessR_Z2 == 0
		save "data_wide_study1_careless_dropped.dta", replace
		
	
		*Study 2
		use "data_wide.dta"
		keep if Study2_CarelessR_Z2 == 0
		save "data_wide_study2_careless_dropped.dta", replace
		
	

	**Anxiety
	mixed Study1_LossAversion        c.Study1_AnxietyComposite
	mixed Study1_RiskyLossAversion   c.Study1_AnxietyComposite
	mixed Study1_SureLossAversion    c.Study1_AnxietyComposite
	mixed Study1_RiskPreference      c.Study1_AnxietyComposite
	mixed Study1_RiskyGainPreference c.Study1_AnxietyComposite
	mixed Study1_SureGainPreference  c.Study1_AnxietyComposite
	mixed Study1_GamblePercentCond6  c.Study1_AnxietyComposite
	
	mixed Study2_LossAversion        c.Study2_AnxietyComposite
	mixed Study2_RiskyLossAversion   c.Study2_AnxietyComposite
	mixed Study2_SureLossAversion    c.Study2_AnxietyComposite
	mixed Study2_RiskPreference      c.Study2_AnxietyComposite
	mixed Study2_RiskyGainPreference c.Study2_AnxietyComposite
	mixed Study2_SureGainPreference  c.Study2_AnxietyComposite
	mixed Study2_GamblePercentCond6  c.Study2_AnxietyComposite
	
	**Depression
	mixed Study1_LossAversion        c.Study1_DepressionComposite
	mixed Study1_RiskyLossAversion   c.Study1_DepressionComposite
	mixed Study1_SureLossAversion    c.Study1_DepressionComposite
	mixed Study1_RiskPreference      c.Study1_DepressionComposite
	mixed Study1_RiskyGainPreference c.Study1_DepressionComposite
	mixed Study1_SureGainPreference  c.Study1_DepressionComposite
	mixed Study1_GamblePercentCond6  c.Study1_DepressionComposite
	
	mixed Study2_LossAversion        c.Study2_DepressionComposite
	mixed Study2_RiskyLossAversion   c.Study2_DepressionComposite
	mixed Study2_SureLossAversion    c.Study2_DepressionComposite
	mixed Study2_RiskPreference      c.Study2_DepressionComposite
	mixed Study2_RiskyGainPreference c.Study2_DepressionComposite
	mixed Study2_SureGainPreference  c.Study2_DepressionComposite
	mixed Study2_GamblePercentCond6  c.Study2_DepressionComposite
