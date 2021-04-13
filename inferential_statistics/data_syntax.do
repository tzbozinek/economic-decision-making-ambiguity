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

***Model 6 Real vs Recovered Parameters

use "low_stakes_model6_real_vs_recovered.dta"
use "high_stakes_model6_real_vs_recovered.dta"

pwcorr Real_C1 Recovered_C1, sig
pwcorr Real_C2 Recovered_C2, sig
pwcorr Real_C3 Recovered_C3, sig
pwcorr Real_C4 Recovered_C4, sig
pwcorr Real_C5 Recovered_C5, sig
pwcorr Real_C6 Recovered_C6, sig
pwcorr Real_C7 Recovered_C7, sig
pwcorr Real_C8 Recovered_C8, sig


***Model 6 Parameters and Gambling Propensity

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
	
	**Condition8 Ambiguous Sure Loss Aversion
	pwcorr Study1_C8Aversion Study1_AmbiguousPercentCond8, sig
	pwcorr Study2_C8Aversion Study2_AmbiguousPercentCond8, sig

	**Condition2 Ambiguous Risky Loss Aversion
	pwcorr Study1_C2Aversion Study1_GamblePercentCond2, sig
	pwcorr Study2_C2Aversion Study2_GamblePercentCond2, sig
	
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
	
	**Condition3 Ambiguous Risky Gain Preference
	pwcorr Study1_C3Preference Study1_GamblePercentCond3, sig
	pwcorr Study2_C3Preference Study2_GamblePercentCond3, sig
	
	**Condition5 Ambiguous Risky Gain Preference
	pwcorr Study1_C5Preference Study1_GamblePercentCond5, sig
	pwcorr Study2_C5Preference Study2_GamblePercentCond5, sig
	
	**Condition7 Ambiguous Sure Gain Preference
	pwcorr Study1_C7Preference Study1_AmbiguousPercentCond7, sig
	pwcorr Study2_C7Preference Study2_AmbiguousPercentCond7, sig
	
	**Condition6 Ambiguous Sure Gain Preference
	pwcorr Study1_C6Preference Study1_GamblePercentCond6, sig
	pwcorr Study2_C6Preference Study2_GamblePercentCond6, sig


***Model 6 - Low vs High Stakes

use "data_wide.dta"
	
	**Loss Aversion
	ttest Study1_LossAversion == 1
	ttest Study2_LossAversion == 1
	ttest Study2_LossAversion == Study1_LossAversion
	
	**C2_Aversion
	ttest Study1_C2Aversion == 1
	ttest Study2_C2Aversion == 1
	ttest Study2_C2Aversion == Study1_C2Aversion
	
	**C8_Aversion
	ttest Study1_C8Aversion == 1
	ttest Study2_C8Aversion == 1
	ttest Study2_C8Aversion == Study1_C8Aversion
	
	**Risk Preference
	ttest Study1_RiskPreference == 1
	ttest Study2_RiskPreference == 1
	ttest Study2_RiskPreference == Study1_RiskPreference
	
	**C3Preference
	ttest Study1_C3Preference == 1
	ttest Study2_C3Preference == 1
	ttest Study2_C3Preference == Study1_C3Preference
	
	**C5Preference
	ttest Study1_C5Preference == 1
	ttest Study2_C5Preference == 1
	ttest Study2_C5Preference == Study1_C5Preference
	
	**C6Preference
	ttest Study1_C6Preference == 1
	ttest Study2_C6Preference == 1
	ttest Study2_C6Preference == Study1_C6Preference
	
	**C7Preference
	ttest Study1_C7Preference == 1
	ttest Study2_C7Preference == 1
	ttest Study2_C7Preference == Study1_C7Preference
	
	**Condition6 Gambling Percentage
	ttest Study1_GamblePercentCond6 == 50
	ttest Study2_GamblePercentCond6 == 50
	ttest Study2_GamblePercentCond6 == Study1_GamblePercentCond6



***Model 6 and Trait Anxiety/Depression

use "data_wide.dta"

	**Anxiety
	mixed Study1_LossAversion        c.Study1_AnxietyComposite
	mixed Study1_C2Aversion          c.Study1_AnxietyComposite
	mixed Study1_C8Aversion          c.Study1_AnxietyComposite
	mixed Study1_RiskPreference      c.Study1_AnxietyComposite
	mixed Study1_C3Preference        c.Study1_AnxietyComposite
	mixed Study1_C5Preference        c.Study1_AnxietyComposite
	mixed Study1_C6Preference        c.Study1_AnxietyComposite
	mixed Study1_C7Preference        c.Study1_AnxietyComposite
	mixed Study1_GamblePercentCond6  c.Study1_AnxietyComposite
	
	mixed Study2_LossAversion        c.Study2_AnxietyComposite
	mixed Study2_C2Aversion          c.Study2_AnxietyComposite
	mixed Study2_C8Aversion          c.Study2_AnxietyComposite
	mixed Study2_RiskPreference      c.Study2_AnxietyComposite
	mixed Study2_C3Preference        c.Study2_AnxietyComposite
	mixed Study2_C5Preference        c.Study2_AnxietyComposite
	mixed Study2_C6Preference        c.Study2_AnxietyComposite
	mixed Study2_C7Preference        c.Study2_AnxietyComposite
	mixed Study2_GamblePercentCond6  c.Study2_AnxietyComposite
	
	**Depression
	mixed Study1_LossAversion        c.Study1_DepressionComposite
	mixed Study1_C2Aversion          c.Study1_DepressionComposite
	mixed Study1_C8Aversion          c.Study1_DepressionComposite
	mixed Study1_RiskPreference      c.Study1_DepressionComposite
	mixed Study1_C3Preference        c.Study1_DepressionComposite
	mixed Study1_C5Preference        c.Study1_DepressionComposite
	mixed Study1_C6Preference        c.Study1_DepressionComposite
	mixed Study1_C7Preference        c.Study1_DepressionComposite
	mixed Study1_GamblePercentCond6  c.Study1_DepressionComposite
	
	mixed Study2_LossAversion        c.Study2_DepressionComposite
	mixed Study2_C2Aversion          c.Study2_DepressionComposite
	mixed Study2_C8Aversion          c.Study2_DepressionComposite
	mixed Study2_RiskPreference      c.Study2_DepressionComposite
	mixed Study2_C3Preference        c.Study2_DepressionComposite
	mixed Study2_C5Preference        c.Study2_DepressionComposite
	mixed Study2_C6Preference        c.Study2_DepressionComposite
	mixed Study2_C7Preference        c.Study2_DepressionComposite
	mixed Study2_GamblePercentCond6  c.Study2_DepressionComposite