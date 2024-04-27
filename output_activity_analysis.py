import os
os.chdir('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/task-interference')
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import statsmodels.api as sm

#%% Load model outputs and trial parameters
mod1_name = 'MM1_monkeyB1245'
path1 = f'./monkey_choice_model/test_data/{mod1_name}_allinds_noisevis0.8mem0.5rec0.1'
mod1_outputs = pickle.load(open(path1 + '_modeloutput.pickle', 'rb'))
mod1_choices = np.argmax(mod1_outputs[:, -1, 2:6], axis=1) + 1
mod1_trialparams = pickle.load(open(path1 + '_trialparams.pickle', 'rb'))
mod1_trial_inds = [mod1_trialparams[i]['trial_ind'] for i in range(mod1_trialparams.shape[0])]
mod1_dsl = np.array([mod1_trialparams[i]['dsl'][-1] for i in range(mod1_trialparams.shape[0])])
mod1_dsf = np.array([mod1_trialparams[i]['dsf'][-1] for i in range(mod1_trialparams.shape[0])])

mod2_name = 'SH2_correctA'
path2 = f'./correct_choice_model/test_data/{mod2_name}_monkeyhist_allinds_noisevis0.8mem0.5rec0.1'
mod2_outputs = pickle.load(open(path2 + '_modeloutput.pickle', 'rb'))
mod2_choices = np.argmax(mod2_outputs[:, -1, 2:6], axis=1) + 1
mod2_trialparams = pickle.load(open(path2 + '_trialparams.pickle', 'rb'))
mod2_trial_inds = [mod2_trialparams[i]['trial_ind'] for i in range(mod2_trialparams.shape[0])]
mod2_dsl = np.array([mod2_trialparams[i]['dsl'][-1] for i in range(mod2_trialparams.shape[0])])
mod2_dsf = np.array([mod2_trialparams[i]['dsf'][-1] for i in range(mod2_trialparams.shape[0])])

assert np.array_equal(mod1_trial_inds, mod2_trial_inds)
aNR = np.array([mod1_trialparams[i]['choice'][-2] != mod1_trialparams[i]['correct'][-2] \
                for i in range(mod1_trialparams.shape[0])])
aNR_inds = np.where(aNR)[0]
aR_inds = np.where(~aNR)[0]

# %% Compute output differences
mod1_SLtaskout = mod1_outputs[..., 0]
mod1_SFtaskout = mod1_outputs[..., 1]
mod1_taskout_diff = mod1_SLtaskout - mod1_SFtaskout

mod1_SFpercout_diff = mod1_outputs[..., 2] - mod1_outputs[..., 3] # SF inc - SF dec
mod1_SLpercout_diff = mod1_outputs[..., 5] - mod1_outputs[..., 4] # SL inc - SL dec

mod1_SLinds = np.where(np.isin(mod1_choices, [3, 4]))[0]
mod1_SFinds = np.where(np.isin(mod1_choices, [1, 2]))[0]

mod2_SLtaskout = mod2_outputs[..., 0]
mod2_SFtaskout = mod2_outputs[..., 1]
mod2_taskout_diff = mod2_SLtaskout - mod2_SFtaskout

mod2_SFpercout_diff = mod2_outputs[..., 2] - mod2_outputs[..., 3] # SF inc - SF dec
mod2_SLpercout_diff = mod2_outputs[..., 5] - mod2_outputs[..., 4] # SL inc - SL dec

mod2_SLinds = np.where(np.isin(mod2_choices, [3, 4]))[0]
mod2_SFinds = np.where(np.isin(mod2_choices, [1, 2]))[0]

# %% Linear regressions

regdata_mod1 = pd.DataFrame({'SLpercoutdiff' : mod1_SLpercout_diff[:, -1], 
                             'SFpercoutdiff' : mod1_SFpercout_diff[:, -1],
                             'dsl' : mod1_dsl, 'dsf' : mod1_dsf, 
                             'taskoutdiff' : mod1_taskout_diff[:, -1]})
# Add interaction terms
regdata_mod1['dsl_dsf'] = regdata_mod1['dsl'] * regdata_mod1['dsf']
regdata_mod1['dsl_taskoutdiff'] = regdata_mod1['dsl'] * regdata_mod1['taskoutdiff']
regdata_mod1['dsf_taskoutdiff'] = regdata_mod1['dsf'] * regdata_mod1['taskoutdiff']

regdata_mod2 = pd.DataFrame({'SLpercoutdiff' : mod2_SLpercout_diff[:, -1], 
                             'SFpercoutdiff' : mod2_SFpercout_diff[:, -1], 
                             'dsl' : mod2_dsl, 'dsf' : mod2_dsf, 
                             'taskoutdiff' : mod2_taskout_diff[:, -1]})
# Add interaction terms
regdata_mod2['dsl_dsf'] = regdata_mod2['dsl'] * regdata_mod2['dsf']
regdata_mod2['dsl_taskoutdiff'] = regdata_mod2['dsl'] * regdata_mod2['taskoutdiff']
regdata_mod2['dsf_taskoutdiff'] = regdata_mod2['dsf'] * regdata_mod2['taskoutdiff']

# SL
X_mod1SL = regdata_mod1.loc[mod1_SLinds, 'dsl':]
X_mod1SL = sm.add_constant(X_mod1SL)
y_mod1SL = regdata_mod1.loc[mod1_SLinds, 'SLpercoutdiff']
model_mod1SL = sm.OLS(y_mod1SL, X_mod1SL).fit()
print('mod1SL', model_mod1SL.summary())

X_mod2SL = regdata_mod2.loc[mod2_SLinds, 'dsl':]
X_mod2SL = sm.add_constant(X_mod2SL)
y_mod2SL = regdata_mod2.loc[mod2_SLinds, 'SLpercoutdiff']
model_mod2SL = sm.OLS(y_mod2SL, X_mod2SL).fit()
print('mod2SL', model_mod2SL.summary())

# SF
X_mod1SF = regdata_mod1.loc[mod1_SFinds, 'dsl':]
X_mod1SF = sm.add_constant(X_mod1SF)
y_mod1SF = regdata_mod1.loc[mod1_SFinds, 'SFpercoutdiff']
model_mod1SF = sm.OLS(y_mod1SF, X_mod1SF).fit()
print('mod1SF', model_mod1SF.summary())

X_mod2SF = regdata_mod2.loc[mod2_SFinds, 'dsl':]
X_mod2SF = sm.add_constant(X_mod2SF)
y_mod2SF = regdata_mod2.loc[mod2_SFinds, 'SFpercoutdiff']
model_mod2SF = sm.OLS(y_mod2SF, X_mod2SF).fit()
print('mod2SF', model_mod2SF.summary())

#%% Plot coefficients with 95% CI. NOTE: Using default CI, not bootstrapped.
n_params = 7
colors = ['m', 'c']
xticklabels = ['intercept', '$\Delta L$', '$\Delta F$', '$TB$', 
               '$\Delta L x \Delta F$', '$\Delta L x TB$', 
               '$\Delta F x TB$']
f, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

# SL
ax[0,0].bar(np.arange(n_params), model_mod1SL.params, 
        yerr=0.5*(model_mod1SL.conf_int()[1] - model_mod1SL.conf_int()[0]),
        label=f'$R^2={model_mod1SL.rsquared_adj:.2f}$', 
        alpha=0.5, capsize=5, color=colors[0], ecolor=colors[0])

ax[0,1].bar(np.arange(n_params), model_mod2SL.params,
        yerr=0.5*(model_mod2SL.conf_int()[1] - model_mod2SL.conf_int()[0]),
        label=f'$R^2={model_mod2SL.rsquared_adj:.2f}$', 
        alpha=0.5, capsize=5, color=colors[0], ecolor=colors[0])

# SF
ax[1,0].bar(np.arange(n_params), model_mod1SF.params, 
        yerr=0.5*(model_mod1SF.conf_int()[1] - model_mod1SF.conf_int()[0]),
        label=f'$R^2={model_mod1SF.rsquared_adj:.2f}$', 
        alpha=0.5, capsize=5, color=colors[1], ecolor=colors[1])

ax[1,1].bar(np.arange(n_params), model_mod2SF.params,
        yerr=0.5*(model_mod2SF.conf_int()[1] - model_mod2SF.conf_int()[0]),
        label=f'$R^2={model_mod2SF.rsquared_adj:.2f}$', 
        alpha=0.5, capsize=5, color=colors[1], ecolor=colors[1])

ax[0,0].set_ylabel('Coefficient value')
ax[0,0].set_title(f'{mod1_name}')
ax[0,1].set_title(f'{mod2_name}')
for i in range(2):
    for j in range(2):
        ax[i,j].set_xticks(np.arange(n_params), xticklabels, rotation=45)
        ax[i,j].legend()

plt.tight_layout()

fig_folder = '/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/Figures'
plt.savefig(fig_folder + f'{mod1_name}v{mod2_name}_SLSFpercoutdiff_LRcoeff.png', dpi=300)
# %%
