import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

#%% Load monkey data
tParams_ndist = pd.read_csv('./data_inds/tParams_ndist.csv')
tParams_new = pd.read_csv('./data_inds/tParams_new.csv')

# Load model outputs and trial parameters
mod1_name = 'MM1_monkeyB1245'
path1 = f'./monkey_choice_model/test_data/{mod1_name}_allinds_noisevis0.8mem0.5rec0.1'
mod1_outputs = pickle.load(open(path1 + '_modeloutput.pickle', 'rb'))
mod1_choices = np.argmax(mod1_outputs[:, -1, 2:6], axis=1) + 1
mod1_trialparams = pickle.load(open(path1 + '_trialparams.pickle', 'rb'))
mod1_trial_inds = [mod1_trialparams[i]['trial_ind'] for i in range(mod1_trialparams.shape[0])]
mod1_og_inds = np.array(tParams_new.loc[mod1_trial_inds, 'og_ind'], dtype=int)

mod2_name = 'SH2_correctA'
path2 = f'./correct_choice_model/test_data/{mod2_name}_monkeyhist_allinds_noisevis0.8mem0.5rec0.1'
mod2_outputs = pickle.load(open(path2 + '_modeloutput.pickle', 'rb'))
mod2_choices = np.argmax(mod2_outputs[:, -1, 2:6], axis=1) + 1
mod2_trialparams = pickle.load(open(path2 + '_trialparams.pickle', 'rb'))
mod2_trial_inds = [mod2_trialparams[i]['trial_ind'] for i in range(mod2_trialparams.shape[0])]
mod2_og_inds = np.array(tParams_new.loc[mod2_trial_inds, 'og_ind'], dtype=int)

# %% Visualize model task outputs
mod1_SLtaskout = mod1_outputs[..., 0]
mod1_SFtaskout = mod1_outputs[..., 1]
mod1_taskout_diff = mod1_SLtaskout - mod1_SFtaskout

f, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].hist(mod1_SLtaskout[:, -1], bins=20, density=True, label='SL', histtype='step')
ax[0].hist(mod1_SFtaskout[:, -1], bins=20, density=True, label='SF', histtype='step')
ax[0].set(xlabel='Task output', ylabel='Density')
ax[0].legend()

ax[1].hist(mod1_taskout_diff[:, -1], bins=20, density=True)
ax[1].set(xlabel='Sum of task outputs')

ax[2].hist(mod1_SLtaskout[:, -1] - mod1_SFtaskout[:, -1], bins=20, density=True)
ax[2].set(xlabel='Difference of task outputs (SL - SF)')

mod2_SLtaskout = mod2_outputs[..., 0]
mod2_SFtaskout = mod2_outputs[..., 1]
mod2_taskout_diff = mod2_SLtaskout - mod2_SFtaskout

# %% Scatterplot of model task outputs difference vs monkey neuronal task belief
og_inds_both = np.intersect1d(mod1_og_inds, tParams_ndist['og_ind'])
mod1_inds2plot = np.where(np.isin(mod1_og_inds, og_inds_both))[0]
tParams_inds2plot = np.where(np.isin(tParams_ndist['og_ind'], og_inds_both))[0]
assert np.array_equal(mod1_og_inds[mod1_inds2plot], tParams_ndist['og_ind'][tParams_inds2plot])

assert np.array_equal(mod1_og_inds, mod2_og_inds)
assert np.array_equal(mod2_og_inds[mod1_inds2plot], tParams_ndist['og_ind'][tParams_inds2plot])

f, ax = plt.subplots(2, 1, figsize=(5, 10), sharex=True, sharey=True)
ax[0].scatter(mod1_taskout_diff[mod1_inds2plot, -1], tParams_ndist['nd_task'][tParams_inds2plot],
            s=5, alpha=0.2)
ax[1].scatter(mod2_taskout_diff[mod1_inds2plot, -1], tParams_ndist['nd_task'][tParams_inds2plot],
            s=5, alpha=0.2)
ax[0].set(xlabel=f'{mod1_name} task output difference (SL - SF)', 
          ylabel='Monkey normalized neuronal task belief')
ax[1].set(xlabel=f'{mod2_name} task output difference (SL - SF)')

# %%
