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
mod1_taskout_sum = mod1_SLtaskout + mod1_SFtaskout
mod2_SLtaskout = mod2_outputs[..., 0]
mod2_SFtaskout = mod2_outputs[..., 1]
mod2_taskout_diff = mod2_SLtaskout - mod2_SFtaskout
mod2_taskout_sum = mod2_SLtaskout + mod2_SFtaskout

f = plt.figure(figsize=(16, 8))
ax00 = f.add_subplot(2, 3, 1)
ax01 = f.add_subplot(2, 3, 2)
ax02 = f.add_subplot(2, 3, 3)
ax10 = f.add_subplot(2, 3, 4, sharex=ax00, sharey=ax00)
ax11 = f.add_subplot(2, 3, 5, sharex=ax01, sharey=ax01 )
ax12 = f.add_subplot(2, 3, 6, sharex=ax02, sharey=ax02)

ax00.hist(mod1_SLtaskout[:, -1], bins=20, density=True, label='SL', histtype='step')
ax00.hist(mod1_SFtaskout[:, -1], bins=20, density=True, label='SF', histtype='step')
ax00.set(ylabel=f'{mod1_name}\nDensity')
ax00.legend()

ax01.hist(mod1_taskout_sum[:, -1], bins=20, density=True)
ax02.hist(mod1_taskout_diff[:, -1], bins=20, density=True)

ax10.hist(mod2_SLtaskout[:, -1], bins=20, density=True, label='SL', histtype='step')
ax10.hist(mod2_SFtaskout[:, -1], bins=20, density=True, label='SF', histtype='step')
ax10.set(xlabel='Task output', ylabel=f'{mod2_name}\nDensity')
ax10.legend()

ax11.hist(mod2_taskout_sum[:, -1], bins=20, density=True)
ax11.set(xlabel='Sum of task outputs')
ax12.hist(mod2_taskout_diff[:, -1], bins=20, density=True)
ax12.set(xlabel='Difference of task outputs (SL - SF)')

plt.tight_layout()

# %% Compare model task outputs with monkey neuronal (7a) task belief
og_inds_both = np.intersect1d(mod1_og_inds, tParams_ndist['og_ind'])
mod1_inds2plot = np.where(np.isin(mod1_og_inds, og_inds_both))[0]
tParams_inds2plot = np.where(np.isin(tParams_ndist['og_ind'], og_inds_both))[0]
assert np.array_equal(mod1_og_inds[mod1_inds2plot], tParams_ndist['og_ind'][tParams_inds2plot])
assert np.array_equal(mod1_og_inds, mod2_og_inds)

# Histogram of monkey neuronal task belief
plt.figure()
plt.hist(tParams_ndist['nd_task'][tParams_inds2plot], bins=50, density=True)
plt.xlabel('Monkey normalized neuronal task belief')
plt.ylabel('Density')

# Scatterplot of monkey neuronal task belief vs model task outputs difference
f, ax = plt.subplots(2, 1, figsize=(5, 10), sharex=True, sharey=True)
ax[0].scatter(mod1_taskout_diff[mod1_inds2plot, -1], tParams_ndist['nd_task'][tParams_inds2plot],
            s=5, alpha=0.2)
ax[1].scatter(mod2_taskout_diff[mod1_inds2plot, -1], tParams_ndist['nd_task'][tParams_inds2plot],
            s=5, alpha=0.2)
ax[0].set(xlabel=f'{mod1_name} task output difference (SL - SF)', 
          ylabel='Monkey normalized neuronal task belief')
ax[1].set(xlabel=f'{mod2_name} task output difference (SL - SF)')
plt.tight_layout()

# Binned violin plot
bins = np.array([-0.8, -0.4, 0, 0.4, 0.8])
mod1_taskout_diff_dig = np.digitize(mod1_taskout_diff[mod1_inds2plot, -1], bins)
mod2_taskout_diff_dig = np.digitize(mod2_taskout_diff[mod1_inds2plot, -1], bins)
assert np.array_equal(np.unique(mod1_taskout_diff_dig), np.unique(mod2_taskout_diff_dig))
n_bins = np.unique(mod1_taskout_diff_dig).shape[0]

f, ax = plt.subplots(2, 1, figsize=(5, 10), sharex=True, sharey=True)
for i in range(n_bins):
    inds1 = np.where(mod1_taskout_diff_dig == i)[0]
    ax[0].violinplot(tParams_ndist['nd_task'][tParams_inds2plot[inds1]], 
                     positions=[i], widths=0.5, showmeans=True, showextrema=False)
    inds2 = np.where(mod2_taskout_diff_dig == i)[0]
    ax[1].violinplot(tParams_ndist['nd_task'][tParams_inds2plot[inds2]], 
                     positions=[i], widths=0.5, showmeans=True, showextrema=False)
ax[0].set(xlabel=f'{mod1_name} task output difference (SL - SF)',
          ylabel='Monkey normalized neuronal task belief')
ax[1].set_xlabel(f'{mod2_name} task output difference (SL - SF)')
ax[1].set_xticks(range(n_bins), [f'< {bins[0]}'] + [f'[{bins[i-1]}, {bins[i]})' \
                    for i in range(1, bins.shape[0])] + [f'>= {bins[-1]}'], rotation=30)
plt.ylim(-4, 4)
plt.tight_layout()

# %%
