#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:09:52 2024

@author: Sol
"""
#%%
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import model_behavior_functions as mf
import scipy.stats as st

#%% side-by-side plot, two models

name1 = 'MM1_monkeyB1245'
data_path1 = f'./monkey_choice_model/test_data/{name1}_allinds_noisevis0.8mem0.5rec0.1'
name2 = 'SH2_correctA'
data_path2 = f'./correct_choice_model/test_data/{name2}_monkeyhist_allinds_noisevis0.8mem0.5rec0.1'
data_paths_ = [data_path1, data_path2]
names_ = [name1, name2]

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
fontsize = 12
rcParams['font.sans-serif'] = 'Helvetica'
rcParams['font.size'] = fontsize
dotsize = 40
lw = 1
X = np.linspace(0, 1, 100)

for i in range(len(data_paths_)):

    data_path = data_paths_[i]
    model_output = pickle.load(open(data_path + '_modeloutput.pickle', 'rb'))
    model_choices = np.argmax(model_output[:, -1, 2:6], axis=1) + 1
    trial_params = pickle.load(open(data_path + '_trialparams.pickle', 'rb'))

    assert model_choices.shape[0] >= 10000, 'test set is too small'
    SL_perf_aR,  SL_perf_aNR, SF_perf_aR, SF_perf_aNR, = \
        mf.perc_perf_same_stim(model_choices, trial_params, n_acc=50, 
                               stim_cond='change_chosen', one_acc_per_cond=False)
    
    perf_aR, perf_aNR = mf.get_perc_acc_afterRvsNR(model_choices, trial_params)

    # stats
    stat_SL, p_SL = st.ranksums(SL_perf_aR, SL_perf_aNR)
    stat_SF, p_SF = st.ranksums(SF_perf_aR, SF_perf_aNR)

    # plot
    ax[i].plot(X, X, color='k', lw=1, ls='-', zorder=0)
    ax[i].scatter(SL_perf_aR, SL_perf_aNR, s=dotsize, facecolors='m', 
                  linewidths=lw, zorder=0, marker='+')
    ax[i].scatter(SF_perf_aR, SF_perf_aNR, s=dotsize, edgecolors='cyan', 
                  facecolors='none', linewidths=lw, zorder=0, marker='o')
    ax[i].scatter(np.mean(SL_perf_aR), np.mean(SL_perf_aNR), s=dotsize*2, 
                  marker='P', edgecolors='k', facecolors='m', linewidths=lw*1.2, 
                  label='SL choice mean, p=%2.1e'%p_SL, alpha=0.9, zorder=2)
    ax[i].scatter(np.mean(SF_perf_aR), np.mean(SF_perf_aNR), s=dotsize*2, 
                  marker='o', edgecolors='k', facecolors='cyan', linewidths=lw*1.2, 
                  label='SF choice mean, p=%2.1e'%p_SF, alpha=0.9, zorder=1)
    ax[i].legend(loc='upper left', fontsize=fontsize-2)
    ax[i].set_title(names_[i], fontsize=fontsize)
    ax[i].set_xlim(left=0.15)
    ax[i].set_ylim(bottom=0.15)
    ax[i].set_aspect('equal')

    print(names_[i] + ' perceptual performance')
    print('across all conditions, after R vs after NR', 
          np.round(perf_aR, 3), np.round(perf_aNR, 3))
    print('difference: ', np.round(perf_aR - perf_aNR, 3))
    print('SL mean, after R vs after NR', 
          np.round(np.mean(SL_perf_aR), 3), np.round(np.mean(SL_perf_aNR), 3))
    print('SF mean, after R vs after NR', 
          np.round(np.mean(SF_perf_aR), 3), np.round(np.mean(SF_perf_aNR), 3))
    print('SL mean difference: ', 
          np.round(np.mean(SL_perf_aR)-np.mean(SL_perf_aNR), 3))
    print('SF mean difference: ', 
          np.round(np.mean(SF_perf_aR)-np.mean(SF_perf_aNR), 3))

ax[0].set_xlabel('After a rewarded trial')
ax[0].set_ylabel('After an unrewarded trial')
plt.suptitle('Perceptual accuracy', fontsize=fontsize)
plt.tight_layout()

rcParams['pdf.fonttype']=42
rcParams['pdf.use14corefonts']=True
#plt.savefig(f'./{name1}_and_{name2}_perceptualaccuracyafterRvsNR.pdf', dpi=300, transparent=True)

# %%
