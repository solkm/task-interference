#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:09:52 2024

@author: Sol
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
from model_behavior_analysis import perc_perf_same_stim

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
dotsize=40
lw=1
X = np.linspace(0, 1, 100)

for i in range(len(data_paths_)):

    data_path = data_paths_[i]
    model_output = pickle.load(open(data_path + '_modeloutput.pickle', 'rb'))
    model_choices = np.argmax(model_output[:, -1, 2:6], axis=1) + 1
    trial_params = pickle.load(open(data_path + '_trialparams.pickle', 'rb'))

    assert model_choices.shape[0] >= 10000, 'test set is too small'
    dsl_perf_aR, dsl_perf_aNR, dsf_perf_aR, dsf_perf_aNR, p_dsl, p_dsf = \
                    perc_perf_same_stim(trial_params, model_choices, n_acc=50)

    # plot
    ax[i].plot(X, X, color='k', lw=1, ls='-', zorder=0)
    ax[i].scatter(dsl_perf_aR, dsl_perf_aNR, s=dotsize, facecolors='m', 
                  linewidths=lw, zorder=1, marker='+')
    ax[i].scatter(dsf_perf_aR, dsf_perf_aNR, s=dotsize, edgecolors='cyan', 
                  facecolors='none', linewidths=lw, zorder=1, marker='o')
    ax[i].scatter(np.mean(dsl_perf_aR), np.mean(dsl_perf_aNR), s=dotsize*2, 
                  marker='P', edgecolors='k', facecolors='m', linewidths=lw*1.2, 
                  label='SL choice mean, p=%2.1e'%p_dsl, alpha=0.9, zorder=3)
    ax[i].scatter(np.mean(dsf_perf_aR), np.mean(dsf_perf_aNR), s=dotsize*2, 
                  marker='o', edgecolors='k', facecolors='cyan', linewidths=lw*1.2, 
                  label='SF choice mean, p=%2.1e'%p_dsf, alpha=0.9, zorder=2)
    ax[i].legend(loc='upper left', fontsize=fontsize-2)
    ax[i].set_title(names_[i], fontsize=fontsize)

    print(names_[i])
    print('SL, after R vs after NR', np.mean(dsl_perf_aR), np.mean(dsl_perf_aNR))
    print('SF, after R vs after NR', np.mean(dsf_perf_aR), np.mean(dsf_perf_aNR))
    print('performance difference, SL: ', np.mean(dsl_perf_aR)-np.mean(dsl_perf_aNR))
    print('performance difference, SF: ',np.mean(dsf_perf_aR)-np.mean(dsf_perf_aNR))

ax[0].set_xlim(left=0.15)
ax[0].set_ylim(bottom=0.15)
ax[0].set_xlabel('After a rewarded trial')
ax[0].set_ylabel('After an unrewarded trial')
plt.suptitle('Perceptual accuracy', fontsize=fontsize)
plt.tick_params(labelsize=fontsize-2)
plt.tight_layout()

rcParams['pdf.fonttype']=42
rcParams['pdf.use14corefonts']=True
#plt.savefig(f'./{name1}_and_{name2}_perceptualaccuracyafterRvsNR.pdf', dpi=300, transparent=True)

# %%
