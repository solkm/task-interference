#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:14:08 2023

@author: Sol
"""
#%%
import os
os.chdir('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/task-interference')
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
#%%
name1 = 'SH2_correctA'
folder1 = 'correct_choice_model/test_data'
testname1 = 'monkeyhist_alltestinds_noisevis0.8mem0.5rec0.1'
outputs1 = pickle.load(open(f'./{folder1}/{name1}_{testname1}_modeloutput.pickle', 'rb'))
trial_params1 = pickle.load(open(f'./{folder1}/{name1}_{testname1}_trialparams.pickle', 'rb'))
test_inds1 = np.array([trial_params1[i]['trial_ind'] for i in range(trial_params1.shape[0])])

name2 = 'MM1_monkeyB1245'
folder2 = 'monkey_choice_model/test_data'
testname2 = 'alltestinds_noisevis0.8mem0.5rec0.1'
outputs2 = pickle.load(open(f'./{folder2}/{name2}_{testname2}_modeloutput.pickle', 'rb'))
trial_params2 = pickle.load(open(f'./{folder2}/{name2}_{testname2}_trialparams.pickle', 'rb'))
test_inds2 = np.array([trial_params2[i]['trial_ind'] for i in range(trial_params2.shape[0])])

prevErr = np.array([trial_params1[i]['choice'][-2]!=trial_params1[i]['correct'][-2] \
                    for i in range(trial_params1.shape[0])])
K=10
aR_inds = np.load(open(f'./data_inds/K{K}trainable_aRinds.npy', 'rb'))
aNR_inds = np.load(open(f'./data_inds/K{K}trainable_aNRinds.npy', 'rb'))

#%% define and plot boundary
t0 = 145
t = np.arange(180-t0)
a = 1.5
b = 1.8
c = 28

bound = a * (1 - b * t/(t+c))

plt.plot(t+t0, bound)
plt.ylim(-0.4, 1.4)
plt.hlines(0,0,180,ls='--', colors='k')

#%% compute RTs

assert np.array_equal(test_inds1, test_inds2), 'test trials must be identical to compute pairwise differences'

RT1 = np.zeros(outputs1.shape[0])
choice_RT1 = np.zeros(outputs1.shape[0])
for i in range(outputs1.shape[0]):
    for t in range(t0, 180):
        decision_bound = a * (1 - b * (t-t0)/((t-t0)+c))
        maxchoiceactivity1 = np.max(outputs1[i, t, 2:6])
        if maxchoiceactivity1 < decision_bound:
            continue
        else:
            RT1[i] = t-t0
            choice_RT1[i] = np.argmax(outputs1[i, t, 2:6]) + 1
            break

RT2 = np.zeros(outputs2.shape[0])
choice_RT2 = np.zeros(outputs2.shape[0])
for i in range(outputs2.shape[0]):
    for t in range(t0, 180):
        decision_bound = a * (1 - b * (t-t0)/((t-t0)+c))
        maxchoiceactivity2 = np.max(outputs2[i, t, 2:6])
        if maxchoiceactivity2 < decision_bound:
            continue
        else:
            RT2[i] = t-t0
            choice_RT2[i] = np.argmax(outputs2[i, t, 2:6]) + 1
            break

RTdiff = RT2 - RT1

_, p_rel_aNR = ttest_rel(RT2[prevErr], RT1[prevErr])
_, p_rel_aR = ttest_rel(RT2[prevErr==False], RT1[prevErr==False])
_, p_diff_aRvsNR = ttest_ind(RTdiff[prevErr], RTdiff[prevErr==False])
print('pairwise aNR: p=', p_rel_aNR)
print('pairwise aR: p=', p_rel_aR)
print('aR vs aNR differences: p=', p_diff_aRvsNR)

#%% difference plot aR vs aNR, proportion of trials histogram

rcParams['font.size'] = 11
rcParams['font.sans-serif'] = 'Helvetica'

bins = np.arange(-14,18,2)
bin_centers = (bins[1:] + bins[:-1])/2
ymax = 0.12
props_aR, _, _ = plt.hist(RTdiff[prevErr==False], alpha=0.4, label='after reward', 
                          color='g', bins=bins, density=True, histtype='step')
props_aNR, _, _  = plt.hist(RTdiff[prevErr], alpha=0.4, label='after non-reward', 
                            color='r', bins=bins, density=True, histtype='step')
plt.plot(bin_centers, props_aR, color='g')
plt.plot(bin_centers, props_aNR, color='r')

plt.vlines(np.mean(RTdiff[prevErr]), 0, ymax, color='r', ls='--', 
           label=f'mean difference = {np.round(np.mean(RTdiff[prevErr])*10,1)}')
plt.vlines(np.mean(RTdiff[prevErr==False]), 0, ymax, color='g', ls='--', 
           label=f'mean difference = {np.round(np.mean(RTdiff[prevErr==False])*10,1)}')

plt.ylim(0, ymax)
plt.xticks(np.arange(-15,18,5), np.arange(-15,18,5)*10)
plt.xlabel(f'Reaction time difference (ms): {name2} - {name1}')
plt.ylabel('Probability density')
plt.legend(loc='upper left')
plt.text(-13, 0.08, 'p=%2.1e'%p_diff_aRvsNR)
plt.tight_layout()

rcParams['pdf.fonttype']=42
rcParams['pdf.use14corefonts']=True

#plt.savefig(f'./{name1}_vs_{name2}_RTdiffhist_testtrials_line.pdf', dpi=300, transparent=True)

# %%
