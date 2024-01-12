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

name = 'MM1_monkeyB1245'
data_path = f'./monkey_choice_model/test_data/{name}_allinds_noisevis0.8mem0.5rec0.1'

model_output = pickle.load(open(data_path + '_modeloutput.pickle', 'rb'))
model_choices = np.argmax(model_output[:, -1, 2:6], axis=1) + 1
trial_params = pickle.load(open(data_path + '_trialparams.pickle', 'rb'))

assert model_choices.shape[0] >= 10000, 'test set is too small'
dsl_perf_aR, dsl_perf_aNR, dsf_perf_aR, dsf_perf_aNR, p_dsl, p_dsf = \
                perc_perf_same_stim(trial_params, model_choices, n_acc=50)

fontsize = 12
rcParams['font.sans-serif'] = 'Helvetica'
rcParams['font.size'] = fontsize

dotsize=40
lw=1
plt.figure(figsize=(6,6))
X = np.linspace(0, 1, 100)
plt.plot(X,X, color='slategrey', lw=1, ls='-', zorder=0)
plt.scatter(dsl_perf_aR, dsl_perf_aNR, s=dotsize, edgecolors='magenta', 
            facecolors='m', linewidths=lw, zorder=1, marker='+')
plt.scatter(dsf_perf_aR, dsf_perf_aNR, s=dotsize, edgecolors='cyan', 
            facecolors='none', linewidths=lw, zorder=1, marker='o')
plt.scatter(np.mean(dsl_perf_aR), np.mean(dsl_perf_aNR), s=dotsize*2, 
            marker='P', edgecolors='k', facecolors='m', linewidths=lw*1.2, 
            label='SL choice mean, p=%2.1e'%p_dsl, alpha=0.9, zorder=3)
plt.scatter(np.mean(dsf_perf_aR), np.mean(dsf_perf_aNR), s=dotsize*2, 
            marker='o', edgecolors='k', facecolors='cyan', linewidths=lw*1.2, 
            label='SF choice mean, p=%2.1e'%p_dsf, alpha=0.9, zorder=2)

minperf = np.min(np.concatenate((dsl_perf_aR, dsl_perf_aNR, dsf_perf_aR, dsf_perf_aNR)))
plt.xlim(left = min(minperf-.02, 0.4))
plt.ylim(bottom = min(minperf-.02, 0.4))
plt.xlabel('After a rewarded trial')
plt.ylabel('After an unrewarded trial')
plt.legend(loc='upper left', fontsize=fontsize-2)
plt.title('Perceptual performance for the same stimulus')
plt.tick_params(labelsize=fontsize-2)
plt.tight_layout()
plt.show()

print('SL, after R vs after NR', np.mean(dsl_perf_aR), np.mean(dsl_perf_aNR))
print('SF, after R vs after NR', np.mean(dsf_perf_aR), np.mean(dsf_perf_aNR))
print('performance difference, SL: ', np.mean(dsl_perf_aR)-np.mean(dsl_perf_aNR))
print('performance difference, SF: ',np.mean(dsf_perf_aR)-np.mean(dsf_perf_aNR))

rcParams['pdf.fonttype']=42
rcParams['pdf.use14corefonts']=True

#plt.savefig(f'./{name}_perceptualaccuracyafterRvsNR.pdf', dpi=300, transparent=True)
