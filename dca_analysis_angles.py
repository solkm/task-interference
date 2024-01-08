#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 12:34:23 2024

@author: Sol
"""
#%%
import os
os.chdir('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/task-interference')
import sys
sys.path.append('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject')
from self_history import Task_SH2
from psychrnn.backend.simulation import BasicSimulator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
rng = np.random.default_rng()
from sklearn.decomposition import PCA
from DCA import dca
from pathlib import Path
from scipy.stats import ttest_ind, ttest_rel, ranksums

#%% PCA --> DCA

vis_noise, mem_noise, rec_noise = 0.8, 0.5, 0.1
N_rec = 200
K = 10
name = 'MM1_monkeyB1245'
folder = 'monkey_choice_model'
tParams_new = pd.read_csv('./data_inds/tParams_new.csv')
weights_path = f'./{folder}/weights/{name}.npz'

n_pca_comp = 10

# choose indices
fname = f'./{folder}/{name}_dca_df.csv'
copy_inds = False

if copy_inds == False:
    all_inds = np.where(tParams_new[f'K{K}trainable']==1)[0]
    n_trial_hists = 2
    
    if Path(fname).exists():
        existing_df = pd.read_csv(fname)
        tested_already = np.array(existing_df['ind'])
        avail_inds = np.delete(all_inds, np.in1d(all_inds, tested_already))
    else:
        avail_inds = all_inds
        
    inds2test = rng.choice(avail_inds, n_trial_hists, replace=False)

else:
    df2copy = pd.read_csv('./PATH_TO_DATAFRAME.csv')
    inds2copy = np.array(df2copy['ind'])
    
    if Path(fname).exists():
        existing_df = pd.read_csv(fname)
        tested_already = np.array(existing_df['ind'])
        avail_inds = np.delete(inds2copy, np.in1d(inds2copy, tested_already))
    else:
        avail_inds = inds2copy
        
    inds2test = avail_inds
    n_trial_hists = avail_inds.shape[0]
    print(n_trial_hists, ' trials')

timepoints = np.arange(70, 121, 5)
n_rew_hist = 3
task_outputs = []
reward_history = []
pca_exp_var = []
angles = []
dcovs_sf = []
dcovs_sl = []

for h in range(n_trial_hists):
    
    inds = [inds2test[h]]
    print('trial hist #', h, ', ind ', inds)
    
    # simulate trials
    N_testbatch = 500
    
    task = Task_SH2(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=N_testbatch, 
                    dat=tParams_new, dat_inds=inds, K=K, fixedDelay=510)

    network_params = task.get_task_params()
    network_params['name'] = name
    network_params['N_rec'] = N_rec
    network_params['rec_noise'] = rec_noise
    
    test_inputs, _, _, trial_params = task.get_trial_batch()
    
    simulator = BasicSimulator(weights_path=weights_path, params=network_params)
    
    model_output, state_var = simulator.run_trials(test_inputs)
    
    sf1 = np.array([trial_params[i]['sf1'] for i in range(trial_params.shape[0])])
    sl1 = np.array([trial_params[i]['sl1'] for i in range(trial_params.shape[0])])
    
    fr_tpts = np.maximum(state_var[:, timepoints, 100:], 0)
    
    task_outputs.append(np.mean(model_output[:, -1, :2], axis=0))
    reward_history.append(np.round(np.mean(np.mean(test_inputs, axis=0), axis=0)
                         [np.arange(5, (K-1)*6, 6)])[-n_rew_hist:])
    
    # PCA
    pca = PCA(n_components=n_pca_comp)
    pca.fit_transform(np.reshape(fr_tpts, (fr_tpts.shape[0]*fr_tpts.shape[1], 
                                           fr_tpts.shape[2])))
    pca_exp_var.append(np.sum(pca.explained_variance_ratio_))
    
    # DCA
    angles_ = np.zeros(timepoints.shape[0])
    dcovs_sl_ = np.zeros(timepoints.shape[0])
    dcovs_sf_ = np.zeros(timepoints.shape[0])
    
    for t in range(timepoints.shape[0]):
        
        print(f't={timepoints[t]}')
        
        fr_t = fr_tpts[:, t, :] @ pca.components_.T
        
        Xs_sl = []
        Xs_sl.append(fr_t.T)
        Xs_sl.append(sl1[:].reshape(1, sl1.shape[0]))

        U_sl, dcovs_sl_[t] = dca(Xs_sl, num_dca_dimensions = 1, 
                                   percent_increase_criterion = 0.01)
        
        Xs_sf = []
        Xs_sf.append(fr_t.T)
        Xs_sf.append(sf1[:].reshape(1, sf1.shape[0]))

        U_sf, dcovs_sf_[t] = dca(Xs_sf, num_dca_dimensions = 1, 
                                   percent_increase_criterion = 0.01)
        
        u1, u2 = np.squeeze(U_sl[0]), np.squeeze(U_sf[0])
        angles_[t] = np.degrees(np.arccos(np.dot(u1, u2) / 
                                    (np.linalg.norm(u1)*np.linalg.norm(u2))))
    
    angles.append(angles_)
    dcovs_sl.append(dcovs_sl_)
    dcovs_sf.append(dcovs_sf_)
    
    
df = pd.DataFrame(data={'ind':inds2test, 'angles':angles, 'dcovs_sl':dcovs_sl,
                        'dcovs_sf':dcovs_sf, 'task_outputs':task_outputs,
                        'reward_history':reward_history, 
                        'pca_exp_var':pca_exp_var})

# save new dataframe
if Path(fname).exists()==False:
    df.to_csv(fname, index=False)
else:
    df_combined = pd.concat([existing_df, df])
    df_combined.to_csv(fname, index=False)

#%%

for i in range(df.shape[0]):
    plt.plot(df.loc[i, 'angles'])






