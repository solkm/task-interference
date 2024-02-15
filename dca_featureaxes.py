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
import pandas as pd
rng = np.random.default_rng()
from sklearn.decomposition import PCA
from DCA import dca
from pathlib import Path

#%% PCA --> DCA

vis_noise, mem_noise, rec_noise = 0.1, 0.5, 0.1
N_rec = 200
K = 10
name = 'MM1_monkeyB1245' # 'SH2_correctA' # 
folder = 'monkey_choice_model' # 'correct_choice_model' # 
tParams_new = pd.read_csv('./data_inds/tParams_new.csv')
weights_path = f'./{folder}/weights/{name}.npz'

# choose indices (test each trial history condition once per model)
fname = f'./{folder}/{name}_DCAdf.pkl'
copy_inds = True# False #

if copy_inds == False:
    all_inds = np.where(tParams_new[f'K{K}trainable']==1)[0]
    N_trialhists = 500
    
    if Path(fname).exists():
        existing_df = pd.read_pickle(fname)
        tested_already = np.array(existing_df['ind'])
        avail_inds = np.delete(all_inds, np.in1d(all_inds, tested_already))
    else:
        avail_inds = all_inds
        
    inds2test = rng.choice(avail_inds, N_trialhists, replace=False)

else:
    df2copy = pd.read_pickle('./correct_choice_model/SH2_correctA_DCAdf.pkl') # pd.read_pickle('./monkey_choice_model/MM1_monkeyB1245_DCAdf.pkl') #   
    inds2copy = np.array(df2copy['ind'])
    
    if Path(fname).exists():
        existing_df = pd.read_pickle(fname)
        tested_already = np.array(existing_df['ind'])
        avail_inds = np.delete(inds2copy, np.in1d(inds2copy, tested_already))
    else:
        avail_inds = inds2copy
        
    inds2test = avail_inds
    N_trialhists = avail_inds.shape[0]
    print(N_trialhists, ' trials')

timepoints = np.arange(70, 121, 10)
n_pca_comp = 10
n_rew_hist = 3
task_outputs = []
reward_hist = []
pca_expvar = []
angles = []
dcovs_sl, dcovs_sf = [], []
ax_sl, ax_sf = [], []

for h in range(N_trialhists):
    
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
    reward_hist.append(np.round(np.mean(np.mean(test_inputs, axis=0), axis=0)
                         [np.arange(5, (K-1)*6, 6)])[-n_rew_hist:])
    
    # PCA
    pca = PCA(n_components=n_pca_comp)
    pca.fit_transform(np.reshape(fr_tpts, (fr_tpts.shape[0]*fr_tpts.shape[1], 
                                           fr_tpts.shape[2])))
    pca_expvar.append(np.sum(pca.explained_variance_ratio_))
    
    # DCA
    angles_ = np.zeros(timepoints.shape[0])
    dcovs_sl_ = np.zeros(timepoints.shape[0])
    ax_sl_ = np.zeros((timepoints.shape[0], n_pca_comp))
    dcovs_sf_ = np.zeros(timepoints.shape[0])
    ax_sf_ = np.zeros((timepoints.shape[0], n_pca_comp))
    
    for t in range(timepoints.shape[0]):
        
        print(f't={timepoints[t]}')
        
        fr_t = fr_tpts[:, t, :] @ pca.components_.T
        
        Xs_sl = []
        Xs_sl.append(fr_t.T)
        Xs_sl.append(sl1[:].reshape(1, sl1.shape[0]))
        U_sl, dcovs_sl_[t] = dca(Xs_sl, num_dca_dimensions=1, 
                                   percent_increase_criterion=0.01)
        ax_sl_[t] =  U_sl[1][0][0] * np.squeeze(U_sl[0])
        
        Xs_sf = []
        Xs_sf.append(fr_t.T)
        Xs_sf.append(sf1[:].reshape(1, sf1.shape[0]))
        U_sf, dcovs_sf_[t] = dca(Xs_sf, num_dca_dimensions=1, 
                                   percent_increase_criterion=0.01)
        ax_sf_[t] = U_sf[1][0][0] * np.squeeze(U_sf[0])
        
        angles_[t] = np.degrees(np.arccos(np.dot(ax_sl_[t], ax_sf_[t])))
        
    angles.append(angles_)
    dcovs_sl.append(dcovs_sl_)
    ax_sl.append(ax_sl_)
    dcovs_sf.append(dcovs_sf_)
    ax_sf.append(ax_sf_)
    
df = pd.DataFrame(data={'ind':inds2test, 'angles':angles, 
                        'dcovs_sl':dcovs_sl, 'dcovs_sf':dcovs_sf, 
                        'ax_sl':ax_sl, 'ax_sf':ax_sf, 
                        'task_outputs':task_outputs, 
                        'reward_history':reward_hist, 
                        'pca_expvar':pca_expvar, 
                        })

# save new dataframe
if Path(fname).exists()==False:
    df.to_pickle(fname)
else:
    df_combined = pd.concat([existing_df, df])
    df_combined = df_combined.reset_index(drop=True)
    df_combined.to_pickle(fname)
