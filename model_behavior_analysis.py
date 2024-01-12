#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:43:38 2024

@author: Sol
"""
import numpy as np
from statsmodels.stats.weightstats import ttest_ind

def perc_perf_same_stim(trial_params, model_choice, n_acc=50):
    """
    Computes perceptual performances for the same stimuli after a reward vs. after a non-reward.

    Parameters
    ----------
    trial_params : array
        Trial parameters that the model was tested on.
    model_choice : array
        The model choices, from {1,2,3,4}.
    n_acc : int
        Minimum number of trials from which to compute accuracies.
        
    Returns
    -------
    dsl_perf_aR : list
    dsl_perf_aNR : list
    dsf_perf_aR : list
    dsf_perf_aNR : list
    p_dsl : float
    p_dsf : float
    """

    N = len(trial_params[:])
    dsl = np.round([trial_params[i]['dsl'][-1]*2 for i in range(N)], 2)
    dsf = np.round([trial_params[i]['dsf'][-1]*2 for i in range(N)], 2)
    prevChoice = [trial_params[i]['choice'][-2] for i in range(N)]
    prevCorrect = [trial_params[i]['correct'][-2] for i in range(N)]
    assert 4>=model_choice.all()>=1, 'model choices are invalid'

    def stim_perf(stim, stim_type = ''):
        stim_unique, stim_unique_inv = np.unique(stim, return_inverse=True)
        stim_perf_aR = []
        stim_perf_aNR = []
        
        for i in range(len(stim_unique)):        
            temp_aR = []
            temp_aNR = []
            if stim_unique[i] == 0: # do not consider zero stimulus change trials
                continue
            for j in range(len(stim_unique_inv)):
                if stim_unique_inv[j] == i:
                    
                    correct_percep = None
                    if stim_type == 'dsl':
                        if model_choice[j]==3 or model_choice[j]==4:
                            correct_percep = ((model_choice[j]==3)&(dsl[j]<0)) | ((model_choice[j]==4)&(dsl[j]>0))
                    elif stim_type == 'dsf':
                        if model_choice[j]==1 or model_choice[j]==2:
                            correct_percep = ((model_choice[j]==1)&(dsf[j]>0)) | ((model_choice[j]==2)&(dsf[j]<0))

                    if prevChoice[j] == prevCorrect[j] and correct_percep is not None:
                        temp_aR.append(int(correct_percep))
                    elif prevChoice[j] != prevCorrect[j] and correct_percep is not None:
                        temp_aNR.append(int(correct_percep))
            
            if len(temp_aR) >= n_acc and len(temp_aNR) >= n_acc:
                n = min(len(temp_aR)//n_acc, len(temp_aNR)//n_acc)
                for k in range(n):
                    stim_perf_aR.append(np.count_nonzero(temp_aR[k*n_acc:(k+1)*n_acc])/n_acc)
                    stim_perf_aNR.append(np.count_nonzero(temp_aNR[k*n_acc:(k+1)*n_acc])/n_acc)
            
        return stim_perf_aR, stim_perf_aNR
    
    dsl_perf_aR, dsl_perf_aNR = stim_perf(dsl, stim_type='dsl')
    dsf_perf_aR, dsf_perf_aNR = stim_perf(dsf, stim_type='dsf')
    _, p_dsl, _ = ttest_ind(dsl_perf_aR, dsl_perf_aNR, alternative='larger')
    _, p_dsf, _ = ttest_ind(dsf_perf_aR, dsf_perf_aNR, alternative='larger')
    
    return dsl_perf_aR, dsl_perf_aNR, dsf_perf_aR, dsf_perf_aNR, p_dsl, p_dsf