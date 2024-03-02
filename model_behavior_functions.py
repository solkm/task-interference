#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:43:38 2024

@author: Sol
"""
import numpy as np
from statsmodels.stats.weightstats import ttest_ind

def get_tasks(choices):
    """
    Returns an array of tasks given an array of choices.
    task 1: SL, task 2: SF
    """
    tasks = np.zeros(choices.shape[0], dtype='int8')
    for i in range(choices.shape[0]):
        if choices[i]==3 or choices[i]==4:
            tasks[i] = 1
        elif choices[i]==1 or choices[i]==2:
            tasks[i] = 2

    return tasks

def get_overall_acc(model_choices, trial_params):

    N_testbatch = trial_params.shape[0]
    correct_choice = [trial_params[i]['correct'][-1] for i in range(N_testbatch)]
    accuracy = np.count_nonzero(correct_choice==model_choices)/N_testbatch

    return accuracy

def get_task_acc(model_choices, trial_params):

    N_testbatch = trial_params.shape[0]
    correct_task = [trial_params[i]['task'][-1] for i in range(N_testbatch)]
    model_task = get_tasks(model_choices)
    task_acc = np.count_nonzero(correct_task==model_task)/N_testbatch

    return task_acc

def get_monkeychoice_acc(model_choices, trial_params):
    
    N_testbatch = trial_params.shape[0]
    monkey_choice = [trial_params[i]['choice'][-1] for i in range(N_testbatch)]
    monkeychoice_acc = np.count_nonzero(monkey_choice==model_choices)/N_testbatch

    return monkeychoice_acc

def get_monkeytask_acc(model_choices, trial_params):
    
    N_testbatch = trial_params.shape[0]
    monkey_task = [trial_params[i]['m_task'][-1] for i in range(N_testbatch)]
    model_task = get_tasks(model_choices)
    monkeytask_acc = np.count_nonzero(monkey_task==model_task)/N_testbatch

    return monkeytask_acc

def get_perc_acc(model_choices, trial_params, dsl=None, dsf=None):
    """
    Parameters
    ----------
    trial_params : array
        Trial parameters that the model was tested on.
    model_choices : array
        The model choices, from {1,2,3,4}.
    dsl, dsf : array, optional alternative to trial_params.
        
    Returns
    -------
    correct_perc : array
        Whether perceptual judements were correct (1) or not (0).
    p_acc : float
        Perceptual accuracy over all trials.
    """
    N_testbatch = model_choices.shape[0]

    if trial_params is not None:
        dsl = [ trial_params[i]['dsl'][-1] for i in range(N_testbatch) ]
        dsf = [ trial_params[i]['dsf'][-1] for i in range(N_testbatch) ]

    correct_perc = np.zeros(N_testbatch)
    for i in range(N_testbatch):
        if model_choices[i]==1 and dsf[i]>0: # SF increase
            correct_perc[i]=1
        elif model_choices[i]==2 and dsf[i]<0: # SF decrease
            correct_perc[i]=1
        elif model_choices[i]==3 and dsl[i]<0: # SL decrease
            correct_perc[i]=1
        elif model_choices[i]==4 and dsl[i]>0: # SL increase
            correct_perc[i]=1

    p_acc = np.count_nonzero(correct_perc)/N_testbatch

    return correct_perc, p_acc

def get_perc_acc_afterRvsNR(model_choices, trial_params):
    """
    Perceptual accuracy after R vs after NR across all stimulus conditions
    """
    prev_choice = np.array([trial_params[i]['choice'][-2] for i in range(trial_params.shape[0])])
    prev_correct = np.array([trial_params[i]['correct'][-2] for i in range(trial_params.shape[0])])
    aR_inds = np.where(prev_choice==prev_correct)[0]
    aNR_inds = np.where(prev_choice!=prev_correct)[0]
    _, perf_aR = get_perc_acc(model_choices[aR_inds], trial_params[aR_inds])
    _, perf_aNR = get_perc_acc(model_choices[aNR_inds], trial_params[aNR_inds])

    return perf_aR, perf_aNR

def perc_perf_same_stim(model_choices, trial_params, n_acc=50, 
                        stim_cond='change_chosen', one_acc_per_cond=False):
    """
    Computes perceptual performances after a reward vs. after a non-reward for the same stimulus condition.

    Parameters
    ----------
    trial_params : array
        Trial parameters that the model was tested on.
    model_choices : array
        The model choices, from {1,2,3,4}.
    n_acc : int
       (Minimum) number of trials from which to compute an accuracy (see one_acc_per_cond).
    stim_cond : str
        'change_chosen' or 'change_both'. 
        If 'change_chosen', a stimulus condition is defined by the feature change amount corresponding to the chosen task only.
        If 'change_both', a stimulus condition is defined by the feature change amounts corresponding to both tasks.
    one_acc_per_cond : bool
        If True, only one accuracy is computed per condition from all >=n_acc trials.
        If False, (# trials)//n_acc accuracies are computed per condition from n_acc trials each.
    Returns
    -------
    SL_perf_aR,  SL_perf_aNR, SF_perf_aR, SF_perf_aNR: arrays of perceptual accuracies.
    """

    assert 4>=model_choices.all()>=1, 'model choices are invalid'
    N = len(trial_params[:])
    dsl = np.round([trial_params[i]['dsl'][-1] for i in range(N)], 2) # round feature change amounts to 2 decimal places
    dsf = np.round([trial_params[i]['dsf'][-1] for i in range(N)], 2)
    chosen_task = get_tasks(model_choices)
    SL_inds = np.where(chosen_task==1)[0]
    SF_inds = np.where(chosen_task==2)[0]
    prev_choice = np.array([trial_params[i]['choice'][-2] for i in range(trial_params.shape[0])])
    prev_correct = np.array([trial_params[i]['correct'][-2] for i in range(trial_params.shape[0])])
    aR_inds = np.where(prev_choice==prev_correct)[0]
    aNR_inds = np.where(prev_choice!=prev_correct)[0]
    correct_perc, _ = get_perc_acc(model_choices, trial_params)

    if stim_cond=='change_both':
        feature_changes = np.vstack((dsl, dsf)).T
        unique_changes, unique_changes_inv = np.unique(feature_changes, axis=0, return_inverse=True)

    SL_perf_aR,  SL_perf_aNR, SF_perf_aR, SF_perf_aNR = [], [], [], []
    SL_conds, SF_conds = [], []

    # SL choices
    if stim_cond=='change_chosen':
        unique_changes, unique_changes_inv = np.unique(dsl, return_inverse=True)

    for i in range(unique_changes.shape[0]):
        if np.any(unique_changes[i]==0): # do not consider no change trials
            continue
        cond_inds = np.where(unique_changes_inv==i)[0]
        SL_cond_inds = np.intersect1d(cond_inds, SL_inds)
        SL_cond_inds_aR = np.intersect1d(SL_cond_inds, aR_inds)
        SL_cond_inds_aNR = np.intersect1d(SL_cond_inds, aNR_inds)
        min_trials = min(SL_cond_inds_aR.shape[0], SL_cond_inds_aNR.shape[0])

        if one_acc_per_cond:
            if min_trials >= n_acc:
                SL_perf_aR.append(np.count_nonzero(correct_perc[SL_cond_inds_aR])/SL_cond_inds_aR.shape[0])
                SL_perf_aNR.append(np.count_nonzero(correct_perc[SL_cond_inds_aNR])/SL_cond_inds_aNR.shape[0])
                SL_conds.append(unique_changes[i])
        else:
            for k in range(min_trials//n_acc):
                SL_perf_aR.append(np.count_nonzero(correct_perc[SL_cond_inds_aR[k*n_acc:(k+1)*n_acc]])/n_acc)
                SL_perf_aNR.append(np.count_nonzero(correct_perc[SL_cond_inds_aNR[k*n_acc:(k+1)*n_acc]])/n_acc)
                SL_conds.append(unique_changes[i])

    # SF choices
    if stim_cond=='change_chosen':
        unique_changes, unique_changes_inv = np.unique(dsf, return_inverse=True)

    for i in range(unique_changes.shape[0]):
        if np.any(unique_changes[i]==0): # do not consider no change trials
            continue
        cond_inds = np.where(unique_changes_inv==i)[0]
        SF_cond_inds = np.intersect1d(cond_inds, SF_inds)
        SF_cond_inds_aR = np.intersect1d(SF_cond_inds, aR_inds)
        SF_cond_inds_aNR = np.intersect1d(SF_cond_inds, aNR_inds)
        min_trials = min(SF_cond_inds_aR.shape[0], SF_cond_inds_aNR.shape[0])

        if one_acc_per_cond:
            if min_trials >= n_acc:
                SF_perf_aR.append(np.count_nonzero(correct_perc[SF_cond_inds_aR])/SF_cond_inds_aR.shape[0])
                SF_perf_aNR.append(np.count_nonzero(correct_perc[SF_cond_inds_aNR])/SF_cond_inds_aNR.shape[0])
                SF_conds.append(unique_changes[i])
        else:
            for k in range(min_trials//n_acc):
                SF_perf_aR.append(np.count_nonzero(correct_perc[SF_cond_inds_aR[k*n_acc:(k+1)*n_acc]])/n_acc)
                SF_perf_aNR.append(np.count_nonzero(correct_perc[SF_cond_inds_aNR[k*n_acc:(k+1)*n_acc]])/n_acc)
                SF_conds.append(unique_changes[i])
    
    SL_perf_aR = np.array(SL_perf_aR)
    SL_perf_aNR = np.array(SL_perf_aNR)
    SF_perf_aR = np.array(SF_perf_aR)
    SF_perf_aNR = np.array(SF_perf_aNR)
    SL_conds = np.array(SL_conds)
    SF_conds = np.array(SF_conds)

    return SL_perf_aR, SL_perf_aNR, SF_perf_aR, SF_perf_aNR, SL_conds, SF_conds

def sliding_window_avg(measure, n_avg, sem=None):
    """
    Returns a sliding window average of a measure over n_avg epochs.
    measure_sw[0] is the mean, measure_sw[1] is the SEM.
        The SEM is computed from standard deviation over the window OR from the 
        individual SEMs if provided.
    """
    n_steps = measure.shape[0] - n_avg
    measure_sw = np.zeros((2, n_steps))
    for i in range(n_steps):
        measure_sw[0, i] = np.mean(measure[i:i+n_avg])
        if sem is not None:
            measure_sw[1, i] = np.sqrt(np.mean(sem[i:i+n_avg]**2))
        else:
            measure_sw[1, i] = np.std(measure[i:i+n_avg])/np.sqrt(n_avg)

    return measure_sw
