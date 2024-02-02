#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:25:29 2023

@author: Sol
"""
from psychrnn.tasks.task import Task
import numpy as np
rng = np.random.default_rng()

class Task_MM1(Task):
    def __init__(self, dat=None, dat_inds=None, dt=10, tau=100, T=1800, 
                 N_batch=100, K=10, mem_noise=0.5, vis_noise=0.5,testall=False, 
                 fixedDelay=None, fixedSL=[None,None], fixedSF=[None,None]):
        N_in = 6*(K-1)+3
        N_out = 7
        super().__init__(N_in, N_out, dt, tau, T, N_batch)
        self.K = K
        self.dat = dat
        self.dat_inds = dat_inds
        self.mem_noise = mem_noise
        self.vis_noise = vis_noise
        self.testall = testall
        self.fixedDelay = fixedDelay
        self.fixedSL = fixedSL
        self.fixedSF = fixedSF
        if testall==True:
            if (self.N_batch != dat_inds.shape[0]):
                print('N_batch does not match data shape')
                
    def generate_trial_params(self, batch, trial):
        
        """"Define parameters for each trial.
    
        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.
    
        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.
    
        Returns:
            dict: Dictionary of trial parameters.
    
        """
        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
            
        dat = self.dat
        dat_inds = self.dat_inds
        K = self.K
        
        if self.testall==False:
            i = np.random.choice(dat_inds)
        else:
            i = dat_inds[trial]

        params = dict()
        params['trial_ind'] = i
        params['choice'] = np.array(dat['choice'][i-K+1:i+1], dtype='int')
        params['correct'] = np.array(dat['correct'][i-K+1:i+1], dtype='int')
        params['dsf'] = np.array(dat['dsf'][i-K+1:i+1])/2
        params['dsl'] = np.array(dat['dsl'][i-K+1:i+1])/2
        params['task'] = np.array(dat['task'][i-K+1:i+1], dtype='int')
        params['m_task'] = np.array(dat['m_task'][i-K+1:i+1], dtype='int')
        
        dsf = params['dsf'][-1]
        params['sf1'] = rng.uniform(2,3)
        params['sf2'] = params['sf1'] + dsf
        
        dsl = params['dsl'][-1]
        params['sl1'] = rng.uniform(2,3)
        params['sl2'] = params['sl1'] + dsl
    
        if self.fixedSF[0] is not None:
            params['sf1'] = self.fixedSF[0]
            
            if self.fixedSF[1] is not None:
                params['sf2'] = self.fixedSF[1]
                dsf = self.fixedSF[1] - self.fixedSF[0]
            else:
                dsf = rng.uniform(-1,1)
                params['sf2'] = self.fixedSF[0] + dsf
        
        else:
            assert self.fixedSF[1] is None, 'cannot fix only the second stimulus'
                
        if self.fixedSL[0] is not None:
            params['sl1'] = self.fixedSL[0]
            
            if self.fixedSL[1] is not None:
                params['sl2'] = self.fixedSL[1]
                dsl = self.fixedSL[1] - self.fixedSL[0]
            else:
                dsl = rng.uniform(-1,1)
                params['sl2'] = self.fixedSL[0] + dsl
        
        else:
            assert self.fixedSL[1] is None, 'cannot fix only the second stimulus'
        
        if self.fixedDelay is not None:
            params['delay2_dur'] = self.fixedDelay
        else:
            params['delay2_dur'] = rng.uniform(300,500) # variable delay period
        
        return params

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.
    
        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.
    
        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()
    
        Returns:
            tuple:
    
            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.

        """
        
        delay1_dur = 500
        stim_dur = 200
        delay2_dur = params['delay2_dur']
        delay3_dur = 150
        choice_dur = 750 - delay2_dur
        
        if delay1_dur + 2*stim_dur + delay2_dur + delay3_dur + choice_dur != self.T:
            print('error: trial event times do not add up!')
        
        stim1_on = delay1_dur
        stim2_on = stim1_on + stim_dur + delay2_dur
        go = stim2_on + stim_dur + delay3_dur
        K = self.K
        # ----------------------------------
        # Initialize with input noise
        # ----------------------------------
        vis_noise = self.vis_noise
        mem_noise = self.mem_noise
        fix_noise = 0.2
        
        x_t = np.zeros(self.N_in)
        x_t[-1] += np.sqrt(2 * fix_noise**2) * rng.standard_normal(1)
        x_t[-3:-1] += np.sqrt(2 * vis_noise**2) * rng.standard_normal(2)
        x_t[:-3] += np.sqrt(2 * mem_noise**2) * rng.standard_normal(self.N_in-3)
        
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)
    
        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        dsl = params['dsl']
        dsf = params['dsf']
        choice = params['choice']
        correct = params['correct']
        sf1 = params['sf1']
        sf2 = params['sf2']
        sl1 = params['sl1']
        sl2 = params['sl2']
        task = params['task']
        m_task = params['m_task']
        
        dStim = np.zeros(K)
        for j in range(K):
            if params['m_task'][j]==1:
                dStim[j] = dsl[j]
            else:
                dStim[j] = dsf[j]
        
        # ----------------------------------
        # Compute values
        # ----------------------------------
        for i in range(0, K-1): # constant trial history inputs for the duration of the trial
            if choice[i] != 0:
                x_t[6*i] += 0.2 + dStim[i] # stimulus change amount, based on the chosen task (signals perceptual difficulty)
                x_t[6*i + choice[i]] += 1 # one-hot encoded choice
                if choice[i] == correct[i]:
                    x_t[6*i+5] += 1 # reward
                else:
                    x_t[6*i+5] += -1 # error
                    
        if time > 100:
            y_t[m_task[-1]-1] = 1
            mask_t[:2] *= 1.5
            if m_task[-2] != m_task[-1]:
                switchTrial=True
                mask_t[:2] *= 1.5
            else:
                switchTrial=False
            
        if stim1_on < time < stim1_on + stim_dur: # stim1 input
            x_t[-3] += sf1
            x_t[-2] += sl1
        
        if stim2_on < time < stim2_on + stim_dur: # stim2 input
            x_t[-3] += sf2
            x_t[-2] += sl2        
    
        if time > go:
            y_t[choice[-1] + 1] = 1
            mask_t[2:6] *= 5 # this is when choice unit outputs should be weighed heavily
            if switchTrial:
                mask_t[2:6] *= 1.5
            
        else: # fixation period
            x_t[-1] += 1
            y_t[-1] = 1
        
        return x_t, y_t, mask_t
