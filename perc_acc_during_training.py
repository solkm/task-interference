# *** NEEDS TO BE EDITED ***
import os
os.chdir('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/MultipleOutputs')
import sys
sys.path.append('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject')
from selfHistory import RNN_SH2, Task_SH2 # RNN_SH1, Task_SH1 
#from TaskStructures import FourAFC_ReportTask
#from psychrnn.backend.models.basic import Basic
from psychrnn.backend.simulation import BasicSimulator
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
#import scipy.io
#import tensorflow as tf
import pickle
import model_functions as mf
import scipy.stats as st

#%% load training history
name = 'SH2_correctA'
folder = 'SelfHistory/SH2/correctA'
K = 10
hist = dict(np.load(open(f'./{folder}/{name}_history.npz', 'rb'), allow_pickle=True))
#%% perc acc during training: 1 datapoint per epoch
N_batch = hist['choice'].shape[0]
N_epochs = hist['choice'].shape[1] - (K-1)

percAcc_training = {'pAcc_aR':[], 'pAcc_aNR':[]}

for p in range(N_epochs):
    model_choice = hist['choice'][:, K-1 + p]
    dsl = hist['dsl'][:, K-1 + p]
    dsf = hist['dsf'][:, K-1 + p]
    correct_choice = hist['correct'][:, K-1 + p]
    
    correctPerception = np.zeros(N_batch)
    
    for i in range(N_batch):
        if model_choice[i]==1 and dsf[i]>0:
            correctPerception[i]=1
        elif model_choice[i]==2 and dsf[i]<0:
            correctPerception[i]=1
        elif model_choice[i]==3 and dsl[i]<0:
            correctPerception[i]=1
        elif model_choice[i]==4 and dsl[i]>0:
            correctPerception[i]=1
    
    aR_inds = np.where(model_choice==correct_choice)[0] + 1
    del_ind = np.where(aR_inds >= N_batch)
    aR_inds = np.delete(aR_inds, del_ind)
    aNR_inds = np.delete(np.arange(N_batch), aR_inds)
    pAcc_aR = np.count_nonzero(correctPerception[aR_inds])/aR_inds.shape[0]
    pAcc_aNR = np.count_nonzero(correctPerception[aNR_inds])/aNR_inds.shape[0]
    
    percAcc_training['pAcc_aR'].append(pAcc_aR)
    percAcc_training['pAcc_aNR'].append(pAcc_aNR)

# plots
pAcc_aR_ = np.array(percAcc_training['pAcc_aR'])
pAcc_aNR_ = np.array(percAcc_training['pAcc_aNR'])

plt.figure()
plt.plot(pAcc_aR_, c='b', alpha=0.6, label='aR')
plt.plot(pAcc_aNR_, c='r', alpha=0.6, label='aNR')
plt.legend()
plt.xlabel('Training epoch')
plt.ylabel('Perceptual accuracy')
plt.tight_layout()
#plt.savefig(f'./{folder}/{name}_percAccDuringTraining.png', dpi=300)

plt.figure()
plt.plot(pAcc_aR_ - pAcc_aNR_, c='darkblue', alpha=0.6)
plt.xlabel('Training epoch')
plt.ylabel('Perceptual accuracy difference\n(after R - after NR)')
plt.tight_layout()
#plt.savefig(f'./{folder}/{name}_percAccDiffDuringTraining.png', dpi=300)

#%% averaging pAcc over epochs
n_steps = 100
pAcc_aR_s = np.zeros((2, n_steps))
pAcc_aNR_s = np.zeros((2, n_steps))

n_avg = int(pAcc_aR_.shape[0]/n_steps)
t = st.t.ppf(q=0.95, df=n_avg-1)

for i in range(n_steps):
    pAcc_aR_s[0, i] = np.mean(pAcc_aR_[i*n_avg:(i+1)*n_avg])
    pAcc_aR_s[1, i] = np.std(pAcc_aR_[i*n_avg:(i+1)*n_avg])
    pAcc_aNR_s[0, i] = np.mean(pAcc_aNR_[i*n_avg:(i+1)*n_avg])
    pAcc_aNR_s[1, i] = np.std(pAcc_aNR_[i*n_avg:(i+1)*n_avg])

plt.fill_between(np.arange(n_steps), pAcc_aR_s[0, :] - t/np.sqrt(n_avg)*pAcc_aR_s[1, :], pAcc_aR_s[0, :] + t/np.sqrt(n_avg)*pAcc_aR_s[1, :], color='b', alpha=0.2, edgecolor='none')
plt.fill_between(np.arange(n_steps), pAcc_aNR_s[0, :] - t/np.sqrt(n_avg)*pAcc_aNR_s[1, :], pAcc_aNR_s[0, :] + t/np.sqrt(n_avg)*pAcc_aNR_s[1, :], color='r', alpha=0.2, edgecolor='none')
plt.plot(pAcc_aR_s[0, :], c='b')
plt.plot(pAcc_aNR_s[0, :], c='r')

pAcc_diff_s = np.zeros((2, n_steps))
pAcc_diff_s[0,:] = pAcc_aR_s[0,:] - pAcc_aNR_s[0,:]
pAcc_diff_s[1,:] = np.sqrt(pAcc_aR_s[1,:]**2 + pAcc_aNR_s[1,:]**2)

plt.figure()
plt.plot(pAcc_diff_s[0,:])
plt.fill_between(np.arange(n_steps), pAcc_diff_s[0,:] - t/np.sqrt(n_avg)*pAcc_diff_s[1,:], pAcc_diff_s[0,:] + t/np.sqrt(n_avg)*pAcc_diff_s[1,:], alpha=0.2)

#%% sliding window, avg pAcc over epochs

n_avg = 200
n_steps = pAcc_aR_.shape[0] - n_avg
pAcc_aR_sw = np.zeros((2, n_steps))
pAcc_aNR_sw = np.zeros((2, n_steps))
pAcc_diff_sw = np.zeros((2, n_steps))

for i in range(n_steps):
    pAcc_aR_sw[0, i] = np.mean(pAcc_aR_[i:i+n_avg])
    pAcc_aR_sw[1, i] = np.std(pAcc_aR_[i:i+n_avg])
    pAcc_aNR_sw[0, i] = np.mean(pAcc_aNR_[i:i+n_avg])
    pAcc_aNR_sw[1, i] = np.std(pAcc_aNR_[i:i+n_avg])
    diffs = pAcc_aR_[i:i+n_avg] - pAcc_aNR_[i:i+n_avg]
    pAcc_diff_sw[0, i] = np.mean(diffs)
    pAcc_diff_sw[1, i] = np.std(diffs)

t = st.t.ppf(q=0.95, df=n_avg-1)

rcParams['pdf.fonttype']=42
rcParams['pdf.use14corefonts']=True

plt.figure(figsize=(5,3))
plt.fill_between(np.arange(n_steps), pAcc_aR_sw[0, :] - t/np.sqrt(n_avg)*pAcc_aR_sw[1, :], pAcc_aR_sw[0, :] + t/np.sqrt(n_avg)*pAcc_aR_sw[1, :], color='b', alpha=0.2, edgecolor='none')
plt.fill_between(np.arange(n_steps), pAcc_aNR_sw[0, :] - t/np.sqrt(n_avg)*pAcc_aNR_sw[1, :], pAcc_aNR_sw[0, :] + t/np.sqrt(n_avg)*pAcc_aNR_sw[1, :], color='r', alpha=0.2, edgecolor='none')
plt.plot(pAcc_aR_sw[0, :], c='b', label='after reward')
plt.plot(pAcc_aNR_sw[0, :], c='r', label='after non-reward')
plt.xlabel('Time during training')
plt.ylabel('Perceptual accuracy')
plt.legend()
plt.tight_layout()
#plt.savefig(f'./{folder}/{name}_percAccDuringTraining_SWavg{n_avg}epochs.pdf', dpi=300, transparent=True)

plt.figure(figsize=(5,3))
plt.plot(pAcc_diff_sw[0,:])
plt.fill_between(np.arange(n_steps), pAcc_diff_sw[0,:] - t/np.sqrt(n_avg)*pAcc_diff_sw[1,:], pAcc_diff_sw[0,:] + t/np.sqrt(n_avg)*pAcc_diff_sw[1,:], alpha=0.2)
plt.hlines(0.0506, 0, n_steps, color='orangered', label='monkeys')
plt.hlines(0, 0, n_steps, color='k', ls='--', zorder=0)
plt.ylim(-0.03, 0.055)
plt.xlabel('Time during training')
plt.ylabel('Perceptual accuracy difference\n(after R - after NR)')
plt.legend()
plt.tight_layout()
#plt.savefig(f'./{folder}/{name}_percAccDiffDuringTraining_SWavg{n_avg}epochs.pdf', dpi=300, transparent=True)

#%% stacked subplots

rcParams['pdf.fonttype']=42
rcParams['pdf.use14corefonts']=True

fig, axs = plt.subplots(2, sharex=True, figsize=(5,4))

axs[0].fill_between(np.arange(n_steps), pAcc_aR_sw[0, :] - t/np.sqrt(n_avg)*pAcc_aR_sw[1, :], pAcc_aR_sw[0, :] + t/np.sqrt(n_avg)*pAcc_aR_sw[1, :], color='b', alpha=0.2, edgecolor='none')
axs[0].fill_between(np.arange(n_steps), pAcc_aNR_sw[0, :] - t/np.sqrt(n_avg)*pAcc_aNR_sw[1, :], pAcc_aNR_sw[0, :] + t/np.sqrt(n_avg)*pAcc_aNR_sw[1, :], color='r', alpha=0.2, edgecolor='none')
axs[0].plot(pAcc_aR_sw[0, :], c='b', label='after reward')
axs[0].plot(pAcc_aNR_sw[0, :], c='r', label='after non-reward')
axs[0].set_ylabel('Perceptual accuracy')
axs[0].legend()

axs[1].plot(pAcc_diff_sw[0,:])
axs[1].fill_between(np.arange(n_steps), pAcc_diff_sw[0,:] - t/np.sqrt(n_avg)*pAcc_diff_sw[1,:], pAcc_diff_sw[0,:] + t/np.sqrt(n_avg)*pAcc_diff_sw[1,:], alpha=0.2)
axs[1].hlines(0.0506, 0, n_steps, color='orangered', label='monkeys')
axs[1].hlines(0, 0, n_steps, color='k', ls='--', zorder=0)
axs[1].set_ylim(-0.03, 0.06)
axs[1].set_xlabel('Time during training')
axs[1].set_ylabel('Perceptual accuracy difference\n(after R - after NR)')
axs[1].legend()

plt.tight_layout()

#plt.savefig(f'./{folder}/{name}_percAccAndDiffDuringTraining_SWavg{n_avg}epochs.pdf', dpi=300, transparent=True)