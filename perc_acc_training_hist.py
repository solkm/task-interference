#
# Perceptual accuracy during training, from training history (correct choice model only)
#
import os
os.chdir('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/task-interference')
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import scipy.stats as st
import model_behavior_functions as mf

#%% calculate perceptual accuracy, 1 datapoint per epoch

name = 'SH2_correctA'
folder = 'correct_choice_model'
K = 10
hist = dict(np.load(open(f'./{folder}/{name}_history.npz', 'rb'), allow_pickle=True))

N_batch = hist['choice'].shape[0]
N_epochs = hist['choice'].shape[1] - (K-1)

percAcc_training = {'pAcc_aR':[], 'pAcc_aNR':[]}

for p in range(N_epochs):
    model_choice = hist['choice'][:, K-1 + p]
    dsl = hist['dsl'][:, K-1 + p]
    dsf = hist['dsf'][:, K-1 + p]
    correct_choice = hist['correct'][:, K-1 + p]
    
    correct_perc, _ = mf.get_perc_acc(model_choice, trial_params=None, 
                                            dsl=dsl, dsf=dsf)
    
    aR_inds = np.where(model_choice==correct_choice)[0] + 1
    del_ind = np.where(aR_inds >= N_batch)
    aR_inds = np.delete(aR_inds, del_ind)
    aNR_inds = np.delete(np.arange(N_batch), aR_inds)
    pAcc_aR = np.count_nonzero(correct_perc[aR_inds])/aR_inds.shape[0]
    pAcc_aNR = np.count_nonzero(correct_perc[aNR_inds])/aNR_inds.shape[0]
    
    percAcc_training['pAcc_aR'].append(pAcc_aR)
    percAcc_training['pAcc_aNR'].append(pAcc_aNR)

pAcc_aR_ = np.array(percAcc_training['pAcc_aR'])
pAcc_aNR_ = np.array(percAcc_training['pAcc_aNR'])

#%% sliding window average over epochs

n_avg = 15
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

#%% stacked subplots

t = st.t.ppf(q=0.975, df=n_avg-1)

fig, axs = plt.subplots(2, sharex=True, figsize=(5,4))

axs[0].fill_between(np.arange(n_steps), 
                    pAcc_aR_sw[0, :] - t/np.sqrt(n_avg)*pAcc_aR_sw[1, :], 
                    pAcc_aR_sw[0, :] + t/np.sqrt(n_avg)*pAcc_aR_sw[1, :], 
                    color='b', alpha=0.2, edgecolor='none')
axs[0].fill_between(np.arange(n_steps), 
                    pAcc_aNR_sw[0, :] - t/np.sqrt(n_avg)*pAcc_aNR_sw[1, :], 
                    pAcc_aNR_sw[0, :] + t/np.sqrt(n_avg)*pAcc_aNR_sw[1, :], 
                    color='r', alpha=0.2, edgecolor='none')
axs[0].plot(pAcc_aR_sw[0, :], c='g', label='after reward')
axs[0].plot(pAcc_aNR_sw[0, :], c='r', label='after non-reward')
axs[0].set_ylabel('Perceptual accuracy')
axs[0].legend()

axs[1].plot(pAcc_diff_sw[0,:])
axs[1].fill_between(np.arange(n_steps), 
                    pAcc_diff_sw[0,:] - t/np.sqrt(n_avg)*pAcc_diff_sw[1,:], 
                    pAcc_diff_sw[0,:] + t/np.sqrt(n_avg)*pAcc_diff_sw[1,:], 
                    alpha=0.2)
axs[1].hlines(0.0506, 0, n_steps, color='grey', ls='--', label='monkeys')
axs[1].hlines(0, 0, n_steps, color='k', ls='--', zorder=0)
axs[1].set_ylim(-0.03, 0.06)
axs[1].set_xlabel('Training epochs')
axs[1].set_ylabel('Difference')
axs[1].legend()

plt.tight_layout()

rcParams['pdf.fonttype']=42
rcParams['pdf.use14corefonts']=True
#plt.savefig(f'./{folder}/{name}_percAccDuringTraining_sw{n_avg}.pdf', dpi=300, transparent=True)
# %%
