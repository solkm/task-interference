#
# Perceptual accuracy during training, from weights (either model)
#
from monkey_model import Task_MM1
from self_history import Task_SH2
from psychrnn.backend.simulation import BasicSimulator
import model_behavior_functions as mf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
    
#%% define function that returns perceptual accuracies as a dataframe

N_rec = 200
vis_noise = 0.8
mem_noise = 0.5
rec_noise = 0.1
K = 10
N = 2500
tParams_new = pd.read_csv('./data_inds/tParams_new.csv')
aR_inds = np.load(open(f'./data_inds/K{K}trainable_aRinds.npy', 'rb'))
aNR_inds = np.load(open(f'./data_inds/K{K}trainable_aNRinds.npy', 'rb'))

seed = 5812
np.random.seed(seed)
test_inds = np.concatenate((np.random.choice(aNR_inds, N, replace=False), 
                            np.random.choice(aR_inds, N, replace=False)))

def get_percAcc_training_df(task, folder, name):

    percAcc_training = {'SL_pAcc_aR':[], 'SL_pAcc_aNR':[], 
                        'SF_pAcc_aR':[], 'SF_pAcc_aNR':[]}
    
    network_params = task.get_task_params()
    network_params['name'] = name
    network_params['N_rec'] = N_rec
    network_params['rec_noise'] = rec_noise
    test_inputs, _, _, trial_params = task.get_trial_batch()

    for tepochs in range(15, 1501, 15):

        weights_path = f'./{folder}/weights/{name}{tepochs}.npz'
        simulator = BasicSimulator(weights_path=weights_path, params=network_params)
        model_output, _ = simulator.run_trials(test_inputs)
        
        model_choice = np.argmax(model_output[:,-1,2:6], axis=1) + 1
        model_task = mf.get_tasks(model_choice)
        
        SL_inds = np.where(model_task==1)[0]
        _, SL_pAcc_aR = mf.get_perc_acc(model_choice[SL_inds[SL_inds>=N]], 
                                        trial_params[SL_inds[SL_inds>=N]])
        _, SL_pAcc_aNR = mf.get_perc_acc(model_choice[SL_inds[SL_inds<N]], 
                                        trial_params[SL_inds[SL_inds<N]])
        
        SF_inds = np.where(model_task==2)[0]
        _, SF_pAcc_aR = mf.get_perc_acc(model_choice[SF_inds[SF_inds>=N]], 
                                        trial_params[SF_inds[SF_inds>=N]])
        _, SF_pAcc_aNR = mf.get_perc_acc(model_choice[SF_inds[SF_inds<N]], 
                                        trial_params[SF_inds[SF_inds<N]])

        percAcc_training['SL_pAcc_aR'].append(SL_pAcc_aR)
        percAcc_training['SL_pAcc_aNR'].append(SL_pAcc_aNR)
        percAcc_training['SF_pAcc_aR'].append(SF_pAcc_aR)
        percAcc_training['SF_pAcc_aNR'].append(SF_pAcc_aNR)
        
        print('epoch ', tepochs)

    percAcc_training_df=pd.DataFrame(percAcc_training)

    return percAcc_training_df

#%% get monkey choice model df

name = 'MM1_monkeyB'
folder = 'monkey_choice_model'
task = Task_MM1(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=2*N, 
                dat=tParams_new, dat_inds=test_inds, K=K, testall=True)
percAcc_training_df = get_percAcc_training_df(task, folder, name)

percAcc_training_df.to_csv(f'./{folder}/{name}_percAccDuringTraining_N{N}seed{seed}.csv', index=False)

#%% get correct choice model df

name = 'SH2_correctA'
folder = 'correct_choice_model'
task = Task_SH2(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=2*N, 
                dat=tParams_new, dat_inds=test_inds, K=K, testall=True)
percAcc_training_df = get_percAcc_training_df(task, folder, name)

percAcc_training_df.to_csv(f'./{folder}/{name}_percAccDuringTraining_N{N}seed{seed}.csv', index=False)

# %% plot

percAcc_training_df = pd.read_csv(f'./{folder}/{name}_percAccDuringTraining_N{N}seed{seed}.csv')
SL_pAcc_aR = percAcc_training_df['SL_pAcc_aR']
SL_pAcc_aNR = percAcc_training_df['SL_pAcc_aNR']
SF_pAcc_aR = percAcc_training_df['SF_pAcc_aR']
SF_pAcc_aNR = percAcc_training_df['SF_pAcc_aNR']

percAccs_aR = 0.5*(SL_pAcc_aR + SF_pAcc_aR)
percAccs_aR_var = 0.5*(SL_pAcc_aR*(1-SL_pAcc_aR)/N + SF_pAcc_aR*(1-SF_pAcc_aR)/N)

percAccs_aNR = 0.5*(SL_pAcc_aNR + SF_pAcc_aNR)
percAccs_aNR_var = 0.5*(SL_pAcc_aNR*(1-SL_pAcc_aNR)/N + SF_pAcc_aNR*(1-SF_pAcc_aNR)/N)
diffs = percAccs_aR - percAccs_aNR
sdev = np.sqrt(percAccs_aR_var + percAccs_aNR_var)

CI = 0.95
z = st.norm.ppf(q=(1+CI)/2)

plt.figure()
c = 'mediumblue'
plt.plot(np.arange(diffs.shape[0]), diffs, color=c, alpha=1)
plt.fill_between(np.arange(diffs.shape[0]), diffs-z*sdev, diffs+z*sdev, 
                 alpha=0.2, color=c, edgecolor='none')
plt.hlines(0.0506, 0, diffs.shape[0], color='grey', lw=1.5, ls='--')
plt.hlines(0, 0, diffs.shape[0], colors='k', ls='--')
plt.ylabel('Perceptual accuracy difference')
plt.xlabel('Training epochs')
plt.tight_layout()

#plt.savefig(f'./{folder}/{name}_percAccDuringTraining_diff.png', dpi=300)
# %%
