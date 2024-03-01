#
# Perceptual accuracy during training, from weights (either model)
#%%
from monkey_model import Task_MM1
from self_history import Task_SH2
from psychrnn.backend.simulation import BasicSimulator
import model_behavior_functions as mf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib import rcParams
    
# define function that returns perceptual accuracies as a dataframe

N_rec = 200
vis_noise = 0.8
mem_noise = 0.5
rec_noise = 0.1
K = 10
N = 2000
tParams_new = pd.read_csv('./data_inds/tParams_new.csv')
aR_inds = np.load(open(f'./data_inds/K{K}trainable_aRinds.npy', 'rb'))
aNR_inds = np.load(open(f'./data_inds/K{K}trainable_aNRinds.npy', 'rb'))

seed = 23
np.random.seed(seed) # set seed
test_inds = np.concatenate((np.random.choice(aNR_inds, N, replace=False), 
                            np.random.choice(aR_inds, N, replace=False)))

def get_percAcc_training_df(task, folder, name, redraw=False):

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
        del model_output, model_choice, model_task

        if tepochs!=1500 and redraw:
            test_inds = np.concatenate((np.random.choice(aNR_inds, N, replace=False), 
                                        np.random.choice(aR_inds, N, replace=False)))
            task.dat_inds = test_inds
            test_inputs, _, _, trial_params = task.get_trial_batch()
            #print('test_ind[0]: ', trial_params[0]['trial_ind']) # test if redraw works

    percAcc_training_df = pd.DataFrame(percAcc_training)

    return percAcc_training_df

#%% get monkey choice model df

name = 'MM1_monkeyB'
folder = 'monkey_choice_model'
task = Task_MM1(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=2*N, 
                dat=tParams_new, dat_inds=test_inds, K=K, testall=True)
redraw = True
percAcc_training_df = get_percAcc_training_df(task, folder, name, redraw=redraw)

percAcc_training_df.to_csv(f'./{folder}/{name}_percAccDuringTraining_N{N}seed{seed}_redraw{redraw}.csv', index=False)

# %% comparison with correct choice model calculated directly from its training history
# with sliding window averaging
    
sw1 = 100
diff1 = np.load(open(f'./correct_choice_model/SH2_correctA_percAccDuringTraining_hist_sw{sw1}.npz', 'rb'))
diff1 = diff1['pAcc_diff_sw']
epochs1 = np.arange(0, 1500-sw1) + sw1/2
t1 = st.t.ppf(q=0.975, df=sw1-1)

plt.hlines(0, 0, 1500, colors='k', ls='--')
plt.hlines(0.0506, 0, 1500, color='grey', lw=1.5, ls='--', label='monkeys')
plt.plot(epochs1, diff1[0], color='darkblue', label='correct choice model')
plt.fill_between(epochs1, diff1[0]-t1*diff1[1], diff1[0]+t1*diff1[1],
                 alpha=0.2, color='darkblue', edgecolor='none')

CI_or_trace = 'trace' # 'CI' or 'trace'
sw2 = 6
N = 2500 #5000 #2000
seed = 5812 #56 #23
redraw = False # True
df2 = pd.read_csv(f'./monkey_choice_model/MM1_monkeyB_percAccDuringTraining_N{N}seed{seed}_redraw{redraw}.csv')

SL_pAcc_aR, SL_pAcc_aNR = df2['SL_pAcc_aR'], df2['SL_pAcc_aNR']
SF_pAcc_aR, SF_pAcc_aNR = df2['SF_pAcc_aR'], df2['SF_pAcc_aNR']
pAcc_aR = 0.5*(SL_pAcc_aR + SF_pAcc_aR)
pAcc_aNR = 0.5*(SL_pAcc_aNR + SF_pAcc_aNR)
diff = pAcc_aR - pAcc_aNR

diff2 = mf.sliding_window_avg(diff, sw2, sem=None)
epochs2 = (np.arange(1, 101-sw2) + sw2/2)*15

# plot raw data or CI and sliding window average
if CI_or_trace=='CI':
    t2 = st.t.ppf(q=0.975, df=sw2-1)
    plt.fill_between(epochs2, diff2[0]-t2*diff2[1], diff2[0]+t2*diff2[1], 
                     alpha=0.2, color='darkorange', edgecolor='none')
else:
    plt.plot(np.arange(15, 1501, 15), diff, color='darkorange', alpha=0.5)

plt.plot(epochs2, diff2[0], color='darkorange', label='monkey choice model')


plt.legend()
plt.xlabel('Training epochs')
plt.ylabel('Perceptual accuracy difference')
plt.tight_layout()

rcParams['pdf.fonttype']=42
rcParams['pdf.use14corefonts']=True
#plt.savefig(f'./SH2_correctA_pAccHist_sw{sw1}_vs_MM1_monkeyB_N2500seed5812sw{sw2}.pdf', dpi=300, transparent=True)

#%% get correct choice model df (monkey history inputs)

name = 'SH2_correctA'
folder = 'correct_choice_model'
task = Task_SH2(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=2*N, 
                dat=tParams_new, dat_inds=test_inds, K=K, testall=True)
percAcc_training_df = get_percAcc_training_df(task, folder, name)

percAcc_training_df.to_csv(f'./{folder}/{name}_percAccDuringTraining_monkeyhist_N{N}seed{seed}.csv', index=False)

#%% plot a comparison of the two models, both tested with monkey history inputs
# optional sliding window averaging

sliding_window = True
n_avg = 6
 
Ns = [2500, 2500]
seeds = [5812, 5812]
colors = ['darkblue', 'darkorange']
dfs = [pd.read_csv(f'./correct_choice_model/SH2_correctA_percAccDuringTraining_monkeyhist_N{Ns[0]}seed{seeds[0]}.csv'), 
       pd.read_csv(f'./monkey_choice_model/MM1_monkeyB_percAccDuringTraining_N{Ns[1]}seed{seeds[1]}.csv')]

for i in range(len(dfs)):

    df = dfs[i]

    SL_pAcc_aR, SL_pAcc_aNR = df['SL_pAcc_aR'], df['SL_pAcc_aNR']
    SF_pAcc_aR, SF_pAcc_aNR = df['SF_pAcc_aR'], df['SF_pAcc_aNR']

    percAccs_aR = 0.5*(SL_pAcc_aR + SF_pAcc_aR)
    percAccs_aR_var = 0.5*(SL_pAcc_aR*(1-SL_pAcc_aR) + SF_pAcc_aR*(1-SF_pAcc_aR))

    percAccs_aNR = 0.5*(SL_pAcc_aNR + SF_pAcc_aNR)
    percAccs_aNR_var = 0.5*(SL_pAcc_aNR*(1-SL_pAcc_aNR) + SF_pAcc_aNR*(1-SF_pAcc_aNR))
    diffs = percAccs_aR - percAccs_aNR
    sem = np.sqrt((percAccs_aR_var + percAccs_aNR_var)/Ns[i])

    if sliding_window:
        diffs_sw = mf.sliding_window_avg(diffs, n_avg, sem=sem) # sem=sem or None
        diffs = diffs_sw[0]
        sem = diffs_sw[1]

    CI = 0.95
    z = st.norm.ppf(q=(1+CI)/2)

    plt.plot(np.arange(diffs.shape[0]), diffs, color=colors[i], alpha=1)
    plt.fill_between(np.arange(diffs.shape[0]), diffs-z*sem, diffs+z*sem, 
                    alpha=0.2, color=colors[i], edgecolor='none')
    if i==0:
        plt.hlines(0.0506, 0, diffs.shape[0], color='grey', lw=1.5, ls='--')
        plt.hlines(0, 0, diffs.shape[0], colors='k', ls='--')
    plt.ylabel('Perceptual accuracy difference')
    plt.xlabel('Training epochs')
    plt.tight_layout()

# %%
