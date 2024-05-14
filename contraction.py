#%%
import numpy as np
import pandas as pd
import pickle
from self_history import Task_SH2
from psychrnn.backend.simulation import BasicSimulator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from model_behavior_functions import get_tasks

#%% Function to generate test inputs, stim1 input + long delay

vis_noise, mem_noise, rec_noise = 0, 0, 0 # no noise
K = 10

def get_test_inputs_longdelay(tparams, delay_dur=500):

    x_t = np.zeros((len(tparams), delay_dur + 70, 6*(K-1) + 3))
    assert [vis_noise, mem_noise, rec_noise] == [0, 0, 0], 'Noise not implemented'

    for tp in range(len(tparams)):

        for i in range(K-1):
            choice_i = tparams[tp]['choice'][i]
            drel = tparams[tp]['dsl'][i] if choice_i > 2.5 else tparams[tp]['dsf'][i]
            x_t[tp, :, 6*i] += 0.2 + drel
            x_t[tp, :, 6*i + choice_i] += 1.0
            x_t[tp, :, 6*i + 5] += 1 if choice_i == tparams[tp]['correct'][i] else -1
        
        x_t[tp, 51:70, -3] += tparams[tp]['sf1']
        x_t[tp, 51:70, -2] += tparams[tp]['sl1']
        x_t[tp, :, -1] += 1 

    return x_t

#%% 
N_rec = 200

# Load the model and monkey trial history params
name, testname, folder = \
    'MM1_monkeyB1245', 'alltestinds_noisevis0.8mem0.5rec0.1', 'monkey_choice_model'
    #'SH2_correctA', 'monkeyhist_alltestinds_noisevis0.8mem0.5rec0.1', 'correct_choice_model'

weights_path = f'./{folder}/weights/{name}.npz'
loaded_tparams = pickle.load(open(f'./{folder}/test_data/{name}_{testname}_trialparams.pickle', 'rb'))
loaded_outputs = pickle.load(open(f'./{folder}/test_data/{name}_{testname}_modeloutput.pickle', 'rb'))
modelchoices = np.argmax(loaded_outputs[:, -1, 2:6], axis=1) + 1
modeltasks = get_tasks(modelchoices)

# Select the test indices from loaded tparams based on # preceding rewards
a2NR, a2R = [], []
for i in range(loaded_tparams.shape[0]):
    if loaded_tparams[i]['choice'][-2] != loaded_tparams[i]['correct'][-2]:
        if loaded_tparams[i]['choice'][-3] == loaded_tparams[i]['correct'][-3]:
            a2NR.append(i)
    else:
        if loaded_tparams[i]['choice'][-3] == loaded_tparams[i]['correct'][-3]:
            a2R.append(i)
"""
taskoutdiff = loaded_outputs[:, -1, 0] - loaded_outputs[:, -1, 1] # SL - SF
plt.hist(taskoutdiff[a2NR], bins=20, alpha=0.5, label='a2NR')
plt.hist(taskoutdiff[a2R], bins=20, alpha=0.5, label='a2R')
plt.legend()
"""
#%% Define test inputs
N_per_cond = 1
np.random.seed(2152)
loaded_inds = np.concatenate((
    np.random.choice(np.intersect1d(a2NR, np.where(modeltasks==1)[0]), N_per_cond, replace=False), 
    np.random.choice(np.intersect1d(a2NR, np.where(modeltasks==2)[0]), N_per_cond, replace=False),
    np.random.choice(np.intersect1d(a2R, np.where(modeltasks==1)[0]), N_per_cond, replace=False),
    np.random.choice(np.intersect1d(a2R, np.where(modeltasks==2)[0]), N_per_cond, replace=False)
    ))
stim1s = [(2.5, 2.3), (2.5, 2.7), (2.3, 2.5), (2.7, 2.5)] # (SL, SF)
cm_dict = {'a2NR_SL':cm['Purples'], 'a2NR_SF':cm['Blues'], 'a2R_SL':cm['Oranges'], 'a2R_SF':cm['Greens']}
colors = []
tparams = []
for i, ind in  enumerate(loaded_inds):
    for j in range(len(stim1s)):
        tparams.append({})
        t = len(stim1s)*i + j
        tparams[t]['trial_ind'] = loaded_tparams[ind]['trial_ind']
        tparams[t]['choice'] = loaded_tparams[ind]['choice']
        tparams[t]['correct'] = loaded_tparams[ind]['correct']
        tparams[t]['dsf'] = loaded_tparams[ind]['dsf']
        tparams[t]['dsl'] = loaded_tparams[ind]['dsl']
        tparams[t]['sf1'] = stim1s[j][0]
        tparams[t]['sl1'] = stim1s[j][1]

        # define colors
        if i < N_per_cond:
            cmap = cm_dict['a2NR_SL']
            x = (stim1s[j][0] - 2.0)
            colors.append(cmap(x))
        elif i >= N_per_cond and i < 2*N_per_cond:
            cmap = cm_dict['a2NR_SF']
            x = (stim1s[j][0] - 2.0)
            colors.append(cmap(x))
        elif i >= 2*N_per_cond and i < 3*N_per_cond:
            cmap = cm_dict['a2R_SL']
            x = (stim1s[j][0] - 2.0)
            colors.append(cmap(x))
        elif i >= 3*N_per_cond:
            cmap = cm_dict['a2R_SF']
            x = (stim1s[j][0] - 2.0)
            colors.append(cmap(x))

test_inputs = get_test_inputs_longdelay(tparams, delay_dur=300)

#%% Run the simulation
N_batch = len(tparams)
task = Task_SH2(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=N_batch, K=K)
network_params = task.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise

simulator = BasicSimulator(weights_path=weights_path, params=network_params)
model_output, state_var = simulator.run_trials(test_inputs)

# %% PCA visualization

sv_pca = state_var[:, 50:, :]
pca = PCA(n_components=3)
pca.fit_transform(sv_pca.reshape(-1, sv_pca.shape[-1]))
print(pca.explained_variance_ratio_)
X = sv_pca @ pca.components_.T

# 3D plot
#%matplotlib widget
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(N_batch):
    c = colors[i]
    ax.plot(X[i, :, 0], X[i, :, 1], X[i, :, 2], color=c, zorder=0)
    ax.scatter(X[i, 19, 0], X[i, 19, 1], X[i, 19, 2], marker='v', edgecolor='k',
               color=c, s=50, alpha=1, zorder=1)
    ax.scatter(X[i, 59, 0], X[i, 59, 1], X[i, 59, 2], marker='o', edgecolor='k', 
               color=c, s=50, alpha=1, zorder=1)
    ax.scatter(X[i, -1, 0], X[i, -1, 1], X[i, -1, 2], marker='X', edgecolor='k', 
               color=c, s=50, alpha=1, zorder=1)
# %%
