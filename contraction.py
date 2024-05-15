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

#%% Functions to generate test inputs

vis_noise, mem_noise, rec_noise = 0, 0, 0 # no noise
K = 10

def get_test_inputs_longdelay1(tparams, delay_dur=500):
    '''
    constant trial history inputs (long delay1)
    '''
    x_t = np.zeros((len(tparams), delay_dur, 6*(K-1) + 3))
    assert [vis_noise, mem_noise] == [0, 0], 'Noise not implemented'

    for tp in range(len(tparams)):
        for i in range(K-1):
            choice_i = tparams[tp]['choice'][i]
            drel = tparams[tp]['dsl'][i] if choice_i > 2.5 else tparams[tp]['dsf'][i]
            x_t[tp, :, 6*i] += 0.2 + drel
            x_t[tp, :, 6*i + choice_i] += 1.0
            x_t[tp, :, 6*i + 5] += 1 if choice_i == tparams[tp]['correct'][i] else -1

    x_t[:, :, -1] += 1 # fixation

    return x_t

def get_test_inputs_longdelay2(tparams, delay_dur=500):
    """
    delay1, stim1 input, long delay2
    """
    x_t = get_test_inputs_longdelay1(tparams, delay_dur + 70)

    for tp in range(len(tparams)):
        x_t[tp, 51:70, -3] += tparams[tp]['sf1']
        x_t[tp, 51:70, -2] += tparams[tp]['sl1']

    return x_t

#%% 
N_rec = 200

# Load the model and monkey trial history params
name, testname, folder = \
    'SH2_correctA', 'monkeyhist_alltestinds_noisevis0.8mem0.5rec0.1', 'correct_choice_model'
    #'MM1_monkeyB1245', 'alltestinds_noisevis0.8mem0.5rec0.1', 'monkey_choice_model'
    
weights_path = f'./{folder}/weights/{name}.npz'
loaded_tparams = pickle.load(open(f'./{folder}/test_data/{name}_{testname}_trialparams.pickle', 'rb'))
loaded_outputs = pickle.load(open(f'./{folder}/test_data/{name}_{testname}_modeloutput.pickle', 'rb'))
modelchoices = np.argmax(loaded_outputs[:, -1, 2:6], axis=1) + 1
modeltasks = get_tasks(modelchoices)
SL_inds = np.where(modeltasks==1)[0]
SF_inds = np.where(modeltasks==2)[0]

# Get number of preceding rewards
a1R, a1NR, a2R, a2NR = [], [], [], []
for i in range(loaded_tparams.shape[0]):
    if loaded_tparams[i]['choice'][-2] == loaded_tparams[i]['correct'][-2]:
        if loaded_tparams[i]['choice'][-3] == loaded_tparams[i]['correct'][-3]:
            a2R.append(i)
        else:
            a1R.append(i)
    else:
        if loaded_tparams[i]['choice'][-3] == loaded_tparams[i]['correct'][-3]:
            a1NR.append(i)
        else:
            a2NR.append(i)

taskoutdiff = loaded_outputs[:, -1, 0] - loaded_outputs[:, -1, 1] # SL - SF
plt.figure()
plt.hist(taskoutdiff[a2R], bins=20, alpha=0.5, label='a2R')
plt.hist(taskoutdiff[a1R], bins=20, alpha=0.5, label='a1R')
plt.hist(taskoutdiff[a1NR], bins=20, alpha=0.5, label='a1NR')
plt.hist(taskoutdiff[a2NR], bins=20, alpha=0.5, label='a2NR')
plt.legend()

#%% Randomly select multiple trials for visualization
N_per_cond = 10
np.random.seed(2152)
loaded_inds = np.concatenate((
    np.random.choice(np.intersect1d(a2NR, SL_inds), N_per_cond, replace=False), 
    np.random.choice(np.intersect1d(a2NR, SF_inds), N_per_cond, replace=False),
    np.random.choice(np.intersect1d(a2R, SL_inds), N_per_cond, replace=False),
    np.random.choice(np.intersect1d(a2R, SF_inds), N_per_cond, replace=False)
    ))
#%% Define test inputs for long delay 1, reward hist conditions
tparams = []
colors = []
for i, ind in  enumerate(loaded_inds):
    tparams.append({})
    tparams[i]['trial_ind'] = loaded_tparams[ind]['trial_ind']
    tparams[i]['choice'] = loaded_tparams[ind]['choice']
    tparams[i]['correct'] = loaded_tparams[ind]['correct']
    tparams[i]['dsl'] = loaded_tparams[ind]['dsl']
    tparams[i]['dsf'] = loaded_tparams[ind]['dsf']

    if i//N_per_cond == 0:
        colors.append(cm['Reds'](0.5))
    elif i//N_per_cond == 1:
        colors.append(cm['Purples'](0.5))
    elif i//N_per_cond == 2:
        colors.append(cm['Blues'](0.5))
    elif i//N_per_cond == 3:
        colors.append(cm['Greens'](0.5))

test_inputs = get_test_inputs_longdelay1(tparams, delay_dur=400)

#%% Define test inputs for long delay 2, reward hist x stim1 conditions
stim1s = [(2.5, 2.3), (2.5, 2.7), (2.3, 2.5), (2.7, 2.5)] # (SL, SF)
cm_dict = {'a2NR_SL':cm['Reds'], 'a2NR_SF':cm['Purples'], 'a2R_SL':cm['Blues'], 'a2R_SF':cm['Greens']}
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
        if i//N_per_cond == 0:
            cmap = cm_dict['a2NR_SL']
            x = (stim1s[j][0] - 2.0)
            colors.append(cmap(x))
        elif i//N_per_cond == 1:
            cmap = cm_dict['a2NR_SF']
            x = (stim1s[j][0] - 2.0)
            colors.append(cmap(x))
        elif i//N_per_cond == 2:
            cmap = cm_dict['a2R_SL']
            x = (stim1s[j][0] - 2.0)
            colors.append(cmap(x))
        elif i//N_per_cond == 3:
            cmap = cm_dict['a2R_SF']
            x = (stim1s[j][0] - 2.0)
            colors.append(cmap(x))

test_inputs = get_test_inputs_longdelay2(tparams, delay_dur=300)

#%% Run the simulation
N_batch = len(tparams)
task = Task_SH2(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=N_batch, K=K)
network_params = task.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise

simulator = BasicSimulator(weights_path=weights_path, params=network_params)
model_output, state_var = simulator.run_trials(test_inputs)

# %% PCA for visualization

t_pca = np.arange(state_var.shape[1]-1, state_var.shape[1])
sv_pca = state_var[:, t_pca, :]
pca = PCA(n_components=3)
pca.fit_transform(sv_pca.reshape(-1, sv_pca.shape[-1]))
print(pca.explained_variance_ratio_)
X = state_var @ pca.components_.T

#%% 3D plot
#%matplotlib widget
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(N_batch):
    c = colors[i]
    ax.plot(X[i, 70:, 0], X[i, 70:, 1], X[i, 70:, 2], color=c, zorder=0)
    '''
    ax.scatter(X[i, 69, 0], X[i, 69, 1], X[i, 69, 2], marker='v', edgecolor='k',
               color=c, s=50, alpha=1, zorder=1)
    ax.scatter(X[i, 109, 0], X[i, 109, 1], X[i, 109, 2], marker='o', edgecolor='k', 
               color=c, s=50, alpha=1, zorder=1)
    '''
    ax.scatter(X[i, -1, 0], X[i, -1, 1], X[i, -1, 2], marker='X', edgecolor='k', 
               color=c, s=50, alpha=1, zorder=1)
    
# %%2D plot 
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
for i in range(N_batch):
    c = colors[i]
    ax.plot(X[i, :, 0], X[i, :, 1], color=c, zorder=0)
    ax.scatter(X[i, -1, 0], X[i, -1, 1], marker='X', edgecolor='k', 
               color=c, s=50, alpha=1, zorder=1)
# %%
