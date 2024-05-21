#%%
import numpy as np
import pandas as pd
import pickle
from self_history import Task_SH2
from psychrnn.backend.simulation import BasicSimulator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib.colors import Normalize
from model_behavior_functions import get_tasks
import scipy.linalg
import itertools

#%% Functions to generate test inputs, and others

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

def get_jacobian_relu(x, W_rec, network_params, returnF=False):
    """
    x: state vector at time t
    """
    tau = network_params['tau']
    N = x.shape[0]
    F = np.zeros((N, N))
    for i in range(N):
        if x[i] >= 0:
            F[i, i] = 1
        else:
            F[i, i] = 0
    J = 1/tau * (- np.eye(N) + W_rec @ F)
    
    if returnF:
        return J, F
    return J

def xdot(x, W_rec, W_in, u, b, network_params):
    """
    x: state vector at time t
    """
    tau = network_params['tau']
    N = x.shape[0]
    xdot = 1/tau * (-x + W_rec @ np.maximum(0, x) + W_in @ u + b)
    
    return xdot

def get_jacobian_relu_discrete(x, W_rec, network_params, returnF=False):
    """
    x: state vector at timestep t
    """
    alpha = network_params['alpha']
    N = x.shape[0]
    F = np.zeros((N, N))
    for i in range(N):
        if x[i] >= 0:
            F[i, i] = 1
        else:
            F[i, i] = 0
    J = (1 - alpha) * np.eye(N) + alpha * W_rec @ F
    
    if returnF:
        return J, F
    return J


#%% Load model weights and existing test data
N_rec = 200

# Load the model and monkey trial history params
name, testname, folder = \
    'MM1_monkeyB1245', 'alltestinds_noisevis0.8mem0.5rec0.1', 'monkey_choice_model'
    #'SH2_correctA', 'monkeyhist_alltestinds_noisevis0.8mem0.5rec0.1', 'correct_choice_model'
    #
print(name)
weights_path = f'./{folder}/weights/{name}.npz'
loaded_tparams = pickle.load(open(f'./{folder}/test_data/{name}_{testname}_trialparams.pickle', 'rb'))
loaded_outputs = pickle.load(open(f'./{folder}/test_data/{name}_{testname}_modeloutput.pickle', 'rb'))
modelchoices = np.argmax(loaded_outputs[:, -1, 2:6], axis=1) + 1
modeltasks = get_tasks(modelchoices)

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
'''
taskoutdiff_loaded = loaded_outputs[:, -1, 0] - loaded_outputs[:, -1, 1] # SL - SF
plt.figure()
plt.hist(taskoutdiff_loaded[a2R], bins=20, alpha=0.5, label='a2R')
plt.hist(taskoutdiff_loaded[a1R], bins=20, alpha=0.5, label='a1R')
plt.hist(taskoutdiff_loaded[a1NR], bins=20, alpha=0.5, label='a1NR')
plt.hist(taskoutdiff_loaded[a2NR], bins=20, alpha=0.5, label='a2NR')
plt.legend()
'''
#%% Define test inputs for long delay 1, reward hist conditions

N_per_cond = 20
np.random.seed(2152)
loaded_inds = np.concatenate((
    np.random.choice(a2NR, N_per_cond, replace=False), 
    np.random.choice(a2R, N_per_cond, replace=False),
    np.random.choice(a1NR, N_per_cond, replace=False),
    np.random.choice(a1R, N_per_cond, replace=False)
    ))

tparams = []
for i, ind in  enumerate(loaded_inds):
    tparams.append({})
    tparams[i]['trial_ind'] = loaded_tparams[ind]['trial_ind']
    tparams[i]['choice'] = loaded_tparams[ind]['choice']
    tparams[i]['correct'] = loaded_tparams[ind]['correct']
    tparams[i]['dsl'] = loaded_tparams[ind]['dsl']
    tparams[i]['dsf'] = loaded_tparams[ind]['dsf']

delaylabel = 'delay1'
dur = 300
test_inputs = get_test_inputs_longdelay1(tparams, delay_dur=dur)

#%% Define test inputs for long delay 2, reward hist x stim1 conditions

N_per_cond = 20
np.random.seed(2152)
loaded_inds = np.concatenate((
    np.random.choice(a2NR, N_per_cond, replace=False), 
    np.random.choice(a2R, N_per_cond, replace=False),
    np.random.choice(a1NR, N_per_cond, replace=False),
    np.random.choice(a1R, N_per_cond, replace=False)
    ))

stim1s = [(2.5, 2.1), (2.5, 2.9), (2.1, 2.5), (2.9, 2.5), (2.5, 2.5)] # (SL, SF)
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
        tparams[t]['sl1'] = stim1s[j][0]
        tparams[t]['sf1'] = stim1s[j][1]
        
delaylabel = 'delay2'
dur = 400
test_inputs = get_test_inputs_longdelay2(tparams, delay_dur=dur)

#%% Run the simulation

N_batch = len(tparams)
task = Task_SH2(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=N_batch, K=K)
network_params = task.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise

simulator = BasicSimulator(weights_path=weights_path, params=network_params)
model_output, state_var = simulator.run_trials(test_inputs)

# %% Color by task output difference

taskoutdiff_longdelay = model_output[:, -1, 0] - model_output[:, -1, 1] # SL - SF
colors_taskoutdiff = cm.coolwarm((taskoutdiff_longdelay + 1)/2)

# %% PCA for visualization, on all trials

tpca_label = 'last' #'stim1off' #
t_pca = np.arange(state_var.shape[1]-1, state_var.shape[1]) # np.arange(70, 71) 
sv_pca = state_var[:, t_pca, :]
pca = PCA(n_components=3)
pca.fit_transform(sv_pca.reshape(-1, sv_pca.shape[-1]))
print(pca.explained_variance_ratio_)
X = state_var @ pca.components_.T

#%% 3D plot, all trials, color by task output difference

#%matplotlib widget
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(N_batch):
    c = colors_taskoutdiff[i]
    ax.plot(X[i, :, 0], X[i, :, 1], X[i, :, 2], color=c, zorder=0, alpha=0.25)
    ax.scatter(X[i, -1, 0], X[i, -1, 1], X[i, -1, 2], marker='X', edgecolor='k', 
               lw=0.5, color=c, s=80, alpha=1, zorder=1)
    
#%% 2D plot, all trials, color by task output difference

fig, ax = plt.subplots(figsize=(8, 8))
rcParams['font.size'] = 12
for i in range(N_batch):
    c = colors_taskoutdiff[i]
    ax.plot(X[i, :, 0], X[i, :, 1], color=c, zorder=0, alpha=0.25)
    ax.scatter(X[i, 50, 0], X[i, 50, 1], marker='s', edgecolor='k', lw=0.4,
               color=c, s=20, alpha=0.7, zorder=1)
    ax.scatter(X[i, -1, 0], X[i, -1, 1], marker='X', edgecolor='k', lw=0.4,
               color=c, s=80, alpha=1, zorder=2)
ax.set_aspect('equal')
ax.set(xlabel='PC1', ylabel='PC2', 
       xticks=np.arange(-8, 13, 4), yticks=np.arange(-12, 1, 4)
       )
cbar = plt.colorbar(cm.ScalarMappable(cmap=cm.coolwarm, norm=Normalize(-1, 1)), 
                    ax=ax, fraction=0.03, label='Task output difference (L - F)')
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
plt.tight_layout()
#plt.savefig(f'./contraction_figs/{name}_{delaylabel}dur{dur}_tPCA{tpca_label}.png', dpi=300)

#%% Choose a subset of task certainty conditions to plot

np.random.seed(2152)
diff_dig = np.digitize(taskoutdiff_longdelay, [-1, -0.8, -0.4, -0.2, 0.2, 0.4, 0.8, 1])
toplot = [np.random.choice(np.intersect1d(np.where(diff_dig==i)[0], 
            np.arange(0, diff_dig.shape[0], len(stim1s)))) for i in [1, 3]] #[1, 3, 5, 7] 
print(taskoutdiff_longdelay[toplot])

# PCA on the trials defined by this subset
tpca_label = 'stim1offplottedtrials'
t_pca = np.arange(70, 71)
pca_trials = np.concatenate([np.arange(i, i+len(stim1s)) for i in toplot])
sv_pca = state_var[pca_trials, t_pca, :]
pca = PCA(n_components=3)
pca.fit_transform(sv_pca.reshape(-1, sv_pca.shape[-1]))
print(pca.explained_variance_ratio_)
X = state_var @ pca.components_.T

#%% 2D plot for a subset of trials, color by relevant/irrelevant stim1

fig, ax = plt.subplots(figsize=(8, 8))
rcParams['font.size'] = 12
colorby = 'rel' # 'irrel' #

for i, ind in enumerate(toplot):

    c1 = colors_taskoutdiff[ind]
    rel = 'L' if taskoutdiff_longdelay[ind] > 0 else 'F'

    ax.scatter(X[ind, 50, 0], X[ind, 50, 1], marker='s', edgecolor='k', 
               lw=0.2, color=c1, s=60, alpha=1, zorder=1)

    for j in range(len(stim1s)):

        if (rel=='L' and colorby=='rel') or (rel=='F' and colorby=='irrel'):
            c2 = cm.plasma((stim1s[j][0] - 2))
            c3 = cm.plasma((stim1s[j][1] - 2))
        elif (rel=='F' and colorby=='rel') or (rel=='L' and colorby=='irrel'):
            c2 = cm.plasma((stim1s[j][1] - 2))
            c3 = cm.plasma((stim1s[j][0] - 2))

        ax.plot(X[ind+j, 50:, 0], X[ind+j, 50:, 1], color=c2, zorder=0, alpha=0.8)
        ax.scatter(X[ind+j, 70, 0], X[ind+j, 70, 1], marker='>', edgecolor=c3, 
                   lw=1.2, color=c2, s=80, alpha=1, zorder=2)
        ax.scatter(X[ind+j, 110, 0], X[ind+j, 110, 1], marker='o', edgecolor=c3, 
                   lw=1.2, color=c2, s=80, alpha=1, zorder=2)
        ax.scatter(X[ind+j, -1, 0], X[ind+j, -1, 1], marker='X', edgecolor='k', 
                   lw=0.2, color=c1, s=80, alpha=1, zorder=1)
 
#ax.set_aspect('equal')
ax.set(xlabel='PC1', ylabel='PC2')
cbar = plt.colorbar(cm.ScalarMappable(cmap=cm.plasma, norm=Normalize(2, 3)), 
                    ax=ax, fraction=0.02, label='Stimulus feature')
cbar.set_ticks([2, 2.5, 3])
plt.tight_layout()

taskdiffs = np.round(taskoutdiff_longdelay[toplot], 3)
#plt.savefig(f'./contraction_figs/{name}_{delaylabel}dur{dur}_tPCA{tpca_label}_{taskdiffs}.png', dpi=300)

# %% 3D plot for a subset of trials, color by relevant/irrelevant stim1

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
rcParams['font.size'] = 12
colorby = 'rel'
# %matplotlib widget

for i, ind in enumerate(toplot):

    c1 = colors_taskoutdiff[ind]
    rel = 'L' if taskoutdiff_longdelay[ind] > 0 else 'F'

    ax.scatter(X[ind, 50, 0], X[ind, 50, 1], X[ind, 50, 2], marker='s', edgecolor='k', 
               lw=0.4, color=c1, s=40, alpha=1, zorder=1)

    for j in range(len(stim1s)):

        if rel=='L' and colorby=='rel':
            c2 = cm.plasma((stim1s[j][0] - 2))
        elif rel=='F' and colorby=='rel':
            c2 = cm.plasma((stim1s[j][1] - 2))

        ax.plot(X[ind+j, 50:, 0], X[ind+j, 50:, 1], X[ind+j, 50:, 2], 
                    color=c2, zorder=0, alpha=0.8)
        ax.scatter(X[ind+j, 70, 0], X[ind+j, 70, 1], X[ind+j, 70, 2], 
                    marker='>', edgecolor='k', lw=0.4, color=c2, s=40, alpha=1, zorder=1)
        ax.scatter(X[ind+j, 110, 0], X[ind+j, 110, 1], X[ind+j, 110, 2], 
            marker='o', edgecolor='k', lw=0.4, color=c2, s=40, alpha=1, zorder=1)
        ax.scatter(X[ind+j, -1, 0], X[ind+j, -1, 1], X[ind+j, -1, 2], 
                    marker='X', edgecolor='k', lw=0.4, color='grey', s=40, alpha=1, zorder=2)


#%% Convergence after stim1? Single context condition example

ind0 = toplot[0]
print(taskoutdiff_longdelay[ind0])
rel = 'L' if taskoutdiff_longdelay[ind0] > 0 else 'F'
rel_ind = 0 if rel=='L' else 1

stim_pair_inds = list(itertools.combinations(np.arange(len(stim1s)), 2))
stim_pair_diffs = []
colormaps = [cm.viridis, cm.viridis] # rel diff, irrel diff
tmax = 250
fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

for i in range(len(stim_pair_inds)):
    diff = np.round(np.abs(np.array(stim1s[stim_pair_inds[i][0]]) \
                           - np.array(stim1s[stim_pair_inds[i][1]])), 1)
    stim_pair_diffs.append(diff)

    if np.all(diff != 0):
        c_rel = colormaps[0](diff[rel_ind])
        c_irrel = colormaps[1](diff[1-rel_ind])
    else:
        if diff[rel_ind] != 0:
            c_rel = colormaps[0](diff[rel_ind])
            c_irrel = colormaps[1](0)
        else:
            c_rel = colormaps[0](0)
            c_irrel = colormaps[1](diff[1-rel_ind])
    
    x1 = state_var[ind0 + stim_pair_inds[i][0], :tmax, :]
    x2 = state_var[ind0 + stim_pair_inds[i][1], :tmax, :]
    delta_x = np.sqrt(np.sum(np.square(x2 - x1), axis=1))

    ax[0].plot(np.arange(delta_x.shape[0]), delta_x, color=c_rel)
    ax[1].plot(np.arange(delta_x.shape[0]), delta_x, color=c_irrel)

ax[0].axvspan(51, 70, color='orange', alpha=0.3)
ax[0].set(xlabel='Timesteps', ylabel='Distance between trajectories')
ax[1].axvspan(51, 70, color='orange', alpha=0.3)

cbar0 = plt.colorbar(cm.ScalarMappable(cmap=colormaps[0]), 
                    ax=ax[0], fraction=0.05, label='$|\Delta f_{Rel}|$')
cbar0.set_ticks([0, 0.4, 0.8])
cbar1 = plt.colorbar(cm.ScalarMappable(cmap=colormaps[1]),
                    ax=ax[1], fraction=0.05, label='$|\Delta f_{Irrel}|$')
cbar1.set_ticks([0, 0.4, 0.8])
plt.tight_layout()
tod3 = np.round(taskoutdiff_longdelay[ind0], 3)
#plt.savefig(f'./contraction_figs/{name}_{delaylabel}_distbetweentrajs_{tod3}.png', dpi=300)

# %% Compute matrix measures for Jacobians J for all trials vs time

tmax = 300
max_Re_eigvals = np.zeros((state_var.shape[0], tmax))
lognorm2 = np.zeros((state_var.shape[0], tmax))
lognorm1 = np.zeros((state_var.shape[0], tmax))

for tr in range(state_var.shape[0]):
    print(tr)
    for t in range(tmax):
        J = get_jacobian_relu(state_var[tr, t, :], simulator.W_rec, network_params)
        max_Re_eigvals[tr, t] = np.max(np.real(np.linalg.eigvals(J)))
        lognorm2[tr, t] = np.max(np.linalg.eigvals((J + J.T)/2))
        lognorm1[tr, t] = max(np.diag(J) + np.sum(np.abs(J) - np.diag(np.diag(J)), axis=1))

# %% Plot matrix measures
rcParams['font.size'] = 14
fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
tmax = 300
time = np.arange(tmax)
alpha = 0.4
lw = 0.5
for i in range(max_Re_eigvals.shape[0]):
    ax[0].plot(time, max_Re_eigvals[i, :tmax], 
            color=colors_taskoutdiff[i], alpha=alpha, lw=lw)
    ax[1].plot(time, lognorm2[i, :tmax],
            color=colors_taskoutdiff[i], alpha=alpha, lw=lw)
    ax[2].plot(time, lognorm1[i, :tmax],
            color=colors_taskoutdiff[i], alpha=alpha, lw=lw)

ax[0].set(xlabel='Timesteps')
ax[0].set(ylabel='$max(Re[\lambda(\ J\ )])$', yticks=[-0.008, -0.004, 0])
ax[1].set(ylabel='$\mu_2(\ J\ )$', ) #yticks=[0.01, 0.015, 0.02]) #
ax[2].set(ylabel='$\mu_1(\ J\ )$', yticks=[0.04, 0.06, 0.08, 0.1]) #yticks=[0.06, 0.08, 0.1, .12]) #

plt.tight_layout()
#plt.savefig(f'./contraction_figs/{name}_{delaylabel}_Jmetrics.png', dpi=300)

# %% Plot distance from last state and xdot vs time

rcParams['font.size'] = 14
fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
lw = 0.5
alpha = 0.4
for i in range(state_var.shape[0]):
    x_last = state_var[i, -1, :]
    distance_from_last = np.sqrt(np.sum(np.square(state_var[i, :, :] - x_last), axis=1))
    ax[0].plot(distance_from_last, color=colors_taskoutdiff[i], alpha=alpha, lw=lw)

ax[0].set(xlabel='Timesteps', ylabel='Distance from last state',
       xticks=np.arange(0, tmax + 1, 100), yticks=np.arange(0, 21, 10), xlim=(-5, tmax + 5))

for i in range(state_var.shape[0]):
    xdot_i = np.zeros((state_var.shape[1], state_var.shape[2]))
    for t in range(state_var.shape[1]):
        xdot_i[t] = xdot(state_var[i, t, :], simulator.W_rec, simulator.W_in, test_inputs[i, t, :], simulator.b_rec, network_params)
    ax[1].plot(np.linalg.norm(xdot_i, axis=1), color=colors_taskoutdiff[i], alpha=alpha, lw=lw)

ax[1].set(xlabel='Timesteps', ylabel='$||\dot{x}||$', yticks=[0, 0.05, 0.1, 0.15])
plt.tight_layout()
#plt.savefig(f'./contraction_figs/{name}_{delaylabel}_distfromlast_xdot.png', dpi=300)

# %% Try metrics for contraction: Single trial example

# Doesn't work

i = 0
x0 = state_var[0, 0, :]
J0, F0 = get_jacobian_relu(x0, simulator.W_rec, network_params, returnF=True)
rate = 0.0001

Theta_guess = scipy.linalg.sqrtm(F0) @ np.diag(np.random.uniform(-1, 1, F0.shape[0]))
M0 = Theta_guess.T @ Theta_guess
matrix = matrix = J.T @ M0 + M0 @ J + 2 * rate * M0
print(np.max(np.linalg.eigvals((matrix + matrix.T)/2)))

# %%
