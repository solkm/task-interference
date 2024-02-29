import numpy as np
import pandas as pd
from self_history import Task_SH2
from monkey_model import Task_MM1
from psychrnn.backend.simulation import BasicSimulator
import matplotlib.pyplot as plt

#%% Define testing parameters, load indices

np.random.seed(46) # set seed
N_rec = 200
vis_noise, mem_noise, rec_noise = 0, 0, 0
K = 10
fixedDelay = 300
fixedSL = [2.5, 2.7]
fixedSF=[2.5, 2.7]

name = 'SH2_correctA'
folder = 'correct_choice_model'
weights_path = f'./{folder}/weights/{name}.npz'

tParams_new = pd.read_csv('./data_inds/tParams_new.csv')
K_trainable_inds = np.where(tParams_new[f'K{K}trainable']==1)[0]
inds2test = np.random.choice(K_trainable_inds, 100, replace=False)
N_testbatch = inds2test.shape[0]

#%% Test with Task_SH2

task = Task_SH2(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=N_testbatch, 
                dat=tParams_new, dat_inds=inds2test, K=K, testall=True,
                fixedDelay=fixedDelay, fixedSL=fixedSL, fixedSF=fixedSF)
network_params = task.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise

test_inputs1, _, _, trial_params1 = task.get_trial_batch()
simulator = BasicSimulator(weights_path=weights_path, params=network_params)
model_output1, state_var1 = simulator.run_trials(test_inputs1)

#%% Test with Task_MM1

task = Task_MM1(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=N_testbatch, 
                dat=tParams_new, dat_inds=inds2test, K=K, testall=True,
                fixedDelay=fixedDelay, fixedSL=fixedSL, fixedSF=fixedSF)
network_params = task.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise

test_inputs2, _, _, trial_params2 = task.get_trial_batch()
simulator = BasicSimulator(weights_path=weights_path, params=network_params)
model_output2, state_var2 = simulator.run_trials(test_inputs2)

#%%
print(np.array_equal(test_inputs1[:, :, :-1], test_inputs2[:, :, :-1]))

plt.plot(test_inputs1[0, :, -1])
plt.plot(test_inputs2[0, :, -1], ls='--')
# %%
