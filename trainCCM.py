# *** NEEDS TO BE EDITED to reflect SH2_correctA model and run in task-interference repo ***
import os
os.chdir('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/MultipleOutputs')
import sys
sys.path.append('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject')
from self_history import RNN_SH2, Task_SH2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% load data and choose training indices

tParams_new = pd.read_csv('./SimplifyData/tParams_new1.csv')
K = 10
K_trainable_inds = np.where(tParams_new[f'K{K}trainable']==1)[0]
first_inds = K_trainable_inds[tParams_new.loc[K_trainable_inds-1, f'K{K}trainable']==0]
np.random.seed(413)
train_first_inds = np.random.choice(first_inds, 140, replace=False)
train_inds = np.zeros(0, dtype='int')
for tfi in train_first_inds:
    fi = np.where(first_inds==tfi)[0][0]
    if fi < first_inds.shape[0]-1:
        fi_range = np.arange(first_inds[fi], first_inds[fi+1])
    else:
        fi_range = np.arange(tfi, K_trainable_inds[-1]+1)
    train_inds = np.append(train_inds, np.intersect1d(fi_range, K_trainable_inds))
#%% TRAIN

N_rec = 200
vis_noise = 0.8
mem_noise = 0.5
rec_noise = 0.1
name = 'SH2_monkeyA'
folder = 'SelfHistory/SH2/monkeyA'
targCorrect = False

# bioconstraints:
autaupses = False
dale = None
# regularizations:
L2_in, L2_out, L2_rec = 0.03, 0.03, 0.03
L2_FR = 0.06

# train params:
train_iters = 300000
N_trainbatch = 200
learn_rate = 0.003

task = Task_SH2(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=N_trainbatch, K=K, dat=tParams_new, dat_inds=train_inds, targCorrect=False)

network_params = task.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise
network_params['L2_in'] = L2_in
network_params['L2_rec'] = L2_rec
network_params['L2_out'] = L2_out
network_params['L2_firing_rate'] = L2_FR
network_params['autapses'] = autaupses
network_params['dale_ratio'] = dale

N_in = task.N_in
N_out = task.N_out
input_connectivity = np.ones((N_rec, N_in))
rec_connectivity = np.ones((N_rec, N_rec))
output_connectivity = np.ones((N_out, N_rec))
assert dale is None
input_connectivity[:int(N_rec/2), -3:] = 0 # ModIn: task module receives no current trial stimulus inputs
input_connectivity[int(N_rec/2):, :-3] = 0 # ModIn: perception module receives no trial history inputs
output_connectivity[:2, int(N_rec/2):] = 0 # ModOut: perception module does not connect to the task output
output_connectivity[2:, :int(N_rec/2)] = 0 # ModOut: task module does not connect to the choice (and fixation) outputs
network_params['input_connectivity'] = input_connectivity
network_params['rec_connectivity'] = rec_connectivity
network_params['output_connectivity'] = output_connectivity

#network_params['load_weights_path'] = 'INSERT_PATH' # If continuing to train a model, load its weights

model = RNN_SH2(network_params)

# Define training parameters
train_params = {}
train_params['training_iters'] = train_iters
train_params['learning_rate'] = learn_rate

# Save weights during training
train_params['training_weights_path'] = f'./{folder}/{name}' 
train_params['loss_epoch'] = 5
train_params['save_training_weights_epoch'] = 15

hist, losses, trainTime, initTime = model.train(task, train_params)
print('initialization time: ', initTime, 'training time: ',trainTime)

np.save(f'./{folder}/{name}_loss', np.array(losses))
np.savez(f'./{folder}/{name}_history', **hist)
np.save(f'./{folder}/{name}_train_inds', train_inds)

# plot loss
plt.figure()
plt.plot(losses)
plt.title('Loss during training')
plt.ylabel('Minibatch loss')
plt.xlabel('Loss epoch')
plt.tight_layout()
plt.savefig(f'./{folder}/{name}_loss', dpi=300)

model.save(f'./{folder}/{name}')
model.destruct()