#%%
from monkey_model import Task_MM1
from self_history import Task_SH2
from psychrnn.backend.simulation import BasicSimulator
import numpy as np
import pandas as pd
import pickle

N_rec = 200
vis_noise = 0.8
mem_noise = 0.5
rec_noise = 0.1
K = 10
tParams_new = pd.read_csv('./data_inds/tParams_new.csv')
aR_inds = np.load(open(f'./data_inds/K{K}trainable_aRinds.npy', 'rb'))
aNR_inds = np.load(open(f'./data_inds/K{K}trainable_aNRinds.npy', 'rb'))

def generate_data(dSLs, dSFs, task, weights_path, N, ff=[2.5]):

    network_params = dict()
    network_params['rec_noise'] = rec_noise
    network_params['alpha'] = task.get_task_params()['alpha']
    network_params['dt'] = task.get_task_params()['dt']
    network_params['tau'] = task.get_task_params()['tau']

    model_choices_all = np.zeros(dSLs.shape[0]*dSFs.shape[0]*len(ff)*N*2, dtype=int)
    trial_params_all = np.zeros(dSLs.shape[0]*dSFs.shape[0]*len(ff)*N*2, dtype=object)
    i = 0
    for dSL in dSLs:
        for dSF in dSFs:
            for f1 in ff:
                print(i, dSL, dSF, f1)

                fixed_SL = [f1, f1 + dSL]
                fixed_SF = [f1, f1 + dSF]
                task.fixedSL = fixed_SL
                task.fixedSF = fixed_SF

                test_inds = np.concatenate((np.random.choice(aNR_inds, N, replace=False), 
                                            np.random.choice(aR_inds, N, replace=False)))
                task.dat_inds = test_inds
                task.testall = True

                test_inputs, _, _, trial_params = task.get_trial_batch()
                simulator = BasicSimulator(weights_path=weights_path, params=network_params)
                model_output, _ = simulator.run_trials(test_inputs)

                model_choice = np.argmax(model_output[:,-1,2:6], axis=1) + 1

                model_choices_all[i*N*2:(i+1)*N*2] = model_choice
                trial_params_all[i*N*2:(i+1)*N*2] = trial_params
            
                i += 1
                del test_inputs, trial_params, simulator, model_output, model_choice

    return model_choices_all, trial_params_all

#%% monkey choice model
name = 'MM1_monkeyB1245'
folder = 'monkey_choice_model'
weights_path = f'./{folder}/weights/{name}.npz'

dSLs = np.round(np.arange(-0.7, 0.71, 0.1), 1)
dSLs = np.delete(dSLs, np.where(dSLs==0)[0])
dSFs = dSLs.copy()
ff = [2.1, 2.3, 2.5, 2.7, 2.9]
N = 200//len(ff)
task = Task_MM1(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=2*N, 
                dat=tParams_new, K=K)

model_choices_all, trial_params_all = generate_data(dSLs, dSFs, task, weights_path, N, ff=ff)

#%% correct choice model
name = 'SH2_correctA'
folder = 'correct_choice_model'
weights_path = f'./{folder}/weights/{name}.npz'

dSLs = np.round(np.arange(-0.7, 0.71, 0.1), 1)
dSLs = np.delete(dSLs, np.where(dSLs==0)[0])
dSFs = dSLs.copy()
ff = [2.1, 2.3, 2.5, 2.7, 2.9]
N = 200//len(ff)
task = Task_SH2(vis_noise=vis_noise, mem_noise=mem_noise, N_batch=2*N, 
                dat=tParams_new, K=K)

model_choices_all, trial_params_all = generate_data(dSLs, dSFs, task, weights_path, N, ff=ff)

#%% Save as .pickle
savename = f'./{folder}/test_data/{name}_forppss_changeboth196condsN{N}ff{ff}_noisevis{vis_noise}mem{mem_noise}rec{rec_noise}'

savefile = open(savename+'_modelchoices.pickle','wb')
pickle.dump(model_choices_all, savefile, protocol=4)
savefile.close()

savefile = open(savename+'_trialparams.pickle','wb')
pickle.dump(trial_params_all, savefile, protocol=4)
savefile.close()
# %%

