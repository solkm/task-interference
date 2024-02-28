import numpy as np
import pickle
import scipy.io as sio

data_folder = './monkey_choice_model/test_data/'
testname = 'MM1_monkeyB1245_allinds_noisevis0.8mem0.5rec0.1'

output = pickle.load(open(data_folder + testname + '_modeloutput.pickle', 'rb'))
trialparams = pickle.load(open(data_folder + testname + '_trialparams.pickle', 'rb'))

#%%
sio.savemat(data_folder + testname + '_modeloutput.mat', {'output': output})
sio.savemat(data_folder + testname + '_trialparams.mat', {'trialparams': trialparams})

# %%
