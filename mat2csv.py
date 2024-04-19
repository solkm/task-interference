import numpy as np
from scipy.io import loadmat
import pandas as pd
#%% Load .mat file
path2mat = '/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/mat_files/tParams_nDist.mat'
data_mat = loadmat(path2mat)
print(data_mat.keys())

#%% Extract data to dataframe
data_df = pd.DataFrame({
    'og_ind' : np.arange(data_mat['r_DSL'].shape[1]),
    'dsl' : data_mat['r_DSL'][0].astype(float),
    'dsf' : data_mat['r_DSF'][0].astype(float),
    'task' : data_mat['r_TASK'][0].astype(int),
    'correct' : data_mat['r_CORRECT'][0].astype(int),
    'choice' : data_mat['r_CHOICE'][0].astype(int),
    'perc_err' : data_mat['r_DERR'][0].astype(int),
    #'RT' : data_mat['r_RT'][0].astype(float),
    #'delay_period' : data_mat['r_DEL'][0].astype(float),
    'nd_task' : data_mat['r_NDB'][0].astype(float),
    'nd_dsl' : data_mat['r_NDSL'][0].astype(float),
    'nd_dsf' : data_mat['r_NDSF'][0].astype(float)
    })

#%% Drop nans
print(data_df.isna().sum())
data_df_nonans = data_df.dropna()
print(data_df.shape)
print(data_df_nonans.shape)
print(data_df_nonans.isna().sum())

# %% Save to .csv
path2csv = './data_inds/tParams_ndist.csv'
data_df_nonans.to_csv(path2csv, index=False)

# %% Compare different tParams files
tParams_new = pd.read_csv('./data_inds/tParams_new.csv')
tParams_time = pd.read_csv('./data_inds/tParams_time.csv')
tParams_og = pd.read_csv('./data_inds/tParams_og.csv')

print(np.array_equal(tParams_new['choice'], tParams_og.loc[tParams_new['og_ind'], 'choice']))

def is_contained_contiguous(subset, mainset):
    subset_length = len(subset)
    for i in range(len(mainset) - subset_length + 1):
        if np.array_equal(subset, mainset[i:i + subset_length]):
            return i
        return False

print(is_contained_contiguous(np.array(tParams_time['choice']), np.array(tParams_og['choice'])))
# %%
