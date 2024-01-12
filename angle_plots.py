#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:38:41 2024

@author: Sol
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/task-interference')

def angles_0to90(a):
    angles = a.copy()
    angles[angles>90] = 180 - angles[angles>90]
    return angles

#%% load dataframes

df1 = pd.read_pickle('monkey_choice_model/MM1_monkeyB1245_DCAdf.pkl')
df2 = pd.read_pickle('./correct_choice_model/SH2_correctA_DCAdf.pkl')

# extract arrays

timepoints = np.arange(70, 121, 10)
N_trialhists = min(df1.shape[0], df2.shape[0])

angles_mat1 = np.zeros((N_trialhists, timepoints.shape[0]))
dcovs_sl_mat1 = np.zeros((N_trialhists, timepoints.shape[0]))
dcovs_sf_mat1 = np.zeros((N_trialhists, timepoints.shape[0]))

angles_mat2 = np.zeros((N_trialhists, timepoints.shape[0]))
dcovs_sl_mat2 = np.zeros((N_trialhists, timepoints.shape[0]))
dcovs_sf_mat2 = np.zeros((N_trialhists, timepoints.shape[0]))

for i in range(N_trialhists):

    angles_mat1[i] = angles_0to90(df1.loc[i, 'angles'])
    dcovs_sl_mat1[i] = df1.loc[i, 'dcovs_sl']
    dcovs_sf_mat1[i] = df1.loc[i, 'dcovs_sf']
    
    angles_mat2[i] = angles_0to90(df2.loc[i, 'angles'])
    dcovs_sl_mat2[i] = df2.loc[i, 'dcovs_sl']
    dcovs_sf_mat2[i] = df2.loc[i, 'dcovs_sf']

min_dcovs1 = np.minimum(dcovs_sl_mat1, dcovs_sf_mat1)
min_dcovs2 = np.minimum(dcovs_sl_mat2, dcovs_sf_mat2)

#%% angles histogram at one timepoint, dcov threshold

tpt = 3
bins = np.arange(0,91,3)
bin_centers = (bins[1:] + bins[:-1])/2

thresh = 0.2*np.mean(np.stack((dcovs_sl_mat1[:,0], dcovs_sf_mat1[:,0],
                                dcovs_sl_mat2[:,0], dcovs_sf_mat2[:,0])))

inds1 = np.where(min_dcovs1[:, tpt]>thresh)[0]
inds2 = np.where(min_dcovs2[:, tpt]>thresh)[0]

props1, _ = np.histogram(angles_mat1[inds1, tpt], bins=bins, density=True)
plt.plot(bin_centers, props1, color='orange', label='monkey choice model')
props2, _ = np.histogram(angles_mat2[inds2, tpt], bins=bins, density=True)
plt.plot(bin_centers, props2, color='darkblue', label='correct choice model', 
         zorder=0)
plt.legend()

print('%d ms into delay'%((timepoints[tpt]-timepoints[0])*10))
print('threshold: %4.3f'%thresh)
print('monkey model proportion of trials: %4.3f'%(inds1.shape[0]/N_trialhists))
print('correct model proportion of trials: %4.3f'%(inds2.shape[0]/N_trialhists))

#%% angle vs time scatterplots

norm_dcovs = 0.4
assert np.all(min_dcovs1 < norm_dcovs) and np.all(min_dcovs2 < norm_dcovs)
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8,4))

for i in range(N_trialhists):
    
    alpha1 = 0.2 * min_dcovs1[i]/norm_dcovs
    ax[0].scatter(timepoints, angles_mat1[i], c='k', alpha=alpha1)
    
    alpha2 = 0.2 * min_dcovs2[i]/norm_dcovs
    ax[1].scatter(timepoints, angles_mat2[i], c='k', alpha=alpha2)
    
fig.supylabel('Angle (deg)')
fig.supxlabel('Time (10 ms)')

ax[0].set_title('monkey choice model')
ax[1].set_title('correct choice model')

#%%

