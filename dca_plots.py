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
from matplotlib import rcParams
import os
import scipy.stats as st
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
aR_inds, aNR_inds = [], []

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
    
    aR_inds.append(i) if df1.loc[i, 'reward_history'][-1] == 1 else aNR_inds.append(i)

aR_inds, aNR_inds = np.array(aR_inds), np.array(aNR_inds)

min_dcovs1 = np.minimum(dcovs_sl_mat1, dcovs_sf_mat1)
max_dcovs1 = np.maximum(dcovs_sl_mat1, dcovs_sf_mat1)

min_dcovs2 = np.minimum(dcovs_sl_mat2, dcovs_sf_mat2)
max_dcovs2 = np.maximum(dcovs_sl_mat2, dcovs_sf_mat2)

#%% magnitude of projection of irrelevant onto relevant axis histogram

proj1 = np.zeros((N_trialhists, timepoints.shape[0]))
proj2 = np.zeros((N_trialhists, timepoints.shape[0]))

for i in range(N_trialhists):
    
    proj1[i] = np.array([np.dot(df1.loc[i, 'ax_sl'][t], df1.loc[i, 'ax_sf'][t]) \
                         for t in range(timepoints.shape[0])])
    proj1[i] *= min_dcovs1[i]

    proj2[i] = np.array([np.dot(df2.loc[i, 'ax_sl'][t], df2.loc[i, 'ax_sf'][t]) \
                      for t in range(timepoints.shape[0])])
    proj2[i] *= min_dcovs2[i]

tpt = 0
lim = 0.15
binsize = 0.005
bins = np.arange(0, lim+binsize, binsize)
bin_centers = (bins[1:] + bins[:-1])/2

props1_aNR, _ = np.histogram(np.abs(proj1[aNR_inds, tpt]), bins=bins, density=True)
props2_aNR, _ = np.histogram(np.abs(proj2[aNR_inds, tpt]), bins=bins, density=True)
props1_aR, _ = np.histogram(np.abs(proj1[aR_inds, tpt]), bins=bins, density=True)
props2_aR, _ = np.histogram(np.abs(proj2[aR_inds, tpt]), bins=bins, density=True)

med1_aNR = np.median(np.abs(proj1[aNR_inds, tpt]))
med2_aNR = np.median(np.abs(proj2[aNR_inds, tpt]))
med1_aR = np.median(np.abs(proj1[aR_inds, tpt]))
med2_aR = np.median(np.abs(proj2[aR_inds, tpt]))

# plot
colors = ['orangered', 'darkblue', 'orange', 'cornflowerblue']
plt.plot(bin_centers, props1_aNR, color=colors[0], ls = '-', lw=2, zorder=3,
         label='monkey choice model, after unrewarded trial')

plt.plot(bin_centers, props2_aNR, color=colors[1], ls = '-', lw=2, zorder=0,
         label='correct choice model, after unrewarded trial')

plt.plot(bin_centers, props1_aR, color=colors[2], ls = '-', lw=2, zorder=2, 
         label='monkey choice model, after rewarded trial')

plt.plot(bin_centers, props2_aR, color=colors[3], ls = '-', lw=2, zorder=1,
         label='correct choice model, after rewarded trial')

med_y = 1.05*np.max(props2_aR) #100
plt.scatter([med1_aNR, med2_aNR, med1_aR, med2_aR], np.tile(med_y, 4),
            c=colors, marker='v')

plt.legend()
plt.xlabel('Magnitude of projection of irrelevant axis onto relevant axis')
plt.ylabel('Probability density')

rcParams['pdf.fonttype']=42
rcParams['pdf.use14corefonts']=True
#plt.savefig(f'./MM1_monkeyB1245_vs_SH2_correctA_projhist_aNRaR_tpt{tpt}_meds.pdf', dpi=300, transparent=True)

#%% statistical testing

_, p_aNR1_aNR2 = st.ranksums(np.abs(proj1[aNR_inds, tpt]), np.abs(proj2[aNR_inds, tpt]))
print('aNR monkey vs aNR correct: p=%4.3e'%p_aNR1_aNR2)

_, p_aR1_aR2 = st.ranksums(np.abs(proj1[aR_inds, tpt]), np.abs(proj2[aR_inds, tpt]))
print('aR monkey vs aR correct: p=%4.3e'%p_aR1_aR2)

_, p_aR1_aNR1 = st.ranksums(np.abs(proj1[aR_inds, tpt]), np.abs(proj1[aNR_inds, tpt]))
print('aR monkey vs aNR monkey: p=%4.3e'%p_aR1_aNR1)

_, p_aNR2_aR2 = st.ranksums(np.abs(proj2[aNR_inds, tpt]), np.abs(proj2[aR_inds, tpt]))
print('aNR correct vs aR correct: p=%4.3e'%p_aNR2_aR2)

_, p_aNR1_aR2 = st.ranksums(np.abs(proj1[aNR_inds, tpt]), np.abs(proj2[aR_inds, tpt]))
print('aNR monkey vs aR correct: p=%4.3e'%p_aNR1_aR2)

_, p_aR1_aNR2 = st.ranksums(np.abs(proj1[aR_inds, tpt]), np.abs(proj2[aNR_inds, tpt]))
print('aR monkey vs aNR correct: p=%4.3e'%p_aR1_aNR2)

#%% projection histogram, pooling aR and aNR trials

tpt = 4
lim = 0.15
binsize = 0.003
bins = np.arange(0, lim+binsize, binsize)
bin_centers = (bins[1:] + bins[:-1])/2

props1, _ = np.histogram(np.abs(proj1[:, tpt]), bins=bins, density=True)
props2, _ = np.histogram(np.abs(proj2[:, tpt]), bins=bins, density=True)

med1 = np.median(np.abs(proj1[:, tpt]))
med2 = np.median(np.abs(proj2[:, tpt]))

# plot
plt.plot(bin_centers, props1, color='orange', ls = '-', lw=2, zorder=3,
         label='monkey choice model')
plt.plot(bin_centers, props2, color='darkblue', ls = '-', lw=2, zorder=2,
         label='correct choice model')
med_y = 80
plt.scatter([med1, med2], np.tile(med_y, 2), c=['orange', 'darkblue'], marker='v')

plt.legend()
plt.xlabel('Magnitude of projection of irrelevant axis onto relevant axis')
plt.ylabel('Probability density')

#%% angles histogram at one timepoint, dcov threshold

tpt = 4
bins = np.arange(0, 91, 3)
bin_centers = (bins[1:] + bins[:-1])/2

thresh_fac = 0.15
thresh = thresh_fac*np.mean(np.stack((dcovs_sl_mat1[:,0], dcovs_sf_mat1[:,0],
                                      dcovs_sl_mat2[:,0], dcovs_sf_mat2[:,0])))

inds1 = np.where(min_dcovs1[:, tpt]>thresh)[0]
inds2 = np.where(min_dcovs2[:, tpt]>thresh)[0]

props1, _ = np.histogram(angles_mat1[inds1, tpt], bins=bins, density=True)
plt.plot(bin_centers, props1, color='orange', label='monkey choice model')
props2, _ = np.histogram(angles_mat2[inds2, tpt], bins=bins, density=True)
plt.plot(bin_centers, props2, color='darkblue', label='correct choice model', zorder=0)
median1 = np.median(angles_mat1[inds1, tpt])
median2 = np.median(angles_mat2[inds2, tpt])
y_med = np.round(max(np.max(props1), np.max(props2))*1.1, 2)
plt.scatter(median1, y_med, color='orange', marker='v', label='median = %3.2f'%median1)
plt.scatter(median2, y_med, color='darkblue', marker='v', label='median = %3.2f'%median2)
plt.legend()
plt.ylabel('Proportion of trials')
plt.xlabel('Angle')
plt.xticks(np.arange(0,91,15))

print('%d ms into delay'%((timepoints[tpt]-timepoints[0])*10))
print('threshold: %4.3f'%thresh)
print('monkey model proportion of trials: %4.3f'%(inds1.shape[0]/N_trialhists))
print('correct model proportion of trials: %4.3f'%(inds2.shape[0]/N_trialhists))

rcParams['pdf.fonttype']=42
rcParams['pdf.use14corefonts']=True
#plt.savefig(f'./MM1_monkeyB1245_vs_SH2_correctA_angleshist_tpt{tpt}_threshfac{thresh_fac}.pdf', dpi=300, transparent=True)

#%% alternative to thresholding: angles histogram weighted by dcovs

def angles_weighted_counts(angles, weights, bins, norm_weights=True):

    angles_dig = np.digitize(angles, bins)
    weighted_counts = np.zeros(bins.shape[0]-1)

    if norm_weights:
        weights = weights/np.sum(weights)
    for i in range(bins.shape[0]-1):
        inds = np.where(angles_dig==i+1)[0]
        weights_i = weights[inds]
        weighted_counts[i] = np.sum(weights_i)

    return weighted_counts

# plot
tpt = 4
bins = np.arange(0, 91, 5)
bin_centers = (bins[1:] + bins[:-1])/2
weights1 = min_dcovs1[:, tpt]
weights2 = min_dcovs2[:, tpt]
weighted_counts1 = angles_weighted_counts(angles_mat1[:, tpt], weights1, bins)
weighted_counts2 = angles_weighted_counts(angles_mat2[:, tpt], weights2, bins)

plt.plot(bin_centers, weighted_counts1, color='orange', label='monkey choice model')
plt.plot(bin_centers, weighted_counts2, color='darkblue', label='correct choice model')
plt.legend()
plt.xlabel('Angle')
plt.ylabel('Weighted counts')

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

# %%
