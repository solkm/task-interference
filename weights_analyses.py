import numpy as np
import matplotlib.pyplot as plt

#%% Load weights
weights_path1 = './correct_choice_model/weights/SH2_correctA.npz'
weights_path2 = './monkey_choice_model/weights/MM1_monkeyB1245.npz'

weights1 = np.load(weights_path1, allow_pickle=True)
weights2 = np.load(weights_path2, allow_pickle=True)

#%% Plot trial history input weight distributions
K = 10
W_in = weights1['W_in']
hist_diff_inds = 6*np.arange(0, K-1)
hist_feedback_inds = 6*np.arange(0, K-1) + 5

f, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

ax[0].hlines(0, 0.5, K-0.5, colors='grey', linestyles='dashed', zorder=0, lw=1)
ax[0].violinplot(W_in[:100, hist_diff_inds], np.arange(K-1, 0, -1), showextrema=False)
ax[0].set(xlabel='# trials back', ylabel='Weight distribution', 
        title='Trial history inputs: perceptual difficulty')
ax[1].hlines(0, 0.5, K-0.5, colors='grey', linestyles='dashed', zorder=0, lw=1)
ax[1].violinplot(W_in[:100, hist_feedback_inds], np.arange(K-1, 0, -1), showextrema=False)
ax[1].set(xlabel='# trials back', ylabel='Weight distribution', 
        title='Trial history inputs: reward feedback')
# %%
