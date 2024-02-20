#%%
import numpy as np 
import pickle
import model_behavior_functions as mbf
import matplotlib.pyplot as plt
from matplotlib import rcParams

# load model outputs and trial parameters
mod1_name = 'MM1_monkeyB1245'
path1 = f'./monkey_choice_model/test_data/{mod1_name}_allinds_noisevis0.8mem0.5rec0.1'
mod1_outputs = pickle.load(open(path1 + '_modeloutput.pickle', 'rb'))
mod1_choices = np.argmax(mod1_outputs[:, -1, 2:6], axis=1) + 1
mod1_trialparams = pickle.load(open(path1 + '_trialparams.pickle', 'rb'))

mod2_name = 'SH2_correctA'
path2 = f'./correct_choice_model/test_data/{mod2_name}_monkeyhist_allinds_noisevis0.8mem0.5rec0.1'
mod2_outputs = pickle.load(open(path2 + '_modeloutput.pickle', 'rb'))
mod2_choices = np.argmax(mod2_outputs[:, -1, 2:6], axis=1) + 1
mod2_trialparams = pickle.load(open(path2 + '_trialparams.pickle', 'rb'))

# %% calculate accuracies

_, mod1_perc_acc = mbf.get_perc_acc(mod1_choices, mod1_trialparams)
_, mod2_perc_acc = mbf.get_perc_acc(mod2_choices, mod2_trialparams)

mod1_task_acc = mbf.get_task_acc(mod1_choices, mod1_trialparams)
mod2_task_acc = mbf.get_task_acc(mod2_choices, mod2_trialparams)

mod1_overall_acc = mbf.get_overall_acc(mod1_choices, mod1_trialparams)
mod2_overall_acc = mbf.get_overall_acc(mod2_choices, mod2_trialparams)

mod1_monkeychoice_acc = mbf.get_monkeychoice_acc(mod1_choices, mod1_trialparams)
mod2_monkeychoice_acc = mbf.get_monkeychoice_acc(mod2_choices, mod2_trialparams)

mod1_monkeytask_acc = mbf.get_monkeytask_acc(mod1_choices, mod1_trialparams)
mod2_monkeytask_acc = mbf.get_monkeytask_acc(mod2_choices, mod2_trialparams)

# %% barplot

fig, ax = plt.subplots()

labels = ['Perceptual', 'Correct choice', 'Correct task', 'Monkey choice', 'Monkey task']
x = np.arange(len(labels))
mod1_accs = [mod1_perc_acc, mod1_overall_acc, mod1_task_acc, 
             mod1_monkeychoice_acc, mod1_monkeytask_acc]
mod1_accs = [100 * acc for acc in mod1_accs]
mod2_accs = [mod2_perc_acc, mod2_overall_acc, mod2_task_acc,  
             mod2_monkeychoice_acc, mod2_monkeytask_acc]
mod2_accs = [100 * acc for acc in mod2_accs]

width = 0.36
hspace = 0.02
ax.bar(x - (width+hspace)/2, mod1_accs, width=width, 
       label='monkey choice network', facecolor='darkorange')
ax.bar(x + (width+hspace)/2, mod2_accs, width=width, 
       label='correct choice network', facecolor='darkblue')
ax.set_xticks(x, labels, rotation=30)
plt.ylim(50, 100)
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout

rcParams['pdf.fonttype']=42
rcParams['pdf.use14corefonts']=True
#plt.savefig(f'./{mod1_name}_vs_{mod2_name}_accuracybarplot.pdf', dpi=300, transparent=True)
# %%
