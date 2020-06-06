import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt

# load results
# evl results in avg ckpts
plot_dataset = pd.read_csv('plot_ckpts_avg_res.csv')

# figure params
n_rows, n_cols = 2, 3
fig_width = 11
rate = fig_width / ((400 * n_cols) / (230 * n_rows))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, rate), sharex=False, sharey=True)
fig.tight_layout()

# fix params
n_trajs = 200
xticks = [1, 2, 3, 4]
suprl_cols = ['suprl_%s' % (idx) for idx in xticks]
single_cols = ['single_%s' % (idx) for idx in xticks]

for row, sample_type in enumerate(['first', 'random']):
    for col, agent_type in enumerate(['DQN', 'DDQN', 'QR-DQN']):
        plot_res = plot_dataset[plot_dataset['agent_type'] == agent_type]
        plot_res = plot_res[plot_res['sample_type'] == sample_type+'_%s'%n_trajs]

        single_vals = plot_res[single_cols].values[0]
        suprl_vals = plot_res[suprl_cols].values[0]

        axs[row, col].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[row, col].plot(xticks, suprl_vals, color='#CC33CC', label='SupRL', marker='o')
        axs[row, col].plot(xticks, single_vals, color='#6633CC', label=agent_type, marker='s')
        axs[row, col].legend(loc='lower right')
        sample_type_upper = 'First' if sample_type == 'first' else 'Random'
        axs[row, col].set_title('%s %s' % (sample_type_upper, n_trajs))
 plt.savefig('LunarLander-v2_K=2_ckpts.png', dpi=300)
plt.show()
