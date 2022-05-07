# Setup
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from math import log, ceil
import pandas as pd
import seaborn as sns
import sys

path_to_module = '/Users/mayleencortez/Desktop/NetworkCausalInference/Code/'
sys.path.append(path_to_module)

import nci_linear_setup as ncls
import nci_polynomial_setup as ncps

save_path = '/Users/mayleencortez/Desktop/NetworkCausalInference/dataFromSSH/'

# Create and save plots
graph = "con-outpwr"
df = pd.read_csv(save_path+graph+'-tp-neurips-quadratic-full-data.csv')

# Plot with all the estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='p', y='Bias', hue='Estimator', style='Estimator', data=df, ci='sd', legend='brief', markers=True)
ax.set_xlabel("Treatment Probability (p)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimator', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(save_path+graph+"-tp-neurips-quadratic-all.pdf")
plt.close()

# Plot with our estimator and Least sqs
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='p', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(['Graph-Agnostic','LeastSqs-Prop','LeastSqs-Num'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Treatment Probability (p)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+"-tp-neurips-quadratic-oursAndLS.pdf")
plt.close()

# Plot with our estimator and interpolation
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='p', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(['Graph-Agnostic','Interp-Lin','Spline-Lin', 'Spline-Quad'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Treatment Probability (p)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+"-tp-neurips-quadratic-oursAndinterp.pdf")
plt.close()