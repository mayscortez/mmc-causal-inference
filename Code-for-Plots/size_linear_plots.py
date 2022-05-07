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
df = pd.read_csv(save_path+graph+'-size-neurips-linear-full-data.csv')

# Plot with all the estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=df, ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(save_path+graph+"-size-neurips-linear-all.pdf")
plt.close()

# Plot with just our estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(['Graph-Agnostic','Graph-Aware','Graph-Agnostic-VR'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+"-size-neurips-linear-ours.pdf")
plt.close()

# Plot with graph agnostic and both OLS estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(['OLS-Num','OLS-Prop','Graph-Agnostic'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+"-size-neurips-linear-agnosticAndOls.pdf")
plt.close()

# Plot with graph agnostic and both difference in means estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(['Diff-Means-Stnd','Diff-Means-Frac','Graph-Agnostic'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+"-size-neurips-linear-agnosticAndDiffMeans.pdf")
plt.close()