'''
Script to plot results from size experiments (linear setting)
(varying size of network; linear model)
'''
# Setup
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

load_path = '/Users/mayleencortez/Desktop/NetworkCausalInference/mmc-causal-inference/outputFiles/'
save_path = '/Users/mayleencortez/Desktop/NetworkCausalInference/mmc-causal-inference/outputFiles/Plots/'
graph = "CON" # configuration model (with out-degrees distributed as power law)
experiment = "-size-linear" # vary size; linear model

# load data
df = pd.read_csv(load_path+graph+experiment+'-full-data.csv')

#########################################
# Plot with combo of estimators
#########################################
fig = plt.figure()
ax = fig.add_subplot(111)
newdf = df.loc[( (df['Rand'] == 'Bernoulli') & df['Estimator'].isin(['Graph-Agnostic-p', 'Diff-Means-Stnd', 'OLS-Prop'])) | ( (df['Rand'] == 'CRD') & df['Estimator'].isin(['Graph-Agnostic-num', 'Diff-Means-Prop']))]

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Rand', data=newdf, ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (BRD & CRD)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(save_path+graph+experiment+'-all.pdf')
plt.close()

#########################################
# Plot with just our estimators (B & CRD)
#########################################
fig = plt.figure()
ax = fig.add_subplot(111)
newdf = df.loc[( (df['Rand'] == 'Bernoulli') & df['Estimator'].isin(['Graph-Agnostic-p','Graph-AgnosticVR'])) | ( (df['Rand'] == 'CRD') & df['Estimator'].isin(['Graph-Agnostic-num']))]

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Rand', data=newdf, ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of OUR Estimators (BRD & CRD)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+'-ours.pdf')
plt.close()

#########################################
# Plot with all the estimators (bernoulli)
#########################################
fig = plt.figure()
ax = fig.add_subplot(111)
newdf = df.loc[(df['Rand'].isin(['Bernoulli'])) & (df['Estimator'] != 'Graph-Agnostic-num')]

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=newdf, ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (BRD)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(save_path+graph+experiment+'-BRD-all.pdf')
plt.close()

#########################################
# Plot with all the estimators (complete)
#########################################
fig = plt.figure()
ax = fig.add_subplot(111)
newdf = df.loc[(df['Rand'].isin(['CRD'])) & (df['Estimator'].isin(['Graph-Agnostic-num', 'Diff-Means-Stnd', 'Diff-Means-Frac', 'OLS-Prop', 'OLS-Num']))]

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=newdf, ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (CRD)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(save_path+graph+experiment+'-CRD-all.pdf')
plt.close()
