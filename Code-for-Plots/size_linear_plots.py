'''
Script to plot results from size-bern-linear experiment
(varying size of network; bernoulli RD; linear model)
'''
# Setup
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

save_path = 'mmc-causal-inference/outputFiles/'
graph = "con-outpwr" # configuration model with out-degrees distributed as power law
experiment = "-size-bern-linear" # vary size; bernoulli RD; linear model

# Create and save plots
df = pd.read_csv(save_path+graph+experiment+'-full-data.csv')

# Plot with all the estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=df, ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(save_path+graph+experiment+'-all.pdf')
plt.close()

# Plot with just our estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(['Graph-Agnostic','Graph-Agnostic-VR'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+'-ours.pdf')
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
plt.savefig(save_path+graph+experiment+'-agnosticAndOls.pdf')
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
plt.savefig(save_path+graph+experiment+'-agnosticAndDiffMeans.pdf')
plt.close()