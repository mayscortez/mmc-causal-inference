'''
Script to plot results from ratio experiments (linear setting)
(varying ratio btw indirect/direct effects; linear model)
'''
# Setup
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

save_path = 'mmc-causal-inference/outputFiles/'
graph = "CON" # configuration model with out-degrees distributed as power law
experiment = "-ratio-linear" # vary ratio; linear model

#df.loc[(df['Rand'].isin(['Bernoulli'])) & (df['Estimator'] != 'Graph-Agnostic-num')]
#df.loc[(df['Rand'].isin(['CRD'])) & (df['Estimator'].isin(['Graph-Agnostic-num', 'Diff-Means-Stnd', 'Diff-Means-Frac', 'OLS-Prop', 'OLS-Num']))]

# Create and save plots
df = pd.read_csv(save_path+graph+experiment+'-full-data.csv')

# Plot with all the estimators (B & CRD)
fig = plt.figure()
ax = fig.add_subplot(111)
sns.lineplot(x='ratio', y='Bias', hue='Estimator', style='Estimator', data=df.loc[(df['Rand'].isin(['Bernoulli'])) & (df['Estimator'] != 'Graph-Agnostic-num')], ci='sd', legend='brief', markers=True)
sns.lineplot(x='ratio', y='Bias', hue='Estimator', style='Estimator', data=df.loc[(df['Rand'].isin(['CRD'])) & (df['Estimator'].isin(['Graph-Agnostic-num', 'Diff-Means-Stnd', 'Diff-Means-Frac', 'OLS-Prop', 'OLS-Num']))], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Ratio Between Indirect & Direct Effects", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of ALL Estimators (BRD & CRD)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(save_path+graph+experiment+'-all.pdf')
plt.close()

# Plot with just our estimators (B & CRD)
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='ratio', y='Bias', hue='Estimator', style='Estimator', data=df.loc[(df['Estimator'].isin(['Graph-Agnostic-p','Graph-Agnostic-VR'])) & (df['Rand'].isin(['Bernoulli']))], ci='sd', legend='brief', markers=True)
sns.lineplot(x='ratio', y='Bias', hue='Estimator', style='Estimator', data=df.loc[(df['Rand'].isin(['CRD'])) & (df['Estimator'].isin(['Graph-Agnostic-num']))], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Ratio Between Indirect & Direct Effects", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of OUR Estimators (BRD & CRD)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+'-ours.pdf')
plt.close()

# Plot with all the estimators (bernoulli)
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='ratio', y='Bias', hue='Estimator', style='Estimator', data=df.loc[(df['Rand'].isin(['Bernoulli'])) & (df['Estimator'] != 'Graph-Agnostic-num')], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Ratio Between Indirect & Direct Effects", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (BRD)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(save_path+graph+experiment+'-BRD-all.pdf')
plt.close()

# Plot with all the estimators (complete)
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='ratio', y='Bias', hue='Estimator', style='Estimator', data=df.loc[(df['Rand'].isin(['CRD'])) & (df['Estimator'].isin(['Graph-Agnostic-num', 'Diff-Means-Stnd', 'Diff-Means-Frac', 'OLS-Prop', 'OLS-Num']))], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Ratio Between Indirect & Direct Effects", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (CRD)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(save_path+graph+experiment+'-CRD-all.pdf')
plt.close()

'''
# Plot with graph agnostic and both OLS estimators (Bern)
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='ratio', y='Bias', hue='Estimator', style='Estimator', data=df.loc[(df['Estimator'].isin(['OLS-Num','OLS-Prop','Graph-Agnostic-p','Graph-AgnosticVR'])) & (df['Rand'].isin(['Bernoulli']))], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Ratio Between Indirect & Direct Effects", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (BRD)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+'-BRD-agnosticAndOls.pdf')
plt.close()

# Plot with graph agnostic and both OLS estimators (CRD)
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='ratio', y='Bias', hue='Estimator', style='Estimator', data=df.loc[(df['Estimator'].isin(['OLS-Num','OLS-Prop','Graph-Agnostic-num'])) & (df['Rand'].isin(['CRD']))], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Ratio Between Indirect & Direct Effects", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (CRD)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+'-CRD-agnosticAndOls.pdf')
plt.close()

# Plot with graph agnostic and both difference in means estimators (bern)
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='ratio', y='Bias', hue='Estimator', style='Estimator', data=df.loc[(df['Estimator'].isin(['Diff-Means-Stnd','Diff-Means-Frac','Graph-Agnostic-p','Graph-AgnosticVR'])) & (df['Rand'].isin(['bernoulli']))], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Ratio Between Indirect & Direct Effects", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (BRD)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+'-BRD-agnosticAndDiffMeans.pdf')
plt.close()

# Plot with graph agnostic and both difference in means estimators (CRD)
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='ratio', y='Bias', hue='Estimator', style='Estimator', data=df.loc[(df['Estimator'].isin(['Diff-Means-Stnd','Diff-Means-Frac','Graph-Agnostic-num'])) & (df['Rand'].isin(['CRD']))], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Ratio Between Indirect & Direct Effects", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (CRD)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+'-CRD-agnosticAndDiffMeans.pdf')
plt.close()
'''