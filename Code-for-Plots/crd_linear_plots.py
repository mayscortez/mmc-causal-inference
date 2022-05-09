'''
Script to plot results from experiments with Complete Randomized Design (CRD)
'''
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

save_path = '/Users/mayleencortez/Desktop/NetworkCausalInference/dataFromSSH/'
graph = "con-outpwr" # configuration model with out-degrees distributed as power law

#######################
# Size experiment
#######################
experiment = '-size-CRD'
df = pd.read_csv(save_path+graph+experiment+'-linear-full-data.csv')

# Plot with all the estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'] != 'Graph-AgnosticVR-C'], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(save_path+graph+experiment+"-linear-all.pdf")
plt.close()

# Plot with just our estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(['Graph-Agnostic-C','Graph-Aware-C'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+"-linear-ours.pdf")
plt.close()

# Plot with graph agnostic and both OLS estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(['OLS-Num-C','OLS-Prop-C','Graph-Agnostic-C'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+"-linear-agnosticAndOls.pdf")
plt.close()

# Plot with graph agnostic and both difference in means estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(['Diff-Means-Stnd-C','Diff-Means-Frac-C','Graph-Agnostic-C'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+"-linear-agnosticAndDiffMeans.pdf")
plt.close()


####################################
# Treatment probability experiment
####################################
experiment = '-tp-CRD'
df = pd.read_csv(save_path+graph+experiment+'-linear-full-data.csv')

# Plot with all the estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='p', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'] != 'Graph-AgnosticVR-C'], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Treatment Probability (p)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (n=10000)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(save_path+graph+experiment+"-linear-all.pdf")
plt.close()

# Plot with just our estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='p', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(['Graph-Agnostic-C','Graph-Aware-C'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Treatment Probability (p)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (n=10000)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+"-linear-ours.pdf")
plt.close()

# Plot with graph agnostic and both OLS estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='p', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(['OLS-Num-C','OLS-Prop-C','Graph-Agnostic-C'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Treatment Probability (p)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (n=10000)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+"-linear-agnosticAndOls.pdf")
plt.close()

# Plot with graph agnostic and both difference in means estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='p', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(['Diff-Means-Stnd-C','Diff-Means-Frac-C','Graph-Agnostic-C'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Treatment Probability (p)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (n=10000)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+"-linear-agnosticAndDiffMeans.pdf")
plt.close()


#######################
# Ratio experiment
#######################
experiment = '-ratio-CRD'
df = pd.read_csv(save_path+graph+experiment+'-linear-full-data.csv')

# Plot with all the estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='ratio', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'] != 'Graph-AgnosticVR-C'], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Ratio Between Indirect & Direct Effects", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (n=10000)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(save_path+graph+experiment+"-linear-all.pdf")
plt.close()

# Plot with just our estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='ratio', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(['Graph-Agnostic-C','Graph-Aware-C'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Ratio Between Indirect & Direct Effects", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (n=10000)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+"-linear-ours.pdf")
plt.close()

# Plot with graph agnostic and both OLS estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='ratio', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(['OLS-Num-C','OLS-Prop-C','Graph-Agnostic-C'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Ratio Between Indirect & Direct Effects", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (n=10000)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+"-linear-agnosticAndOls.pdf")
plt.close()

# Plot with graph agnostic and both difference in means estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='ratio', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(['Diff-Means-Stnd-C','Diff-Means-Frac-C','Graph-Agnostic-C'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Ratio Between Indirect & Direct Effects", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (n=10000)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+"-linear-agnosticAndDiffMeans.pdf")
plt.close()