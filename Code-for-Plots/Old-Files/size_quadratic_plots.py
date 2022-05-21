'''
Script to plot results from size-bern-quadratic experiment
(varying size of network; bernoulli RD; quadratic model)
'''
# Setup
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

load_path = 'mmc-causal-inference/outputFiles/'
save_path = 'mmc-causal-inference/outputFiles/Plots/'
graph = "CON" # configuration model with out-degrees distributed as power law
experiment = "-size-quadratic" # vary size; bernoulli RD; quadratic model

# load data
df = pd.read_csv(load_path+graph+experiment+'-full-data.csv')

#########################################
# Plot with combo of estimators (B & CRD)
#########################################
fig = plt.figure()
ax = fig.add_subplot(111)
newdf = df.loc[( (df['Rand'] == 'Bernoulli') & df['Estimator'].isin(['Graph-Agnostic', 'LeastSqs-Prop', 'Spline-Lin'])) | ( (df['Rand'] == 'CRD') & df['Estimator'].isin(['Graph-Agnostic', 'LeastSqs-Prop', 'Spline-Lin']))]

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=newdf, ci='sd', legend='brief', markers=True)
sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=df.loc[(df['Rand'] == 'CRD') & (df['Estimator'] != 'Graph-AgnosticVR')], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of ALL Estimators (BRD & CRD)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(save_path+graph+experiment+'-all.pdf')
plt.close()

#########################################
# Plot with just our estimators (B & CRD)
#########################################
fig = plt.figure()
ax = fig.add_subplot(111)
newdf = df.loc[( (df['Rand'] == 'Bernoulli') & df['Estimator'].isin(['Graph-Agnostic','Graph-AgnosticVR'])) | ( (df['Rand'] == 'CRD') & df['Estimator'].isin(['Graph-Agnostic']))]

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
newdf = df.loc[df['Rand']=='Bernoulli']

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=newdf, ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (BRD)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(load_path+graph+experiment+'-BRD-all.pdf')
plt.close()

#########################################
# Plot with all the estimators (complete)
#########################################
fig = plt.figure()
ax = fig.add_subplot(111)
newdf = df.loc[(df['Rand']=='CRD') & (df['Estimator']!='Graph-AgnosticVR')]

sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=newdf, ci='sd', legend='brief', markers=True)
ax.set_xlabel("Size of Population (n)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (CRD)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(save_path+graph+experiment+'-CRD-all.pdf')
plt.close()
