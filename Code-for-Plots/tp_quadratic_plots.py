'''
Script to plot results from size-bern-linear experiment
(varying size of network; bernoulli RD; quadratic model)
'''
# Setup
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

save_path = 'mmc-causal-inference/outputFiles/'
graph = "con-outpwr" # configuration model with out-degrees distributed as power law
experiment = "-tp-bern-quadratic" # vary treatent probability; bernoulli RD; quadratic model

# Create and save plots
df = pd.read_csv(save_path+graph+experiment+'-full-data.csv')

# Plot with all the estimators
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='p', y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['p'] > 0.10], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Treatment Probability (p)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimator (n=50000)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.savefig(save_path+graph+experiment+'-all.pdf')
plt.close()

# Plot with our estimator and Least sqs
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='p', y='Bias', hue='Estimator', style='Estimator', data=df.loc[(df['p'] > 0.10) & df['Estimator'].isin(['Graph-Agnostic','LeastSqs-Prop','LeastSqs-Num'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Treatment Probability (p)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (n=50000)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+'-oursAndLS.pdf')
plt.close()

# Plot with our estimator and interpolation
fig = plt.figure()
ax = fig.add_subplot(111)

sns.lineplot(x='p', y='Bias', hue='Estimator', style='Estimator', data=df.loc[(df['p'] > 0.10) & df['Estimator'].isin(['Graph-Agnostic','Interp-Lin','Spline-Lin', 'Spline-Quad'])], ci='sd', legend='brief', markers=True)
ax.set_xlabel("Treatment Probability (p)", fontsize = 12)
ax.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators (n=50000)', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(save_path+graph+experiment+'-oursAndinterp.pdf')
plt.close()