'''
Script to plot results from ratio-bern-quadratic experiment
(varying ratio btw indirect/direct effects; bernoulli RD; quadratic model)
'''

# Setup
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import importlib
import nbformat

load_path = 'outputFiles/save/'
save_path = 'outputFiles/Plots/'

def main():
    graph = "CON" # configuration model with out-degrees distributed as power law
    #x_label = ['ratio', 'tp', 'size']#['varying']
    #x_var = ['ratio', 'p', 'n']#['beta']
    #model = ['deg2','linear']#['deg']
    x_label = ['varying']
    x_var = ['beta']
    model = ['deg']
    for b in model:
        for ind in range(len(x_var)):
            plot(graph,x_var[ind],x_label[ind],b)

def plot(graph,x_var,x_label,model):
    CRD_est = ['PI($k/n$)', 'DM', 'DM($0.75$)', 'LS-Prop', 'LS-Num']
    our_est = ['PI($p$)', 'PI($k/n$)', 'PI($\hat{k}/n$)']

    experiment = '-'+x_label+'-'+model
    print(experiment)

    # Create and save plots
    df = pd.read_csv(load_path+graph+experiment+'-full-data.csv')
    if model == 'linear':
        df = df.assign(Estimator = lambda df: df.Estimator.replace({'Graph-Agnostic-p':'PI($p$)', 'Graph-Agnostic-num':'PI($k/n$)', 'Graph-AgnosticVR': 'PI($\hat{k}/n)$', 'OLS-Prop':'LS-Prop','OLS-Num':'LS-Num','Diff-Means-Stnd': 'DM', 'Diff-Means-Frac-0.75':'DM($0.75$)'}))
        #df.rename(columns={'Graph-Agnostic-p':'PI(p)', 'Graph-Agnostic-num':'PI(k/n)', 'Graph-AgnosticVR': 'PI(kHat/n)', 'OLS-Prop':'LS-Prop','OLS-Num':'LS-Num','Diff-Means-Stnd': 'DM', 'Diff-Means-Frac-0.75':'DM(0.75)'})
    else:
        df = df.assign(Estimator = lambda df: df.Estimator.replace({'Graph-Agnostic-p':'PI($p$)', 'Graph-Agnostic-num':'PI($k/n$)', 'Graph-AgnosticVR': 'PI($\hat{k}/n$)', 'LeastSqs-Prop':'LS-Prop','LeastSqs-Num':'LS-Num','Diff-Means-Stnd': 'DM', 'Diff-Means-Frac-0.75':'DM(0.75)'}))
        #df.rename(columns={'Graph-Agnostic-p':'PI(p)', 'Graph-Agnostic-num':'PI(k/n)', 'Graph-AgnosticVR': 'PI(kHat/n)', 'LeastSqs-Prop':'LS-Prop','LeastSqs-Num':'LS-Num','Diff-Means-Stnd': 'DM', 'Diff-Means-Frac-0.75':'DM(0.75)'})

    plt.rc('text', usetex=True)
    
    # Plot with all the estimators
    fig = plt.figure()
    ax = fig.add_subplot(111)
    newData = df.loc[df['Estimator'].isin(CRD_est)]

    sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', data=newData, ci='sd', legend='brief', markers=True)
    ax.set_ylim(-1,1)
    ax.set_xlabel(x_var, fontsize = 12)
    ax.set_ylabel("Relative Bias", fontsize = 12)
    ax.set_title('Performance of Estimators under CRD', fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    plt.savefig(save_path+graph+experiment+'-allCRD.pdf')
    plt.close()
    

    # Plot with our the estimators
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)

    sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(our_est)], ci='sd', legend='brief', markers=True)
    ax.set_ylim(-0.2,0.2)
    ax.set_xlabel(x_var, fontsize = 12)
    ax.set_ylabel("Relative Bias", fontsize = 12)
    ax.set_title('Performance of our Estimators', fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    plt.savefig(save_path+graph+experiment+'-ours.pdf')
    plt.close()


if __name__ == "__main__":
    main()
