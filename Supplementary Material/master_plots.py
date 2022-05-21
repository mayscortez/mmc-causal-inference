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

load_path = 'Plots/'
save_path = 'Plots/'

def main():
    graph = "CON" # configuration model with out-degrees distributed as power law
    
    title = ['$\\beta=1, n=5000, k/n=0.05$','$\\beta=1, n=5000, r=1.25$','$\\beta=1, k/n=0.05, r=1.25$']
    x_label = ['ratio', 'tp', 'size']
    x_var = ['ratio', 'p', 'n']
    x_plot = ['$r$', '$k/n$', '$n$']
    model = ['linear','deg2']
    for b in model:
        for ind in range(len(x_var)):
            plot(graph,x_var[ind],x_label[ind],b,x_plot[ind],title[ind])

    
    title = ['$n=15000, k/n=0.5, r=1.25$']
    x_label = ['varying']
    x_var = ['beta']
    x_plot = ['$\\beta$']
    model = ['deg']
    for b in model:
        for ind in range(len(x_var)):
            plot(graph,x_var[ind],x_label[ind],b,x_plot[ind],title[ind],True)

def plot(graph,x_var,x_label,model,x_plot,title,permute=False):
    # CRD_est = ['$\text{PI}(k/n)$', '$\text{DM}$', '$\text{DM}(0.75)$', '$\text{LS-Prop}$', '$\text{LS-Num}$']
    # our_est = ['$\text{PI}(p)$', '$\text{PI}(k/n)$', '$\text{PI}(\hat{k}/n)$']
    CRD_est = ['PI($k/n$)', 'DM', 'DM($0.75$)', 'LS-Prop', 'LS-Num']
    our_est = ['PI($p$)', 'PI($k/n$)', 'PI($\hat{k}/n$)']

    experiment = '-'+x_label+'-'+model
    print(experiment)

    # Create and save plots
    df = pd.read_csv(load_path+graph+experiment+'-full-data.csv')
    if model == 'linear':
        df = df.assign(Estimator = lambda df: df.Estimator.replace({'Graph-Agnostic-p':our_est[0], 'Graph-Agnostic-num':our_est[1], 'Graph-AgnosticVR': our_est[2], 'OLS-Prop':CRD_est[3],'OLS-Num':CRD_est[4],'Diff-Means-Stnd': CRD_est[1], 'Diff-Means-Frac-0.75':CRD_est[2]}))
        #df.rename(columns={'Graph-Agnostic-p':'PI(p)', 'Graph-Agnostic-num':'PI(k/n)', 'Graph-AgnosticVR': 'PI(kHat/n)', 'OLS-Prop':'LS-Prop','OLS-Num':'LS-Num','Diff-Means-Stnd': 'DM', 'Diff-Means-Frac-0.75':'DM(0.75)'})
    else:
        df = df.assign(Estimator = lambda df: df.Estimator.replace({'Graph-Agnostic-p':our_est[0], 'Graph-Agnostic-num':our_est[1], 'Graph-AgnosticVR': our_est[2], 'LeastSqs-Prop':CRD_est[3],'LeastSqs-Num':CRD_est[4],'Diff-Means-Stnd': CRD_est[1], 'Diff-Means-Frac-0.75':CRD_est[2]}))
        #df.rename(columns={'Graph-Agnostic-p':'PI(p)', 'Graph-Agnostic-num':'PI(k/n)', 'Graph-AgnosticVR': 'PI(kHat/n)', 'LeastSqs-Prop':'LS-Prop','LeastSqs-Num':'LS-Num','Diff-Means-Stnd': 'DM', 'Diff-Means-Frac-0.75':'DM(0.75)'})

    plt.rc('text', usetex=True)
    
    # Plot with all the estimators
    fig = plt.figure()
    ax = fig.add_subplot(111)
    newData = df.loc[df['Estimator'].isin(CRD_est)]

    sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', data=newData, ci='sd', legend='brief', markers=True)
    ax.set_ylim(-1,1)
    ax.set_xlabel(x_plot, fontsize = 18)
    ax.set_ylabel("Relative Bias", fontsize = 18)
    ax.set_title(title, fontsize=20)
    handles, labels = ax.get_legend_handles_labels()

    if permute:
        order = [0,3,4,1,2]
        ax.legend(handles=[handles[i] for i in order], labels=[labels[i] for i in order], loc='upper right', fontsize = 14)
    else:
        ax.legend(handles=handles, labels=labels, loc='upper right', fontsize = 14)

    plt.savefig(save_path+graph+experiment+'-allCRD.pdf')
    #plt.savefig(save_path+graph+experiment+'-allBRD.pdf')
    plt.close()
    

    # Plot with our the estimators
    fig = plt.figure(figsize=(5,3.5))
    ax = fig.add_subplot(111)

    sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(our_est)], ci='sd', legend='brief', markers=True)
    ax.set_ylim(-0.2,0.2)
    ax.set_xlabel(x_plot, fontsize = 14)
    ax.set_ylabel("Relative Bias", fontsize = 14)
    ax.set_title(title, fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=10)
    plt.tight_layout()

    plt.savefig(save_path+graph+experiment+'-oursCRD.pdf')
    #plt.savefig(save_path+graph+experiment+'-oursBRD.pdf')
    plt.close()


if __name__ == "__main__":
    main()
