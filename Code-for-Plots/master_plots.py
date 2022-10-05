'''
Script to plot results from staggered rollout experiments:
    - vary size of network
    - vary treatment budget
    - vary ratio of direct:indirect effects
    - vary beta
Staggered Rollout Estimators: Bernoulli RD, Complete RD, and Beroulli Realized

Take careful look at commented lines before running... may need to uncomment some and comment others
'''

# Setup
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

load_path = 'outputFiles/save/'
#save_path = 'outputFiles/save/Configuration-Model-Plots/'
save_path = 'outputFiles/save/Erdos-Renyi-Plots/'

def main():
    #graph = "CON"  # configuration model
    graph = "ER"    # erdos-renyi

    #rand_design = "-allCRD"     # Complete
    rand_design = "-allBRD"    # Bernoulli

    #title = ['$\\beta=1, n=15000, k/n=0.2$','$\\beta=1, n=15000, r=2$','$\\beta=1, k/n=0.2, r=2$']   # Complete 
    title = ['$\\beta=1, n=15000, p=0.2$','$\\beta=1, n=15000, r=2$','$\\beta=1, p=0.2, r=2$']    # Bernoulli
    x_label = ['ratio', 'tp', 'size']
    x_var = ['ratio', 'p', 'n']
    #x_plot = ['$r$', '$k/n$', '$n$']   # Complete
    x_plot = ['$r$', '$p$', '$n$']    # Bernoulli
    model = ['linear']
    for b in model:
        for ind in range(len(x_var)):
            plot(graph,x_var[ind],x_label[ind],b,x_plot[ind],title[ind],rand_design)
    
    
    #title = ['$\\beta=2, n=15000, k/n=0.2$','$\\beta=2, n=15000, r=2$','$\\beta=2, k/n=0.2, r=2$'] # Complete
    title = ['$\\beta=2, n=15000, p=0.2$','$\\beta=2, n=15000, r=2$','$\\beta=2, p=0.2, r=2$']  # Bernoulli
    x_label = ['ratio', 'tp', 'size']
    x_var = ['ratio', 'p', 'n']
    #x_plot = ['$r$', '$k/n$', '$n$'] #Complete
    x_plot = ['$r$', '$p$', '$n$'] #Bernoulli
    model = ['deg2']
    for b in model:
        for ind in range(len(x_var)):
            plot(graph,x_var[ind],x_label[ind],b,x_plot[ind],title[ind],rand_design)

    #title = ['$n=15000, k/n=0.2, r=2$'] #Complete
    title = ['$n=15000, p=0.2, r=2$'] #Bernoulli
    x_label = ['varying']
    x_var = ['beta']
    x_plot = ['$\\beta$']
    model = ['deg']
    for b in model:
        for ind in range(len(x_var)):
            plot(graph,x_var[ind],x_label[ind],b,x_plot[ind],title[ind],rand_design)
    

def plot(graph,x_var,x_label,model,x_plot,title,rand_design,permute=False):
    CRD_est = ['PI($k/n$)', 'DM', 'DM($0.75$)', 'LS-Prop', 'LS-Num']
    BRD_est = ['PI($p$)', 'DM', 'DM($0.75$)', 'LS-Prop', 'LS-Num']
    our_est = ['PI($p$)', 'PI($k/n$)', 'PI($\hat{k}/n$)']

    experiment = '-'+x_label+'-'+model
    print(experiment)

    # Create and save plots
    df = pd.read_csv(load_path+graph+experiment+'-full-data.csv')
    if model == 'linear':
        df = df.assign(Estimator = lambda df: df.Estimator.replace({'Graph-Agnostic-p':our_est[0], 'Graph-Agnostic-num':our_est[1], 'Graph-AgnosticVR': our_est[2], 'OLS-Prop':CRD_est[3],'OLS-Num':CRD_est[4],'Diff-Means-Stnd': CRD_est[1], 'Diff-Means-Frac-0.75':CRD_est[2]}))
    else:
        df = df.assign(Estimator = lambda df: df.Estimator.replace({'Graph-Agnostic-p':our_est[0], 'Graph-Agnostic-num':our_est[1], 'Graph-AgnosticVR': our_est[2], 'LeastSqs-Prop':BRD_est[3],'LeastSqs-Num':BRD_est[4],'Diff-Means-Stnd': CRD_est[1], 'Diff-Means-Frac-0.75':CRD_est[2]}))
    

    if experiment == '-varying-deg':
        df = df.loc[df['beta'].isin([0,1,2,3])]

    plt.rc('text', usetex=True)
    
    # Plot with all the estimators
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #newData = df.loc[df['Estimator'].isin(CRD_est)]    #Complete
    newData = df.loc[df['Estimator'].isin(BRD_est)]     #Bernoulli

    sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', data=newData, ci='sd', legend='brief', markers=True)
    ax.set_ylim(-0.5,0.5)
    ax.set_xlabel(x_plot, fontsize = 18)
    ax.set_ylabel("Relative Bias", fontsize = 18)
    ax.set_title(title, fontsize=20)
    handles, labels = ax.get_legend_handles_labels()

    if permute:
        order = [0,3,4,1,2]
        ax.legend(handles=[handles[i] for i in order], labels=[labels[i] for i in order], loc='upper right', fontsize = 14)
    else:
        ax.legend(handles=handles, labels=labels, loc='upper right', fontsize = 14)

    plt.savefig(save_path+graph+experiment+rand_design+'.pdf')
    plt.close()
    
    
    # Plot with our the estimators
    fig = plt.figure(figsize=(5,3.5))
    ax = fig.add_subplot(111)

    sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(our_est)], ci='sd', legend='brief', markers=True)
    ax.set_ylim(-0.15,0.15)
    ax.set_xlabel(x_plot, fontsize = 14)
    ax.set_ylabel("Relative Bias", fontsize = 14)
    ax.set_title(title, fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=10)
    plt.tight_layout()

    plt.savefig(save_path+graph+experiment+'-ours.pdf')
    plt.close()

if __name__ == "__main__":
    main()
