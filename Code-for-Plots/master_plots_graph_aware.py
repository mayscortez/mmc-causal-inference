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

load_path = 'outputFiles/graph_aware/'
save_path = 'outputFiles/graph_aware/'

def main():    
    x_label = ['ratio', 'tp', 'size']
    x_var = ['ratio', 'p', 'n']
    x_plot = ['$r$', '$p$', '$n$']
    # graph_list = ["CON-prev","CON","er","sw-ring","SBM"]
    # for graph in graph_list:
    graph = "er"
    for beta in [1,2]:
        if graph == "sw":
            title = ['$\\beta='+str(beta)+', n=9216, p=0.2$','$\\beta='+str(beta)+', n=9216, r=2$','$\\beta='+str(beta)+', p=0.2, r=2$']
        else:
            title = ['$\\beta='+str(beta)+', n=15000, p=0.2$','$\\beta='+str(beta)+', n=15000, r=2$','$\\beta='+str(beta)+', p=0.2, r=2$']
        est_names = ['SNIPE('+str(beta)+')', 'DM', 'DM($0.75$)', 'LS-Prop', 'LS-Num']
        for ind in [0,1,2]:
            plot(graph,x_var[ind],x_label[ind],'deg'+str(beta),x_plot[ind],title[ind],est_names)


def plot(graph,x_var,x_label,model,x_plot,title,est_names,permute=False):
    experiment = '-'+x_label+'-'+model
    print(experiment)
    #est_names = ['SNIPES', 'DM', 'DM($0.75$)', 'LS-Prop', 'LS-Num']

    # Create and save plots
    df = pd.read_csv(load_path+graph+experiment+'-graph_aware.csv')
    df = df.assign(Estimator = lambda df: df.Estimator.replace({'Graph-Aware':est_names[0], 'LeastSqs-Prop':est_names[3],'LeastSqs-Num':est_names[4],'Diff-Means-Stnd': est_names[1], 'Diff-Means-Frac-0.75':est_names[2]}))
        
    plt.rc('text', usetex=True)
    
    # Plot with all the estimators
    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', data=df, ci='sd', legend='brief', markers=True)
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

    plt.savefig(save_path+graph+experiment+'_graph_aware.pdf')
    plt.close()

if __name__ == "__main__":
    main()
