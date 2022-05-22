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
    graph = "sw"
    for beta in [2]:
        if graph == "sw":
            title = ['$\\beta='+str(beta)+', n=9216, p=0.2$','$\\beta='+str(beta)+', n=9216, r=2$','$\\beta='+str(beta)+', p=0.2, r=2$']
        else:
            title = ['$\\beta='+str(beta)+', n=15000, p=0.2$','$\\beta='+str(beta)+', n=15000, r=2$','$\\beta='+str(beta)+', p=0.2, r=2$']
        for ind in range(len(x_var)):
            plot(graph,x_var[ind],x_label[ind],'deg'+str(beta),x_plot[ind],title[ind])


def plot(graph,x_var,x_label,model,x_plot,title,permute=False):
    experiment = '-'+x_label+'-'+model
    print(experiment)

    # Create and save plots
    df = pd.read_csv(load_path+graph+experiment+'-graph_aware.csv')

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
