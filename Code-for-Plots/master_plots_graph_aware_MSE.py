# Plot MSE

# Setup
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

load_path = 'outputFiles/graph_aware/'
save_path = 'outputFiles/graph_aware/'

def main():    
    x_label = ['ratio', 'tp', 'size']
    x_var = ['ratio', 'p', 'n']
    x_plot = ['$r$', '$p$', '$n$']

    graph = "er"
    for beta in [1,2]:
        title = ['$\\beta='+str(beta)+', n=15000, p=0.2$','$\\beta='+str(beta)+', n=15000, r=2$','$\\beta='+str(beta)+', p=0.2, r=2$']
        est_names = ['SNIPE('+str(beta)+')', 'DM', 'DM($0.75$)', 'LS-Prop', 'LS-Num']
        for ind in [0,1,2]:
            plot(graph,x_var[ind],x_label[ind],'deg'+str(beta),x_plot[ind],title[ind],est_names,permute=True)


def plot(graph,x_var,x_label,model,x_plot,title,est_names,permute=False):
    experiment = '-'+x_label+'-'+model
    print(experiment)

    # Create and save plots
    df = pd.read_csv(load_path+graph+experiment+'-graph_aware.csv')
    df = df.assign(Estimator = lambda df: df.Estimator.replace({'Graph-Aware':est_names[0], 'LeastSqs-Prop':est_names[3],'LeastSqs-Num':est_names[4],'Diff-Means-Stnd': est_names[1], 'Diff-Means-Frac-0.75':est_names[2]}))

    df["biassq"] = df["Bias"]**2
    df2 = df.groupby([x_var,'Estimator']).agg('mean')

    # Uncomment line below for LaTeX font
    plt.rc('text', usetex=True)
    
    # Plot with all the estimators
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # '#2077b4' - blue; '#ff7f0e' - orange; '#2ba02c' - green; '#d62728' - red; '#9468bd' - purple
    sns.lineplot(x=x_var, y='biassq', hue='Estimator', style='Estimator', data=df2, palette=['#ff7f0e', '#2ba02c', '#9468bd', '#d62728', '#2077b4'], legend='brief', markers=True)

    if model == 'deg1':
        ax.set_ylim(0,0.05)
    else:
        ax.set_ylim(0,0.5)
    ax.set_xlabel(x_plot, fontsize = 18)
    ax.set_ylabel("MSE", fontsize = 18)
    ax.set_title(title, fontsize=20)
    handles, labels = ax.get_legend_handles_labels()

    if permute:
        order = [4,0,1,3,2] #1 - DM75, 0- DM, 3-LSprop, 4-SNIpe, 2-LSnum
        ax.legend(handles=[handles[i] for i in order], labels=[labels[i] for i in order], loc='upper right', fontsize = 14)
    else:
        ax.legend(handles=handles, labels=labels, loc='upper right', fontsize = 14)

    plt.savefig(save_path+graph+experiment+'_graph_aware_MSE.pdf')
    plt.close()

if __name__ == "__main__":
    main()
