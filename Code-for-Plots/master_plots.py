'''
Script to plot results from ratio-bern-quadratic experiment
(varying ratio btw indirect/direct effects; bernoulli RD; quadratic model)
'''

# Setup
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

load_path = 'outputFiles/save/'
save_path = 'outputFiles/Plots/'

def main():
    graph = "CON" # configuration model with out-degrees distributed as power law
    x_label = ['ratio', 'tp', 'size']#['varying']
    x_var = ['ratio', 'p', 'n']#['beta']
    model = ['deg2','linear']#['deg']
    #x_label = ['varying']
    #x_var = ['beta']
    #model = ['deg']
    for b in model:
        for ind in range(len(x_var)):
            plot(graph,x_var[ind],x_label[ind],b)

def plot(graph,x_var,x_label,model):
    CRD_est_lin = ['Graph-Agnostic-num', 'Diff-Means-Stnd', 'Diff-Means-Frac-0.75', 'OLS-Prop', 'OLS-Num']
    CRD_est_deg2 = ['Graph-Agnostic-num', 'Diff-Means-Stnd', 'Diff-Means-Frac-0.75', 'LeastSqs-Prop', 'LeastSqs-Num']
    our_est = ['Graph-Agnostic-p', 'Graph-Agnostic-num', 'Graph-AgnosticVR', ]
    experiment = '-'+x_label+'-'+model
    print(experiment)

    # Create and save plots
    df = pd.read_csv(load_path+graph+experiment+'-full-data.csv')

    """
    # Plot with all the estimators
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if model == 'linear':
        newData = df.loc[df['Estimator'].isin(CRD_est_lin)]
    else:
        newData = df.loc[df['Estimator'].isin(CRD_est_deg2)]

    sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', data=newData, ci='sd', legend='brief', markers=True)
    ax.set_ylim(-1,1)
    ax.set_xlabel(x_var, fontsize = 12)
    ax.set_ylabel("Relative Bias", fontsize = 12)
    ax.set_title('Performance of Estimators under CRD', fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    plt.savefig(save_path+graph+experiment+'-allCRD.pdf')
    plt.close()
    """

    # Plot with our the estimators
    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', data=df.loc[df['Estimator'].isin(our_est)], ci='sd', legend='brief', markers=True)
    ax.set_ylim(-0.25,0.25)
    ax.set_xlabel(x_var, fontsize = 12)
    ax.set_ylabel("Relative Bias", fontsize = 12)
    ax.set_title('Performance of our Estimators', fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    plt.savefig(save_path+graph+experiment+'-ours.pdf')
    plt.close()


if __name__ == "__main__":
    main()
