'''
Experiments: Linear setting

Runs three experiments and saves the results
- Experiment 1: varying size of the network
- Experiment 2: varying treatment budget
- Experiment 3: varying ratio between indirect and direct effects
'''

# Setup
import numpy as np
import networkx as nx
from math import log, ceil
import pandas as pd
import seaborn as sns
import sys
import time
import scipy.sparse
import setup as ncls

save_path = 'New-Data/'
save_path_graphs = 'Graphs/'

def main():
    G = 30             # number of graphs we want to average over
    T = 100            # number of trials per graph
    graphStr = "CON"   # configuration model

    f = open(save_path+'experiments_output.txt', 'w')

    ###########################################
    # Run Experiment: Varying Size of Network
    ###########################################
    startTime1 = time.time()
    diag = 1        # maximum norm of direct effect
    r = 1.25        # ratio between indirect and direct effects
    p = 0.05        # treatment probability

    results = []
    sizes = np.array([1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000])

    for n in sizes:
        print("n = {}".format(n))
        startTime2 = time.time()

        results.extend(run_experiment(G,T,n,p,r,graphStr,diag))

        executionTime = (time.time() - startTime2)
        print('Runtime (in seconds) for n = {} step: {}'.format(n,executionTime),file=f)
        print('Runtime (in seconds) for n = {} step: {}'.format(n,executionTime))

    executionTime = (time.time() - startTime1)
    print('Runtime (size experiment) in minutes: {}'.format(executionTime/60),file=f)   
    print('Runtime (size experiment) in minutes: {}'.format(executionTime/60))        
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path+graphStr+'-size-linear-full-data.csv')

    ################################################
    # Run Experiment: Varying Treatment Probability 
    ################################################
    startTime2 = time.time()
    n = 5000        # number of nodes in network
    diag = 1        # maximum norm of direct effect
    r = 1.25        # ratio between indirect and direct effects

    results = []
    p_treatments = np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]) # treatment probabilities

    for p in p_treatments:
        print("Treatment Probability: {}".format(p))
        startTime3 = time.time()

        results.extend(run_experiment(G,T,n,p,r,graphStr,diag))

        executionTime = (time.time() - startTime3)
        print('Runtime (in seconds) for p = {} step: {}'.format(p,executionTime),file=f)
        print('Runtime (in seconds) for p = {} step: {}'.format(p,executionTime))

    executionTime = (time.time() - startTime2)
    print('Runtime (tp experiment) in minutes: {}'.format(executionTime/60),file=f)  
    print('Runtime (tp experiment) in minutes: {}'.format(executionTime/60))        
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path+graphStr+'-tp-linear-full-data.csv')

    ###########################################################
    # Run Experiment: Varying Ratio of Indirect & Direct Effects 
    ###########################################################
    n = 5000
    p = 0.05       # treatment probability
    diag = 1       # maximum norm of direct effect

    results = []
    ratio = [0.01, 0.1, 0.25,0.5,0.75,1,1/0.75,1/0.5,3,1/0.25]

    for r in ratio:
        print('ratio: {}'.format(r))
        startTime3 = time.time()

        results.extend(run_experiment(G,T,n,p,r,graphStr,diag))

        executionTime = (time.time() - startTime3)
        print('Runtime (in seconds) for r = {} step: {}'.format(r,executionTime),file=f)
        print('Runtime (in seconds) for r = {} step: {}'.format(r,executionTime))

    executionTime = (time.time() - startTime2)
    print('Runtime (ratio experiment) in minutes: {}'.format(executionTime/60),file=f)     
    print('Runtime (ratio experiment) in minutes: {}'.format(executionTime/60))          
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path+graphStr+'-ratio-linear-full-data.csv')

    executionTime = (time.time() - startTime1)
    print('Runtime (whole script) in minutes: {}'.format(executionTime/60),file=f)
    print('Runtime (whole script) in minutes: {}'.format(executionTime/60))

    sys.stdout.close()

def run_experiment(G,T,n,p,r,graphStr,diag=1,loadGraphs=False):
    '''
    G (int): Number of graphs to average over
    T (int): Number of trials to average over
    n (int): Size of network (number of nodes)
    p (float): treatment budget
    r (float): ration between indirect and direct effects
    graphStr (str): type of graph e.g. "ER" for erdos renyi
    diag (float): maximum norm of the direct effects
    loadGraphs (bool): if set to True, uses pre-saved graphs
    '''
    offdiag = r*diag   # maximum norm of indirect effects

    results = []
    dict_base = {'p': p, 'ratio': r, 'n': n}

    sz = str(n) + '-'
    for g in range(G):
        graph_rep = str(g)
        dict_base.update({'Graph':sz+graph_rep})

        if loadGraphs:
            # load weighted graph
            name = save_path_graphs + graphStr + sz + graph_rep
            A = scipy.sparse.load_npz(name+'-A.npz')
            rand_wts = np.load(name+'-wts.npy')
        else:
            A = ncls.config_model_nx(n)
            rand_wts = np.random.rand(n,3)

        alpha = rand_wts[:,0].flatten()
        C = ncls.simpleWeights(A, diag, offdiag, rand_wts[:,1].flatten(), rand_wts[:,2].flatten())
        
        # potential outcomes model
        fy = lambda z: ncls.linear_pom(C,alpha,z)

        # compute and print true TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
        # print("Ground-Truth TTE: {}".format(TTE))

        ####### Estimate ########
        estimators = []
        estimators.append(lambda y,z: (1/(p*n))*np.sum(y - fy(np.zeros(n))))
        estimators.append(lambda y,z: (1/np.sum(z))*np.sum(y - fy(np.zeros(n))))
        estimators.append(lambda y,z: (1/np.sum(z))*np.sum(y - fy(np.zeros(n))))
        estimators.append(lambda y,z: ncls.diff_in_means_naive(y,z))
        estimators.append(lambda y,z: ncls.diff_in_means_fraction(n,y,A,z,0.75))
        estimators.append(lambda y,z: ncls.est_ols_gen(y,A,z))
        estimators.append(lambda y,z: ncls.est_ols_treated(y,A,z))

        alg_names = ['Graph-Agnostic-p', 'Graph-Agnostic-num', 'Graph-AgnosticVR', 'Diff-Means-Stnd', 'Diff-Means-Frac-0.75', 'OLS-Prop', 'OLS-Num']

        # Compare against Complete Randomized Design
        #bern_est = [0,2]
        #CRD_est = [1,3,4,5,6]

        # Compare against Bernoulli Randomized Design
        bern_est = [0,2,3,4,5,6]
        CRD_est = [1]        

        for i in range(T):
            dict_base.update({'rep': i, 'Rand': 'CRD'})
            z = ncls.completeRD(n,int(np.floor(p*n)))
            y = fy(z)

            for ind in range(len(CRD_est)):
                est = estimators[CRD_est[ind]](y,z)
                dict_base.update({'Estimator': alg_names[CRD_est[ind]], 'Bias': (est-TTE)/TTE})
                results.append(dict_base.copy())

            dict_base.update({'Rand': 'Bernoulli'})
            z = ncls.bernoulli(n,p)
            y = fy(z)

            for ind in range(len(bern_est)):
                est = estimators[bern_est[ind]](y,z)
                dict_base.update({'Estimator': alg_names[bern_est[ind]], 'Bias': (est-TTE)/TTE})
                results.append(dict_base.copy())

    return results


if __name__ == "__main__":
    main()