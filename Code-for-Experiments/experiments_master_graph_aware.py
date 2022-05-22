'''
Experiments: polynomial setting
'''

# Setup
import numpy as np
import random
import networkx as nx
from math import log, ceil
import pandas as pd
import seaborn as sns
import sys
import time
import scipy.sparse
import nci_linear_setup as ncls
import nci_polynomial_setup as ncps

path_to_module = 'Code-for-Experiments/'
#sys.path.append(path_to_module)
save_path = 'outputFiles/graph_aware/'
save_path_graphs = 'graphs/'

def main(argv):
    if len(argv) > 1:
        beta = int(argv[0])
    else:
        beta = 2

    G = 10          # number of graphs we want to average over
    T = 500          # number of trials per graph

    # graph_list = ["CON-prev","CON","er","sw-ring","SBM"]

    # for graphStr in graph_list:

    graphStr = "er"

    if graphStr = "sw":
        loadGraphs = True
    else:
        loadGraphs = False

    for beta in [2]:

        f = open(save_path+'experiments_output_deg'+str(beta)+'_graph_aware.txt', 'w')
        startTime1 = time.time()

        ###########################################
        # Run Experiment: Varying Size of Network
        ###########################################
        diag = 10        # controls magnitude of direct effects
        r = 2        # ratio between indirect and direct effects
        p = 0.2        # treatment probability

        results = []
        if graphStr == "sw":
            sizes = np.array([16, 24, 32, 48, 64, 96])
        else:
            sizes = np.array([ 5000, 10000, 15000, 20000, 25000])
            # sizes = np.array([10000, 15000, 20000, 30000, 40000])

        for n in sizes:
            print("n = {}".format(n))
            startTime2 = time.time()

            results.extend(run_experiment(G,T,n,p,r,graphStr,diag,beta,loadGraphs))

            executionTime = (time.time() - startTime2)
            print('Runtime (in seconds) for n = {} step: {}'.format(n,executionTime),file=f)
            print('Runtime (in seconds) for n = {} step: {}'.format(n,executionTime))

        executionTime = (time.time() - startTime1)
        print('Runtime (size experiment) in minutes: {}'.format(executionTime/60),file=f)  
        print('Runtime (size experiment) in minutes: {}'.format(executionTime/60))       
        df = pd.DataFrame.from_records(results)
        df.to_csv(save_path+graphStr+'-size-deg'+str(beta)+'-graph_aware.csv')

        ################################################
        # Run Experiment: Varying Treatment Probability 
        ################################################
        startTime2 = time.time()
        if graphStr == "sw":
            n = 96
        else:
            n = 15000        # number of nodes in network
        diag = 10     # maximum norm of direct effect
        r = 2        # ratio between indirect and direct effects

        results = []
        p_treatments = np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]) # treatment probabilities

        for p in p_treatments:
            print("Treatment Probability: {}".format(p))
            startTime3 = time.time()

            results.extend(run_experiment(G,T,n,p,r,graphStr,diag,beta,loadGraphs))

            executionTime = (time.time() - startTime3)
            print('Runtime (in seconds) for p = {} step: {}'.format(p,executionTime),file=f)
            print('Runtime (in seconds) for p = {} step: {}'.format(p,executionTime))

        executionTime = (time.time() - startTime2)
        print('Runtime (tp experiment) in minutes: {}'.format(executionTime/60),file=f)  
        print('Runtime (tp experiment) in minutes: {}'.format(executionTime/60))           
        df = pd.DataFrame.from_records(results)
        df.to_csv(save_path+graphStr+'-tp-deg'+str(beta)+'-graph_aware.csv')

        ###########################################################
        # Run Experiment: Varying Ratio of Indirect & Direct Effects 
        ###########################################################
        startTime2 = time.time()
        if graphStr == "sw":
            n = 96
        else:
            n = 15000        # number of nodes in network
        p = 0.2    # treatment probability
        diag = 10   # maximum norm of direct effect

        results = []
        ratio = [0.01, 0.1, 0.25,0.5,0.75,1,1/0.75,1/0.5,3,1/0.25]

        for r in ratio:
            print('ratio: {}'.format(r))
            startTime3 = time.time()

            results.extend(run_experiment(G,T,n,p,r,graphStr,diag,beta,loadGraphs))

            executionTime = (time.time() - startTime3)
            print('Runtime (in seconds) for r = {} step: {}'.format(r,executionTime),file=f)
            print('Runtime (in seconds) for r = {} step: {}'.format(r,executionTime))

        executionTime = (time.time() - startTime2)
        print('Runtime (ratio experiment) in minutes: {}'.format(executionTime/60),file=f)   
        print('Runtime (ratio experiment) in minutes: {}'.format(executionTime/60))           
        df = pd.DataFrame.from_records(results)
        df.to_csv(save_path+graphStr+'-ratio-deg'+str(beta)+'-graph_aware.csv')

        executionTime = (time.time() - startTime1)
        print('Runtime (whole script) in minutes: {}'.format(executionTime/60),file=f)
        print('Runtime (whole script) in minutes: {}'.format(executionTime/60))

        f.close()

def run_experiment(G,T,n,p,r,graphStr,diag=1,beta=2,loadGraphs=False):
    
    offdiag = r*diag   # maximum norm of indirect effect

    results = []
    dict_base = {'p': p, 'ratio': r, 'n': n, 'beta': beta}

    sz = str(n) + '-'
    for g in range(G):
        graph_rep = str(g)
        dict_base.update({'Graph':sz+graph_rep})

        if loadGraphs:
            if graphStr == "sw":
                A = ncls.loadGraph(save_path_graphs+'SW'+str(n)+'.txt', n*n, True)
                n = n*n
                dict_base.update({'n':n})
                rand_wts = np.random.rand(n,3)
            else:
                # load weighted graph
                name = save_path_graphs + graphStr + sz + graph_rep
                A = scipy.sparse.load_npz(name+'-A.npz')
                rand_wts = np.load(name+'-wts.npy')
        else:
            if graphStr == "CON-prev":
                A = ncls.config_model_nx_prev(n,1000*n)
            if graphStr == "CON":
                A = ncls.config_model_nx(n)
            elif graphStr == "er":
                deg = 10
                A = ncls.erdos_renyi(n,deg/n)
            elif graphStr == "sw-ring":
                A = ncls.small_world(n,10,0.1)
            elif graphStr == "SBM":
                clusterSize = int(n/10)
                n = 10*clusterSize
                prob = 0.02 * np.random.beta(0.5, 0.5, (10, 10)) + np.diagflat(0.08 * np.random.beta(0.5, 0.5, (10, 1)))
                A = ncls.SBM(clusterSize, 10*prob/n)
            rand_wts = np.random.rand(n,3)

    
        alpha = rand_wts[:,0].flatten()
        C = ncls.simpleWeights(A, diag, offdiag, rand_wts[:,1].flatten(), rand_wts[:,2].flatten())

        # C = ncls.weights_node_deg_unif(A)
        # C = C*A
        # C = ncls.normalized_weights(C, diag=10, offdiag=20)
        
        # potential outcomes model
        if beta == 1:
            fy = lambda z: ncls.linear_pom(C,alpha,z)
        else:
            fy = ncps.ppom(beta, C, alpha)

        # compute and print true TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
        # print("Ground-Truth TTE: {}".format(TTE))

        ####### Estimate ########
        estimators = []
        if beta == 1:
            estimators.append(lambda y,z: ncls.est_us(n, p, y, A, z))
        else:
            estimators.append(lambda y,z: ncps.graph_aware_estimator(n, p, y, A, z, beta))
        estimators.append(lambda y,z: ncls.diff_in_means_naive(y,z))
        estimators.append(lambda y,z: ncls.diff_in_means_fraction(n,y,A,z,0.75))
        if beta == 1:
            estimators.append(lambda y,z: ncls.est_ols_gen(y,A,z))
            estimators.append(lambda y,z: ncls.est_ols_treated(y,A,z))
        else:
            estimators.append(lambda y,z: ncps.poly_regression_prop(beta, y, A, z))
            estimators.append(lambda y,z: ncps.poly_regression_num(beta, y, A, z))

        alg_names = ['Graph-Aware', 'Diff-Means-Stnd', 'Diff-Means-Frac-0.75', 'LeastSqs-Prop', 'LeastSqs-Num']

        for i in range(T):
            dict_base.update({'rep':i, 'Rand': 'Bernoulli'})
            z = ncls.bernoulli(n,p)
            y = fy(z)

            for ind in range(len(estimators)):
                est = estimators[ind](y,z)
                dict_base.update({'Estimator': alg_names[ind], 'Bias': (est-TTE)/TTE})
                results.append(dict_base.copy())

    return results


if __name__ == "__main__":
    main(sys.argv[1:])