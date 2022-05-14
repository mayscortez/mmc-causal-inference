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
save_path = 'mmc-causal-inference/outputFiles/'
save_path_graphs = 'graphs/'

def main():
    G = 20          # number of graphs we want to average over
    T = 50          # number of trials per graph
    graphStr = "CON"   # configuration model

    sys.stdout = open(save_path+'experiments_output_quadratic.txt', 'w')

    ###########################################
    # Run Experiment: Varying Size of Network
    ###########################################
    startTime1 = time.time()
    diag = 6        # controls magnitude of direct effects
    offdiag = 8     # controls magnitude of indirect effects
    r = offdiag/diag
    p = 0.50        # treatment probability

    results = []
    sizes = np.array([5000, 9000, 15000, 19000, 23000, 27000, 31000, 35000])

    for n in sizes:
        # print("n = {}".format(n))
        startTime2 = time.time()

        results.extend(run_experiment(G,T,n,p,r,graphStr,diag))

        executionTime = (time.time() - startTime2)
        print('Runtime (in seconds) for n = {} step: {}'.format(n,executionTime))

    executionTime = (time.time() - startTime1)
    print('Runtime (size experiment) in minutes: {}\n'.format(executionTime/60))        
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path+graphStr+'-size-quadratic-full-data.csv')

    ################################################
    # Run Experiment: Varying Treatment Probability 
    ################################################
    startTime2 = time.time()
    n = 21000        # number of nodes in network
    diag = 6     # maximum norm of direct effect
    offdiag = 8  # maximum norm of indirect effect
    r = offdiag/diag

    results = []
    p_treatments = np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]) # treatment probabilities

    for p in p_treatments:
        # print("Treatment Probability: {}".format(p))
        startTime3 = time.time()

        results.extend(run_experiment(G,T,n,p,r,graphStr,diag))

        executionTime = (time.time() - startTime3)
        print('Runtime (in seconds) for p = {} step: {}'.format(p,executionTime))

    executionTime = (time.time() - startTime2)
    print('Runtime (tp experiment) in minutes: {}\n'.format(executionTime/60))        
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path+graphStr+'-tp-quadratic-full-data.csv')

    ###########################################################
    # Run Experiment: Varying Ratio of Indirect & Direct Effects 
    ###########################################################
    n = 21000
    p = 0.50    # treatment probability
    diag = 10   # maximum norm of direct effect

    results = []
    ratio = [0.25,0.5,0.75,1,1/0.75,1/0.5,3,1/0.25]

    for r in ratio:
        # print('ratio: {}'.format(r))
        startTime3 = time.time()

        results.extend(run_experiment(G,T,n,p,r,graphStr,diag))

        executionTime = (time.time() - startTime3)
        print('Runtime (in seconds) for r = {} step: {}'.format(r,executionTime))

    executionTime = (time.time() - startTime2)
    print('Runtime (ratio experiment) in minutes: {}\n'.format(executionTime/60))          
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path+graphStr+'-ratio-quadratic-full-data.csv')

    executionTime = (time.time() - startTime1)
    print('Runtime (whole script) in hours: {}'.format(executionTime/3600))

    sys.stdout.close()

def run_experiment(G,T,n,p,r,graphStr,diag=1,beta=2,loadGraphs=False):
    
    offdiag = r*diag   # maximum norm of indirect effect

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
        fy = ncps.ppom(ncps.f_quadratic, C, alpha)

        # compute and print true TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
        # print("Ground-Truth TTE: {}".format(TTE))

        ####### Estimate ########
        estimators = []
        estimators.append(lambda y,z,sums,L,K,Lr: ncps.graph_agnostic(n, sums, L))
        estimators.append(lambda y,z,sums,L,K,Lr: ncps.graph_agnostic(n, sums, Lr))
        estimators.append(lambda y,z,sums,L,K,Lr: ncps.poly_regression_prop(beta, y, A, z))
        estimators.append(lambda y,z,sums,L,K,Lr: ncps.poly_regression_num(beta, y, A, z))
        estimators.append(lambda y,z,sums,L,K,Lr: ncps.poly_interp_linear(n, K, sums))
        estimators.append(lambda y,z,sums,L,K,Lr: ncps.poly_interp_splines(n, K, sums, 'quadratic'))

        alg_names = ['Graph-Agnostic', 'Graph-AgnosticVR', 'LeastSqs-Prop', 'LeastSqs-Num','Spline-Lin','Spline-Quad']

        K = ncps.seq_treated(beta,p,n)            # sequence of treated for CRD + staggered rollout
        L = ncps.complete_coeffs(beta, n, K)    # coefficents for GASR estimator under CRD
        P = ncps.seq_treatment_probs(beta,p)    # sequence of probabilities for bern staggered rollout RD
        H = ncps.bern_coeffs(beta,P)            # coefficents for GASR estimator under Bernoulli design
        
        for i in range(T):
            dict_base.update({'rep': i, 'Rand': 'CRD'})
            Z = ncps.staggered_rollout_complete(beta, n, K)
            z = Z[beta,:]
            y = fy(z)
            sums = ncps.outcome_sums(beta, fy, Z)

            for alg in range(len(alg_names)):
                est = estimators[alg](y,z,sums,L,K,L)
                dict_base.update({'Estimator': alg_names[alg], 'Bias': (est-TTE)/TTE})
                results.append(dict_base.copy())

            dict_base.update({'Rand': 'Bernoulli'})
            Z = ncps.staggered_rollout_bern(beta, n, P)
            z = Z[beta,:]
            y = fy(z)
            sums = ncps.outcome_sums(beta, fy, Z)
            Kr = np.sum(Z,1) # realized number of people treated at each time step
            Lr = ncps.complete_coeffs(beta, n, Kr) # coeffs for variance reduction
            
            for alg in range(len(alg_names)):
                est = estimators[alg](y,z,sums,H,P,Lr)
                dict_base.update({'Estimator': alg_names[alg], 'Bias': (est-TTE)/TTE})
                results.append(dict_base.copy())

    return results


if __name__ == "__main__":
    main()