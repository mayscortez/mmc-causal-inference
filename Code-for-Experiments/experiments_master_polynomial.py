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
save_path = 'outputFiles/'
save_path_graphs = 'graphs/'

def main():
    G = 20          # number of graphs we want to average over
    T = 50          # number of trials per graph
    beta = 2
    graphStr = "CON"   # configuration model

    f = open(save_path+'experiments_output_'+str(beta)+'.txt', 'w')

    ###########################################
    # Run Experiment: Varying Size of Network
    ###########################################
    startTime1 = time.time()
    diag = 6        # controls magnitude of direct effects
    offdiag = 8     # controls magnitude of indirect effects
    r = offdiag/diag
    p = 0.50        # treatment probability

    results = []
    sizes = np.array([1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000])

    for n in sizes:
        print("n = {}".format(n))
        startTime2 = time.time()

        results.extend(run_experiment(G,T,n,p,r,graphStr,diag,beta))

        executionTime = (time.time() - startTime2)
        print('Runtime (in seconds) for n = {} step: {}'.format(n,executionTime))
        print('Runtime (in seconds) for n = {} step: {}'.format(n,executionTime),file=f)

    executionTime = (time.time() - startTime1)
    print('Runtime (size experiment) in minutes: {}\n'.format(executionTime/60))
    print('Runtime (size experiment) in minutes: {}\n'.format(executionTime/60),file=f)         
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path+graphStr+'-size-'+str(beta)+'-full-data.csv')

    ################################################
    # Run Experiment: Varying Treatment Probability 
    ################################################
    startTime2 = time.time()
    n = 5000        # number of nodes in network
    diag = 6     # maximum norm of direct effect
    offdiag = 8  # maximum norm of indirect effect
    r = offdiag/diag

    results = []
    p_treatments = np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]) # treatment probabilities

    for p in p_treatments:
        print("Treatment Probability: {}".format(p))
        startTime3 = time.time()

        results.extend(run_experiment(G,T,n,p,r,graphStr,diag,beta))

        executionTime = (time.time() - startTime3)
        print('Runtime (in seconds) for p = {} step: {}'.format(p,executionTime))
        print('Runtime (in seconds) for p = {} step: {}'.format(p,executionTime),file=f)

    executionTime = (time.time() - startTime2)
    print('Runtime (tp experiment) in minutes: {}\n'.format(executionTime/60))        
    print('Runtime (tp experiment) in minutes: {}\n'.format(executionTime/60),file=f)     
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path+graphStr+'-tp-'+str(beta)+'-full-data.csv')

    ###########################################################
    # Run Experiment: Varying Ratio of Indirect & Direct Effects 
    ###########################################################
    n = 5000
    p = 0.50    # treatment probability
    diag = 10   # maximum norm of direct effect

    results = []
    ratio = [0.25,0.5,0.75,1,1/0.75,1/0.5,3,1/0.25]

    for r in ratio:
        print('ratio: {}'.format(r))
        startTime3 = time.time()

        results.extend(run_experiment(G,T,n,p,r,graphStr,diag,beta))

        executionTime = (time.time() - startTime3)
        print('Runtime (in seconds) for r = {} step: {}'.format(r,executionTime))
        print('Runtime (in seconds) for r = {} step: {}'.format(r,executionTime),file=f)

    executionTime = (time.time() - startTime2)
    print('Runtime (ratio experiment) in minutes: {}\n'.format(executionTime/60))          
    print('Runtime (ratio experiment) in minutes: {}\n'.format(executionTime/60),file=f)    
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path+graphStr+'-ratio-'+str(beta)+'-full-data.csv')

    executionTime = (time.time() - startTime1)
    print('Runtime (whole script) in hours: {}'.format(executionTime/3600))
    print('Runtime (whole script) in hours: {}'.format(executionTime/3600),file=f)

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
        if beta == 2:
            fy = ncps.ppom(ncps.f_quadratic, C, alpha)
        elif beta == 3:
            fy = ncps.ppom(ncps.f_cubic, C, alpha)
        elif beta == 4:
            fy = ncps.ppom(ncps.f_quadratic, C, alpha)
        else:
            print("ERROR: invalid degree")

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

        # M represents number of measurements - 1 (not including 0), which we assume to be beta
        M = beta

        K = ncps.seq_treated(M,p,n)            # sequence of treated for CRD + staggered rollout
        L = ncps.complete_coeffs(n, K)    # coefficents for GASR estimator under CRD
        P = ncps.seq_treatment_probs(M,p)    # sequence of probabilities for bern staggered rollout RD
        H = ncps.bern_coeffs(P)            # coefficents for GASR estimator under Bernoulli design
        
        for i in range(T):
            dict_base.update({'rep': i, 'Rand': 'CRD'})
            Z = ncps.staggered_rollout_complete(n, K)
            z = Z[M,:]
            y = fy(z)
            sums = ncps.outcome_sums(fy, Z)

            for alg in range(len(alg_names)):
                est = estimators[alg](y,z,sums,L,K/n,L)
                dict_base.update({'Estimator': alg_names[alg], 'Bias': (est-TTE)/TTE})
                results.append(dict_base.copy())

            dict_base.update({'Rand': 'Bernoulli'})
            Z = ncps.staggered_rollout_bern(n, P)
            z = Z[M,:]
            y = fy(z)
            sums = ncps.outcome_sums(fy, Z)
            Kr = np.sum(Z,1) # realized number of people treated at each time step
            Lr = ncps.complete_coeffs(n, Kr) # coeffs for variance reduction
            
            for alg in range(len(alg_names)):
                est = estimators[alg](y,z,sums,H,P,Lr)
                dict_base.update({'Estimator': alg_names[alg], 'Bias': (est-TTE)/TTE})
                results.append(dict_base.copy())

    return results


if __name__ == "__main__":
    main()