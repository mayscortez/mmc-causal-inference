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
save_path = 'outputFiles/save/'
save_path_graphs = 'graphs/'

def main():
    G = 30          # number of graphs we want to average over
    T = 100          # number of trials per graph
    graphStr = "CON"   # configuration model

    f = open(save_path+'experiments_output_varying_deg.txt', 'w')

    ###########################################
    # Run Experiment: Varying Size of Network
    ###########################################
    startTime1 = time.time()
    r = 1.25
    p = 0.5        # treatment probability
    n = 15000
    diag=1

    results = []

    for beta in range(4):
        startTime2 = time.time()

        results.extend(run_experiment(G,T,n,p,r,graphStr,diag,beta))

        executionTime = (time.time() - startTime2)
        print('Runtime (in seconds) for beta = {} step: {}'.format(beta,executionTime))
        print('Runtime (in seconds) for beta = {} step: {}'.format(beta,executionTime),file=f)

    executionTime = (time.time() - startTime1)
    print('Runtime (size experiment) in minutes: {}\n'.format(executionTime/60))
    print('Runtime (size experiment) in minutes: {}\n'.format(executionTime/60),file=f)         
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_path+graphStr+'-varying-deg-full-data.csv')

    sys.stdout.close()

def run_experiment(G,T,n,p,r,graphStr,diag=1,beta=2,loadGraphs=False):
    
    offdiag = r*diag   # maximum norm of indirect effect

    results = []
    dict_base = {'p': p, 'ratio': r, 'n': n, 'beta': beta}

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
        fy = ncps.ppom(beta, C, alpha)

        # compute and print true TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
        #dict_base.update({'TTE': TTE})
        # print("Ground-Truth TTE: {}".format(TTE))

        ####### Estimate ########
        estimators = []
        estimators.append(lambda y,z,sums,L,K,Lr: ncps.graph_agnostic(n, sums, L))
        estimators.append(lambda y,z,sums,L,K,Lr: ncps.graph_agnostic(n, sums, Lr))
        estimators.append(lambda y,z,sums,L,K,Lr: ncps.poly_regression_prop(beta, y, A, z))
        estimators.append(lambda y,z,sums,L,K,Lr: ncps.poly_regression_num(beta, y, A, z))
        # estimators.append(lambda y,z,sums,L,K,Lr: ncps.poly_interp_linear(n, K, sums))
        # estimators.append(lambda y,z,sums,L,K,Lr: ncps.poly_interp_splines(n, K, sums, 'quadratic'))
        estimators.append(lambda y,z,sums,L,K,Lr: ncls.diff_in_means_naive(y,z))
        estimators.append(lambda y,z,sums,L,K,Lr: ncls.diff_in_means_fraction(n,y,A,z,0.75))

        alg_names = ['Graph-Agnostic', 'Graph-AgnosticVR', 'LeastSqs-Prop', 'LeastSqs-Num','Diff-Means-Stnd', 'Diff-Means-Frac-0.75']#, 'Spline-Lin','Spline-Quad']

        bern_est = [0,1,2]
        CRD_est = [1,2,3,4,5]

        # M represents number of measurements - 1 (not including 0), which we assume to be beta
        M = max(1,beta)

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

            for ind in range(len(CRD_est)):
                est = estimators[CRD_est[ind]](y,z,sums,L,K/n,L)
                dict_base.update({'Estimator': alg_names[CRD_est[ind]], 'Bias': (est-TTE)/TTE})
                results.append(dict_base.copy())

            dict_base.update({'Rand': 'Bernoulli'})
            Z = ncps.staggered_rollout_bern(n, P)
            z = Z[M,:]
            y = fy(z)
            sums = ncps.outcome_sums(fy, Z)
            Kr = np.sum(Z,1) # realized number of people treated at each time step
            Lr = ncps.complete_coeffs(n, Kr) # coeffs for variance reduction
            
            for ind in range(len(bern_est)):
                est = estimators[bern_est[ind]](y,z,sums,H,P,Lr)
                dict_base.update({'Estimator': alg_names[bern_est[ind]], 'Bias': (est-TTE)/TTE})
                results.append(dict_base.copy())

    return results


if __name__ == "__main__":
    main()