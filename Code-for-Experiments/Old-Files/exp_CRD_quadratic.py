'''
Experiments with complete RD under polynomial setting
'''
# Setup
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from math import log, ceil
import pandas as pd
import seaborn as sns
import sys
import time
import scipy.sparse

path_to_module = 'Code-for-Experiments/'
sys.path.append(path_to_module)

import nci_linear_setup as ncls
import nci_polynomial_setup as ncps

save_path = 'mmc-causal-inference/outputFiles/'
save_path_graphs = 'mmc-causal-inference/graphs/'

startTime = time.time()
G = 1          # number of graphs per value of n
T = 1          # number of trials per graph
beta = 2        # degree of the outcomes model
# note if we change beta we have to change the corresponding potential outcomes function

###########################################
# Run Experiment: Varying Size of Network
###########################################
p = 0.50        # treatment probability
diag = 6        # controls magnitude of direct effects
offdiag = 8     # controls magnitude of indirect effects
r = offdiag/diag
graph = "CON"   # configuration model
sizes = np.array([5000, 9000, 15000, 19000, 23000, 27000, 31000, 35000])
results = []

for n in sizes:
    print("n = {}".format(n))
    sz = str(n) + '-'
    K = ncps.seq_treated(beta,p,n)            # sequence of treated for CRD + staggered rollout
    L = ncps.complete_coeffs(beta, n, K)    # coefficents for GASR estimator under CRD
    startTime1 = time.time()

    for g in range(G):
        if g % 5 == 0:
            print("Graph #{}".format(g))
        graph_rep = str(g)
        
        # load weighted graph
        name = save_path_graphs + graph + sz + graph_rep
        A = scipy.sparse.load_npz(name+'-A.npz')
        rand_wts = np.load(name+'-wts.npy')
        alpha = rand_wts[:,0].flatten()
        C = ncls.simpleWeights(A, diag, offdiag, rand_wts[:,1].flatten(), rand_wts[:,2].flatten())
        
        # potential outcomes model
        fy = ncps.ppom(ncps.f_quadratic, C, alpha)

        # calculate and print ground-truth TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
        #print("Ground-Truth TTE: {}\n".format(TTE))

        ####### Estimate ########

        for i in range(T):
            Z = ncps.staggered_rollout_complete(beta, n, K)
            y = fy(Z[beta,:])
            sums = ncps.outcome_sums(beta, fy, Z)
            TTE_gasr = ncps.graph_agnostic(n, sums, L)
            TTE_pol1 = ncps.poly_regression_prop(beta, y, A, Z[beta,:])
            TTE_pol2 = ncps.poly_regression_num(beta, y, A, Z[beta,:])
            TTE_linpoly, TTE_linspl = ncps.poly_interp_linear(n, K, sums)
            TTE_quadspl = ncps.poly_interp_splines(n, K, sums, 'quadratic')

            results.append({'Estimator': 'Graph-Agnostic-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_gasr-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'LeastSqs-Prop-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_pol1-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'LeastSqs-Num-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_pol2-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Interp-Lin-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_linpoly-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Spline-Lin-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_linspl-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Spline-Quad-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_quadspl-TTE)/TTE, 'Graph':sz+graph_rep})
    executionTime1 = (time.time() - startTime1)
    print('Runtime (in seconds) for n = {} step: {}\n'.format(n,executionTime1))

executionTime = (time.time() - startTime)
print('Runtime of size experiment in minutes: {}\n'.format(executionTime/60))    
df = pd.DataFrame.from_records(results)
df.to_csv(save_path+graph+'-size-CRD-quadratic-full-data.csv') 

################################################
# Run Experiment: Varying Treatment Probability 
################################################
n = 21000       # number of nodes in network
sz = str(n) + '-'
diag = 6        # maximum norm of direct effect
offdiag = 8     # maximum norm of indirect effect
r = offdiag/diag
graph = "CON"   # configuration model
p_treatments = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]) # treatment probabilities
results = []
startTime_tp = time.time()

for p in p_treatments:
    print("Treatment Probability: {}\n".format(p))
    K = ncps.seq_treated(beta,p,n)            # sequence of treated for CRD + staggered rollout
    L = ncps.complete_coeffs(beta, n, K)    # coefficents for GASR estimator under CRD
    pr = str(p) + '-'
    startTime1 = time.time()

    for g in range(G):
        if g % 5 == 0:
            print("Graph #{}".format(g))
        graph_rep = str(g) 
        
        # load weighted graph
        name = save_path_graphs + graph + sz + graph_rep
        A = scipy.sparse.load_npz(name+'-A.npz')
        rand_wts = np.load(name+'-wts.npy')
        alpha = rand_wts[:,0].flatten()
        C = ncls.simpleWeights(A, diag, offdiag, rand_wts[:,1].flatten(), rand_wts[:,2].flatten())
        
        # potential outcomes model
        fy = ncps.ppom(ncps.f_quadratic, C, alpha)

        # calculate and print ground-truth TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
        #print("Ground-Truth TTE: {}\n".format(TTE))

        ####### Estimate ########

        for i in range(T):
            Z = ncps.staggered_rollout_complete(beta, n, K)
            y = fy(Z[beta,:])
            sums = ncps.outcome_sums(beta, fy, Z)
            TTE_gasr = ncps.graph_agnostic(n, sums, L)
            TTE_pol1 = ncps.poly_regression_prop(beta, y, A, Z[beta,:])
            TTE_pol2 = ncps.poly_regression_num(beta, y, A, Z[beta,:])
            TTE_linpoly, TTE_linspl = ncps.poly_interp_linear(n, K, sums)
            TTE_quadspl = ncps.poly_interp_splines(n, K, sums, 'quadratic')

            results.append({'Estimator': 'Graph-Agnostic-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_gasr-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'LeastSqs-Prop-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_pol1-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'LeastSqs-Num-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_pol2-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Interp-Lin-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_linpoly-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Spline-Lin-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_linspl-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Spline-Quad-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_quadspl-TTE)/TTE, 'Graph':sz+graph_rep})
    executionTime1 = (time.time() - startTime1)
    print('Runtime (in seconds) for p = {} step: {}\n'.format(p,executionTime1))

executionTime = (time.time() - startTime_tp)
print('Runtime of tp experiment in minutes: {}\n'.format(executionTime/60)) 
df = pd.DataFrame.from_records(results)
df.to_csv(save_path+graph+'-tp-CRD-quadratic-full-data.csv')  

###########################################################
# Run Experiment: Varying Ratio of Indirect & Direct Effects 
###########################################################
p = 0.50        # treatment probability
n = 21000       # number of nodes in network
sz = str(n) + '-'
diag = 10       # maximum norm of direct effect
graph = "CON"   # configuration model
K = ncps.seq_treated(beta,p,n)            # sequence of treated for CRD + staggered rollout
L = ncps.complete_coeffs(beta, n, K)    # coefficents for GASR estimator under CRD
ratio = [0.25,0.5,0.75,1,1/0.75,1/0.5,3,1/0.25]
results = []
start_Time_rat = time.time()

for r in ratio:
    print('ratio: {}'.format(r))
    offdiag = r*diag   # maximum norm of indirect effect
    rat = str(r) + '-'
    startTime1 = time.time()

    for g in range(G):
        if g % 5 == 0:
            print("Graph #{}".format(g))
        graph_rep = str(g)

        # load weighted graph
        name = save_path_graphs + graph + sz + graph_rep
        A = scipy.sparse.load_npz(name+'-A.npz')
        rand_wts = np.load(name+'-wts.npy')
        alpha = rand_wts[:,0].flatten()
        C = ncls.simpleWeights(A, diag, offdiag, rand_wts[:,1].flatten(), rand_wts[:,2].flatten())
        
        # potential outcomes model
        fy = ncps.ppom(ncps.f_quadratic, C, alpha)

        # calculate and print ground-truth TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
        #print("Ground-Truth TTE: {}\n".format(TTE))

        ####### Estimate ########

        for i in range(T):
            Z = ncps.staggered_rollout_complete(beta, n, K)
            y = fy(Z[beta,:])
            sums = ncps.outcome_sums(beta, fy, Z)
            TTE_gasr = ncps.graph_agnostic(n, sums, L)
            TTE_pol1 = ncps.poly_regression_prop(beta, y, A, Z[beta,:])
            TTE_pol2 = ncps.poly_regression_num(beta, y, A, Z[beta,:])
            TTE_linpoly, TTE_linspl = ncps.poly_interp_linear(n, K, sums)
            TTE_quadspl = ncps.poly_interp_splines(n, K, sums, 'quadratic')

            results.append({'Estimator': 'Graph-Agnostic-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_gasr-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'LeastSqs-Prop-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_pol1-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'LeastSqs-Num-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_pol2-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Interp-Lin-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_linpoly-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Spline-Lin-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_linspl-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Spline-Quad-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_quadspl-TTE)/TTE, 'Graph':sz+graph_rep})
    executionTime1 = (time.time() - startTime1)
    print('Runtime (in seconds) for r = {} step: {}\n'.format(r,executionTime1))

executionTime = (time.time() - start_Time_rat)
print('Runtime of tp experiment in minutes: {}\n'.format(executionTime/60))   
df = pd.DataFrame.from_records(results)
df.to_csv(save_path+graph+'-ratio-CRD-quadratic-full-data.csv')  

executionTime = (time.time() - startTime)
print('Runtime of entire script in minutes: {}'.format(executionTime/60))  