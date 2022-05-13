'''
Mayleen Cortez
Experiments: the size of the graph
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
# Run Experiment
G = 1          # number of graphs per value of n
T = 1          # number of trials per graph
beta = 2        # degree of the outcomes model
p = 0.50        # treatment probability
diag = 6        # controls magnitude of direct effects
offdiag = 8     # controls magnitude of indirect effects
r = offdiag/diag
graph = "CON"   # configuration model
P = ncps.seq_treatment_probs(beta,p)    # sequence of probabilities for bern staggered rollout RD
H = ncps.bern_coeffs(beta,P)            # coefficents for GASR estimator under Bernoulli design
sizes = np.array([5000, 9000, 15000, 19000, 23000, 27000, 31000, 35000])
results = []

for n in sizes:
    print("n = {}".format(n))
    sz = str(n) + '-'
    startTime2 = time.time()

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
            Z = ncps.staggered_rollout_bern(beta, n, P)
            K = np.sum(Z,1)
            L = ncps.complete_coeffs(beta, n, K)
            y = fy(Z[beta,:])
            sums = ncps.outcome_sums(beta, fy, Z)
            TTE_gasr = ncps.graph_agnostic(n, sums, H)
            TTE_gasr_VR = ncps.graph_agnostic(n, sums, L)
            TTE_pol1 = ncps.poly_regression_prop(beta, y, A, Z[beta,:])
            TTE_pol2 = ncps.poly_regression_num(beta, y, A, Z[beta,:])
            TTE_linpoly, TTE_linspl = ncps.poly_interp_linear(n, P, sums)
            TTE_quadspl = ncps.poly_interp_splines(n, P, sums, 'quadratic')

            results.append({'Estimator': 'Graph-Agnostic', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_gasr-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Graph-AgnosticVR', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_gasr_VR-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'LeastSqs-Prop', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_pol1-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'LeastSqs-Num', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_pol2-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Interp-Lin', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_linpoly-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Spline-Lin', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_linspl-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Spline-Quad', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_quadspl-TTE)/TTE, 'Graph':sz+graph_rep})
    executionTime2 = (time.time() - startTime2)
    print('Runtime (in seconds) for n = {} step: {}\n'.format(n,executionTime2))

executionTime = (time.time() - startTime)
print('Runtime of entire script in minutes: {}'.format(executionTime/60))    
df = pd.DataFrame.from_records(results)
df.to_csv(save_path+graph+'-size-bern-quadratic-full-data.csv')  