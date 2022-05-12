'''
Mayleen Cortez
Experiments: Varying the treatment probability
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

path_to_module = 'mmc-causal-inference/Code-for-Experiments/'
sys.path.append(path_to_module)

import nci_linear_setup as ncls
import nci_polynomial_setup as ncps

save_path = 'mmc-causal-inference/outputFiles/'
save_path_graphs = 'mmc-causal-inference/graphs/'

startTime = time.time()
# Run Experiment
beta = 2        # degree of outcomes model
n = 21000       # number of nodes in network
sz = str(n) + '-'
diag = 6        # maximum norm of direct effect
offdiag = 8     # maximum norm of indirect effect
r = offdiag/diag
G = 2      # number of graphs per value of p
T = 10      # number of trials per graph

p_treatments = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]) # treatment probabilities
results = []
graph = "CON"

for p in p_treatments:
    print("Treatment Probability: {}\n".format(p))
    startTime2 = time.time()
    P = ncps.seq_treatment_probs(beta,p)    # sequence of probabilities for bern staggered rollout RD
    H = ncps.bern_coeffs(beta,P)            # coefficents for GASR estimator under Bernoulli design

    for g in range(G):
        if g % 5 == 0:
            print("Graph #{}".format(g))
        graph_rep = + str(g)
        pr = str(p) + '-' 
        
        # load weighted graph
        name = save_path_graphs + graph + sz + graph_rep + '-C'
        C,alpha = ncls.loadGraph(name, n)
        A = (C > 0) + 0
        
        # potential outcomes model
        fy = lambda z: ncls.linear_pom(C,alpha,z)

        # calculate and print ground-truth TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
        #print("Ground-Truth TTE: {}\n".format(TTE))

        ####### Estimate ########
        TTE_gasr, TTE_pol1, TTE_pol2, TTE_linpoly, TTE_linspl, TTE_quadspl = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)

        for i in range(T):
            Z = ncps.staggered_rollout_bern(beta, n, P)
            y = fy(Z[beta,:])
            sums = ncps.outcome_sums(beta, fy, Z)
            TTE_gasr[i] = ncps.graph_agnostic(n, sums, H)
            TTE_pol1[i] = ncps.poly_regression_prop(beta, y, A, Z[beta,:])
            TTE_pol2[i] = ncps.poly_regression_num(beta, y, A, Z[beta,:])
            TTE_linpoly[i], TTE_linspl[i] = ncps.poly_interp_linear(n, P, sums)
            TTE_quadspl[i] = ncps.poly_interp_splines(n, P, sums, 'quadratic')

            results.append({'Estimator': 'Graph-Agnostic', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_gasr[i]-TTE)/TTE, 'Graph':graph_rep})
            results.append({'Estimator': 'LeastSqs-Prop', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_pol1[i]-TTE)/TTE, 'Graph':graph_rep})
            results.append({'Estimator': 'LeastSqs-Num', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_pol2[i]-TTE)/TTE, 'Graph':graph_rep})
            results.append({'Estimator': 'Interp-Lin', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_linpoly[i]-TTE)/TTE, 'Graph':graph_rep})
            results.append({'Estimator': 'Spline-Lin', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_linspl[i]-TTE)/TTE, 'Graph':graph_rep})
            results.append({'Estimator': 'Spline-Quad', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_quadspl[i]-TTE)/TTE, 'Graph':graph_rep})
    executionTime2 = (time.time() - startTime2)
    print('Runtime (in seconds) for p = {} step: {}'.format(p,executionTime2))

executionTime = (time.time() - startTime)
print('Runtime of entire script in minutes: {}'.format(executionTime/60)) 
df = pd.DataFrame.from_records(results)
df.to_csv(save_path+graph+'-tp-bern-quadratic-full-data.csv')  