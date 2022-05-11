'''
Mayleen Cortez
Experiments: Varying the ratio
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

startTime = time.time()
# Run Experiment
G = 25          # number of graphs per value of n
T = 25          # number of trials per graph
beta = 2        # degree of the outcomes model
p = 0.50        # treatment probability
diag = 5        # maximum norm of direct effect
offdiag = 5     # maximum norm of indirect effect
r = offdiag/diag

sizes = np.array([5000, 7500, 10000, 12500, 15000, 17500, 20000])
results = []

for n in sizes:
    print(n)

    for g in range(G):
        graph_rep = str(n) + '-' + str(g)
        
        # Generate network
        A = ncls.config_model_nx(n, t = n*1000, law = "out")
        graph = "con-outpwr"

        # weights from simple model
        C = ncls.simpleWeights(A, diag, offdiag)

        # Generate (normalized) weights
        # C = ncls.weights_node_deg_unif(A)
        # C = C*A
        # C = ncls.normalized_weights(C, diag=diag_max, offdiag=offdiag_max)

        # baseline parameters
        alpha = np.random.rand(n)
        
        # potential outcomes model
        fy = lambda z: ncls.linear_pom(C,alpha,z)

        # calculate and print ground-truth TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
        #print("Ground-Truth TTE: {}\n".format(TTE))

        ####### Estimate ########
        TTE_gasr, TTE_pol1, TTE_pol2, TTE_linpoly, TTE_linspl, TTE_quadspl = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)

        for i in range(T):
            Z, H, P = ncps.staggered_rollout_bern(beta, p, n)
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

executionTime = (time.time() - startTime)
print('Runtime in seconds: {}'.format(executionTime))       
df = pd.DataFrame.from_records(results)
df.to_csv(save_path+graph+'-size-bern-quadratic-full-data.csv')  