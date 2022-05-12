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
save_path_graphs = 'mmc-causal-inference/graphs/'

startTime = time.time()
# Run Experiment
n = 21000       # number of nodes in network
sz = str(n) + '-'
diag = 10       # maximum norm of direct effect
beta = 2        # degree of outcomes model
p = 0.50        # treatment probability
G = 2          # number of graphs per value of r
T = 10          # number of trials per graph

P = ncps.seq_treatment_probs(beta,p)    # sequence of probabilities for bern staggered rollout RD
H = ncps.bern_coeffs(beta,P)            # coefficents for GASR estimator under Bernoulli design
ratio = [0.25,0.5,0.75,1,1/0.75,1/0.5,3,1/0.25]
results = []
graph = "CON"

# baseline parameters
alpha = np.random.rand(n)

for r in ratio:
    print('ratio: {}'.format(r))
    offdiag = r*diag   # maximum norm of indirect effect
    rat = str(r) + '-'
    startTime2 = time.time()

    for g in range(G):
        if g % 5 == 0:
            print("Graph #{}".format(g))
        graph_rep = str(g)

        # load graph
        name = save_path_graphs + graph + sz + graph_rep + '-A'
        A = ncls.loadGraph(name, symmetric=False)
        
        # null effects
        alpha = np.random.rand(n)

        # weights from simple model
        C = ncls.simpleWeights(A, diag, offdiag)

        # Save weights
        name = save_path_graphs + graph + sz + rat + graph_rep + '-C'
        ncls.printWeights(C, alpha, name)

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

            results.append({'Estimator': 'Graph-Agnostic', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_gasr[i]-TTE)/TTE, 'Graph':rat+graph_rep})
            results.append({'Estimator': 'LeastSqs-Prop', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_pol1[i]-TTE)/TTE, 'Graph':rat+graph_rep})
            results.append({'Estimator': 'LeastSqs-Num', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_pol2[i]-TTE)/TTE, 'Graph':rat+graph_rep})
            results.append({'Estimator': 'Interp-Lin', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_linpoly[i]-TTE)/TTE, 'Graph':rat+graph_rep})
            results.append({'Estimator': 'Spline-Lin', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_linspl[i]-TTE)/TTE, 'Graph':rat+graph_rep})
            results.append({'Estimator': 'Spline-Quad', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_quadspl[i]-TTE)/TTE, 'Graph':rat+graph_rep})
    executionTime2 = (time.time() - startTime2)
    print('Runtime (in seconds) for r = {} step: {}'.format(r,executionTime2))

executionTime = (time.time() - startTime)
print('Runtime of entire script in minutes: {}'.format(executionTime/60))    
df = pd.DataFrame.from_records(results)
df.to_csv(save_path+graph+'-ratio-bern-quadratic-full-data.csv')  