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

path_to_module = '/home/mayleencortez/mmc-causal-inference/Code-for-Experiments'
sys.path.append(path_to_module)

import nci_linear_setup as ncls
import nci_polynomial_setup as ncps

save_path = '/home/mayleencortez/datafiles/'

n = 50000        # number of nodes in network
diag_max = 5     # maximum norm of direct effect
offdiag_max = 5  # maximum norm of indirect effect
r = offdiag_max/diag_max
beta = 2

T = 100  # number of trials
d = 1     # influence and malleability dimension size

p_treatments = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]) # treatment probabilities
results = []

A = ncls.config_model_nx(n, t = n*1000, law = "out")
graph = "con-outpwr"

# Generate (normalized) weights
C = ncls.weights_node_deg_unif(A, d)
C = C*A
C = ncls.normalized_weights(C, diag=diag_max, offdiag=offdiag_max)

# baseline parameters
alpha = np.random.rand(n)

# potential outcomes model
fy = lambda z: ncls.linear_pom(C,alpha,z)

for p in p_treatments:
    print("Treatment Probability: {}\n".format(p))

    # calculate and print ground-truth TTE
    TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
    print("Ground-Truth TTE: {}\n".format(TTE))

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

        results.append({'Estimator': 'Graph-Agnostic', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_gasr[i]-TTE)/TTE})
        results.append({'Estimator': 'LeastSqs-Prop', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_pol1[i]-TTE)/TTE})
        results.append({'Estimator': 'LeastSqs-Num', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_pol2[i]-TTE)/TTE})
        results.append({'Estimator': 'Interp-Lin', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_linpoly[i]-TTE)/TTE})
        results.append({'Estimator': 'Spline-Lin', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_linspl[i]-TTE)/TTE})
        results.append({'Estimator': 'Spline-Quad', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_quadspl[i]-TTE)/TTE})
        
df = pd.DataFrame.from_records(results)
df.to_csv(save_path+graph+'-tp-bern-quadratic-full-data.csv')  