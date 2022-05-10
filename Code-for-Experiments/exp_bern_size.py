'''
Mayleen Cortez
Experiments: Varying the size of the graph
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

path_to_module = 'Code-for-Experiments/'
sys.path.append(path_to_module)

import nci_linear_setup as ncls
import nci_polynomial_setup as ncps

save_path = 'ouputFiles/'

# Run Experiment
T = 100        # number of trials
diag = 6        # controls magnitude of direct effects
offdiag = 8     # controls magnitude of indirect effects
r = offdiag/diag
p = 0.05        # treatment probability

results = []

sizes = np.array([500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000])
for n in sizes:
    print(n)

    # Generate random adjacency matrix
    A = ncls.config_model_nx(n,t=n*1000)
    graph = "con-outpwr"

    # null effects
    alpha = np.random.rand(n)

    #simplified model for generating the weights
    C_diag = diag*np.random.rand(n)
    C_offdiag = offdiag*np.random.rand(n)

    in_deg = np.sum(A,axis=1)  # array of the in-degree of each node
    out_deg = np.sum(A,axis=0)  # array of the in-degree of each node
    C = np.dot(np.diag(in_deg), A - np.eye(n))
    col_sum = np.sum(C,axis=0)
    col_sum = np.where(col_sum != 0, col_sum, col_sum+1)
    C = C / col_sum
    C = np.dot(C, np.diag(C_offdiag))
    np.fill_diagonal(C, C_diag)

    # # Generate normalized weights
    # C = ncls.weights_node_deg_unif(A)
    # C = C*A
    # C = ncls.normalized_weights(C, diag, offdiag)

    # potential outcomes model
    fy = lambda z: ncls.linear_pom(C,alpha,z)

    # compute and print true TTE
    TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
    print("Ground-Truth TTE: {}\n".format(TTE))

    ####### Estimate ########
    TTE_ols, TTE_ols2, TTE_gasr, TTE_aware, TTE_reduction, TTE_diff_means_naive, TTE_diff_means_fraction = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)

    for i in range(T):
        z = ncls.bernoulli(n,p)
        y = fy(z)
        y_diff = y - fy(np.zeros(n))
        degs = np.sum(A,axis=1)
        treated_degs = A.dot(z)

        TTE_ols[i] = ncls.est_ols_gen(y,A,z)
        TTE_ols2[i] = ncls.est_ols_treated(y,A,z)
        TTE_gasr[i] = 1/(p*n) * np.sum(y_diff)
        TTE_aware[i] = sum([y_diff[i] * degs[i]/treated_degs[i] * 1/(1-(1-p)**degs[i]) for i in range(n) if treated_degs[i] > 0]) * 1/n
        TTE_reduction[i] = 1/(1-(1-p)**n) * np.sum(y_diff)/np.sum(z)
        TTE_diff_means_naive[i] = ncls.diff_in_means_naive(y,z)
        TTE_diff_means_fraction[i] = ncls.diff_in_means_fraction(n,y,A,z,0.2)

        results.append({'Estimator': 'Graph-Agnostic', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_gasr[i]-TTE)/TTE})
        results.append({'Estimator': 'Graph-Aware', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_aware[i]-TTE)/TTE})
        results.append({'Estimator': 'Graph-Agnostic-VR', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_reduction[i]-TTE)/TTE})
        results.append({'Estimator': 'OLS-Prop', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_ols[i]-TTE)/TTE})
        results.append({'Estimator': 'OLS-Num', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_ols2[i]-TTE)/TTE})
        results.append({'Estimator': 'Diff-Means-Stnd', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_diff_means_naive[i]-TTE)/TTE})
        results.append({'Estimator': 'Diff-Means-Frac', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_diff_means_fraction[i]-TTE)/TTE})

df = pd.DataFrame.from_records(results)
df.to_csv(save_path+graph+'-size-bern-linear-full-data.csv')