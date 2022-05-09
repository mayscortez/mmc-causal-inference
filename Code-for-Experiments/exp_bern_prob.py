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

n = 10000        # number of nodes in network
diag_max = 6     # maximum norm of direct effect
offdiag_max = 8  # maximum norm of indirect effect
r = offdiag_max/diag_max

T = 1000  # number of trials
d = 1     # influence and malleability dimension size

p_treatments = np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]) # treatment probabilities
results = []

A = ncls.config_model_nx(n, t = n*1000, law = "out")
graph = "con-outpwr"

for p in p_treatments:
    print("Treatment Probability: {}\n".format(p))

    # Generate (normalized) weights
    C = ncls.weights_node_deg_unif(A, d)
    C = C*A
    C = ncls.normalized_weights(C, diag=diag_max, offdiag=offdiag_max)

    # baseline parameters
    alpha = np.random.rand(n)

    # potential outcomes model
    fy = lambda z: ncls.linear_pom(C,alpha,z)

    # calculate and print ground-truth TTE
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
        TTE_diff_means_fraction[i] = ncls.diff_in_means_fraction(n,y,A,z,tol=0.2)

        results.append({'Estimator': 'Graph-Agnostic', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_gasr[i]-TTE)/TTE})
        results.append({'Estimator': 'Graph-Aware', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_aware[i]-TTE)/TTE})
        results.append({'Estimator': 'Graph-Agnostic-VR', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_reduction[i]-TTE)/TTE})
        results.append({'Estimator': 'OLS-Prop', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_ols[i]-TTE)/TTE})
        results.append({'Estimator': 'OLS-Num', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_ols2[i]-TTE)/TTE})
        results.append({'Estimator': 'Diff-Means-Stnd', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_diff_means_naive[i]-TTE)/TTE})
        results.append({'Estimator': 'Diff-Means-Frac', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_diff_means_fraction[i]-TTE)/TTE})
        
df = pd.DataFrame.from_records(results)
df.to_csv(save_path+graph+'-tp-bern-linear-full-data.csv')  