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
import time

path_to_module = 'mmc-causal-inference/Code-for-Experiments/'
sys.path.append(path_to_module)

import nci_linear_setup as ncls
import nci_polynomial_setup as ncps

save_path = 'mmc-causal-inference/outputFiles/'
save_path_graphs = 'mmc-causal-inference/outputFiles/Graphs/'

startTime = time.time()
# Run Experiment
G = 10          # number of graphs we want to average over
T = 20          # number of trials per graph
diag = 6        # controls magnitude of direct effects
offdiag = 8     # controls magnitude of indirect effects
r = offdiag/diag
p = 0.05        # treatment probability
graph = "config"
#experiment = "-size-bern-lin-" # vary size; bernoulli RD; linear model

results = []

sizes = np.array([1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000])
for n in sizes:
    print("n = {}".format(n))

    startTime2 = time.time()
    for g in range(G):
        if g % 5 == 0:
            print("Graph #{}".format(g))
        graph_rep = str(g)
        sz = str(n) + '-'
        
        # Generate random adjacency matrix
        A = ncls.config_model_nx(n,t=n*1000)

        # load weighted graph
        name = save_path_graphs + graph + sz + graph_rep + '-C'
        C,alpha = ncls.loadGraph(name, n)

        # potential outcomes model
        fy = lambda z: ncls.linear_pom(C,alpha,z)

        # compute and print true TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
        # print("Ground-Truth TTE: {}\n".format(TTE))

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
            #TTE_aware[i] = sum([y_diff[i] * degs[i]/treated_degs[i] * 1/(1-(1-p)**degs[i]) for i in range(n) if treated_degs[i] > 0]) * 1/n
            TTE_reduction[i] = 1/(1-(1-p)**n) * np.sum(y_diff)/np.sum(z)
            TTE_diff_means_naive[i] = ncls.diff_in_means_naive(y,z)
            TTE_diff_means_fraction[i] = ncls.diff_in_means_fraction(n,y,A,z,0.2)

            results.append({'Estimator': 'Graph-Agnostic', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_gasr[i]-TTE)/TTE, 'Graph':sz+graph_rep})
            #results.append({'Estimator': 'Graph-Aware', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_aware[i]-TTE)/TTE, 'Graph':graph_rep})
            results.append({'Estimator': 'Graph-Agnostic-VR', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_reduction[i]-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'OLS-Prop', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_ols[i]-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'OLS-Num', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_ols2[i]-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Diff-Means-Stnd', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_diff_means_naive[i]-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Diff-Means-Frac', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_diff_means_fraction[i]-TTE)/TTE, 'Graph':sz+graph_rep})
    executionTime2 = (time.time() - startTime2)
    print('Runtime (in seconds) for n = {} step: {}'.format(n,executionTime2))

executionTime = (time.time() - startTime)
print('Runtime of entire script in minutes: {}'.format(executionTime/60))
df = pd.DataFrame.from_records(results)
df.to_csv(save_path+graph+'-size-bern-linear-full-data.csv')