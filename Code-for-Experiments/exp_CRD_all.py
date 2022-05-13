'''
Mayleen Cortez
Experiments: Complete Randomized Design 
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

path_to_module = 'Code-for-Experiments/'

sys.path.append(path_to_module)

import nci_linear_setup as ncls
import nci_polynomial_setup as ncps

save_path = 'outputFiles/'
save_path_graphs = 'graphs/'

G = 1          # number of graphs we want to average over
T = 1          # number of trials per graph

###########################################
# Run Experiment: Varying Size of Network
###########################################
startTime1 = time.time()
diag = 6        # controls magnitude of direct effects
offdiag = 8     # controls magnitude of indirect effects
r = offdiag/diag
p = 0.05        # treatment probability
graph = "CON"   # configuration model

results = []
sizes = np.array([1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000])

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
        fy = lambda z: ncls.linear_pom(C,alpha,z)

        # compute and print true TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
        print("Ground-Truth TTE: {}\n".format(TTE))

        ####### Estimate ########
        TTE_ols, TTE_ols2, TTE_gasr, TTE_aware, TTE_reduction, TTE_diff_means_naive, TTE_diff_means_fraction = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)

        # coefficients for graph agnostic with complete staggered rollout
        pn = np.floor(p*n)
        l0 = (n - pn)/(-pn) - 1
        l1 = n/pn

        for i in range(T):
            z = ncls.completeRD(n,p)
            y = fy(z)

            TTE_gasr[i] = (1/n)*np.sum(l0*fy(np.zeros(n)) + l1*y)

            # degs = A.sum(axis=1)
            # treated_degs = A.dot(z)
            #TTE_aware[i] = sum([y_diff[i] * degs[i]/treated_degs[i] * 1/(1-(1-p)**degs[i]) for i in range(n) if treated_degs[i] > 0]) * 1/n
            
            y0 = fy(np.zeros(n))
            y_diff = y - y0
            TTE_reduction[i] = 1/(1-(1-p)**n) * np.sum(y_diff)/np.sum(z)

            TTE_diff_means_naive[i] = ncls.diff_in_means_naive(y,z)
            TTE_diff_means_fraction[i] = ncls.diff_in_means_fraction(n,y,A,z,0.2)
            TTE_ols[i] = ncls.est_ols_gen(y,A,z)
            TTE_ols2[i] = ncls.est_ols_treated(y,A,z)

            results.append({'Estimator': 'Graph-Agnostic-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_gasr[i]-TTE)/TTE, 'Graph':sz+graph_rep})
            #results.append({'Estimator': 'Graph-Aware-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_aware[i]-TTE)/TTE})
            results.append({'Estimator': 'Graph-AgnosticVR-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_reduction[i]-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'OLS-Prop-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_ols[i]-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'OLS-Num-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_ols2[i]-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Diff-Means-Stnd-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_diff_means_naive[i]-TTE)/TTE, 'Graph':sz+graph_rep})
            results.append({'Estimator': 'Diff-Means-Frac-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_diff_means_fraction[i]-TTE)/TTE, 'Graph':sz+graph_rep})
    executionTime = (time.time() - startTime2)
    print('Runtime (in seconds) for n = {} step: {}'.format(n,executionTime))
executionTime = (time.time() - startTime1)
print('Runtime (size experiment) in minutes: {}'.format(executionTime/60))        
df = pd.DataFrame.from_records(results)
df.to_csv(save_path+graph+'-size-CRD-linear-full-data.csv')

################################################
# Run Experiment: Varying Treatment Probability 
################################################
startTime2 = time.time()
n = 5000        # number of nodes in network
sz = str(n) + '-'
diag = 6     # maximum norm of direct effect
offdiag = 8  # maximum norm of indirect effect
r = offdiag/diag

p_treatments = np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]) # treatment probabilities
results = []
graph = "CON"

for p in p_treatments:
    print("Treatment Probability: {}\n".format(p))
    startTime3 = time.time()

    for g in range(G):
        if g % 5 == 0:
            print("Graph #{}".format(g))
        graph_rep = str(g)
        pr = str(p) + '-' 

        # # load weighted graph
        # name = save_path_graphs + graph + sz + graph_rep + '-C'
        # C,alpha = ncls.loadGraph(name, n)
        # A = (C > 0) + 0

        name = save_path_graphs + graph + sz + graph_rep
        A = scipy.sparse.load_npz(name+'-A.npz')
        rand_wts = np.load(name+'-wts.npy')
        alpha = rand_wts[:,0].flatten()
        C = ncls.simpleWeights(A, diag, offdiag, rand_wts[:,1].flatten(), rand_wts[:,2].flatten())

        # potential outcomes model
        fy = lambda z: ncls.linear_pom(C,alpha,z)

        # calculate and print ground-truth TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
        # print("Ground-Truth TTE: {}\n".format(TTE))
    
        ####### Estimate ########
        TTE_ols, TTE_ols2, TTE_gasr, TTE_aware, TTE_reduction, TTE_diff_means_naive, TTE_diff_means_fraction = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
        
        # coefficients for graph agnostic with complete staggered rollout
        pn = np.floor(p*n)
        l0 = (n - pn)/(-pn) - 1
        l1 = n/pn

        for i in range(T):
            z = ncls.completeRD(n,p)
            y = fy(z)

            TTE_gasr[i] = (1/n)*np.sum(l0*fy(np.zeros(n)) + l1*y)
            
            # degs = A.sum(axis=1)
            # treated_degs = A.dot(z)
            #TTE_aware[i] = sum([y_diff[i] * degs[i]/treated_degs[i] * 1/(1-(1-p)**degs[i]) for i in range(n) if treated_degs[i] > 0]) * 1/n
            
            y0 = fy(np.zeros(n))
            y_diff = y - y0
            TTE_reduction[i] = 1/(1-(1-p)**n) * np.sum(y_diff)/np.sum(z)
            TTE_diff_means_naive[i] = ncls.diff_in_means_naive(y,z)
            TTE_diff_means_fraction[i] = ncls.diff_in_means_fraction(n,y,A,z,0.2)
            TTE_ols[i] = ncls.est_ols_gen(y,A,z)
            TTE_ols2[i] = ncls.est_ols_treated(y,A,z)

            results.append({'Estimator': 'Graph-Agnostic-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_gasr[i]-TTE)/TTE, 'Graph':pr+graph_rep})
            #results.append({'Estimator': 'Graph-Aware-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_aware[i]-TTE)/TTE})
            results.append({'Estimator': 'Graph-AgnosticVR-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_reduction[i]-TTE)/TTE, 'Graph':pr+graph_rep})
            results.append({'Estimator': 'OLS-Prop-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_ols[i]-TTE)/TTE, 'Graph':pr+graph_rep})
            results.append({'Estimator': 'OLS-Num-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_ols2[i]-TTE)/TTE, 'Graph':pr+graph_rep})
            results.append({'Estimator': 'Diff-Means-Stnd-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_diff_means_naive[i]-TTE)/TTE, 'Graph':pr+graph_rep})
            results.append({'Estimator': 'Diff-Means-Frac-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_diff_means_fraction[i]-TTE)/TTE, 'Graph':pr+graph_rep})
    executionTime = (time.time() - startTime3)
    print('Runtime (in seconds) for p = {} step: {}'.format(p,executionTime))

executionTime = (time.time() - startTime2)
print('Runtime (tp experiment) in minutes: {}'.format(executionTime/60))        
df = pd.DataFrame.from_records(results)
df.to_csv(save_path+graph+'-tp-CRD-linear-full-data.csv')

###########################################################
# Run Experiment: Varying Ratio of Indirect & Direct Effects 
###########################################################
n = 5000
sz = str(n) + '-'
diag = 10   # maximum norm of direct effect

p = 0.06    # treatment probability

ratio = [0.25,0.5,0.75,1,1/0.75,1/0.5,3,1/0.25]
results = []
graph = "CON"

# coefficients for graph agnostic with complete staggered rollout
pn = np.floor(p*n)
l0 = (n - pn)/(-pn) - 1
l1 = n/pn

####### Experiments ########
for r in ratio:
    print('ratio: {}'.format(r))
    offdiag = r*diag   # maximum norm of indirect effect
    rat = str(r) + '-'
    startTime3 = time.time()

    for g in range(G):
        if g % 5 == 0:
            print("Graph #{}".format(g))
        graph_rep = str(g)

        # # load graph
        # name = save_path_graphs + graph + sz + rat + graph_rep + '-C'
        # C, alpha = ncls.loadWeights(name, symmetric=False)
        # A = (C > 0) + 0

        name = save_path_graphs + graph + sz + graph_rep
        A = scipy.sparse.load_npz(name+'-A.npz')
        rand_wts = np.load(name+'-wts.npy')
        alpha = rand_wts[:,0].flatten()
        C = ncls.simpleWeights(A, diag, offdiag, rand_wts[:,1].flatten(), rand_wts[:,2].flatten())

        ## Potential Outcomes Model ##
        fy = lambda z: ncls.linear_pom(C,alpha,z)

        # Calculate and print ground truth TTE
        TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
        #print("Ground-Truth TTE: {}\n".format(TTE))

        ####### Estimate ########
        TTE_ols, TTE_ols2, TTE_gasr, TTE_aware, TTE_reduction, TTE_diff_means_naive, TTE_diff_means_fraction = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)

        for i in range(T):
            z = ncls.completeRD(n,p)
            y = fy(z)

            TTE_gasr[i] = (1/n)*np.sum(l0*fy(np.zeros(n)) + l1*y)

            # degs = A.sum(axis=1)
            # treated_degs = A.dot(z)
            #TTE_aware[i] = sum([y_diff[i] * degs[i]/treated_degs[i] * 1/(1-(1-p)**degs[i]) for i in range(n) if treated_degs[i] > 0]) * 1/n
            
            y0 = fy(np.zeros(n))
            y_diff = y - y0
            TTE_reduction[i] = 1/(1-(1-p)**n) * np.sum(y_diff)/np.sum(z)

            TTE_diff_means_naive[i] = ncls.diff_in_means_naive(y,z)
            TTE_diff_means_fraction[i] = ncls.diff_in_means_fraction(n,y,A,z,0.2)
            TTE_ols[i] = ncls.est_ols_gen(y,A,z)
            TTE_ols2[i] = ncls.est_ols_treated(y,A,z)

            results.append({'Estimator': 'Graph-Agnostic-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_gasr[i]-TTE)/TTE, 'Graph':rat+graph_rep})
            #results.append({'Estimator': 'Graph-Aware-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_aware[i]-TTE)/TTE})
            results.append({'Estimator': 'Graph-AgnosticVR-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_reduction[i]-TTE)/TTE, 'Graph':rat+graph_rep})
            results.append({'Estimator': 'OLS-Prop-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_ols[i]-TTE)/TTE, 'Graph':rat+graph_rep})
            results.append({'Estimator': 'OLS-Num-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_ols2[i]-TTE)/TTE, 'Graph':rat+graph_rep})
            results.append({'Estimator': 'Diff-Means-Stnd-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_diff_means_naive[i]-TTE)/TTE, 'Graph':rat+graph_rep})
            results.append({'Estimator': 'Diff-Means-Frac-C', 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_diff_means_fraction[i]-TTE)/TTE, 'Graph':rat+graph_rep})
    executionTime = (time.time() - startTime3)
    print('Runtime (in seconds) for r = {} step: {}'.format(r,executionTime))

executionTime = (time.time() - startTime2)
print('Runtime (ratio experiment) in minutes: {}'.format(executionTime/60))  

executionTime = (time.time() - startTime1)
print('Runtime (whole script) in hours: {}'.format(executionTime/3600))        
df = pd.DataFrame.from_records(results)
df.to_csv(save_path+graph+'-ratio-CRD-linear-full-data.csv')