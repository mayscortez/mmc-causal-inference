'''
Script to generate random graphs for different values of n
'''
# Setup
import numpy as np
import random
import networkx as nx
from math import log, ceil
import sys
import time
import scipy.sparse

path_to_module = 'Code-for-Experiments/'
sys.path.append(path_to_module)

import nci_linear_setup as ncls
import nci_polynomial_setup as ncps

save_path = 'graphs/'
save_path_graphs = 'graphs/'

startTime = time.time()
prevTime = startTime

# Run Experiment
G = 1               # number of graphs we want to average over
diag = 6            # controls magnitude of direct effects
offdiag = 8         # controls magnitude of indirect effects
r = offdiag/diag    # ratio btw indirect & direct effects
graph = "CON"       # configuration model
# graph = "ER"     # Erdos Renyi

sizes = np.array([1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000, 21000, 23000, 25000, 27000, 29000, 31000, 33000, 35000, 38000])
for n in sizes:
    print(n)

    for g in range(G):
        graph_rep = str(g)
        sz = str(n)
        # Generate Network
        A = ncls.config_model_nx(n,t=n*1000)

        # save graph
        name = save_path_graphs + graph + sz + '-' + graph_rep + '-A'
        #ncls.printGraph(A, name, symmetric=False)
        scipy.sparse.save_npz(name,A)
        
        # # null effects
        # alpha = np.random.rand(n)

        # # weights from simple model
        # C = ncls.simpleWeights(A, diag, offdiag)

        # # # Generate normalized weights
        # # C = ncls.weights_node_deg_unif(A)
        # # C = C*A
        # # C = ncls.normalized_weights(C, diag, offdiag)

        # # Save weights
        # name = save_path_graphs + graph + sz + '-' + graph_rep + '-C'
        # ncls.printWeights(C, alpha, name)
        
        # random numbers to generate weights; each of three columns are for alpha, rand_diag, rand_offdiag
        rand_wts = np.random.rand(n,3)
        # Save weights
        name = save_path_graphs + graph + sz + '-' + graph_rep + '-wts'
        np.save(name,rand_wts)

    print('Time to generate graphs of size {} in seconds: {}'.format(n,time.time() - prevTime))
    prevTime = time.time()

executionTime = (time.time() - startTime)
print('Total runtime in seconds: {}'.format(executionTime))

