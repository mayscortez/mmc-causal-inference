import numpy as np
import random
#import matplotlib.pyplot as plt
import networkx as nx
#from math import log, ceil
#import pandas as pd
#import seaborn as sns
#from numpy.core.fromnumeric import shape

# https://stackoverflow.com/a/3590105/9969751
def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

def constrained_sum_sample_nonneg(n, total):
    """Return a randomly chosen list of n nonnegative integers summing to total.
    Each such list is equally likely to occur."""

    return [x - 1 for x in constrained_sum_sample_pos(n, total + n)]

# functions to create random networks
def erdos_renyi(n,p,undirected=False):
    ''' 
    Generates a random network of n nodes using the Erdos-Renyi method,
    where the probability that an edge exists between two nodes is p.

    Returns the adjacency matrix of the network as an n by n numpy array
    '''
    A = np.random.rand(n,n)
    A = (A < p) + 0
    A[range(n),range(n)] = 1   # everyone is affected by their own treatment
    if undirected:
        A = symmetrizeGraph(A)
    return A

def config_model(di):
    '''
    Given a sequence of in-degrees di, returns the adjacency matrix of a random
    network generated by a configuration model. The out-degree is constant-ish.
    '''
    n = di.shape[0]
    ds = np.sum(di)          # in-degree sum
    do = np.ones(n)*ds//n    # sequence of out-degrees
    
    for i in range(n):
        if ds > np.sum(do): do[i] += 1
    
    heads, tails = [], []
    for i in range(n):
        heads += [i]*int(do[i])
        tails += [i]*int(di[i])
    
    random.shuffle(tails)

    # fake it: if we try to add repeated edge, just don't
    # TODO: maybe don't fake it?
    
    A = np.eye(n,n)  # everyone is affected by their own treatment
    A[tails, heads] = 1

    return A

def config_model_nx(N, t = 50000, law = "out"):
    '''
    Returns the adjacency matrix A (as a numpy array) of a networkx configuration 
    model with power law degree sequences 
    
    N (int): number of nodes
    t (int): number of tries for powerlaw sequence
    law (str): inicates whether in-, out- or both in- and out-degrees should be distributed as a power law
        "out" : out-degrees distributed as powerlaw, in-degrees sum up to same # as out-degrees
        "in" : in-degrees distributed as powerlaw, out-degrees sum up to same # as in-degrees
        "both" : both in- and out-degrees distributed as powerlaw
    '''
    assert law in ["out", "in", "both"], "law must = 'out', 'in', or 'both'"
    if law == "out":
        deg_seq_out = nx.random_powerlaw_tree_sequence(N,tries=t)
        deg_seq_in = constrained_sum_sample_nonneg(N,np.sum(deg_seq_out))
    elif law == "in":
        deg_seq_in = nx.random_powerlaw_tree_sequence(N,tries=t)
        deg_seq_out = constrained_sum_sample_nonneg(N,np.sum(deg_seq_in))
    else:
        deg_seq_out = nx.random_powerlaw_tree_sequence(N,tries=t)
        deg_seq_in = nx.random_powerlaw_tree_sequence(N,tries=t)
    
    G = nx.generators.degree_seq.directed_configuration_model(deg_seq_in,deg_seq_out)
    
    G.remove_edges_from(nx.selfloop_edges(G)) # remove self-loops
    G = nx.DiGraph(G)                         # remove parallel edges
    A = nx.to_numpy_array(G)                  # retrieve adjacency matrix
    np.fill_diagonal(A, 1)                    # everyone is affected by their own treatment
    
    return A

def small_world(n,k,p):
    '''
    Returns adjacency matrix (A, numpy array) of random network using the Watts-
    Strogatz graph function in the networkx package. 
    
    n (int): number of nodes
    k (int): Each node is joined with its k nearest neighbors in a ring topology
    p (float, in [0,1]): probability of rewiring each edge
    '''
    G = nx.watts_strogatz_graph(n, k, p)
    return nx.to_numpy_array(G)

def SBM(clusterSize, probabilities):
    '''
    Returns adjacency matrix (A, numpy array) of a stochastic block matrix where
    
    clusterSize (): 
    probabilities ():
        
    **ASSUME SYMMETRIC PROBABILITY MATRIX**
    '''
    p = np.kron(probabilities, np.ones((clusterSize,clusterSize)))
    n = p.shape[0]
    A = np.random.rand(n,n)
    A = (A < p) + 0
    A[range(n),range(n)] = 1   # everyone is affected by their own treatment
    return A

def symmetrizeGraph(A):
  n = A.shape[0]
  if A.shape[1] != n:
    print("Error: adjacency matrix is not square!")
    return A
  for i in range(n):
    for j in range(i):
      A[i,j] = A[j,i]
  return A

def printGraph(A,filename, symmetric=True):
  f = open(filename, 'w')
  print("# graph", file=f)
  print("# Nodes: "+str(A.shape[0]), file=f)
  print("# NodeId\tNodeId", file=f)
  indices = np.argwhere(A)
  for i in indices:
    if symmetric and i[0] > i[1]:
      continue
    print(str(i[0])+"\t"+str(i[1]), file=f)
  f.close()

def loadGraph(filename, n, symmetric=True):
  A = np.zeros((n,n))
  f = open(filename, 'r')
  next(f)
  next(f)
  next(f)
  for line in f:
    line = line.strip()
    ind = line.split()
    A[int(ind[0]),int(ind[1])] = 1
    if symmetric:
      A[int(ind[1]),int(ind[0])] = 1
  return A

def loadPartition(filename, n):
  partition = np.zeros(n)
  f = open(filename, 'r')
  next(f)
  c_id = 0
  for line in f:
    line = line.strip()
    ind = line.split()
    for i in ind:
      partition[int(i)] = c_id
    c_id += 1
  
  clusters = np.zeros((n,c_id))
  for i in range(n):
    clusters[i,int(partition[int(i)])] = 1

  return clusters

# Functions to generate network weights

def weights_im_normal(n, d=1, sigma=0.1, neg=0):
    '''
    Returns weights C (numpy array) under the influence and malleability framework,
    with Gaussian mean-zero noise.
    
    n (int): number of individuals
    d (int): number of influence dimensions
    sigma (float): standard deviation of noise
    neg (0 or 1): 0 if restricted to non-negative weights, 1 otherwise
    '''
    X = np.random.rand(d,n)              # influence
    W = np.random.rand(d,n)              # malliability
    
    if neg==0:
      E = np.abs(np.random.normal(scale=sigma, size=(n,n)))
    else:
      E = np.random.normal(scale=sigma, size=(n,n))
    C = X.T.dot(W)+E
    return C

def weights_im_beta(n, d=1, alpha=0.5, gamma=0.5):
    '''
    Returns weights C (numpy array) under the influence and malleability framework,
    with noise distributed as Beta(alpha, gamma)
    
    n (int): number of individuals
    d (int): number of influence dimensions
    alpha (float): shape parameter of Beta distribtion
    gamma (float): shape parameter of Beta distribtion
    '''
    X = np.random.rand(d,n)              # influence
    W = np.random.rand(d,n)              # malliability
    E = np.random.beta(alpha, gamma, size=(n,n))
    C = X.T.dot(W)+E
    return C

def weights_im_expo(n, d=1, lam=1.5):
    '''
    Returns weights C (numpy array) under the influence and malleability framework,
    with noise distributed as Expo(lam)
    
    n (int): number of individuals
    d (int): number of influence dimensions
    lam (float): the mean of the Exponential distribution
    '''
    X = np.random.rand(d,n)              # influence
    W = np.random.rand(d,n)              # malliability
    E = np.random.exponential(lam, size=(n,n))
    C = X.T.dot(W)+E
    return C

def weights_im_unif(n, d=1):
    '''
    Returns weights C (numpy array) under the influence and malleability framework,
    with noise distributed as Uniform(0,1)
    
    n (int): number of individuals
    d (int): number of influence dimensions
    '''
    X = np.random.rand(d,n)              # influence
    W = np.random.rand(d,n)              # malliability
    E = np.random.rand(n,n)
    C = X.T.dot(W)+E
    return C

def weights_discrete(n, diag=2, offdiag=1.5):
    '''
    Returns normalized weights from discrete set ~ binomial
    n (int): number of nodes
    diag  (float): controls magnitude of diagonal elements
    offdiag (float): controls magnitude of off-diagonal elements
    '''
    C = np.random.rand(n,n)
    C[C < 1/16] = -2
    C[C >= 15/16] = 2
    C[np.logical_and(C >= 1/16, C < 5/16)] = -1
    C[np.logical_and(C >= 5/16, C < 11/16)] = 0
    C[np.logical_and(C >= 11/16, C < 15/16)] = 1
    
    # remove diagonal elements and normalize off-diagonal elements
    np.fill_diagonal(C, 0)
    col_norms = np.linalg.norm(C, axis=0)
    C = (C / col_norms[None,:]) * offdiag * np.random.rand(n)[None,:]
    
    # diagonal elements 
    C_diag = np.ones(n) * diag * np.random.rand(n)
    
    # add back the diagonal
    C += np.diag(C_diag)
    
    return C

def weights_discrete_nonneg(n, diag=4, offdiag=3):
    '''
    Returns normalized non-negative weights from discrete set ~ binomial
    n (int): number of nodes
    diag  (float): controls magnitude of diagonal elements
    offdiag (float): controls magnitude of off-diagonal elements
    '''
    C = np.random.rand(n,n)
    C[C < 1/16] = 0
    C[C >= 15/16] = 4
    C[np.logical_and(C >= 1/16, C < 5/16)] = 1
    C[np.logical_and(C >= 5/16, C < 11/16)] = 1.5
    C[np.logical_and(C >= 11/16, C < 15/16)] = 2
    
    # remove diagonal elements and normalize off-diagonal elements
    np.fill_diagonal(C, 0)
    col_norms = np.linalg.norm(C, axis=0)
    C = (C / col_norms[None,:]) * offdiag * np.random.rand(n)[None,:]
    
    # diagonal elements 
    C_diag = np.ones(n) * diag * np.random.rand(n)
    
    # add back the diagonal
    C += np.diag(C_diag)
    
    return C

def weights_node_deg_expo(A, d=1, lam=1, prop=1):
    '''
    Returns weighted adjacency matrix C (numpy array) where weights depend on
    node-degree as Expo(node_deg) or Expo(1/node_deg) or Expo(0)
    
    A (square numpy array): adjacency matrix of your network
    d (int): influence/malleability dimension
    lam (float): rate of exponential distribution governing noise
    prop (int): governs the dependence of the weights on the node degrees
        prop = 0: both influence & malleability are inverse proportional to node deg
        prop = 1: both influence & malleability are directly proportional to node deg
        prop = 2: influence is inverse proportional to node deg, malleability directly prop
        prop = 3: influence is directly proportional to node deg, malleability inversely prop
    '''
    n = A.shape[0]
    out_deg = np.sum(A,axis=1) # array of the out-degree of each node
    in_deg = np.sum(A,axis=0)  # array of the in-degree of each node
    M_out = np.max(out_deg)    # max out-degree
    M_in = np.max(in_deg)      # max in-degree
    
    # ensures we don't pass "infinity" as the scale in the next step
    scaled_out_deg = M_out / out_deg
    scaled_in_deg = M_in / in_deg
    scaled_out_deg[scaled_out_deg > M_out] = 0
    scaled_in_deg[scaled_in_deg > M_in] = 0
    
    if prop == 0:
        X = np.random.exponential(scale=scaled_out_deg, size=(d,n))
        W = np.random.exponential(scale=scaled_in_deg, size=(d,n))
    elif prop == 1:
        X = np.random.exponential(scale=out_deg, size=(d,n))
        W = np.random.exponential(scale=in_deg, size=(d,n))
    elif prop == 2:
        X = np.random.exponential(scale=scaled_out_deg, size=(d,n))
        W = np.random.exponential(scale=in_deg, size=(d,n))
    else:
        X = np.random.exponential(scale=out_deg, size=(d,n))
        W = np.random.exponential(scale=scaled_in_deg, size=(d,n))
    
    E = np.random.exponential(lam, size=(n,n))
    
    return (X.T.dot(W)+E)

def weights_node_deg_rayleigh(A, d=1, sig=2, prop=1):
    '''
    Returns weighted adjacency matrix C (numpy array) where weights depend on
    node-degree as Rayleigh(node_deg) or Rayleigh(1/node_deg) or Rayleigh(0)
    
    A (square numpy array): adjacency matrix of your network
    d (int): influence/malleability dimension
    sig (float): scale of rayleigh distribution governing noise
    
    prop (int): governs the dependence of the weights on the node degrees
        prop = 0: both influence & malleability are inverse proportional to node deg
        prop = 1: both influence & malleability are directly proportional to node deg
        prop = 2: influence is inverse proportional to node deg, malleability directly prop
        prop = 3: influence is directly proportional to node deg, malleability inversely prop
    '''
    n = A.shape[0]
    out_deg = np.sum(A,axis=1) # array of the out-degree of each node
    in_deg = np.sum(A,axis=0) # array of the in-degree of each node
    
    # ensures we don't pass "infinity" as the scale in the next step
    scaled_out_deg = 1 / out_deg
    scaled_in_deg = 1 / in_deg
    scaled_out_deg[scaled_out_deg == np.inf] = 0
    scaled_in_deg[scaled_in_deg == np.inf] = 0
    
    if prop == 0:
        X = np.random.rayleigh(scale=(0.5*out_deg), size=(d,n))
        W = np.random.rayleigh(scale=(0.5*in_deg), size=(d,n))
    elif prop == 1:
        X = np.random.rayleigh(scale=scaled_out_deg, size=(d,n))
        W = np.random.rayleigh(scale=scaled_in_deg, size=(d,n))
    elif prop == 2: 
        X = np.random.rayleigh(scale=(0.5*out_deg), size=(d,n))
        W = np.random.rayleigh(scale=scaled_in_deg, size=(d,n))
    else:
        X = np.random.rayleigh(scale=scaled_out_deg, size=(d,n))
        W = np.random.rayleigh(scale=(0.5*in_deg), size=(d,n))
    
    E = np.random.rayleigh(sig, size=(n,n))
    
    return (X.T.dot(W)+E)

def weights_node_deg_unif(A, d=1, prop=1, vals=np.array([0,1])):
    '''
    Returns weighted adjacency matrix C (numpy array) where weights depend on
    node-degree as Uniform(0,node_deg) or Uniform(0,1/node_deg) or Uniform(0,0)
    
    A (square numpy array): adjacency matrix of your network
    d (int): influence/malleability dimension
    lam (float): rate of exponential distribution governing noise
    prop (int): governs the dependence of the weights on the node degrees
        prop = 0: both influence & malleability are inverse proportional to node deg
        prop = 1: both influence & malleability are directly proportional to node deg
        prop = 2: influence is inverse proportional to node deg, malleability directly prop
        prop = 3: influence is directly proportional to node deg, malleability inversely prop
    vals (numpy array): [low, high] => Noise ~ Uniform[low,high]
    '''
    n = A.shape[0]
    out_deg = np.sum(A,axis=1) # array of the out-degree of each node
    in_deg = np.sum(A,axis=0) # array of the in-degree of each node
    
    # ensures we don't pass "infinity" as the scale in the next step
    scaled_out_deg = 1 / out_deg
    scaled_in_deg = 1 / in_deg
    scaled_out_deg[scaled_out_deg == np.inf] = 0
    scaled_in_deg[scaled_in_deg == np.inf] = 0
    
    if prop == 0:
        X = np.random.uniform(low=0, high=scaled_out_deg, size=(d,n))
        W = np.random.uniform(low=0, high=scaled_in_deg, size=(d,n))
    elif prop == 1:
        X = np.random.uniform(low=0, high=out_deg, size=(d,n))
        W = np.random.uniform(low=0, high=in_deg, size=(d,n))
    elif prop == 2: 
        X = np.random.uniform(low=0, high=scaled_out_deg, size=(d,n))
        W = np.random.uniform(low=0, high=in_deg, size=(d,n))
    else:
        X = np.random.uniform(low=0, high=out_deg, size=(d,n))
        W = np.random.uniform(low=0, high=scaled_in_deg, size=(d,n))
    
    E = np.random.uniform(low=vals[0], high=vals[1], size=(n,n))
    
    return (X.T.dot(W)+E)

def normalized_weights(C, diag=10, offdiag=8):
  '''
  Returns normalized weights (or normalized weighted adjacency matrix) as numpy array

  C (square numpy array): weight matrix (or weighted adjacency matrix)
  diag (float): controls the magnitude of the diagonal elements
  offdiag (float): controls the magnitude of the off-diagonal elements
  '''
  n = C.shape[0]

  # diagonal elements
  C_diag = np.ones(n) * diag * np.random.rand(n)

  # remove diagonal elements and normalize off-diagonal elements
  np.fill_diagonal(C, 0)
  col_norms = np.linalg.norm(C, ord=1, axis=0)
  col_norms = np.where(col_norms != 0, col_norms, col_norms+1)
  C = (C / col_norms) * offdiag * np.random.rand(n)

  # add back the diagonal
  C += np.diag(C_diag)

  return C

def printWeights(C,alpha,filename):
  f = open(filename, 'w')
  n = C.shape[0]
  print("baseline values", file=f)
  print("n: "+str(n), file=f)
  for i in range(n):
    print(str(alpha[i]), file=f)
  nnz = np.count_nonzero(C)
  print("treatment effect weights", file=f)
  print("edges: "+str(nnz), file=f)
  (ind1,ind2) = np.nonzero(C)
  for i in range(nnz):
    a = ind1[i]
    b = ind2[i]
    print(str(a)+"\t"+str(b)+"\t"+str(C[a,b]), file=f)
  f.close()

def loadWeights(filename,n):
    f = open(filename, 'r')
    next(f)
    next(f)
    alpha = np.zeros(n)
    for i in range(n):
        line = next(f)
        line.strip()
        alpha[i] = float(line)
    next(f)
    next(f)
    C = np.zeros((n,n))
    for line in f:
        line = line.strip()
        ind = line.split()
        C[int(ind[0]),int(ind[1])] = float(ind[2])
    return (C,alpha)

# Potential Outcomes Models

linear_pom = lambda C,alpha, z : C.dot(z) + alpha

# Treatment Assignment

bernoulli = lambda n,p : (np.random.rand(n) < p) + 0

def cluster_randomization(clusters,p):
    '''
    TODO
    '''
    return clusters.dot(np.random.rand(clusters.shape[1]) < p)

def threenet(A):
    '''
    TODO
    '''
    A_sq  = 1*(A.dot(A) > 0)
    A_cubed = 1*(A.dot(A_sq) > 0)
    vert = np.arange(A.shape[0])
    center = []
    while vert.size > 0:
        ind = np.random.randint(vert.size)
        center.append(ind)
        neighb = np.flatnonzero(A_cubed[ind,:])
        np.delete(vert, neighb)
    clusters = np.zeros(A.shape[0], len(center))
    vert = np.arange(A.shape[0])
    for i in range(len(center)):
        ind = center[i]
        clusters[ind,i] = 1
        neighb = np.flatnonzero(A[ind,:])
        add_vert = np.intersect1d(vert, neighb, assume_unique=True)
        for j in add_vert:
            clusters[j,i] = 1
        np.delete(vert, neighb)
    for v in vert:
        neighb_2 = np.flatnonzero(A_sq[v,:])
        cent, comm1, comm2 = np.intersect1d(center, neighb_2, assume_unique=True, return_indices=True)
        if len(cent) > 0:
            clusters[v,comm1[0]] = 1
        else:
            neighb_3 = np.flatnonzero(A_cubed[v,:])
            cent, comm1, comm2 = np.intersect1d(center, neighb_3, assume_unique=True, return_indices=True)
            clusters[v,comm1[0]] = 1
    s=np.sum(clusters,axis=1)
    print(np.amax(s))
    print(np.amin(s))
    return clusters

# Estimators
# n = number of individuals
# p = treatment probability
# fy = potential outcomes model
# A = network adjacency matrix
# z = treatment vector

def est_us(n, p, y, A, z): 
    '''
    Returns an estimate of the TTE using our proposed estimator
    
    n (int): number of individuals
    p (float): treatment probability
    fy (function): potential outcomes model
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    '''
    zz = z/p - (1-z)/(1-p)
    return 1/n * y.dot(A.dot(zz))

def est_us_clusters(n, p, y, A, z, clusters=np.array([])): 
    '''
    TODO
    
    n (int): number of individuals
    p (float): treatment probability
    y (TODO): TODO
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    cluster (numpy array): TODO
    '''
    if clusters.size == 0:
        z_c = z
        A_c = A
    else:
        z_c = 1*(np.sum(np.multiply(z.reshape((n,1)),clusters),axis=0)>0)
        A_c = 1*(A.dot(clusters) >0)
    zz = z_c/p - (1-z_c)/(1-p)
    return 1/n * y.dot(A_c.dot(zz))

def est_ols(n, p, y, A, z): 
    '''
    Returns an estimate of the TTE using OLS (TODO: regresses  over proportion?)
    Uses numpy.linalg.solve and normal equations
    
    n (int): number of individuals
    p (float): treatment probability
    y (numpy array): observed outcomes
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    '''
    M = np.ones((n,3))
    M[:,1] = z
    M[:,2] = A.dot(z) / (A.dot(np.ones(n))+1e-10)
    
    v = np.linalg.solve(M.T.dot(M),M.T.dot(y))
    return v[1]+v[2]

def est_ols_gen(n, p, y, A, z): 
    '''
    Returns an estimate of the TTE using OLS (TODO: regresses  over proportion?)
    Uses numpy.linalg.lstsq without the use of the normal equations
    
    n (int): number of individuals
    p (float): treatment probability
    y (numpy array): observed outcomes
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    '''
    M = np.ones((n,3))
    M[:,1] = z
    M[:,2] = A.dot(z) / (A.dot(np.ones(n))+1e-10)
    
    v = np.linalg.lstsq(M,y)[0]
    return v[0]+v[1]

def diff_in_means_naive(n, p, y, A, z):
    '''
    TODO: Returns an estimate of the TTE using standard (naive, simple?) 
    difference in means on... 
    
    n (int): number of individuals
    p (float): treatment probability
    y (numpy array): observed outcomes
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    '''
    return y.dot(z)/np.sum(z) - y.dot(1-z)/np.sum(1-z)

def diff_in_means_fraction(n, p, y, A, z, lam):
    '''
    TODO: Returns an estimate of the TTE using standard (naive, simple?) 
    difference in means on...
    
    n (int): number of individuals
    p (float): treatment probability
    y (numpy array): observed outcomes
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    lam (float): TODO
    '''
    treated = 1*(A.dot(z) / (A.dot(np.ones(n))+1e-10) > lam)
    control = 1*(A.dot(1-z) / (A.dot(np.ones(n))+1e-10) > lam)
    return y.dot(treated)/np.sum(treated) - y.dot(control)/np.sum(control) 

#Horvitz-Thompson 
def est_ht(n, p, y, A, z, clusters=np.array([])):
  if clusters.size == 0:
    zz = np.prod(np.tile(z/p,(n,1)),axis=1, where=A==1) - np.prod(np.tile((1-z)/(1-p),(n,1)),axis=1, where=A==1)
  else:
    deg = np.sum(clusters,axis=1)
    wt_T = np.power(p,deg)
    wt_C = np.power(1-p,deg)
    zz = np.multiply(np.prod(A*z,axis=1),wt_T) - np.multiply(np.prod(A*(1-z),axis=1),wt_C)
  return 1/n * y.dot(zz)

#Hajek
def est_hajek(n, p, y, A, z, clusters=np.array([])): 
  if clusters.size == 0:
    zz_T = np.prod(np.tile(z/p,(n,1)), axis=1, where=A==1)
    zz_C = np.prod(np.tile((1-z)/(1-p),(n,1)), axis=1, where=A==1)
  else:
    deg = np.sum(clusters,axis=1)
    wt_T = np.power(p,deg)
    wt_C = np.power(1-p,deg)
    zz_T = np.multiply(np.prod(A*z,axis=1),wt_T) 
    zz_C = np.multiply(np.prod(A*(1-z),axis=1),wt_C)
  all_ones = np.ones(n)
  est_T = 0
  est_C=0
  if all_ones.dot(zz_T) > 0:
    est_T = y.dot(zz_T) / all_ones.dot(zz_T)
  if all_ones.dot(zz_C) > 0:
    est_C = y.dot(zz_C) / all_ones.dot(zz_C)
  return est_T - est_C