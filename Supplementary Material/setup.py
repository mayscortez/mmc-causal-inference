import numpy as np
import random
import networkx as nx
import scipy.sparse
from scipy import interpolate

########################################
# Functions to generate random networks
########################################

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

def config_model_nx(N, exp = 2.5, law = "out"):
    '''
    Returns the adjacency matrix A (as a numpy array) of a networkx configuration
    model with power law degree sequences

    N (int): number of nodes
    law (str): inicates whether in-, out- or both in- and out-degrees should be distributed as a power law
        "out" : out-degrees distributed as powerlaw, in-degrees sum up to same # as out-degrees
        "in" : in-degrees distributed as powerlaw, out-degrees sum up to same # as in-degrees
        "both" : both in- and out-degrees distributed as powerlaw
    '''
    assert law in ["out", "in", "both"], "law must = 'out', 'in', or 'both'"
    powerlaw_out, powerlaw_in = powerlaw_degrees(N, exp)
    if law == "out":
        deg_seq_out  = powerlaw_out
        deg_seq_in = uniform_degrees(N,np.sum(deg_seq_out))
    elif law == "in":
        deg_seq_in = powerlaw_in
        deg_seq_out = uniform_degrees(N,np.sum(deg_seq_in))
    else:
        deg_seq_out = powerlaw_out
        deg_seq_in = powerlaw_in

    G = nx.generators.degree_seq.directed_configuration_model(deg_seq_in,deg_seq_out)

    G.remove_edges_from(nx.selfloop_edges(G)) # remove self-loops
    G = nx.DiGraph(G)                         # remove parallel edges
    A = nx.to_scipy_sparse_matrix(G)          # retrieve adjacency matrix
    A.setdiag(np.ones(N))                     # everyone is affected by their own treatment

    return A

def powerlaw_degrees(N, exp=2.5):
    '''
    Returns out- and in-degree sequences distributed according to a powerlaw with exp
    The two sequences sum up to the same number
    See networkx utils.powerlaw_sequence for more details

    N (int): : number of nodes in graph
    exp (float): exponent in powerlaw distribution pdf
    '''
    S_out = np.around(nx.utils.powerlaw_sequence(N, exponent=exp), decimals=0).astype(int)
    out_sum = np.sum(S_out)
    if (out_sum % 2 != 0):
        ind = np.random.randint(N)
        S_out[ind] += 1
    
    S_in = np.around(nx.utils.powerlaw_sequence(N, exponent=exp), decimals=0).astype(int)
    while (np.sum(S_in) != out_sum):
        ind = np.random.randint(N)
        if (np.sum(S_in) > out_sum):
            S_in[ind] -= 1
        else:
            S_in[ind] += 1

    
    return S_out, S_in

def uniform_degrees(n,sum):
    '''
    Given n and sum, returns array whose entries add up to sum where each entry is in {sum/n, (sum,n)+1}
    i.e. to create uniform degrees for a network that add up to a specific number

    n: size of network
    sum: number that the entries of the array must add up to
    '''
    degs = (np.ones(n)*np.floor(sum/n)).astype(int)
    i = 0
    while np.sum(degs) != sum:
        degs[i] += 1
        i += 1
    return degs

def symmetrizeGraph(A):
    n = A.shape[0]
    if A.shape[1] != n:
        print("Error: adjacency matrix is not square!")
        return A
    for i in range(n):
        for j in range(i):
            A[i,j] = A[j,i]
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
    Returns adjacency matrix A (numpy array) of a stochastic block matrix where

    # TODO:
    clusterSize
    probabilities

    **ASSUME SYMMETRIC PROBABILITY MATRIX**
    '''
    p = np.kron(probabilities, np.ones((clusterSize,clusterSize)))
    n = p.shape[0]
    A = np.random.rand(n,n)
    A = (A < p) + 0
    A[range(n),range(n)] = 1   # everyone is affected by their own treatment
    return A

########################################
# Functions to generate network weights
########################################

def simpleWeights(A, diag=5, offdiag=5, rand_diag=np.array([]), rand_offdiag=np.array([])):
    '''
    Returns weights generated from model described in Experiments Section

    A (numpy array): adjacency matrix of the network
    diag (float): maximum norm of direct effects
    offidiag (float): maximum norm of the indirect effects
    rand_diag (numpy array):
    rand_offdiag (numpy arry):
    '''
    n = A.shape[0]

    if rand_offdiag.size == 0:
        rand_offdiag = np.random.rand(n)
    C_offdiag = offdiag*rand_offdiag

    in_deg = scipy.sparse.diags(np.array(A.sum(axis=1)).flatten(),0)  # array of the in-degree of each node
    C = in_deg.dot(A - scipy.sparse.eye(n))
    col_sum = np.array(C.sum(axis=0)).flatten()
    col_sum[col_sum==0] = 1
    temp = scipy.sparse.diags(C_offdiag/col_sum)
    C = C.dot(temp)

    # out_deg = np.array(A.sum(axis=0)).flatten() # array of the out-degree of each node
    # out_deg[out_deg==0] = 1
    # temp = scipy.sparse.diags(C_offdiag/out_deg)
    # C = A.dot(temp)

    if rand_diag.size == 0:
        rand_diag = np.random.rand(n)
    C_diag = diag*rand_diag
    C.setdiag(C_diag)

    return C

########################################
# Potential Outcomes Models
########################################
linear_pom = lambda C,alpha, z : C.dot(z) + alpha

# Scale the effects of higher order terms
a1 = 1      # for linear effects
a2 = 1    # for quadratic effects
a3 = 1   # for cubic effects
a4 = 1   # for quartic effects

# Define f(z)
f_linear = lambda alpha, z, gz: alpha + a1*z # should be equivalent to linear_pom
f_quadratic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz)
f_cubic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz) + a3*np.power(gz,3)
f_quartic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz) + a3*np.power(gz,3) + a4*np.power(gz,4)

def ppom(beta, C, alpha):
  '''
  Returns k-degree polynomial potential outcomes (POM) function fy
  
  beta (int): degree of POM 
  C (np.array): weighted adjacency matrix
  alpha (np.array): vector of null effects
  '''
  g = lambda z : C.dot(z) / np.array(np.sum(C,1)).flatten()

  if beta == 0:
      return lambda z: alpha + a1*z
  elif beta == 1:
      f = f_linear
  elif beta == 2:
      f = f_quadratic
  elif beta == 3:
      f = f_cubic
  elif beta == 4:
      f = f_quadratic
  else:
      print("ERROR: invalid degree")

  return lambda z: f(alpha, C.dot(z), g(z)) 

#####################################################
# Treatment Assignment Mechanisms (Randomized Design)
#####################################################

bernoulli = lambda n,p : (np.random.rand(n) < p) + 0

def completeRD(n,treat):
    '''
    Returns a treatment vector using complete randomized design

    n (int): number of individuals
    p (float): fraction of individuals you want to be assigned to treatment
    '''
    z = np.zeros(shape=(n,))
    z[0:treat] = np.ones(shape=(treat))
    rng = np.random.default_rng()
    rng.shuffle(z)
    return z

def staggered_rollout_bern(n, P):
  '''
  Returns Treatment Samples from Bernoulli Staggered Rollout

  beta (int): degree of potential outcomes model
  n (int): size of population
  P (numpy array): treatment probabilities for each time step
  '''

  ### Initialize ###
  Z = np.zeros(shape=(P.size,n))   # for each treatment sample z_t
  U = np.random.rand(n)

  ### staggered rollout experiment ###
  for t in range(P.size):
    ## sample treatment vector ##
    Z[t,:] = (U < P[t])+0

  return Z

def staggered_rollout_complete(n, K):
  '''
  Returns Treatment Samples Z from Complete Staggered Rollout and number of people treated by each time step K

  beta (int): degree of potential outcomes model
  n (int): size of population
  K (numpy array): total number of individuals treated by each timestep
  '''

  ### Initialize ###
  Z = np.zeros(shape=(K.size,n))   # for each treatment sample, z_t
  indices = np.random.permutation(np.arange(n))           # random permutation of the individuals

  ### staggered rollout experiment ###
  # indices: holds indices of entries equal to 0 in treatment vector
  # to_treat: from the next set of indiv in the random permutation
  for t in range(K.size-1):
    to_treat = indices[K[t]:K[t+1]+1]
    Z[t+1:,to_treat] = 1 

  return Z

def outcome_sums(Y, Z):
  '''
  Returns the sums of the outcomes Y(z_t) for each timestep t

  Y (function): potential outcomes model
  Z (numpy array): treatment vectors z_t for each timestep t
   - each row should correspond to a timestep, i.e. Z should be beta+1 by n
  '''
  sums = np.zeros(Z.shape[0]) 
  for t in range(Z.shape[0]):
    sums[t] = np.sum(Y(Z[t,:]))
  return sums

################################################
# Setup for graph agnostic w/ staggered rollout
################################################
def bern_coeffs(P):
  '''
  Returns Coefficients h_t from Bernoulli Staggered Rollout

  P (numpy array): treatment probabilities for each time step
  '''

  ### Initialize ###
  H = np.zeros(P.size)

  ### Coefficients ###
  for t in range(P.size):
    one_minusP = 1 - P            # [1-p0, 1-p1, ... , 1-p_beta]
    pt_minusP = P[t] - P          # [pt-p0, pt-p1, ... , pt-p_beta]
    minusP = -1*P                 # [-p0, -p1, ... , -p_beta]
    one_minusP[t] = 1; pt_minusP[t] = 1; minusP[t] = 1
    fraction1 = one_minusP/pt_minusP
    fraction2 = minusP/pt_minusP
    H[t] = np.prod(fraction1) - np.prod(fraction2)

  return H

def seq_treatment_probs(M, p):
  '''
  Returns sequence of treatment probabilities for Bernoulli staggered rollout

  M (int): fineness of measurements in staggered rollout (# timesteps - 1, not counting the time zero)
  p (float): treatment budget e.g. if you can treat 5% of population, p = 0.05
  '''
  fun = lambda i: (i)*p/(M)
  P = np.fromfunction(fun, shape=(M+1,))
  return P

def complete_coeffs(n, K):
  '''
  Returns coefficients l_t from Complete Staggered Rollout

  n (int): size of population
  K (numpy array): total number of individuals treated by each timestep
  '''

  ### Initialize ###
  L = np.zeros(K.size)             # for the coefficients L_t

  for t in range(K.size):
    n_minusK = n - K            # [n-k0, n-k1, ... , n-k_beta]
    kt_minusK = K[t] - K        # [kt-k0, kt-k1, ... , kt-k_beta]
    minusK = -1*K               # [-k0, -k1, ... , -k_beta]
    n_minusK[t] = 1; kt_minusK[t] = 1; minusK[t] = 1
    fraction1 = n_minusK/kt_minusK
    fraction2 = minusK/kt_minusK
    L[t] = np.prod(fraction1) - np.prod(fraction2)

  return L

def seq_treated(M, p, n, K=np.array([])):
  '''
  Returns number of people treated by each time step with K = [k0, k1, ... , kM] via ki = i*n*p/M
  
  M (int): fineness of measurements in staggered rollout (# timesteps - 1, not counting the time zero)
  p (float): treatment budget e.g. if you can treat 5% of population, p = 0.05
  n (int): size of population
  '''
  if K.size == 0:
    fun = lambda i: np.floor(p*n*i/M).astype(int)
    K = np.fromfunction(fun, shape=(M+1,))
  return K

########################################
# Estimators
########################################
def graph_agnostic(n, sums, H):
    '''
    Returns an estimate of the TTE with (beta+1) staggered rollout design

    n (int): popluation size
    H (numpy array): PPOM coefficients h_t or l_t
    sums (numpy array): sums of outcomes at each time step
    '''
    return (1/n)*H.dot(sums)

###### Least Squares Regression ######
def est_ols_gen(y, A, z):
    '''
    Returns an estimate of the TTE using OLS (regresses over proportion of neighbors treated)
    Uses numpy.linalg.lstsq without the use of the normal equations

    y (numpy array): observed outcomes
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    '''
    n = A.shape[0]
    X = np.ones((n,3))
    X[:,1] = z
    X[:,2] = (A.dot(z) - z) / (np.array(A.sum(axis=1)).flatten()-1+1e-10)

    v = np.linalg.lstsq(X,y,rcond=None)[0] # solve for v in y = Xv
    return v[1]+v[2]

def poly_regression_prop(beta, y, A, z):
  '''
  Returns an estimate of the TTE using polynomial regression using
  numpy.linalg.lstsq

  beta (int): degree of polynomial
  y (numpy array): observed outcomes
  A (square numpy array): network adjacency matrix
  z (numpy array): treatment vector
  '''
  n = A.shape[0]

  if beta == 0:
      X = np.ones((n,2))
      X[:,1] = z
  else:
      X = np.ones((n,2*beta+1))
      count = 1
      treated_neighb = (A.dot(z)-z)/(np.array(A.sum(axis=1)).flatten()-1+1e-10)
      for i in range(beta):
          X[:,count] = np.multiply(z,np.power(treated_neighb,i))
          X[:,count+1] = np.power(treated_neighb,i+1)
          count += 2

  v = np.linalg.lstsq(X,y,rcond=None)[0]
  return np.sum(v)-v[0]

def est_ols_treated(y, A, z):
    '''
    Returns an estimate of the TTE using OLS (regresses over number neighbors treated)
    Uses numpy.linalg.lstsq without the use of the normal equations

    y (numpy array): observed outcomes
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    '''
    n = A.shape[0]
    X = np.ones((n,3))
    X[:,1] = z
    X[:,2] = A.dot(z) - z

    v = np.linalg.lstsq(X,y,rcond=None)[0] # solve for v in y = Xv
    return v[1]+(v[2]*(np.sum(A)-n)/n)

def poly_regression_num(beta, y, A, z):
  '''
  Returns an estimate of the TTE using polynomial regression using
  numpy.linalg.lstsq

  beta (int): degree of polynomial
  y (numpy array): observed outcomes
  A (square numpy array): network adjacency matrix
  z (numpy array): treatment vector
  '''
  n = A.shape[0]

  if beta == 0:
      X = np.ones((n,2))
      X[:,1] = z
  else:
      X = np.ones((n,2*beta+1))
      count = 1
      treated_neighb = (A.dot(z)-z)
      for i in range(beta):
          X[:,count] = np.multiply(z,np.power(treated_neighb,i))
          X[:,count+1] = np.power(treated_neighb,i+1)
          count += 2

  # least squares regression
  v = np.linalg.lstsq(X,y,rcond=None)[0]

  # Estimate TTE
  count = 1
  treated_neighb = np.array(A.sum(axis=1)).flatten()-1
  for i in range(beta):
      X[:,count] = np.power(treated_neighb,i)
      X[:,count+1] = np.power(treated_neighb,i+1)
      count += 2
  TTE_hat = np.sum((X @ v) - v[0])/n
  return TTE_hat

###### Difference in Means ######
def diff_in_means_naive(y, z):
    '''
    Returns an estimate of the TTE using difference in means
    (mean outcome of individuals in treatment) - (mean outcome of individuals in control)

    y (numpy array): observed outcomes
    z (numpy array): treatment vector
    '''
    return y.dot(z)/np.sum(z) - y.dot(1-z)/np.sum(1-z)

def diff_in_means_fraction(n, y, A, z, tol):
    '''
    Returns an estimate of the TTE using weighted difference in means where 
    we only count neighborhoods with at least tol fraction of the neighborhood being
    assigned to treatment or control

    n (int): number of individuals
    y (numpy array): observed outcomes
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    tol (float): neighborhood fraction treatment/control "threshhold"
    '''
    z = np.reshape(z,(n,1))
    treated = 1*(A.dot(z)-1 >= tol*(A.dot(np.ones((n,1)))-1))
    treated = np.multiply(treated,z).flatten()
    control = 1*(A.dot(1-z)-1 >= tol*(A.dot(np.ones((n,1)))-1))
    control = np.multiply(control,1-z).flatten()

    est = 0
    if np.sum(treated) > 0:
        est = est + y.dot(treated)/np.sum(treated)
    if np.sum(control) > 0:
        est = est - y.dot(control)/np.sum(control)
    return est

###### Spline Interpolation ######
def poly_interp_splines(n, P, sums, spltyp = 'quadratic'):
  '''
  Returns estimate of TTE using spline polynomial interpolation 
  via scipy.interpolate.interp1d

  n (int): popluation size
  P (numpy array): sequence of probabilities p_t
  sums (numpy array): sums of outcomes at each time step
  spltyp (str): type of spline, can be 'quadratic, or 'cubic'
  '''
  assert spltyp in ['quadratic', 'cubic'], "spltyp must be 'quadratic', or 'cubic'"
  f_spl = interpolate.interp1d(P, sums, kind=spltyp, fill_value='extrapolate')
  TTE_hat = (1/n)*(f_spl(1) - f_spl(0))
  return TTE_hat

def poly_interp_linear(n, P, sums):
  '''
  Returns estimate of TTE using linear spline interpolation 
  via scipy.interpolate.interp1d

  n (int): popluation size
  P (numpy array): sequence of probabilities p_t
  sums (numpy array): sums of outcomes at each time step
  '''

  f_spl = interpolate.interp1d(P, sums, kind='slinear', fill_value='extrapolate')
  TTE_hat2 = (1/n)*(f_spl(1) - f_spl(0))
  return TTE_hat2

########################################
# To save graphs to be C++ compatible
########################################
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
        line = line.strip()
        alpha[i] = float(line)
    next(f)
    next(f)
    C = np.zeros((n,n))
    for line in f:
        line = line.strip()
        ind = line.split()
        C[int(ind[0]),int(ind[1])] = float(ind[2])
    return (C,alpha)
