import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from math import log, ceil
import pandas as pd
import seaborn as sns
import nci_linear_setup as ncls
from scipy import interpolate

# Scale down the effects of higher order terms
a1 = 1      # for linear effects
a2 = 1/2    # for quadratic effects
a3 = 1/4   # for cubic effects

# Define f(z)
f_linear = lambda alpha, z, gz: alpha + a1*z + 0*gz
f_quadratic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz)
f_cubic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz) + a3*np.power(gz,3)

def ppom(f, C, alpha):
  '''
  Returns k-degree polynomial potential outcomes function fy
  
  f (function): must be of the form f(z) = alpha + z + a2*z^2 + a3*z^3 + ... + ak*z^k
  C (np.array): weighted adjacency matrix
  alpha (np.array): vector of null effects
  '''
  n = C.shape[0]
  assert np.all(f(alpha, np.zeros(n), np.zeros(n)) == alpha), 'f(0) should equal alpha'
  #assert np.all(np.around(f(alpha, np.ones(n)) - alpha - np.ones(n), 10) >= 0), 'f must include linear component'
  g = lambda z : C.dot(z)/np.sum(C,1)
  return lambda z: f(alpha, C.dot(z), g(z)) 

def staggered_rollout_bern(beta, p, n, P=np.array([])):
  '''
  Returns Treatment Samples and Coefficients h_t from Bernoulli Staggered Rollout

  beta (int): degree of potential outcomes model
  p (float): treatment budget e.g. if you can treat 5% of population, p = 0.05
  n (int): size of population
  P (numpy array): treatment probabilities for each time step
  '''
  ### Compute P = [p0, p1, ... , pbeta] via pi = (i+1)p/(beta+1) ###
  if P.size == 0:
    fun = lambda i: (i)*p/(beta)
    P = np.fromfunction(fun, shape=(beta+1,))

  ### Initialize ###
  H = np.zeros(beta+1)   # for the coefficients h_t
  Z = np.zeros(shape=(beta+1,n))   # for each treatment sample z_t
  U = np.random.rand(n)

  ### staggered rollout experiment ###
  for t in range(beta+1):
    ## sample treatment vector ##
    Z[t,:] = (U < P[t])+0

    ## computing h_t ##
    one_minusP = 1 - P            # [1-p0, 1-p1, ... , 1-p_beta]
    pt_minusP = P[t] - P          # [pt-p0, pt-p1, ... , pt-p_beta]
    minusP = -1*P                 # [-p0, -p1, ... , -p_beta]
    one_minusP[t] = 1; pt_minusP[t] = 1; minusP[t] = 1
    fraction1 = one_minusP/pt_minusP
    fraction2 = minusP/pt_minusP
    H[t] = np.prod(fraction1) - np.prod(fraction2)

  return Z, H, P

def staggered_rollout_complete(beta, p, n, K=np.array([])):
  '''
  Returns Treatment Samples and Coefficients l_t from Complete Staggered Rollout

  beta (int): degree of potential outcomes model
  p (float): treatment budget e.g. if you can treat 5% of population, p = 0.05
  n (int): size of population
  P (numpy array): treatment probabilities for each time step
  '''
  ### Compute P = [p0, p1, ... , pbeta] via pi = (i+1)p/(beta+1) ###
  if K.size == 0:
    #TODO
    pass

  ### Initialize ###
  L = np.zeros(beta+1)   # for the coefficients L_t
  Z = np.zeros(shape=(beta+1,n))   # for each treatment sample, z_t
  z0 = np.zeros(n)       # initial treatment vector

  ### staggered rollout experiment ###
  for t in range(beta+1):
    """
    ## sample treatment vector ##
    # TODO: should do something about replacement?
    zt = ncls.completeRD(n,p)

    ## if treated before, should still be treated ##
    if t > 0:
      Z[t,:] = np.where(Z[t-1,:] > 0, Z[t-1,:], zt)
    else:
      Z[t,:] = zt
    """

    ## computing _t ##
    n_minusK = n - K            # [n-k0, n-k1, ... , n-k_beta]
    kt_minusK = K[t] - K        # [kt-k0, kt-k1, ... , kt-k_beta]
    minusK = -1*K               # [-k0, -k1, ... , -k_beta]
    n_minusK[t] = 1; kt_minusK[t] = 1; minusK[t] = 1
    fraction1 = n_minusK/kt_minusK
    fraction2 = minusK/kt_minusK
    L[t] = np.prod(fraction1) - np.prod(fraction2)

  return Z, L

def outcome_sums(beta, Y, Z):
  '''
  Returns the sums of the outcomes Y(z_t) for each timestep t

  beta (int): degree of potential outcomes model
  Y (function): potential outcomes model
  Z (numpy array): treatment vectors z_t for each timestep t
   - each row should correspond to a timestep, i.e. Z should be beta+1 by n
  '''
  sums = np.zeros(beta+1) 
  for t in range(beta+1):
    sums[t] = np.sum(Y(Z[t,:]))
  return sums

def graph_agnostic(n, sums, H):
    '''
    Returns an estimate of the TTE with (beta+1) staggered rollout design

    n (int): popluation size
    H (numpy array): PPOM coefficients h_t or l_t
    sums (numpy array): sums of outcomes at each time step
    '''
    return (1/n)*H.dot(sums)

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
  Returns two estimates of TTE using linear polynomial interpolation 
  via scipy.interpolate.interp1d
  - the first is with kind = 'linear' (as in... ?)
  - the second is with kind = 'slinear' (as in linear spline)

  n (int): popluation size
  P (numpy array): sequence of probabilities p_t
  sums (numpy array): sums of outcomes at each time step
  '''

  f_lin = interpolate.interp1d(P, sums, fill_value='extrapolate')
  f_spl = interpolate.interp1d(P, sums, kind='slinear', fill_value='extrapolate')
  TTE_hat1 = (1/n)*(f_lin(1) - f_lin(0))
  TTE_hat2 = (1/n)*(f_spl(1) - f_spl(0))
  return TTE_hat1, TTE_hat2


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

  X = np.ones((n,2*beta+1))
  count = 1
  treated_neighb = (A.dot(z)-z)/((np.sum(A,1)-1)+1e-10)
  for i in range(beta):
      X[:,count] = np.multiply(z,np.power(treated_neighb,i))
      X[:,count+1] = np.power(treated_neighb,i+1)
      count += 2

  v = np.linalg.lstsq(X,y,rcond=None)[0]
  return np.sum(v)-v[0]

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
  treated_neighb = np.sum(A,1)-1
  for i in range(beta):
      X[:,count] = np.multiply(z,np.power(treated_neighb,i))
      X[:,count+1] = np.power(treated_neighb,i+1)
      count += 2
  TTE_hat = np.sum((X @ v) - v[0])/n
  
  return TTE_hat

