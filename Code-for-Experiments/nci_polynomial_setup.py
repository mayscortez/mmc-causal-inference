from dataclasses import replace
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from math import log, ceil
import pandas as pd
import seaborn as sns
import nci_linear_setup as ncls
from scipy import interpolate, special

# Scale down the effects of higher order terms
a1 = 1      # for linear effects
a2 = 1    # for quadratic effects
a3 = 1   # for cubic effects
a4 = 1   # for quartic effects

# Define f(z)
f_linear = lambda alpha, z, gz: alpha + a1*z
f_quadratic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz)
f_cubic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz) + a3*np.power(gz,3)
f_quartic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz) + a3*np.power(gz,3) + a4*np.power(gz,4)

def ppom(beta, C, alpha):
  '''
  Returns k-degree polynomial potential outcomes function fy
  
  f (function): must be of the form f(z) = alpha + z + a2*z^2 + a3*z^3 + ... + ak*z^k
  C (np.array): weighted adjacency matrix
  alpha (np.array): vector of null effects
  '''
  # n = C.shape[0]
  # assert np.all(f(alpha, np.zeros(n), np.zeros(n)) == alpha), 'f(0) should equal alpha'
  #assert np.all(np.around(f(alpha, np.ones(n)) - alpha - np.ones(n), 10) >= 0), 'f must include linear component'

  if beta == 0:
      return lambda z: alpha + a1*z
  elif beta == 1:
      f = f_linear
      return lambda z: alpha + a1*C.dot(z)
  else:
      g = lambda z : C.dot(z) / np.array(np.sum(C,1)).flatten()
      if beta == 2:
          f = f_quadratic
      elif beta == 3:
          f = f_cubic
      elif beta == 4:
          f = f_quartic
      else:
          print("ERROR: invalid degree")
      return lambda z: f(alpha, C.dot(z), g(z)) 

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

  #f_lin = interpolate.interp1d(P, sums, fill_value='extrapolate')
  f_spl = interpolate.interp1d(P, sums, kind='slinear', fill_value='extrapolate')
  #TTE_hat1 = (1/n)*(f_lin(1) - f_lin(0))
  TTE_hat2 = (1/n)*(f_spl(1) - f_spl(0))
  #return TTE_hat1, TTE_hat2
  return TTE_hat2


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

def poly_regression_prop_cy(beta, y, A, z):
  n = A.shape[0]
  X = np.ones((n,2*beta+2))
  z = z.reshape((n,1))
  treated_neighb = (A.dot(z)-z)/(np.array(A.sum(axis=1)).flatten()-1+1e-10)
  # temp = 1
  # for i in range(beta+1):
  #     X[:,i] = np.multiply(z,temp)
  #     X[:,beta+1+i] = np.multiply(1-z,temp)
  #     temp = temp * treated_neighb
  treated_neighb = np.power(treated_neighb.reshape((n,1)), np.arange(beta+1).reshape((1,beta+1)))
  X[:,:beta+1] = z.dot(treated_neighb)
  X[:,beta+1:] = (1-z).dot(treated_neighb)

  v = np.linalg.lstsq(X,y,rcond=None)[0]
  return np.sum(v[:beta+1])-v[beta+1]

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

def poly_regression_num_cy(beta, y, A, z):
  n = A.shape[0]

  X = np.ones((n,2*beta+2))
  z = z.reshape((n,1))
  treated_neighb = (A.dot(z)-z)
  # temp = 1
  # for i in range(beta+1):
  #     X[:,i] = np.multiply(z,temp)
  #     X[:,beta+1+i] = np.multiply(1-z,temp)
  #     temp = temp * treated_neighb
  treated_neighb = np.power(treated_neighb.reshape((n,1)), np.arange(beta+1).reshape((1,beta+1)))
  X[:,:beta+1] = z.dot(treated_neighb)
  X[:,beta+1:] = (1-z).dot(treated_neighb)

  # least squares regression
  v = np.linalg.lstsq(X,y,rcond=None)[0]

  # Estimate TTE
  X = np.zeros((n,2*beta+2))
  deg = np.array(A.sum(axis=1)).flatten()-1
  # temp = 1
  # for i in range(beta+1):
  #     X[:,i] = temp
  #     temp = temp * deg
  X[:,:beta+1] = np.power(deg.reshape((n,1)), np.arange(beta+1).reshape((1,beta+1)))
  TTE_hat = np.sum((X @ v) - v[beta+1])/n

  
  return TTE_hat

def graph_aware_estimator(n, p, y, A, z, beta):
  # n = z.size
  # z = z.reshape((n,1))
  treated_neighb = A.dot(z)
  control_neighb = A.dot(1-z)
  est = 0
  for i in range(n):
    w = 0
    a_lim = min(beta,int(treated_neighb[i]))
    for a in range(a_lim+1):
      b_lim = min(beta - a,int(control_neighb[i]))
      for b in range(b_lim+1):
        w = w + ((1-p)**(a+b) - (-p)**(a+b)) * p**(-a) * (p-1)**(-b) * special.binom(treated_neighb[i],a)  * special.binom(control_neighb[i],b)
    est = est + y[i]*w

  return est/n
