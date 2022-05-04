from nci_lin_setup import *
import pandas as pd

diagmax = 10   # maximum norm of direct effect
r = 2
offdiagmax = r*diagmax   # maximum norm of indirect effect

p = 0.5    # treatment probability
T = 1000   # number of trials

sizes = np.array([16,24,32,48,64,96])#
results = []

for s in sizes:
  n = s*s
  print('n '+str(n))
  A = loadGraph('graphs/SW'+str(s)+'.txt', n, symmetric=True)
  graph = "sw"

  # baseline parameters
  alpha = np.zeros(n)

  # Generate (normalized) weights
  C = weights_node_deg_unif(A)
  C = C*A
  C = normalized_weights(C, diag=diagmax, offdiag=offdiagmax)

  # Potential Outcomes Model
  fy = lambda z: linear_pom(C,alpha,z)

  TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))

  for i in range(T):
    z = bernoulli(n,p)
    y = fy(z)

    TTE_est = diff_in_means_fraction(n, p, y, A, z, 0.75)
    results.append({'Estimator': 'diff_means_frac', 'graph': graph, 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_est-TTE)/TTE})
    
    TTE_est = diff_in_means_naive(n, p, y, A, z)
    results.append({'Estimator': 'diff_means', 'graph': graph, 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_est-TTE)/TTE})

    TTE_est = est_us(n,p,y,A,z)
    results.append({'Estimator': 'SumIPS-Linear', 'graph': graph, 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_est-TTE)/TTE})
    
    TTE_est = est_ols(n,p,y,A,z)
    results.append({'Estimator': 'OLS', 'graph': graph, 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_est-TTE)/TTE})

    TTE_est = 1/(p*n) * np.sum(y - fy(np.zeros(n)))
    results.append({'Estimator': 'Graph-Agnostic', 'graph': graph, 'rep': i, 'n': n, 'p': p, 'ratio': r, 'Bias': (TTE_est-TTE)/TTE})
    
df = pd.DataFrame.from_records(results)
df.to_csv('outputFiles/'+graph+'-uniprop-size.csv')