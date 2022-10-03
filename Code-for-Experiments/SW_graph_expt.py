from nci_linear_setup import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

diagmax = 10   # maximum norm of direct effect
r = 2
offdiagmax = r*diagmax   # maximum norm of indirect effect

p = 0.5    # treatment probability
T = 1000   # number of trials

sizes = np.array([16,24])#,32,48,64,96])#
results = []

for s in sizes:
  n = s*s
  print('n '+str(n))
  A = loadGraph('graphs/SW'+str(s)+'.txt', n, symmetric=True)
  graph = "sw"

  # baseline parameters
  alpha = np.random.rand(n)

  # Generate (normalized) weights
  C = weights_node_deg_unif(A)
  C = C*A
  C = normalized_weights(C, diag=diagmax, offdiag=offdiagmax)

  # print/load weights
  printWeights(C,alpha,'graphs/SW'+str(s)+'_weights.txt')
  (C,alpha) = loadWeights('graphs/SW'+str(s)+'_weights.txt',n)

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

fig = plt.figure()
ax = fig.add_subplot(111)

p = sns.lineplot(x='n', y='Bias', hue='Estimator', style='Estimator', data=df, ci='sd', legend='brief', markers=True)
p.set_xlabel("Population Size (n)", fontsize = 12)

p.set_ylabel("Relative Bias", fontsize = 12)
ax.set_title('Performance of Estimators', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels) # title="Custom Title"s

plt.savefig('outputFiles/'+graph+'-uniprop-size.pdf',format='pdf')