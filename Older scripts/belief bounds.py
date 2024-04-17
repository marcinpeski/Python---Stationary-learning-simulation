#to create the environment (not needed anymore): python -m venv myenv
#to activate the enviro: myenv\Scripts\activate
#to load the modules used: pip install -r requirements.txt

#THis script draws two plots for n=3 and n=4: Plot of transition functions and also plot of beliefs for different values of pi

n_grid = 100
epsilon = 1e-7
pi = 0.8

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from math import sqrt

def g_1_old(x, pi=pi):
    return x**3 + 3*x*(1-x)*pi

def g_1(x, pi=pi, n=3):
    return x**n + sum([comb(n,i)*x**i*(1-x)**(n-i)*pi for i in range(1,n)])

def psi(x, k, n):
    return x**k*(1-x)**(n-k)

def psi_1(x):
    return x*(1-x)**2

def psi_2(x):
    return x**2*(1-x)

def compute_ranges(pi, n=3):
#generate uniform grid from epsilon to 1-epsilon
    x = np.linspace(epsilon, g_1(epsilon), n_grid)
    e = epsilon
    data = [x]
    k=0
    while e<1-epsilon:
        e = g_1(e, pi=pi, n=n)
        x = g_1(x, pi=pi, n=n)
        data += [x]
        k += 1
    psi_data = {k: [psi(x, k, n) for x in data] for k in range(1,n)}
    psi_sum = {k:np.sum(psi_data[k], axis=0) for k in psi_data}
    p = {k: psi_sum[k]/(psi_sum[k]+psi_sum[n-k]) for k in psi_sum}
    p_range = {k:(min(p[k]), max(p[k])) for k in p}
    return p_range

plotsize = 4
ns = [3, 4]
ranges = {n:{} for n in ns}
fig, axs = plt.subplots(2, len(ns), figsize=(plotsize*len(ns),plotsize*2))
for i in range(len(ns)):
    n = ns[i]
    pi_min = (1+sqrt(1-4/n**2))/2
    pis = np.linspace(pi_min, 1, 50)
    ranges[n] = {pi:compute_ranges(pi, n) for pi in pis}
    axs[0][i].set_xlabel('$\pi$')
    axs[0][i].set_ylabel('beliefs')
    axs[0][i].grid()
    axs[0][i].set_title(f'$n={n}$')
    axs[0][i].set_ylim(0,1)
    for k in range(1,n):
        axs[0][i].fill_between(pis, [ranges[n][pi][k][0] for pi in pis], [ranges[n][pi][k][1] for pi in pis], alpha=0.5, label = f'$p_{k}$' )
    axs[0][i].legend(loc='best', frameon=False)
    axs[1][i].set_xlabel('$x$')
    axs[1][i].set_ylabel(f'$g_1(x)$ for $\pi={pi_min:.2f}$')
    axs[1][i].grid()
    axs[1][i].plot(np.linspace(epsilon, 1-epsilon, n_grid), g_1(np.linspace(epsilon, 1-epsilon, n_grid), pi=pi_min, n=n), label = f'$g_1$')
    axs[1][i].plot(np.linspace(epsilon, 1-epsilon, n_grid), 1-g_1(1-np.linspace(epsilon, 1-epsilon, n_grid), pi=pi_min, n=n), label = f'$g_0$')
    axs[1][i].legend(loc='best', frameon=False)


plt.tight_layout()
plt.savefig('figures/belief_bounds.png')
plt.show()




