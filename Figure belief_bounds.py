#to create the environment (not needed anymore): python -m venv myenv
#to activate the enviro: myenv\Scripts\activate
#to load the modules used: pip install -r requirements.txt

#THis script draws Figure from the paper "belief_bounds" (given n= 3, 4, or 5): 
#   Plot of transition functions,
#   plot of beliefs for different values of pi,


n_grid = 100
n_density = 20000
no_bins = 200
epsilon = 1e-7
delta = 1e-7
pi = 0.8

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from math import sqrt
from functools import partial

def g_1_old(x, pi=pi):
    return x**3 + 3*x*(1-x)*pi

def g_1(x, pi=pi, n=3):
    return x**n + sum([comb(n,i)*x**i*(1-x)**(n-i)*pi for i in range(1,n)])

#alpha is the probability of playing 1 (rather than acquiring information) at sample k=1
#in the regular case, alpha == 0
#only case n=3
def g_1_weird(x, pi=pi, n=3, alpha=0.5):
    return x**3 + 3*x**2*(1-x)*(pi*(1-alpha)) + 3*x*(1-x)**2*(pi*(1-alpha)+alpha)

def psi(x, k, n):
    return x**k*(1-x)**(n-k)

def psi_1(x):
    return x*(1-x)**2

def psi_2(x):
    return x**2*(1-x)

def compute_ranges(pi, n=3, weird = False, alpha = False):
#generate uniform grid from epsilon to 1-epsilon
    if n==3 and alpha != False:
        g = partial(g_1_weird, alpha = alpha)
    else:
        g = g_1
    g1_epsilon = g(epsilon, pi=pi, n=n)
    x = np.linspace(epsilon, g1_epsilon, n_grid)
    e = epsilon
    data = [x]
    while e<1-epsilon:
        e = g(e, pi=pi, n=n)
        x = g(x, pi=pi, n=n)
        data += [x]
    psi_data = {k: [psi(x, k, n) for x in data] for k in range(1,n)}
    psi_sum = {k:np.sum(psi_data[k], axis=0) for k in psi_data}
    p = {k: psi_sum[k]/(psi_sum[k]+psi_sum[n-k]) for k in psi_sum}
    p_range = {k:(min(p[k]), max(p[k])) for k in p}
    return p_range

def compute_density(pi, n=3, alpha = False):
#generate uniform grid from epsilon to 1-epsilon
    if n==3 and alpha != False:
        g = partial(g_1_weird, alpha = alpha)
    else:
        g = g_1
    g1_epsilon = g(epsilon, pi=pi, n=n)
    density = np.exp(np.linspace(np.log(epsilon), np.log(g1_epsilon), n_density))
    x_old = density
    e = epsilon
    while e<1-epsilon:
        e = g(e, pi=pi, n=n)
        x = g(x_old, pi=pi, n=n)
        density = np.concatenate((density, x))
        x_old = x
    density = density[(density >= delta) & (density <= 1-delta)]
    bin_edges = np.linspace(delta, 1-delta, no_bins + 1)
    d1, _ = np.histogram(density, bins=bin_edges)
    d0, _ = np.histogram(1-density, bins=bin_edges)
    return np.log(d0), np.log(d1)

def generate_plots(n=3, pi_show = 0.9, alpha = False):
    plotsize = 4
    fig, axs = plt.subplots(1, 3, figsize=(plotsize*3,plotsize))
    pi_min = (1+sqrt(1-4/n**2))/2
    pis = np.linspace(pi_min, 1, 50)
    ranges = {pi:[] for pi in pis}
    for pi in pis:
        print(f'Computing ranges for pi={pi:.2f}, n={n} and alpha={alpha}', end="\r")
        print(f'Computing ranges for pi={pi:.2f}, n={n} and alpha={alpha}', end="\r")
        ranges[pi] = compute_ranges(pi, n, alpha = alpha)
    d0, d1 = compute_density(pi_show, n, alpha = alpha)

    #Draw transition functions
    f = 0
    axs[f].set_xlabel('$x$')
    axs[f].set_ylabel(f'$g_0(x)$, $g_1(x)$ for $\pi={pi_show:.2f}$')
    axs[f].grid(False)
    axs[f].plot(np.linspace(epsilon, 1-epsilon, n_grid), 1-g_1(1-np.linspace(epsilon, 1-epsilon, n_grid), pi=pi_show, n=n), label = f'$g_0$')
    axs[f].plot(np.linspace(epsilon, 1-epsilon, n_grid), g_1(np.linspace(epsilon, 1-epsilon, n_grid), pi=pi_show, n=n), label = f'$g_1$')
    axs[f].legend(loc='best', frameon=False)
    
    #Draw densities
    f = 1
    axs[f].set_xlabel('$x$')
    axs[f].set_ylabel(f'$log(f_0)(x)$, $log(f_1)(x)$ for $\pi={pi_show:.2f}$')
    axs[f].grid(False)
    axs[f].plot(np.linspace(delta, 1-delta, no_bins), d0, label='$log(f_0)$')
    axs[f].plot(np.linspace(delta, 1-delta, no_bins), d1, label='$log(f_1)$')
    axs[f].legend(loc='best', frameon=False)

    #Draw beliefs
    f = 2
    axs[f].set_xlabel('precision $\pi$')
    axs[f].set_ylabel('equilibrium beliefs')
    axs[f].grid(False)
    axs[f].set_ylim(0,1)
    p_min = {k:[ranges[pi][k][0] for pi in pis] for k in range(1,n)}
    p_max = {k:[ranges[pi][k][1] for pi in pis] for k in range(1,n)}
    colors = ['green', 'purple']
    for k in range(1,n):
        color = colors[k-1]
        axs[f].plot(pis, p_min[k], color=color, label = f'$p_{k}$' )
        axs[f].plot(pis, p_max[k], color=color)
        axs[f].fill_between(pis, p_min[k], p_max[k], alpha=0.5, facecolor=color)
    axs[f].legend(loc='best', frameon=False)

    for a in axs:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        
    return fig, axs


pi_show = 0.99
alpha = 0
for n in [3]:
    fig, axs = generate_plots(n=n, pi_show = pi_show, alpha = alpha)
    plt.tight_layout()
    plt.savefig(f'figures/belief_bounds n={n}.png')
    #plt.show()




