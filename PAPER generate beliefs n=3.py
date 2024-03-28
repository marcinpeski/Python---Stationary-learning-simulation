#to create the environment (not needed anymore): python -m venv myenv
#to activate the enviro: myenv\Scripts\activate
#to load the modules used: pip install -r requirements.txt

#THis script draws three plots (given n= 3, 4, or 5): 
#   Plot of transition functions,
#   plot of beliefs for different values of pi,


n_alphas = 400
n_pis = 400

epsilon = 1e-4
delta = 1e-3
pi = 0.8
first_pi = 0.51



import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from math import sqrt
from functools import partial


# abs(alpha) is the probability of playing 1 (if alpha <1) or 0 (if alhpa >0) 
# rather than acquiring information at sample k=1
def g_1(x, pi=0.55, alpha=0.3):
    return x**3 + 3*x**2*(1-x)*(max(0,alpha)+pi*(1-abs(alpha))) + 3*x*(1-x)**2*(max(0,-alpha)+pi*(1-abs(alpha)))

def psi(x, k, n):
    return x**k*(1-x)**(n-k)

def alpha_range(pi):
    #g_0<1
    if pi <2/3:
        alpha_min = 1 - 1/(3*(1-pi))
    else: 
        alpha_min = (2-3*pi)/(4-3*pi)
    alpha_max = 1 - 1/3/pi
    #g_1<1

    return alpha_min+delta, alpha_max-delta

def compute_beliefs_for_alpha(pi=0.55, n=3, alpha = 0.3):
#generate uniform grid from epsilon to 1-epsilon
    g = partial(g_1, pi=pi, alpha = alpha)
    g1_epsilon = g(epsilon)
    x = np.linspace(epsilon, g1_epsilon, 101)
    e = epsilon
    data = [x]
    all_xs = x
    fail = False
    k = 0
    while e<1-epsilon and not fail:
        e_old, e = e, g(e)
        fail = abs(e-e_old)<0.001*epsilon
        x = g(x)
        data += [x]
        all_xs = np.append(all_xs, x)
        k += 1
    if not fail:
        psi_data = {k: [psi(x, k, n) for x in data] for k in range(1,n)}
        psi_sum = {k:np.sum(psi_data[k], axis=0) for k in psi_data}
        p1 = psi_sum[1]/(psi_sum[1]+psi_sum[2])
        p1_range = (min(p1), max(p1))
        g0 = 3*(max(0,-alpha)+(1-pi)*(1-abs(alpha)))
        g1 = 3*(max(0,-alpha)+pi*(1-abs(alpha)))
        p0 = -np.log(g0)/(-np.log(g0)+np.log(g1))
        return p0, p1_range
    else:
        return None

def compute_beliefs(pi):
    alpha_min, alpha_max = alpha_range(pi)
    alpha_min, alpha_max = alpha_min + delta, alpha_max - delta
    alphas = np.linspace(alpha_min, alpha_max, n_alphas)
    if alpha_min<=0<=alpha_max:
        alphas = np.append(alphas, 0)
    beliefs = {}
    for alpha in alphas: 
        print(f'computing pi = {pi}, alpha = {alpha}', end='\r')
        beliefs[alpha] =compute_beliefs_for_alpha(pi, alpha=alpha)
    return beliefs

pis = np.linspace(first_pi, 1, n_pis)
beliefs = {pi:None for pi in pis}
for pi in pis:
    beliefs[pi] = compute_beliefs(pi)
#save beliefs to file
import pickle
with open('PAPER beliefs.pkl', 'wb') as f:
    pickle.dump(beliefs, f)
