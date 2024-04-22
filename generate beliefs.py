#to create the environment (not needed anymore): python -m venv myenv
#to activate the enviro: myenv\Scripts\activate
#to load the modules used: pip install -r requirements.txt

#THis script draws three plots (given n= 3, 4, or 5): 
#   Plot of transition functions,
#   plot of beliefs for different values of pi,
n_alphas = 200
n_pis = 200
n_xs = 101

epsilon = 1e-4
delta = 1e-3
pi = 0.8
first_pi = 0.52


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

def generate_alphas(pi):
    if pi <2/3:
        alpha_min = 1 - 1/(3*(1-pi))
    else: 
        alpha_min = 2/3*pi - 1
    alpha_max = 1 - 1/3/pi
    alpha_min, alpha_max = alpha_min + delta, alpha_max - delta
    alphas_first = np.linspace(alpha_min, alpha_max, n_alphas)
    if alpha_min<=0<=alpha_max:
        alphas_first = np.append(alphas_first, 0)
    alphas = []
    for alpha in alphas_first:
        g = partial(g_1, pi=pi, alpha = alpha)
        e = epsilon
        fail = False 
        while e<1-epsilon and not fail:
            e_old, e = e, g(e)
            fail = abs(e-e_old)<0.001*epsilon
        if not fail:
            alphas.append(alpha)
    return alphas

def compute_beliefs_for_alpha(pi=0.55, n=3, alpha = 0.3):
    print(f'computing pi = {pi}, alpha = {alpha}', end='\r')
    g = partial(g_1, pi=pi, alpha = alpha)
    g1_epsilon = g(epsilon)
    x = np.linspace(epsilon, g1_epsilon, n_xs)
    data = [x]
    while g(x[0])<1-epsilon:
        x = g(x)
        data += [x]
    psi_data = {k: [psi(x, k, n) for x in data] for k in range(1,n)}
    psi_sum = {k:np.sum(psi_data[k], axis=0) for k in psi_data}
    p1 = psi_sum[1]/(psi_sum[1]+psi_sum[2])
    p1_range = (min(p1), max(p1))
    g0 = 3*(max(0,-alpha)+(1-pi)*(1-abs(alpha)))
    g1 = 3*(max(0,-alpha)+pi*(1-abs(alpha)))
    p0 = -np.log(g0)/(-np.log(g0)+np.log(g1))
    return p0, p1_range
    
#generate beliefs
pis = np.linspace(first_pi, 1, n_pis)
beliefs = {pi:{alpha:compute_beliefs_for_alpha(pi, alpha=alpha) for alpha in generate_alphas(pi)} for pi in pis}

#save beliefs to file
import pickle
with open('beliefs.pkl', 'wb') as f:
    pickle.dump(beliefs, f)
