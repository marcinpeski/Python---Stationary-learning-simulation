#to create the environment (not needed anymore): python -m venv myenv
#to activate the enviro: myenv\Scripts\activate
#to load the modules used: pip install -r requirements.txt

#THis script draws three plots (given n= 3, 4, or 5): 
#   Plot of transition functions,
#   plot of beliefs for different values of pi,


n_grid = 100
n_density = 20000
no_bins = 200
epsilon = 1e-7
delta = 1e-7
pi = 0.8

phat_step = 1e-3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import pickle
with open('PAPER beliefs.pkl', 'rb') as f:
    beliefs = pickle.load(f)

#select for profiles and beliefs only if alpha is positive if p1_min<0.5 and negative if  p1_max>0.5
profiles = {pi:{alpha:beliefs[pi][alpha] for alpha in beliefs[pi]           \
                if (beliefs[pi][alpha] is not None and                     \
                    ((alpha>=0 and beliefs[pi][alpha][1][0] <= 0.5)         \
                    or (alpha<=0 and beliefs[pi][alpha][1][1] >= 0.5)))} for pi in beliefs}

def payoffs(pi, alpha, beliefs):
    
    if alpha ==0:
        phat_min=max(beliefs[1][0], 1-beliefs[1][1])
        phat_max = pi
        etype = "0"
    elif alpha>0:
        phat_min = 1-beliefs[1][1]
        phat_max = 1-beliefs[1][0]
        etype = "+"
    else:
        phat_min = beliefs[1][0]
        phat_max = beliefs[1][1]
        etype = "-"
    phat_min = max(0.5, min(phat_min, pi))
    phat_max = max(0.5, min(phat_max, pi))
    ps = np.arange(phat_min, phat_max, phat_step)
    p0=beliefs[0]
    def a(phat,p0):
        if p0<1-phat:
            return '0'
        else: 
            return 'A'
    payoffs = [{'type':etype+a(p,p0), 'payoffs':max(1-p0, p), 'phat':p, 'p0':p0, 'p1':beliefs[1][0]} for p in ps]
    return payoffs

equilibria = {pi:{alpha:payoffs(pi, alpha, beliefs[pi][alpha]) for alpha in profiles[pi]} for pi in profiles if pi < 1}
test_cs = [0, 0.1]
payoff = {c:{} for c in test_cs}
for c in test_cs:
    for pi in equilibria:
        available = [e for alpha in equilibria[pi] for e in equilibria[pi][alpha] if e['phat'] <= pi - c]
        if len(available) > 0:
            payoff[c][pi] = max([e['payoffs'] for e in available])
        else:
            payoff[c][pi] = 0.5

regions = ['+0', '+A', '00', '0A', '-0', '-A']
ebr, phat_max_r, phat_min_r = {region:{} for region in regions}, {region:{} for region in regions}, {region:{} for region in regions}
for region in regions:
    ebr[region] = {pi:[e for alpha in equilibria[pi] for e in equilibria[pi][alpha] if e['type'] == region] for pi in equilibria}
    for pi in ebr[region]:
        if len(ebr[region][pi]) > 0:        
            phat_max_r[region][pi] = max([e['phat'] for e in ebr[region][pi]])
            phat_min_r[region][pi] = min([e['phat'] for e in ebr[region][pi]])

colors, hatching = {}, {}
for region in regions:
    if region[1] == '0':
        colors[region] = 'green'
    else:
        colors[region] = 'orange'
    if region[0] == '+':
        hatching[region] = '---'
    elif region[0] == '0':
        hatching[region] = ''
    else:
        hatching[region] = '|||'
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label=f'$\\beta_0=0$'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label=f'$\\beta_0>0$'),
    Patch(facecolor='white', edgecolor='black', label=f'$\\beta_1=1$'),
    Patch(facecolor='white', edgecolor='black', hatch='---', label=f'$\\beta_1<1$, $\\alpha_1 (1/2)=0$'),
    Patch(facecolor='white', edgecolor='black', hatch='|||', label=f'$\\beta_1<1$, $\\alpha_1 (1/2)=1$'),
]
for region in ebr:
    pis = list(phat_max_r[region].keys())
    max_vals = [phat_max_r[region][pi] for pi in pis]
    min_vals = [phat_min_r[region][pi] for pi in pis] 
    plt.fill_between(pis, min_vals, max_vals, alpha=0.5, color = colors[region], hatch = hatching[region], edgecolor='black')
for c in test_cs:
    plt.plot(list(payoff[c].keys()), list(payoff[c].values()), ':', label=f'$c={c}$')
plt.ylabel("$\hat{p}$")
plt.xlabel("$\pi$")
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(handles=legend_elements, loc='upper left', frameon=False)
plt.savefig(f'figures/equilibria n=3 regions.png')
plt.show()




#The rest
all_payoffs = {pi:[e['payoffs'] for alpha in equilibria[pi] for e in equilibria[pi][alpha]] for pi in equilibria}  
max_payoffs = {pi:max(all_payoffs[pi]) for pi in all_payoffs}
colors = {'+0':'blue', '+A':'green', '00':'orange', '0A':'yellow', '-0':'purple', '-A':'red'}
shading = {'+0':'blue', '+A':'green', '00':'orange', '0A':'yellow', '-0':'purple', '-A':'red'}
explanations = {}
for key in colors:
    explanations[key] = ''
    if key[1] == '0':
        explanations[key] += 'k=0 never acquires'
    else: 
        explanations[key] += 'k=0 sometimes acquires'
    explanations[key] += ', '
    if key[0] == '+':
        explanations[key] += 'k=1 acquires or plays 0'
    elif key[0] == '-':
        explanations[key] += 'k=1 acquires or plays 1'
    else:
        explanations[key] += 'k=1 always acquires'
    

legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=explanations[key], markerfacecolor=value, markersize=10) for key, value in colors.items()]

drawing = []
for pi in equilibria:
    for alpha in equilibria[pi]:
        for e in equilibria[pi][alpha]:
            p = (pi, e['phat'], colors[e['type']])
            drawing.append(p)
print(len(drawing))
for d in drawing:
    plt.plot(d[0], d[1], marker = 'o', color = d[2], alpha =0.7, markersize=1)    
for pi in max_payoffs:
    plt.plot(pi, max_payoffs[pi], marker = 'o', color = 'red', markersize=1)
plt.ylabel("$\hat{p}$")
plt.xlabel("$\pi$")
plt.legend(handles=legend_elements, loc='upper left', frameon=False)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(f'figures/equilibria n=3 old.png')
plt.show()




