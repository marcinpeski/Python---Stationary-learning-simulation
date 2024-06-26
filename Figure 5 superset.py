#to create the environment (not needed anymore): python -m venv myenv
#to activate the enviro: myenv\Scripts\activate
#to load the modules used: pip install -r requirements.txt

#THis script draws Figure "equilibria n=3 regions" from beliefs generated by 
#   Plot of transition functions,
#   plot of beliefs for different values of pi,


phat_step = 1e-3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import pickle
with open('beliefs.pkl', 'rb') as f:
    beliefs = pickle.load(f)

#select for profiles and beliefs only if alpha is positive if p1_min<0.5 and negative if  p1_max>0.5
profiles = {pi:{alpha:beliefs[pi][alpha] for alpha in beliefs[pi]           \
                if (beliefs[pi][alpha] is not None and                     \
                    ((alpha>=0 and beliefs[pi][alpha][1][0] <= 0.5)         \
                    or (alpha<=0 and beliefs[pi][alpha][1][1] >= 0.5)))} for pi in beliefs}

def payoffs(pi, alpha, beliefs):
    
    if alpha <0:
        phat_min = max(beliefs[1][0], 0.5)
        phat_max = min(beliefs[1][1], pi)
        etype = "-"
    elif alpha ==0:
        phat_min=max(beliefs[1][0], 1-beliefs[1][1], 0.5)
        phat_max = pi
        etype = "0"
    elif alpha>0:
        phat_min = max(1-beliefs[1][1], 0.5)
        phat_max = min(1-beliefs[1][0], pi)
        etype = "+" 
    ps = np.arange(phat_min, phat_max, phat_step)
    p0=beliefs[0]
    def a(phat,p0):
        if p0<1-phat:
            return '0'
        else: 
            return 'A'
    payoffs = [{'region':etype+a(p,p0), 'payoffs':max(1-p0, p), 'phat':p, 'p0':p0, 'p1':beliefs[1][0]} for p in ps]
    return payoffs

def find_superset_equilibria(beliefs):
    #select for profiles and beliefs only if they satisfy corrollary 1 conditions
    corollary1 = {pi:{alpha:payoffs(pi, alpha, beliefs[pi][alpha]) for alpha in beliefs[pi]           
                    if (pi < 1 and beliefs[pi][alpha] is not None and              
                        ((alpha<0 and beliefs[pi][alpha][1][1] >= 0.5 and beliefs[pi][alpha][1][0] <= pi) or
                         (alpha==0 and max(beliefs[pi][alpha][1][0], 1-beliefs[pi][alpha][1][0]) <= pi ) or 
                         (alpha>0 and beliefs[pi][alpha][1][0] <= 0.5 and 1- beliefs[pi][alpha][1][1] <= pi)))} for pi in beliefs}
    equilibria = {pi:{} for pi in corollary1}
    for pi in corollary1:
        for alpha in corollary1[pi]:
            for e in corollary1[pi][alpha]:
                if not e['phat'] in equilibria[pi]:
                    equilibria[pi][e['phat']] = []
                equilibria[pi][e['phat']].append(e)
    return equilibria

equilibria = find_superset_equilibria(beliefs)

test_cs = [0, 0.1, 0.3]
payoff = {c:{} for c in test_cs}
for c in test_cs:
    for pi in equilibria:
        available = [e for phat in equilibria[pi] for e in equilibria[pi][phat] if phat <= pi - c]
        if len(available) > 0:
            payoff[c][pi] = max([e['payoffs'] for e in available])
        else:
            payoff[c][pi] = 0.5

regions = ['+0', '+A', '00', '0A', '-0', '-A']
ebr, phat_max_r, phat_min_r = {region:{} for region in regions}, {region:{} for region in regions}, {region:{} for region in regions}
for region in regions:
    ebr[region] = {pi:[e for phat in equilibria[pi] for e in equilibria[pi][phat] if e['region'] == region] for pi in equilibria}
    for pi in ebr[region]:
        if len(ebr[region][pi]) > 0:        
            phat_max_r[region][pi] = max([e['phat'] for e in ebr[region][pi]])
            phat_min_r[region][pi] = min([e['phat'] for e in ebr[region][pi]])

colors, hatching, transparency = {}, {}, {}
for region in regions:
    if region[1] == '0':
        colors[region] = 'green'
        transparency[region] = 0.6
    else:
        colors[region] = 'orange'
        transparency[region] = 0.6
    if region[0] == '+':
        hatching[region] = '---'
    elif region[0] == '0':
        hatching[region] = ''
    else:
        hatching[region] = '|||'
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label=f'$\\beta(0)>0$'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label=f'$\\beta(0)=0$'),
]
legend_hatchings = [
    Patch(facecolor='white', edgecolor='grey', label=f'$\\beta(1)=1$'),
    Patch(facecolor='white', edgecolor='grey', hatch='---', label=f'$\\beta(1)<1$, $\\alpha(1, 1/2)=0$'),
    Patch(facecolor='white', edgecolor='grey', hatch='|||', label=f'$\\beta(1)<1$, $\\alpha(1, 1/2)=1$'),
]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for region in ebr:
    pis = list(phat_max_r[region].keys())
    max_vals = [phat_max_r[region][pi] for pi in pis]
    min_vals = [phat_min_r[region][pi] for pi in pis] 
    ax[0].fill_between(pis, min_vals, max_vals, alpha=transparency[region], color = colors[region], hatch = hatching[region], edgecolor='grey')
    ax[1].fill_between(pis, min_vals, max_vals, alpha=0.8, facecolor = 'white', edgecolor='lightgrey')
ax[0].set_ylabel("$\hat{p}$")
ax[0].set_xlabel("$\pi$")
legend1 = ax[0].legend(handles=legend_elements, loc='center left', frameon=False)
legend2 = ax[0].legend(handles=legend_hatchings, loc='upper left', frameon=False)
ax[0].add_artist(legend1)

colors = {0:'blue', 0.1:'purple', 0.3:'red'}
linestyle = {0:'-', 0.1:'--', 0.3:':'}

for c in test_cs:
    x = list(payoff[c].keys())
    y = list(payoff[c].values())
    ax[1].plot(x, y, ':', color = colors[c], linestyle = linestyle[c], label = f'$c={c}$')
    #ax[1].text(x[-1], y[-1], f'$c={c}$', va='center', color = colors[c])
ax[1].legend(loc='upper left', frameon=False)
ax[1].set_ylabel("welfare")
ax[1].set_xlabel("$\pi$")

for a in ax:
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)

plt.savefig(f'figures/Figure 5 superset.png')
plt.show()

