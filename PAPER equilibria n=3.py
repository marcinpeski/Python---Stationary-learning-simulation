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

import numpy as np
import matplotlib.pyplot as plt

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
        phat_min = min (phat_min, phat_max)
        etype = "0"
    else:
        phat_min = max(1-beliefs[1][1], beliefs[1][0])
        phat_max = max(1-beliefs[1][0], beliefs[1][1])
        if alpha>0:
            etype = "+"
        else:
            etype = "-"
    phat_min = min(phat_min, pi)
    phat_max = min(phat_max, pi)
    ps = np.arange(phat_min, phat_max, 0.01)
    p0=beliefs[0]
    def a(phat,p0):
        if p0<1-phat:
            return '0'
        else: 
            return 'A'
    payoffs = [{'type':etype+a(p,p0), 'payoffs':max(1-p0, p), 'phat':p, 'p0':p0, 'p1':beliefs[1][0]} for p in ps]
    return payoffs

equilibria = {pi:{alpha:payoffs(pi, alpha, beliefs[pi][alpha]) for alpha in profiles[pi]} for pi in profiles if pi < 1}
colors = {'+0':'blue', '+A':'green', '00':'orange', '0A':'yellow', '-0':'purple', '-A':'red'}
explanations = {}
for key in colors:
    if key[0] == '+':
        explanations[key] = 'k=1 acquires (regular), '
    elif key[0] == '-':
        explanations[key] = 'k=1 acquires (weird), '
    else:
        explanations[key] = 'k=1 doesn\'t ac., '
    if key[1] == '0':
        explanations[key] += 'k=0 doesn\'t ac.'
    else: 
        explanations[key] += 'k=0 acquires'

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
plt.ylabel("$\hat{p}$")
plt.xlabel("$\pi$")
plt.legend(handles=legend_elements, loc='upper left', frameon=False)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(f'figures/equilibria n=3.png')
plt.show()




