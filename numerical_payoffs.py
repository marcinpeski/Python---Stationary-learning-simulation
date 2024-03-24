#to create the environment (not needed anymore): python -m venv myenv
#to activate the enviro: myenv\Scripts\activate
#to load the modules used: pip install -r requirements.txt

import matplotlib
matplotlib.use('Agg')  # Set a non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import nStrategy as ns

#link to current directory
figures_dir = os.path.dirname(__file__) + '/figures/'
def payoff_picture(pis=[], n=3, cs=[]):
    if n <=4:
        regular = ns.StrategyClass(n = n, pattern = {1:'A0'}, name = 'regular')
        weird = ns.StrategyClass(n = n, pattern = {1:'A1'}, name = 'weird')
    strategies = [regular, weird]
    payoffs = {strategy.name:{pi:[] for pi in pis} for strategy in strategies}
    scatterX = {strategy.name:[] for strategy in strategies}
    scatterY = {strategy.name:[] for strategy in strategies}
    scattercX = {index_c:[] for index_c in range(len(cs))}
    scattercY = {index_c:[] for index_c in range(len(cs))}
    for pi in pis:
        for strategy in strategies:
            strategy.find_equilibrium_sb_pairs(pi)
            payoffs = strategy.find_equilibrium_payoffs()
            scatterX[strategy.name] += [pi]*len(payoffs)
            scatterY[strategy.name] += payoffs 
        for index_c in range(len(cs)):
            for strategy in strategies:
                payoffs = strategy.find_equilibrium_payoffs(c=cs[index_c])
                scattercX[index_c] += [pi]*len(payoffs)
                scattercY[index_c] += payoffs

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))  # 1 row, 3 columns
    
    dot_size = 5
    axs.scatter(scatterX['regular'], scatterY['regular'], color='orange', label='regular', s=dot_size)
    axs.scatter(scatterX['weird'], scatterY['weird'], color='green', label='weird', s=dot_size)
    for index_c in range(len(cs)):
        axs.scatter(scattercX[index_c], scattercY[index_c], label=f'c={cs[index_c]}', s=dot_size)
    axs.set_title(f'Payoffs for n={n}')
    axs.set_xlabel('$\\pi$ - probability of correct signal')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.grid(False)
    axs.legend(loc='best', frameon=False) 

    plt.tight_layout()
    plt.savefig(figures_dir + f'payoffs_{n}.png')


pi_size = 100
payoff_picture(pis = np.linspace(0.55, 0.95, pi_size), n=3, cs=[0.05, 0.1, 0.2])
#payoff_picture(pis = np.linspace(0.55, 0.99, pi_size), n=4)
