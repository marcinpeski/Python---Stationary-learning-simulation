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
delta = 1e-3
no_bins = 1000

#Main picture
def draw(alpha_show, strategy_class):
    sb_pairs = [(strategy, strategy.beliefs()) for strategy in strategy_class.generate_strategies()]
    equilibria = [pair for pair in sb_pairs if strategy_class.is_equilibrium(pair[1])]
    
    strategy_show = strategy_class.new_strategy(alpha_show)

    # Create a labeled graph
    fig, axs = plt.subplots(1, 4, figsize=(12, 5))  # 1 row, 3 columns
    fig.suptitle(f'{strategy_show}', fontsize=16)

    # Plot your data in the first subplot
    alphas = [pair[0].alpha for pair in sb_pairs]
    p0s = [pair[1][0] for pair in sb_pairs]
    p1s = [pair[1][1] for pair in sb_pairs]
    alpha_min, alpha_max = strategy_class.alpha_range()

    if equilibria != []:
        for pair in equilibria:
            axs[0].axvline(x=pair[0].alpha, color=(255/255, 192/255, 203/255), linestyle='-')
        axs[0].text((equilibria[0][0].alpha+equilibria[-1][0].alpha)/2, 0.2, '\nequilibrium\nzone', horizontalalignment='center', verticalalignment='center')    
    axs[0].axhline(y=0.5, color='lightgrey', linestyle=':')    
    axs[0].plot(alphas, p0s, linestyle='-', color='orange', label='$p_0$')
    axs[0].plot(alphas, p1s, linestyle='-', color = 'black', label='$p_1$')
    axs[0].axvline(x=alpha_show, color='red', linestyle=':', label = '$\\alpha$')
    axs[0].set_xlabel('$\\alpha$ - probability of info acquisition')
    axs[0].set_title(f'beliefs')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].grid(False)
    axs[0].legend(loc='best', frameon=False) 

    xs = np.linspace(0, 1, 1000)
    y0s = strategy_show.g(0, xs)
    y1s = strategy_show.g(1, xs)
    axs[1].plot(xs, y0s, linestyle='-', color = 'green', label='$g_0$')
    axs[1].plot(xs, y1s, linestyle='-', color = 'blue', label='$g_1$')
    axs[1].plot([0, 0], [1,1], linestyle=':', color = 'lightgrey')
    axs[1].set_xlabel("state $x$")
    axs[1].set_title(f'transition functions')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].grid(False)
    axs[1].legend(loc='best', frameon=False) 

    distribution_show = strategy_show.generate_f1_distribution(K= ns.K_show)
    d1s = distribution_show
    d0s = 1-distribution_show
    bin_edges = np.linspace(delta, 1-delta, no_bins+1)
    f0s, _ = np.histogram(d0s, bins=bin_edges)
    f0s = f0s/np.max(f0s)
    f1s, _ = np.histogram(d1s, bins=bin_edges)
    f1s = f1s/np.max(f1s)
    axs[2].plot(bin_edges[:-1], np.log(f0s), color='green', label='$log(f_0)$')
    axs[2].plot(bin_edges[:-1], np.log(f1s), color='blue', label='$log(f_1)$')
    axs[2].set_xlabel('state $x$')
    axs[2].set_title(f'log densities')
    axs[2].tick_params(left = False, right = False , labelleft = False)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].grid(False)
    axs[2].legend(loc='best', frameon=False)

    axs[3].plot(bin_edges[:-1], np.log(f1s)-np.log(f0s), color="#008080")
    axs[3].set_xlabel('state $x$')
    axs[3].set_title('$log(f_1)-log(f_0)$')
    axs[3].axhline(y=0, color='lightgrey', linestyle=':')
    axs[3].spines['top'].set_visible(False)
    axs[3].spines['right'].set_visible(False)
    axs[3].grid(False)
    
    # Display the plot
    plt.tight_layout()
    plt.savefig(figures_dir + f'plot_{strategy_show}.png')
    #plt.show()
    return fig

n = 3
regular = ns.StrategyClass(n = n, pattern = {1:'A0'}, name = 'regular')
weird = ns.StrategyClass(n = n, pattern = {1:'A1'}, name = 'weird')
pis= np.linspace(0.9, 0.99, 5)
strategy_class = regular
for pi in pis:
    strategy_class.set_pi(pi) 
    alphas = strategy_class.alphas_show()
    for alpha_show in alphas:
        print("***** Computations for "f'alpha={alpha_show:.2f}, pi={pi:.2f}, name={strategy_class}')
        fig = draw(alpha_show, strategy_class)
