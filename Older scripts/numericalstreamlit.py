#to create the environment (not needed anymore): python -m venv myenv
#to activate the enviro: myenv\Scripts\activate
#to load the modules used: pip install -r requirements.txt

# To test run it locally, run: streamlit run numericalstreamlit.py
# To stop the local server, press Ctrl+C (in the terminal window)

#TOO BLOODY SLOW

import streamlit as st
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
 
import matplotlib.pyplot as plt
import numpy as np


#Parameters for belief computations
gamma = 1e-3
density = 1000
epsilon = 1e-3
a = 1e-10
K=100

#Parameters for show
K_show = 100000
delta = 1e-3
no_bins = 200


def g0(x, alpha, pi):
    return x*x*x + 3*x*x*(1-x)*alpha*(1-pi) + 3*x*(1-x)*(1-x)*(alpha*(1-pi) + 1-alpha)

def g1(x, alpha, pi):
    return x*x*x + 3*x*x*(1-x)*alpha*pi + 3*x*(1-x)*(1-x)*(alpha*pi + 1-alpha)

def psi1(x):
    return x*(1-x)*(1-x)

def psi2(x):
    return x*x*(1-x)

#Main picture
def draw(alpha_show, pi, mode):
    global K
    alpha_min = 2/3/pi
    
    #Compute beliefs
    I1s = []
    I2s = []
    ps = []
    alphas = []
    for d in range(density):
        alpha = alpha_min + gamma + d*(1-alpha_min- gamma)/density
        I1 = 0
        I2 = 0

        x0 = a
        x1 = g1(x0, alpha, pi)
        logys = np.linspace(np.log(x0), np.log(x1), K+1)
        ys = np.exp(logys)
        
        while x1 < 1 - epsilon:
            x0 = x1
            x1 = g1(x0, alpha, pi)
            ys = g1(ys, alpha, pi)
            I1 += 1/K*np.sum(psi1(ys))
            I2 += 1/K*np.sum(psi2(ys))
        
        alphas.append(alpha)
        I1s.append(I1)
        I2s.append(I2)
        ps.append(I1/(I1+I2))

    K = K_show
    alpha = alpha_show
    x0 = a
    x1 = g1(x0, alpha, pi)
    logys = np.linspace(np.log(x0), np.log(x1), K+1)
    ys = np.exp(logys)
    all_ys = []
    I1, I2 = 0, 0
    while x1 < 1 - epsilon:
        x0 = x1
        x1 = g1(x0, alpha, pi)
        ys = g1(ys, alpha, pi)
        all_ys = np.concatenate((all_ys, ys))
        I1 += 1/K*np.sum(psi1(ys))
        I2 += 1/K*np.sum(psi2(ys))
    show = [alpha, I1/(I1+I2)]

    # Create a labeled graph
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))  # 1 row, 3 columns

    # Plot your data in the first subplot
    axs[0].plot(alphas, ps, linestyle='-')
    axs[0].plot(show[0], show[1], marker = 'o', color = 'red')
    axs[0].set_xlabel('Alphas')
    axs[0].set_ylabel('Ps')
    axs[0].set_title('Beliefs $p_1$')
    axs[0].grid(True)

    xs = np.linspace(0, 1, 1000)
    y0s = g0(xs, show[0], pi)
    y1s = g1(xs, show[0], pi)
    axs[1].plot(y0s, xs, linestyle='-', color = 'green')
    axs[1].plot([0, 0.1], [1,1], linestyle='-', color = 'green')
    axs[1].text(0.2, 1, '$g_0$', horizontalalignment='center', verticalalignment='center')

    axs[1].plot(y1s, xs, linestyle='-', color = 'blue')
    axs[1].plot([0, 0.1], [0.95,0.95], linestyle='-', color = 'blue')
    axs[1].text(0.2, 0.95, '$g_1$', horizontalalignment='center', verticalalignment='center')
    axs[1].set_xlabel("state $x$")
    axs[1].set_title(f'$g_0$ and $g_1$ for $\\alpha$={alpha}')
    axs[1].grid(False)

    d1s = all_ys
    d0s = 1-all_ys
    bin_edges = np.linspace(delta, 1-delta, no_bins+1)
    f0s, _ = np.histogram(d0s, bins=bin_edges)
    f0s = f0s/np.max(f0s)
    f1s, _ = np.histogram(d1s, bins=bin_edges)
    f1s = f1s/np.max(f1s)
    axs[2].plot(bin_edges[:-1], np.log(f0s), color='green', label='$f_0$')
    axs[2].plot([0.2, 0.3], [1,1], linestyle='-', color = 'green')
    axs[2].text(0.4, 1, '$f_0$', horizontalalignment='center', verticalalignment='center')
    axs[2].plot(bin_edges[:-1], np.log(f1s), color='blue', label='$f_1$')
    axs[2].plot([0.2, 0.3], [0.5,0.5], linestyle='-', color = 'blue')
    axs[2].text(0.4, 0.5, '$f_1$', horizontalalignment='center', verticalalignment='center')
    axs[2].set_xlabel('state $x$')
    axs[2].set_title(f'log densities for $\\alpha$={alpha}')
    axs[2].tick_params(left = False, right = False , labelleft = False)
    axs[2].grid(False)

    axs[3].plot(bin_edges[:-1], np.log(f1s)-np.log(f0s), color='red')
    axs[3].set_xlabel('state $x$')
    axs[3].set_title('$log(f_1)-log(f_0)$')
    axs[3].grid(True)

    if mode == 'development':
        plt.show()
    else:
        return fig

mode = 'deployment' 
#mode = 'development'

if mode == 'deployment':
    # Create columns
    col0, col1 = st.columns([5, 5])
    with col0:
        pi_value = st.slider('pi:', min_value=2/3, max_value=float(0.999), value=float(0.8), step=0.01, format='%.3f')
    with col1:
        alpha_value = st.slider('alpha:', min_value=2/3, max_value=float(1), value=0.8, step=0.01, format='%.3f')
    fig = draw(alpha_show = alpha_value, pi = pi_value, mode = mode)
    st.pyplot(fig)
    
else:
    draw(alpha_show = 0.8, pi = 0.99, mode = mode)