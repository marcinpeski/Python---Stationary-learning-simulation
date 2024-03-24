#This 

import numpy as np
import matplotlib.pyplot as plt

no_elements = 5000
no_bins = 500

ro = 0.35
epsilon = 0.00000001
delta = 0.0001

steps = 200


def g(x):
    return x**3 +ro*x*((1-x)**2)

def f(x):
    return x*(1-x)**2

def compute(ro):
    g_epsilon = 1-g(1-epsilon)
    x = np.linspace(0, 1, no_elements)
    x = x*np.log(g_epsilon/epsilon)
    x = 1 - epsilon * np.exp(x)

    data = np.array([])
    for s in range(steps):
        data = np.hstack((data, x))
        x = g(x)

    s0 = np.sum(f(data))
    s1 = np.sum(f(1-data))
    p_1 = s1/(s0+s1)
    p_0 = (-np.log(ro))/(np.log(3)-np.log(ro))

    return p_0, p_1, data

def show(p_0, p_1, data):
    data = data[(data >= delta) & (data <= 1-delta)]
    d0 = data
    d1 = 1-data
        

    fig, (ax0, ax1, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    # Plot the histogram in the left panel
    bin_edges = np.linspace(delta, 1-delta, no_bins + 1)
    d0, _ = np.histogram(d0, bins=bin_edges)
    d1, _ = np.histogram(d1, bins=bin_edges)

    dd = d1/(d0+d1)
    ax3.plot(bin_edges[:-1], dd, color='red')
    ax3.set_title('f1/(f0+f1)')
    ax3.axhline(y=0.5, color='gray', linestyle='dotted')
    ax3.axhline(y=1-p_0, color='black', linestyle='dashed')
    ax3.axhline(y=p_1, color='blue')
    ax3.axhline(y=p_0, color='black', linestyle='dashed')

    ax1.plot(bin_edges[:-1], np.log(d0), color='black', label='log(f0)')
    ax1.plot(bin_edges[:-1], np.log(d1), color='green', label='log(f1)')
    ax1.set_xlabel('Values')
    ax1.set_ylabel('log values')
    ax1.set_title('log(f0) (black), log(f1) (green)')

    # Plot the function g(x) = x**2 in the right panel
    x_values = np.linspace(0, 1, 100)
    g_x = g(x_values)
    ax0.plot(x_values, g_x, color='black', label='g0(x)')
    ax0.plot(1-x_values, 1-g_x, color='green', label='g1(x)')
    ax0.plot(x_values, x_values, color='gray', linestyle='dashed')
    ax0.set_xlabel('x')
    ax0.set_ylabel('g0(x)')
    ax0.set_title('Function g0(x) (black), g1(x) (green)')

    # Adjust spacing between the two panels # Show the plot
    plt.tight_layout()
    plt.show()

def show_ps(results):
    plt.clf()
    fig, (ax0) = plt.subplots(1, 1, figsize=(4, 4))

    ax0.plot(results[:,0], results[:,1], color='black', label='p0')
    ax0.plot(results[:,0], results[:,2], color='green', label='p1')
    ax0.set_xlabel('ro')
    ax0.set_ylabel('p')
    ax0.set_title('p0 (black), p1 (green)')
    ax0.legend()
    plt.tight_layout()
    # Adjust spacing between the two panels
    plt.show()

#plt.ion()

ros = np.linspace(0.38, 0.5, 20)
results = []
for ro in ros:
    p_0, p_1, data = compute(ro)
    print(ro, p_0, p_1)
    results += [[ro, p_0, 1-p_1]]
    #show_ps(np.array(results))
    
#plt.ioff()

# Show the final plot (optional)
show_ps(np.array(results))

# If you want to keep the final plot open after the loop, add this line:
#plt.show()
