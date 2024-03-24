#part of numerical package

import numpy as np
import itertools

density = {1:600, 2:100}
gamma = 3e-3
epsilon = 1e-3
x_min = 1e-5
eq_gamma = 1e-3

K = 100
K_show = 100000

def psi(x, n, k):
    return x**k * (1-x)**(n-k)

class StrategyClass:
    def __init__(self, n=3, pattern = {1:'A0'}, name = ''):
        self.n = n
        self.kmax = n//2
        self.pattern = pattern
        self.pi = 0.5
        self.name = name
        self.sb_pairs = []

    def set_pi(self, pi):
        self.pi = pi

    #compute the range of alphas for a given pi
    def alpha_range(self):
        pi = self.pi
        if self.n<=4 and self.pattern == {1:'A0'}:
            self.alpha_min = 1/self.n/np.sqrt(pi*(1-pi))+gamma
            self.alpha_max = 1/self.n/(1-pi)-gamma
        if self.n<=4 and self.pattern == {1:'A1'}:
            self.alpha_min = (self.n-1)/self.n/pi+gamma
            self.alpha_max = 1-gamma
        self.alpha_min = min(self.alpha_min, 1-gamma)
        self.alpha_max = min(self.alpha_max, 1-gamma)
        return self.alpha_min, self.alpha_max

    #prepare a list of few alphas for detailed nalysis
    def alphas_show(self):
        amin, amax = self.alpha_range()
        if amin<amax:
            if self.n<=4:
                return np.linspace(amin, amax, 10)
        else:
            return []
        
    #prepare a list of many alphas for computing beliefes
    def alphas_computations(self):
        amin, amax = self.alpha_range()
        if amin<amax:
            if self.n<=4:
                return np.linspace(amin, amax, density[self.kmax])
        else:
            return []
        
    def __repr__(self):
        if self.name == '':
            pattern = '; '+'_'.join([f'{k}{v}' for k,v in self.pattern.items()])
        else:
            pattern = ''
        return f'{self.name} (n={self.n}; pi={self.pi:.2f}{pattern})'
    
    def new_strategy(self, alpha):
        return Strategy(alpha, self.pi, self)
    
    #generate strategies for a given pi from the set of alphas 
    def generate_strategies(self):
        alphas = self.alphas_computations()
        return [self.new_strategy(alpha) for alpha in alphas]
    
    #check if a particular strategy is an equilibrium given beliefs
    def is_equilibrium(self, beliefs, c= None):
        if self.n <= 4 and self.pattern == {1:'A0'}:
            equilibrium = beliefs[0] < beliefs[1] and beliefs[1] < 0.5+gamma and beliefs[1] > 1 - self.pi-gamma
            if c is not None:
                equilibrium = equilibrium and abs(beliefs[1] - (1-self.pi+c)) < eq_gamma
        if self.n <= 4 and self.pattern == {1:'A1'}:
            equilibrium = beliefs[0] < 1-beliefs[1] and beliefs[1] > 0.5-gamma and beliefs[1] < self.pi+gamma
            if c is not None:
                equilibrium = equilibrium and abs(beliefs[1] - (self.pi-c)) < eq_gamma
        
        return equilibrium
    
    def find_equilibrium_sb_pairs(self, pi):
        self.set_pi(pi)
        self.sb_pairs = [(strategy, strategy.beliefs()) for strategy in self.generate_strategies()]
        return self.sb_pairs
    
    #compute possible equilibrium payoffs for a given pi
    def find_equilibrium_payoffs(self, sb_pairs = None, c=None):
        sb_pairs = self.sb_pairs
        equilibria = [pair for pair in sb_pairs if self.is_equilibrium(pair[1], c=c)]
        payoffs = [1-pair[1][0] for pair in equilibria]
        return payoffs

class Strategy:
    def __init__(self, alpha, pi, strategy_class):
        self.strategy_class = strategy_class
        self.alpha = alpha
        self.pi = pi
        self.pa1s = [self.probabilities_action_1(0), self.probabilities_action_1(1)]

    def probabilities_action_1(self, state):
        n = self.strategy_class.n
        prob_1 = state*self.pi + (1-state)*(1-self.pi)  #probability that signal is 1
        '''
        pas = np.zeros(n+1)
        pas[0] = 0
        pas[1] = 1
        if n%2 == 0:
            pas[n//2] = prob_1
        for k,v in self.strategy_class.pattern.items():
            if 'A' in v:
                pas[k] = self.alpha*prob_1
                pas[n-k] = self.alpha*prob_1
            if v == 'A0':
                pas[n-k] += 1-self.alpha
            if v == 'A1':
                pas[k] += 1-self.alpha
        '''
        if self.strategy_class.pattern == {1:'A0'}:
            if n==3:
                return (0, 3* self.alpha*prob_1, 3* (self.alpha*prob_1 + 1-self.alpha), 1)
            if n==4:
                return (0, 4* self.alpha*prob_1, 6*prob_1, 4* (self.alpha*prob_1 + 1-self.alpha), 1)
        if self.strategy_class.pattern == {1:'A1'}:
            if n==3:
                return (0, 3* (self.alpha*prob_1 + 1-self.alpha), 3* self.alpha*prob_1, 1)
            if n==4:
                return (0, 4* (self.alpha*prob_1 + 1-self.alpha), 6*prob_1, 4* self.alpha*prob_1, 1)

    def g(self, state, x):
        return sum([self.pa1s[state][k] * (x**k) * ((1-x)**(self.strategy_class.n-k)) for k in range(self.strategy_class.n+1)])
       
    #Generates log-uniform at 0 distribution of F1
    def generate_f1_distribution(self, K = K):
        x0 = x_min
        x1 = self.g(1, x0)
        xs = np.exp(np.linspace(np.log(x0), np.log(x1), K+1))
        all_xs = xs
        while xs[-1] < 1 - epsilon:
            xs = self.g(1, xs)
            all_xs = np.concatenate((all_xs, xs))
        return all_xs

    def beliefs(self):
        print('Computing beliefs for ', end="")
        print(self, end="\r")
        kmax = self.strategy_class.kmax
        ps = np.zeros(kmax+1)
        g0_0= self.g(0, x_min)/x_min
        g1_0= self.g(1, x_min)/x_min
        ps[0] = -np.log(g0_0)/(np.log(g1_0)-np.log(g0_0))
        xs = self.generate_f1_distribution()
        for k in range(kmax):
            I1 = np.sum(psi(xs, self.strategy_class.n, k+1))
            I2 = np.sum(psi(xs, self.strategy_class.n, self.strategy_class.n-k-1))
            ps[k+1] = I1/ (I1 +I2)
        self.beliefs = ps
        return ps

    def __repr__(self):
        strategy_class = self.strategy_class.__repr__()[:-1]
        return strategy_class + f'; alpha={self.alpha:.2f})'
    
