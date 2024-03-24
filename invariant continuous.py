import numpy as np
import matplotlib.pyplot as plt

#key constants
niterations=100      #number of iterations
N = 10000   # grid of points
n = 5       # sample size
lam=0.2     # lambda
y=0.7      # parameter of the function

#Dynamics 
gin=[]
gin.append(lambda x: (x/y)**(1/n))
gin.append(lambda x: 1 - gin[0](1-x))

#Initialize
domain=np.linspace(0, 1, num=N)
F=[]
ga=[]
for i in range(2):
    F.append(np.linspace(0, 1, num=N))
    a=np.array(gin[i](domain))
    a[a>1]=1
    a[a<0]=0
    ga.append(a)
colors=['b','r']

#plot
plt.ion()
f = plt.figure(figsize=(15,5), dpi=100)
pgdynamics=f.add_subplot(1,3,1)
pcdf=f.add_subplot(1,3,2)
pratio=f.add_subplot(1,3,3)

pgdynamics.plot(domain, domain, '--','g')
for i in range(2):
    pgdynamics.plot(domain, ga[i], colors[i], label='g'+str(i)+' inverse')
    pcdf.plot(domain, F[i], colors[i], label='F'+str(i))
    pratio.plot(domain, F[1]/F[0])
#plt.xlabel('CDF')
#plt.ylabel('population fraction')
#plt.title("Invariant distributions")
f.canvas.draw()


#Iterate
for count in range(niterations):
    Ftemp=[]
    for i in range(2):
        Ftemp.append(np.zeros(N))
        for t in range(N):
            kt = gin[i](t/(N-1)) * (N-1)
            k1 = max(0, min(int(kt), N-1))
            k2 = max(0, min(k1+1, N-1))
            alpha = max(0,min(k1+1-kt,1))
            Ftemp[i][t] = alpha* F[i][k1] + (1-alpha)* F[i][k2]
        Ftemp[i][0]=0
        Ftemp[i][N-1]=1

    pcdf.cla()
    pratio.cla()
    for i in range(2):
        F[i]=(1-lam) * Ftemp[i] + lam* Ftemp[1-i]
        pcdf.plot(domain, F[i], colors[i], label='F'+str(i))
    pratio.plot(domain, F[1]/F[0])
    f.canvas.draw()
    print(count)




