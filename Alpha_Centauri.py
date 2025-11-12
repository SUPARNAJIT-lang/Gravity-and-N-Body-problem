import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from numba import njit
#My imports:
from Stardata import pcalc2 as pc2
from Stardata import get_velocity_arrays as gva

m0=1.98847e+30 #1 solar-mass
mass=np.array([1.1,0.907,0.122])*m0

#pos0= START FROM HERE
#vel0= START FROM HERE









G=6.674e-11
N=3

ν=3.156*10**10
t=np.linspace(0,100*ν,100000)


pos0=np.random.rand(N,3)*1.5*10e16
vel0=np.random.rand(N,3)*5*1000
mass=np.random.rand(N)*70*10e30

state0 = np.hstack((pos0.flatten(), vel0.flatten()))

@njit
def acc(state,t):
            pos=state[:3*N].reshape(N,3)
            vel=state[3*N:].reshape(N,3)
            acc = np.zeros((N, 3))

            for i in range (N):
                for j in range (N):
                    if i!=j:
                        rji=pos[j]-pos[i]
                        θ=np.linalg.norm(rji)
                        γ=((G*(mass[j]))/(θ**3))
                        φ=γ*rji
                        
                        acc[i]+=φ
            return np.hstack((vel.flatten(),acc.flatten()))
                         


sol=odeint(acc,state0,t)
            

fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")

ax.plot(sol.T[0],sol.T[1],sol.T[2])
ax.plot(sol.T[3],sol.T[4],sol.T[5])
ax.plot(sol.T[6],sol.T[7],sol.T[8])

plt.show()

