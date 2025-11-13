import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

#My imports:
from Stardata import pcalc2 as pc2
from Stardata import get_velocity_arrays as gva


AU = 1.496e11

G=6.674e-11
N=3

ν=3.156e7

t_span=(0,80*ν)
t=np.linspace(t_span[0],t_span[1],20000)






m0=1.98847e30 #1 solar-mass
mass=np.array([1.1*m0,0.907*m0,0.122*m0]) 

pos0=pc2()
vel0=gva()
print(vel0)




state0 = np.hstack((pos0.flatten(), vel0.flatten()))


def acc(state,t):
            pos=state[:3*N].reshape(N,3)
            vel=state[3*N:].reshape(N,3)
            acc = np.zeros((N, 3))

            for i in range (N):
                for j in range (N):
                    if i!=j:
                        rji=pos[j]-pos[i]
                        θ=np.linalg.norm(rji)
                        γ=((G*(mass[i]))/(θ**3))
                        φ=γ*rji
                        
                        acc[i]+=φ
            return np.hstack((vel.flatten(),acc.flatten()))
                         



# After your integration:
sol = odeint(acc,state0,t)

# Create two separate plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})

# Plot 1: Alpha Cen A & B ONLY (in AU)
ax1.plot(sol[:, 0]/AU, sol[:, 1]/AU, sol[:, 2]/AU, 'orange', label='Alpha Cen A')
ax1.plot(sol[:, 3]/AU, sol[:, 4]/AU, sol[:, 5]/AU, 'blue', label='Alpha Cen B')
ax1.set_xlabel('X (AU)')
ax1.set_ylabel('Y (AU)')
ax1.set_zlabel('Z (AU)')
ax1.set_title('Alpha Cen A & B (0 years)')
ax1.legend()

# Plot 2: All three stars
ax2.plot(sol[:, 0]/AU, sol[:, 1]/AU, sol[:, 2]/AU, 'orange', label='Alpha Cen A', linewidth=0.5)
ax2.plot(sol[:, 3]/AU, sol[:, 4]/AU, sol[:, 5]/AU, 'blue', label='Alpha Cen B', linewidth=0.5)
ax2.plot(sol[:, 6]/AU, sol[:, 7]/AU, sol[:, 8]/AU, 'red', label='Proxima', linewidth=2)
ax2.set_xlabel('X (AU)')
ax2.set_ylabel('Y (AU)')
ax2.set_zlabel('Z (AU)')
ax2.set_title('All three stars (Proxima scale)')
ax2.legend()

plt.tight_layout()
plt.show()

