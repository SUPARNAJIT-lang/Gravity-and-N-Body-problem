import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#My imports:

from Stardata import pcalc2 as pc2
from Stardata import get_velocity_arrays as gva
from Stardata import v_relative as vr



AU = 1.496e11

G=6.674e-11
N=3

ν=3.156e7

t_span=(0,80*ν)
t=np.linspace(t_span[0],t_span[1],20000)






m0=1.98847e30 #1 solar-mass
mass=np.array([1.1*m0,0.907*m0,0.122*m0]) 

pos00=pc2()
vel00=gva()
print(vel00)

vel01=vr()


state00 = np.hstack((pos00.flatten(), vel00.flatten()))
state01=np.hstack((pos00.flatten(), vel01.flatten()))


def acc(t,state):
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
                         



# After your integration:
sol = solve_ivp(acc,t_span,state00,method='RK45',t_eval=t)
print(sol)

sol0=solve_ivp(acc,t_span,state01,method='RK45',t_eval=t)
print(sol0)



fig = plt.figure(figsize=(14, 12))

# --- 1️⃣ Alpha Cen A & B (original sol)
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot(sol.y[0, :]/AU, sol.y[1, :]/AU, sol.y[2, :]/AU, 'orange', label='Alpha Cen A')
ax1.plot(sol.y[3, :]/AU, sol.y[4, :]/AU, sol.y[5, :]/AU, 'blue', label='Alpha Cen B')
ax1.set_xlabel('X (AU)')
ax1.set_ylabel('Y (AU)')
ax1.set_zlabel('Z (AU)')
ax1.set_title('Alpha Cen A & B (original)')
ax1.legend()

# --- 2️⃣ All three (original sol)
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot(sol.y[0, :]/AU, sol.y[1, :]/AU, sol.y[2, :]/AU, 'orange', linewidth=0.5, label='Alpha Cen A')
ax2.plot(sol.y[3, :]/AU, sol.y[4, :]/AU, sol.y[5, :]/AU, 'blue', linewidth=0.5, label='Alpha Cen B')
ax2.plot(sol.y[6, :]/AU, sol.y[7, :]/AU, sol.y[8, :]/AU, 'red', linewidth=1.5, label='Proxima')
ax2.set_xlabel('X (AU)')
ax2.set_ylabel('Y (AU)')
ax2.set_zlabel('Z (AU)')
ax2.set_title('All three stars (original)')
ax2.legend()

# --- 3️⃣ Alpha Cen A & B (relative velocities)
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.plot(sol0.y[0, :]/AU, sol0.y[1, :]/AU, sol0.y[2, :]/AU, 'orange', label='Alpha Cen A')
ax3.plot(sol0.y[3, :]/AU, sol0.y[4, :]/AU, sol0.y[5, :]/AU, 'blue', label='Alpha Cen B')
ax3.set_xlabel('X (AU)')
ax3.set_ylabel('Y (AU)')
ax3.set_zlabel('Z (AU)')
ax3.set_title('Alpha Cen A & B (relative)')
ax3.legend()

# --- 4️⃣ All three (relative velocities)
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.plot(sol0.y[0, :]/AU, sol0.y[1, :]/AU, sol0.y[2, :]/AU, 'orange', linewidth=0.5, label='Alpha Cen A')
ax4.plot(sol0.y[3, :]/AU, sol0.y[4, :]/AU, sol0.y[5, :]/AU, 'blue', linewidth=0.5, label='Alpha Cen B')
ax4.plot(sol0.y[6, :]/AU, sol0.y[7, :]/AU, sol0.y[8, :]/AU, 'red', linewidth=1.5, label='Proxima')
ax4.set_xlabel('X (AU)')
ax4.set_ylabel('Y (AU)')
ax4.set_zlabel('Z (AU)')
ax4.set_title('All three stars (relative)')
ax4.legend()

plt.tight_layout()
plt.show()



