import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from numba import njit



#My part

G=6.674e-11
N=10

ν=3.156e10
t=np.linspace(0,10e7*ν,10000)



pos0=np.random.rand(N,3)*1.5*10e20

vel0=np.random.rand(N,3)*5*1000
mass=np.random.rand(N)*70*10e35

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
                        θ=np.sqrt(rji[0]**2 + rji[1]**2 + rji[2]**2)  # Manual norm for numba
                        θ=max(θ, 1e14)  # Collision softening - prevents division by near-zero
                        γ=((G*(mass[j]))/(θ**3))
                        φ=γ*rji
                        
                        acc[i]+=φ
            return np.hstack((vel.flatten(),acc.flatten()))
                         


#Claude's part 

sol = odeint(acc, state0, t)

# Extract positions correctly
positions = sol[:, :3*N].reshape(-1, N, 3)

# Create clean plot
fig = plt.figure(figsize=(10, 8), facecolor='white')
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor('white')

# Generate colors for any N bodies
colors = plt.cm.tab10(np.linspace(0, 1, N))

# Plot trajectories as lines only
for i in range(N):
    # Line trajectory
    ax.plot(positions[:, i, 0], positions[:, i, 1], positions[:, i, 2],
           color=colors[i], linewidth=1.2, alpha=0.8, label=f'Body {i+1}')
    
    # Mark final position only
    ax.scatter(positions[-1, i, 0], positions[-1, i, 1], positions[-1, i, 2],
              color=colors[i], s=200, marker='o', edgecolors='black', 
              linewidths=2, zorder=10)

# Clean styling
ax.set_xlabel('X (m)', fontsize=11)
ax.set_ylabel('Y (m)', fontsize=11)
ax.set_zlabel('Z (m)', fontsize=11)
ax.set_title(f'{N}-Body Gravitational Simulation', fontsize=13, fontweight='bold', pad=15)

# Scientific notation
ax.ticklabel_format(style='scientific', scilimits=(0,0))

# Clean grid
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.legend(fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.show()