import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from numba import njit

G = 6.674e-11
N = 2



# Time setup: 100,000 years in seconds
year_in_seconds = 3.156e7  # seconds per year
t_max = 100000000 * year_in_seconds
t = np.linspace(0, t_max, 10000)  # Reduced points for efficiency

# Initial conditions
pos0 = np.random.rand(N, 3) * 1.5e16  # meters
vel0 = np.random.rand(N, 3) * 5e3     # m/s
mass = np.random.rand(N) * 7e31       # kg (solar mass scale)

state0 = np.hstack((pos0.flatten(), vel0.flatten()))

@njit
def derivatives(state, t):
    pos = state[:3*N].reshape(N, 3)
    vel = state[3*N:].reshape(N, 3)
    acc = np.zeros((N, 3))
    
    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = pos[j] - pos[i]
                r_mag = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
                
                # Gravitational acceleration
                acc_mag = G * mass[j] / (r_mag**3)
                acc[i] += acc_mag * r_vec
    
    return np.hstack((vel.flatten(), acc.flatten()))

# Solve ODE
print("Solving N-body problem...")
sol = odeint(derivatives, state0, t)

# Extract positions for each body
positions = sol[:, :3*N].reshape(-1, N, 3)

# Plot trajectories
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

colors = ['r', 'b', 'g']
for i in range(N):
    ax.plot(positions[:, i, 0], positions[:, i, 1], positions[:, i, 2], 
            color=colors[i], label=f'Body {i+1}', linewidth=0.5)
    # Mark starting position
    ax.scatter(positions[0, i, 0], positions[0, i, 1], positions[0, i, 2], 
               color=colors[i], s=100, marker='o')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3-Body Gravitational Simulation')
ax.legend()

plt.tight_layout()
plt.show()