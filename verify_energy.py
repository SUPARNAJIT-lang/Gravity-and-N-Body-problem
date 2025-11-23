import numpy as np
from physics_engine import PhysicsEngine

def calculate_energy(state, masses, G, softening=0.0):
    N = len(masses)
    pos = state[:3*N].reshape(N, 3)
    vel = state[3*N:].reshape(N, 3)
    
    # Kinetic Energy
    ke = 0.5 * np.sum(masses[:, np.newaxis] * vel**2)
    
    # Potential Energy
    pe = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = pos[j] - pos[i]
            r_mag = np.sqrt(np.sum(r_vec**2))
            effective_r = np.sqrt(r_mag**2 + softening**2)
            pe -= (G * masses[i] * masses[j]) / effective_r
            
    return ke + pe

def verify_energy_conservation():
    print("Verifying Energy Conservation...")
    
    # Setup a simple 2-body system (Earth-Sun like)
    G = 6.67430e-11
    m1 = 1.989e30  # Sun
    m2 = 5.972e24  # Earth
    r = 1.496e11   # 1 AU
    v = 29780.0    # Earth velocity
    
    masses = np.array([m1, m2])
    
    pos = np.array([
        [0.0, 0.0, 0.0],       # Sun at origin
        [r, 0.0, 0.0]          # Earth at x=r
    ])
    
    vel = np.array([
        [0.0, 0.0, 0.0],       # Sun stationary
        [0.0, v, 0.0]          # Earth moving in y
    ])
    
    state0 = np.hstack((pos.flatten(), vel.flatten()))
    
    # Simulation parameters
    year = 365.25 * 24 * 3600
    t_span = (0, 5 * year) # 5 years
    t_eval = np.linspace(0, 5 * year, 1000)
    
    engine = PhysicsEngine(units='SI', method='DOP853', rtol=1e-11, atol=1e-11)
    
    sol = engine.run_simulation(state0, t_span, masses, t_eval=t_eval)
    
    # Check energy at each step
    energies = []
    for i in range(len(sol.t)):
        state_i = sol.y[:, i]
        e = calculate_energy(state_i, masses, G)
        energies.append(e)
        
    energies = np.array(energies)
    e0 = energies[0]
    max_deviation = np.max(np.abs((energies - e0) / e0))
    
    print(f"Initial Energy: {e0:.6e} J")
    print(f"Max Relative Energy Deviation: {max_deviation:.6e}")
    
    if max_deviation < 1e-9:
        print("✅ Energy conservation test PASSED (Deviation < 1e-9)")
    else:
        print("❌ Energy conservation test FAILED")

if __name__ == "__main__":
    verify_energy_conservation()
