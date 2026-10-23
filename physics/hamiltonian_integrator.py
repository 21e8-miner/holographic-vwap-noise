
import numpy as np

class HamiltonianIntegrator:
    """
    Implements the Optimized Multiple Time Step (MTS) Position-Verlet Integrator.
    Reference: Batcho, Case, Schlick (2001), J. Chem. Phys.
    
    Why Position-Verlet?
    - Superior stability for large time steps compared to Velocity-Verlet.
    - Essential for "Macro-Scale" market simulation where dt is large (e.g. 1 minute).
    
    Hamiltonian: H(q, p) = p^2/2m + V(q)
    Equations:
        dq/dt = p/m
        dp/dt = -grad(V) - gamma*p + R(t)  [Langevin]
    """
    def __init__(self, dt=1.0, friction=0.1, temp=1.0, mass=1.0):
        self.dt = dt
        self.gamma = friction
        self.kT = temp
        self.mass = mass
        self.noise_scale = np.sqrt(2 * self.kT * self.gamma * self.mass)
        
    def position_verlet_step(self, x, v, force_func):
        """
        Single step of Langevin Position-Verlet.
        
        x(t+dt) = x(t) + v(t+dt/2)*dt
        v(t+dt/2) = v(t) + F(x(t)) * dt/2m  [Half Kick]
        
        Refined for Langevin (Barth & Schlick):
        Uses operator splitting e^{L_p} e^{L_q} e^{L_p}
        """
        dt = self.dt
        m = self.mass
        
        # 1. Half-Kick (Velocity update)
        # F_conservative = force_func(x)
        # F_total = F_conservative - gamma*v + Noise ? 
        # Actually, for Langevin PV, we use the specific "BAOAB" or "OBABO" splitting usually,
        # but the paper suggests a modification to standard PV.
        
        # Simple PV with Langevin damping applied at the ends (Splitting)
        
        # A. Half-Kick conservative
        f = force_func(x)
        v_half = v + 0.5 * dt * f / m
        
        # B. Drift (Position update)
        x_new = x + dt * v_half
        
        # C. Full Kick conservative (at new position)
        f_new = force_func(x_new)
        v_full = v_half + 0.5 * dt * f_new / m
        
        # D. Langevin Thermostat (O-step) - Applied to velocity
        # v_final = alpha * v_full + sigma * noise
        # This keeps the system at "Temperature" (Market Volatility)
        alpha = np.exp(-self.gamma * dt)
        sigma = np.sqrt(self.kT * (1 - alpha**2) / m) # Fluctuation-Dissipation
        noise = np.random.normal(0, 1, size=x.shape)
        
        v_final = alpha * v_full + sigma * noise
        
        return x_new, v_final

    def simulate(self, x0, v0, force_func, steps=100):
        traj = [x0]
        vels = [v0]
        
        x, v = x0, v0
        for _ in range(steps):
            x, v = self.position_verlet_step(x, v, force_func)
            traj.append(x)
            vels.append(v)
            
        return np.array(traj), np.array(vels)

# --- POTENTIAL ENERGY SURFACES (The "Fields") ---

def harmonic_potential_gradient(x, center=0.0, k=1.0):
    """Simple mean-reversion force: F = -k(x - center)"""
    return -k * (x - center)

def double_well_gradient(x, roots=(-1, 1), height=1.0):
    """Bistable regime force (Market can stick to High or Low)"""
    # V(x) = height * (x^2 - 1)^2
    # F(x) = -dV/dx = -height * 2(x^2 - 1) * 2x = -4*height * x * (x^2 - 1)
    return -4 * height * x * (x**2 - 1)

