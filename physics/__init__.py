"""
Physics Engine Package
======================
Physics-informed analysis modules for trading signal generation.

Solid Math (use these):
  - decision_engine: Particle filter, Bayesian confidence, VOI
  - koopman_sheaf_engine: DMD, sheaf cohomology, RG flow, MDL
  - graph_physics: Laplacian arbitrage, Fiedler regime detection
  - hamiltonian_integrator: Langevin dynamics for price simulation

Heuristic (useful but metaphorical naming):
  - torsion_physics: Autocorrelation, volume profile, range compression
  - physics_orderflow_engine: Order flow → physics pattern matching
"""

# Re-export for convenience
try:
    from .decision_engine import DecisionEngine
except ImportError:
    pass

try:
    from .graph_physics import GraphPhysicsEngine
except ImportError:
    pass

try:
    from .torsion_physics import TorsionPhysicsModel
except ImportError:
    pass
