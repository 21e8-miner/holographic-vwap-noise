"""
================================================================================
KOOPMAN-SHEAF RENORMALIZED GENERATOR
================================================================================
Abstract Math Kernels for FMTR v3.0:

1. Koopman Spectral Decomposition (complex phase eigenfunctions)
2. Sheaf Coboundary Energy (local vs global consistency obstruction)
3. RG Correlation Length ℓ(x) (structure persistence depth)
4. Information Geometry KL Divergence (regime transition detector)
5. MDL Scaffold Selection (grammar competition)
6. Boltzmann Constrained Decoder (stat-mech objective)

This module provides the mathematical backbone for the
"Constrained Renormalized Complex Option Generator" architecture.
================================================================================
"""

import numpy as np
# import torch  # REMOVED: Using pure NumPy for compatibility
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from numpy.fft import fft, fftfreq, ifft  # Use numpy instead of scipy

# PHYSICS INTEGRATION (Batcho-Case-Schlick 2001)
try:
    from hamiltonian_integrator import HamiltonianIntegrator
    HAMILTONIAN_AVAILABLE = True
except ImportError:
    HAMILTONIAN_AVAILABLE = False


# ============================================================================
# 1. KOOPMAN SPECTRAL DECOMPOSITION
# ============================================================================
class KoopmanSpectralEngine:
    """
    Approximates Koopman eigenfunctions from time series.
    
    The Koopman operator K acts on observables g:
        (Kg)(x) = g(F(x))
    
    Eigenfunctions ψ_j satisfy: Kψ_j = λ_j ψ_j
    When λ_j ≈ e^{iω_j Δt}, the eigenfunction carries rotating complex phase.
    """
    
    def __init__(self, n_modes: int = 5, device='cpu', fiedler_weight: float = 0.7):
        self.n_modes = n_modes
        self.device = device
        self.fiedler_weight = fiedler_weight  # 0 = no Fiedler weighting, 1 = full Fiedler penalty
        
    @staticmethod
    def compute_optimal_ridge_lambda(spectrum: np.ndarray, n_samples: int, noise_var: float = 1.0) -> float:
        """
        Computes the optimal ridge regularization λ⋆ using Dimension-Free Ridge Regression theory.
        
        Implements arXiv:2210.08571v3 (Cheng & Montanari, 2025).
        
        For polynomially varying spectrum σₖ ~ k^(-α), the optimal λ⋆ = c⋆ * σₙ where c⋆ solves:
            1 = ν/c⋆ + (π/α)/sin(π/α) * c⋆^(-1/α)
        
        This provides mathematically optimal bias-variance tradeoff for spectral forecasting.
        
        Args:
            spectrum: array of eigenvalues/singular values (sorted descending)
            n_samples: number of samples (time points)
            noise_var: noise variance τ²
            
        Returns:
            lambda_star: optimal regularization parameter
        """
        if len(spectrum) < 2:
            return 1.0
            
        # Estimate spectral decay exponent α from log-log regression
        k_vals = np.arange(1, len(spectrum) + 1)
        log_k = np.log(k_vals + 1e-9)
        log_sigma = np.log(np.abs(spectrum) + 1e-9)
        
        # Linear regression: log(σ) = -α*log(k) + const
        # α = -slope
        if len(log_k) >= 2:
            slope = np.polyfit(log_k, log_sigma, 1)[0]
            alpha = max(1.01, -slope)  # α > 1 required for convergence
        else:
            alpha = 2.0  # Default
            
        # Solve for c⋆ using bisection (Theorem 5, Case 1)
        # Fixed-point equation: 1 = ν/c + (π/α)/sin(π/α) * c^(-1/α)
        # We use ν = 0.5 (balanced regularization)
        nu = 0.5
        
        def fixed_point_residual(c):
            if c <= 0:
                return float('inf')
            try:
                return nu / c + (np.pi / alpha) / np.sin(np.pi / alpha) * (c ** (-1.0 / alpha)) - 1.0
            except:
                return float('inf')
        
        # Bisection search for c⋆
        c_low, c_high = 0.01, 100.0
        for _ in range(50):
            c_mid = (c_low + c_high) / 2
            if fixed_point_residual(c_mid) > 0:
                c_low = c_mid
            else:
                c_high = c_mid
        c_star = (c_low + c_high) / 2
        
        # λ⋆ = c⋆ * σₙ (Eq. 34 in paper)
        sigma_n = spectrum[min(n_samples - 1, len(spectrum) - 1)] if n_samples <= len(spectrum) else spectrum[-1]
        lambda_star = c_star * abs(sigma_n)
        
        return max(1e-6, lambda_star)
        
    def compute_fiedler_weights(self, N: int, freq_indices: np.ndarray) -> np.ndarray:
        """
        Compute Fiedler alignment weights for each frequency mode.
        
        For a path graph of length N, the k-th Laplacian eigenvector is:
            φ_k(n) = cos(πk(n + 0.5) / N)
        
        The Fiedler vector (k=1) is the fundamental structural mode.
        Higher k values represent increasingly "noisy" high-frequency structure.
        
        We weight modes by: w_j = exp(-α * k_j / N)
        where k_j is the effective Fiedler order of mode j.
        
        This implements the insight from arXiv:2510.26745 that geometric memory
        aligns with Fiedler vectors, and non-Fiedler components are "adulteration".
        
        Returns:
            weights: array of shape (len(freq_indices),) with values in (0, 1]
        """
        if N < 2:
            return np.ones(len(freq_indices))
        
        # Map FFT frequency indices to effective Fiedler order
        # FFT freq[j] = j/N for j < N/2, maps to Fiedler k ≈ |j|
        fiedler_order = np.abs(freq_indices).astype(float)
        
        # Normalize by N so that k/N is the "spectral position"
        normalized_order = fiedler_order / N
        
        # Exponential decay weight: fundamental modes (low k) get weight ≈ 1
        # High-frequency modes (k → N/2) get exponentially suppressed
        # α = 4.0 gives reasonable decay: k=N/4 has weight ≈ 0.37, k=N/2 has weight ≈ 0.14
        alpha = 4.0
        weights = np.exp(-alpha * normalized_order)
        
        # Ensure at least some weight for all modes (floor at 0.1)
        weights = np.maximum(weights, 0.1)
        
        return weights
        
    def extract_koopman_modes(self, time_series: List[float]) -> Dict:
        """
        Extract dominant Koopman modes via DMD-like spectral decomposition.
        
        QC ENHANCED: Added eigenvalue stability analysis.
        - |λ| < 1: Stable (decaying) mode
        - |λ| = 1: Neutral (oscillating) mode
        - |λ| > 1: Unstable (growing) mode
        
        Returns:
            - eigenvalues λ_j (complex)
            - frequencies ω_j
            - mode amplitudes ρ_j
            - phases φ_j
            - stability: list of 'STABLE', 'NEUTRAL', 'UNSTABLE'
        """
        if len(time_series) < 10:
            return {'eigenvalues': [], 'frequencies': [], 'amplitudes': [], 'phases': [], 'stability': [], 'koopman_coherence': 0, 'dominant_frequency': 0}
        
        x = np.array(time_series, dtype=np.float64)
        
        # FFT-based spectral decomposition (DMD approximation)
        N = len(x)
        spectrum = fft(x - np.mean(x))
        freqs = fftfreq(N)
        
        # Extract top n_modes by amplitude
        amplitudes = np.abs(spectrum)
        indices = np.argsort(amplitudes)[::-1][:self.n_modes]
        
        # Compute Fiedler alignment weights (arXiv:2510.26745)
        fiedler_weights = self.compute_fiedler_weights(N, indices)
        
        eigenvalues = []
        frequencies = []
        mode_amplitudes = []
        phases = []
        stability = []
        
        for idx in indices:
            omega = 2 * np.pi * freqs[idx]
            
            # QC FIX: Compute eigenvalue with decay/growth from amplitude trend
            # For pure FFT, we approximate stability from local amplitude ratio
            if idx > 0 and idx < N - 1:
                # Local amplitude ratio gives decay information
                local_amp = amplitudes[idx]
                neighbor_avg = (amplitudes[idx-1] + amplitudes[idx+1]) / 2
                decay_factor = min(1.2, max(0.8, neighbor_avg / (local_amp + 1e-9)))
            else:
                decay_factor = 1.0
            
            # Koopman eigenvalue: λ = r * e^(iω) where r is magnitude
            # STABILITY FIX: Force r <= 1.0 to prevent exponential explosion (Divergence Guard)
            lam = min(1.0, decay_factor) * np.exp(1j * omega)
            
            eigenvalues.append(complex(lam))
            frequencies.append(float(omega))
            mode_amplitudes.append(float(amplitudes[idx] / N))
            phases.append(float(np.angle(spectrum[idx])))
            
            # Classify stability
            mag = abs(lam)
            if mag < 0.99:
                stability.append('STABLE')
            elif mag > 1.01:
                stability.append('UNSTABLE')
            else:
                stability.append('NEUTRAL')
        
        # Compute "Koopman coherence" - how well dominated by few modes
        total_energy = np.sum(amplitudes**2)
        top_energy = np.sum(amplitudes[indices]**2)
        koopman_coherence = top_energy / (total_energy + 1e-9)
        
        return {
            'eigenvalues': eigenvalues,
            'frequencies': frequencies,
            'amplitudes': mode_amplitudes,
            'phases': phases,
            'stability': stability,
            'fiedler_weights': fiedler_weights.tolist(),  # NEW: Fiedler alignment weights
            'koopman_coherence': koopman_coherence,
            'dominant_frequency': frequencies[0] if frequencies else 0.0
        }


    def generate_koopman_forecast(self, time_series: List[float], steps: int = 10) -> Dict:
        """
        Generate forecast using Koopman eigenvalue propagation.
        
        Theory: x̂(t+k) = Σⱼ wⱼ · φⱼ · λⱼᵏ · ψⱼ(x_t)
        
        Where:
            - λⱼ are Koopman eigenvalues (on unit circle)
            - φⱼ are mode amplitudes
            - ψⱼ are phases
            - wⱼ are Fiedler alignment weights (arXiv:2510.26745)
        
        The Fiedler weights implement a soft penalty on non-fundamental modes,
        preferring geometric memory over associative/noisy components.
        
        Returns:
            - forecast: list of predicted price deltas
            - confidence: based on Koopman coherence
            - fiedler_purity: how Fiedler-aligned the forecast is
        """
        modes = self.extract_koopman_modes(time_series)
        
        if not modes['eigenvalues'] or modes['koopman_coherence'] < 0.1:
            # Not enough structure - return zero forecast
            return {
                'forecast': [0.0] * steps,
                'confidence': 0.0,
                'dominant_frequency': 0.0,
                'fiedler_purity': 0.0
            }
        
        # Get Fiedler weights
        fiedler_weights = np.array(modes.get('fiedler_weights', [1.0] * len(modes['eigenvalues'])))
        
        # --- DIMENSION-FREE RIDGE REGRESSION (arXiv:2210.08571) ---
        # Compute optimal regularization λ⋆ from spectral decay characteristics
        mode_amplitudes_arr = np.array(modes['amplitudes'])
        if len(mode_amplitudes_arr) >= 2:
            lambda_star = self.compute_optimal_ridge_lambda(
                spectrum=mode_amplitudes_arr,
                n_samples=len(time_series)
            )
            # Apply ridge regularization to mode weights (Eq. 8-9: Effective Bias/Variance)
            # w_k = σ_k / (σ_k + λ⋆) - shrinks noisy high-frequency modes
            ridge_weights = mode_amplitudes_arr / (mode_amplitudes_arr + lambda_star + 1e-9)
        else:
            ridge_weights = np.ones(len(mode_amplitudes_arr))
        
        # Blend: Fiedler (geometric) + Ridge (optimal regularization)
        # Combined weighting: Fiedler ensures geometric alignment, Ridge ensures optimal shrinkage
        base_weights = ridge_weights * ((1 - self.fiedler_weight) + self.fiedler_weight * fiedler_weights)
        
        # --- mHC: MANIFOLD-CONSTRAINED MODAL INTERACTION ---
        # Construct interaction matrix: Diag(weights) + small crosstalk
        # This represents the 'Hyper-Connection' betweeen spectral modes.
        n = len(base_weights)
        crosstalk = 0.05  # Slight energy mixing
        
        # Log-space construction for Sinkhorn stability
        # M_ij = weight_i if i==j else crosstalk
        interaction_matrix = np.full((n, n), np.log(crosstalk + 1e-9))
        np.fill_diagonal(interaction_matrix, np.log(base_weights + 1e-9))
        
        # Project onto Doubly Stochastic Manifold (Birkhoff Polytope)
        doubly_stochastic_modes = self.sinkhorn_knopp_projection(interaction_matrix)
        
        # Extract the new stabilized diagonal weights (Self-Loop Energy)
        # We scale by n to recover magnitude (since Sinkhorn averages to 1/n)
        effective_weights = np.diag(doubly_stochastic_modes) * n
        
        # Compute Fiedler purity with NEW weights
        weighted_energy = np.sum(np.array(modes['amplitudes']) ** 2 * effective_weights)
        total_energy = np.sum(np.array(modes['amplitudes']) ** 2) + 1e-9
        fiedler_purity = float(weighted_energy / total_energy)
        
        forecast = []
        for k in range(1, steps + 1):
            predicted_delta = 0.0
            
            # Sum over all modes: wⱼ · φⱼ · λⱼᵏ · exp(i·ψⱼ)
            for i, (lam, amp, phase) in enumerate(zip(
                modes['eigenvalues'], 
                modes['amplitudes'], 
                modes['phases']
            )):
                # Apply mHC-stabilized weight
                w = effective_weights[i] if i < len(effective_weights) else 1.0
                
                # Propagate eigenvalue k steps forward
                lam_k = lam ** k
                
                # Mode contribution (real part only for price)
                mode_val = w * amp * np.real(lam_k * np.exp(1j * phase))
                predicted_delta += mode_val
            
            forecast.append(float(predicted_delta))
        
        return {
            'forecast': forecast,
            'confidence': float(modes['koopman_coherence']),
            'dominant_frequency': float(modes['dominant_frequency']),
            'fiedler_purity': fiedler_purity
        }

    def generate_hamiltonian_forecast(self, time_series: List[float], steps: int = 10, temp_scale: float = 1.0) -> Dict:
        """
        Generate forecast using Langevin Position-Verlet Dynamics (Physics Simulation).
        
        Instead of projecting λ^k (Kinematic), we simulate a particle with mass m
        moving through a potential V(x) defined by the Koopman Mode structure.
        
        Benefits (from Batcho/Schlick Paper):
        - Stable at larger time steps (Position-Verlet).
        - Samples probability distribution via Langevin Noise (not just point estimate).
        - Captures "Momentum" of the market (Inertia).
        """
        if not HAMILTONIAN_AVAILABLE:
            return self.generate_koopman_forecast(time_series, steps)
            
        # 1. Get Linear Baseline (The "Attractor")
        linear_forecast = self.generate_koopman_forecast(time_series, steps)
        path_attractor = np.array(linear_forecast['forecast'])
        
        # 2. Physics Setup
        # Current state
        x0 = np.array([0.0]) # Delta relative to now
        
        # Initial velocity (Momentum) estimated from last few points
        if len(time_series) >= 2:
            v0 = np.array([time_series[-1] - time_series[-2]])
        else:
            v0 = np.array([0.0])
            
        # Define Potential Force: Pulls towards the Koopman path
        # V(x, t) = k/2 * (x - x_koopman(t))^2
        # F(x, t) = -k * (x - x_koopman(t))
        # This models the Spectral solution as the "low energy valley"
        
        # We need a simulation loop that steps through time, updating the force target
        # Mass = 2.0 (Heavy market)
        integrator = HamiltonianIntegrator(dt=1.0, friction=0.5, temp=temp_scale, mass=2.0)
        
        sim_traj = []
        x, v = x0, v0
        
        for t in range(steps):
            # Target for this step
            target = path_attractor[t] if t < len(path_attractor) else path_attractor[-1]
            
            # Force Function closure for this step
            def current_force(pos):
                k = 0.8 # Spring constant (Mean Reversion strength to Spectral Path)
                return -k * (pos - target)
            
            x, v = integrator.position_verlet_step(x, v, current_force)
            sim_traj.append(float(x[0]))
            
        return {
            'forecast': sim_traj,
            'confidence': linear_forecast['confidence'], # Inherit coherence
            'dominant_frequency': linear_forecast['dominant_frequency'],
            'physics_temp': temp_scale,
            'method': 'Langevin Position-Verlet'
        }


    @staticmethod
    def hilbert_transform(signal: np.ndarray) -> np.ndarray:
        """
        Compute analytic signal using Hilbert Transform.
        
        The analytic signal Z = x + i*H(x) where H is the Hilbert transform.
        This gives us instantaneous phase and amplitude.
        
        Implementation via FFT (equivalent to scipy.signal.hilbert):
        1. FFT the signal
        2. Zero out negative frequencies
        3. Double positive frequencies
        4. IFFT
        """
        N = len(signal)
        spectrum = fft(signal)
        
        # Create the frequency multiplier
        h = np.zeros(N)
        if N % 2 == 0:  # Even length
            h[0] = 1
            h[1:N//2] = 2
            h[N//2] = 1
        else:  # Odd length
            h[0] = 1
            h[1:(N+1)//2] = 2
        
        analytic = ifft(spectrum * h)
        return analytic

    @staticmethod
    def sinkhorn_knopp_projection(log_matrix: np.ndarray, iterations: int = 20) -> np.ndarray:
        """
        Projects a matrix onto the Birkhoff polytope (Doubly Stochastic Manifold).
        
        Implements mHC (Manifold-Constrained Hyper-Connections) from arXiv:2512.24880.
        
        Algorithm:
        1. Exponentiate: M = exp(log_matrix) to ensure positivity
        2. Iterative Normalization: Alternate row and column scaling
        3. Converges to M where rows and cols sum to 1.
        
        This restores the Identity Mapping property in deep residual streams (energy conservation).
        Used for stabilizing multi-stream spectral recombination.
        """
        # 1. Start in positive domain
        M = np.exp(log_matrix)
        
        # 2. Sinkhorn iterations
        r = np.ones(M.shape[0])
        c = np.ones(M.shape[1])
        
        for _ in range(iterations):
            # Row normalization
            r = 1.0 / (M @ c + 1e-9)
            # Column normalization
            c = 1.0 / (r @ M + 1e-9)
            
        # 3. Construct doubly stochastic matrix
        # P = diag(r) * M * diag(c)
        P = (M * r[:, None]) * c[None, :]
        
        return P



# ============================================================================
# 2. SHEAF COBOUNDARY ENERGY (Global Consistency Obstruction)
# ============================================================================
class SheafConsistencyEngine:
    """
    Implements sheaf-theoretic gluing across overlapping windows.
    
    Cover the timeline by overlapping windows U_i.
    On each window, compute local structure label s_i.
    On overlaps U_i ∩ U_j, restriction maps must agree.
    
    The obstruction energy E_sheaf = ||δs||² measures how locally-consistent
    pieces refuse to glue into a global structure.
    
    Non-consensus insight: regime shifts become TOPOLOGICAL OBSTRUCTIONS
    to gluing structure across scales.
    """
    
    def __init__(self, window_size: int = 20, overlap: int = 10):
        self.window_size = window_size
        self.overlap = overlap
        
    def compute_local_parses(self, symbols: List[str]) -> List[Dict]:
        """Compute local structure on each overlapping window."""
        parses = []
        step = self.window_size - self.overlap
        
        for i in range(0, len(symbols) - self.window_size + 1, step):
            window = symbols[i:i + self.window_size]
            
            # Local structure: deflation depth + L/S ratio
            l_count = window.count('L')
            s_count = window.count('S')
            ratio = l_count / (l_count + s_count + 1e-9)
            
            # Try local deflation
            depth = self._deflation_depth(window)
            
            parses.append({
                'start': i,
                'end': i + self.window_size,
                'ratio': ratio,
                'depth': depth,
                'window': window
            })
        
        return parses
    
    def _deflation_depth(self, seq: List[str]) -> int:
        """Count how many times we can deflate the sequence."""
        def deflate(s):
            result = []
            i = 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == 'L' and s[i+1] == 'S':
                    result.append('L')
                    i += 2
                elif s[i] == 'L':
                    result.append('S')
                    i += 1
                else:
                    return None
            return result
        
        current = seq
        depth = 0
        for _ in range(20):
            deflated = deflate(current)
            if deflated is None or len(deflated) == 0:
                break
            depth += 1
            current = deflated
        return depth
    
    def compute_coboundary_energy(self, parses: List[Dict]) -> Dict:
        """
        Compute Čech coboundary norm ||δs||².
        
        This measures disagreement on overlaps - the obstruction to gluing.
        """
        if len(parses) < 2:
            return {'energy': 0.0, 'disagreements': [], 'gluing_possible': True}
        
        disagreements = []
        total_energy = 0.0
        
        for i in range(len(parses) - 1):
            p1, p2 = parses[i], parses[i+1]
            
            # Restriction to overlap: compare structure on shared region
            overlap_start = p2['start']
            overlap_end = p1['end']
            
            if overlap_end > overlap_start:
                # Structure disagreement: ratio difference + depth difference
                ratio_diff = abs(p1['ratio'] - p2['ratio'])
                depth_diff = abs(p1['depth'] - p2['depth'])
                
                # Weighted coboundary contribution
                local_energy = ratio_diff**2 + (depth_diff / 10.0)**2
                total_energy += local_energy
                
                if local_energy > 0.1:  # Significant disagreement
                    disagreements.append({
                        'location': overlap_start,
                        'energy': local_energy,
                        'type': 'STRUCTURE_FRACTURE' if depth_diff > 2 else 'RATIO_SHIFT'
                    })
        
        # Normalize by number of overlaps
        normalized_energy = total_energy / max(1, len(parses) - 1)
        
        return {
            'energy': normalized_energy,
            'raw_energy': total_energy,
            'disagreements': disagreements,
            'gluing_possible': len(disagreements) == 0,
            'fracture_count': len(disagreements)
        }
    
    def compute_sheaf_obstruction(self, time_series: List[float]) -> Dict:
        """Full sheaf analysis pipeline."""
        # Symbolize
        symbols = []
        for i in range(1, len(time_series)):
            symbols.append('L' if time_series[i] > time_series[i-1] else 'S')
        
        if len(symbols) < self.window_size:
            return {'energy': 0.0, 'gluing_possible': True, 'regime': 'INSUFFICIENT_DATA'}
        
        parses = self.compute_local_parses(symbols)
        coboundary = self.compute_coboundary_energy(parses)
        
        # Regime classification based on sheaf structure
        if coboundary['energy'] < 0.05 and coboundary['gluing_possible']:
            regime = 'GLOBALLY_CONSISTENT'  # Perfect structure
        elif coboundary['energy'] < 0.2:
            regime = 'LOCAL_STRUCTURE'  # Structure exists locally but with mild obstruction
        elif coboundary['energy'] < 0.5:
            regime = 'STRUCTURAL_FRACTURE'  # Significant gluing failure
        else:
            regime = 'NOISE'  # No consistent structure
        
        coboundary['regime'] = regime
        return coboundary


# ============================================================================
# 3. RG CORRELATION LENGTH ℓ(x)
# ============================================================================
class RGCorrelationEngine:
    """
    Computes the RG correlation length: how far structure persists across scales.
    
    ℓ(x) = max{k : D^k(x) exists}
    
    This is the "largest scale at which deflation remains well-defined."
    """
    
    def compute_correlation_length(self, time_series: List[float]) -> Dict:
        """
        Compute multi-scale correlation length via iterative deflation.
        """
        # Symbolize
        symbols = []
        for i in range(1, len(time_series)):
            symbols.append('L' if time_series[i] > time_series[i-1] else 'S')
        
        if len(symbols) < 5:
            return {'correlation_length': 0, 'max_depth': 0, 'scale_coherence': []}
        
        # Track coherence at each deflation level
        current = symbols
        scale_coherence = []
        max_depth = 0
        
        for k in range(20):
            # Attempt deflation
            deflated = self._deflate(current)
            if deflated is None or len(deflated) < 2:
                break
            
            # Measure coherence at this scale
            l_ratio = deflated.count('L') / (len(deflated) + 1e-9)
            coherence = 4 * l_ratio * (1 - l_ratio)  # Variance-like measure (0 at extremes)
            scale_coherence.append({
                'level': k + 1,
                'length': len(deflated),
                'coherence': 1 - coherence  # Invert so 1 = pure structure
            })
            
            max_depth = k + 1
            current = deflated
        
        # Correlation length = weighted average of depths where coherence > 0.5
        weighted_sum = sum(sc['level'] * sc['coherence'] for sc in scale_coherence)
        weight_total = sum(sc['coherence'] for sc in scale_coherence) + 1e-9
        correlation_length = weighted_sum / weight_total
        
        return {
            'correlation_length': correlation_length,
            'max_depth': max_depth,
            'scale_coherence': scale_coherence,
            'is_fixed_point': max_depth >= 5  # Approaches fixed point
        }
    
    def _deflate(self, seq: List[str]) -> Optional[List[str]]:
        """Fibonacci deflation: LS -> L, L -> S"""
        result = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == 'L' and seq[i+1] == 'S':
                result.append('L')
                i += 2
            elif seq[i] == 'L':
                result.append('S')
                i += 1
            else:
                return None
        return result


# ============================================================================
# 4. INFORMATION GEOMETRY (KL Divergence for Regime Transitions)
# ============================================================================
class InfoGeometryEngine:
    """
    Tracks movement on the statistical manifold of scale-posteriors.
    
    d(t) = KL(q(t) || q(t-1))
    
    Spikes in d(t) are STRUCTURAL DISPLACEMENTS, not just volatility.
    """
    
    def __init__(self):
        self.posterior_history = []  # Track q(t) over time
        
    def update_posterior(self, q_vector: Dict[str, float]) -> Dict:
        """
        Add new posterior to history and compute KL divergence.
        
        q_vector: Dict mapping scale names to posterior means
        """
        # Convert to array for computation
        scales = sorted(q_vector.keys())
        q_new = np.array([q_vector[s] for s in scales]) + 1e-9
        q_new = q_new / q_new.sum()  # Normalize
        
        kl_divergence = 0.0
        structural_displacement = False
        
        if len(self.posterior_history) > 0:
            q_old = self.posterior_history[-1]
            
            # KL(q_new || q_old)
            kl_divergence = np.sum(q_new * np.log(q_new / (q_old + 1e-9)))
            
            # Threshold for "structural displacement"
            structural_displacement = kl_divergence > 0.1
        
        self.posterior_history.append(q_new)
        if len(self.posterior_history) > 100:
            self.posterior_history.pop(0)
        
        return {
            'kl_divergence': float(kl_divergence),
            'structural_displacement': structural_displacement,
            'manifold_velocity': float(kl_divergence)  # Rate of movement on manifold
        }


# ============================================================================
# 5. MDL SCAFFOLD SELECTION
# ============================================================================
class MDLScaffoldSelector:
    """
    Minimum Description Length selection between candidate grammars.
    
    M* = argmin_M [L(M) + L(x|M)]
    
    This turns "structure hunting" into "find the shortest program."
    """
    
    def __init__(self):
        # Candidate scaffolds with their complexity costs
        self.scaffolds = {
            'FIBONACCI': {'rule': 'LS->L, L->S', 'complexity': 2},
            'TRIBONACCI': {'rule': 'LLS->L, LS->L, L->S', 'complexity': 3},
            'BINARY_RUN': {'rule': 'alternating runs', 'complexity': 1},
            'RANDOM': {'rule': 'no structure', 'complexity': 0}
        }
    
    def select_scaffold(self, time_series: List[float]) -> Dict:
        """
        Select the MDL-optimal scaffold for the observed data.
        """
        symbols = []
        for i in range(1, len(time_series)):
            symbols.append('L' if time_series[i] > time_series[i-1] else 'S')
        
        if len(symbols) < 10:
            return {'scaffold': 'INSUFFICIENT_DATA', 'mdl_score': float('inf')}
        
        scores = {}
        
        for name, scaffold in self.scaffolds.items():
            # L(M): model complexity
            model_cost = scaffold['complexity']
            
            # L(x|M): data cost given model
            data_cost = self._compute_data_cost(symbols, name)
            
            scores[name] = model_cost + data_cost
        
        # Select minimum
        best = min(scores, key=scores.get)
        
        return {
            'scaffold': best,
            'mdl_score': scores[best],
            'all_scores': scores,
            'confidence': 1 - (scores[best] / (max(scores.values()) + 1e-9))
        }
    
    def _compute_data_cost(self, symbols: List[str], scaffold_name: str) -> float:
        """Compute bits needed to encode data under scaffold."""
        n = len(symbols)
        
        if scaffold_name == 'RANDOM':
            # Entropy cost: n * H(p) where p = L ratio
            l_ratio = symbols.count('L') / n
            if l_ratio == 0 or l_ratio == 1:
                return 0.0
            entropy = -l_ratio * np.log2(l_ratio) - (1-l_ratio) * np.log2(1-l_ratio)
            return n * entropy
        
        elif scaffold_name == 'FIBONACCI':
            # Cost = violations of Fibonacci constraint
            violations = self._count_violations(symbols, 'FIBONACCI')
            return violations * np.log2(n + 1)
        
        elif scaffold_name == 'TRIBONACCI':
            violations = self._count_violations(symbols, 'TRIBONACCI')
            return violations * np.log2(n + 1)
        
        else:  # BINARY_RUN
            # Cost = number of run-length changes
            runs = 1
            for i in range(1, len(symbols)):
                if symbols[i] != symbols[i-1]:
                    runs += 1
            return runs * np.log2(n / runs + 1)
    
    def _count_violations(self, symbols: List[str], grammar: str) -> int:
        """Count grammar violations."""
        if grammar == 'FIBONACCI':
            # Violation: SS (two consecutive S without L)
            violations = 0
            for i in range(len(symbols) - 1):
                if symbols[i] == 'S' and symbols[i+1] == 'S':
                    violations += 1
            return violations
        return 0


# ============================================================================
# 6. UNIFIED KOOPMAN-SHEAF ENGINE
# ============================================================================
@dataclass
class KoopmanSheafState:
    """Complete state of the Koopman-Sheaf Renormalized Generator."""
    koopman_coherence: float
    dominant_frequency: float
    sheaf_energy: float
    sheaf_regime: str
    correlation_length: float
    rg_depth: int
    kl_divergence: float
    structural_displacement: bool
    mdl_scaffold: str
    mdl_confidence: float
    gluing_possible: bool
    fracture_count: int


class KoopmanSheafEngine:
    """
    The Unified Stack: Koopman–Sheaf Renormalized Generator with MDL-selected scaffolds.
    
    Each layer consumes outputs of the previous and produces a quantity
    with a clear decision role.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.koopman = KoopmanSpectralEngine(n_modes=5, device=device)
        self.sheaf = SheafConsistencyEngine(window_size=20, overlap=10)
        self.rg = RGCorrelationEngine()
        self.info_geom = InfoGeometryEngine()
        self.mdl = MDLScaffoldSelector()
        
    def process(self, time_series: List[float]) -> KoopmanSheafState:
        """
        Full Koopman-Sheaf analysis pipeline.
        
        Signal → Koopman modes → Symbolization → RG parse → 
        Sheaf gluing → MDL selection → Regime state
        """
        if len(time_series) < 20:
            return KoopmanSheafState(
                koopman_coherence=0, dominant_frequency=0,
                sheaf_energy=0, sheaf_regime='INSUFFICIENT_DATA',
                correlation_length=0, rg_depth=0,
                kl_divergence=0, structural_displacement=False,
                mdl_scaffold='NONE', mdl_confidence=0,
                gluing_possible=True, fracture_count=0
            )
        
        # 1. Koopman spectral decomposition
        koopman_result = self.koopman.extract_koopman_modes(time_series)
        
        # 2. Sheaf coboundary analysis
        sheaf_result = self.sheaf.compute_sheaf_obstruction(time_series)
        
        # 3. RG correlation length
        rg_result = self.rg.compute_correlation_length(time_series)
        
        # 4. Info geometry update (using scale coherence as posterior proxy)
        scale_coherence = {f'scale_{i}': sc['coherence'] 
                          for i, sc in enumerate(rg_result['scale_coherence'])}
        if not scale_coherence:
            scale_coherence = {'default': 0.5}
        info_result = self.info_geom.update_posterior(scale_coherence)
        
        # 5. MDL scaffold selection
        mdl_result = self.mdl.select_scaffold(time_series)
        
        return KoopmanSheafState(
            koopman_coherence=koopman_result.get('koopman_coherence', 0),
            dominant_frequency=koopman_result.get('dominant_frequency', 0),
            sheaf_energy=sheaf_result.get('energy', 0),
            sheaf_regime=sheaf_result.get('regime', 'UNKNOWN'),
            correlation_length=rg_result.get('correlation_length', 0),
            rg_depth=rg_result.get('max_depth', 0),
            kl_divergence=info_result.get('kl_divergence', 0),
            structural_displacement=info_result.get('structural_displacement', False),
            mdl_scaffold=mdl_result.get('scaffold', 'NONE'),
            mdl_confidence=mdl_result.get('confidence', 0),
            gluing_possible=sheaf_result.get('gluing_possible', True),
            fracture_count=sheaf_result.get('fracture_count', 0)
        )
    
    def get_metacontrol_policy(self, state: KoopmanSheafState) -> Dict:
        """
        Metacontroller option selection based on Koopman-Sheaf state.
        
        Π(k,r|t) ∝ exp(η·E[q_k(t)] - γ·E_sheaf(t) - μ·L(M_r))
        
        Returns action recommendation.
        """
        # Preference weights
        eta = 1.0   # Prefer stable scales
        gamma = 2.0  # Penalize fractured structure
        mu = 0.5    # Slight preference for simpler scaffolds
        
        # Compute policy score
        stability_score = eta * state.koopman_coherence
        fracture_penalty = gamma * state.sheaf_energy
        complexity_penalty = mu * (1 - state.mdl_confidence)
        
        policy_score = stability_score - fracture_penalty - complexity_penalty
        
        # Action recommendation
        if state.sheaf_regime == 'GLOBALLY_CONSISTENT' and state.rg_depth >= 3:
            action = 'TREND_FOLLOW'
            confidence = min(0.95, state.koopman_coherence)
        elif state.sheaf_regime == 'STRUCTURAL_FRACTURE':
            action = 'REDUCE_EXPOSURE'
            confidence = 0.5
        elif state.structural_displacement:
            action = 'REGIME_TRANSITION_WAIT'
            confidence = 0.3
        else:
            action = 'MEAN_REVERT'
            confidence = min(0.7, 1 - state.sheaf_energy)
        
        return {
            'action': action,
            'confidence': confidence,
            'policy_score': policy_score,
            'reasoning': {
                'stability': stability_score,
                'fracture_penalty': fracture_penalty,
                'complexity_penalty': complexity_penalty
            }
        }


# ============================================================================
# INTEGRATION WITH FMTR
# ============================================================================
def integrate_koopman_sheaf_into_fmtr(price_history: List[float], fmtr_kernel=None) -> Dict:
    """
    Bridge function to integrate Koopman-Sheaf analysis into existing FMTR kernel.
    """
    engine = KoopmanSheafEngine()
    state = engine.process(price_history)
    policy = engine.get_metacontrol_policy(state)
    
    # Map sheaf_regime to forecast_direction
    direction = 'FLAT'
    if state.sheaf_regime:
        if 'UP' in state.sheaf_regime:
            direction = 'BULLISH'
        elif 'DOWN' in state.sheaf_regime:
            direction = 'BEARISH'
            
    return {
        # Original keys
        'koopman_coherence': state.koopman_coherence,
        'dominant_frequency': state.dominant_frequency,
        
        # Sheaf outputs
        'sheaf_energy': state.sheaf_energy,
        'sheaf_regime': state.sheaf_regime,
        'gluing_possible': state.gluing_possible,
        'fracture_count': state.fracture_count,
        
        # RG outputs
        'correlation_length': state.correlation_length,
        'rg_depth': state.rg_depth,
        
        # Info geometry
        'kl_divergence': state.kl_divergence,
        'structural_displacement': state.structural_displacement,
        
        # MDL
        'mdl_scaffold': state.mdl_scaffold,
        'mdl_confidence': state.mdl_confidence,
        
        # Metacontrol
        'recommended_action': policy['action'],
        'action_confidence': policy['confidence'],
        'policy_score': policy['policy_score'],
        
        # Server expected keys
        'forecast_direction': direction,
        'spectral_confidence': state.koopman_coherence,
        'mdl_grammar': state.mdl_scaffold,
        'action': policy['action']
    }
