"""
HOLOGRAPHIC VWAP ENGINE
=======================

Extends the VWAP Scanner with tensor network analysis and holographic principles.

Core Innovations:
1. SECTOR TENSOR NETWORK — Treats market sectors as entangled nodes
2. VWAP CURVATURE FIELD — Models deviation from VWAP as a geometric manifold  
3. FLOW ENTANGLEMENT — Uses order flow clustering to measure hidden correlations
4. RESONANCE DETECTION — Finds when multiple stocks signal simultaneously

Physics Inspiration:
- The VWAP acts as a "ground state" — the fair value energy level
- Deviations are "excitations" that want to decay back
- When sectors are entangled, excitations propagate (contagion)
- The curvature of the deviation field predicts return velocity

This is NOT conventional VWAP mean reversion.
This is GEOMETRIC FIELD THEORY applied to equity microstructure.

Compatible with existing AnchoredVWAPScanner.

Author: FMTR Research
Date: 2024-12-13
"""

import numpy as np
import pandas as pd
from scipy import linalg, stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class GeometricPhase(Enum):
    """Market phase as classified by VWAP geometry"""
    GROUND_STATE = "ground_state"       # At VWAP, low vol - stable
    EXCITATION = "excitation"           # Moving away from VWAP
    DECAY = "decay"                     # Returning to VWAP
    RESONANCE = "resonance"             # Multiple stocks in sync
    QUANTUM_JUMP = "quantum_jump"       # Gap/sudden regime change
    SUPERPOSITION = "superposition"     # Mixed signals, uncertain
    STOCHASTIC_CLIFF = "stochastic_cliff" # High risk of false summit


@dataclass
class VWAPCurvature:
    """Curvature metrics for a single stock's VWAP deviation"""
    ticker: str
    deviation: float            # Current distance from VWAP (in std devs)
    velocity: float             # Rate of change of deviation
    acceleration: float         # Second derivative (curvature)
    energy: float               # |deviation|^2 + |velocity|^2 (total excitation)
    decay_rate: float           # Estimated rate of return to VWAP
    tail_var: float             # Expected shortfall at current deviation
    phase: GeometricPhase       # Current phase


@dataclass
class SectorEntanglement:
    """Entanglement between sectors"""
    sector1: str
    sector2: str
    correlation: float          # Return correlation
    flow_correlation: float     # Order flow correlation (if available)
    phase_sync: float           # Phase synchronization (0-1)
    mutual_information: float   # Information shared between sectors


@dataclass
class HolographicVWAPState:
    """Complete holographic state of the VWAP universe"""
    timestamp: datetime
    
    # Per-stock metrics
    curvatures: Dict[str, VWAPCurvature]
    
    # Sector-level
    sector_entanglement: Dict[str, float]
    dominant_sector: Optional[str]
    sector_entropy: float
    
    # Global
    total_energy: float         # Sum of all excitations
    resonance_score: float      # How aligned are all signals?
    phase: GeometricPhase       # Global market phase
    barrier_distance: float     # Distance to critical threshold
    
    # Predictions
    mean_reversion_strength: float  # Expected rate of return to VWAP
    contagion_risk: float           # Risk of sector spillover


# =============================================================================
# VWAP CURVATURE CALCULATOR
# =============================================================================

class VWAPCurvatureCalculator:
    """
    Computes the geometric curvature of VWAP deviation.
    
    The key insight: The DEVIATION from VWAP is not just a number —
    it's a TRAJECTORY in price space. The curvature of this trajectory
    tells us how "bent" the path is, which predicts reversal probability.
    
    High curvature = sharp bend = likely reversal
    Low curvature = straight path = momentum continues
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.deviation_history: Dict[str, List[float]] = {}
        
    def update(self, ticker: str, current_price: float, vwap: float, vwap_std: float) -> VWAPCurvature:
        """
        Update curvature metrics for a ticker.
        
        Args:
            ticker: Stock symbol
            current_price: Current price
            vwap: Anchored VWAP value
            vwap_std: Standard deviation of VWAP (for normalization)
        """
        # Compute deviation in standard deviations
        deviation = (current_price - vwap) / vwap_std if vwap_std > 0 else 0
        
        # Update history
        if ticker not in self.deviation_history:
            self.deviation_history[ticker] = []
        self.deviation_history[ticker].append(deviation)
        
        # Trim history
        if len(self.deviation_history[ticker]) > self.lookback:
            self.deviation_history[ticker] = self.deviation_history[ticker][-self.lookback:]
        
        history = np.array(self.deviation_history[ticker])
        
        # Compute derivatives
        if len(history) >= 3:
            velocity = history[-1] - history[-2]
            acceleration = (history[-1] - 2*history[-2] + history[-3]) if len(history) >= 3 else 0
        elif len(history) >= 2:
            velocity = history[-1] - history[-2]
            acceleration = 0
        else:
            velocity = 0
            acceleration = 0
        
        # Energy = kinetic + potential
        energy = deviation**2 + velocity**2
        
        # Decay rate estimation (mean reversion speed)
        # Use autocorrelation of deviation as proxy
        if len(history) >= 10:
            autocorr = np.corrcoef(history[:-1], history[1:])[0, 1]
            decay_rate = 1 - autocorr if not np.isnan(autocorr) else 0.5
        else:
            decay_rate = 0.5
        
        # --- MEHTA HARDENING: TailVaR (Expected Shortfall) ---
        # Formula: TVaR = mu - sigma * (phi(z) / (1 - alpha))
        # Here we use deviation as z, mu=0 (fair value), sigma=1 (normalized)
        alpha = 0.95
        z_thresh = stats.norm.ppf(alpha)
        phi_z = stats.norm.pdf(deviation)
        tail_var = abs(deviation) + (phi_z / (1 - alpha)) if abs(deviation) > z_thresh else 0.0
        
        # Phase classification
        phase = self._classify_phase(deviation, velocity, acceleration, tail_var)
        
        return VWAPCurvature(
            ticker=ticker,
            deviation=deviation,
            velocity=velocity,
            acceleration=acceleration,
            energy=energy,
            decay_rate=decay_rate,
            tail_var=tail_var,
            phase=phase
        )
    
    def _classify_phase(self, deviation: float, velocity: float, acceleration: float, tail_var: float = 0.0) -> GeometricPhase:
        """Classify the geometric phase"""
        
        # Mehta Hardening: High TailVaR = Stochastic Cliff (Extreme risk)
        if tail_var > 4.0:
            return GeometricPhase.STOCHASTIC_CLIFF
        
        # Near VWAP = ground state
        if abs(deviation) < 0.5:
            return GeometricPhase.GROUND_STATE
        
        # Moving toward VWAP = decay
        if deviation > 0 and velocity < 0:
            return GeometricPhase.DECAY
        if deviation < 0 and velocity > 0:
            return GeometricPhase.DECAY
        
        # High acceleration = quantum jump
        if abs(acceleration) > 0.5:
            return GeometricPhase.QUANTUM_JUMP
        
        # Moving away from VWAP = excitation
        if (deviation > 0 and velocity > 0) or (deviation < 0 and velocity < 0):
            return GeometricPhase.EXCITATION
        
        return GeometricPhase.SUPERPOSITION


# =============================================================================
# SECTOR TENSOR NETWORK
# =============================================================================

class SectorTensorNetwork:
    """
    Models sectors as nodes in a tensor network.
    
    Each sector has a "state" (average deviation, flow direction).
    Edges represent correlations between sectors.
    
    When sectors are highly entangled (correlated), signals in one
    sector predict signals in another. This is the "contagion" effect.
    """
    
    # Standard sector mapping
    SECTORS = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLV': 'Healthcare',
        'XLI': 'Industrials',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLU': 'Utilities',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLC': 'Communications'
    }
    
    # Stock to sector mapping (simplified)
    STOCK_SECTORS = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'AMZN': 'Technology', 'META': 'Technology', 'NVDA': 'Technology',
        'TSLA': 'Consumer Discretionary', 'AMD': 'Technology', 'NFLX': 'Technology',
        'JPM': 'Financials', 'BAC': 'Financials', 'GS': 'Financials',
        'XOM': 'Energy', 'CVX': 'Energy', 'OXY': 'Energy',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
        'BA': 'Industrials', 'CAT': 'Industrials', 'GE': 'Industrials',
        'WMT': 'Consumer Staples', 'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
        'DIS': 'Communications', 'CMCSA': 'Communications', 'T': 'Communications',
        'SPY': 'Market', 'QQQ': 'Technology', 'IWM': 'Small Cap',
    }
    
    def __init__(self):
        self.sector_states: Dict[str, List[float]] = {}  # Sector -> deviation history
        self.sector_flows: Dict[str, List[float]] = {}   # Sector -> flow direction history
        
    def update(self, ticker: str, deviation: float, flow_direction: float = 0.0):
        """Update sector state with a stock's data"""
        sector = self.STOCK_SECTORS.get(ticker, 'Other')
        
        if sector not in self.sector_states:
            self.sector_states[sector] = []
            self.sector_flows[sector] = []
            
        self.sector_states[sector].append(deviation)
        self.sector_flows[sector].append(flow_direction)
        
        # Trim
        if len(self.sector_states[sector]) > 50:
            self.sector_states[sector] = self.sector_states[sector][-50:]
            self.sector_flows[sector] = self.sector_flows[sector][-50:]
    
    def compute_entanglement_matrix(self) -> np.ndarray:
        """
        Compute the sector-sector entanglement matrix.
        
        Uses correlation as a proxy for entanglement.
        """
        sectors = list(self.sector_states.keys())
        n = len(sectors)
        
        if n == 0:
            return np.array([])
        
        # Build correlation matrix
        C = np.eye(n)
        
        for i, s1 in enumerate(sectors):
            for j, s2 in enumerate(sectors):
                if i >= j:
                    continue
                    
                dev1 = np.array(self.sector_states[s1])
                dev2 = np.array(self.sector_states[s2])
                
                min_len = min(len(dev1), len(dev2))
                if min_len < 5:
                    C[i, j] = 0
                    C[j, i] = 0
                else:
                    corr = np.corrcoef(dev1[-min_len:], dev2[-min_len:])[0, 1]
                    if np.isnan(corr):
                        corr = 0
                    C[i, j] = corr
                    C[j, i] = corr
        
        return C
    
    def compute_sector_entropy(self) -> float:
        """
        Compute the entropy of sector states.
        
        High entropy = sectors moving independently (diversified)
        Low entropy = sectors moving together (systemic risk)
        """
        C = self.compute_entanglement_matrix()
        
        if len(C) == 0:
            return 0.0
        
        try:
            eigenvalues, _ = linalg.eigh(C)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            
            if len(eigenvalues) == 0:
                return 0.0
            
            # Normalize to probability
            eigenvalues = eigenvalues / np.sum(eigenvalues)
            
            # Von Neumann entropy
            entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
            
            return float(entropy)
        except:
            return 0.0
    
    def get_dominant_sector(self) -> Tuple[Optional[str], float]:
        """
        Find the sector with the strongest signal (largest deviation).
        """
        if not self.sector_states:
            return (None, 0.0)
        
        dominant = None
        max_dev = 0.0
        
        for sector, devs in self.sector_states.items():
            if devs:
                avg_dev = np.mean(np.abs(devs[-5:])) if len(devs) >= 5 else np.mean(np.abs(devs))
                if avg_dev > max_dev:
                    max_dev = avg_dev
                    dominant = sector
        
        return (dominant, max_dev)
    
    def predict_contagion(self) -> float:
        """
        Predict the likelihood of sector contagion.
        
        High when:
        - Sectors are highly correlated
        - One sector is far from VWAP (potential domino)
        """
        C = self.compute_entanglement_matrix()
        
        if len(C) == 0:
            return 0.0
        
        # Average off-diagonal correlation
        n = len(C)
        avg_corr = (np.sum(np.abs(C)) - n) / (n * (n - 1)) if n > 1 else 0
        
        # Max sector deviation
        _, max_dev = self.get_dominant_sector()
        
        # Contagion = high correlation * high deviation
        contagion = avg_corr * min(max_dev, 3) / 3
        
        return float(contagion)


# =============================================================================
# RESONANCE DETECTOR
# =============================================================================

class ResonanceDetector:
    """
    Detects when multiple stocks' VWAP deviations are "in resonance."
    
    Resonance occurs when:
    1. Multiple stocks have the same phase (all decaying to VWAP)
    2. The deviations are correlated in time
    3. The sectors align
    
    Resonance amplifies signal strength — like constructive interference.
    """
    
    def __init__(self, min_stocks: int = 3):
        self.min_stocks = min_stocks
        self.phase_history: Dict[str, List[GeometricPhase]] = {}
        
    def update(self, ticker: str, phase: GeometricPhase):
        """Track phase history"""
        if ticker not in self.phase_history:
            self.phase_history[ticker] = []
        self.phase_history[ticker].append(phase)
        
        if len(self.phase_history[ticker]) > 20:
            self.phase_history[ticker] = self.phase_history[ticker][-20:]
    
    def compute_resonance(self, curvatures: Dict[str, VWAPCurvature]) -> Tuple[float, List[str]]:
        """
        Compute the resonance score and list of resonating tickers.
        
        Returns:
            (score 0-1, list of tickers in resonance)
        """
        if len(curvatures) < self.min_stocks:
            return (0.0, [])
        
        # Group by phase
        decay_tickers = []
        excitation_tickers = []
        
        for ticker, curv in curvatures.items():
            if curv.phase == GeometricPhase.DECAY:
                decay_tickers.append(ticker)
            elif curv.phase == GeometricPhase.EXCITATION:
                excitation_tickers.append(ticker)
        
        # Resonance = fraction of stocks in dominant phase
        max_group = max(len(decay_tickers), len(excitation_tickers))
        total = len(curvatures)
        
        resonance_score = max_group / total
        
        if len(decay_tickers) >= len(excitation_tickers):
            resonating = decay_tickers
        else:
            resonating = excitation_tickers
            
        # --- MEHTA HARDENING: False Summit Filter ---
        # 90% of perceived summits are false. We discount resonance if 
        # the dominant group has just recently entered the phase.
        recent_count = 0
        for t in resonating:
            hist = self.phase_history.get(t, [])
            if len(hist) < 5 or all(p == hist[-1] for p in hist[-3:]):
                # Established phase
                pass
            else:
                recent_count += 1
        
        # If > 50% of the resonating group is "new" to the phase, 
        # apply the 90% False Summit discount.
        if resonating and (recent_count / len(resonating)) > 0.5:
            resonance_score *= 0.1 # Severe penalty for immature signals
        
        return (resonance_score, resonating)
    
    def is_resonating(self, curvatures: Dict[str, VWAPCurvature], threshold: float = 0.6) -> bool:
        """Check if market is in resonance"""
        score, _ = self.compute_resonance(curvatures)
        return score >= threshold


# =============================================================================
# HOLOGRAPHIC VWAP ENGINE
# =============================================================================

class HolographicVWAPEngine:
    """
    The master engine that combines:
    - VWAP curvature analysis
    - Sector tensor network
    - Resonance detection
    
    Produces holographic state for the entire VWAP universe.
    """
    
    def __init__(self):
        self.curvature_calc = VWAPCurvatureCalculator()
        self.sector_network = SectorTensorNetwork()
        self.resonance_detector = ResonanceDetector()
        
        self.curvatures: Dict[str, VWAPCurvature] = {}
        self.barrier_level: float = 10.0  # Energy barrier
        self.energy_history: List[float] = []
        
    def update(self, ticker: str, current_price: float, vwap: float, 
               vwap_std: float, flow_direction: float = 0.0) -> VWAPCurvature:
        """
        Update state with new data for a ticker.
        
        Args:
            ticker: Stock symbol
            current_price: Current price
            vwap: Anchored VWAP
            vwap_std: VWAP standard deviation
            flow_direction: Order flow direction (-1 to +1)
        """
        # Compute curvature
        curv = self.curvature_calc.update(ticker, current_price, vwap, vwap_std)
        self.curvatures[ticker] = curv
        
        # Update sector network
        self.sector_network.update(ticker, curv.deviation, flow_direction)
        
        # Update resonance detector
        self.resonance_detector.update(ticker, curv.phase)
        
        return curv
    
    def get_holographic_state(self) -> HolographicVWAPState:
        """
        Compute the full holographic state.
        """
        # Total energy
        total_energy = sum(c.energy for c in self.curvatures.values())
        self.energy_history.append(total_energy)
        if len(self.energy_history) > 100:
            self.energy_history = self.energy_history[-100:]
        
        # Dynamic barrier
        if len(self.energy_history) > 20:
            self.barrier_level = np.percentile(self.energy_history, 95)
        
        barrier_distance = total_energy / self.barrier_level if self.barrier_level > 0 else 0
        
        # Sector metrics
        sector_entropy = self.sector_network.compute_sector_entropy()
        dominant_sector, _ = self.sector_network.get_dominant_sector()
        contagion_risk = self.sector_network.predict_contagion()
        
        # Entanglement bonds as dict
        C = self.sector_network.compute_entanglement_matrix()
        sectors = list(self.sector_network.sector_states.keys())
        sector_entanglement = {}
        for i, s1 in enumerate(sectors):
            for j, s2 in enumerate(sectors):
                if i < j:
                    key = f"{s1}-{s2}"
                    sector_entanglement[key] = float(abs(C[i, j])) if len(C) > 0 else 0.0
        
        # Resonance
        resonance_score, _ = self.resonance_detector.compute_resonance(self.curvatures)
        
        # Mean reversion strength
        if len(self.curvatures) > 0:
            avg_decay = np.mean([c.decay_rate for c in self.curvatures.values()])
            mean_reversion_strength = avg_decay * (1 + resonance_score)
        else:
            mean_reversion_strength = 0.0
        
        # Global phase
        if resonance_score > 0.7:
            phase = GeometricPhase.RESONANCE
        elif barrier_distance > 1.0:
            phase = GeometricPhase.QUANTUM_JUMP
        elif total_energy < 5:
            phase = GeometricPhase.GROUND_STATE
        else:
            # Count phases
            phase_counts = {}
            for c in self.curvatures.values():
                phase_counts[c.phase] = phase_counts.get(c.phase, 0) + 1
            if phase_counts:
                phase = max(phase_counts, key=phase_counts.get)
            else:
                phase = GeometricPhase.SUPERPOSITION
        
        return HolographicVWAPState(
            timestamp=datetime.now(),
            curvatures=self.curvatures.copy(),
            sector_entanglement=sector_entanglement,
            dominant_sector=dominant_sector,
            sector_entropy=sector_entropy,
            total_energy=total_energy,
            resonance_score=resonance_score,
            phase=phase,
            barrier_distance=barrier_distance,
            mean_reversion_strength=mean_reversion_strength,
            contagion_risk=contagion_risk
        )
    
    def get_signal_strength(self, ticker: str) -> float:
        """
        Get the signal strength for a ticker, amplified by holographic factors.
        
        Returns:
            Multiplier (0.5 to 2.0)
        """
        state = self.get_holographic_state()
        curv = self.curvatures.get(ticker)
        
        if curv is None:
            return 1.0
        
        base_strength = 1.0
        
        # Amplify if in resonance and decaying to VWAP
        if state.phase == GeometricPhase.RESONANCE:
            if curv.phase == GeometricPhase.DECAY:
                base_strength *= (1 + state.resonance_score)
        
        # Amplify if high curvature (sharp turn)
        if abs(curv.acceleration) > 0.3:
            base_strength *= 1.2
        
        # Reduce if barrier breached (too risky)
        if state.barrier_distance > 1.0 or curv.phase == GeometricPhase.STOCHASTIC_CLIFF:
            base_strength *= 0.3 # Harder cut for TailVaR risk
        
        # Amplify if same direction as dominant sector
        ticker_sector = self.sector_network.STOCK_SECTORS.get(ticker, 'Other')
        if ticker_sector == state.dominant_sector:
            base_strength *= 1.1
            
        # --- PAPER INTEGRATION: Pecking Order & Sharpe Efficiency ---
        try:
            from pecking_order import pecking_order_engine
            from portfolio_efficiency import sharpe_engine
            
            # 1. Pecking Order (Paper 2)
            po_score = pecking_order_engine.get_pecking_order_score(ticker)
            # High structural health (po_score -> 1.0) boosts signal confidence
            base_strength *= (0.8 + po_score * 0.4) 
            
            # 2. Portfolio Diversity (Paper 3 - Roy et al.)
            # Boost assets that contribute to diversification (low correlation)
            diversity_multiplier = sharpe_engine.get_diversity_multiplier(ticker, 0.02, 0.5) # Heuristic vol/corr
            base_strength *= diversity_multiplier
            
            # 3. Structured Event Representation (Paper 6 - Li et al., 2025)
            # Modulate strength via 'Contextual Event Multiplier' from SER attention.
            from event_representation import ser_engine, StructuredEvent
            sig_events = [
                StructuredEvent("Signal", "State", "Resonant" if state.resonance_score > 0.7 else "Normal"),
                StructuredEvent(ticker, "Trend", "Holographic" if base_strength > 1.0 else "Stochastic")
            ]
            event_inf = ser_engine.get_event_influence(ticker, sig_events)
            # Map [-0.05, 0.05] influence to [0.95, 1.05] multiplier
            base_strength *= (1.0 + event_inf)

        except ImportError:
            pass
        
        return float(np.clip(base_strength, 0.5, 3.0)) # Expanded cap for boosted diversifiers


# =============================================================================
# INTEGRATION WITH EXISTING SCANNER
# =============================================================================

def enhance_vwap_signal(scanner_data: dict, engine: HolographicVWAPEngine) -> dict:
    """
    Enhance VWAP scanner output with holographic analysis.
    
    Args:
        scanner_data: Output from AnchoredVWAPScanner.get_scan_results()
        engine: HolographicVWAPEngine instance
        
    Returns:
        Enhanced data with holographic fields
    """
    enhanced = scanner_data.copy()
    
    # Update engine with scanner data
    for row in enhanced.get('data', []):
        ticker = row.get('ticker')
        price = row.get('current_price', 0)
        vwap = row.get('vwap', price)
        std = row.get('vwap_std', 1)
        
        if ticker and price > 0:
            engine.update(ticker, price, vwap, std)
    
    # Get holographic state
    state = engine.get_holographic_state()
    
    # Add holographic fields to each row
    for row in enhanced.get('data', []):
        ticker = row.get('ticker')
        curv = state.curvatures.get(ticker)
        
        if curv:
            row['holographic'] = {
                'deviation': curv.deviation,
                'velocity': curv.velocity,
                'curvature': curv.acceleration,
                'energy': curv.energy,
                'phase': curv.phase.value,
                'signal_strength': engine.get_signal_strength(ticker)
            }
    
    # Add global holographic state
    enhanced['holographic_state'] = {
        'phase': state.phase.value,
        'total_energy': state.total_energy,
        'resonance_score': state.resonance_score,
        'sector_entropy': state.sector_entropy,
        'dominant_sector': state.dominant_sector,
        'barrier_distance': state.barrier_distance,
        'mean_reversion_strength': state.mean_reversion_strength,
        'contagion_risk': state.contagion_risk
    }
    
    return enhanced


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🌌 HOLOGRAPHIC VWAP ENGINE - DEMO")
    print("="*70)
    
    # Initialize
    engine = HolographicVWAPEngine()
    
    # Simulate some data
    np.random.seed(42)
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'JPM', 'XOM']
    
    print("\n📊 Simulating VWAP deviations...")
    
    # Common factor (market)
    n = 50
    market = np.cumsum(np.random.randn(n) * 0.1)
    
    for i in range(n):
        for j, ticker in enumerate(tickers):
            # Random VWAP deviation with sector correlation
            base_dev = market[i] + np.random.randn() * 0.3
            
            # Simulate mean reversion
            if i > 0:
                prev = engine.curvatures.get(ticker)
                if prev:
                    base_dev = prev.deviation * 0.9 + np.random.randn() * 0.2
            
            engine.update(
                ticker=ticker,
                current_price=100 + base_dev * 5,
                vwap=100,
                vwap_std=5
            )
    
    # Get state
    state = engine.get_holographic_state()
    
    print(f"\n🧠 HOLOGRAPHIC STATE:")
    print(f"   Phase:              {state.phase.value.upper()}")
    print(f"   Total Energy:       {state.total_energy:.2f}")
    print(f"   Resonance Score:    {state.resonance_score:.2%}")
    print(f"   Sector Entropy:     {state.sector_entropy:.4f}")
    print(f"   Dominant Sector:    {state.dominant_sector}")
    print(f"   Barrier Distance:   {state.barrier_distance:.2%}")
    print(f"   Contagion Risk:     {state.contagion_risk:.2%}")
    
    print(f"\n📈 CURVATURE METRICS:")
    for ticker, curv in state.curvatures.items():
        strength = engine.get_signal_strength(ticker)
        phase_icon = {
            GeometricPhase.GROUND_STATE: "⚪",
            GeometricPhase.EXCITATION: "🔴",
            GeometricPhase.DECAY: "🟢",
            GeometricPhase.RESONANCE: "🟣",
            GeometricPhase.QUANTUM_JUMP: "⚡",
            GeometricPhase.SUPERPOSITION: "❓"
        }.get(curv.phase, "❓")
        
        print(f"   {ticker:6} | Dev: {curv.deviation:+6.2f}σ | Vel: {curv.velocity:+5.2f} | "
              f"κ: {curv.acceleration:+5.2f} | {phase_icon} | Str: {strength:.2f}x")
    
    print(f"\n🔗 SECTOR ENTANGLEMENT:")
    for bond, strength in list(state.sector_entanglement.items())[:5]:
        bar = "█" * int(abs(strength) * 20)
        print(f"   {bond:30} {strength:+.3f} {bar}")
    
    print("\n" + "="*70)
    print("✅ Holographic VWAP Engine Operational")
    print("="*70)
