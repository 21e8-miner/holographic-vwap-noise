"""
GRAPH PHYSICS ENGINE
=====================
Implements Laplacian Arbitrage (Gaussian Free Fields) for market analysis.

Mathematical Foundation:
- Source: GRAINVDB_MOONSHOT_IDEAS.md TIER 5
- The market is modeled as a correlation graph
- Laplacian L = D - A captures "tension" between assets
- Potential φ = L @ returns identifies mean-reversion opportunities
- Fiedler Value (λ₂) indicates regime: Fragmented/Normal/Hyper-Coupled

Integration:
- Call `GraphPhysicsEngine.analyze()` with price dataframe
- Returns: fiedler_value, regime, potential_signals
"""

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger("GraphPhysics")


class GraphPhysicsEngine:
    """
    Implements Laplacian Arbitrage for mean-reversion signal generation.
    
    Key Metrics:
    - Fiedler Value: Algebraic connectivity of the market graph
    - Laplacian Potential: Measures deviation from graph equilibrium
    """
    
    def __init__(self, correlation_threshold: float = 0.4):
        """
        Args:
            correlation_threshold: Minimum correlation to create graph edge (default 0.4)
        """
        self.correlation_threshold = correlation_threshold
        self.last_fiedler = 0.0
        self.last_regime = "UNKNOWN"
    
    def analyze(self, prices: pd.DataFrame, lookback_hours: int = 4) -> Dict:
        """
        Analyze market using Graph Physics.
        
        Args:
            prices: DataFrame with columns as asset symbols, rows as time
            lookback_hours: Hours to consider for momentum calculation
            
        Returns:
            Dict containing:
            - fiedler_value: Float, algebraic connectivity
            - regime: str, "FRAGMENTED" | "NORMAL" | "HYPER_COUPLED"
            - signals: List[Dict] with asset, potential, signal, neighbors
        """
        if prices is None or prices.empty or len(prices) < 5:
            return {
                'fiedler_value': 0.0,
                'regime': 'INSUFFICIENT_DATA',
                'signals': []
            }
        
        try:
            # 1. Compute Returns & Correlation
            returns = prices.pct_change().fillna(0)
            corr_matrix = returns.corr().values
            
            # 2. Build Adjacency Matrix (Graph Edges)
            adj_matrix = np.where(
                np.abs(corr_matrix) > self.correlation_threshold, 
                np.abs(corr_matrix), 
                0
            )
            np.fill_diagonal(adj_matrix, 0)  # No self-loops
            
            # 3. Compute Graph Laplacian (L = D - A)
            L = laplacian(adj_matrix, normed=False)
            
            # 4. Spectral Analysis (Regime Detection)
            evals, evecs = eigh(L)
            fiedler_value = float(evals[1]) if len(evals) > 1 else 0.0
            
            # Classify regime based on Fiedler value
            if fiedler_value < 0.1:
                regime = "FRAGMENTED"  # Assets moving independently
            elif fiedler_value > 2.0:
                regime = "HYPER_COUPLED"  # Crash/Bubble risk
            else:
                regime = "NORMAL"  # Healthy correlation
            
            # 5. Compute Laplacian Potential
            lookback = min(lookback_hours, len(returns))
            recent_performance = returns.iloc[-lookback:].sum().values
            potential = L @ recent_performance
            
            # 6. Generate Signals
            assets = prices.columns.tolist()
            signals = []
            
            for i, asset in enumerate(assets):
                phi = float(potential[i])
                
                if phi < -0.05:
                    signal = "LONG"
                    strength = abs(phi)
                elif phi > 0.05:
                    signal = "SHORT"
                    strength = abs(phi)
                else:
                    signal = "NEUTRAL"
                    strength = 0.0
                    
                signals.append({
                    'asset': asset,
                    'potential': phi,
                    'signal': signal,
                    'strength': strength,
                    'neighbors': int(np.sum(adj_matrix[i] > 0))
                })
            
            # Sort by absolute potential (strongest signals first)
            signals.sort(key=lambda x: abs(x['potential']), reverse=True)
            
            # Cache
            self.last_fiedler = fiedler_value
            self.last_regime = regime
            
            return {
                'fiedler_value': fiedler_value,
                'regime': regime,
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"Graph Physics analysis failed: {e}")
            return {
                'fiedler_value': 0.0,
                'regime': 'ERROR',
                'signals': []
            }
    
    def get_top_opportunities(self, prices: pd.DataFrame, n: int = 5) -> List[Dict]:
        """
        Get top N mean-reversion opportunities.
        
        Returns signals with highest absolute potential (most dislocated from graph equilibrium).
        """
        result = self.analyze(prices)
        return result['signals'][:n]
    
    def is_hyper_coupled(self, prices: pd.DataFrame) -> bool:
        """
        Quick check if market is in HYPER_COUPLED regime (crash/bubble risk).
        
        Use this to:
        - Reduce position sizes
        - Tighten stops
        - Switch to portfolio-level hedging (SIG philosophy)
        """
        result = self.analyze(prices)
        return result['regime'] == 'HYPER_COUPLED'


# Singleton instance for easy import
_engine = None

def get_engine(correlation_threshold: float = 0.4) -> GraphPhysicsEngine:
    """Get or create the global GraphPhysicsEngine instance."""
    global _engine
    if _engine is None:
        _engine = GraphPhysicsEngine(correlation_threshold)
    return _engine


def analyze_market(prices: pd.DataFrame) -> Dict:
    """Convenience function: Analyze market using the global engine."""
    return get_engine().analyze(prices)


def get_fiedler_regime(prices: pd.DataFrame) -> Tuple[float, str]:
    """Convenience function: Get (fiedler_value, regime) tuple."""
    result = get_engine().analyze(prices)
    return result['fiedler_value'], result['regime']
