import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

@dataclass
class TorsionSignal:
    direction: int  # 1 (Long), -1 (Short), 0 (Neutral)
    strength: float # 0.0 to 1.0
    node: float     # Target/Stop level
    meta: str       # Reasoning

class TorsionPhysicsModel:
    """
    TORSION PHYSICS MODEL
    =====================
    Models market price action as a twisting elastic substrate.
    
    Key Concepts (Statistical Equivalents):
    1. Chirality: Autocorrelation sign of returns (trending vs mean-reverting)
    2. Angular Momentum Density (AMD): Volume-weighted center of mass of price changes
    3. Substrate Compression (SCF): Volume-to-range compression ratio
    4. Refraction: Price deviation and gravity attraction to high-density support/resistance nodes
    """
    def __init__(self, lookback_window=20):
        self.nodes = []
        self.chirality = 0
        self.lookback = lookback_window
        self.history_prices = []
        self.history_volumes = []
        
    def update_history(self, close_price, open_price, volume):
        self.history_prices.append({'close': close_price, 'open': open_price})
        self.history_volumes.append(volume)
        
        # Keep buffer finite
        if len(self.history_prices) > self.lookback * 2:
            self.history_prices.pop(0)
            self.history_volumes.pop(0)

    def calculate_amd(self):
        """Angular Momentum Density: Rotational force of price * volume."""
        if len(self.history_prices) < self.lookback:
            return 0.0
            
        recent_prices = self.history_prices[-self.lookback:]
        recent_vols = self.history_volumes[-self.lookback:]
        
        # Bodies as mass
        bodies = np.array([abs(p['close'] - p['open']) for p in recent_prices])
        volumes = np.array(recent_vols)
        
        # Position relative to center of mass (time)
        center = len(bodies) / 2
        positions = np.arange(len(bodies)) - center
        
        # Torque = r x F (Position x BodySize*Volume)
        # Summing gives total angular momentum
        amd = np.sum(bodies * positions * volumes)
        
        # Normalize by total volume to get density
        total_vol = np.sum(volumes) + 1e-8
        return amd / (total_vol * self.lookback)

    def detect_chirality(self):
        """Detect dominant 'handedness' (serial correlation sign)."""
        if len(self.history_prices) < self.lookback:
            return 0
            
        closes = pd.Series([p['close'] for p in self.history_prices[-self.lookback:]])
        returns = closes.pct_change().dropna()
        
        if len(returns) < 5:
            return 0
            
        # Autocorrelation of returns
        corr = returns.autocorr(lag=1)
        
        # Chirality = sign of lag-1 autocorrelation of returns
        #   Corr > 0: Momentum / Trending (Right-handed/Persisting)
        #   Corr < 0: Mean Reversion / Ranging (Left-handed/Anti-persisting)
        
        return np.sign(corr) if abs(corr) > 0.1 else 0

    def substrate_compression(self, current_volatility, oi_change=0):
        """SCF: Substrate Compression Factor. Tension in the field."""
        # Using Realized Vol (rv) as base. Ideally compare Short-term IV vs Long-term RV.
        # Here we approximate: High Vol + High OI = High Tension? 
        # Actually usually High Vol + Dropping OI = Release.
        # User Formula: (iv / rv) * (1 + abs(oi_change))
        
        # Since we might not have live IV/OI, we use:
        # Volatility of Volatility (VolOfVol) or range compression?
        
        # Implementation for typical price-only stream:
        # Compression = Low price range relative to volume
        # Return 1.0 as the neutral baseline value if we have insufficient history
        if len(self.history_prices) < 5: return 1.0
        
        recent = self.history_prices[-10:]
        high_low_range = np.max([p['close'] for p in recent]) - np.min([p['close'] for p in recent])
        avg_vol = np.mean(self.history_volumes[-10:]) + 1
        
        # If range is tight but volume is high -> HIGH COMPRESSION
        # Inverted: Volume / Range
        if high_low_range == 0: return 0.0
        compression = (avg_vol / high_low_range) * current_volatility
        
        # Normalize roughly to 1.0 baseline
        return np.log1p(compression)

    def find_nodes(self, current_price):
        """Update standing wave nodes based on price clustering."""
        if len(self.history_prices) < 50:
            return [current_price]
            
        prices = [p['close'] for p in self.history_prices]
        
        # Simple clustering (Kernel Density Estimation approach simplified)
        # We look for modes in the price distribution
        bins = np.linspace(min(prices), max(prices), 20)
        hist, bin_edges = np.histogram(prices, bins=bins)
        
        nodes = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                nodes.append((bin_edges[i] + bin_edges[i+1])/2)
                
        self.nodes = sorted(nodes)
        return self.nodes

    def analyze(self, current_price, current_vol, volatility) -> Tuple[TorsionSignal, Dict]:
        """Main analysis loop."""
        # Update nodes
        self.find_nodes(current_price)
        
        # Calc metrics
        amd = self.calculate_amd()
        chirality = self.detect_chirality()
        scf = self.substrate_compression(volatility)
        
        # Find nearest node
        if not self.nodes:
            self.nodes = [current_price]
            
        nearest_node = min(self.nodes, key=lambda n: abs(n - current_price))
        distance_pct = abs(current_price - nearest_node) / current_price * 100
        
        signal = TorsionSignal(0, 0.0, nearest_node, "Neutral")
        
        # INTERPRETATION LOGIC
        
        # 1. High Compression (SCF) + Momentum (Chirality > 0) -> BREAKOUT from Node
        if scf > 2.0 and chirality > 0:
            if current_price > nearest_node:
                signal = TorsionSignal(1, min(scf/5, 1.0), nearest_node, f"Torsion Spring Breakout Up (SCF: {scf:.1f})")
            else:
                signal = TorsionSignal(-1, min(scf/5, 1.0), nearest_node, f"Torsion Spring Breakout Down (SCF: {scf:.1f})")
                
        # 2. Low Compression + Mean Reversion (Chirality < 0) -> GRAVITY to Node
        elif scf < 1.0 and chirality < 0:
            if distance_pct > 0.1: # If significantly away from node
                direction = 1 if current_price < nearest_node else -1
                signal = TorsionSignal(direction, 0.5, nearest_node, f"Node Gravity Attraction (Dist: {distance_pct:.2f}%)")
                
        return signal, {"amd": amd, "chirality": chirality, "scf": scf, "nodes": self.nodes}
