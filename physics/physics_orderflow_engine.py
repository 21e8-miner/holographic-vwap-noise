#!/usr/bin/env python3
"""
PHYSICS-INFORMED ORDER FLOW ENGINE
===================================
Maps real-time market microstructure to learned physics patterns.

Core Insight:
- Order flow dynamics follow fluid dynamics equations
- Bid-ask spread ≈ pressure gradient
- Order book depth ≈ density field
- Large trades ≈ shocks/discontinuities
- Market makers ≈ viscosity/damping

By querying GrainVDB (trained on MHD, turbulence, reaction-diffusion, etc.),
we can predict market evolution using physics pattern matching.

Created: January 15, 2026
"""

import sys
import os
import numpy as np
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRESH_BOT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "fresh_bot", "python", "lab"))
sys.path.insert(0, FRESH_BOT_PATH)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("PhysicsOrderFlow")

# Import GrainVDB
try:
    from grain_vdb import GrainVDB
    GRAINVDB_AVAILABLE = True
    logger.info("✅ GrainVDB loaded successfully")
except ImportError as e:
    GRAINVDB_AVAILABLE = False
    logger.warning(f"⚠️ GrainVDB not available: {e}")


# =============================================================================
# PHYSICS PATTERN DEFINITIONS
# =============================================================================

@dataclass
class PhysicsPattern:
    """Defines a physics pattern and its market interpretation."""
    dataset: str
    description: str
    market_analog: str
    signal_type: str  # 'momentum', 'mean_reversion', 'trend', 'volatility'
    confidence_weight: float = 1.0


# =============================================================================
# Φ EUCLID OBSTRUCTION LOGIC
# =============================================================================

class EuclidObstruction:
    """
    Implements the Euclid Obstruction Form (QS) to detect "Prime Novelty".
    
    Goal: Quantify how much a market state escapes the "Finite Admissible Closure" 
    of currently known physics patterns in the database (i.e. novelty score).
    """
    
    def __init__(self, admissibility_threshold: float = 0.5):
        self.admissibility_threshold = admissibility_threshold
        
    def calculate_obstruction_form(self, similarity_scores: List[float]) -> float:
        """
        Calculates the ℓ0-energy (QS) of the novelty map.
        QS(m) = sum(1[similarity < threshold])
        
        High QS = The state is 'prime' relative to the known physics set.
        """
        if not similarity_scores:
            return 0.0
            
        # Count matches that fail to reach admissibility
        obstructions = [1 for s in similarity_scores if s < self.admissibility_threshold]
        qs_score = sum(obstructions)
        
        # Normalize to [0, 1] relative to the known set size
        return qs_score / len(similarity_scores)

    def is_canonical_extension(self, qs_normalized: float) -> bool:
        """
        Determines if the state represents a 'Forced Extension' (Theorem 1).
        If normalized QS is 1.0, the state is completely outside finite closure.
        """
        return qs_normalized >= 1.0


PHYSICS_INTERPRETATIONS = {
    "MHD_64": PhysicsPattern(
        dataset="MHD_64",
        description="Magnetohydrodynamic waves - coherent wave propagation",
        market_analog="Institutional order flow with momentum",
        signal_type="momentum",
        confidence_weight=1.2
    ),
    "MHD_256": PhysicsPattern(
        dataset="MHD_256",
        description="High-resolution MHD - complex wave interactions",
        market_analog="Multi-timeframe momentum convergence",
        signal_type="momentum",
        confidence_weight=1.3
    ),
    "turbulence_gravity_cooling": PhysicsPattern(
        dataset="turbulence_gravity_cooling",
        description="Turbulent flow with cooling - energy dissipation",
        market_analog="High volatility with mean reversion tendency",
        signal_type="mean_reversion",
        confidence_weight=1.0
    ),
    "gray_scott_reaction_diffusion": PhysicsPattern(
        dataset="gray_scott_reaction_diffusion",
        description="Pattern formation through reaction-diffusion",
        market_analog="Emerging trend with structure formation",
        signal_type="trend",
        confidence_weight=1.1
    ),
    "acoustic_scattering_maze": PhysicsPattern(
        dataset="acoustic_scattering_maze",
        description="Wave scattering through complex geometry",
        market_analog="Order flow navigating support/resistance",
        signal_type="breakout",
        confidence_weight=0.9
    ),
    "acoustic_scattering_inclusions": PhysicsPattern(
        dataset="acoustic_scattering_inclusions",
        description="Wave interaction with obstacles",
        market_analog="Price testing key levels",
        signal_type="bounce",
        confidence_weight=0.9
    ),
    "helmholtz_staircase": PhysicsPattern(
        dataset="helmholtz_staircase",
        description="Stepped wave patterns - quantized states",
        market_analog="Price in consolidation with clear levels",
        signal_type="range",
        confidence_weight=0.8
    ),
    "active_matter": PhysicsPattern(
        dataset="active_matter",
        description="Self-propelled particle dynamics",
        market_analog="Algorithmic trading swarms",
        signal_type="volatility",
        confidence_weight=1.0
    ),
    "supernova_explosion_128": PhysicsPattern(
        dataset="supernova_explosion_128",
        description="Explosive expansion from point source",
        market_analog="News-driven breakout/gap",
        signal_type="breakout",
        confidence_weight=1.4
    ),
    "viscoelastic_instability": PhysicsPattern(
        dataset="viscoelastic_instability",
        description="Stress buildup leading to instability",
        market_analog="Compression before volatility expansion",
        signal_type="volatility",
        confidence_weight=1.3
    ),
}


# =============================================================================
# ORDER FLOW STATE VECTOR
# =============================================================================

@dataclass
class OrderFlowState:
    """Captures the current state of order flow for embedding."""
    timestamp: datetime
    ticker: str
    
    # Price dynamics (normalized)
    price: float
    price_change_1m: float  # 1-minute change
    price_change_5m: float  # 5-minute change
    price_velocity: float   # Rate of change
    price_acceleration: float  # Second derivative
    
    # Volume dynamics
    volume_ratio: float     # Current vs average
    buy_volume_ratio: float # Buy vs sell
    volume_acceleration: float
    
    # Order book shape (if available)
    bid_depth: float        # Normalized bid depth
    ask_depth: float        # Normalized ask depth
    spread: float           # Bid-ask spread
    imbalance: float        # Order book imbalance
    
    # Derived metrics
    vwap_deviation: float   # Distance from VWAP
    volatility: float       # Recent volatility
    momentum: float         # RSI-like momentum
    
    def to_embedding_vector(self) -> np.ndarray:
        """Convert state to 16-dimensional embedding vector."""
        vec = np.array([
            self.price_change_1m,
            self.price_change_5m,
            self.price_velocity,
            self.price_acceleration,
            self.volume_ratio,
            self.buy_volume_ratio,
            self.volume_acceleration,
            self.bid_depth,
            self.ask_depth,
            self.spread,
            self.imbalance,
            self.vwap_deviation,
            self.volatility,
            self.momentum,
            # Derived features
            self.price_velocity * self.volume_ratio,  # Flow = velocity * volume
            self.imbalance * self.volatility,  # Pressure = imbalance * vol
        ], dtype=np.float32)
        
        # Normalize to unit norm for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec


# =============================================================================
# PHYSICS ORDER FLOW ENGINE
# =============================================================================

class PhysicsOrderFlowEngine:
    """
    Maps order flow dynamics to physics patterns for prediction.
    
    Uses GrainVDB trained on physics datasets to find analogous patterns
    in market microstructure.
    """
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.state_history: deque = deque(maxlen=1000)
        self.signal_history: deque = deque(maxlen=100)
        
        # Load GrainVDB
        self.vdb = None
        if GRAINVDB_AVAILABLE:
            try:
                self.vdb = GrainVDB(embed_dim=embedding_dim, max_size=5000)
                self.vdb.load()
                logger.info(f"📖 Loaded GrainVDB with {self.vdb.count} physics patterns")
            except Exception as e:
                logger.warning(f"⚠️ Could not load GrainVDB: {e}")
                self.vdb = None
        
        # Projection layer: 16D order flow -> 256D physics space
        # We use a learned random projection (orthogonal for better geometry)
        np.random.seed(42)  # Reproducible
        self.projection_matrix = self._create_projection_matrix(16, embedding_dim)
        
        # Statistics for normalization
        self.feature_means = np.zeros(16)
        self.feature_stds = np.ones(16)
        self.samples_seen = 0
        
        # Φ Euclid Obstruction (Novelty Detection)
        self.euclid = EuclidObstruction(admissibility_threshold=0.45) # 45% similarity floor
        
        logger.info("🚀 Physics Order Flow Engine initialized")
    
    def _create_projection_matrix(self, input_dim: int, output_dim: int) -> np.ndarray:
        """Create orthogonal random projection matrix."""
        # Start with random matrix
        random_matrix = np.random.randn(input_dim, output_dim).astype(np.float32)
        
        # Orthogonalize using QR decomposition
        q, r = np.linalg.qr(random_matrix.T)
        projection = q.T[:input_dim, :output_dim]
        
        # Scale to preserve norms
        projection *= np.sqrt(output_dim / input_dim)
        
        return projection
    
    def project_to_physics_space(self, order_flow_vec: np.ndarray) -> np.ndarray:
        """Project 16D order flow vector to 256D physics embedding space."""
        # Apply projection
        physics_vec = order_flow_vec @ self.projection_matrix
        
        # L2 normalize for cosine similarity
        norm = np.linalg.norm(physics_vec)
        if norm > 0:
            physics_vec = physics_vec / norm
        
        return physics_vec
    
    def update_statistics(self, order_flow_vec: np.ndarray):
        """Online update of feature statistics for normalization."""
        self.samples_seen += 1
        
        # Welford's online algorithm
        delta = order_flow_vec - self.feature_means
        self.feature_means += delta / self.samples_seen
        delta2 = order_flow_vec - self.feature_means
        variance = (self.feature_stds ** 2) * (self.samples_seen - 1)
        variance += delta * delta2
        if self.samples_seen > 1:
            self.feature_stds = np.sqrt(variance / (self.samples_seen - 1))
    
    def normalize_features(self, order_flow_vec: np.ndarray) -> np.ndarray:
        """Z-score normalize features."""
        normalized = (order_flow_vec - self.feature_means) / (self.feature_stds + 1e-8)
        return normalized
    
    def query_physics_patterns(self, state: OrderFlowState, k: int = 5) -> List[Dict]:
        """
        Query GrainVDB for physics patterns matching current order flow.
        
        Returns list of matches with:
        - physics_pattern: PhysicsPattern with interpretation
        - score: similarity score
        - signal: derived trading signal
        """
        if self.vdb is None or self.vdb.count == 0:
            logger.warning("⚠️ GrainVDB empty or not available")
            return []
        
        # Convert state to embedding
        raw_vec = state.to_embedding_vector()
        self.update_statistics(raw_vec)
        normalized_vec = self.normalize_features(raw_vec)
        physics_vec = self.project_to_physics_space(normalized_vec)
        
        # Query VDB
        matches = self.vdb.search(physics_vec, k=k)
        
        # Interpret matches
        results = []
        for match in matches:
            if match['metadata'] is None:
                continue
            
            dataset = match['metadata'].get('dataset', 'unknown')
            pattern = PHYSICS_INTERPRETATIONS.get(dataset)
            
            if pattern:
                # Compute signal based on pattern type
                signal = self._interpret_signal(pattern, state, match['score'])
                
                results.append({
                    'physics_pattern': pattern,
                    'score': match['score'],
                    'training_loss': match['metadata'].get('loss', None),
                    'signal': signal
                })
        
        return results
    
    def _interpret_signal(self, pattern: PhysicsPattern, state: OrderFlowState, score: float) -> Dict:
        """Interpret physics pattern into trading signal."""
        confidence = score * pattern.confidence_weight
        
        signal = {
            'type': pattern.signal_type,
            'direction': 'neutral',
            'strength': 0.0,
            'confidence': confidence,
            'reasoning': f"Matches {pattern.description}"
        }
        
        # Interpret based on signal type
        if pattern.signal_type == 'momentum':
            # Momentum: follow the current direction if order flow confirms
            if state.momentum > 0.5 and state.imbalance > 0:
                signal['direction'] = 'long'
                signal['strength'] = state.momentum * state.imbalance
            elif state.momentum < -0.5 and state.imbalance < 0:
                signal['direction'] = 'short'
                signal['strength'] = abs(state.momentum * state.imbalance)
            signal['reasoning'] = f"MHD wave pattern: {pattern.market_analog}"
                
        elif pattern.signal_type == 'mean_reversion':
            # Mean reversion: fade extremes
            if state.vwap_deviation > 1.5:
                signal['direction'] = 'short'
                signal['strength'] = min(state.vwap_deviation / 2, 1.0)
            elif state.vwap_deviation < -1.5:
                signal['direction'] = 'long'
                signal['strength'] = min(abs(state.vwap_deviation) / 2, 1.0)
            signal['reasoning'] = f"Turbulence pattern: {pattern.market_analog}"
                
        elif pattern.signal_type == 'trend':
            # Trend: reaction-diffusion suggests emerging structure
            if state.price_velocity > 0 and state.volume_ratio > 1.2:
                signal['direction'] = 'long'
                signal['strength'] = state.price_velocity * 0.5
            elif state.price_velocity < 0 and state.volume_ratio > 1.2:
                signal['direction'] = 'short'
                signal['strength'] = abs(state.price_velocity) * 0.5
            signal['reasoning'] = f"Pattern formation: {pattern.market_analog}"
                
        elif pattern.signal_type == 'breakout':
            # Breakout: supernova-like expansion expected
            if state.volatility < 0.5 and abs(state.imbalance) > 0.3:
                signal['direction'] = 'long' if state.imbalance > 0 else 'short'
                signal['strength'] = abs(state.imbalance)
                signal['reasoning'] = f"Compression before expansion: {pattern.market_analog}"
                
        elif pattern.signal_type == 'volatility':
            # High volatility expected - reduce size or wait
            signal['direction'] = 'neutral'
            signal['strength'] = state.volatility
            signal['reasoning'] = f"Instability detected: {pattern.market_analog}"
        
        signal['strength'] = min(max(signal['strength'], 0), 1)  # Clamp to [0, 1]
        
        return signal
    
    def generate_composite_signal(self, state: OrderFlowState) -> Dict:
        """
        Generate a composite trading signal from all physics pattern matches.
        
        Uses weighted consensus across all matching physics domains.
        """
        matches = self.query_physics_patterns(state, k=5)
        
        if not matches:
            return {
                'direction': 'neutral',
                'strength': 0,
                'confidence': 0,
                'physics_patterns': [],
                'reasoning': 'No physics patterns matched'
            }
        
        # Aggregate signals
        long_score = 0
        short_score = 0
        total_weight = 0
        
        pattern_summaries = []
        
        for match in matches:
            signal = match['signal']
            weight = match['score'] * match['physics_pattern'].confidence_weight
            total_weight += weight
            
            if signal['direction'] == 'long':
                long_score += signal['strength'] * weight
            elif signal['direction'] == 'short':
                short_score += signal['strength'] * weight
            
            pattern_summaries.append({
                'dataset': match['physics_pattern'].dataset,
                'signal_type': match['physics_pattern'].signal_type,
                'score': match['score'],
                'direction': signal['direction'],
                'strength': signal['strength']
            })
        
        # Normalize
        if total_weight > 0:
            long_score /= total_weight
            short_score /= total_weight
        
        # --- Φ EUCLID OBSTRUCTION ANALYSIS ---
        all_scores = [m['score'] for m in matches]
        qs_normalized = self.euclid.calculate_obstruction_form(all_scores)
        prime_extension = self.euclid.is_canonical_extension(qs_normalized)
        
        if prime_extension:
            logger.info("📐 NO FINITE CLOSURE: Prime Extension Event Detected (QS=1.0)")
        
        # Determine composite signal
        if long_score > short_score and long_score > 0.3:
            direction = 'long'
            strength = long_score
        elif short_score > long_score and short_score > 0.3:
            direction = 'short'
            strength = short_score
        else:
            direction = 'neutral'
            strength = 0
        
        # Overall confidence based on pattern agreement and match quality
        avg_score = np.mean(all_scores) if all_scores else 0
        direction_agreement = abs(long_score - short_score)
        
        # Adjust confidence by novelty (High novelty = Lower confidence in existing patterns)
        confidence = avg_score * (0.5 + 0.5 * direction_agreement) * (1.0 - qs_normalized * 0.5)
        
        return {
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'physics_patterns': pattern_summaries,
            'long_score': long_score,
            'short_score': short_score,
            'euclid_obstruction': qs_normalized,
            'prime_extension': prime_extension,
            'reasoning': f"Composite from {len(matches)} physics patterns (QS: {qs_normalized:.2f})"
        }
    
    def analyze_orderbook_shape(self, bids: List[Tuple[float, float]], 
                                 asks: List[Tuple[float, float]]) -> Dict:
        """
        Analyze order book shape and find matching physics patterns.
        
        Args:
            bids: List of (price, size) tuples for bid side
            asks: List of (price, size) tuples for ask side
            
        Returns:
            Physics interpretation of order book shape
        """
        if not bids or not asks:
            return {'error': 'Empty orderbook'}
        
        # Extract shape features
        bid_prices = np.array([b[0] for b in bids])
        bid_sizes = np.array([b[1] for b in bids])
        ask_prices = np.array([a[0] for a in asks])
        ask_sizes = np.array([a[1] for a in asks])
        
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        spread = ask_prices[0] - bid_prices[0]
        
        # Compute gradients (like pressure gradients in fluids)
        bid_gradient = np.gradient(bid_sizes)
        ask_gradient = np.gradient(ask_sizes)
        
        # Compute imbalance
        total_bid = np.sum(bid_sizes)
        total_ask = np.sum(ask_sizes)
        imbalance = (total_bid - total_ask) / (total_bid + total_ask + 1e-8)
        
        # Shape features
        bid_concentration = np.max(bid_sizes) / (np.mean(bid_sizes) + 1e-8)
        ask_concentration = np.max(ask_sizes) / (np.mean(ask_sizes) + 1e-8)
        
        # Map to state
        state = OrderFlowState(
            timestamp=datetime.now(),
            ticker="ORDERBOOK",
            price=mid_price,
            price_change_1m=0,
            price_change_5m=0,
            price_velocity=np.mean(bid_gradient) - np.mean(ask_gradient),
            price_acceleration=0,
            volume_ratio=1.0,
            buy_volume_ratio=total_bid / (total_bid + total_ask + 1e-8),
            volume_acceleration=0,
            bid_depth=total_bid,
            ask_depth=total_ask,
            spread=spread / mid_price,  # Normalized spread
            imbalance=imbalance,
            vwap_deviation=0,
            volatility=abs(np.std(bid_gradient) + np.std(ask_gradient)),
            momentum=imbalance  # Use imbalance as momentum proxy
        )
        
        # Get physics interpretation
        signal = self.generate_composite_signal(state)
        
        return {
            'mid_price': mid_price,
            'spread': spread,
            'imbalance': imbalance,
            'bid_concentration': bid_concentration,
            'ask_concentration': ask_concentration,
            'physics_signal': signal
        }


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo_physics_orderflow():
    """Demonstrate the Physics Order Flow Engine."""
    
    print("=" * 80)
    print("PHYSICS-INFORMED ORDER FLOW ENGINE")
    print("=" * 80)
    print()
    
    engine = PhysicsOrderFlowEngine()
    
    if engine.vdb is None or engine.vdb.count == 0:
        print("⚠️  GrainVDB is empty - generating synthetic demo")
        print("   (Run this after physics training completes for real results)")
        print()
    
    # Create sample order flow states
    scenarios = [
        {
            "name": "Institutional Buying",
            "state": OrderFlowState(
                timestamp=datetime.now(),
                ticker="SPY",
                price=500.0,
                price_change_1m=0.1,
                price_change_5m=0.3,
                price_velocity=0.5,
                price_acceleration=0.1,
                volume_ratio=1.8,
                buy_volume_ratio=0.7,
                volume_acceleration=0.3,
                bid_depth=1.5,
                ask_depth=0.8,
                spread=0.01,
                imbalance=0.4,
                vwap_deviation=0.5,
                volatility=0.3,
                momentum=0.7
            )
        },
        {
            "name": "Panic Selling",
            "state": OrderFlowState(
                timestamp=datetime.now(),
                ticker="SPY",
                price=490.0,
                price_change_1m=-0.5,
                price_change_5m=-1.2,
                price_velocity=-1.0,
                price_acceleration=-0.3,
                volume_ratio=3.0,
                buy_volume_ratio=0.2,
                volume_acceleration=0.8,
                bid_depth=0.3,
                ask_depth=2.0,
                spread=0.05,
                imbalance=-0.6,
                vwap_deviation=-2.0,
                volatility=1.5,
                momentum=-0.9
            )
        },
        {
            "name": "Pre-Breakout Compression",
            "state": OrderFlowState(
                timestamp=datetime.now(),
                ticker="QQQ",
                price=420.0,
                price_change_1m=0.0,
                price_change_5m=0.1,
                price_velocity=0.05,
                price_acceleration=0.0,
                volume_ratio=0.5,
                buy_volume_ratio=0.52,
                volume_acceleration=0.0,
                bid_depth=1.0,
                ask_depth=1.0,
                spread=0.005,
                imbalance=0.35,
                vwap_deviation=0.1,
                volatility=0.1,
                momentum=0.1
            )
        },
        {
            "name": "Mean Reversion Setup",
            "state": OrderFlowState(
                timestamp=datetime.now(),
                ticker="AAPL",
                price=180.0,
                price_change_1m=0.2,
                price_change_5m=1.5,
                price_velocity=0.3,
                price_acceleration=-0.1,
                volume_ratio=1.2,
                buy_volume_ratio=0.55,
                volume_acceleration=-0.2,
                bid_depth=1.2,
                ask_depth=1.4,
                spread=0.02,
                imbalance=0.1,
                vwap_deviation=2.0,
                volatility=0.8,
                momentum=0.3
            )
        }
    ]
    
    for scenario in scenarios:
        print("-" * 80)
        print(f"📊 Scenario: {scenario['name']}")
        print("-" * 80)
        
        state = scenario['state']
        signal = engine.generate_composite_signal(state)
        
        # Display results
        direction_emoji = {
            'long': '🟢 LONG',
            'short': '🔴 SHORT',
            'neutral': '⚪ NEUTRAL'
        }
        
        print(f"   Ticker: {state.ticker}")
        print(f"   Price: ${state.price:.2f}")
        print(f"   Momentum: {state.momentum:+.2f}")
        print(f"   Imbalance: {state.imbalance:+.2f}")
        print(f"   VWAP Dev: {state.vwap_deviation:+.2f}")
        print()
        
        print(f"   🧲 Physics Patterns Matched:")
        if signal['physics_patterns']:
            for p in signal['physics_patterns']:
                print(f"      • {p['dataset']}: {p['signal_type']} → {p['direction']} ({p['score']:.2f})")
        else:
            print(f"      (No patterns matched - VDB may be empty)")
        
        print()
        print(f"   📈 COMPOSITE SIGNAL: {direction_emoji[signal['direction']]}")
        print(f"      Strength: {'█' * int(signal['strength'] * 10)}{' ' * (10 - int(signal['strength'] * 10))} {signal['strength']:.2f}")
        print(f"      Confidence: {signal['confidence']*100:.1f}%")
        print(f"      Long Score: {signal.get('long_score', 0):.2f} | Short Score: {signal.get('short_score', 0):.2f}")
        print()
    
    print("=" * 80)
    print("ENGINE DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("1. Wait for physics training to complete (GrainVDB population)")
    print("2. Connect to live order flow data")
    print("3. Backtest signals against historical data")
    print("4. Paper trade before going live")


if __name__ == "__main__":
    demo_physics_orderflow()
