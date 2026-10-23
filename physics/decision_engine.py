"""
DECISION ENGINE - Algorithms for Decision Making Integration
=============================================================
Implements core algorithms from Kochenderfer et al. (MIT Press, 2022):

1. Particle Filter (Ch. 19.6) - Regime tracking with proper belief updates
2. Value of Information (Ch. 6.6) - Should we run expensive analysis?
3. Bayesian Confidence (Ch. 4) - Proper posterior belief calculation

This replaces heuristic confidence with principled probabilistic reasoning.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import time
import logging
import math
from statistics import NormalDist

logger = logging.getLogger("DecisionEngine")

# =============================================================================
# 1. MARKET REGIME DEFINITIONS (Hidden States)
# =============================================================================

class MarketRegime(Enum):
    """Hidden market states we want to infer"""
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    MEAN_REVERT = "MEAN_REVERT"
    HIGH_VOL = "HIGH_VOL"
    LOW_VOL = "LOW_VOL"
    TRANSITION = "TRANSITION"

# Regime transition probabilities (learned from historical data)
# P(next_regime | current_regime) - Markov transition matrix
REGIME_TRANSITIONS = {
    MarketRegime.TREND_UP: {
        MarketRegime.TREND_UP: 0.70,
        MarketRegime.TREND_DOWN: 0.05,
        MarketRegime.MEAN_REVERT: 0.10,
        MarketRegime.HIGH_VOL: 0.05,
        MarketRegime.LOW_VOL: 0.05,
        MarketRegime.TRANSITION: 0.05,
    },
    MarketRegime.TREND_DOWN: {
        MarketRegime.TREND_UP: 0.05,
        MarketRegime.TREND_DOWN: 0.70,
        MarketRegime.MEAN_REVERT: 0.10,
        MarketRegime.HIGH_VOL: 0.05,
        MarketRegime.LOW_VOL: 0.05,
        MarketRegime.TRANSITION: 0.05,
    },
    MarketRegime.MEAN_REVERT: {
        MarketRegime.TREND_UP: 0.15,
        MarketRegime.TREND_DOWN: 0.15,
        MarketRegime.MEAN_REVERT: 0.50,
        MarketRegime.HIGH_VOL: 0.05,
        MarketRegime.LOW_VOL: 0.10,
        MarketRegime.TRANSITION: 0.05,
    },
    MarketRegime.HIGH_VOL: {
        MarketRegime.TREND_UP: 0.10,
        MarketRegime.TREND_DOWN: 0.10,
        MarketRegime.MEAN_REVERT: 0.15,
        MarketRegime.HIGH_VOL: 0.50,
        MarketRegime.LOW_VOL: 0.05,
        MarketRegime.TRANSITION: 0.10,
    },
    MarketRegime.LOW_VOL: {
        MarketRegime.TREND_UP: 0.10,
        MarketRegime.TREND_DOWN: 0.10,
        MarketRegime.MEAN_REVERT: 0.20,
        MarketRegime.HIGH_VOL: 0.05,
        MarketRegime.LOW_VOL: 0.50,
        MarketRegime.TRANSITION: 0.05,
    },
    MarketRegime.TRANSITION: {
        MarketRegime.TREND_UP: 0.20,
        MarketRegime.TREND_DOWN: 0.20,
        MarketRegime.MEAN_REVERT: 0.20,
        MarketRegime.HIGH_VOL: 0.15,
        MarketRegime.LOW_VOL: 0.15,
        MarketRegime.TRANSITION: 0.10,
    },
}


# =============================================================================
# 2. PARTICLE FILTER FOR REGIME TRACKING (Ch. 19.6)
# =============================================================================

@dataclass
class Particle:
    """A single particle representing a hypothesis about the market state"""
    regime: MarketRegime
    weight: float = 1.0
    # Additional state variables tracked per particle
    momentum: float = 0.0
    volatility: float = 0.01
    trend_strength: float = 0.0


class ParticleFilterRegimeTracker:
    """
    Particle Filter for Market Regime Detection
    
    From Ch. 19.6 of "Algorithms for Decision Making":
    - Maintains N particles representing belief distribution over regimes
    - Updates weights based on observation likelihood
    - Resamples when effective sample size drops
    
    This replaces heuristic regime detection with proper Bayesian inference.
    """
    
    def __init__(self, n_particles: int = 500, resample_threshold: float = 0.5):
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.particles: List[Particle] = []
        self.history: List[Dict] = []
        self._initialize_particles()
        
    def _initialize_particles(self):
        """Initialize particles uniformly across regimes"""
        regimes = list(MarketRegime)
        base = self.n_particles // len(regimes)
        rem = self.n_particles - base * len(regimes)
        
        self.particles = []
        for i, regime in enumerate(regimes):
            n = base + (1 if i < rem else 0)
            for _ in range(n):
                self.particles.append(Particle(
                    regime=regime,
                    weight=1.0 / self.n_particles,
                    momentum=np.random.normal(0, 0.01),
                    volatility=np.random.uniform(0.005, 0.03),
                    trend_strength=np.random.uniform(-0.5, 0.5)
                ))
        
        # Normalize weights
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize particle weights to sum to 1"""
        total = sum(p.weight for p in self.particles)
        if total > 0:
            for p in self.particles:
                p.weight /= total
    
    def _compute_observation_likelihood(self, particle: Particle, 
                                         returns: np.ndarray) -> float:
        """
        P(observation | hidden state)
        
        Compute how likely the observed returns are given the particle's regime hypothesis.
        This is the measurement model.
        """
        if len(returns) < 2:
            return 1.0
        
        ret_mean = np.mean(returns)
        ret_std = np.std(returns) + 1e-9
        
        likelihood = 1.0
        
        if particle.regime == MarketRegime.TREND_UP:
            # Expect positive returns, moderate volatility
            likelihood *= np.exp(-((ret_mean - 0.002) ** 2) / (2 * 0.005 ** 2))
            likelihood *= np.exp(-((ret_std - 0.01) ** 2) / (2 * 0.01 ** 2))
            
        elif particle.regime == MarketRegime.TREND_DOWN:
            # Expect negative returns, moderate volatility
            likelihood *= np.exp(-((ret_mean + 0.002) ** 2) / (2 * 0.005 ** 2))
            likelihood *= np.exp(-((ret_std - 0.01) ** 2) / (2 * 0.01 ** 2))
            
        elif particle.regime == MarketRegime.MEAN_REVERT:
            # Expect returns near zero, low volatility
            likelihood *= np.exp(-((ret_mean) ** 2) / (2 * 0.002 ** 2))
            likelihood *= np.exp(-((ret_std - 0.005) ** 2) / (2 * 0.005 ** 2))
            
        elif particle.regime == MarketRegime.HIGH_VOL:
            # Expect high volatility, any direction
            likelihood *= np.exp(-((ret_std - 0.025) ** 2) / (2 * 0.015 ** 2))
            
        elif particle.regime == MarketRegime.LOW_VOL:
            # Expect low volatility
            likelihood *= np.exp(-((ret_std - 0.003) ** 2) / (2 * 0.003 ** 2))
            
        elif particle.regime == MarketRegime.TRANSITION:
            # High uncertainty - flat likelihood
            likelihood *= 0.5
        
        return max(likelihood, 1e-10)
    
    def _transition_particle(self, particle: Particle) -> Particle:
        """
        Sample next state from transition model P(s' | s)
        """
        current_regime = particle.regime
        transition_probs = REGIME_TRANSITIONS[current_regime]
        
        regimes = list(transition_probs.keys())
        probs = list(transition_probs.values())
        
        # Sample next regime
        next_regime = np.random.choice(regimes, p=probs)
        
        # Evolve continuous state variables with noise
        new_momentum = particle.momentum * 0.9 + np.random.normal(0, 0.005)
        new_volatility = particle.volatility * 0.95 + np.random.uniform(0, 0.01)
        new_trend = particle.trend_strength * 0.8 + np.random.normal(0, 0.1)
        
        return Particle(
            regime=next_regime,
            weight=particle.weight,
            momentum=np.clip(new_momentum, -0.05, 0.05),
            volatility=np.clip(new_volatility, 0.001, 0.1),
            trend_strength=np.clip(new_trend, -1.0, 1.0)
        )
    
    def _effective_sample_size(self) -> float:
        """
        Compute effective sample size (ESS) to determine if resampling is needed.
        ESS = 1 / sum(w_i^2)
        """
        weights_squared = sum(p.weight ** 2 for p in self.particles)
        return 1.0 / (weights_squared + 1e-10)
    
    def _resample(self):
        """
        Systematic resampling (low variance resampling from Ch. 19.6)
        """
        n = len(self.particles)
        weights = [p.weight for p in self.particles]
        
        # Cumulative sum
        cumsum = np.cumsum(weights)
        
        # Starting point
        step = 1.0 / n
        u = np.random.uniform(0, step)
        
        new_particles = []
        j = 0
        
        for i in range(n):
            target = u + i * step
            while j < n - 1 and cumsum[j] < target:
                j += 1
            
            # Clone particle with equal weight
            old_p = self.particles[j]
            new_particles.append(Particle(
                regime=old_p.regime,
                weight=1.0 / n,
                momentum=old_p.momentum + np.random.normal(0, 0.001),  # Jitter
                volatility=old_p.volatility + np.random.uniform(-0.001, 0.001),
                trend_strength=old_p.trend_strength + np.random.normal(0, 0.02)
            ))
        
        self.particles = new_particles
    
    def update(self, returns: np.ndarray) -> Dict:
        """
        Main particle filter update step.
        
        Algorithm from Ch. 19.6:
        1. Predict: Propagate particles through transition model
        2. Update: Weight particles by observation likelihood
        3. Resample: If ESS too low, resample particles
        
        Args:
            returns: Recent log returns (last 10-20 observations)
            
        Returns:
            Dict with regime beliefs and confidence
        """
        # 1. PREDICT: Transition each particle
        self.particles = [self._transition_particle(p) for p in self.particles]
        
        # 2. UPDATE: Weight by observation likelihood
        for particle in self.particles:
            likelihood = self._compute_observation_likelihood(particle, returns)
            particle.weight *= likelihood
        
        # Normalize weights
        self._normalize_weights()
        
        # 3. RESAMPLE if needed
        ess = self._effective_sample_size()
        if ess < self.resample_threshold * self.n_particles:
            self._resample()
        
        # Compute regime probabilities (belief distribution)
        regime_beliefs = self._compute_regime_beliefs()
        
        # Compute confidence as max belief
        max_belief = max(regime_beliefs.values())
        entropy = -sum(p * np.log(p + 1e-10) for p in regime_beliefs.values())
        max_entropy = np.log(len(MarketRegime))
        
        # Confidence: 1 when one regime dominates, 0 when uniform
        confidence = 1.0 - (entropy / max_entropy)
        
        # Get most likely regime
        most_likely_regime = max(regime_beliefs, key=regime_beliefs.get)
        
        result = {
            'regime': most_likely_regime.value,
            'regime_beliefs': {k.value: v for k, v in regime_beliefs.items()},
            'confidence': confidence,
            'effective_sample_size': ess / self.n_particles,
            'timestamp': time.time()
        }
        
        self.history.append(result)
        if len(self.history) > 1000:
            self.history = self.history[-500:]
        
        return result
    
    def _compute_regime_beliefs(self) -> Dict[MarketRegime, float]:
        """Compute belief distribution over regimes from particle weights"""
        beliefs = {regime: 0.0 for regime in MarketRegime}
        
        for particle in self.particles:
            beliefs[particle.regime] += particle.weight
        
        return beliefs
    
    def get_recommended_action(self) -> Tuple[str, float]:
        """
        Map regime belief to recommended trading action.
        
        Returns:
            (action, confidence) tuple
        """
        beliefs = self._compute_regime_beliefs()
        
        # Weighted action scores
        trend_score = beliefs[MarketRegime.TREND_UP] - beliefs[MarketRegime.TREND_DOWN]
        revert_score = beliefs[MarketRegime.MEAN_REVERT] + beliefs[MarketRegime.LOW_VOL]
        volatile_score = beliefs[MarketRegime.HIGH_VOL] + beliefs[MarketRegime.TRANSITION]
        
        if volatile_score > 0.4:
            return ("REDUCE_EXPOSURE", volatile_score)
        elif abs(trend_score) > 0.3:
            action = "TREND_FOLLOW" if trend_score > 0 else "TREND_FOLLOW_SHORT"
            return (action, abs(trend_score))
        elif revert_score > 0.3:
            return ("MEAN_REVERT", revert_score)
        else:
            return ("HOLD", 0.5)


# =============================================================================
# 3. VALUE OF INFORMATION (Ch. 6.6)
# =============================================================================

class ValueOfInformation:
    """
    Computes whether additional analysis is worth the computational cost.
    
    From Ch. 6.6: VOI = E[U(a* | new info)] - E[U(a* | current info)]
    
    If VOI > cost of acquiring info, then acquire it.
    """
    
    def __init__(self, analysis_cost: float = 0.01):
        """
        Args:
            analysis_cost: Relative cost of running analysis (in utility units)
        """
        self.analysis_cost = analysis_cost
        self.history: List[Dict] = []
    
    def compute_voi(self, 
                    current_confidence: float,
                    expected_info_gain: float,
                    potential_pnl: float) -> Dict:
        """
        Compute Value of Information for running additional analysis.
        
        Args:
            current_confidence: Current confidence in trading decision [0, 1]
            expected_info_gain: Expected reduction in uncertainty from analysis
            potential_pnl: Potential P&L if decision is correct
            
        Returns:
            Dict with VOI, recommendation, and reasoning
        """
        # Current expected utility (simplified)
        # U(correct) = potential_pnl, U(incorrect) = -potential_pnl (symmetric)
        current_eu = current_confidence * potential_pnl - (1 - current_confidence) * potential_pnl
        current_eu = (2 * current_confidence - 1) * potential_pnl
        
        # Expected utility with additional info
        # Assume info improves confidence toward 1 or 0 (more decisive)
        improved_confidence = min(1.0, current_confidence + expected_info_gain * (1 - current_confidence))
        improved_eu = (2 * improved_confidence - 1) * potential_pnl
        
        # Value of information
        voi = improved_eu - current_eu
        
        # Decision
        should_analyze = voi > self.analysis_cost
        
        result = {
            'voi': voi,
            'current_eu': current_eu,
            'improved_eu': improved_eu,
            'analysis_cost': self.analysis_cost,
            'should_analyze': should_analyze,
            'reasoning': self._generate_reasoning(voi, self.analysis_cost, current_confidence)
        }
        
        self.history.append(result)
        
        return result
    
    def _generate_reasoning(self, voi: float, cost: float, confidence: float) -> str:
        """Generate human-readable reasoning"""
        if voi <= 0:
            return "VOI ≤ 0: Already confident enough, skip analysis."
        elif voi > cost * 2:
            return f"VOI ({voi:.4f}) >> Cost ({cost:.4f}): Strongly recommend analysis."
        elif voi > cost:
            return f"VOI ({voi:.4f}) > Cost ({cost:.4f}): Analysis worthwhile."
        else:
            return f"VOI ({voi:.4f}) < Cost ({cost:.4f}): Skip analysis, too expensive."
    
    def should_run_deep_analysis(self, 
                                  trigger_distance_std: float,
                                  current_pnl_pct: float,
                                  time_in_trade_mins: float) -> bool:
        """
        Specific decision: Should we run expensive holographic/spectral analysis?
        
        Args:
            trigger_distance_std: How far from VWAP in std devs
            current_pnl_pct: Current unrealized P&L %
            time_in_trade_mins: How long the position has been open
        """
        # Heuristic confidence based on distance
        if abs(trigger_distance_std) > 2.5:
            current_confidence = 0.8  # Strong signal
        elif abs(trigger_distance_std) > 1.5:
            current_confidence = 0.6
        else:
            current_confidence = 0.4
        
        # Expected info gain depends on distance (more uncertain = more info gain)
        expected_info_gain = max(0.1, 0.5 - current_confidence)
        
        # Potential P&L scales with how far from VWAP
        potential_pnl = abs(trigger_distance_std) * 0.5  # 0.5% per std dev
        
        voi_result = self.compute_voi(current_confidence, expected_info_gain, potential_pnl)
        
        return voi_result['should_analyze']


# =============================================================================
# 4. BAYESIAN CONFIDENCE UPDATER (Ch. 4)
# =============================================================================

class BayesianConfidenceUpdater:
    """
    Proper Bayesian updating of confidence based on observations.
    
    Replaces heuristic confidence = koopman_confidence * fiedler_purity
    with proper posterior calculation.
    """
    
    def __init__(self, prior_alpha: float = 2.0, prior_beta: float = 2.0):
        """
        Initialize with Beta prior on success probability.
        
        Args:
            prior_alpha: Prior successes + 1
            prior_beta: Prior failures + 1
        """
        self.alpha = prior_alpha
        self.beta = prior_beta
        self.observation_history: List[bool] = []
    
    def update(self, success: bool):
        """
        Update beliefs with new observation.
        
        Args:
            success: Whether the prediction was correct
        """
        if success:
            self.alpha += 1
        else:
            self.beta += 1
        
        self.observation_history.append(success)
        
        # Limit history to prevent unbounded growth
        if len(self.observation_history) > 1000:
            # Decay old observations
            self.alpha = max(2.0, self.alpha * 0.99)
            self.beta = max(2.0, self.beta * 0.99)
    
    def get_confidence(self) -> float:
        """
        Get posterior mean (expected success rate).
        
        E[θ] = α / (α + β) for Beta(α, β)
        """
        return self.alpha / (self.alpha + self.beta)
    
    def get_confidence_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """
        Get credible interval for confidence using normal approximation.
        """
        mean = self.get_confidence()
        var = self.get_uncertainty()
        alpha_tail = (1.0 - level) / 2.0
        z = NormalDist().inv_cdf(1.0 - alpha_tail)
        std = math.sqrt(max(0.0, var))
        return (max(0.0, mean - z * std), min(1.0, mean + z * std))
    
    def get_uncertainty(self) -> float:
        """
        Get posterior variance (uncertainty in confidence).
        
        Var[θ] = αβ / [(α+β)²(α+β+1)]
        """
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total ** 2 * (total + 1))
    
    def should_trust_prediction(self, threshold: float = 0.6) -> Tuple[bool, str]:
        """
        Decide whether to trust the current prediction system.
        
        Args:
            threshold: Minimum confidence to trust
            
        Returns:
            (should_trust, reasoning)
        """
        confidence = self.get_confidence()
        uncertainty = self.get_uncertainty()
        
        if confidence >= threshold and uncertainty < 0.05:
            return (True, f"Confidence {confidence:.2%} ≥ {threshold:.0%}, low uncertainty")
        elif confidence >= threshold:
            return (True, f"Confidence {confidence:.2%} ≥ {threshold:.0%}, but uncertain")
        else:
            return (False, f"Confidence {confidence:.2%} < {threshold:.0%}")


# =============================================================================
# 5. INTEGRATED DECISION ENGINE
# =============================================================================

class DecisionEngine:
    """
    Unified decision engine combining all components.
    
    Integrates:
    - Particle filter for regime detection
    - Value of Information for analysis decisions
    - Bayesian confidence for trust calibration
    """
    
    def __init__(self, n_particles: int = 500):
        self.regime_tracker = ParticleFilterRegimeTracker(n_particles=n_particles)
        self.voi_calculator = ValueOfInformation(analysis_cost=0.01)
        self.confidence_updater = BayesianConfidenceUpdater()
        
        self.last_update = 0
        self.update_count = 0
    
    def process_observation(self, 
                            price_history: List[float],
                            prediction_outcome: Optional[bool] = None) -> Dict:
        """
        Main entry point for processing new market data.
        
        Args:
            price_history: Recent prices (last 20-50 observations)
            prediction_outcome: If available, whether last prediction was correct
            
        Returns:
            Comprehensive decision state
        """
        # 1. Compute returns
        if len(price_history) < 5:
            return {'error': 'Insufficient data', 'regime': 'UNKNOWN'}
        
        prices = np.array(price_history, dtype=np.float64)
        returns = np.diff(np.log(prices + 1e-9))
        
        # 2. Update regime tracker
        regime_result = self.regime_tracker.update(returns[-20:] if len(returns) >= 20 else returns)
        
        # 3. Update confidence if outcome provided
        if prediction_outcome is not None:
            self.confidence_updater.update(prediction_outcome)
        
        # 4. Get recommended action
        action, action_confidence = self.regime_tracker.get_recommended_action()
        
        # 5. Get Bayesian confidence
        bayesian_confidence = self.confidence_updater.get_confidence()
        
        # 6. Combine into unified confidence
        # Geometric mean of particle filter confidence and Bayesian confidence
        unified_confidence = np.sqrt(regime_result['confidence'] * bayesian_confidence)
        
        self.update_count += 1
        self.last_update = time.time()
        
        return {
            'regime': regime_result['regime'],
            'regime_beliefs': regime_result['regime_beliefs'],
            'particle_confidence': regime_result['confidence'],
            'bayesian_confidence': bayesian_confidence,
            'unified_confidence': unified_confidence,
            'recommended_action': action,
            'action_confidence': action_confidence,
            'effective_sample_size': regime_result['effective_sample_size'],
            'update_count': self.update_count,
            'timestamp': self.last_update
        }
    
    def should_run_analysis(self, 
                            trigger_distance: float, 
                            current_pnl: float) -> bool:
        """Check if expensive analysis is worth running"""
        return self.voi_calculator.should_run_deep_analysis(
            trigger_distance_std=trigger_distance,
            current_pnl_pct=current_pnl,
            time_in_trade_mins=0  # Simplified
        )

    def get_decision(self, price: float, returns: List[float], volatility: float) -> Dict:
        """
        Wrapper to allow direct feeding of pre-computed returns and volatility
        """
        # If returns is empty, default to zero returns
        if not returns or len(returns) == 0:
            returns = [0.0]
        # Update particle filter with returns
        regime_result = self.regime_tracker.update(np.array(returns))
        action, action_confidence = self.regime_tracker.get_recommended_action()
        bayesian_confidence = self.confidence_updater.get_confidence()
        
        # Calculate Value of Information based on price/volatility
        # Estimate expected utility
        voi_result = self.voi_calculator.compute_voi(
            current_confidence=regime_result['confidence'],
            expected_info_gain=max(0.1, 0.5 - regime_result['confidence']),
            potential_pnl=volatility
        )
        
        return {
            'regime': regime_result['regime'],
            'regime_beliefs': regime_result['regime_beliefs'],
            'regime_confidence': regime_result['confidence'],
            'bayesian_confidence': bayesian_confidence,
            'action': action,
            'voi': voi_result['voi']
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_decision_engine: Optional[DecisionEngine] = None

def get_decision_engine() -> DecisionEngine:
    """Get or create the singleton decision engine"""
    global _decision_engine
    if _decision_engine is None:
        _decision_engine = DecisionEngine(n_particles=500)
        logger.info("Initialized DecisionEngine with 500 particles")
    return _decision_engine


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DECISION ENGINE TEST - Algorithms for Decision Making Integration")
    print("=" * 70)
    
    engine = get_decision_engine()
    
    # Simulate price history
    np.random.seed(42)
    base_price = 21000
    prices = [base_price]
    
    # Generate trending up data
    for _ in range(100):
        ret = np.random.normal(0.001, 0.01)  # Slight upward drift
        prices.append(prices[-1] * (1 + ret))
    
    print("\n--- Testing with UPTREND data ---")
    result = engine.process_observation(prices)
    print(f"Detected Regime: {result['regime']}")
    print(f"Regime Beliefs: {result['regime_beliefs']}")
    print(f"Particle Confidence: {result['particle_confidence']:.2%}")
    print(f"Recommended Action: {result['recommended_action']}")
    print(f"Action Confidence: {result['action_confidence']:.2%}")
    
    # Simulate mean-reverting data
    prices_mr = [base_price]
    for i in range(100):
        mean_ret = -0.01 * (prices_mr[-1] - base_price) / base_price
        ret = mean_ret + np.random.normal(0, 0.005)
        prices_mr.append(prices_mr[-1] * (1 + ret))
    
    print("\n--- Testing with MEAN REVERT data ---")
    engine2 = DecisionEngine(n_particles=500)
    for _ in range(5):  # Multiple updates to converge
        result2 = engine2.process_observation(prices_mr)
    print(f"Detected Regime: {result2['regime']}")
    print(f"Regime Beliefs: {result2['regime_beliefs']}")
    print(f"Recommended Action: {result2['recommended_action']}")
    
    # Test VOI
    print("\n--- Testing Value of Information ---")
    voi_calc = ValueOfInformation()
    voi_result = voi_calc.compute_voi(
        current_confidence=0.6,
        expected_info_gain=0.2,
        potential_pnl=0.02
    )
    print(f"VOI: {voi_result['voi']:.4f}")
    print(f"Should Analyze: {voi_result['should_analyze']}")
    print(f"Reasoning: {voi_result['reasoning']}")
    
    print("\n✅ Decision Engine Test Complete")
