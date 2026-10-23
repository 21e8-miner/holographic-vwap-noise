#!/usr/bin/env python3
"""
SPECTRAL ORDER FLOW OPERATOR
==============================
Applies Fourier/Spectral analysis to order book data, then queries GrainVDB
for matching physics patterns to predict evolution.

This is the core "Holographic Neural Operator" implementation that bridges
financial microstructure to learned physical laws.

Key Concepts:
- Order book liquidity curve → Spectral representation (FFT)
- Query GrainVDB for similar physics patterns (MHD waves, turbulence, etc.)
- Apply retrieved "physics operator" to evolve the spectral state
- Inverse FFT to get predicted market state

Created: January 17, 2026
"""

import sys
import os
import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, fftshift
from scipy.signal import find_peaks, savgol_filter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRESH_BOT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "fresh_bot", "python", "lab"))
sys.path.insert(0, FRESH_BOT_PATH)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("SpectralOperator")

# Try to import GrainVDB
try:
    from grain_vdb import GrainVDB
    GRAINVDB_AVAILABLE = True
except ImportError:
    GRAINVDB_AVAILABLE = False
    logger.warning("GrainVDB not available - using simulation mode")

# Import Monster Resonance module
try:
    from monster_resonance import MONSTER_WAVELENGTH, MonsterResonanceDetector
    MONSTER_LOGIC_AVAILABLE = True
except ImportError:
    MONSTER_LOGIC_AVAILABLE = False
    logger.warning("MonsterResonance module not found")


# =============================================================================
# SPECTRAL FEATURE EXTRACTION
# =============================================================================

@dataclass
class SpectralSignature:
    """Spectral representation of an order book state."""
    timestamp: datetime
    ticker: str
    
    # Frequency domain features
    power_spectrum: np.ndarray      # |FFT|^2 - energy distribution
    phase_spectrum: np.ndarray      # Phase angles
    dominant_frequencies: List[float]  # Peak frequencies
    spectral_centroid: float        # "Center of mass" of spectrum
    spectral_bandwidth: float       # Spread of spectrum
    spectral_rolloff: float         # Frequency below which 85% energy
    spectral_flatness: float        # Noisiness (1=white noise, 0=tonal)
    
    # Derived physics-like features
    energy_density: float           # Total spectral energy
    coherence_score: float          # Phase coherence (like physics coherence)
    entropy: float                  # Spectral entropy
    
    def to_embedding(self, target_dim: int = 256) -> np.ndarray:
        """Convert to fixed-size embedding for GrainVDB query."""
        # Combine all features into a feature vector
        features = [
            self.spectral_centroid,
            self.spectral_bandwidth,
            self.spectral_rolloff,
            self.spectral_flatness,
            self.energy_density,
            self.coherence_score,
            self.entropy,
            len(self.dominant_frequencies),
        ]
        
        # Add power spectrum (resample to fixed size)
        ps_resampled = np.interp(
            np.linspace(0, 1, target_dim - len(features) - 10),
            np.linspace(0, 1, len(self.power_spectrum)),
            self.power_spectrum
        )
        
        # Add top 10 dominant frequencies (padded)
        dom_freq = np.zeros(10)
        for i, f in enumerate(self.dominant_frequencies[:10]):
            dom_freq[i] = f
        
        embedding = np.concatenate([
            np.array(features),
            ps_resampled,
            dom_freq
        ]).astype(np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            # Optionally normalize by Monster Wavelength for universal scaling
            scale = MONSTER_WAVELENGTH if MONSTER_LOGIC_AVAILABLE else 1.0
            embedding = (embedding / norm) * scale
        
        return embedding


class SpectralExtractor:
    """
    Extracts spectral features from order book and price data.
    
    Maps financial waveforms to the same "spectral space" as our 
    physics patterns (MHD waves, turbulence, etc.).
    """
    
    def __init__(self, sample_rate: float = 1.0):
        """
        Args:
            sample_rate: Samples per second (for frequency scaling)
        """
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
        
    def extract_1d_spectrum(self, signal: np.ndarray) -> SpectralSignature:
        """
        Extract spectral features from a 1D signal (e.g., price series).
        
        Args:
            signal: 1D numpy array of values
            
        Returns:
            SpectralSignature with all computed features
        """
        n = len(signal)
        
        # Detrend and normalize
        signal = signal - np.mean(signal)
        std = np.std(signal)
        if std > 0:
            signal = signal / std
        
        # Apply Hanning window to reduce spectral leakage
        windowed = signal * np.hanning(n)
        
        # Compute FFT
        fft_result = fft(windowed)
        
        # Power spectrum (one-sided)
        power_spectrum = np.abs(fft_result[:n//2])**2
        power_spectrum = power_spectrum / np.sum(power_spectrum + 1e-10)  # Normalize
        
        # Phase spectrum
        phase_spectrum = np.angle(fft_result[:n//2])
        
        # Frequency axis
        freqs = np.fft.fftfreq(n, d=1/self.sample_rate)[:n//2]
        
        # Find dominant frequencies (peaks in power spectrum)
        peaks, properties = find_peaks(power_spectrum, height=0.01, distance=3)
        dominant_frequencies = freqs[peaks].tolist() if len(peaks) > 0 else [0.0]
        
        # Spectral centroid (center of mass)
        spectral_centroid = np.sum(freqs * power_spectrum) / (np.sum(power_spectrum) + 1e-10)
        
        # Spectral bandwidth (spread around centroid)
        spectral_bandwidth = np.sqrt(
            np.sum(((freqs - spectral_centroid)**2) * power_spectrum) / 
            (np.sum(power_spectrum) + 1e-10)
        )
        
        # Spectral rolloff (85% energy)
        cumsum = np.cumsum(power_spectrum)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        spectral_rolloff = freqs[min(rolloff_idx, len(freqs)-1)]
        
        # Spectral flatness (geometric mean / arithmetic mean)
        log_power = np.log(power_spectrum + 1e-10)
        geometric_mean = np.exp(np.mean(log_power))
        arithmetic_mean = np.mean(power_spectrum)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        # Energy density
        energy_density = np.sum(power_spectrum)
        
        # Phase coherence (measure of phase consistency)
        # High coherence = organized wave, Low = noise
        phase_diff = np.diff(phase_spectrum)
        phase_diff = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi  # Wrap to [-π, π]
        coherence_score = 1.0 - np.std(phase_diff) / np.pi
        
        # Spectral entropy
        ps_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        entropy = -np.sum(ps_norm * np.log2(ps_norm + 1e-10))
        
        return SpectralSignature(
            timestamp=datetime.now(),
            ticker="",
            power_spectrum=power_spectrum,
            phase_spectrum=phase_spectrum,
            dominant_frequencies=dominant_frequencies,
            spectral_centroid=spectral_centroid,
            spectral_bandwidth=spectral_bandwidth,
            spectral_rolloff=spectral_rolloff,
            spectral_flatness=spectral_flatness,
            energy_density=energy_density,
            coherence_score=coherence_score,
            entropy=entropy
        )
    
    def extract_orderbook_spectrum(self, 
                                    bids: List[Tuple[float, float]],
                                    asks: List[Tuple[float, float]]) -> SpectralSignature:
        """
        Extract spectral features from L2 order book.
        
        Treats the order book as a "liquidity wave" where:
        - X-axis = price levels
        - Y-axis = liquidity (size)
        
        Low frequencies = broad liquidity patterns
        High frequencies = sharp walls/cliffs
        """
        if not bids or not asks:
            raise ValueError("Empty order book")
        
        # Create unified liquidity curve
        # Bids (negative distance from mid) | Mid | Asks (positive distance)
        mid_price = (bids[0][0] + asks[0][0]) / 2
        
        # Normalize prices to distance from mid
        bid_curve = [(mid_price - b[0], b[1]) for b in bids]
        ask_curve = [(a[0] - mid_price, a[1]) for a in asks]
        
        # Create symmetric liquidity array
        # Sample at regular price intervals
        max_distance = max(bid_curve[-1][0] if bid_curve else 0, 
                          ask_curve[-1][0] if ask_curve else 0)
        
        # Use Monster-aligned resolution if available
        n_samples = int(MONSTER_WAVELENGTH * 10) if MONSTER_LOGIC_AVAILABLE else 128
        price_steps = np.linspace(-max_distance, max_distance, n_samples)
        liquidity = np.zeros(n_samples)
        
        # Interpolate bid liquidity (negative side)
        for i, p in enumerate(price_steps[:n_samples//2]):
            dist = -p
            # Find nearest bid level
            for bd, bs in bid_curve:
                if bd >= dist:
                    liquidity[i] = bs
                    break
        
        # Interpolate ask liquidity (positive side)
        for i, p in enumerate(price_steps[n_samples//2:], start=n_samples//2):
            dist = p
            for ad, as_ in ask_curve:
                if ad >= dist:
                    liquidity[i] = as_
                    break
        
        # Normalize liquidity
        if np.max(liquidity) > 0:
            liquidity = liquidity / np.max(liquidity)
        
        # Extract spectrum
        sig = self.extract_1d_spectrum(liquidity)
        sig.ticker = "ORDERBOOK"
        
        return sig


# =============================================================================
# SPECTRAL MANIFOLD OPERATOR (The Core Innovation)
# =============================================================================

class SpectralManifoldOperator:
    """
    The "Holographic Neural Operator" that:
    1. Maps market data to spectral representation
    2. Queries GrainVDB for matching physics patterns
    3. Applies the retrieved "operator" to predict evolution
    4. Maps back to market predictions
    
    This is the bridge between financial microstructure and physical laws.
    """
    
    def __init__(self, embed_dim: int = 256, store_path: str = None):
        self.embed_dim = embed_dim
        self.extractor = SpectralExtractor()
        
        # Initialize Monster Resonance Detector
        self.monster_detector = MonsterResonanceDetector() if MONSTER_LOGIC_AVAILABLE else None
        
        # Default store path is in fresh_bot lab
        if store_path is None:
            store_path = os.path.join(FRESH_BOT_PATH, "grain_vdb_store")
        
        # Load GrainVDB with physics patterns
        self.vdb = None
        if GRAINVDB_AVAILABLE:
            try:
                # Change to store directory for loading
                original_cwd = os.getcwd()
                os.chdir(FRESH_BOT_PATH)
                
                self.vdb = GrainVDB(embed_dim=embed_dim, max_size=5000)
                self.vdb.load()
                logger.info(f"📖 Loaded GrainVDB with {self.vdb.count} physics patterns")
                
                os.chdir(original_cwd)
            except Exception as e:
                logger.warning(f"Could not load GrainVDB: {e}")
                try:
                    os.chdir(original_cwd)
                except:
                    pass
        
        # Physics pattern interpretations
        self.pattern_effects = {
            "MHD_64": {"type": "wave_propagation", "damping": 0.1, "velocity": 0.8},
            "MHD_256": {"type": "wave_propagation", "damping": 0.05, "velocity": 0.9},
            "turbulence_gravity_cooling": {"type": "dissipation", "damping": 0.3, "velocity": 0.4},
            "supernova_explosion_128": {"type": "expansion", "damping": 0.0, "velocity": 1.5},
            "gray_scott_reaction_diffusion": {"type": "pattern_formation", "damping": 0.2, "velocity": 0.6},
            "acoustic_scattering_maze": {"type": "reflection", "damping": 0.15, "velocity": 0.7},
            "viscoelastic_instability": {"type": "stress_release", "damping": 0.25, "velocity": 1.2},
            "rayleigh_benard": {"type": "convection", "damping": 0.18, "velocity": 0.65},
        }
    
    def query_physics_pattern(self, spectral_sig: SpectralSignature, k: int = 3) -> List[Dict]:
        """
        Query GrainVDB for physics patterns matching this spectral signature.
        """
        if self.vdb is None:
            return []
        
        embedding = spectral_sig.to_embedding(self.embed_dim)
        matches = self.vdb.search(embedding, k=k)
        
        results = []
        for match in matches:
            if match['metadata']:
                dataset = match['metadata'].get('dataset', 'unknown')
                effects = self.pattern_effects.get(dataset, {})
                
                results.append({
                    'dataset': dataset,
                    'score': match['score'],
                    'loss': match['metadata'].get('loss', 0),
                    'effects': effects,
                    'vector': match.get('vector')
                })
        
        return results
    
    def evolve_spectrum(self, 
                        spectral_sig: SpectralSignature,
                        physics_pattern: Dict,
                        steps: int = 1) -> np.ndarray:
        """
        Apply physics-based evolution to the spectral representation.
        
        This is the "operator" part - we're not just predicting,
        we're simulating forward using physical rules.
        """
        power = spectral_sig.power_spectrum.copy()
        phase = spectral_sig.phase_spectrum.copy()
        
        effects = physics_pattern.get('effects', {})
        damping = effects.get('damping', 0.1)
        velocity = effects.get('velocity', 1.0)
        pattern_type = effects.get('type', 'wave_propagation')
        
        n = len(power)
        freqs = np.arange(n)
        
        for _ in range(steps):
            if pattern_type == 'wave_propagation':
                # MHD-like: coherent wave with slight damping
                # High frequencies decay faster
                decay = np.exp(-damping * freqs / n)
                power = power * decay
                # Phase advances proportional to frequency
                phase = phase + velocity * freqs * 0.1
                
            elif pattern_type == 'dissipation':
                # Turbulence-like: energy cascades to smaller scales
                # Transfer energy from low to high frequencies
                cascade = np.roll(power, 1) * 0.1
                power = power * (1 - damping) + cascade
                phase = phase + np.random.normal(0, 0.1, n)  # Phase randomization
                
            elif pattern_type == 'expansion':
                # Supernova-like: rapid outward expansion
                # Shift energy to higher frequencies (faster oscillations)
                power = np.roll(power, int(velocity))
                power = power * 1.1  # Energy increase
                phase = phase + velocity * 0.2
                
            elif pattern_type == 'pattern_formation':
                # Reaction-diffusion: emergence of characteristic scales
                # Enhance certain frequency bands
                resonance = np.sin(2 * np.pi * freqs / n * 3) ** 2
                power = power * (1 + 0.1 * resonance)
                
            elif pattern_type == 'stress_release':
                # Viscoelastic: sudden release after buildup
                # Inject energy at low frequencies
                release = np.exp(-freqs / (n * 0.1))
                power = power + release * velocity
                
            elif pattern_type == 'reflection':
                # Acoustic scattering: interference patterns
                interference = np.cos(2 * np.pi * freqs / n * 5)
                power = power * (1 + 0.2 * interference)
                
            elif pattern_type == 'convection':
                # Rayleigh-Benard: cells of circulation
                cells = np.sin(2 * np.pi * freqs / n * 2) ** 2
                power = power * (1 + 0.15 * cells)
        
        # Normalize
        power = power / (np.sum(power) + 1e-10)
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        
        return power, phase
    
    def predict_orderbook_evolution(self,
                                     bids: List[Tuple[float, float]],
                                     asks: List[Tuple[float, float]],
                                     horizon: int = 5) -> Dict:
        """
        Predict how the order book will evolve over the next `horizon` steps.
        
        Returns predictions for:
        - Mid price direction
        - Spread change
        - Imbalance shift
        - Volatility forecast
        """
        # Extract current spectral signature
        current_spectrum = self.extractor.extract_orderbook_spectrum(bids, asks)
        
        # Query physics patterns
        patterns = self.query_physics_pattern(current_spectrum, k=3)
        
        if not patterns:
            return {
                'error': 'No physics patterns available',
                'direction': 'neutral',
                'confidence': 0
            }
        
        # Use best matching pattern
        best_pattern = patterns[0]
        
        # Evolve spectrum
        evolved_power, evolved_phase = self.evolve_spectrum(
            current_spectrum, best_pattern, steps=horizon
        )
        
        # Interpret spectral changes
        current_centroid = current_spectrum.spectral_centroid
        current_bandwidth = current_spectrum.spectral_bandwidth
        
        # Approximate new centroid from evolved spectrum
        freqs = np.arange(len(evolved_power)) / len(evolved_power) * self.extractor.nyquist
        evolved_centroid = np.sum(freqs * evolved_power) / (np.sum(evolved_power) + 1e-10)
        evolved_bandwidth = np.sqrt(
            np.sum(((freqs - evolved_centroid)**2) * evolved_power) / 
            (np.sum(evolved_power) + 1e-10)
        )
        
        # Derive predictions
        centroid_shift = evolved_centroid - current_centroid
        bandwidth_change = evolved_bandwidth - current_bandwidth
        
        # Spectral centroid increase → energy moving to high frequencies → volatility
        # Spectral bandwidth increase → spread widening → uncertainty
        
        pattern_type = best_pattern['effects'].get('type', 'unknown')
        
        # Direction prediction based on pattern type
        if pattern_type == 'wave_propagation':
            # MHD: momentum continues
            direction = 'continuation'
        elif pattern_type == 'dissipation':
            # Turbulence: mean reversion
            direction = 'reversion'
        elif pattern_type == 'expansion':
            # Supernova: breakout
            direction = 'breakout'
        elif pattern_type == 'stress_release':
            # Viscoelastic: snap
            direction = 'breakout'
        else:
            direction = 'neutral'
        
        # Normalize confidence to 0-1 using sigmoid
        raw_score = best_pattern['score']
        confidence = 1.0 / (1.0 + np.exp(-raw_score / 50))  # Sigmoid normalization
        
        # Monster Resonance Check
        monster_alignments = []
        if self.monster_detector:
            monster_alignments = self.monster_detector.detect_alignments(current_spectrum.dominant_frequencies)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'pattern_type': pattern_type,
            'dataset_match': best_pattern['dataset'],
            'centroid_shift': float(centroid_shift),
            'bandwidth_change': float(bandwidth_change),
            'volatility_forecast': 'increasing' if centroid_shift > 0 else 'stable',
            'spread_forecast': 'widening' if bandwidth_change > 0 else 'tightening',
            'monster_alignments': monster_alignments,
            'physics_details': {
                'damping': best_pattern['effects'].get('damping', 0),
                'velocity': best_pattern['effects'].get('velocity', 0),
            },
            'evolved_power': evolved_power.tolist(),
            'all_matches': [
                {'dataset': p['dataset'], 'score': p['score'], 'type': p['effects'].get('type')}
                for p in patterns
            ]
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Spectral Manifold Operator."""
    print("=" * 80)
    print("SPECTRAL MANIFOLD OPERATOR")
    print("Holographic Neural Operator for Order Flow Prediction")
    print("=" * 80)
    print()
    
    operator = SpectralManifoldOperator()
    
    print(f"GrainVDB Status: {'✅ Loaded' if operator.vdb else '❌ Not available'}")
    # skip count if not available
    print()
    
    # Simulate order book scenarios
    scenarios = [
        {
            "name": "Strong Bid Wall",
            "bids": [(100.0, 5000), (99.99, 3000), (99.98, 2000), (99.95, 1000), (99.90, 500)],
            "asks": [(100.01, 500), (100.02, 500), (100.05, 800), (100.10, 1000), (100.20, 1500)]
        },
        {
            "name": "Thin Liquidity",
            "bids": [(100.0, 100), (99.95, 100), (99.90, 200), (99.80, 300), (99.70, 400)],
            "asks": [(100.05, 100), (100.10, 100), (100.20, 200), (100.30, 300), (100.40, 400)]
        },
        {
            "name": "Balanced Book",
            "bids": [(100.0, 1000), (99.98, 1000), (99.96, 1000), (99.94, 1000), (99.92, 1000)],
            "asks": [(100.02, 1000), (100.04, 1000), (100.06, 1000), (100.08, 1000), (100.10, 1000)]
        },
        {
            "name": "Ask Cliff",
            "bids": [(100.0, 500), (99.98, 600), (99.96, 700), (99.94, 800), (99.92, 900)],
            "asks": [(100.01, 200), (100.02, 100), (100.03, 50), (100.05, 5000), (100.10, 1000)]
        }
    ]
    
    for scenario in scenarios:
        print("-" * 80)
        print(f"📊 Scenario: {scenario['name']}")
        print("-" * 80)
        
        # Extract spectral signature
        extractor = SpectralExtractor()
        spectrum = extractor.extract_orderbook_spectrum(scenario['bids'], scenario['asks'])
        
        print(f"   Spectral Analysis:")
        print(f"      Centroid: {spectrum.spectral_centroid:.4f}")
        print(f"      Bandwidth: {spectrum.spectral_bandwidth:.4f}")
        print(f"      Flatness: {spectrum.spectral_flatness:.4f} ({'noisy' if spectrum.spectral_flatness > 0.5 else 'structured'})")
        print(f"      Coherence: {spectrum.coherence_score:.4f}")
        print(f"      Entropy: {spectrum.entropy:.4f}")
        print()
        
        # Get prediction
        prediction = operator.predict_orderbook_evolution(
            scenario['bids'], 
            scenario['asks'],
            horizon=5
        )
        
        direction_emoji = {
            'continuation': '➡️',
            'reversion': '↩️',
            'breakout': '🚀',
            'neutral': '⚖️'
        }
        
        print(f"   Physics-Based Prediction:")
        print(f"      Direction: {direction_emoji.get(prediction['direction'], '?')} {prediction['direction'].upper()}")
        print(f"      Confidence: {prediction['confidence']*100:.1f}%")
        print(f"      Pattern Match: {prediction.get('dataset_match', 'N/A')} ({prediction.get('pattern_type', 'N/A')})")
        print(f"      Volatility: {prediction.get('volatility_forecast', 'N/A')}")
        print(f"      Spread: {prediction.get('spread_forecast', 'N/A')}")
        
        if prediction.get('all_matches'):
            print(f"   All Matches:")
            for m in prediction['all_matches']:
                print(f"      • {m['dataset']}: {m['score']:.3f} ({m.get('type', 'N/A')})")
        print()
    
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demo()
