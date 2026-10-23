# Holographic VWAP + `Noise` Forward-Only Trajectory Engine

[![Live Real-Time Analytics Dashboard](https://img.shields.io/badge/Live_Dashboard-GitHub_Pages-38bdf8?style=for-the-badge)](https://21e8-miner.github.io/holographic-vwap-noise/)

**Live Interactive Terminal:** **[https://21e8-miner.github.io/holographic-vwap-noise/](https://21e8-miner.github.io/holographic-vwap-noise/)**

A real-time cryptocurrency VWAP deviation & geometric curvature scanner combined with a **backprop-free, forward-only credit assignment optimizer** adapted from [`qlabs-eng/noise`](https://github.com/qlabs-eng/noise).

---

## Why Forward-Only Credit Assignment (`Noise`) for Quantitative Trading?

Trading execution pipelines contain inherently **non-differentiable step functions**: discrete entry gates ($|z| > \sigma_{\text{entry}}$), breakeven stop triggers, volatility-adaptive trailing stops, discrete time-exits, slippage, and fixed gas fees. Standard gradient descent (`loss.backward()`) cannot differentiate through these discrete regime gates.

This repository implements **`noise_optimizer.py`**, bringing three core algorithms from `qlabs-eng/noise` directly to intraday multi-asset market microstructure:

1. **8x8 Walsh-Hadamard Central-Difference Probes (`hadamard_cut_step`)**:
   Perturbs all 8 non-differentiable strategy thresholds simultaneously along random Rademacher-flipped orthogonal Hadamard rows ($\theta \pm \sigma \, D H_i$), extracting low-variance forward credit in 16 ultra-fast forward evaluations per step.
2. **ANVIL-Style RMS-Normalized Momentum (`Boost -> Cut`)**:
   Uses a 20% wide-exploration *Boost* phase followed by an 80% fine-grained *Cut* phase with $\beta = (0.5, 0.98)$ momentum normalization and prior weight decay.
3. **9-EMA Trajectory Coordinate Merge (`optimize_and_merge`)**:
   Maintains 9 exponential moving averages ($\alpha \in \{0.50, 0.35, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02, 0.01\}$) along the optimization trajectory and runs a two-pass coordinate sweep on held-out bars to lock in the out-of-sample regime blend.

---

## Intraday Benchmark (1,250 1-Minute Bars Across `BTC-USD`, `ETH-USD`, `SOL-USD`)

### Out-of-Sample Unseen Future (Bars 750 $\to$ 1,250)
| Metric | Baseline (v3.0 Static) | `Noise` (All Patterns) | `Noise` (Holographic Reversion) | Alpha Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Net PnL** | `-$469.97` | `-$198.82` | **`+$18.08`** | **`+$488.05`** |
| **Max Drawdown** | `0.470%` | `0.199%` | **`0.025%`** | **`-95% DD`** |
| **Win Rate** | `11.8%` | `27.3%` | **`42.9%`** | **`+31.1% WR`** |
| **Total Trades** | `51` | `11` | `7` | Filtered 44 chop trades |

### Full-Day Benchmark (All 1,250 Bars)
| Metric | Baseline (v3.0 Static) | `Noise` (All Patterns) | `Noise` (Holographic Reversion) | Alpha Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Net PnL** | `-$1,258.72` (`-1.26%`) | `-$150.83` (`-0.15%`) | **`+$44.43` (`+0.04%`)** | **`+$1,303.15`** |
| **Max Drawdown** | `1.259%` | `0.199%` | **`0.025%`** | **`-98% DD`** |
| **Win Rate** | `13.5%` | `45.0%` | **`53.3%`** | **`+39.8% WR`** |
| **Total Trades** | `126` | `20` | `15` | High-conviction only |

---

## Learned Parameter Shift

| Parameter | Baseline v3.0 | `Noise` 9-EMA Merge | Structural Diagnosis |
| :--- | :---: | :---: | :--- |
| `ENTRY_DEVIATION_SIGMA` | `1.500 σ` | **`2.664 σ`** | Waits for genuine exhaustion outside $2.66\sigma$ before fading |
| `STOP_LOSS_SIGMA` | `1.000 σ` | **`4.026 σ`** | Places initial stop outside 1-minute noise (~$60 BTC whipsaw zone) |
| `TSL_BASE_THRESHOLD` | `2.000 %` | **`2.112 %`** | Adaptive trailing stop once mean-reversion begins |
| `BREAKEVEN_THRESHOLD` | `0.700 %` | **`0.585 %`** | Protects green trades once reversion clears roundtrip slippage |
| `MIN_CONFIDENCE` | `0.550` | **`0.717`** | Requires Physics Decision Engine + Holographic Resonance alignment |
| `TAKE_PROFIT_PCT` | `2.000 %` | **`2.772 %`** | Captures full reversion impulse back to session VWAP |
| `MAX_HOLD_MINUTES` | `30.0 m` | **`85.1 m`** | Prevents premature time-exits on winning reversions |
| `MOM_5M_THRESHOLD` | `0.300 %` | **`5.000 %` (Gated)** | Automatically disables false 5m breakout whipsaws in range regimes |

---

## Real-Time Web Dashboard Architecture (`index.html`)

The static web app hosted on GitHub Pages connects directly from the client's browser to:
- **Coinbase Exchange WebSocket (`wss://ws-feed.exchange.coinbase.com`)**: Sub-second real-time price ticks across `BTC-USD`, `ETH-USD`, `SOL-USD`, `AVAX-USD`, `LINK-USD`, and `UNI-USD`.
- **Coinbase 1m Candle REST API (`https://api.exchange.coinbase.com/products/{ticker}/candles?granularity=60`)**: Computes live rolling session VWAP, $\pm 1\sigma / \pm 2.66\sigma$ standard deviation bands, 2nd-order geometric phase (`DECAY`, `EXPANSION`, `COHERENT`), and in-browser 8x8 Walsh-Hadamard trajectory recalibration.

## Running the Python Optimizer Locally

```bash
python3 noise_optimizer.py
```

Outputs updated trajectory parameters and intraday trade attribution to `data/noise_learned_params.json`, which `position_manager.py` and `index.html` automatically load.
