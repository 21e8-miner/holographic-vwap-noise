# Holographic VWAP + `Noise` v2.1 Forward-Only Trajectory Engine

[![Live Real-Time Analytics Dashboard](https://img.shields.io/badge/Live_Dashboard-GitHub_Pages-38bdf8?style=for-the-badge)](https://21e8-miner.github.io/holographic-vwap-noise/)

**Live Real-Time Action & Forward P&L Dashboard:** **[https://21e8-miner.github.io/holographic-vwap-noise/](https://21e8-miner.github.io/holographic-vwap-noise/)**

A real-time cryptocurrency multi-asset VWAP deviation, cross-asset holographic resonance, and 2nd-order geometric curvature scanner combined with a **backprop-free, forward-only credit assignment optimizer** adapted from [`qlabs-eng/noise`](https://github.com/qlabs-eng/noise).

---

## Middle-Out QC & `Noise` v2.1 Architecture

Trading execution pipelines contain inherently **non-differentiable step functions**: discrete entry gates ($|z| > \sigma_{\text{entry}}$), breakeven stop triggers, volatility-adaptive trailing stops, dynamic VWAP z-score reversion exits ($|z| \le z_{\text{exit}}$), discrete time-exits, slippage, and fixed gas fees. Standard gradient descent (`loss.backward()`) cannot differentiate through these discrete regime gates.

`Noise` v2.1 audits and upgrades the entire pipeline from the **middle out** (`data/noise_learned_params.json` contract $\leftrightarrow$ `noise_optimizer.py` $\leftrightarrow$ `index.html`):

1. **8x8 Walsh-Hadamard Central-Difference Probes (`hadamard_cut_step`)**:
   Perturbs all 8 non-differentiable strategy thresholds simultaneously along random Rademacher-flipped orthogonal Hadamard rows ($\theta \pm \sigma \, D H_i$), extracting low-variance forward credit in 16 forward evaluations per step.
2. **Top-4 Empirical Credit Covariance Subspace + 9-EMA Trajectory Merge (`optimize_and_merge`)**:
   Extracts the top-4 principal eigenvectors $V_4$ of the empirical forward-credit covariance matrix $C = \sum_t g_t g_t^\top$ via SVD and performs a joint coordinate sweep across both the 9 exponential moving averages ($\alpha \in \{0.50, 0.35, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02, 0.01\}$) and the top-4 coupled interaction directions on held-out validation bars.
3. **Portfolio Directional Coherence, Cross-Asset Breadth Guard & Stale-Impulse Protection (`evaluate_forward`)**:
   - **Portfolio Directional Coherence**: Prevents holding conflicting `BUY` and `SELL` mean-reversion positions simultaneously across correlated L1/DeFi assets.
   - **Cross-Asset Breadth Filter**: Requires higher conviction (`+0.032` confidence) for counter-trend dip buys when $\ge 4$ of 5 assets are trading below session VWAP.
   - **Stale-Impulse Protection**: Cuts stalled underwater reversions at 22 minutes if peak excursion never reached `+10 bps`.
4. **Unified Middle-Layer Telemetry Contract (`data/noise_learned_params.json` $\to$ `index.html`)**:
   Synchronizes the exact 30-bar rolling `(close - vwap)` standard deviation formula, per-asset empirical EV (`live_action_plan`), 1,345-bar intraday equity curve (`bar_equity_curve`), and executed trade log (`trades`) between Python and the browser.

---

## Intraday Benchmark (1,345 1-Minute Bars Across 5 L1/DeFi Assets)

### Out-of-Sample Unseen Future (Bars 807 $\to$ 1,345)
| Metric | Baseline (v3.0 Static) | `Noise` v2.1 (Subspace 9-EMA) | Alpha Delta |
| :--- | :---: | :---: | :---: |
| **Net PnL** | `-$1,070.99` | **`+$88.62`** | **`+$1,159.61`** |
| **Max Drawdown** | `1.071%` | **`0.000%`** | **`-100% DD`** |
| **Win Rate** | `16.0%` | **`100.0%`** (`3/3`) | **`+84.0% WR`** |
| **Profit Factor** | `0.14` | **`9.99`** | **`+9.85 PF`** |
| **Total Trades** | `75` | `3` | Filtered 72 chop trades |

### Full-Day Benchmark (All 1,345 Bars)
| Metric | Baseline (v3.0 Static) | `Noise` v2.1 (Subspace 9-EMA) | Alpha Delta |
| :--- | :---: | :---: | :---: |
| **Net PnL** | `-$2,241.62` (`-2.24%`) | **`+$141.13` (`+0.14%`)** | **`+$2,382.75`** |
| **Max Drawdown** | `2.242%` | **`0.039%`** | **`-98.3% DD`** |
| **Win Rate** | `14.2%` | **`81.8%`** (`9/11`) | **`+67.6% WR`** |
| **Profit Factor** | `0.10` | **`3.64`** | **`+3.54 PF`** |
| **Total Trades** | `169` | `11` | High-conviction only |

---

## Per-Asset Causal Forward-Credit Attribution

| Asset | Baseline v3.0 PnL | `Noise` v2.1 PnL (Trades) | `Noise` Win Rate | Causal Alloc Weight | Alpha Saved / Generated |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`SOL-USD`** | `-$538.58` | **`+$84.44`** (`5`) | `80.0%` | `1.55x` (Overweight) | **`+$623.02`** |
| **`LINK-USD`** | `-$543.19` | **`+$47.97`** (`4`) | `75.0%` | `1.55x` (Overweight) | **`+$591.16`** |
| **`ETH-USD`** | `-$353.70` | **`+$8.72`** (`2`) | `100.0%` | `1.55x` (Overweight) | **`+$362.42`** |
| **`AVAX-USD`** | `-$406.87` | **`$0.00`** (`0`) | `—` | `0.00x` (Pruned) | **`+$406.87`** |
| **`BTC-USD`** | `-$398.98` | **`$0.00`** (`0`) | `—` | `0.00x` (Pruned) | **`+$398.98`** |

---

## Running the Python Optimizer Locally

```bash
python3 noise_optimizer.py
```

Outputs updated trajectory parameters, `live_action_plan`, `bar_equity_curve`, and intraday trade logs to `data/noise_learned_params.json`, which `index.html` automatically loads.
