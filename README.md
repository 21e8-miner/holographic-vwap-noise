# Holographic VWAP + `Noise` v2.2 Continuous Trajectory Engine

[![Live Real-Time Analytics Dashboard](https://img.shields.io/badge/Live_Dashboard-GitHub_Pages-38bdf8?style=for-the-badge)](https://21e8-miner.github.io/holographic-vwap-noise/)

**Live Real-Time Action & Forward P&L Dashboard:** **[https://21e8-miner.github.io/holographic-vwap-noise/](https://21e8-miner.github.io/holographic-vwap-noise/)**

A real-time cryptocurrency multi-asset VWAP deviation, cross-asset holographic resonance, and 2nd-order geometric curvature scanner combined with a **backprop-free, forward-only credit assignment optimizer** adapted from [`qlabs-eng/noise`](https://github.com/qlabs-eng/noise).

---

## Middle-Out QC & `Noise` v2.2 Architecture

Trading execution pipelines contain inherently **non-differentiable step functions**: discrete entry gates ($|z| > \sigma_{\text{entry}}$), breakeven stop triggers, volatility-adaptive trailing stops, dynamic VWAP z-score reversion exits ($|z| \le z_{\text{exit}}$), discrete time-exits, slippage, and fixed gas fees. Standard gradient descent (`loss.backward()`) cannot differentiate through these discrete regime gates.

`Noise` v2.2 audits and upgrades the entire pipeline from the **middle out** (`data/noise_learned_params.json` contract $\leftrightarrow$ `noise_optimizer.py` $\leftrightarrow$ `index.html`):

1. **8x8 Walsh-Hadamard Central-Difference Probes (`hadamard_cut_step`)**:
   Perturbs all 8 non-differentiable strategy thresholds simultaneously along random Rademacher-flipped orthogonal Hadamard rows ($\theta \pm \sigma \, D H_i$), extracting low-variance forward credit in 16 forward evaluations per step.
2. **Top-4 Empirical Credit Covariance Subspace + 9-EMA Trajectory Merge (`optimize_and_merge`)**:
   Extracts the top-4 principal eigenvectors $V_4$ of the empirical forward-credit covariance matrix $C = \sum_t g_t g_t^\top$ via SVD and performs a joint coordinate sweep across both the 9 exponential moving averages ($\alpha \in \{0.50, 0.35, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02, 0.01\}$) and the top-4 coupled interaction directions on held-out validation bars.
3. **Continuous Rolling 24-Hour Microstructure Tape (`1,440` 1-Minute Bars)**:
   Fetches continuous 2-day high-resolution tape sliced to the exact rolling 1,440 bars, ensuring 100% statistically valid sample sizes across Asian, European, and US sessions regardless of UTC execution time.
4. **Complete Cascade Breadth Lock & Directional Coherence (`evaluate_forward`)**:
   - **Cascade Lock**: Automatically disables counter-trend dip-buying whenever $\ge 4$ or $5/5$ assets are locked below session VWAP, eliminating knife-catches during market-wide liquidity flushes.
   - **Directional Coherence**: Blocks conflicting `BUY` and `SELL` mean-reversion positions across correlated L1/DeFi assets.
   - **Stale-Impulse Protection**: Automatically cuts stalled underwater trades at 22 minutes if peak favorable excursion fails to clear $+10\text{ bps}$.
5. **Interactive Execution Inspector & One-Click Copy (`index.html`)**:
   Real-time order specification card displaying exact Limit Entry, Take Profit, Stop Loss, and Expected Net EV with clipboard copy support.

---

## Dual-Regime Performance Benchmark

### 1. Trending Liquidity Flush / 24-Hour Crash Regime (1,440 Bars)
| Metric | Baseline (v3.0 Static) | `Noise` v2.2 (Subspace 9-EMA) | Alpha Delta / Capital Preserved |
| :--- | :---: | :---: | :---: |
| **Full 24h Net PnL** | `-$2,658.00` (`-2.66%`) | **`-$68.44` (`-0.068%`)** | **`+$2,589.56` Capital Preserved** |
| **Max Drawdown** | `2.658%` | **`0.071%`** | **`-97.3%` Drawdown Reduction** |
| **Total Trades** | `153` trades (chopped) | **`5` high-conviction trades** | Filtered 148 loss trades |
| **Out-of-Sample PnL** | `-$1,173.03` | **`-$57.19`** | **`+$1,115.84` OOS Alpha** |

### 2. Mean-Reversion High-Resonance Regime (1,345 Bars)
| Metric | Baseline (v3.0 Static) | `Noise` v2.2 (Subspace 9-EMA) | Alpha Delta |
| :--- | :---: | :---: | :---: |
| **Full Day Net PnL** | `-$2,241.62` (`-2.24%`) | **`+$141.13` (`+0.14%`)** | **`+$2,382.75`** |
| **Out-of-Sample PnL** | `-$1,070.99` | **`+$88.62`** | **`+$1,159.61` OOS Alpha** |
| **Win Rate** | `14.2%` | **`81.8%`** (`9/11` trades) | **`+67.6% WR`** |
| **Profit Factor** | `0.10` | **`3.64`** | **`+3.54 PF`** |
| **Max Drawdown** | `2.242%` | **`0.039%`** | **`-98.3% DD`** |

---

## Per-Asset Capital Preservation & Credit Attribution

| Asset | Baseline v3.0 PnL | `Noise` v2.2 PnL | Allocation Weight | Drawdown Saved |
| :--- | :---: | :---: | :---: | :---: |
| **`BTC-USD`** | `-$661.31` | `-$50.82` | `1.00x` | **`+$610.49`** |
| **`SOL-USD`** | `-$630.31` | `$0.00` | `0.00x` (Pruned) | **`+$630.31`** |
| **`LINK-USD`** | `-$512.01` | `$0.00` | `0.00x` (Pruned) | **`+$512.01`** |
| **`AVAX-USD`** | `-$451.51` | `$0.00` | `0.00x` (Pruned) | **`+$451.51`** |
| **`ETH-USD`** | `-$402.86` | `-$17.62` | `1.00x` | **`+$385.24`** |

---

## Running the Python Optimizer Locally

```bash
python3 noise_optimizer.py
```

Outputs updated trajectory parameters, `live_action_plan`, `bar_equity_curve`, and trade logs to `data/noise_learned_params.json`, which `index.html` automatically loads.
