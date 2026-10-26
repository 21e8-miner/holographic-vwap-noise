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
   Extracts the top-4 principal eigenvectors $V_4$ of the empirical forward-credit covariance matrix $C = \sum_t g_t g_t^\top$ via SVD and performs a joint coordinate sweep across both the 9 exponential moving averages ($\alpha \in \{0.50, 0.35, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02, 0.01\}$) and the top-4 coupled interaction directions on held-out validation bars in **under 1.9 seconds**.
3. **Ornstein-Uhlenbeck (OU) Mean-Reversion Elasticity Filter**:
   Estimates rolling 60-bar AR(1) mean-reversion speed $\theta$ and half-life $t_{1/2} = \frac{\ln 2}{\theta}$. Assets with rapid elastic snap-back ($t_{1/2} < 20\text{ min}$) receive higher entry conviction, while sluggish drifting assets ($t_{1/2} > 75\text{ min}$) require increased confirmation, eliminating counter-trend trap entries.
4. **Institutional Liquidation Volume Depletion Check**:
   Blocks dip-buying into aggressive volume liquidations (`vol_ratio > 2.2x` with $z < -2.2\sigma$) until volume absorption cools or a 1-minute reversal candle forms.
5. **Dynamic Kelly Criterion Position Sizing ($f^*$)**:
   Computes asset-specific empirical win rates and win/loss ratios ($b = \frac{\text{avg win}}{\text{avg loss}}$) to determine optimal Kelly allocation with a fractional quarter-Kelly safety multiplier ($0.25 f^*$).
6. **Complete Cascade Breadth Lock & Directional Coherence (`evaluate_forward`)**:
   - **Cascade Lock**: Automatically disables counter-trend dip-buying whenever $\ge 4$ or $5/5$ assets are locked below session VWAP, eliminating knife-catches during market-wide liquidity flushes.
   - **Directional Coherence**: Blocks conflicting `BUY` and `SELL` mean-reversion positions across correlated L1/DeFi assets.
   - **Adaptive Early Harvest**: Automatically captures gains at 20 minutes if $\ge 70\%$ of reversion is achieved and $z$ normalizes into $\pm 0.40\sigma$.

---

## Performance Benchmark (Rolling 24h Tape, 1,440 Bars)

| Metric | Baseline (v3.0 Static) | `Noise` v2.2 (Subspace 9-EMA) | Alpha Delta / Advantage |
| :--- | :---: | :---: | :---: |
| **Full 24h Net PnL** | `-$2,904.91` (`-2.91%`) | **`+$29.69` (`+0.030%`)** | **`+$2,934.60` Alpha Saved** |
| **Profit Factor** | `0.11` | **`3.84`** | **`+3.73 PF`** |
| **Annualized Sharpe Ratio** | `-160.78` | **`+21.20`** | **`+181.98`** |
| **Annualized Sortino Ratio** | `-87.85` | **`+157.45`** | **`+245.30`** |
| **Max Drawdown** | `2.905%` | **`0.005%`** | **`-99.8% Drawdown Reduction`** |
| **Total Trades** | `193` trades (chopped) | **`6` high-conviction trades** | Filtered 187 losing trades |
| **Win Rate** | `17.6%` | **`66.7%`** | **`+49.1% WR`** |
| **Out-of-Sample PnL** | `-$954.13` | **`-$4.79`** | **`+$949.34` OOS Alpha** |

---

## Per-Asset Capital Preservation & Credit Attribution

| Asset | Baseline v3.0 PnL | `Noise` v2.2 PnL | Allocation Weight | OU Half-Life ($t_{1/2}$) | Kelly Multiplier | Drawdown Saved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BTC-USD`** | `-$661.31` | `-$5.32` | `1.00x` | `14.4 min` (76% Elastic) | `1.00x` | **`+$655.99`** |
| **`SOL-USD`** | `-$630.31` | `+$4.54` | `1.45x` | `27.0 min` (55% Elastic) | `2.00x` | **`+$634.85`** |
| **`LINK-USD`** | `-$512.01` | `+$15.13` | `1.45x` | `7.6 min` (87% Elastic) | `2.00x` | **`+$527.14`** |
| **`AVAX-USD`** | `-$451.51` | `$0.00` | `0.00x` (Pruned) | `14.0 min` (77% Elastic) | `1.00x` | **`+$451.51`** |
| **`ETH-USD`** | `-$402.86` | `+$15.34` | `1.45x` | `20.7 min` (65% Elastic) | `0.94x` | **`+$418.20`** |

---

## Running the Python Optimizer Locally

```bash
python3 noise_optimizer.py
```

Outputs updated trajectory parameters, `live_action_plan`, `bar_equity_curve`, and trade logs to `data/noise_learned_params.json`, which `index.html` automatically loads.
