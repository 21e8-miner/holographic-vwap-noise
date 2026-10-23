# Holographic VWAP + `Noise` v2.0 Subspace 9-EMA Trajectory Engine

[![Live Real-Time Analytics Dashboard](https://img.shields.io/badge/Live_Dashboard-GitHub_Pages-38bdf8?style=for-the-badge)](https://21e8-miner.github.io/holographic-vwap-noise/)

**Live Interactive Terminal:** **[https://21e8-miner.github.io/holographic-vwap-noise/](https://21e8-miner.github.io/holographic-vwap-noise/)**

A real-time cryptocurrency multi-asset VWAP deviation, cross-asset holographic resonance, and 2nd-order geometric curvature scanner combined with a **backprop-free, forward-only credit assignment optimizer** adapted from [`qlabs-eng/noise`](https://github.com/qlabs-eng/noise).

---

## What's New in `Noise` v2.0 (10x Quantitative & Interactive Upgrade)

Trading execution pipelines contain inherently **non-differentiable step functions**: discrete entry gates ($|z| > \sigma_{\text{entry}}$), breakeven stop triggers, volatility-adaptive trailing stops, dynamic VWAP z-score reversion exits ($|z| \le z_{\text{exit}}$), discrete time-exits, slippage, and fixed gas fees. Standard gradient descent (`loss.backward()`) cannot differentiate through these discrete regime gates.

`Noise` v2.0 upgrades both the quantitative forward-only credit assignment engine (`noise_optimizer.py`) and the real-time web analytics terminal (`index.html`):

1. **8x8 Walsh-Hadamard Central-Difference Probes (`hadamard_cut_step`)**:
   Perturbs all 8 non-differentiable strategy thresholds simultaneously along random Rademacher-flipped orthogonal Hadamard rows ($\theta \pm \sigma \, D H_i$), extracting low-variance forward credit in 16 ultra-fast forward evaluations per step.
2. **ANVIL-Style RMS-Normalized Momentum (`Boost -> Cut`)**:
   Uses a 20% wide-exploration *Boost* phase followed by an 80% fine-grained *Cut* phase with $\beta = (0.5, 0.98)$ momentum normalization and prior weight decay.
3. **Top-4 Empirical Credit Covariance Subspace + 9-EMA Trajectory Merge (`optimize_and_merge`)**:
   Constructs the empirical forward-credit covariance matrix $C = \sum_t g_t g_t^\top$, extracts the top-4 principal eigenvectors $V_4$ via SVD, and performs a joint coordinate sweep across both the 9 exponential moving averages ($\alpha \in \{0.50, 0.35, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02, 0.01\}$) and the top-4 coupled interaction directions on held-out validation bars.
4. **Causal Forward-Credit Asset Allocation & Conviction-Scaled Sizing**:
   Evaluates forward credit per asset across the 5-asset L1/DeFi universe (`BTC-USD`, `ETH-USD`, `SOL-USD`, `AVAX-USD`, `LINK-USD`), scaling notional exposure (`1.1x`–`1.45x`) on positive-expectancy cross-resonant assets (`BTC`, `ETH`, `AVAX`, `LINK`) and automatically pruning negative-credit momentum-bleed regimes (`SOL` `0.0x`).
5. **Interactive 3-Canvas Quantitative Terminal & What-If Parameter Sandbox**:
   - **Live Price & Session VWAP Canvas** with $\pm 1\sigma / \pm \sigma_{\text{entry}}$ bands, interactive crosshair inspector, and execution markers.
   - **Cumulative Intraday Equity Curve** comparing Baseline v3.0 vs. `Noise` v2.0 with the `Bar 773` Out-of-Sample Unseen Future boundary.
   - **20-Step `Boost -> Cut` Loss Convergence & Subspace 9-EMA Merge Curve** (`+1,164.1` $\to$ `-15.0`).
   - **Interactive What-If Sandbox**: Adjust $\sigma_{\text{entry}}$, $\sigma_{\text{stop}}$, $z_{\text{exit}}$, and `MIN_CONFIDENCE` sliders in real time to re-run the non-differentiable forward backtest in the browser, inspect per-asset causal attribution, and export full trade telemetry to CSV.

---

## Intraday Benchmark (1,289 1-Minute Bars Across 5 L1/DeFi Assets)

### Out-of-Sample Unseen Future (Bars 773 $\to$ 1,289)
| Metric | Baseline (v3.0 Static) | `Noise` v2.0 (All Patterns) | `Noise` v2.0 (Subspace 9-EMA) | Alpha Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Net PnL** | `-$1,003.11` | `-$198.60` | **`+$18.67`** | **`+$1,021.78`** |
| **Max Drawdown** | `1.003%` | `0.214%` | **`0.072%`** | **`-92.8% DD`** |
| **Win Rate** | `17.8%` | `46.2%` | **`62.5%`** | **`+44.7% WR`** |
| **Profit Factor** | `0.15` | `0.33` | **`1.22`** | **`+1.07 PF`** |
| **Total Trades** | `73` | `13` | `8` | Filtered 65 chop trades |

### Full-Day Benchmark (All 1,289 Bars)
| Metric | Baseline (v3.0 Static) | `Noise` v2.0 (All Patterns) | `Noise` v2.0 (Subspace 9-EMA) | Alpha Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Net PnL** | `-$2,134.36` (`-2.13%`) | `-$93.62` (`-0.09%`) | **`+$110.74` (`+0.11%`)** | **`+$2,245.10`** |
| **Max Drawdown** | `2.134%` | `0.214%` | **`0.072%`** | **`-96.6% DD`** |
| **Win Rate** | `14.2%` | `59.1%` | **`68.8%`** | **`+54.6% WR`** |
| **Profit Factor** | `0.10` | `0.72` | **`2.10`** | **`+2.00 PF`** |
| **Total Trades** | `162` | `22` | `16` | High-conviction only |

---

## Per-Asset Causal Forward-Credit Attribution

| Asset | Baseline v3.0 PnL (Trades) | `Noise` v2.0 PnL (Trades) | `Noise` Win Rate | Causal Alloc Weight | Alpha Saved / Generated |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`AVAX-USD`** | `-$536.79` (`39`) | **`+$76.00`** (`6`) | `66.7%` | `1.45x` (Overweight) | **`+$612.79`** |
| **`LINK-USD`** | `-$512.77` (`34`) | **`+$22.87`** (`4`) | `75.0%` | `1.45x` (Overweight) | **`+$535.64`** |
| **`BTC-USD`** | `-$280.17` (`28`) | **`+$19.37`** (`3`) | `66.7%` | `1.35x` (Core Anchor) | **`+$299.54`** |
| **`ETH-USD`** | `-$374.16` (`28`) | **`-$7.50`** (`3`) | `66.7%` | `1.10x` (Neutral) | **`+$366.66`** |
| **`SOL-USD`** | `-$430.48` (`33`) | **`$0.00`** (`0`) | `—` | `0.00x` (Pruned) | **`+$430.48`** |

---

## Learned Parameter Shift (`Noise` v2.0 Subspace 9-EMA Merge)

| Parameter | Baseline v3.0 | `Noise` v2.0 Subspace Merge | Structural Diagnosis |
| :--- | :---: | :---: | :--- |
| `ENTRY_DEVIATION_SIGMA` | `1.500 σ` | **`1.954 σ`** | Captures high-conviction exhaustion outside $1.95\sigma$ when Holographic Resonance aligns |
| `STOP_LOSS_SIGMA` | `1.000 σ` | **`4.433 σ`** | Places initial stop outside 1-minute liquidity sweep zone |
| `TSL_BASE_THRESHOLD` | `2.000 %` | **`2.097 %`** | Volatility-adaptive trailing stop once mean-reversion begins |
| `BREAKEVEN_THRESHOLD` | `0.700 %` | **`0.315 %`** | Rapidly locks breakeven stop once reversion clears roundtrip fee/slippage hurdle |
| `MIN_CONFIDENCE` | `0.550` | **`0.717`** | Requires Physics Decision Engine + 5-Asset Holographic Resonance alignment |
| `TAKE_PROFIT_PCT` | `2.000 %` | **`2.104 %`** | Full impulse target when volatility expands |
| `MAX_HOLD_MINUTES` | `30.0 m` | **`86.7 m`** | Prevents premature time-exits on multi-bar VWAP reversions |
| `EXIT_VWAP_Z` | `0.000 z` | **`0.294 z`** | Dynamic VWAP z-score harvest: books profit at $\pm 0.294\sigma$ before VWAP centerline flip |

---

## Running the Python Optimizer Locally

```bash
python3 noise_optimizer.py
```

Outputs updated trajectory parameters, 20-step `Boost -> Cut` telemetry, per-asset attribution, and intraday trade logs to `data/noise_learned_params.json`, which `position_manager.py` and `index.html` automatically load.
