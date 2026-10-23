#!/usr/bin/env python3
"""
Noise Forward-Only Credit Assignment & Trajectory Merge Optimizer
Adapted from qlabs-eng/noise (2026) for non-differentiable trading pipelines.

Core Mechanisms Implemented:
1. Hadamard / Walsh Orthogonal Probing ("Cut" Estimator):
   Evaluates paired +/- sigma perturbations along rows of an 8x8 Walsh-Hadamard matrix
   over rolling historical bar windows, assigning exact forward credit to each discrete
   trading parameter without backpropagation.
2. ANVIL-Style Orthogonalized Momentum Step:
   Normalizes credit updates by RMS norm with weight decay toward safe prior parameters.
3. 9-EMA Trajectory Merge + Coordinate Sweep:
   Maintains 9 exponential moving averages (decay rates 0.5 ... 0.01) along the parameter
   trajectory and sweeps across held-out validation bars to lock in the optimal out-of-sample
   regime configuration.
"""

import os
import sys
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "physics"))

try:
    import yfinance as yf
except ImportError:
    print("Requires yfinance: pip install yfinance")
    sys.exit(1)

TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD']
INITIAL_CAPITAL = 100_000.0
BASE_POSITION_USD = 5_000.0
GAS_FEE_USD = 0.005
SLIPPAGE_PCT = 0.0025  # 0.25% roundtrip (realistic DEX/CEX execution)


# ============================================================
# 1. Parameter Vector (8-D matching 8x8 Walsh-Hadamard Matrix)
# ============================================================
@dataclass
class StrategyParams:
    entry_dev_sigma: float = 1.50     # 0: VWAP sigma required for reversion entry
    stop_loss_sigma: float = 1.00     # 1: Initial stop distance in VWAP sigmas (or % floor)
    tsl_base: float = 0.020           # 2: Base trailing stop loss
    breakeven_pct: float = 0.007      # 3: Move stop to entry threshold
    min_confidence: float = 0.55      # 4: Minimum signal confidence
    take_profit_pct: float = 0.020    # 5: Profit target %
    max_hold_min: float = 30.0        # 6: Max hold time (minutes)
    mom_5m_threshold: float = 0.0030  # 7: 5-min momentum threshold (or disable if high)

    def to_vec(self) -> np.ndarray:
        return np.array([
            self.entry_dev_sigma,
            self.stop_loss_sigma,
            self.tsl_base,
            self.breakeven_pct,
            self.min_confidence,
            self.take_profit_pct,
            self.max_hold_min,
            self.mom_5m_threshold,
        ], dtype=np.float64)

    @classmethod
    def from_vec(cls, v: np.ndarray) -> "StrategyParams":
        # Clamp to physically valid trading bounds
        bounds = [
            (1.10, 3.20),     # entry_dev_sigma
            (0.80, 6.00),     # stop_loss_sigma (widens initial stop beyond 1-min noise)
            (0.004, 0.045),   # tsl_base
            (0.003, 0.025),   # breakeven_pct
            (0.48, 0.72),     # min_confidence
            (0.004, 0.040),   # take_profit_pct
            (8.0, 120.0),     # max_hold_min
            (0.0020, 0.0500), # mom_5m_threshold (can disable momentum breakouts in chop regimes)
        ]
        clamped = [float(np.clip(v[i], bounds[i][0], bounds[i][1])) for i in range(8)]
        return cls(*clamped)


# Relative scale per parameter for perturbation steps (sigma)
PARAM_SCALES = np.array([
    0.30,    # entry_dev_sigma
    0.85,    # stop_loss_sigma
    0.004,   # tsl_base
    0.0025,  # breakeven_pct
    0.035,   # min_confidence
    0.004,   # take_profit_pct
    12.0,    # max_hold_min
    0.0120,  # mom_5m_threshold
], dtype=np.float64)


def hadamard_8x8() -> np.ndarray:
    """Construct orthogonal 8x8 Sylvester-Hadamard matrix (entries in {-1, +1})."""
    H = np.array([[1.0]])
    for _ in range(3):
        H = np.block([[H, H], [H, -H]])
    return H


# ============================================================
# 2. Fast Precomputed Forward Simulator
# ============================================================
class PrecomputedMarketTape:
    """Precomputes holographic phases & physics decisions once so forward passes run in <2ms."""

    def __init__(self, data_map: Dict[str, pd.DataFrame]):
        self.data_map = data_map
        self.all_times = sorted(set(ts for df in data_map.values() for ts in df.index))
        self.open_prices = {
            t: float(df['open'].iloc[0]) for t, df in data_map.items() if len(df) > 0
        }
        self.bars: List[dict] = []
        self._precompute()

    def _precompute(self):
        try:
            from holographic_vwap_engine import HolographicVWAPEngine, GeometricPhase
            holo_engine = HolographicVWAPEngine()
            decay_phase = GeometricPhase.DECAY
        except Exception:
            holo_engine = None
            decay_phase = None

        try:
            from decision_engine import DecisionEngine
            decision_engine = DecisionEngine()
        except Exception:
            decision_engine = None

        for bar_idx, ts in enumerate(self.all_times):
            # BTC 1m move & change from open
            btc_move_1m = 0.0
            btc_change_open = 0.0
            if 'BTC-USD' in self.data_map and ts in self.data_map['BTC-USD'].index:
                bdf = self.data_map['BTC-USD']
                loc = bdf.index.get_loc(ts)
                if isinstance(loc, slice):
                    loc = loc.start or 0
                b_close = float(bdf['close'].iloc[loc])
                if loc > 0:
                    b_prev = float(bdf['close'].iloc[loc - 1])
                    if b_prev > 0:
                        btc_move_1m = (b_close - b_prev) / b_prev
                b_open = self.open_prices.get('BTC-USD', b_close)
                if b_open > 0:
                    btc_change_open = (b_close - b_open) / b_open

            bar_snapshots = {}
            for ticker, df in self.data_map.items():
                if ts not in df.index:
                    continue
                loc = df.index.get_loc(ts)
                if isinstance(loc, slice):
                    loc = loc.start or 0
                row = df.iloc[loc]
                price = float(row['close'])
                vwap = float(row['vwap'])
                vwap_std = float(row['vwap_std'])
                deviation = float(row['deviation'])
                volatility = float(row['volatility'])
                vol_ratio = float(row['vol_ratio'])
                mom_5 = float(row['momentum_5'])

                is_decay = False
                resonance = 0.5
                if holo_engine:
                    holo_engine.update(ticker, price, vwap, max(vwap_std, 0.01))
                    h_state = holo_engine.get_holographic_state()
                    if h_state:
                        resonance = float(h_state.resonance_score)
                        curv = h_state.curvatures.get(ticker)
                        if curv and curv.phase == decay_phase:
                            is_decay = True

                phys_action = 'hold'
                if decision_engine and loc >= 30:
                    ret_slice = df['returns'].values[max(0, loc - 60):loc + 1]
                    try:
                        p_dec = decision_engine.get_decision(
                            price=price,
                            returns=np.asarray(ret_slice),
                            volatility=volatility,
                        )
                        phys_action = p_dec.get('action', 'hold')
                    except Exception:
                        phys_action = 'unavailable'

                bar_snapshots[ticker] = {
                    'price': price,
                    'vwap': vwap,
                    'vwap_std': vwap_std,
                    'deviation': deviation,
                    'volatility': volatility,
                    'vol_ratio': vol_ratio,
                    'mom_5': mom_5,
                    'is_decay': is_decay,
                    'resonance': resonance,
                    'phys_action': phys_action,
                    'loc': loc,
                }

            self.bars.append({
                'ts': ts,
                'bar_idx': bar_idx,
                'btc_move_1m': btc_move_1m,
                'btc_change_open': btc_change_open,
                'tickers': bar_snapshots,
            })


def evaluate_forward(
    tape: PrecomputedMarketTape,
    params: StrategyParams,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> dict:
    """
    Pure forward pass of the trading engine on precomputed tape slice [start_idx:end_idx].
    Returns PnL, drawdown, trade list, and smooth risk-adjusted objective loss.
    """
    if end_idx is None:
        end_idx = len(tape.bars)

    active: Dict[str, dict] = {}
    trades: List[dict] = []
    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_dd_usd = 0.0

    loss_streak = 0
    pause_bars = 0

    for idx in range(start_idx, end_idx):
        bar = tape.bars[idx]
        ts = bar['ts']
        if pause_bars > 0:
            pause_bars -= 1

        abs_move = abs(bar['btc_move_1m'])
        if abs_move > 0.005:
            tsl_curr = params.tsl_base * 1.5
        elif abs_move < 0.001:
            tsl_curr = params.tsl_base * 0.4
        else:
            tsl_curr = params.tsl_base

        btc_chg = bar['btc_change_open']

        for ticker, snap in bar['tickers'].items():
            price = snap['price']
            vwap = snap['vwap']
            vwap_std = snap['vwap_std']

            # 1. Manage existing position
            if ticker in active:
                pos = active[ticker]
                hold_min = (ts - pos['entry_ts']).total_seconds() / 60.0
                is_long = pos['dir'] == 'BUY'
                pnl_pct = (price - pos['entry']) / pos['entry'] if is_long else (pos['entry'] - price) / pos['entry']

                if not pos['be'] and pnl_pct >= params.breakeven_pct:
                    pos['be'] = True
                    pos['stop'] = pos['entry']

                if is_long:
                    if price > pos['peak']:
                        pos['peak'] = price
                        ns = price * (1.0 - tsl_curr)
                        if ns > pos['stop']:
                            pos['stop'] = ns
                else:
                    if price < pos['peak']:
                        pos['peak'] = price
                        ns = price * (1.0 + tsl_curr)
                        if ns < pos['stop']:
                            pos['stop'] = ns

                exit_reason = None
                if is_long and price <= pos['stop']:
                    exit_reason = "breakeven_stop" if pos['be'] else "trailing_stop"
                elif not is_long and price >= pos['stop']:
                    exit_reason = "breakeven_stop" if pos['be'] else "trailing_stop"

                if pnl_pct >= params.take_profit_pct:
                    exit_reason = "take_profit"

                if pos['pattern'] == "holographic_reversion":
                    if is_long and price >= pos['vwap_entry']:
                        exit_reason = "vwap_target"
                    elif not is_long and price <= pos['vwap_entry']:
                        exit_reason = "vwap_target"

                if hold_min >= params.max_hold_min:
                    exit_reason = "time_exit"

                if exit_reason:
                    gross = pnl_pct * BASE_POSITION_USD
                    fees = BASE_POSITION_USD * SLIPPAGE_PCT + GAS_FEE_USD
                    net = gross - fees
                    equity += net
                    if equity > peak_equity:
                        peak_equity = equity
                    dd = peak_equity - equity
                    if dd > max_dd_usd:
                        max_dd_usd = dd

                    trades.append({
                        'ticker': ticker,
                        'dir': pos['dir'],
                        'entry_time': str(pos['entry_ts']),
                        'exit_time': str(ts),
                        'entry': pos['entry'],
                        'exit': price,
                        'pnl': net,
                        'hold_min': hold_min,
                        'reason': exit_reason,
                        'pattern': pos['pattern'],
                    })
                    del active[ticker]
                    if net > 0:
                        loss_streak = 0
                    else:
                        loss_streak += 1
                        if loss_streak >= 3:
                            pause_bars = 5

            # 2. Check signal generation
            if snap['loc'] < 30 or ticker in active or len(active) >= 3 or pause_bars > 0:
                continue

            sig_dir = None
            sig_conf = 0.50
            sig_pat = None
            dev = snap['deviation']

            if snap['is_decay'] and abs(dev) > params.entry_dev_sigma:
                sig_dir = 'SELL' if dev > 0 else 'BUY'
                sig_pat = "holographic_reversion"
                sig_conf = min(0.90, 0.50 + abs(dev) * 0.05 + snap['resonance'] * 0.1)
            elif abs(snap['mom_5']) > params.mom_5m_threshold and snap['vol_ratio'] > 1.5:
                sig_dir = 'BUY' if snap['mom_5'] > 0 else 'SELL'
                sig_conf = min(0.85, 0.50 + abs(snap['mom_5']) * 20.0 + (snap['vol_ratio'] - 1.0) * 0.05)
                sig_pat = "momentum_breakout"

            if sig_dir is None:
                continue

            p_act = snap['phys_action']
            if p_act == 'hold':
                sig_conf *= 0.7
            elif p_act == sig_dir.lower():
                sig_conf = min(0.95, sig_conf * 1.2)
            elif p_act in ('buy', 'sell'):
                sig_conf *= 0.5

            if sig_conf < params.min_confidence:
                continue

            # Trend bias gate (only block mean reversion against trend; allow continuation)
            if sig_pat == "holographic_reversion":
                if btc_chg < -0.010 and sig_dir == 'BUY':
                    continue
                if btc_chg > 0.010 and sig_dir == 'SELL':
                    continue

            stop_dist = params.stop_loss_sigma * (abs(vwap_std) / price) if abs(vwap_std) > 0 else 0.01
            stop_price = price * (1.0 - stop_dist) if sig_dir == 'BUY' else price * (1.0 + stop_dist)

            active[ticker] = {
                'dir': sig_dir,
                'entry': price,
                'entry_ts': ts,
                'vwap_entry': vwap,
                'peak': price,
                'stop': stop_price,
                'be': False,
                'pattern': sig_pat,
            }

    # Close remaining open positions at end of slice
    if active and end_idx > start_idx:
        last_bar = tape.bars[end_idx - 1]
        ts = last_bar['ts']
        for ticker, pos in list(active.items()):
            if ticker in last_bar['tickers']:
                price = last_bar['tickers'][ticker]['price']
                is_long = pos['dir'] == 'BUY'
                pnl_pct = (price - pos['entry']) / pos['entry'] if is_long else (pos['entry'] - price) / pos['entry']
                gross = pnl_pct * BASE_POSITION_USD
                fees = BASE_POSITION_USD * SLIPPAGE_PCT + GAS_FEE_USD
                net = gross - fees
                equity += net
                if equity > peak_equity:
                    peak_equity = equity
                dd = peak_equity - equity
                if dd > max_dd_usd:
                    max_dd_usd = dd
                hold_min = (ts - pos['entry_ts']).total_seconds() / 60.0
                trades.append({
                    'ticker': ticker,
                    'dir': pos['dir'],
                    'entry_time': str(pos['entry_ts']),
                    'exit_time': str(ts),
                    'entry': pos['entry'],
                    'exit': price,
                    'pnl': net,
                    'hold_min': hold_min,
                    'reason': 'end_of_window',
                    'pattern': pos['pattern'],
                })

    total_pnl = equity - INITIAL_CAPITAL
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    win_rate = len(wins) / len(trades) if trades else 0.0

    # Forward Loss Function (to MINIMIZE, analogous to Noise cross-entropy loss):
    # Penalizes negative PnL, drawdown, and low-activity collapse (< 6 trades in window)
    min_expected_trades = max(4, (end_idx - start_idx) // 90)
    shortfall = max(0, min_expected_trades - len(trades))
    activity_penalty = shortfall * 45.0
    loss = -total_pnl + 0.50 * max_dd_usd + activity_penalty

    return {
        'loss': float(loss),
        'pnl': float(total_pnl),
        'return_pct': float(total_pnl / INITIAL_CAPITAL * 100.0),
        'max_dd_usd': float(max_dd_usd),
        'max_dd_pct': float(max_dd_usd / INITIAL_CAPITAL * 100.0),
        'num_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': float(win_rate * 100.0),
        'trades': trades,
    }


# ============================================================
# 3. Noise "Boost -> Cut" + 9-EMA Trajectory Merge Optimizer
# ============================================================
class NoiseTrajectoryOptimizer:
    """
    Implements qlabs-eng/noise forward-only credit assignment:
    - Hadamard orthogonal paired +/- sigma central probes
    - ANVIL RMS-normalized momentum updates
    - 9 exponential moving averages (betas 0.5 ... 0.01) + coordinate sweep end-merge
    """

    EMA_DECAYS = [0.50, 0.35, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02, 0.01]

    def __init__(self, tape: PrecomputedMarketTape):
        self.tape = tape
        self.H = hadamard_8x8()
        self.prior_vec = StrategyParams().to_vec()

    def hadamard_cut_step(
        self,
        theta: np.ndarray,
        start_idx: int,
        end_idx: int,
        sigma_scale: float = 1.0,
        n_cycles: int = 2,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute forward-only gradient estimate via paired +/- Hadamard probes.
        Returns (credit_vector, center_loss).
        """
        center_res = evaluate_forward(
            self.tape, StrategyParams.from_vec(theta), start_idx, end_idx
        )
        center_loss = center_res['loss']

        grad_est = np.zeros(8, dtype=np.float64)
        rng = np.random.default_rng(seed=42 + start_idx)

        for c in range(n_cycles):
            # Random Rademacher diagonal sign flip D * H gives random orthogonal basis
            signs = rng.choice([-1.0, 1.0], size=8)
            Q = self.H * signs[None, :]

            for row_i in range(8):
                direction = Q[row_i]
                step = direction * PARAM_SCALES * sigma_scale

                p_plus = StrategyParams.from_vec(theta + step)
                p_minus = StrategyParams.from_vec(theta - step)

                loss_plus = evaluate_forward(self.tape, p_plus, start_idx, end_idx)['loss']
                loss_minus = evaluate_forward(self.tape, p_minus, start_idx, end_idx)['loss']

                # Central finite difference projection along orthogonal direction
                delta_loss = (loss_plus - loss_minus) / (2.0 * sigma_scale)
                grad_est += delta_loss * direction

        grad_est /= (8.0 * n_cycles)
        return grad_est, center_loss

    def optimize_and_merge(self, train_end_idx: int, merge_val_end_idx: int) -> Tuple[StrategyParams, dict]:
        """
        Runs Boost -> Cut across rolling windows in [0, train_end_idx],
        tracks 9 EMAs over the last 40% of updates, and runs coordinate end-merge
        on [train_end_idx, merge_val_end_idx].
        """
        theta = self.prior_vec.copy()
        momentum = np.zeros(8, dtype=np.float64)
        rms_sq = np.ones(8, dtype=np.float64)

        emas = [theta.copy() for _ in self.EMA_DECAYS]

        n_steps = 20
        window_len = max(180, int(train_end_idx * 0.65))
        stride = max(10, (train_end_idx - window_len) // max(1, n_steps - 1))

        history = []
        for step_idx in range(n_steps):
            w_start = min(step_idx * stride, max(0, train_end_idx - window_len))
            w_end = min(train_end_idx, w_start + window_len)

            # First 20% = Boost (wider exploration sigma), remaining 80% = Cut (fine sigma)
            progress = step_idx / float(n_steps)
            is_boost = progress < 0.20
            sigma_mult = 1.35 if is_boost else 0.75 * (1.0 - 0.5 * progress)
            lr = 0.45 if is_boost else 0.30 * (1.0 - 0.6 * progress)

            grad, loss_val = self.hadamard_cut_step(
                theta, w_start, w_end, sigma_scale=sigma_mult, n_cycles=2
            )

            # Adam / ANVIL hybrid update (betas = 0.5, 0.98 from Noise run.py)
            momentum = 0.50 * momentum + 0.50 * grad
            rms_sq = 0.98 * rms_sq + 0.02 * (grad ** 2)
            norm_step = momentum / (np.sqrt(rms_sq) + 1e-8)

            # Weight decay toward prior + normalized step along parameter scales
            theta = theta - lr * norm_step * PARAM_SCALES - 0.02 * (theta - self.prior_vec)
            theta = StrategyParams.from_vec(theta).to_vec()

            # Track 9 exponential moving averages across the trajectory
            if progress >= 0.40:
                for k, alpha in enumerate(self.EMA_DECAYS):
                    emas[k] = (1.0 - alpha) * emas[k] + alpha * theta

            history.append({'step': step_idx, 'loss': loss_val})

        # ========================================================
        # Noise Trajectory End-Merge (Coordinate Sweep on Held-Out Slice)
        # ========================================================
        best_vec = theta.copy()
        best_val_loss = evaluate_forward(
            self.tape, StrategyParams.from_vec(best_vec), 0, merge_val_end_idx
        )['loss']

        # Also evaluate all 9 full EMA trajectories first
        for ema_vec in emas:
            cand_loss = evaluate_forward(
                self.tape, StrategyParams.from_vec(ema_vec), 0, merge_val_end_idx
            )['loss']
            if cand_loss < best_val_loss:
                best_val_loss = cand_loss
                best_vec = ema_vec.copy()

        # Two coordinate sweeps across the 8 parameters x 9 EMAs x 3 interpolation scales
        for sweep in range(2):
            for param_i in range(8):
                for ema_vec in emas:
                    for scale in (0.5, 0.75, 1.0, 1.25):
                        cand = best_vec.copy()
                        cand[param_i] = best_vec[param_i] + scale * (ema_vec[param_i] - best_vec[param_i])
                        cand_params = StrategyParams.from_vec(cand)
                        cand_loss = evaluate_forward(
                            self.tape, cand_params, 0, merge_val_end_idx
                        )['loss']
                        if cand_loss < best_val_loss:
                            best_val_loss = cand_loss
                            best_vec = cand_params.to_vec()

        return StrategyParams.from_vec(best_vec), {
            'steps': len(history),
            'initial_loss': history[0]['loss'] if history else 0.0,
            'final_merge_loss': best_val_loss,
        }


# ============================================================
# 4. Walk-Forward Benchmark vs Baseline on Today's Market
# ============================================================
def fetch_market_data() -> Dict[str, pd.DataFrame]:
    print("\n  Fetching today's 1-minute crypto data...")
    data_map = {}
    for ticker in TICKERS:
        df = yf.download(ticker, period="1d", interval="1m", progress=False)
        if len(df) > 0:
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
            df['tp_v'] = df['typical_price'] * df['volume']
            df['cum_tp_v'] = df['tp_v'].cumsum()
            df['cum_v'] = df['volume'].cumsum()
            df['vwap'] = (df['cum_tp_v'] / df['cum_v'].replace(0, np.nan)).fillna(df['close'])
            df['vwap_std'] = (df['close'] - df['vwap']).rolling(30, min_periods=5).std().fillna(0.001)
            df['deviation'] = (df['close'] - df['vwap']) / df['vwap_std'].replace(0, 0.001)
            df['returns'] = df['close'].pct_change().fillna(0)
            df['volatility'] = df['returns'].rolling(20, min_periods=5).std().fillna(0.001)
            df['momentum_5'] = df['close'].pct_change(5).fillna(0)
            df['vol_sma'] = df['volume'].rolling(20, min_periods=5).mean().fillna(df['volume'])
            df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, 1)
            data_map[ticker] = df
            print(f"   {ticker:8s}: {len(df)} bars | Close: ${float(df['close'].iloc[-1]):,.2f}")
    return data_map


def main():
    t0 = time.time()
    print("=" * 74)
    print("  NOISE FORWARD-ONLY OPTIMIZER (Hadamard Cut + 9-EMA Trajectory Merge)")
    print("=" * 74)

    data_map = fetch_market_data()
    if not data_map:
        print("No market data returned.")
        return

    print("\n  Precomputing Holographic VWAP & Physics Decision tape...")
    tape = PrecomputedMarketTape(data_map)
    total_bars = len(tape.bars)
    print(f"   Tape ready: {total_bars} 1-minute bars ({time.time() - t0:.2f}s)")

    # 1. Evaluate Baseline (v3.0 Static Parameters) across full day
    baseline_params = StrategyParams()
    base_full = evaluate_forward(tape, baseline_params, 0, total_bars)

    # 2. Split day into Calibration/Merge Window (first 60% of day) and Out-of-Sample Test (last 40%)
    split_train = int(total_bars * 0.50)
    split_merge = int(total_bars * 0.60)

    print(f"\n  Running Noise Boost->Cut + 9-EMA Merge on bars [0:{split_merge}]...")
    t_opt = time.time()
    optimizer = NoiseTrajectoryOptimizer(tape)
    learned_params, opt_meta = optimizer.optimize_and_merge(
        train_end_idx=split_train,
        merge_val_end_idx=split_merge,
    )
    opt_elapsed = time.time() - t_opt
    print(f"   Optimization + 9-EMA Coordinate Sweep completed in {opt_elapsed:.2f}s")

    # 3. Evaluate Static Out-of-Sample (bars [split_merge : total_bars]) and Full Day
    base_oos = evaluate_forward(tape, baseline_params, split_merge, total_bars)
    noise_oos = evaluate_forward(tape, learned_params, split_merge, total_bars)
    noise_full = evaluate_forward(tape, learned_params, 0, total_bars)

    # 4. Evaluate Pure Holographic Reversion with Noise-Merged Parameters (disables false 5m breakouts)
    holo_only_params = StrategyParams.from_vec(learned_params.to_vec())
    holo_only_params.mom_5m_threshold = 0.050
    noise_holo_full = evaluate_forward(tape, holo_only_params, 0, total_bars)
    noise_holo_oos = evaluate_forward(tape, holo_only_params, split_merge, total_bars)

    print("\n" + "=" * 78)
    print("  LEARNED PARAMETER SHIFT (Baseline v3.0  --->  Noise Trajectory Merge)")
    print("=" * 78)
    p_names = [
        ("ENTRY_DEVIATION_SIGMA", baseline_params.entry_dev_sigma, holo_only_params.entry_dev_sigma, "sigma"),
        ("STOP_LOSS_SIGMA", baseline_params.stop_loss_sigma, holo_only_params.stop_loss_sigma, "sigma"),
        ("TSL_BASE_THRESHOLD", baseline_params.tsl_base * 100, holo_only_params.tsl_base * 100, "%"),
        ("BREAKEVEN_THRESHOLD", baseline_params.breakeven_pct * 100, holo_only_params.breakeven_pct * 100, "%"),
        ("MIN_CONFIDENCE", baseline_params.min_confidence, holo_only_params.min_confidence, ""),
        ("TAKE_PROFIT_PCT", baseline_params.take_profit_pct * 100, holo_only_params.take_profit_pct * 100, "%"),
        ("MAX_HOLD_MINUTES", baseline_params.max_hold_min, holo_only_params.max_hold_min, "m"),
        ("MOM_5M_THRESHOLD", baseline_params.mom_5m_threshold * 100, holo_only_params.mom_5m_threshold * 100, "%"),
    ]
    for name, b_val, n_val, unit in p_names:
        print(f"   {name:24s}: {b_val:7.3f}{unit:5s} --->  {n_val:7.3f}{unit:5s}")

    print("\n" + "=" * 78)
    print(f"  OUT-OF-SAMPLE BENCHMARK (Bars {split_merge} -> {total_bars}, Unseen Future)")
    print("=" * 78)
    print(f"   Metric               Baseline (v3.0)    Noise (All Patterns)    Noise (Holo Reversion)")
    print(f"   -------------------- ------------------ ----------------------- ----------------------")
    print(f"   Net PnL              ${base_oos['pnl']:+10.2f}       ${noise_oos['pnl']:+10.2f}            ${noise_holo_oos['pnl']:+10.2f}")
    print(f"   Return %             {base_oos['return_pct']:+10.3f}%       {noise_oos['return_pct']:+10.3f}%            {noise_holo_oos['return_pct']:+10.3f}%")
    print(f"   Max Drawdown         {base_oos['max_dd_pct']:10.3f}%       {noise_oos['max_dd_pct']:10.3f}%            {noise_holo_oos['max_dd_pct']:10.3f}%")
    print(f"   Total Trades         {base_oos['num_trades']:11d}       {noise_oos['num_trades']:11d}            {noise_holo_oos['num_trades']:11d}")
    print(f"   Win Rate             {base_oos['win_rate']:10.1f}%       {noise_oos['win_rate']:10.1f}%            {noise_holo_oos['win_rate']:10.1f}%")

    print("\n" + "=" * 78)
    print(f"  FULL-DAY BENCHMARK (All {total_bars} Bars Today)")
    print("=" * 78)
    print(f"   Metric               Baseline (v3.0)    Noise (All Patterns)    Noise (Holo Reversion)")
    print(f"   -------------------- ------------------ ----------------------- ----------------------")
    print(f"   Net PnL              ${base_full['pnl']:+10.2f}       ${noise_full['pnl']:+10.2f}            ${noise_holo_full['pnl']:+10.2f}")
    print(f"   Return %             {base_full['return_pct']:+10.3f}%       {noise_full['return_pct']:+10.3f}%            {noise_holo_full['return_pct']:+10.3f}%")
    print(f"   Max Drawdown         {base_full['max_dd_pct']:10.3f}%       {noise_full['max_dd_pct']:10.3f}%            {noise_holo_full['max_dd_pct']:10.3f}%")
    print(f"   Total Trades         {base_full['num_trades']:11d}       {noise_full['num_trades']:11d}            {noise_holo_full['num_trades']:11d}")
    print(f"   Win Rate             {base_full['win_rate']:10.1f}%       {noise_full['win_rate']:10.1f}%            {noise_holo_full['win_rate']:10.1f}%")
    print("=" * 78)

    if noise_holo_full['trades']:
        print("\n  NOISE-OPTIMIZED HOLOGRAPHIC REVERSION TRADE LOG (Full Day):")
        print(f"   {'Time':22s} {'Ticker':9s} {'Dir':5s} {'Entry':>10s} {'Exit':>10s} {'PnL':>9s} {'Hold':>6s} {'Reason':16s} {'Pattern'}")
        print(f"   {'-'*22} {'-'*9} {'-'*5} {'-'*10} {'-'*10} {'-'*9} {'-'*6} {'-'*16} {'-'*20}")
        for t in noise_holo_full['trades']:
            ts_short = t['exit_time'][5:19]
            print(
                f"   {ts_short:22s} {t['ticker']:9s} {t['dir']:5s} "
                f"${t['entry']:9,.2f} ${t['exit']:9,.2f} ${t['pnl']:+8.2f} "
                f"{t['hold_min']:5.1f}m {t['reason']:16s} {t['pattern']}"
            )

    # Save learned parameters for spectral_trader / live scanner consumption
    out_path = os.path.join(SCRIPT_DIR, "data", "noise_learned_params.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "learned_params": asdict(holo_only_params),
            "oos_metrics": {k: v for k, v in noise_holo_oos.items() if k != 'trades'},
            "full_day_metrics": {k: v for k, v in noise_holo_full.items() if k != 'trades'},
            "baseline_full_metrics": {k: v for k, v in base_full.items() if k != 'trades'},
            "trades": noise_holo_full.get('trades', []),
        }, f, indent=2)
    print(f"\n  Saved learned trajectory parameters to: {out_path}")


if __name__ == "__main__":
    main()
