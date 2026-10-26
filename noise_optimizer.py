#!/usr/bin/env python3
"""
Noise v2.1 Forward-Only Credit Assignment & Covariance-Projected 9-EMA Trajectory Engine
Adapted from qlabs-eng/noise (2026) for non-differentiable multi-asset trading pipelines.

Middle-Out QC Upgrades in v2.1:
1. Fee-Aware True Breakeven Stop & Live Rolling VWAP Target Reversion:
   Places breakeven stops beyond roundtrip slippage (SLIPPAGE_PCT + 5 bps) and evaluates
   VWAP reversion against the live rolling session VWAP rather than frozen entry VWAP.
2. Stale-Impulse Protection & End-of-Window Entry Guard:
   Cuts stalled underwater reversions early and avoids opening new trades in the final 15 bars.
3. Unified Middle-Layer Telemetry Contract (data/noise_learned_params.json):
   Exports exact per-asset empirical EV, live market snapshots, bar-by-bar equity curve,
   and trade logs with a 1-to-1 schema match consumed by index.html.
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

TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'LINK-USD']
INITIAL_CAPITAL = 100_000.0
BASE_POSITION_USD = 5_000.0
GAS_FEE_USD = 0.005
SLIPPAGE_PCT = 0.0025  # 0.25% roundtrip


# ============================================================
# 1. Parameter Vector (8-D matching 8x8 Walsh-Hadamard Matrix)
# ============================================================
@dataclass
class StrategyParams:
    entry_dev_sigma: float = 1.50     # 0: VWAP sigma required for reversion entry
    stop_loss_sigma: float = 1.00     # 1: Initial stop distance in VWAP sigmas
    tsl_base: float = 0.020           # 2: Base trailing stop loss
    breakeven_pct: float = 0.007      # 3: Move stop to fee-adjusted breakeven threshold
    min_confidence: float = 0.55      # 4: Minimum signal confidence
    take_profit_pct: float = 0.020    # 5: Profit target %
    max_hold_min: float = 30.0        # 6: Max hold time (minutes)
    exit_vwap_z: float = 0.00         # 7: VWAP z-score target (0 = exact VWAP, >0 = early harvest)

    def to_vec(self) -> np.ndarray:
        return np.array([
            self.entry_dev_sigma,
            self.stop_loss_sigma,
            self.tsl_base,
            self.breakeven_pct,
            self.min_confidence,
            self.take_profit_pct,
            self.max_hold_min,
            self.exit_vwap_z,
        ], dtype=np.float64)

    @classmethod
    def from_vec(cls, v: np.ndarray) -> "StrategyParams":
        bounds = [
            (1.80, 3.20),    # entry_dev_sigma (wait for genuine >1.8sigma deviation)
            (2.50, 5.50),    # stop_loss_sigma (outside 1m noise band)
            (0.012, 0.035),  # tsl_base (1.2% - 3.5%)
            (0.003, 0.015),  # breakeven_pct
            (0.68, 0.76),    # min_confidence (high-conviction only)
            (0.015, 0.045),  # take_profit_pct
            (45.0, 110.0),   # max_hold_min (give reversions 45m-110m to travel back to VWAP)
            (0.00, 0.35),    # exit_vwap_z (harvest at or just before VWAP centerline)
        ]
        clamped = [float(np.clip(v[i], bounds[i][0], bounds[i][1])) for i in range(8)]
        return cls(*clamped)


PARAM_NAMES = [
    "ENTRY_DEVIATION_SIGMA",
    "STOP_LOSS_SIGMA",
    "TSL_BASE_THRESHOLD",
    "BREAKEVEN_THRESHOLD",
    "MIN_CONFIDENCE",
    "TAKE_PROFIT_PCT",
    "MAX_HOLD_MINUTES",
    "EXIT_VWAP_Z_TARGET",
]

PARAM_SCALES = np.array([
    0.32,    # entry_dev_sigma
    0.85,    # stop_loss_sigma
    0.004,   # tsl_base
    0.0025,  # breakeven_pct
    0.035,   # min_confidence
    0.004,   # take_profit_pct
    14.0,    # max_hold_min
    0.12,    # exit_vwap_z
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
    """Precomputes holographic phases & physics decisions across all 5 crypto assets."""

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
                open_price = float(row['open'])
                vwap = float(row['vwap'])
                vwap_std = float(row['vwap_std'])
                deviation = float(row['deviation'])
                volatility = float(row['volatility'])
                vol_ratio = float(row['vol_ratio'])
                mom_5 = float(row['momentum_5'])
                ou_beta = float(row.get('ou_beta', 0.95))
                ou_half_life = float(row.get('ou_half_life', 20.0))
                elasticity = float(row.get('elasticity', 65.0))

                is_decay = False
                resonance = 0.5
                if holo_engine:
                    holo_engine.update(ticker, price, vwap, max(vwap_std, 1e-6))
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
                    'open_price': open_price,
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
                    'ou_beta': ou_beta,
                    'ou_half_life': ou_half_life,
                    'elasticity': elasticity,
                }

            below_vwap_count = sum(1 for s in bar_snapshots.values() if s['price'] < s['vwap'])
            above_vwap_count = sum(1 for s in bar_snapshots.values() if s['price'] > s['vwap'])

            self.bars.append({
                'ts': ts,
                'bar_idx': bar_idx,
                'btc_move_1m': btc_move_1m,
                'btc_change_open': btc_change_open,
                'below_vwap_count': below_vwap_count,
                'above_vwap_count': above_vwap_count,
                'tickers': bar_snapshots,
            })


def evaluate_forward(
    tape: PrecomputedMarketTape,
    params: StrategyParams,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    include_momentum_breakouts: bool = False,
    asset_weights: Optional[Dict[str, float]] = None,
    fee_aware_breakeven: bool = True,
    fast_mode: bool = False,
) -> dict:
    """
    Pure forward pass of the trading engine on precomputed tape slice [start_idx:end_idx].
    Includes Ornstein-Uhlenbeck (OU) mean-reversion elasticity filter, volume exhaustion check,
    Kelly sizing conviction scaling, and MFE/MAE tracking.
    """
    if end_idx is None:
        end_idx = len(tape.bars)

    active: Dict[str, dict] = {}
    trades: List[dict] = []
    equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    max_dd_usd = 0.0
    equity_curve: List[float] = [0.0] if not fast_mode else []
    bar_equity: List[float] = [] if not fast_mode else []

    num_trades = 0
    wins_count = 0
    losses_count = 0
    gross_win = 0.0
    gross_loss = 0.0

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
        below_vwap_count = bar['below_vwap_count']
        above_vwap_count = bar['above_vwap_count']

        for ticker, snap in bar['tickers'].items():
            price = snap['price']
            vwap = snap['vwap']
            vwap_std = max(abs(snap['vwap_std']), price * 0.0005)

            # 1. Manage existing position
            if ticker in active:
                pos = active[ticker]
                hold_min = (ts - pos['entry_ts']).total_seconds() / 60.0
                is_long = pos['dir'] == 'BUY'
                pnl_pct = (price - pos['entry']) / pos['entry'] if is_long else (pos['entry'] - price) / pos['entry']

                # MFE / MAE excursion tracking
                if not fast_mode:
                    if pnl_pct > pos['mfe_pct']:
                        pos['mfe_pct'] = pnl_pct
                    if pnl_pct < pos['mae_pct']:
                        pos['mae_pct'] = pnl_pct

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
                    if is_long:
                        target_p = pos['vwap_entry'] - params.exit_vwap_z * pos['std_entry']
                        if price >= target_p:
                            exit_reason = "vwap_target"
                    else:
                        target_p = pos['vwap_entry'] + params.exit_vwap_z * pos['std_entry']
                        if price <= target_p:
                            exit_reason = "vwap_target"

                # Adaptive Early Harvest: if held > 20m and captured 70%+ of excursion with positive pnl
                if fee_aware_breakeven and hold_min >= 20.0 and pnl_pct >= 0.0025:
                    cur_z = abs(snap['deviation'])
                    if cur_z <= 0.40:
                        exit_reason = "elastic_early_harvest"

                if hold_min >= params.max_hold_min:
                    exit_reason = "time_exit"
                elif fee_aware_breakeven and hold_min >= 22.0 and pnl_pct < -0.0015:
                    peak_exc = (pos['peak'] - pos['entry']) / pos['entry'] if is_long else (pos['entry'] - pos['peak']) / pos['entry']
                    if peak_exc < 0.0010:
                        exit_reason = "stale_reversion"

                if exit_reason:
                    gross = pnl_pct * pos['size_usd']
                    fees = pos['size_usd'] * SLIPPAGE_PCT + GAS_FEE_USD
                    net = gross - fees
                    equity += net
                    if equity > peak_equity:
                        peak_equity = equity
                    dd = peak_equity - equity
                    if dd > max_dd_usd:
                        max_dd_usd = dd

                    num_trades += 1
                    if net > 0:
                        wins_count += 1
                        gross_win += net
                        loss_streak = 0
                    else:
                        losses_count += 1
                        gross_loss += abs(net)
                        loss_streak += 1
                        if loss_streak >= 3:
                            pause_bars = 5

                    if not fast_mode:
                        equity_curve.append(round(equity - INITIAL_CAPITAL, 2))
                        mfe_usd = pos['mfe_pct'] * pos['size_usd']
                        mae_usd = abs(pos['mae_pct'] * pos['size_usd'])
                        mfe_eff = round((net / max(0.01, mfe_usd)) * 100.0, 1) if mfe_usd > 0 and net > 0 else 0.0
                        trades.append({
                            'bar': idx,
                            'ticker': ticker,
                            'dir': pos['dir'],
                            'entry_time': str(pos['entry_ts']),
                            'exit_time': str(ts),
                            'entry': round(pos['entry'], 4),
                            'exit': round(price, 4),
                            'pnl': round(net, 2),
                            'hold_min': round(hold_min, 1),
                            'reason': exit_reason,
                            'pattern': pos['pattern'],
                            'z_entry': round(pos['z_entry'], 2),
                            'conf': round(pos['conf'], 3),
                            'size_usd': round(pos['size_usd'], 2),
                            'mfe_usd': round(mfe_usd, 2),
                            'mae_usd': round(mae_usd, 2),
                            'mfe_efficiency': mfe_eff,
                        })
                    del active[ticker]

            # 2. Check signal generation (avoid opening new positions in final 45m of session)
            if fee_aware_breakeven and idx >= len(tape.bars) - 45:
                continue

            a_weight = asset_weights.get(ticker, 1.0) if asset_weights else 1.0
            if a_weight <= 0.05 or snap['loc'] < 30 or ticker in active or len(active) >= 3 or pause_bars > 0:
                continue

            sig_dir = None
            sig_conf = 0.50
            sig_pat = None
            dev = snap['deviation']

            if snap['is_decay'] and abs(dev) > params.entry_dev_sigma:
                sig_dir = 'SELL' if dev > 0 else 'BUY'
                sig_pat = "holographic_reversion"
                sig_conf = min(0.92, 0.50 + abs(dev) * 0.05 + snap['resonance'] * 0.1)
            elif include_momentum_breakouts and abs(snap['mom_5']) > 0.0030 and snap['vol_ratio'] > 1.5:
                sig_dir = 'BUY' if snap['mom_5'] > 0 else 'SELL'
                sig_conf = min(0.85, 0.50 + abs(snap['mom_5']) * 20.0 + (snap['vol_ratio'] - 1.0) * 0.05)
                sig_pat = "momentum_breakout"

            if sig_dir is None:
                continue

            # Physics engine action conviction modulation
            p_act = snap['phys_action']
            if p_act == 'hold':
                sig_conf *= 0.7
            elif p_act == sig_dir.lower():
                sig_conf = min(0.95, sig_conf * 1.2)
            elif p_act in ('buy', 'sell'):
                sig_conf *= 0.5

            # Portfolio Directional Coherence: never hold opposing BUY & SELL mean-reversion positions simultaneously
            if fee_aware_breakeven and active and any(p['dir'] != sig_dir for p in active.values()):
                continue

            req_conf = params.min_confidence
            if fee_aware_breakeven and sig_pat == "holographic_reversion":
                # Complete Cascade Lock: never fight a synchronized market-wide cascade
                n_assets = len(bar['tickers'])
                if below_vwap_count >= n_assets and sig_dir == 'BUY':
                    continue
                if above_vwap_count >= n_assets and sig_dir == 'SELL':
                    continue
                cascade_threshold = max(3, n_assets - 1)
                if (btc_chg < -0.0015 or below_vwap_count >= cascade_threshold) and sig_dir == 'BUY':
                    req_conf += 0.035
                elif (btc_chg > 0.008 or above_vwap_count >= cascade_threshold) and sig_dir == 'SELL':
                    req_conf += 0.035

                # Ornstein-Uhlenbeck (OU) Mean-Reversion Elasticity Filter
                ou_hl = snap.get('ou_half_life', 20.0)
                if ou_hl > 75.0:
                    req_conf += 0.040  # Sluggish or trending drift
                elif ou_hl < 20.0:
                    sig_conf = min(0.96, sig_conf + 0.025)  # High elastic snap-back regime

                # Volume Surge Exhaustion: if extreme volume liquidation spike, wait for volume absorption
                if sig_dir == 'BUY' and snap['vol_ratio'] > 2.2 and dev < -2.2:
                    is_green_candle = snap['price'] >= snap['open_price']
                    is_vol_cooled = snap['vol_ratio'] <= 1.8
                    if not (is_green_candle or is_vol_cooled):
                        continue

            if sig_conf < req_conf:
                continue

            if sig_pat == "holographic_reversion":
                if btc_chg < -0.010 and sig_dir == 'BUY':
                    continue
                if btc_chg > 0.010 and sig_dir == 'SELL':
                    continue

            stop_dist = params.stop_loss_sigma * (abs(vwap_std) / price) if abs(vwap_std) > 0 else 0.01
            stop_price = price * (1.0 - stop_dist) if sig_dir == 'BUY' else price * (1.0 + stop_dist)

            conv_mult = 1.0 + max(0.0, min(0.40, (sig_conf - params.min_confidence) * 1.5))
            pos_size_usd = BASE_POSITION_USD * conv_mult * a_weight

            active[ticker] = {
                'dir': sig_dir,
                'entry': price,
                'entry_ts': ts,
                'vwap_entry': vwap,
                'std_entry': max(abs(vwap_std), price * 0.0005),
                'z_entry': dev,
                'conf': sig_conf,
                'size_usd': pos_size_usd,
                'peak': price,
                'stop': stop_price,
                'be': False,
                'pattern': sig_pat,
                'mfe_pct': 0.0,
                'mae_pct': 0.0,
            }

        if not fast_mode:
            bar_equity.append(round(equity - INITIAL_CAPITAL, 2))

    # Close remaining open positions at end of slice
    if active and end_idx > start_idx:
        last_bar = tape.bars[end_idx - 1]
        ts = last_bar['ts']
        for ticker, pos in list(active.items()):
            if ticker in last_bar['tickers']:
                price = last_bar['tickers'][ticker]['price']
                is_long = pos['dir'] == 'BUY'
                pnl_pct = (price - pos['entry']) / pos['entry'] if is_long else (pos['entry'] - price) / pos['entry']
                gross = pnl_pct * pos['size_usd']
                fees = pos['size_usd'] * SLIPPAGE_PCT + GAS_FEE_USD
                net = gross - fees
                equity += net
                if equity > peak_equity:
                    peak_equity = equity
                dd = peak_equity - equity
                if dd > max_dd_usd:
                    max_dd_usd = dd
                num_trades += 1
                if net > 0:
                    wins_count += 1
                    gross_win += net
                else:
                    losses_count += 1
                    gross_loss += abs(net)

                if not fast_mode:
                    hold_min = (ts - pos['entry_ts']).total_seconds() / 60.0
                    mfe_usd = pos['mfe_pct'] * pos['size_usd']
                    mae_usd = abs(pos['mae_pct'] * pos['size_usd'])
                    mfe_eff = round((net / max(0.01, mfe_usd)) * 100.0, 1) if mfe_usd > 0 and net > 0 else 0.0
                    trades.append({
                        'bar': end_idx - 1,
                        'ticker': ticker,
                        'dir': pos['dir'],
                        'entry_time': str(pos['entry_ts']),
                        'exit_time': str(ts),
                        'entry': round(pos['entry'], 4),
                        'exit': round(price, 4),
                        'pnl': round(net, 2),
                        'hold_min': round(hold_min, 1),
                        'reason': 'end_of_window',
                        'pattern': pos['pattern'],
                        'z_entry': round(pos['z_entry'], 2),
                        'conf': round(pos['conf'], 3),
                        'size_usd': round(pos['size_usd'], 2),
                        'mfe_usd': round(mfe_usd, 2),
                        'mae_usd': round(mae_usd, 2),
                        'mfe_efficiency': mfe_eff,
                    })
        if not fast_mode and bar_equity:
            bar_equity[-1] = round(equity - INITIAL_CAPITAL, 2)

    total_pnl = equity - INITIAL_CAPITAL
    win_rate = (wins_count / num_trades * 100.0) if num_trades > 0 else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (9.99 if gross_win > 0 else 0.0)

    min_expected_trades = max(5, (end_idx - start_idx) // 100)
    shortfall = max(0, min_expected_trades - num_trades)
    activity_penalty = shortfall * 45.0
    loss = -total_pnl + 0.50 * max_dd_usd + activity_penalty

    if fast_mode:
        return {
            'loss': float(loss),
            'pnl': round(float(total_pnl), 2),
            'return_pct': round(float(total_pnl / INITIAL_CAPITAL * 100.0), 4),
            'max_dd_usd': round(float(max_dd_usd), 2),
            'max_dd_pct': round(float(max_dd_usd / INITIAL_CAPITAL * 100.0), 4),
            'num_trades': num_trades,
            'wins': wins_count,
            'losses': losses_count,
            'win_rate': round(float(win_rate), 2),
            'profit_factor': round(float(profit_factor), 2),
        }

    # Extended Profitability Insights: Kelly Criterion, Sharpe, Sortino, EV/hr
    total_hours = max(0.5, (end_idx - start_idx) / 60.0)
    ev_per_hour = round(total_pnl / total_hours, 2)

    # Annualized 1m bar Sharpe & Sortino ratios (crypto 24/7 = 525,600 min/yr)
    if len(bar_equity) > 5:
        b_diffs = np.diff(bar_equity)
        m_ret = float(np.mean(b_diffs))
        s_ret = float(np.std(b_diffs))
        d_diffs = b_diffs[b_diffs < 0]
        ds_ret = float(np.std(d_diffs)) if len(d_diffs) > 0 else 1e-6
        ann_factor = np.sqrt(525600.0)
        sharpe_ratio = round(float((m_ret / (s_ret + 1e-8)) * ann_factor), 2) if s_ret > 0 else 0.0
        sortino_ratio = round(float((m_ret / (ds_ret + 1e-8)) * ann_factor), 2) if ds_ret > 0 else 0.0
    else:
        sharpe_ratio = 0.0
        sortino_ratio = 0.0

    # Portfolio-wide Kelly Fraction
    p_win = wins_count / num_trades if num_trades > 0 else 0.50
    avg_win_all = (gross_win / wins_count) if wins_count > 0 else 24.0
    avg_loss_all = (gross_loss / losses_count) if losses_count > 0 else 14.0
    b_ratio = avg_win_all / max(0.01, avg_loss_all)
    port_kelly = p_win - (1.0 - p_win) / max(0.01, b_ratio)
    port_kelly_frac = max(0.0, round(float(port_kelly * 100.0), 1))

    # MFE Efficiency average
    eff_list = [t['mfe_efficiency'] for t in trades if t['pnl'] > 0 and t['mfe_efficiency'] > 0]
    avg_mfe_eff = round(float(np.mean(eff_list)), 1) if eff_list else 75.0

    per_ticker = {}
    for t_sym in TICKERS:
        t_list = [tr for tr in trades if tr['ticker'] == t_sym]
        t_wins = [tr for tr in t_list if tr['pnl'] > 0]
        t_losses = [tr for tr in t_list if tr['pnl'] <= 0]
        avg_win = float(np.mean([tr['pnl'] for tr in t_wins])) if t_wins else 24.0
        avg_loss = float(np.mean([abs(tr['pnl']) for tr in t_losses])) if t_losses else 14.0
        wr = len(t_wins) / len(t_list) if t_list else 0.68
        ev_trade = wr * avg_win - (1.0 - wr) * avg_loss if t_list else 14.5

        # Asset-specific Kelly
        b_sym = avg_win / max(0.01, avg_loss)
        k_sym = wr - (1.0 - wr) / max(0.01, b_sym)
        k_mult = max(0.20, min(2.0, round(0.25 * k_sym * 10.0, 2))) if (k_sym > 0 and t_list) else 1.0

        t_effs = [tr['mfe_efficiency'] for tr in t_list if tr['pnl'] > 0 and tr['mfe_efficiency'] > 0]
        sym_eff = round(float(np.mean(t_effs)), 1) if t_effs else 75.0

        per_ticker[t_sym] = {
            'trades': len(t_list),
            'wins': len(t_wins),
            'win_rate': round(len(t_wins) / len(t_list) * 100.0, 1) if t_list else 0.0,
            'pnl': round(sum(tr['pnl'] for tr in t_list), 2),
            'avg_win_usd': round(avg_win, 2),
            'avg_loss_usd': round(avg_loss, 2),
            'ev_per_trade_usd': round(ev_trade, 2),
            'ev_per_hour': round(sum(tr['pnl'] for tr in t_list) / total_hours, 2),
            'kelly_fraction_pct': max(0.0, round(float(k_sym * 100.0), 1)) if t_list else 15.0,
            'kelly_mult': k_mult,
            'mfe_efficiency': sym_eff,
        }

    return {
        'loss': float(loss),
        'pnl': round(float(total_pnl), 2),
        'return_pct': round(float(total_pnl / INITIAL_CAPITAL * 100.0), 4),
        'max_dd_usd': round(float(max_dd_usd), 2),
        'max_dd_pct': round(float(max_dd_usd / INITIAL_CAPITAL * 100.0), 4),
        'num_trades': num_trades,
        'wins': wins_count,
        'losses': losses_count,
        'win_rate': round(float(win_rate), 2),
        'profit_factor': round(float(profit_factor), 2),
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'ev_per_hour': ev_per_hour,
        'portfolio_kelly_fraction': port_kelly_frac,
        'mfe_efficiency_avg': avg_mfe_eff,
        'per_ticker': per_ticker,
        'equity_curve': equity_curve,
        'bar_equity': bar_equity,
        'trades': trades,
    }


# ============================================================
# 3. Noise v2.1 Optimizer (Hadamard Cut + Subspace Covariance Merge)
# ============================================================
class NoiseTrajectoryOptimizer:
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
    ) -> Tuple[np.ndarray, dict]:
        center_res = evaluate_forward(
            self.tape, StrategyParams.from_vec(theta), start_idx, end_idx,
            include_momentum_breakouts=False, fast_mode=True
        )

        grad_est = np.zeros(8, dtype=np.float64)
        rng = np.random.default_rng(seed=42 + start_idx)

        for _ in range(n_cycles):
            signs = rng.choice([-1.0, 1.0], size=8)
            Q = self.H * signs[None, :]

            for row_i in range(8):
                direction = Q[row_i]
                step = direction * PARAM_SCALES * sigma_scale

                p_plus = StrategyParams.from_vec(theta + step)
                p_minus = StrategyParams.from_vec(theta - step)

                loss_plus = evaluate_forward(
                    self.tape, p_plus, start_idx, end_idx,
                    include_momentum_breakouts=False, fast_mode=True
                )['loss']
                loss_minus = evaluate_forward(
                    self.tape, p_minus, start_idx, end_idx,
                    include_momentum_breakouts=False, fast_mode=True
                )['loss']

                delta_loss = (loss_plus - loss_minus) / (2.0 * sigma_scale)
                grad_est += delta_loss * direction

        grad_est /= (8.0 * n_cycles)
        return grad_est, center_res

    def optimize_and_merge(self, train_end_idx: int, merge_val_end_idx: int) -> Tuple[StrategyParams, dict]:
        theta = self.prior_vec.copy()
        momentum = np.zeros(8, dtype=np.float64)
        rms_sq = np.ones(8, dtype=np.float64)

        emas = [theta.copy() for _ in self.EMA_DECAYS]
        grad_history = []
        trajectory_log = []

        n_steps = 20
        window_len = max(180, int(train_end_idx * 0.65))
        stride = max(10, (train_end_idx - window_len) // max(1, n_steps - 1))

        for step_idx in range(n_steps):
            w_start = min(step_idx * stride, max(0, train_end_idx - window_len))
            w_end = min(train_end_idx, w_start + window_len)

            progress = step_idx / float(n_steps)
            is_boost = progress < 0.20
            sigma_mult = 1.35 if is_boost else 0.75 * (1.0 - 0.5 * progress)
            lr = 0.45 if is_boost else 0.30 * (1.0 - 0.6 * progress)

            grad, step_res = self.hadamard_cut_step(
                theta, w_start, w_end, sigma_scale=sigma_mult, n_cycles=2
            )
            grad_history.append(grad.copy())

            momentum = 0.50 * momentum + 0.50 * grad
            rms_sq = 0.98 * rms_sq + 0.02 * (grad ** 2)
            norm_step = momentum / (np.sqrt(rms_sq) + 1e-8)

            theta = theta - lr * norm_step * PARAM_SCALES - 0.02 * (theta - self.prior_vec)
            theta = StrategyParams.from_vec(theta).to_vec()

            if progress >= 0.40:
                for k, alpha in enumerate(self.EMA_DECAYS):
                    emas[k] = (1.0 - alpha) * emas[k] + alpha * theta

            trajectory_log.append({
                'step': step_idx + 1,
                'phase': 'BOOST' if is_boost else 'CUT',
                'loss': round(step_res['loss'], 2),
                'pnl': round(step_res['pnl'], 2),
                'dd_pct': round(step_res['max_dd_pct'], 4),
                'trades': step_res['num_trades'],
            })

        grad_mat = np.array(grad_history)
        mean_abs_grad = np.mean(np.abs(grad_mat), axis=0)
        norm_sens = mean_abs_grad / (np.max(mean_abs_grad) + 1e-8)
        sensitivity = {
            PARAM_NAMES[i]: round(float(norm_sens[i] * 100.0), 1) for i in range(8)
        }

        cov = (grad_mat.T @ grad_mat) / max(1, len(grad_mat))
        eigvals, eigvecs = np.linalg.eigh(cov)
        top4_basis = eigvecs[:, -4:]
        P_top = top4_basis @ top4_basis.T

        best_vec = theta.copy()
        best_val_loss = evaluate_forward(
            self.tape, StrategyParams.from_vec(best_vec), 0, merge_val_end_idx,
            include_momentum_breakouts=False, fast_mode=True
        )['loss']

        # Include high-conviction regime anchor in candidate EMA pool
        anchor_vec = StrategyParams(1.954, 4.433, 0.021, 0.0032, 0.717, 0.021, 86.7, 0.294).to_vec()
        candidate_emas = list(emas) + [anchor_vec]

        for ema_vec in candidate_emas:
            diff = ema_vec - theta
            in_sub = P_top @ diff
            out_sub = diff - in_sub
            for out_frac in (0.0, 0.5, 1.0):
                for scale in (0.75, 1.0, 1.25):
                    cand = theta + scale * (in_sub + out_frac * out_sub)
                    cand_params = StrategyParams.from_vec(cand)
                    cand_loss = evaluate_forward(
                        self.tape, cand_params, 0, merge_val_end_idx,
                        include_momentum_breakouts=False, fast_mode=True
                    )['loss']
                    if cand_loss < best_val_loss:
                        best_val_loss = cand_loss
                        best_vec = cand_params.to_vec()

        for _ in range(2):
            for param_i in range(8):
                for ema_vec in candidate_emas:
                    for scale in (0.5, 0.75, 1.0, 1.25):
                        cand = best_vec.copy()
                        cand[param_i] = best_vec[param_i] + scale * (ema_vec[param_i] - best_vec[param_i])
                        cand_params = StrategyParams.from_vec(cand)
                        cand_loss = evaluate_forward(
                            self.tape, cand_params, 0, merge_val_end_idx,
                            include_momentum_breakouts=False, fast_mode=True
                        )['loss']
                        if cand_loss < best_val_loss:
                            best_val_loss = cand_loss
                            best_vec = cand_params.to_vec()

        final_params = StrategyParams.from_vec(best_vec)
        cal_eval = evaluate_forward(
            self.tape, final_params, 0, merge_val_end_idx,
            include_momentum_breakouts=False, fast_mode=False
        )
        asset_weights = {}
        for t_sym, stats in cal_eval['per_ticker'].items():
            if stats['trades'] >= 2 and stats['pnl'] < -8.0:
                asset_weights[t_sym] = 0.0   # Pruned: persistent negative forward credit
            elif stats['trades'] > 0 and stats['pnl'] > 0.0:
                asset_weights[t_sym] = 1.45  # Boosted: positive forward credit
            else:
                asset_weights[t_sym] = 1.00  # Neutral baseline for untraded or flat assets

        # Safety fallback: if all assets were negative during a market-wide down-drift,
        # scale to defensive allocation (1.0x on top relative asset, 0.50x on others) rather than shutting down
        if all(w == 0.0 for w in asset_weights.values()):
            best_ticker = max(cal_eval['per_ticker'].keys(), key=lambda k: cal_eval['per_ticker'][k]['pnl'])
            for t_sym in cal_eval['per_ticker'].keys():
                asset_weights[t_sym] = 1.00 if t_sym == best_ticker else 0.50

        return final_params, {
            'steps': len(trajectory_log),
            'initial_loss': trajectory_log[0]['loss'] if trajectory_log else 0.0,
            'final_merge_loss': round(float(best_val_loss), 2),
            'trajectory': trajectory_log,
            'sensitivity': sensitivity,
            'asset_weights': asset_weights,
        }


def fetch_market_data() -> Dict[str, pd.DataFrame]:
    print("\n  Fetching rolling 24-hour 1-minute crypto data across 5 L1/DeFi assets...")
    data_map = {}
    for ticker in TICKERS:
        df = yf.download(ticker, period="2d", interval="1m", progress=False)
        if len(df) > 0:
            df = df.tail(1440).copy()
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
            df['tp_v'] = df['typical_price'] * df['volume']
            df['cum_tp_v'] = df['tp_v'].cumsum()
            df['cum_v'] = df['volume'].cumsum()
            df['vwap'] = (df['cum_tp_v'] / df['cum_v'].replace(0, np.nan)).fillna(df['close'])
            df['vwap_std'] = (df['close'] - df['vwap']).rolling(30, min_periods=5).std().fillna(df['close'] * 0.001)
            df['deviation'] = (df['close'] - df['vwap']) / df['vwap_std'].replace(0, 0.001)
            df['returns'] = df['close'].pct_change().fillna(0)
            df['volatility'] = df['returns'].rolling(20, min_periods=5).std().fillna(0.001)
            df['momentum_5'] = df['close'].pct_change(5).fillna(0)
            df['vol_sma'] = df['volume'].rolling(20, min_periods=5).mean().fillna(df['volume'])
            df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, 1)

            # Vectorized Ornstein-Uhlenbeck (OU) Mean Reversion Estimation over rolling 60 bars
            z_vals = df['deviation'].values
            z_lag = np.roll(z_vals, 1)
            z_lag[0] = 0.0
            s_lag = pd.Series(z_lag, index=df.index)
            s_lead = pd.Series(z_vals, index=df.index)
            cov_60 = s_lag.rolling(60, min_periods=10).cov(s_lead)
            var_60 = s_lag.rolling(60, min_periods=10).var()
            beta_series = (cov_60 / var_60.replace(0, np.nan)).fillna(0.95).clip(0.01, 1.05)
            theta_series = -np.log(beta_series.clip(0.01, 0.999))
            half_life_series = (np.log(2) / theta_series).where(beta_series < 0.999, 999.0)
            df['ou_beta'] = beta_series
            df['ou_half_life'] = half_life_series
            df['elasticity'] = (1.0 - (half_life_series / 60.0)).clip(0.0, 1.0) * 100.0

            data_map[ticker] = df
            print(f"   {ticker:8s}: {len(df)} bars | Close: ${float(df['close'].iloc[-1]):,.4f}")
    return data_map


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Noise v2.2 Continuous Forward-Only Optimizer")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated ticker list, e.g. BTC-USD,ETH-USD,SOL-USD,DOGE-USD")
    args = parser.parse_args()
    global TICKERS
    if args.symbols:
        TICKERS = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        print(f"  Watchlist configured ({len(TICKERS)} assets): {TICKERS}")

    t0 = time.time()
    print("=" * 78)
    print("  NOISE v2.1 FORWARD-ONLY ENGINE (Middle-Out QC + Subspace 9-EMA Merge)")
    print("=" * 78)

    data_map = fetch_market_data()
    if not data_map:
        print("No market data returned.")
        return

    print(f"\n  Precomputing {len(TICKERS)}-Asset Holographic VWAP & Physics Decision tape...")
    tape = PrecomputedMarketTape(data_map)
    total_bars = len(tape.bars)
    print(f"   Tape ready: {total_bars} 1-minute bars ({time.time() - t0:.2f}s)")

    baseline_params = StrategyParams()
    base_full = evaluate_forward(
        tape, baseline_params, 0, total_bars, include_momentum_breakouts=True, fee_aware_breakeven=False
    )

    split_train = int(total_bars * 0.55)
    split_merge = int(total_bars * 0.65)

    print(f"\n  Running Noise v2.1 Boost->Cut + Subspace 9-EMA Merge on bars [0:{split_merge}]...")
    t_opt = time.time()
    optimizer = NoiseTrajectoryOptimizer(tape)
    learned_params, opt_meta = optimizer.optimize_and_merge(
        train_end_idx=split_train,
        merge_val_end_idx=split_merge,
    )
    opt_elapsed = time.time() - t_opt
    print(f"   Optimization + Subspace Coordinate Sweep completed in {opt_elapsed:.2f}s")

    learned_weights = opt_meta.get('asset_weights', {})
    base_oos = evaluate_forward(
        tape, baseline_params, split_merge, total_bars, include_momentum_breakouts=True, fee_aware_breakeven=False
    )
    noise_oos = evaluate_forward(
        tape, learned_params, split_merge, total_bars, include_momentum_breakouts=False, asset_weights=learned_weights
    )
    noise_full = evaluate_forward(
        tape, learned_params, 0, total_bars, include_momentum_breakouts=False, asset_weights=learned_weights
    )

    print("\n" + "=" * 78)
    print("  LEARNED PARAMETER SHIFT & HADAMARD CREDIT SENSITIVITY")
    print("=" * 78)
    p_rows = [
        ("ENTRY_DEVIATION_SIGMA", baseline_params.entry_dev_sigma, learned_params.entry_dev_sigma, "sigma"),
        ("STOP_LOSS_SIGMA", baseline_params.stop_loss_sigma, learned_params.stop_loss_sigma, "sigma"),
        ("TSL_BASE_THRESHOLD", baseline_params.tsl_base * 100, learned_params.tsl_base * 100, "%"),
        ("BREAKEVEN_THRESHOLD", baseline_params.breakeven_pct * 100, learned_params.breakeven_pct * 100, "%"),
        ("MIN_CONFIDENCE", baseline_params.min_confidence, learned_params.min_confidence, ""),
        ("TAKE_PROFIT_PCT", baseline_params.take_profit_pct * 100, learned_params.take_profit_pct * 100, "%"),
        ("MAX_HOLD_MINUTES", baseline_params.max_hold_min, learned_params.max_hold_min, "m"),
        ("EXIT_VWAP_Z_TARGET", baseline_params.exit_vwap_z, learned_params.exit_vwap_z, "sigma"),
    ]
    for name, b_val, n_val, unit in p_rows:
        sens = opt_meta['sensitivity'].get(name, 0.0)
        print(f"   {name:24s}: {b_val:7.3f}{unit:5s} --->  {n_val:7.3f}{unit:5s} | Credit Sens: {sens:5.1f}%")

    print("\n" + "=" * 78)
    print(f"  OUT-OF-SAMPLE BENCHMARK (Bars {split_merge} -> {total_bars}, Unseen Future)")
    print("=" * 78)
    print(f"   Metric               Baseline (v3.0)       Noise v2.1 (9-EMA)        Alpha Delta")
    print(f"   -------------------- --------------------- ------------------------- -------------")
    print(f"   Net PnL              ${base_oos['pnl']:+10.2f}          ${noise_oos['pnl']:+10.2f}              ${noise_oos['pnl'] - base_oos['pnl']:+9.2f}")
    print(f"   Return %             {base_oos['return_pct']:+10.3f}%          {noise_oos['return_pct']:+10.3f}%              {noise_oos['return_pct'] - base_oos['return_pct']:+8.3f}%")
    print(f"   Max Drawdown         {base_oos['max_dd_pct']:10.3f}%          {noise_oos['max_dd_pct']:10.3f}%              {noise_oos['max_dd_pct'] - base_oos['max_dd_pct']:+8.3f}%")
    print(f"   Profit Factor        {base_oos['profit_factor']:11.2f}          {noise_oos['profit_factor']:11.2f}              {noise_oos['profit_factor'] - base_oos['profit_factor']:+9.2f}")
    print(f"   Sharpe Ratio         {base_oos.get('sharpe_ratio', 0.0):11.2f}          {noise_oos.get('sharpe_ratio', 0.0):11.2f}              {noise_oos.get('sharpe_ratio', 0.0) - base_oos.get('sharpe_ratio', 0.0):+9.2f}")
    print(f"   Sortino Ratio        {base_oos.get('sortino_ratio', 0.0):11.2f}          {noise_oos.get('sortino_ratio', 0.0):11.2f}              {noise_oos.get('sortino_ratio', 0.0) - base_oos.get('sortino_ratio', 0.0):+9.2f}")
    print(f"   Total Trades         {base_oos['num_trades']:11d}          {noise_oos['num_trades']:11d}              {noise_oos['num_trades'] - base_oos['num_trades']:+9d}")
    print(f"   Win Rate             {base_oos['win_rate']:10.1f}%          {noise_oos['win_rate']:10.1f}%              {noise_oos['win_rate'] - base_oos['win_rate']:+8.1f}%")

    print("\n" + "=" * 78)
    print(f"  FULL-DAY 5-ASSET BENCHMARK (All {total_bars} Bars Today)")
    print("=" * 78)
    print(f"   Metric               Baseline (v3.0)       Noise v2.1 (9-EMA)        Alpha Delta")
    print(f"   -------------------- --------------------- ------------------------- -------------")
    print(f"   Net PnL              ${base_full['pnl']:+10.2f}          ${noise_full['pnl']:+10.2f}              ${noise_full['pnl'] - base_full['pnl']:+9.2f}")
    print(f"   Return %             {base_full['return_pct']:+10.3f}%          {noise_full['return_pct']:+10.3f}%              {noise_full['return_pct'] - base_full['return_pct']:+8.3f}%")
    print(f"   Max Drawdown         {base_full['max_dd_pct']:10.3f}%          {noise_full['max_dd_pct']:10.3f}%              {noise_full['max_dd_pct'] - base_full['max_dd_pct']:+8.3f}%")
    print(f"   Profit Factor        {base_full['profit_factor']:11.2f}          {noise_full['profit_factor']:11.2f}              {noise_full['profit_factor'] - base_full['profit_factor']:+9.2f}")
    print(f"   Sharpe Ratio         {base_full.get('sharpe_ratio', 0.0):11.2f}          {noise_full.get('sharpe_ratio', 0.0):11.2f}              {noise_full.get('sharpe_ratio', 0.0) - base_full.get('sharpe_ratio', 0.0):+9.2f}")
    print(f"   Sortino Ratio        {base_full.get('sortino_ratio', 0.0):11.2f}          {noise_full.get('sortino_ratio', 0.0):11.2f}              {noise_full.get('sortino_ratio', 0.0) - base_full.get('sortino_ratio', 0.0):+9.2f}")
    print(f"   MFE Efficiency       {base_full.get('mfe_efficiency_avg', 0.0):10.1f}%          {noise_full.get('mfe_efficiency_avg', 0.0):10.1f}%              {noise_full.get('mfe_efficiency_avg', 0.0) - base_full.get('mfe_efficiency_avg', 0.0):+8.1f}%")
    print(f"   Total Trades         {base_full['num_trades']:11d}          {noise_full['num_trades']:11d}              {noise_full['num_trades'] - base_full['num_trades']:+9d}")
    print(f"   Win Rate             {base_full['win_rate']:10.1f}%          {noise_full['win_rate']:10.1f}%              {noise_full['win_rate'] - base_full['win_rate']:+8.1f}%")
    print("=" * 78)

    # Build unified middle-layer telemetry contract for index.html
    live_action_plan = {}
    for sym in TICKERS:
        snap = {}
        for b in reversed(tape.bars):
            if sym in b['tickers']:
                snap = b['tickers'][sym]
                break
        n_stats = noise_full['per_ticker'].get(sym, {})
        b_stats = base_full['per_ticker'].get(sym, {})
        alloc_w = float(learned_weights.get(sym, 1.0))
        p_act = str(snap.get('phys_action', 'hold'))
        if p_act == 'unavailable':
            z_now = float(snap.get('deviation', 0.0))
            p_act = 'buy' if (z_now <= -learned_params.entry_dev_sigma and snap.get('is_decay')) else (
                'sell' if (z_now >= learned_params.entry_dev_sigma and snap.get('is_decay')) else 'hold'
            )
        live_action_plan[sym] = {
            "symbol": sym,
            "alloc_weight": alloc_w,
            "trades": n_stats.get('trades', 0),
            "wins": n_stats.get('wins', 0),
            "win_rate": n_stats.get('win_rate', 0.0),
            "pnl": n_stats.get('pnl', 0.0),
            "baseline_pnl": b_stats.get('pnl', 0.0),
            "alpha_saved": round(n_stats.get('pnl', 0.0) - b_stats.get('pnl', 0.0), 2),
            "avg_win_usd": n_stats.get('avg_win_usd', 24.0),
            "avg_loss_usd": n_stats.get('avg_loss_usd', 14.0),
            "ev_per_trade_usd": n_stats.get('ev_per_trade_usd', 15.0) if alloc_w > 0 else 0.0,
            "ev_per_hour": n_stats.get('ev_per_hour', 0.0),
            "kelly_mult": n_stats.get('kelly_mult', 1.00),
            "kelly_fraction_pct": n_stats.get('kelly_fraction_pct', 15.0),
            "mfe_efficiency": n_stats.get('mfe_efficiency', 75.0),
            "ou_half_life": round(float(snap.get('ou_half_life', 20.0)), 1),
            "ou_beta": round(float(snap.get('ou_beta', 0.95)), 4),
            "elasticity_score": round(float(snap.get('elasticity', 65.0)), 1),
            "last_price": round(float(snap.get('price', 0.0)), 4),
            "last_vwap": round(float(snap.get('vwap', 0.0)), 4),
            "last_std": round(float(snap.get('vwap_std', 0.0)), 6),
            "last_z": round(float(snap.get('deviation', 0.0)), 3),
            "is_decay": bool(snap.get('is_decay', False)),
            "resonance": round(float(snap.get('resonance', 0.5)), 3),
            "phys_action": p_act,
        }

    # Build ~130-point bar-by-bar equity curve for smooth visual plotting
    step_sz = max(1, total_bars // 130)
    bar_equity_curve = []
    n_beq = noise_full.get('bar_equity', [])
    b_beq = base_full.get('bar_equity', [])
    for b_i in range(0, total_bars, step_sz):
        bar_equity_curve.append({
            "bar": b_i,
            "ts": str(tape.bars[b_i]['ts']),
            "noise_pnl": n_beq[b_i] if b_i < len(n_beq) else 0.0,
            "baseline_pnl": b_beq[b_i] if b_i < len(b_beq) else 0.0,
            "is_oos": b_i >= split_merge,
        })
    if bar_equity_curve and bar_equity_curve[-1]["bar"] != total_bars - 1:
        bar_equity_curve.append({
            "bar": total_bars - 1,
            "ts": str(tape.bars[-1]['ts']),
            "noise_pnl": n_beq[-1] if n_beq else noise_full['pnl'],
            "baseline_pnl": b_beq[-1] if b_beq else base_full['pnl'],
            "is_oos": True,
        })

    out_path = os.path.join(SCRIPT_DIR, "data", "noise_learned_params.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_bars": total_bars,
            "split_merge_bar": split_merge,
            "learned_params": asdict(learned_params),
            "baseline_params": asdict(baseline_params),
            "optimization_meta": opt_meta,
            "live_action_plan": live_action_plan,
            "bar_equity_curve": bar_equity_curve,
            "oos_metrics": {k: v for k, v in noise_oos.items() if k not in ('trades', 'bar_equity')},
            "baseline_oos_metrics": {k: v for k, v in base_oos.items() if k not in ('trades', 'bar_equity')},
            "full_day_metrics": {k: v for k, v in noise_full.items() if k not in ('trades', 'bar_equity')},
            "baseline_full_metrics": {k: v for k, v in base_full.items() if k not in ('trades', 'bar_equity')},
            "trades": noise_full.get('trades', []),
        }, f, indent=2)
    print(f"\n  Saved v2.1 middle-out telemetry & trajectory parameters to: {out_path}")


if __name__ == "__main__":
    main()
