#!/usr/bin/env python3
"""
POSITION MANAGER - Trailing Stops & Anti-Whipsaw Protection v3.0
=================================================================
Manages active trade states, handles trailing stop-loss logic, 
and prevents over-trading during high-volatility "whipsaw" regimes.

v3.0 Improvements (Backtest-Driven):
- Progressive pause system (5m → 30m → 1h based on severity)
- FIXED: Volatility-adaptive TSL (high vol = wide, low vol = tight)
- FIXED: Regime change logging (no more log spam)
- NEW: Profit target exit (2%) and time-based exit (30 min)
- Breakeven stop after 0.7% profit
- Win/Loss streak tracking for position sizing
"""

import os
import json
import time
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

logger = logging.getLogger("PositionManager")

@dataclass
class Position:
    symbol: str
    side: str  # 'BUY' (Long WETH) or 'SELL' (Short WETH)
    entry_price: float
    peak_price: float
    stop_loss: float
    amount_weth: float
    timestamp: float
    metadata: dict
    breakeven_activated: bool = False  # NEW: Track if breakeven stop is active

class PositionManager:
    def __init__(self, tsl_threshold: float = 0.020, whipsaw_threshold: int = 3):
        self.positions: Dict[str, Position] = {}
        self.base_tsl_threshold = tsl_threshold  # Use constructor arg (default 2%)
        self.tsl_threshold = tsl_threshold
        self.whipsaw_threshold = whipsaw_threshold
        self.loss_streak = 0
        self.win_streak = 0
        self.last_pause_time = 0
        
        # Regime tracking (prevents log spam)
        self._current_regime = 'normal'
        
        # PROGRESSIVE PAUSE DURATIONS (based on loss severity)
        self.pause_tiers = {
            3: 300,    # 3 losses = 5 min cooldown
            5: 1800,   # 5 losses = 30 min cooldown
            7: 3600,   # 7+ losses = 1 hour full pause
        }
        
        # BREAKEVEN THRESHOLD: Move stop to entry after this profit
        self.breakeven_threshold = 0.007  # 0.7% profit triggers breakeven
        
        # PROFIT TARGET: Hard exit at this profit level
        self.profit_target = 0.02  # 2% take-profit
        
        # TIME EXIT: Force close after this many seconds
        self.max_hold_seconds = 1800  # 30 minutes
        
        # Load Noise forward-only trajectory-merged parameters if available
        noise_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "noise_learned_params.json")
        if os.path.exists(noise_file):
            try:
                with open(noise_file, "r") as nf:
                    n_cfg = json.load(nf).get("learned_params", {})
                if "tsl_base" in n_cfg:
                    self.base_tsl_threshold = float(n_cfg["tsl_base"])
                    self.tsl_threshold = self.base_tsl_threshold
                if "breakeven_pct" in n_cfg:
                    self.breakeven_threshold = float(n_cfg["breakeven_pct"])
                if "take_profit_pct" in n_cfg:
                    self.profit_target = float(n_cfg["take_profit_pct"])
                if "max_hold_min" in n_cfg:
                    self.max_hold_seconds = int(float(n_cfg["max_hold_min"]) * 60)
                logger.info(
                    f"Loaded Noise Trajectory Parameters: TSL={self.base_tsl_threshold:.2%}, "
                    f"BE={self.breakeven_threshold:.2%}, TP={self.profit_target:.2%}, "
                    f"MaxHold={self.max_hold_seconds//60}m"
                )
            except Exception as e:
                logger.warning(f"Could not load noise_learned_params.json: {e}")

        self.state_file = "position_state.json"
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    for sym, pos_data in data.get('positions', {}).items():
                        # Handle old format without breakeven_activated
                        if 'breakeven_activated' not in pos_data:
                            pos_data['breakeven_activated'] = False
                        self.positions[sym] = Position(**pos_data)
                    self.loss_streak = data.get('loss_streak', 0)
                    self.win_streak = data.get('win_streak', 0)
                    self.last_pause_time = data.get('last_pause_time', 0)
            except Exception as e:
                logger.error(f"Failed to load position state: {e}")

    def _save_state(self):
        try:
            data = {
                'positions': {sym: asdict(pos) for sym, pos in self.positions.items()},
                'loss_streak': self.loss_streak,
                'win_streak': self.win_streak,
                'last_pause_time': self.last_pause_time
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save position state: {e}")

    def get_pause_duration(self) -> int:
        """Get pause duration based on loss streak severity."""
        for threshold, duration in sorted(self.pause_tiers.items(), reverse=True):
            if self.loss_streak >= threshold:
                return duration
        return 0

    def set_market_regime(self, btc_1m_move: float):
        """Adjust TSL threshold based on market volatility regime.
        
        FIXED v3.0: Corrected inverted logic.
        - High vol → WIDEN stops (avoid noise exits)
        - Low vol → TIGHTEN stops (take profits faster, less room needed)
        - Only logs on regime CHANGES (eliminates spam)
        """
        abs_move = abs(btc_1m_move)
        
        if abs_move > 0.005:  # High volatility (> 0.5% in 1 min)
            new_regime = 'high_vol'
            self.tsl_threshold = 0.030  # Widen to 3.0% — give room on volatile days
        elif abs_move < 0.001:  # Low volatility (< 0.1% in 1 min)
            new_regime = 'low_vol'
            self.tsl_threshold = 0.008  # Tighten to 0.8% — capture small moves
        else:
            new_regime = 'normal'
            self.tsl_threshold = self.base_tsl_threshold  # Default 2.0%
        
        # Only log on regime CHANGES (fixes log spam)
        if new_regime != self._current_regime:
            self._current_regime = new_regime
            regime_labels = {'high_vol': '🔥 HIGH VOL', 'low_vol': '😴 LOW VOL', 'normal': '⚖️ NORMAL'}
            logger.info(f"{regime_labels[new_regime]} REGIME: TSL set to {self.tsl_threshold:.2%}")

    def get_position_size_multiplier(self) -> float:
        """Calculate position size multiplier based on streak."""
        if self.win_streak >= 2:
            return 1.25  # 25% increase after 2 wins
        elif self.loss_streak >= 2:
            return 0.5   # 50% reduction after 2 losses
        return 1.0

    def add_position(self, symbol: str, side: str, entry_price: float, amount_weth: float, metadata: dict = None):
        """Register a new position."""
        # Initial stop uses current TSL threshold (volatility-adaptive)
        if side == 'BUY':
            stop_loss = entry_price * (1 - self.tsl_threshold)
        else:
            stop_loss = entry_price * (1 + self.tsl_threshold)
        
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            peak_price=entry_price,
            stop_loss=stop_loss,
            amount_weth=amount_weth,
            timestamp=time.time(),
            metadata=metadata or {},
            breakeven_activated=False
        )
        self._save_state()
        logger.info(f"🆕 Position Added: {side} {symbol} @ ${entry_price:.2f} | Stop: ${stop_loss:.2f} ({self.tsl_threshold:.1%})")

    def update_peak(self, symbol: str, current_price: float):
        """Update peak price and shift trailing stop."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        
        # BREAKEVEN CHECK: If up 0.5%, move stop to entry price
        if not pos.breakeven_activated:
            profit_pct = (current_price - pos.entry_price) / pos.entry_price
            if pos.side == 'SELL': profit_pct = -profit_pct
            
            if profit_pct >= self.breakeven_threshold:
                pos.breakeven_activated = True
                pos.stop_loss = pos.entry_price
                logger.info(f"🛡️ BREAKEVEN STOP: {symbol} now risk-free @ ${pos.entry_price:.2f}")
                self._save_state()
                return
        
        # TRAILING STOP LOGIC
        if pos.side == 'BUY':
            if current_price > pos.peak_price:
                pos.peak_price = current_price
                new_stop = current_price * (1 - self.tsl_threshold)
                if new_stop > pos.stop_loss:
                    pos.stop_loss = new_stop
                    logger.info(f"📈 TSL UP: {symbol} Peak ${current_price:.2f}, New Stop ${new_stop:.2f}")
        else: # SELL (Shorting) - peak is lowest price
            if current_price < pos.peak_price:
                pos.peak_price = current_price
                new_stop = current_price * (1 + self.tsl_threshold)
                if new_stop < pos.stop_loss:
                    pos.stop_loss = new_stop
                    logger.info(f"📉 TSL DOWN: {symbol} Low ${current_price:.2f}, New Stop ${new_stop:.2f}")
        
        self._save_state()

    def check_exits(self, current_prices: Dict[str, float]) -> List[Position]:
        """Check if any positions should be exited.
        
        v3.0: Added profit target and time-based exits.
        Priority: Profit Target > Stop Loss > Time Exit
        """
        to_exit = []
        now = time.time()
        
        for symbol, pos in list(self.positions.items()):
            price = current_prices.get(symbol)
            if not price:
                continue
                
            self.update_peak(symbol, price)
            
            # Calculate current PnL
            pnl_pct = (price - pos.entry_price) / pos.entry_price
            if pos.side == 'SELL': pnl_pct = -pnl_pct
            
            should_exit = False
            reason = ""
            
            # 1. PROFIT TARGET (highest priority — lock in winners)
            if pnl_pct >= self.profit_target:
                should_exit = True
                reason = f"🎯 Take Profit ({pnl_pct:.2%})"
            
            # 2. STOP LOSS / TRAILING STOP
            elif pos.side == 'BUY' and price <= pos.stop_loss:
                should_exit = True
                reason = "🛡️ Breakeven Stop" if pos.breakeven_activated else "📉 Trailing Stop"
            elif pos.side == 'SELL' and price >= pos.stop_loss:
                should_exit = True
                reason = "🛡️ Breakeven Stop" if pos.breakeven_activated else "📈 Trailing Stop"
            
            # 3. TIME EXIT (positions shouldn't linger in a scalping system)
            elif (now - pos.timestamp) > self.max_hold_seconds:
                should_exit = True
                hold_mins = (now - pos.timestamp) / 60
                reason = f"⏰ Time Exit ({hold_mins:.0f}m)"
                
            if should_exit:
                logger.info(f"🚫 EXIT: {symbol} {reason} @ ${price:.2f} (PnL: {pnl_pct:+.2%})")
                to_exit.append(pos)
                del self.positions[symbol]
                
                if pnl_pct < 0:
                    self.loss_streak += 1
                    self.win_streak = 0
                else:
                    self.win_streak += 1
                    self.loss_streak = 0
                
                # PROGRESSIVE PAUSE
                if self.loss_streak >= self.whipsaw_threshold:
                    self.last_pause_time = time.time()
                    pause_mins = self.get_pause_duration() // 60
                    logger.warning(f"⚠️ WHIPSAW PROTECTION: {self.loss_streak} losses → Pausing {pause_mins}m")
        
        if to_exit:
            self._save_state()
            
        return to_exit

    def is_paused(self) -> bool:
        """Check if trading is currently paused due to whipsaw protection."""
        if self.last_pause_time == 0:
            return False
        
        pause_duration = self.get_pause_duration()
        elapsed = time.time() - self.last_pause_time
        
        if elapsed < pause_duration:
            return True
        else:
            # Pause expired, reset
            self.last_pause_time = 0
            # Don't reset streak - it stays until a win
            self._save_state()
            logger.info(f"✅ Whipsaw pause expired. Resuming trading (streak: {self.loss_streak})")
            return False
