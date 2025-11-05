import pandas as pd
import numpy as np
import json
import uuid
import os
import math
from datetime import datetime, time as dtime
from dataclasses import dataclass, field
from queue import Queue, Empty
import logging

# ==================================================================================================
# --- IMPORT SENTINEL-PRIME CORE COMPONENTS ---
# ==================================================================================================
try:
    from main1 import (
        AbstractTrader, Position, PositionStatus, OrderSide, OptionType,
        RiskManager, PositionManager, MicrostructureMonitor, RegimeClassifier,
        BarStore, InstrumentBook, Engine, StoreActor, OrderActor,
        BaseStrategy, MomentumBreakoutStrategy, TrendPullbackStrategy,
        MeanReversionStrategy, VolatilityMeanReversionStrategy, OpeningRangeBreakout,
        StrategyName, Regime, Clock,
        load_config, now_ist, _get_underlying,
        black_scholes_price, bs_delta, calculate_greeks,
        calculate_historical_volatility, calculate_iv, _calculate_time_to_expiry,
        PERSIST_DIR, IST
    )
except ImportError as e:
    print(f"FATAL: Could not import core classes from main1.py. Make sure this file is in the same directory.")
    print(f"Error: {e}")
    exit(1)

L = logging.getLogger("SENTINEL-PRIME-BACKTESTER")
L.setLevel(logging.INFO)
if not L.handlers:
    L.addHandler(logging.StreamHandler())

# ==================================================================================================
# --- 1. SIMULATED "SENSES" (HIGH-FIDELITY PRICEBUS) ---
# ==================================================================================================

class SimulatedPriceBus:
    """
    A high-fidelity "fake" PriceBus. It serves real futures data
    and dynamically estimates option prices on-the-fly when asked.
    """
    def __init__(self, book: InstrumentBook, engine: 'BacktestEngine'):
        self.book = book
        self.engine = engine # To get IV cache, config, etc.
        self._current_futures_ticks = {}
        self._current_futures_bar = None
        self._risk_free_rate = 0.05 # Hardcoded for BS
        self._market_close_time = dtime.fromisoformat("15:30:00")
        
        # Cache for estimated prices to avoid re-calculating 5x per bar
        self._bar_option_cache = {}

    def update_futures_bar(self, bar: pd.Series):
        """Called by the BacktestEngine on each new bar."""
        self._current_futures_bar = bar
        self._bar_option_cache = {} # Clear cache on new bar
        
        # We only update the FUTURES ticks
        for token, prefix in [(256265, "NIFTY"), (260105, "BN"), (257281, "VIX")]:
            if f"{prefix}_open" in bar.index:
                self._current_futures_ticks[token] = {
                    "instrument_token": token,
                    "last_price": bar[f"{prefix}_close"],
                    "open": bar[f"{prefix}_open"],
                    "high": bar[f"{prefix}_high"],
                    "low": bar[f"{prefix}_low"],
                    "close": bar[f"{prefix}_close"],
                    "volume": bar[f"{prefix}_volume"],
                    "depth": { # Simulate basic depth
                        "buy": [{"price": bar[f"{prefix}_close"] * 0.999, "quantity": 100}],
                        "sell": [{"price": bar[f"{prefix}_close"] * 1.001, "quantity": 100}]
                    }
                }
    
    def ltp(self, token: int) -> float | None:
        """Gets the Last Traded Price (simulated)."""
        if token in self._current_futures_ticks:
            return self._current_futures_ticks[token]["last_price"]
        
        # If it's an option, estimate its LTP
        tick = self.get_full_tick(token)
        return tick.get("last_price") if tick else None

    def get_full_tick(self, token: int) -> dict | None:
        """
        --- THIS IS THE CORE OF THE SIMULATION ---
        If a future is requested, returns the bar data.
        If an option is requested, it calculates its price *on-the-fly*.
        """
        # 1. Check if it's a future
        if token in self._current_futures_ticks:
            return self._current_futures_ticks[token]
            
        # 2. Check if it's an option we've already priced this bar
        if token in self._bar_option_cache:
            return self.get_estimated_option_ohlc(token) # Returns full bar dict

        # 3. If it's a new option request, estimate its price (LTP)
        try:
            opt_details = self.book.df_by_token.loc[token]
            strike = opt_details['strike']
            is_call = opt_details['instrument_type'] == 'CE'
            expiry_date = opt_details['expiry'].date()
            
            underlying_name = _get_underlying(opt_details['name'])
            underlying_token = 256265 if "NIFTY" in underlying_name else 260105
            
            spot = self.ltp(underlying_token)
            if not spot: return None # Can't price
                
            T = _calculate_time_to_expiry(expiry_date, now_ist(), self._market_close_time)
            
            # Get IV from the *real* engine's cache
            sigma = self.engine.engine.atm_iv_cache.get(underlying_name, 0.2) # Default 20% IV
            
            price = black_scholes_price(spot, strike, T, self._risk_free_rate, sigma, is_call)
            
            # Return a "fake" tick
            return {
                "instrument_token": token,
                "last_price": price,
                "volume": 100000, # Fake
                "open_interest": 100000, # Fake
                "depth": { # Return a perfectly neutral book so OBI is 0
                    "buy": [{"price": price * 0.999, "quantity": 100}],
                    "sell": [{"price": price * 1.001, "quantity": 100}]
                }
            }
        except Exception as e:
            # L.warning(f"Failed to estimate price for token {token}: {e}")
            return None
            
    def get_estimated_option_ohlc(self, token: int) -> dict | None:
        """
        Estimates the OHLC for an option token based on the
        underlying futures bar AND the VIX bar.
        """
        if token in self._bar_option_cache:
            return self._bar_option_cache[token]
            
        try:
            # --- THIS IS THE NEW CODE ---
            # Get the fudge factor from the config, defaulting to 0%
            backtest_cfg = self.engine.config.get("backtester", {})
            fudge_factor = backtest_cfg.get("volatility_fudge_factor_pct", 0.0) / 100.0
            # --- END NEW CODE ---

            opt_details = self.book.df_by_token.loc[token]
            strike = opt_details['strike']
            is_call = opt_details['instrument_type'] == 'CE'
            expiry_date = opt_details['expiry'].date()
            
            underlying_name = _get_underlying(opt_details['name'])
            underlying_token = 256265 if "NIFTY" in underlying_name else 260105
            
            fut_bar = self._current_futures_ticks.get(underlying_token)
            if not fut_bar: return None
                
            T = _calculate_time_to_expiry(expiry_date, now_ist(), self._market_close_time)

            # --- MODIFICATION START ---
            # REMOVED: sigma = self.engine.engine.atm_iv_cache.get(underlying_name, 0.2)
            
            # 1. Get the VIX bar for dynamic IV
            #    (257281 is the INDIA VIX token, which update_futures_bar already processes)
            vix_bar = self._current_futures_ticks.get(257281) 
            
            if not vix_bar:
                # Fallback if VIX data is missing for this bar
                L.warning(f"VIX bar not found for IV simulation. Falling back to 20% flat IV.")
                sigma_open = sigma_high = sigma_low = sigma_close = 0.20
            else:
                # 2. Use VIX OHLC for dynamic IV, ensuring a floor of 1%
                sigma_open = max(0.01, vix_bar['open'] / 100.0)
                sigma_high = max(0.01, vix_bar['high'] / 100.0)
                sigma_low = max(0.01, vix_bar['low'] / 100.0)
                sigma_close = max(0.01, vix_bar['close'] / 100.0)

            # 3. Calculate OHLC prices using the corresponding dynamic IV
            opt_open = black_scholes_price(fut_bar['open'], strike, T, self._risk_free_rate, sigma_open, is_call)
            opt_high = black_scholes_price(fut_bar['high'], strike, T, self._risk_free_rate, sigma_high, is_call)
            opt_low = black_scholes_price(fut_bar['low'], strike, T, self._risk_free_rate, sigma_low, is_call)
            opt_close = black_scholes_price(fut_bar['close'], strike, T, self._risk_free_rate, sigma_close, is_call)
            # --- MODIFICATION END ---


            # Handle call/put high/low inversion
            if is_call:
                o_h, o_l = opt_high, opt_low
            else:
                # Inverted for puts: a drop in underlying (fut_bar['low'])
                # combined with a spike in IV (sigma_high) creates the
                # *highest* price for the put option.
                o_h, o_l = opt_low, opt_high # Inverted for puts

            # --- APPLY THE FUDGE FACTOR ---
            # Artificially widen the bar to simulate real-world gamma/vega risk
            # We add a % to the high and subtract a % from the low.
            o_h = o_h * (1.0 + fudge_factor)
            o_l = o_l * (1.0 - fudge_factor)
            o_l = max(0.01, o_l) # Ensure low price doesn't go to or below zero
            # --- END MODIFICATION ---

            bar_dict = {
                "instrument_token": token,
                "last_price": opt_close,
                "open": opt_open,
                "high": o_h,  # <-- Use modified high
                "low": o_l,   # <-- Use modified low
                "close": opt_close,
                "volume": 100000, # Fake
                "last_traded_quantity": 100, # Fake
                "exchange_timestamp": now_ist()
            }
            self._bar_option_cache[token] = bar_dict # Cache it
            return bar_dict
            
        except Exception as e:
            # L.warning(f"Failed to estimate OHLC for token {token}: {e}")
            return None
    
    def subscribe(self, tokens: list[int]):
        pass # Not needed for simulation

# ==================================================================================================
# --- 2. SIMULATED "LIMBS" (FAKE TRADER) ---
# ==================================================================================================

@dataclass
class ClosedTradeLog:
    id: str
    tradingsymbol: str
    strategy: str
    regime: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    exit_reason: str
    
class SimulatedTrader(AbstractTrader):
    def __init__(self,
                 engine: 'BacktestEngine',
                 book: InstrumentBook,
                 prices: SimulatedPriceBus,
                 store: 'SimulatedStore',
                 config: dict):
        
        super().__init__(engine, book, prices, store, config, perf_callback=None)
        
        self.engine = engine
        self.prices = prices
        self.store = store
        
        self.backtest_config = config.get("backtester", {})
        self.slippage_pct = self.backtest_config.get("slippage_pct", 0.05) / 100.0
        self.commission_per_lot = self.backtest_config.get("commission_per_lot", 40)
        
        self.positions: dict[str, Position] = {}
        self.closed_trades_log: list[ClosedTradeLog] = []
        self.trade_id_counter = 1
        self.daily_realized_pnl = 0.0

    def _get_lot_size(self, tradingsymbol: str) -> int:
        underlying = _get_underlying(tradingsymbol)
        return self.book.lot_size(underlying)

    def _calculate_fill_price(self, target_price: float, side: OrderSide) -> float:
        if side == OrderSide.BUY:
            return target_price * (1 + self.slippage_pct)
        else: # SELL
            return target_price * (1 - self.slippage_pct)

    def _calculate_commission(self, tradingsymbol: str, qty: int) -> float:
        lot_size = self._get_lot_size(tradingsymbol)
        if not lot_size or lot_size == 0:
            L.warning(f"Could not find lot size for {tradingsymbol}, commission will be 0")
            return 0
        lots = qty / lot_size
        return lots * self.commission_per_lot # Commission is for a round trip

    def open_position(self, trade_params: dict) -> Position | None:
        """
        Simulates an immediate fill. The `trade_params` now contain
        the *real* option contract chosen by the planner.
        """
        current_bar = self.prices._current_futures_bar
        if current_bar is None: return None

        opt = trade_params['opt']
        
        # Get the estimated fill price, which was already calculated by the
        # planner and stored in `ltp_opt`. We'll use this as the "open" price.
        target_price = trade_params['ltp_opt']
        fill_price = self._calculate_fill_price(target_price, OrderSide.BUY)
        
        lot_size = self._get_lot_size(opt['tradingsymbol'])
        if not lot_size: return None
             
        qty = trade_params['lots'] * lot_size
        pos_id = f"BACKTEST_{self.trade_id_counter}"
        self.trade_id_counter += 1

        pos = Position(
            id=pos_id,
            tradingsymbol=opt['tradingsymbol'],
            token=int(opt['instrument_token']),
            option_type=opt['instrument_type'],
            qty=qty,
            initial_qty=qty,
            entry_price=fill_price,
            initial_sl_price=fill_price - trade_params['option_sl_points'],
            sl_price=fill_price - trade_params['option_sl_points'],
            tp_price=fill_price + trade_params['option_tp_points'],
            opened_at=current_bar.name, # bar.name is the timestamp
            strategy=trade_params['strategy'],
            market_regime_at_entry=trade_params['regime'],
            underlying_sl_level=trade_params['underlying_sl'],
            status=PositionStatus.ACTIVE.value,
            entry_order_id=pos_id,
            scaled_out_qty=0,
            trailing_sl_armed=False,
            initial_risk_points=trade_params['option_sl_points'],
            option_sl_points=trade_params['option_sl_points'],
            option_tp_points=trade_params['option_tp_points'],
            high_price_since_entry=fill_price,
            scale_out_rules=trade_params['scale_out_rules'],
            greeks=trade_params['greeks'],
            max_trade_duration_minutes=trade_params.get('max_trade_duration_minutes', 90),
            oi_profit_target=trade_params.get('oi_profit_target'),
            intended_risk_rupees=trade_params.get('total_trade_risk', 0.0),
        )
        
        self.positions[pos.id] = pos
        self.store.upsert_position(pos)
        
        L.info(f"[{current_bar.name}] ✅ OPENED {pos.strategy} {pos.tradingsymbol} Qty={pos.qty} @ {pos.entry_price:.2f}")
        return pos

    def close_position(self, p: Position, reason: str, exit_price: float) -> bool:
        if p.status == PositionStatus.CLOSED.value:
            return True

        p.status = PositionStatus.CLOSED.value
        p.exit_reason = reason
        
        final_exit_price = self._calculate_fill_price(exit_price, OrderSide.SELL)
        commission = self._calculate_commission(p.tradingsymbol, p.initial_qty)
        
        pnl = (final_exit_price - p.entry_price) * p.initial_qty - commission
        self.daily_realized_pnl += pnl
        
        self.engine.risk_manager.update_performance_metrics(pnl)
        
        log_entry = ClosedTradeLog(
            id=p.id, tradingsymbol=p.tradingsymbol,
            strategy=p.strategy, regime=p.market_regime_at_entry,
            entry_time=p.opened_at, exit_time=self.prices._current_futures_bar.name,
            entry_price=p.entry_price, exit_price=final_exit_price,
            qty=p.initial_qty, pnl=pnl, exit_reason=reason
        )
        
        self.closed_trades_log.append(log_entry)
        self.positions.pop(p.id, None)
        self.store.log_closed_trade(p, final_exit_price, reason)
        
        L.info(f"[{self.prices._current_futures_bar.name}] ❌ CLOSED {p.tradingsymbol} @ {final_exit_price:.2f} ({reason}). PnL: {pnl:.2f}")
        return True

    def modify_sl(self, p: Position, new_trigger: float):
        if new_trigger > p.sl_price:
            L.debug(f"[{self.prices._current_futures_bar.name}] 📈 TSL {p.tradingsymbol} from {p.sl_price:.2f} to {new_trigger:.2f}")
            p.sl_price = new_trigger
            self.store.upsert_position(p)

    def scale_out(self, p: Position, qty_to_close: int) -> bool:
        if qty_to_close <= 0 or qty_to_close > p.qty:
            return False
            
        # Estimate current price
        current_price = self.prices.ltp(p.token)
        if not current_price:
            L.warning(f"Could not get LTP for {p.token} to scale out.")
            return False
            
        exit_price = self._calculate_fill_price(current_price, OrderSide.SELL)
        commission = self._calculate_commission(p.tradingsymbol, qty_to_close)
        pnl = (exit_price - p.entry_price) * qty_to_close - commission
        
        self.daily_realized_pnl += pnl
        self.engine.risk_manager.update_performance_metrics(pnl)
        
        L.info(f"[{self.prices._current_futures_bar.name}] 💰 SCALED OUT {qty_to_close} of {p.tradingsymbol} @ {exit_price:.2f}. PnL: {pnl:.2f}")
        
        p.qty -= qty_to_close
        p.scaled_out_qty += qty_to_close
        p.status = PositionStatus.PARTIALLY_CLOSED.value
        self.store.upsert_position(p)
        return True

    def unrealized_pnl(self) -> float:
        pnl = 0.0
        for p in self.positions.values():
            if p.status in [PositionStatus.ACTIVE.value, PositionStatus.PARTIALLY_CLOSED.value]:
                current_price = self.prices.ltp(p.token)
                if current_price:
                    pnl += (current_price - p.entry_price) * p.qty
        return pnl

    def place_bracket_orders(self, p: Position) -> bool:
        p.status = PositionStatus.ACTIVE.value
        return True
    def cancel_pending_entry(self, p: Position) -> bool: return True
    def execute_simulated_sl(self, p: Position) -> bool: return True

# ==================================================================================================
# --- 3. MODIFIED POSITION MANAGER (WITH FULL TSL) ---
# ==================================================================================================

class BacktestPositionManager(PositionManager):
    """
    Overrides the real PositionManager to work with historical bars
    and includes the FULL TSL logic from main1.py.
    """
    
    # --- COPIED FROM main1.py ---
    def _calculate_trailing_stop(self, p: Position) -> float | None:
        try:
            trail_params = self.trailing_sl_config
            trail_tf = trail_params["timeframe_scaled_out"] if p.triggered_scale_out_targets else trail_params["timeframe"]
            
            # This now reads the ESTIMATED option bar data from the BarStore
            df = self.engine.get_ohlc(p.token, trail_tf) 

            period = trail_params["chandelier_period"]
            if len(df) < period:
                return None

            try:
                import pandas_ta as ta
                df.ta.atr(length=period, append=True)
                atr_col = f'ATRr_{period}'
                if atr_col not in df.columns:
                     raise Exception(f"Failed to calculate ATR, column {atr_col} not found.")
                atr = df[atr_col].iloc[-1]
            except Exception as e:
                L.warning(f"Could not get ATR for TSL: {e}. Install pandas-ta.")
                return None

            if pd.isna(atr):
                return None

            multiplier = trail_params["chandelier_multiplier_scaled_out"] if p.triggered_scale_out_targets else trail_params["chandelier_multiplier"]
            high_over_period = df['high'].rolling(period).max().iloc[-1]
            new_sl_price = high_over_period - atr * multiplier
            return new_sl_price
        except Exception as e:
            L.warning(f"Could not calculate trailing stop for {p.tradingsymbol}: {e}")
            return None
    # --- END COPY ---

    def update_option_bars_for_open_positions(self):
        """
        Estimates and adds the current bar's OHLC for all open
        options to the BarStore. This is VITAL for the TSL.
        """
        for p in self.trader.positions.values():
            if p.status in [PositionStatus.ACTIVE.value, PositionStatus.PARTIALLY_CLOSED.value]:
                # Get the estimated OHLC bar for this option
                opt_bar = self.prices.get_estimated_option_ohlc(p.token)
                if opt_bar:
                    # Add it to the BarStore so TSL can read it
                    self.engine.bars.add_tick(opt_bar)

    def manage_positions_backtest(self, bar_timestamp: datetime):
        """
        Checks for SL/TP hits against the estimated High/Low.
        """
        with self.trader.lock:
            active_positions = list(self.trader.positions.values())

        now = bar_timestamp

        for p in active_positions:
            with self.trader.lock:
                if p.id not in self.trader.positions:
                    continue
            
            # Get the estimated OHLC for this option
            opt_bar = self.prices.get_estimated_option_ohlc(p.token)
            if not opt_bar:
                L.warning(f"Could not get estimated OHLC for {p.tradingsymbol}, skipping mgmt.")
                continue
                
            bar_high = opt_bar['high']
            bar_low = opt_bar['low']
            ltp = opt_bar['close']

            # --- CRITICAL HIT CHECKING ---
            if bar_low <= p.sl_price:
                L.debug(f"[{now}] SL HIT for {p.tradingsymbol} (BarLow: {bar_low:.2f} <= SL: {p.sl_price:.2f})")
                self.trader.close_position(p, "SL_HIT_BACKTEST", p.sl_price)
                continue

            if bar_high >= p.tp_price and p.tp_price > 0:
                L.debug(f"[{now}] TP HIT for {p.tradingsymbol} (BarHigh: {bar_high:.2f} >= TP: {p.tp_price:.2f})")
                self.trader.close_position(p, "TP_HIT_BACKTEST", p.tp_price)
                continue

            if (now - p.opened_at).total_seconds() / 60 > p.max_trade_duration_minutes:
                L.info(f"[{now}] TIME STOP for {p.tradingsymbol}. Closing at bar open.")
                self.trader.close_position(p, "TIME_STOP_EXIT", opt_bar['open'])
                continue

            # --- IF NO HITS, run normal TSL logic ---
            p.high_price_since_entry = max(p.high_price_since_entry, ltp)

            # --- FULL TRAILING STOP LOGIC (from main1.py) ---
            with self.trader.lock:
                profit_points = ltp - p.entry_price
                current_rr = profit_points / p.initial_risk_points if p.initial_risk_points > 0 else 0
                highest_sl_floor = p.initial_sl_price

                trailing_stages = self.trade_mgmt_config.get("trailing_stop_stages", [])
                for stage in trailing_stages:
                    if current_rr >= stage['rr_target']:
                        new_floor = p.entry_price + (p.initial_risk_points * stage['trail_behind_rr'])
                        highest_sl_floor = max(highest_sl_floor, new_floor)

                final_new_sl = highest_sl_floor

                if not p.trailing_sl_armed and profit_points >= p.initial_risk_points * self.trading_config['trailing_sl_activation_rr']:
                    p.trailing_sl_armed = True
                    L.info(f"[{now}] Chandelier TSL armed for {p.tradingsymbol}")

                if p.trailing_sl_armed:
                    if calculated_chandelier_sl := self._calculate_trailing_stop(p):
                        final_new_sl = max(final_new_sl, calculated_chandelier_sl)

                if final_new_sl > p.sl_price:
                    if final_new_sl >= p.entry_price and p.tp_price > 0:
                        L.info(f"[{now}] TSL for {p.tradingsymbol} is profitable, cancelling static TP.")
                        p.tp_price = 0 # 0 means no TP
                        
                    self.trader.modify_sl(p, final_new_sl)

            self.store_actor.upsert_position(p)

# ==================================================================================================
# --- 4. FAKE STORE & CLOCK (Dependencies) ---
# ==================================================================================================

class SimulatedStore:
    """A fake, in-memory store that mimics the StoreActor's interface."""
    def __init__(self):
        self.positions = {}
        self.closed_trades = []
        self.kv = {}
        L.info("SimulatedStore (in-memory) initialized.")

    def upsert_position(self, pos: Position): self.positions[pos.id] = pos
    def log_closed_trade(self, pos: Position, price: float, reason: str):
        self.closed_trades.append(pos); self.positions.pop(pos.id, None)
    def log_strategy_performance(self, name: str, pnl: float): pass
    def get_strategy_performance(self, lookback_days: int) -> pd.DataFrame:
        return pd.DataFrame(columns=["strategy_name", "pnl"])
    def load_open_positions(self) -> dict[str, Position]: return {}
    def get_todays_trades_stats(self) -> tuple[int, int]: return 0, 0
    def set_kv(self, key: str, value: str): self.kv[key] = value
    def get_kv(self, key: str, default: str | None = None) -> str | None:
        return self.kv.get(key, default)

class BacktestClock(Clock):
    """A Clock that gets its time from the BacktestEngine."""
    def __init__(self): self._now = datetime.now()
    def set_time(self, new_time: datetime): self._now = new_time
    def now(self) -> datetime: return self._now

# ==================================================================================================
# --- 5. THE BACKTEST ENGINE (The "Runner") ---
# ==================================================================================================

class BacktestEngine:
    def __init__(self, config_dict: dict, historical_data_path: str): # <-- MODIFIED
        L.info("Initializing BacktestEngine...")
        self.config = config_dict # <-- MODIFIED
        self.historical_data = self._load_data(historical_data_path)
        
        # --- Create FAKE Components ---
        self.clock = BacktestClock()
        self.store = SimulatedStore()
        
        global now_ist
        now_ist = self.clock.now
        
        # --- Create REAL Components ---
        self.book = self._load_real_book()
        
        # SimulatedPriceBus needs the book and engine
        self.prices = SimulatedPriceBus(self.book, self)
        
        # The Engine is the "coordinator"
        self.engine = Engine(None, self.book, self.prices, self.config)
        
        self.trader = SimulatedTrader(self, self.book, self.prices, self.store, self.config)
        self.risk_manager = RiskManager(self.engine, self.trader, self.book, self.prices, self.store, self.config)
        self.micro_monitor = MicrostructureMonitor(self.prices, self.config)
        self.pos_manager = BacktestPositionManager(self.engine, self.trader, self.book, self.prices, self.store, self.risk_manager, self.config)

        # --- Link all components together ---
        self.engine.set_dependencies(self.trader, self.risk_manager, self.micro_monitor, self.pos_manager)
        
        # Set the (real) BarStore on the (real) Engine
        self.engine.bars = BarStore(timeframes=[1, 3, 5, 15])
        
        # Set the (real) classifier on the (real) Engine
        self.engine.nifty_token = 256265
        self.engine.bn_token = 260105
        self.engine.vix_token = 257281
        self.engine.classifier = RegimeClassifier(self.engine, 256265, 260105, 257281, self.config["strategies"]["regime_classifier"])
        
        L.info("BacktestEngine initialized successfully.")
        
    # In backtester.py, add this new function inside the BacktestEngine class

    def _generate_synthetic_microstructure(self, bar: pd.Series):
        """
        Generates plausible OBI/TFI scores based on the underlying bar
        and injects them into the MicrostructureMonitor.
        """
        for token, prefix in [(256265, "NIFTY"), (260105, "BN")]:
            if f"{prefix}_close" not in bar.index:
                continue

            bar_close = bar[f"{prefix}_close"]
            bar_open = bar[f"{prefix}_open"]
            bar_high = bar[f"{prefix}_high"]
            bar_low = bar[f"{prefix}_low"]

            if bar_close == bar_open:
                continue # Do-nothing on a doji bar

            # 1. Calculate bar "strength" (e.g., -1.0 to +1.0)
            # This is a simple % of bar that was a green/red candle
            bar_range = bar_high - bar_low
            if bar_range == 0: continue
            
            body = bar_close - bar_open
            strength = (body / bar_range) * 2.0 # Scale to roughly -2.0 to +2.0
            strength = max(-1.0, min(1.0, strength)) # Clamp to -1.0 to +1.0

            # 2. Get all option contracts for this underlying
            underlying_name = _get_underlying(self.book.get_symbol(token))
            expiry = self.book.find_nearest_expiry_date(underlying_name)
            if not expiry: continue
            
            chain = self.book.get_option_chain(underlying_name, expiry)
            if chain.empty: continue
            
            # 3. Inject synthetic data for ALL options in the chain
            for opt_token in chain['instrument_token']:
                # --- Fake TFI Score ---
                # A strong up-bar (strength=1.0) creates a TFI score of +350
                # A strong down-bar (strength=-1.0) creates a TFI score of -350
                fake_tfi_score = strength * 350 # 350 is just below the 400 threshold
                
                # Add some randomness so it's not perfect
                fake_tfi_score += np.random.normal(0, 50) 
                
                # Inject this fake score into the *real* monitor
                if opt_token not in self.micro_monitor.tfi_score_history:
                    self.micro_monitor.tfi_score_history[opt_token] = deque(maxlen=self.micro_monitor.persistence_window)
                self.micro_monitor.tfi_score_history[opt_token].append(fake_tfi_score)


                # --- Fake OBI Score ---
                # A strong up-bar (strength=1.0) creates OBI of 65% (0.65)
                # A strong down-bar (strength=-1.0) creates OBI of 35% (0.35)
                # 0.5 is the neutral mid-point
                fake_obi_ratio = 0.5 + (strength * 0.15) # 0.15 maps to the 65/35 range
                
                # Add some randomness
                fake_obi_ratio += np.random.normal(0, 0.05)
                fake_obi_ratio = max(0.01, min(0.99, fake_obi_ratio)) # Clamp
                
                # Inject this fake score
                if opt_token not in self.micro_monitor.obi_ratio_history:
                    self.micro_monitor.obi_ratio_history[opt_token] = deque(maxlen=self.micro_monitor.persistence_window)
                self.micro_monitor.obi_ratio_history[opt_token].append(fake_obi_ratio)
        
    def _load_real_book(self) -> InstrumentBook:
        """Loads the real InstrumentBook from the cached CSV."""
        L.info("Loading REAL InstrumentBook...")
        instrument_file = os.path.join(PERSIST_DIR, "instruments_nfo.csv")
        if not os.path.exists(instrument_file):
            L.critical(f"CRITICAL: `instruments_nfo.csv` not found in `{PERSIST_DIR}`.")
            L.critical("Please run `main1.py` once in live mode to generate this file.")
            exit(1)
            
        try:
            book = InstrumentBook(store_actor=None, order_actor=None)
            book.df = pd.read_csv(instrument_file, parse_dates=["expiry"])
            book.df['expiry'] = pd.to_datetime(book.df['expiry']).dt.tz_localize(IST) # Ensure TZ
            book.df_by_token = book.df.set_index('instrument_token')
            book.df_by_symbol = book.df.set_index('tradingsymbol')
            L.info(f"Real InstrumentBook loaded with {len(book.df)} instruments.")
            return book
        except Exception as e:
            L.critical(f"Failed to load real InstrumentBook: {e}", exc_info=True)
            exit(1)


    def _load_data(self, path: str) -> pd.DataFrame:
        """
        Loads and prepares the historical data.
        NOTE: This NO LONGER creates fake OPT_ columns.
        """
        L.info(f"Loading historical data from {path}...")
        try:
            df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
            df = df.tz_localize('UTC').tz_convert('Asia/Kolkata') # Ensure IST
            
            # Remove any fake columns if they exist
            df = df.loc[:, ~df.columns.str.startswith('OPT_')]
            
            L.info(f"Data loaded. {len(df)} bars from {df.index[0]} to {df.index[-1]}")
            return df
        except Exception as e:
            L.critical(f"Failed to load data: {e}")
            exit(1)

    def _prime_barstore(self):
        """
        PRE-LOADS THE ENTIRE DATASET for futures/VIX
        to ensure indicators are fully "warmed up".
        """
        L.info(f"Pre-loading BarStore with {len(self.historical_data)} FUTURES bars...")
        
        # NIFTY
        if 'NIFTY_open' in self.historical_data.columns:
            prime_data_nifty = self.historical_data[['NIFTY_open', 'NIFTY_high', 'NIFTY_low', 'NIFTY_close', 'NIFTY_volume']].copy()
            prime_data_nifty.columns = ['open', 'high', 'low', 'close', 'volume']
            prime_data_nifty['date'] = prime_data_nifty.index
            self.engine.bars.prime(256265, prime_data_nifty) # NIFTY
        
        # BANKNIFTY
        if 'BN_open' in self.historical_data.columns:
            prime_data_bn = self.historical_data[['BN_open', 'BN_high', 'BN_low', 'BN_close', 'BN_volume']].copy()
            prime_data_bn.columns = ['open', 'high', 'low', 'close', 'volume']
            prime_data_bn['date'] = prime_data_bn.index
            self.engine.bars.prime(260105, prime_data_bn) # BN

        # VIX
        if 'VIX_open' in self.historical_data.columns:
            prime_data_vix = self.historical_data[['VIX_open', 'VIX_high', 'VIX_low', 'VIX_close', 'VIX_volume']].copy()
            prime_data_vix.columns = ['open', 'high', 'low', 'close', 'volume']
            prime_data_vix['date'] = prime_data_vix.index
            self.engine.bars.prime(257281, prime_data_vix) # VIX
        
        L.info("BarStore (Futures/VIX) pre-loading complete.")


    def run_backtest(self):
        L.info("--- STARTING HIGH-FIDELITY BACKTEST RUN ---")
        
        self._prime_barstore()
        
        warmup_period = 200 # Must match your longest indicator
        L.info(f"Skipping first {warmup_period} bars for indicator warmup...")

        for i, bar in enumerate(self.historical_data.iloc[warmup_period:].itertuples()):
            try:
                # 1. Update "Time" and "Senses" (Futures only)
                self.clock.set_time(bar.Index)
                self.prices.update_futures_bar(bar)
                
                # 2. Update BarStore (Futures only, for indicators)
                for token, prefix in [(256265, "NIFTY"), (260105, "BN")]:
                     if f"{prefix}_close" in bar.index:
                         vol = bar[f"{prefix}_volume"]
                         self.engine.bars.add_tick({
                             "instrument_token": token,
                             "exchange_timestamp": bar.Index,
                             "last_price": bar[f"{prefix}_close"],
                             "last_traded_quantity": vol / 10 if vol > 10 else 1,
                         })

                # 3. Run "Brain" - Phase 1 (Regime & IV)
                self.engine.run_regime_classification(bar.Index)
                self.engine._update_atm_iv_cache() # CRITICAL: Update IV
                
                self._generate_synthetic_microstructure(bar)
                
                # 4. Run "Brain" - Phase 2 (Planner)
                # This runs the REAL planner. It will call the
                # SimulatedPriceBus to get on-the-fly option prices.
                self.engine._run_strategic_planner()
                
                # 5. Update Option Bars for TSL
                # This feeds the *estimated* option OHLC to the BarStore
                # so the PositionManager can calculate TSL.
                self.pos_manager.update_option_bars_for_open_positions()
                
                # 6. Manage Open Trades (Check SL/TP)
                self.pos_manager.manage_positions_backtest(bar.Index)
                
                # 7. Execute Queued Trades (Simulate Fills)
                while not self.engine.trade_signal_queue.empty():
                    try:
                        trade_params = self.engine.trade_signal_queue.get_nowait()
                        self.trader.open_position(trade_params)
                    except Empty:
                        break
                
                # 8. Update PnL for RiskManager
                self.risk_manager.last_unrealized_pnl = self.trader.unrealized_pnl()

            except Exception as e:
                L.error(f"Error on bar {bar.Index}: {e}", exc_info=True)
                
            if i % 1000 == 0 and i > 0:
                L.info(f"Processed {i} bars... Current PnL: {self.trader.daily_realized_pnl:.2f}")

        L.info("--- BACKTEST RUN COMPLETE ---")
        self.generate_report()

    def generate_report(self, return_metrics: bool = False): # <-- MODIFIED
        """Prints a final PnL and trade summary."""
        L.info("--- Backtest Report ---")
        
        if not self.trader.closed_trades_log:
            L.warning("No trades were executed.")
            if return_metrics: # <-- ADDED
                return {"total_pnl": 0, "total_trades": 0, "win_rate": 0, "profit_factor": 0}
            return

        df = pd.DataFrame(self.trader.closed_trades_log)
        total_pnl = df['pnl'].sum()
        total_trades = len(df)
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        
        total_loss = losses['pnl'].sum()
        total_win = wins['pnl'].sum()
        profit_factor = abs(total_win / total_loss) if total_loss != 0 else float('inf')
        
        print("\n" + "="*30)
        print("PERFORMANCE SUMMARY")
        print("="*30)
        print(f"Total Net PnL:     {total_pnl:,.2f}")
        print(f"Total Trades:      {total_trades}")
        print(f"Win Rate:          {win_rate:.2f}%")
        print(f"Profit Factor:     {profit_factor:.2f}")
        print(f"Average Win:       {avg_win:,.2f}")
        print(f"Average Loss:      {avg_loss:,.2f}")
        
        print("\n" + "="*30)
        print("PNL BY STRATEGY")
        print("="*30)
        print(df.groupby('strategy')['pnl'].sum())
        
        print("\n" + "="*30)
        print("PNL BY REGIME")
        print("="*30)
        print(df.groupby('regime')['pnl'].sum())
        
        if return_metrics:
            return {
                "total_pnl": total_pnl,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor
            }
        
        # Plotting (optional, but very useful)
        try:
            import matplotlib.pyplot as plt
            df.set_index('exit_time')['pnl'].cumsum().plot(title='Cumulative PnL Over Time', grid=True)
            plt.show()
        except ImportError:
            L.info("Install `matplotlib` to see a PnL equity curve.")

# ==================================================================================================
# --- 6. MAIN EXECUTION SCRIPT ---
# ==================================================================================================

if __name__ == "__main__":
    CONFIG_FILE = "config.json"
    
    # --- !!! YOU MUST CREATE THIS FILE !!! ---
    # This CSV needs to have (at minimum):
    # timestamp,NIFTY_open,NIFTY_high,NIFTY_low,NIFTY_close,NIFTY_volume,BN_open,...
    HISTORICAL_DATA_FILE = "historical_data.csv"
    
    # You will also need the 'instruments_nfo.csv' file in your 'PERSIST_DIR'
    
    L.info("Starting High-Fidelity Backtester...")
    
    try:
        backtester = BacktestEngine(
            config_path=CONFIG_FILE,
            historical_data_path=HISTORICAL_DATA_FILE
        )
        backtester.run_backtest()
        
    except FileNotFoundError as e:
        L.critical(f"CRITICAL: Could not find file: {e.filename}")
        L.critical(f"Please create {CONFIG_FILE}, {HISTORICAL_DATA_FILE}, and run `main1.py` to create `instruments_nfo.csv`")
    except Exception as e:
        L.critical(f"An unhandled error occurred: {e}", exc_info=True)