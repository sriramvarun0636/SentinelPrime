from __future__ import annotations

import random
import os
import sys
import time
import math
import json
import queue
import signal
import logging
import threading
import uuid
import sqlite3
import stat
from enum import Enum, auto
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import List, Dict, Optional, Tuple, Callable, Type
from datetime import datetime, timedelta, time as dtime, date
from abc import ABC, abstractmethod
from collections import deque


from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pytz
import pandas_market_calendars as mcal

# --- Dependency Checks ---
try:
    import pandas_ta as ta
except ImportError:
    raise RuntimeError("pandas-ta library not found. Please install it: pip install pandas-ta")
try:
    from kiteconnect import KiteConnect, KiteTicker
    from kiteconnect.exceptions import KiteException
except ImportError:
    raise RuntimeError("KiteConnect library not found. Please install it: pip install kiteconnect")
try:
    import requests
except ImportError:
    requests = None
try:
    from flask import Flask
    from prometheus_flask_exporter import PrometheusMetrics
    from prometheus_client import Gauge, REGISTRY, PROCESS_COLLECTOR, PLATFORM_COLLECTOR
    from waitress import serve
except ImportError:
    print("WARNING: Flask/Prometheus dependencies not found. Metrics server will be disabled.", file=sys.stderr)
    Flask = None
    Gauge = None
    REGISTRY = None
    PROCESS_COLLECTOR = None
    PLATFORM_COLLECTOR = None
    serve = None
try:
    from scipy.optimize import newton
    from scipy.stats import norm
except ImportError:
    raise RuntimeError("scipy library not found. Please install it: pip install scipy")

# ==================================================================================================
# ACTOR MODEL - FOR CONCURRENCY SAFETY
# ==================================================================================================
L_ACTORS = logging.getLogger("SENTINEL-PRIME.ACTORS")

class OrderActor(threading.Thread):
    """A single-threaded actor for all broker API I/O."""
    def __init__(self, kite_client: GovernedKite):
        super().__init__(daemon=True, name="OrderActor")
        self.q = queue.Queue()
        self.kite = kite_client
        self.running = True

    def run(self):
        while self.running:
            try:
                msg = self.q.get(timeout=1.0)
                if msg is None:
                    continue

                L_ACTORS.debug(f"OrderActor processing: {msg.get('type')}")
                msg_type = msg['type']
                reply_q = msg.get('reply_q') # Reply queue is optional

                try:
                    res = None
                    # --- Write/Trade Methods ---
                    if msg_type == "place_order":
                        res = self.kite.place_order(**msg['params'])
                    elif msg_type == "modify_order":
                        res = self.kite.modify_order(**msg['params'])
                    elif msg_type == "cancel_order":
                        res = self.kite.cancel_order(**msg['params'])
                    
                    # --- Read/Data Methods ---
                    elif msg_type == "order_history":
                        res = self.kite.order_history(**msg['params'])
                    
                    # --- NEWLY ADDED METHODS ---
                    elif msg_type == "orders":
                        res = self.kite.orders()
                    elif msg_type == "positions":
                        res = self.kite.positions()
                    elif msg_type == "margins":
                        res = self.kite.margins()
                    elif msg_type == "instruments":
                        res = self.kite.instruments(**msg['params'])
                    elif msg_type == "historical_data":
                        res = self.kite.historical_data(**msg['params'])
                    elif msg_type == "quote":
                        res = self.kite.quote(**msg['params'])
                    # --- END NEW METHODS ---

                    else:
                        L_ACTORS.error(f"OrderActor unknown message type: {msg_type}")

                    if reply_q:
                        reply_q.put({"ok": True, "res": res})

                except Exception as e:
                    L_ACTORS.error(f"OrderActor error on {msg_type}: {e}", exc_info=False)
                    if reply_q:
                        reply_q.put({"ok": False, "error": e})
                
                self.q.task_done()

            except queue.Empty:
                continue

    def stop(self):
        self.running = False
        self.q.put(None) # Unblock .get()

class StoreActor(threading.Thread):
    """A single-threaded actor for all database I/O."""
    def __init__(self, store: Store):
        super().__init__(daemon=True, name="StoreActor")
        self.q = queue.Queue()
        self.store = store # The *only* thread that can use this object
        self.running = True

    def run(self):
        while self.running:
            try:
                msg = self.q.get(timeout=1.0)
                if msg is None:
                    continue

                L_ACTORS.debug(f"StoreActor processing: {msg.get('type')}")
                msg_type = msg['type']
                reply_q = msg.get('reply_q') # Reply is optional

                try:
                    res = None
                    # --- Write Methods ---
                    if msg_type == "upsert_position":
                        self.store.upsert_position(msg['pos'])
                    elif msg_type == "log_closed_trade":
                        self.store.log_closed_trade(msg['pos'], msg['price'], msg['reason'])
                    elif msg_type == "log_strategy_performance":
                        self.store.log_strategy_performance(msg['name'], msg['pnl'])
                    elif msg_type == "set_kv":
                        self.store.set_kv(msg['key'], msg['value'])
                    
                    # --- Read Methods ---
                    elif msg_type == "get_kv":
                        res = self.store.get_kv(msg['key'], msg.get('default'))
                    elif msg_type == "load_open_positions":
                        res = self.store.load_open_positions()
                    
                    # --- NEWLY ADDED METHODS ---
                    elif msg_type == "get_strategy_performance":
                        res = self.store.get_strategy_performance(msg['lookback_days'])
                    elif msg_type == "get_todays_trades_stats":
                        res = self.store.get_todays_trades_stats()
                    # --- END NEW METHODS ---
                    
                    else:
                        L_ACTORS.error(f"StoreActor unknown message type: {msg_type}")

                    if reply_q:
                        reply_q.put({"ok": True, "res": res})

                except Exception as e:
                    L_ACTORS.error(f"StoreActor error on {msg_type}: {e}", exc_info=True)
                    if reply_q:
                        reply_q.put({"ok": False, "error": e})

                self.q.task_done()

            except queue.Empty:
                continue
    
    def stop(self):
        self.running = False
        self.q.put(None) # Unblock .get()

class Clock(ABC):
    @abstractmethod
    def now(self) -> datetime:
        pass

class RealTimeClock(Clock):
    """The clock for live trading."""
    def now(self) -> datetime:
        return datetime.now(tz=IST)

# ==================================================================================================
# CONFIGURATION LOADER
# ==================================================================================================
def load_config(path: str = "config.json") -> Dict:
    """Loads and validates the configuration from a JSON file."""
    logging.info(f"Loading configuration from: {path}")
    try:
        with open(path, 'r') as f:
            config = json.load(f)

        timing_keys = ["market_open", "market_settling_time", "final_entry_time", "eod_flatten_time", "market_close", "final_expiry_entry_time"]
        for key in timing_keys:
            if key in config["timings"]:
                try:
                    config["timings"][key] = dtime.fromisoformat(config["timings"][key])
                except (ValueError, TypeError):
                    logging.warning(f"Invalid or missing time format for '{key}'. Using default or ignoring.")

        required_keys = ["trading", "timings", "strategies", "technical"]
        if not all(key in config for key in required_keys):
            raise ValueError("Config file is missing one of the main keys: trading, timings, strategies, technical")

        return config
    except FileNotFoundError:
        raise SystemExit(f"FATAL: Configuration file '{path}' not found.")
    except (json.JSONDecodeError, ValueError) as e:
        raise SystemExit(f"FATAL: Error in configuration file '{path}': {e}")
    except Exception as e:
        raise SystemExit(f"FATAL: Unhandled error loading configuration: {e}")

# ==================================================================================================
# GLOBAL CONSTANTS & LOGGING SETUP
# ==================================================================================================
IST = pytz.timezone("Asia/Kolkata")
PERSIST_DIR = os.environ.get("PERSIST_DIR", "./persist_sentinel_prime")
DATA_LOG_DIR = os.path.join(PERSIST_DIR, "daily_data_logs")
DB_PATH = os.path.join(PERSIST_DIR, "state_sentinel_prime.db")
TOKEN_FILE_PATH = os.path.join(PERSIST_DIR, "kite_token.json")
LOG_FILE = os.environ.get("LOG_FILE", "sentinel_prime_bot.log")
KILL_SWITCH_FILE = os.path.join(PERSIST_DIR, "HALT.txt")
os.makedirs(PERSIST_DIR, exist_ok=True)

try:
    os.chmod(PERSIST_DIR, 0o700)
    if os.path.exists(DB_PATH):
        os.chmod(DB_PATH, 0o600)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(threadName)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)]
)
L = logging.getLogger("SENTINEL-PRIME")

# --- Prometheus Metrics Server Setup ---
if Gauge:
    try:
        REGISTRY.unregister(PROCESS_COLLECTOR)
        REGISTRY.unregister(PLATFORM_COLLECTOR)
        if 'python_gc_objects_collected_total' in REGISTRY._collector_registry:
            REGISTRY.unregister(REGISTRY._collector_registry['python_gc_objects_collected_total'])
    except Exception as e:
        L.warning(f"Could not unregister default Prometheus collectors: {e}")

    G_PNL_REALIZED = Gauge("sentinel_pnl_realized_rupees", "Daily realized PnL")
    G_PNL_UNREALIZED = Gauge("sentinel_pnl_unrealized_rupees", "Current unrealized PnL of open positions")
    G_DAILY_DRAWDOWN_PCT = Gauge("sentinel_daily_drawdown_pct", "Current daily drawdown from the high-water mark")
    G_HALTED_STATUS = Gauge("sentinel_trading_halted_status", "Trading status (1 = Halted, 0 = Active)")
    G_WS_CONNECTED = Gauge("sentinel_websocket_connected_status", "PriceBus WebSocket connection status (1 = Connected, 0 = Disconnected)")
    G_LAST_TICK_AGE_SECONDS = Gauge("sentinel_last_tick_age_seconds", "Age of the last received tick from the PriceBus in seconds")
    G_CURRENT_REGIME = Gauge("sentinel_market_regime_enum", "Current market regime as an Enum integer", labelnames=["regime_name"])
    G_PORTFOLIO_DELTA = Gauge("sentinel_portfolio_net_delta", "Net Delta exposure of the entire portfolio")
    G_PORTFOLIO_VEGA = Gauge("sentinel_portfolio_net_vega", "Net Vega exposure of the entire portfolio")
    G_PORTFOLIO_GAMMA = Gauge("sentinel_portfolio_net_gamma", "Net Gamma exposure of the entire portfolio")
    G_PORTFOLIO_THETA = Gauge("sentinel_portfolio_net_theta", "Net Theta exposure of the entire portfolio")
    METRICS_APP = Flask(__name__)
    metrics = PrometheusMetrics(METRICS_APP, export_defaults=False)
else:
    G_PNL_REALIZED = G_PNL_UNREALIZED = G_DAILY_DRAWDOWN_PCT = G_HALTED_STATUS = G_WS_CONNECTED = G_LAST_TICK_AGE_SECONDS = G_CURRENT_REGIME = None
    G_PORTFOLIO_DELTA = G_PORTFOLIO_VEGA = G_PORTFOLIO_GAMMA = G_PORTFOLIO_THETA = None
    METRICS_APP = None


def start_metrics_server(port: int = 9095):
    if not serve or not METRICS_APP:
        L.warning("Flask/Waitress not found. Prometheus metrics server is disabled.")
        return

    def run_server():
        L.info(f"Starting Prometheus metrics server on http://0.0.0.0:{port}/metrics")
        serve(METRICS_APP, host='0.0.0.0', port=port, threads=4)

    metrics_thread = threading.Thread(target=run_server, name="MetricsServer", daemon=True)
    metrics_thread.start()


# --- Global Constants (to be populated in main) ---
APP_ENV = os.environ.get("APP_ENV", "STAGING").upper()
PAPER_TRADING = True
TG_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
INDEX_TRADING_SYMBOLS = {"BANKNIFTY": "NIFTY BANK", "NIFTY": "NIFTY 50", "INDIA VIX": "INDIA VIX"}


class Regime(Enum):
    COMPRESSION = auto()
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    CHOP = auto()
    CHAOS = auto()
    UNCLEAR = auto()


class StrategyName(Enum):
    MOMENTUM_BREAKOUT = "MomentumBreakout"
    TREND_PULLBACK = "TrendPullback"
    MEAN_REVERSION = "MeanReversion"
    VOLATILITY_MEAN_REVERSION = "VolatilityMeanReversion"
    OPENING_RANGE_BREAKOUT = "OpeningRangeBreakout"


class PositionStatus(Enum):
    PENDING_SUBMISSION = "PENDING_SUBMISSION"
    REJECTED = "REJECTED"
    PENDING_ENTRY = "PENDING_ENTRY"
    OPEN_AWAITING_BRACKETS = "OPEN_AWAITING_BRACKETS"
    ACTIVE = "ACTIVE"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    PENDING_CLOSURE = "PENDING_CLOSURE"
    PENDING_SL_EXIT = "PENDING_SL_EXIT"
    CLOSED = "CLOSED"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OptionType(Enum):
    CE = "CE"
    PE = "PE"


# ==================================================================================================
# CORE DATA STRUCTURES
# ==================================================================================================
@dataclass
class ScaleOut:
    rr_target: float
    pct_to_close: int


@dataclass
class Position:
    id: str
    tradingsymbol: str
    token: int
    option_type: str
    qty: int
    initial_qty: int
    entry_price: float
    initial_sl_price: float
    sl_price: float
    tp_price: float
    opened_at: datetime
    strategy: str
    market_regime_at_entry: str
    underlying_sl_level: Optional[float] = None
    status: str = PositionStatus.PENDING_SUBMISSION.value
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    partial_exit_order_ids: List[str] = field(default_factory=list)
    exit_reason: Optional[str] = None
    slm_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    scaled_out_qty: int = 0
    trailing_sl_armed: bool = False
    initial_risk_points: float = 0.0
    option_sl_points: float = 0.0
    option_tp_points: float = 0.0
    high_price_since_entry: float = 0.0
    exit_price: Optional[float] = None
    scale_out_rules: List[Dict] = field(default_factory=list)
    triggered_scale_out_targets: List[float] = field(default_factory=list)
    greeks: Dict[str, float] = field(default_factory=dict)
    max_trade_duration_minutes: int = 90
    oi_profit_target: Optional[float] = None
    intended_risk_rupees: float = 0.0
    is_entry_order_open: bool = False
    entry_stage: int = 0
    last_entry_modification: Optional[datetime] = None

    def __post_init__(self):
        """Sanitizes numeric types to prevent silent crashes from NumPy types."""
        numeric_fields = [
            'qty', 'initial_qty', 'entry_price', 'initial_sl_price',
            'sl_price', 'tp_price', 'underlying_sl_level', 'scaled_out_qty',
            'initial_risk_points', 'option_sl_points', 'option_tp_points',
            'high_price_since_entry', 'exit_price', 'oi_profit_target',
            'intended_risk_rupees'
        ]
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if value is None:
                continue

            if isinstance(value, (int, np.integer)):
                setattr(self, field_name, int(value))
            elif isinstance(value, (float, np.floating)):
                setattr(self, field_name, float(value))


@dataclass
class TradeSignal:
    strategy_name: StrategyName
    side: OrderSide
    risk_points: float
    reward_points: float
    vega: float = 0.0


# ==================================================================================================
# UTILITIES & GREEK CALCULATIONS
# ==================================================================================================
def now_ist() -> datetime:
    return datetime.now(tz=IST)


def send_alert(text: str, level: str = "info"):
    log_func = getattr(L, level, L.info)
    log_func(text)
    if not all([TG_BOT_TOKEN, TG_CHAT_ID, requests]):
        return
    for attempt in range(2):
        try:
            requests.post(
                f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                json={"chat_id": TG_CHAT_ID, "text": text[:4000]},
                timeout=5
            )
            return
        except Exception as e:
            L.warning(f"Telegram alert failed on attempt {attempt+1}: {e}")
            time.sleep(1)


def _calculate_time_to_expiry(expiry_date: date, current_datetime: datetime, market_close_time: dtime) -> float:
    """Calculates time to expiry in years based on calendar time."""
    expiry_dt = datetime.combine(expiry_date, market_close_time, tzinfo=IST)
    if expiry_dt < current_datetime:
        return 1e-9

    time_delta_seconds = (expiry_dt - current_datetime).total_seconds()
    seconds_in_year = 365.25 * 24 * 60 * 60
    T = time_delta_seconds / seconds_in_year
    return max(1e-9, T)
# --- ADD THIS NEW FUNCTION TO main1.py ---

def calculate_trading_time_to_expiry(
    current_datetime: datetime, 
    expiry_date: date, 
    market_open_time: dtime, 
    market_close_time: dtime,
    nse_calendar
) -> float:
    """
    Calculates time to expiry (T) in fractional years based on TRADING time.
    T = (Trading Days Remaining) / 252.0
    For intraday expiry, it calculates the fraction of the trading day left.
    """
    
    # --- Constants ---
    # Total trading minutes in a standard session (e.g., 9:15 to 15:30 = 375 mins)
    total_session_minutes = (
        (market_close_time.hour * 60 + market_close_time.minute) -
        (market_open_time.hour * 60 + market_open_time.minute)
    )
    if total_session_minutes <= 0:
        total_session_minutes = 375 # Failsafe
        
    TRADING_DAYS_PER_YEAR = 252.0
    MINUTES_PER_TRADING_YEAR = TRADING_DAYS_PER_YEAR * total_session_minutes

    current_date = current_datetime.date()
    current_time = current_datetime.time()

    # --- Case 1: Already Past Expiry ---
    # If it's expiry day and past market close
    if current_date > expiry_date or (current_date == expiry_date and current_time >= market_close_time):
        return 1e-9 # Return a tiny non-zero number to avoid division by zero

    # --- Case 2: On Expiry Day (Intraday Calculation) ---
    if current_date == expiry_date:
        if current_time < market_open_time:
            # Pre-market on expiry day, counts as 1 full day
            minutes_remaining = total_session_minutes
        else:
            # Mid-session on expiry day
            minutes_remaining = (
                (market_close_time.hour * 60 + market_close_time.minute) -
                (current_time.hour * 60 + current_time.minute)
            )
        
        # Return the fraction of the *entire trading year* that is left today
        return max(1e-9, minutes_remaining / MINUTES_PER_TRADING_YEAR)

    # --- Case 3: Before Expiry Day (Interday Calculation) ---
    # Use the market calendar to find the number of trading days
    try:
        trading_days = nse_calendar.valid_days(
            start_date=current_date, 
            end_date=expiry_date
        )
        
        # .size includes the start day. We need to adjust.
        # If today is a trading day and market is open: count today + future days
        # If today is a trading day and market is closed: count only future days
        # If today is a holiday: count only future days
        
        is_today_trading_day = current_date in trading_days
        
        if is_today_trading_day and current_time < market_close_time:
            # Market is open, or pre-market. Today counts as a full day
            # (plus fractional part if we want to be hyper-accurate, but for
            # interday, this is good enough)
            trading_days_left = len(trading_days)
        else:
            # Market is closed, or it's a holiday. Today doesn't count.
            trading_days_left = len(trading_days)
            if is_today_trading_day:
                trading_days_left -= 1 # Don't count today
                
        return max(1e-9, trading_days_left / TRADING_DAYS_PER_YEAR)
        
    except Exception as e:
        L.warning(f"Failed to calculate trading days, falling back to calendar estimate: {e}")
        # Fallback to a (better) calendar estimate
        days_left = (expiry_date - current_date).days
        return max(1e-9, (days_left * (TRADING_DAYS_PER_YEAR / 365.25)) / TRADING_DAYS_PER_YEAR)


def _get_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[Optional[float], Optional[float]]:
    if T <= 1e-9 or sigma <= 1e-9 or S <= 0 or K <= 0:
        return None, None
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2
    except (ValueError, ZeroDivisionError):
        return None, None


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    d1, d2 = _get_d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return max(0.0, S - K) if is_call else max(0.0, K - S)

    price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2) if is_call else \
            K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    d1, _ = _get_d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return (1.0 if S > K else 0.0) if is_call else (-1.0 if S < K else 0.0)
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, _ = _get_d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return 0.0
    return S * norm.pdf(d1) * math.sqrt(T) / 100.0


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, _ = _get_d1_d2(S, K, T, r, sigma)
    if d1 is None or S <= 0 or sigma <= 0 or T <= 0:
        return 0.0
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    d1, d2 = _get_d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return 0.0

    p1 = - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    if is_call:
        p2 = - r * K * math.exp(-r * T) * norm.cdf(d2)
    else:  # Put
        p2 = r * K * math.exp(-r * T) * norm.cdf(-d2)

    return (p1 + p2) / 365.0


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> Dict[str, float]:
    """Calculates primary greeks for a given option."""
    return {
        "delta": bs_delta(S, K, T, r, sigma, is_call),
        "vega": bs_vega(S, K, T, r, sigma),
        "gamma": bs_gamma(S, K, T, r, sigma),
        "theta": bs_theta(S, K, T, r, sigma, is_call)
    }


def calculate_historical_volatility(price_series: pd.Series, window: int = 20, timeframe_minutes: int = 1) -> Optional[float]:
    """Calculates annualized historical volatility from intraday data."""
    if len(price_series) < window:
        return None

    trading_minutes_per_day = 375
    bars_per_day = trading_minutes_per_day / timeframe_minutes
    bars_per_year = bars_per_day * 252
    annualization_factor = math.sqrt(bars_per_year)

    log_returns = np.log(price_series / price_series.shift(1))
    std_dev = log_returns.rolling(window=window).std().iloc[-1]

    annualized_vol = std_dev * annualization_factor
    return annualized_vol if not pd.isna(annualized_vol) else None


def calculate_iv(target_price: float, S: float, K: float, T: float, r: float, is_call: bool, initial_guess: float = 0.5, hv_fallback: Optional[float] = None) -> float:
    """Calculates Implied Volatility using Newton-Raphson method."""
    def error_func(sigma, *args):
        price = black_scholes_price(S, K, T, r, sigma, is_call)
        return price - target_price

    try:
        iv = newton(error_func, initial_guess, tol=1e-5, maxiter=50)
        return iv if iv > 0 else (hv_fallback or initial_guess)
    except (RuntimeError, ValueError):
        L.warning(f"IV calculation failed for S={S}, K={K}, P={target_price}. Falling back.")
        return hv_fallback or initial_guess


def _get_underlying(tradingsymbol: str) -> str:
    upper_symbol = tradingsymbol.upper()
    if "BANKNIFTY" in upper_symbol:
        return "BANKNIFTY"
    if "NIFTY" in upper_symbol:
        return "NIFTY"
    return "UNKNOWN"


def calculate_dynamic_risk_params(
    ohlc_df_1m: pd.DataFrame,
    base_sl_multiplier: float,
    base_tp_multiplier: float
) -> Tuple[float, float]:
    """
    Calculates dynamic SL/TP multipliers based on short-term vs long-term volatility.
    """
    if len(ohlc_df_1m) < 50:
        L.warning("Not enough 1m data for dynamic risk calculation.")
        return 0.0, 0.0

    atr_short = ohlc_df_1m.ta.atr(length=5).iloc[-1]
    atr_long = ohlc_df_1m.ta.atr(length=50).iloc[-1]

    if atr_long == 0 or pd.isna(atr_short) or pd.isna(atr_long):
        L.warning("Invalid ATR values for dynamic risk calculation.")
        return 0.0, 0.0

    vol_ratio = atr_short / atr_long

    if vol_ratio > 1.5:  # Volatility expanding
        final_sl_multiplier = base_sl_multiplier * 1.25
        final_tp_multiplier = base_tp_multiplier * 0.75
    elif vol_ratio < 0.7:  # Volatility contracting
        final_sl_multiplier = base_sl_multiplier * 0.80
        final_tp_multiplier = base_tp_multiplier * 1.20
    else:  # Normal
        final_sl_multiplier = base_sl_multiplier
        final_tp_multiplier = base_tp_multiplier

    current_atr = ohlc_df_1m.ta.atr(14).iloc[-1]
    if pd.isna(current_atr):
        L.warning("Invalid 14-period ATR for dynamic risk calculation.")
        return 0.0, 0.0

    risk_points = current_atr * final_sl_multiplier
    reward_points = current_atr * final_tp_multiplier
    return risk_points, reward_points


class ApiGovernor:
    """A thread-safe API rate limiter and retry wrapper."""
    def __init__(self, rate_limit: int = 3, per_seconds: int = 1):
        self.rate_limit = rate_limit
        self.per_seconds = per_seconds
        self.call_log = []
        self.lock = threading.Lock()

    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            max_retries = 3
            delay = 1
            for attempt in range(max_retries):
                try:
                    with self.lock:
                        now = time.time()
                        self.call_log = [t for t in self.call_log if now - t < self.per_seconds]
                        if len(self.call_log) >= self.rate_limit:
                            sleep_time = self.per_seconds - (now - self.call_log[0])
                            if sleep_time > 0:
                                L.warning(f"API rate limit hit for {func.__name__}. Sleeping for {sleep_time:.2f}s.")
                                time.sleep(sleep_time)
                        self.call_log.append(time.time())
                    return func(*args, **kwargs)
                except KiteException as e:
                    L.warning(f"API call {func.__name__} failed (attempt {attempt+1}/{max_retries}): {e}")
                    if e.code in [503, 504]:
                        time.sleep(delay)
                        delay *= 2
                    else:
                        L.error(f"Non-retriable KiteException in {func.__name__}: {e}")
                        return None
                except Exception as e:
                    L.error(f"An unexpected error in governed call {func.__name__}: {e}", exc_info=True)
                    raise
            L.error(f"API call {func.__name__} failed after {max_retries} retries.")
            return None
        return wrapper


class GovernedKite:
    """A wrapper class that applies the ApiGovernor decorator to all KiteConnect API methods."""
    def __init__(self, kite_instance: KiteConnect):
        self._kite = kite_instance
        self.governor = ApiGovernor()

        methods_to_wrap = [
            "place_order", "modify_order", "cancel_order", "positions", "margins",
            "orders", "order_history", "historical_data", "quote",
            "profile", "instruments", "ltp"
        ]

        for method in methods_to_wrap:
            if hasattr(self._kite, method):
                setattr(self, method, self.governor(getattr(self._kite, method)))

        for attr in dir(self._kite):
            if attr.isupper() and not hasattr(self, attr):
                setattr(self, attr, getattr(self._kite, attr))

# ==================================================================================================
# PERSISTENCE LAYER
# ==================================================================================================
class Store:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self.lock = None
        self.lock = threading.Lock()

        try:
            if os.path.exists(self.path):
                os.chmod(self.path, 0o600)
            else:
                original_umask = os.umask(0o077)
                self._get_connection().close()
                os.umask(original_umask)
                L.info(f"Created new database at {self.path} with 600 permissions.")
        except Exception as e:
            L.warning(f"Could not set secure file permissions on {self.path}. This is normal on Windows. Error: {e}")

        self._initialize_db()

    def _get_connection(self):
        return sqlite3.connect(self.path, check_same_thread=False, timeout=10)

    def _initialize_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA user_version")
            db_version = cursor.fetchone()[0]
            L.info(f"Database schema version: {db_version}")

            if db_version < 8:  # --- MODIFIED: Target version is now 8 ---
                L.info("Running database migrations...")
                
                if db_version < 1:
                    cursor.execute("""
                    CREATE TABLE positions (
                        id TEXT PRIMARY KEY, tradingsymbol TEXT, token INTEGER, option_type TEXT, qty INTEGER, initial_qty INTEGER,
                        entry_price REAL, initial_sl_price REAL, sl_price REAL, tp_price REAL, opened_at TEXT,
                        strategy TEXT, market_regime_at_entry TEXT, underlying_sl_level REAL, status TEXT,
                        entry_order_id TEXT, slm_order_id TEXT, tp_order_id TEXT, scaled_out_qty INTEGER,
                        breakeven_armed INTEGER, trailing_sl_armed INTEGER, initial_risk_points REAL, 
                        option_sl_points REAL, option_tp_points REAL, high_price_since_entry REAL
                    )""")
                    cursor.execute("""
                        CREATE TABLE trade_log (
                            id TEXT PRIMARY KEY, tradingsymbol TEXT, strategy TEXT, entry_time TEXT, exit_time TEXT,
                            entry_price REAL, exit_price REAL, qty INTEGER, pnl REAL, exit_reason TEXT, market_regime_at_entry TEXT
                        )""")
                    cursor.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")

                if db_version < 2:
                    cursor.execute("ALTER TABLE positions ADD COLUMN exit_order_id TEXT")
                    cursor.execute("ALTER TABLE positions ADD COLUMN exit_reason TEXT")
                    cursor.execute("ALTER TABLE positions ADD COLUMN exit_price REAL")

                if db_version < 3:
                    cursor.execute("ALTER TABLE positions ADD COLUMN greeks TEXT DEFAULT '{}'")

                if db_version < 4:
                    cursor.execute("ALTER TABLE positions ADD COLUMN max_trade_duration_minutes INTEGER DEFAULT 90")
                    cursor.execute("ALTER TABLE positions ADD COLUMN oi_profit_target REAL")

                if db_version < 5:
                    cursor.execute("ALTER TABLE positions ADD COLUMN intended_risk_rupees REAL DEFAULT 0.0")
                    cursor.execute("ALTER TABLE positions ADD COLUMN is_entry_order_open INTEGER DEFAULT 0")
                    cursor.execute("ALTER TABLE positions ADD COLUMN scale_out_rules TEXT DEFAULT '[]'")
                    cursor.execute("ALTER TABLE positions ADD COLUMN triggered_scale_out_targets TEXT DEFAULT '[]'")

                if db_version < 6:
                    cursor.execute("ALTER TABLE positions ADD COLUMN entry_stage INTEGER DEFAULT 0")
                    cursor.execute("ALTER TABLE positions ADD COLUMN last_entry_modification TEXT")
                    L.info("DB Schema Migration: breakeven_armed column is now obsolete and will not be used.")

                if db_version < 7:
                    cursor.execute("ALTER TABLE positions ADD COLUMN partial_exit_order_ids TEXT DEFAULT '[]'")

                # --- NEW MIGRATION FOR VERSION 8 ---
                if db_version < 8:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS strategy_performance (
                            id TEXT PRIMARY KEY, 
                            strategy_name TEXT NOT NULL, 
                            close_time TEXT NOT NULL, 
                            pnl REAL NOT NULL
                        )""")
                    L.info("Database migration to v8 (strategy_performance) complete.")
                # --- END NEW MIGRATION ---

                # --- UPDATE FINAL VERSION ---
                cursor.execute("PRAGMA user_version = 8")
                L.info(f"Database migrations complete. Now at version {8}.")
            
            else:
                L.info(f"Database is up-to-date at version {db_version}.")
    
    # In Store class
    def log_strategy_performance(self, strategy_name: str, pnl: float):
        trade_id = f"strat_pnl_{uuid.uuid4()}"
        with self.lock, self._get_connection() as conn:
            conn.execute(
                """INSERT INTO strategy_performance (id, strategy_name, close_time, pnl)
                   VALUES (?, ?, ?, ?)""",
                (trade_id, strategy_name, now_ist().isoformat(), pnl)
            )
            conn.commit()

    def get_strategy_performance(self, lookback_days: int = 7) -> pd.DataFrame:
        cutoff_date = (now_ist() - timedelta(days=lookback_days)).isoformat()
        with self.lock, self._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT strategy_name, pnl FROM strategy_performance WHERE close_time >= ?",
                conn,
                params=(cutoff_date,)
            )
        return df

    def upsert_position(self, p: Position):
        sql = """
            INSERT INTO positions (id, tradingsymbol, token, option_type, qty, initial_qty, entry_price, initial_sl_price, sl_price, tp_price, opened_at, strategy, market_regime_at_entry, underlying_sl_level, status, entry_order_id, slm_order_id, tp_order_id, scaled_out_qty, trailing_sl_armed, initial_risk_points, option_sl_points, option_tp_points, high_price_since_entry, scale_out_rules, exit_order_id, exit_reason, exit_price, greeks, max_trade_duration_minutes, oi_profit_target, intended_risk_rupees, is_entry_order_open, triggered_scale_out_targets, entry_stage, last_entry_modification, partial_exit_order_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                qty=excluded.qty, sl_price=excluded.sl_price, tp_price=excluded.tp_price, status=excluded.status, slm_order_id=excluded.slm_order_id, tp_order_id=excluded.tp_order_id, scaled_out_qty=excluded.scaled_out_qty, trailing_sl_armed=excluded.trailing_sl_armed, high_price_since_entry=excluded.high_price_since_entry, entry_price=excluded.entry_price, opened_at=excluded.opened_at, entry_order_id=excluded.entry_order_id, exit_order_id=excluded.exit_order_id, exit_reason=excluded.exit_reason, exit_price=excluded.exit_price, greeks=excluded.greeks, max_trade_duration_minutes=excluded.max_trade_duration_minutes, oi_profit_target=excluded.oi_profit_target, intended_risk_rupees=excluded.intended_risk_rupees, is_entry_order_open=excluded.is_entry_order_open, triggered_scale_out_targets=excluded.triggered_scale_out_targets, entry_stage=excluded.entry_stage, last_entry_modification=excluded.last_entry_modification, partial_exit_order_ids=excluded.partial_exit_order_ids
        """
        rules_json = json.dumps(p.scale_out_rules)
        greeks_json = json.dumps(p.greeks)
        triggered_json = json.dumps(p.triggered_scale_out_targets)
        partial_exits_json = json.dumps(p.partial_exit_order_ids)
        last_mod_iso = p.last_entry_modification.isoformat() if p.last_entry_modification else None

        params = (p.id, p.tradingsymbol, p.token, p.option_type, p.qty, p.initial_qty, p.entry_price, p.initial_sl_price, p.sl_price, p.tp_price, p.opened_at.isoformat(), p.strategy, p.market_regime_at_entry, p.underlying_sl_level, p.status, p.entry_order_id, p.slm_order_id, p.tp_order_id, p.scaled_out_qty, int(p.trailing_sl_armed), p.initial_risk_points, p.option_sl_points, p.option_tp_points, p.high_price_since_entry, rules_json, p.exit_order_id, p.exit_reason, p.exit_price, greeks_json, p.max_trade_duration_minutes, p.oi_profit_target, p.intended_risk_rupees, int(p.is_entry_order_open), triggered_json, p.entry_stage, last_mod_iso, partial_exits_json)
        with self.lock, self._get_connection() as conn:
            conn.execute(sql, params)
            conn.commit()

    def log_closed_trade(self, p: Position, exit_price: float, reason: str):
        pnl = (exit_price - p.entry_price) * p.initial_qty
        with self.lock, self._get_connection() as conn:
            conn.execute(
                """INSERT INTO trade_log (id, tradingsymbol, strategy, entry_time, exit_time, entry_price, exit_price, qty, pnl, exit_reason, market_regime_at_entry)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (p.id, p.tradingsymbol, p.strategy, p.opened_at.isoformat(), now_ist().isoformat(), p.entry_price, exit_price, p.initial_qty, pnl, reason, p.market_regime_at_entry)
            )
            conn.commit()

    def load_open_positions(self) -> Dict[str, Position]:
        with self.lock, self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM positions WHERE status != 'CLOSED'")
            rows = cursor.fetchall()
        positions = {}
        for r_dict in [dict(row) for row in rows]:
            r_dict['scale_out_rules'] = json.loads(r_dict.get('scale_out_rules', '[]'))
            r_dict['greeks'] = json.loads(r_dict.get('greeks', '{}'))
            r_dict['triggered_scale_out_targets'] = json.loads(r_dict.get('triggered_scale_out_targets', '[]'))
            r_dict['partial_exit_order_ids'] = json.loads(r_dict.get('partial_exit_order_ids', '[]'))
            if 'breakeven_armed' in r_dict:
                del r_dict['breakeven_armed']
            r_dict['trailing_sl_armed'] = bool(r_dict.get("trailing_sl_armed"))
            r_dict['is_entry_order_open'] = bool(r_dict.get("is_entry_order_open"))
            r_dict['opened_at'] = datetime.fromisoformat(r_dict["opened_at"])
            if r_dict.get("last_entry_modification"):
                r_dict['last_entry_modification'] = datetime.fromisoformat(r_dict["last_entry_modification"])

            extra_keys = set(r_dict.keys()) - set(f.name for f in Position.__dataclass_fields__)
            for key in extra_keys:
                del r_dict[key]

            pos = Position(**r_dict)
            positions[pos.id] = pos
        return positions

    def get_todays_trades_stats(self) -> Tuple[int, int]:
        today_str = date.today().isoformat()
        wins, losses = 0, 0
        with self.lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pnl FROM trade_log WHERE entry_time LIKE ?", (f"{today_str}%",))
            rows = cursor.fetchall()
            for row in rows:
                if row[0] > 0:
                    wins += 1
                else:
                    losses += 1
        return wins, losses

    def set_kv(self, key: str, value: str):
        with self.lock, self._get_connection() as conn:
            conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
            conn.commit()

    def get_kv(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self.lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM meta WHERE key=?", (key,))
            row = cursor.fetchone()
        return row[0] if row else default

# ==================================================================================================
# DATA MANAGEMENT
# ==================================================================================================
class InstrumentBook:
    def __init__(self, store_actor: StoreActor, order_actor: OrderActor):
        self.store_actor = store_actor
        self.order_actor = order_actor
        self.df = None
        self.path = os.path.join(PERSIST_DIR, "instruments_nfo.csv")
        self.df_by_token = None
        self.df_by_symbol = None
        self.special_tokens: Dict[str, int] = {}
        self.lock = None
        self.lock = threading.RLock()

    def load(self):
        try:
            # --- FIXED: Use StoreActor for DB read ---
            # This is a blocking call, safe to do on startup
            L.debug("InstrumentBook checking refresh status with StoreActor...")
            reply_q = queue.Queue()
            self.store_actor.q.put({
                "type": "get_kv",
                "key": "instruments_refreshed",
                "default": None,
                "reply_q": reply_q
            })
            
            instruments_refreshed = None
            try:
                resp = reply_q.get(timeout=10.0) # Wait for actor's reply
                if resp['ok']:
                    instruments_refreshed = resp['res']
                else:
                    L.warning(f"StoreActor failed to get 'instruments_refreshed' key: {resp.get('error')}")
            except queue.Empty:
                L.error("Timeout waiting for StoreActor reply in InstrumentBook.load()")

            if not os.path.exists(self.path) or instruments_refreshed != str(date.today()):
                self.refresh() # This will also use actors
            else:
                L.info("Loading NFO instruments from local cache.")
                self.df = pd.read_csv(self.path, parse_dates=["expiry"])
        except Exception as e:
            L.warning(f"Could not load instrument file from cache, attempting refresh: {e}")
            self.refresh()

        if self.df is None:
            L.critical("NFO Instrument data failed to load from both API and cache.")
            raise SystemExit("System cannot run without instrument data.")

        self.df_by_token = self.df.set_index('instrument_token')
        self.df_by_symbol = self.df.set_index('tradingsymbol')
        self._load_special_tokens()
        return self

    def refresh(self):
        L.info("Refreshing NFO instrument list via OrderActor...")
        try:
            # --- FIXED: Use OrderActor for API call ---
            reply_q = queue.Queue()
            self.order_actor.q.put({
                "type": "instruments",
                "params": {"exchange": "NFO"},
                "reply_q": reply_q
            })

            instruments = None
            try:
                resp = reply_q.get(timeout=10.0) # Wait for actor's reply
                if resp['ok']:
                    instruments = resp['res']
                else:
                    L.error(f"OrderActor failed to get instruments: {resp.get('error')}")
                    raise KiteException(f"OrderActor failed: {resp.get('error')}")
            except queue.Empty:
                L.error("Timeout waiting for OrderActor reply in InstrumentBook.refresh()")
                raise KiteException("Timeout waiting for OrderActor reply")

            if instruments is None:
                raise KiteException("Failed to fetch NFO instruments from OrderActor after multiple retries.")
            
            df = pd.DataFrame(instruments)
            if "expiry" in df.columns:
                df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
            df.to_csv(self.path, index=False)
            self.df = df
            
            # --- FIXED: Use StoreActor for DB write ---
            # This is a non-blocking "fire-and-forget" message
            self.store_actor.q.put({
                "type": "set_kv",
                "key": "instruments_refreshed",
                "value": str(date.today())
            })
            
            L.info(f"NFO Instrument list refreshed and saved. Rows: {len(df)}")
        except Exception as e:
            L.error(f"API fetch for NFO instruments failed: {e}")
            if os.path.exists(self.path):
                L.warning("Loading NFO instruments from local cache as a fallback.")
                self.df = pd.read_csv(self.path, parse_dates=["expiry"])
            else:
                L.critical("API fetch failed and no NFO instrument cache is available. Cannot continue.")
                self.df = None

    def _load_special_tokens(self):
        L.info("Loading special instrument tokens...")
        # These methods are safe, they only read from self.df
        nifty_fut = self.find_current_futures_contract("NIFTY")
        bn_fut = self.find_current_futures_contract("BANKNIFTY")
        if nifty_fut:
            self.special_tokens["NIFTY"] = nifty_fut['instrument_token']
        if bn_fut:
            self.special_tokens["BANKNIFTY"] = bn_fut['instrument_token']

        # Try to find VIX dynamically
        try:
            vix_inst = self.df[self.df["name"] == "INDIA VIX"].iloc[0]
            self.special_tokens["INDIA VIX"] = int(vix_inst['instrument_token'])
        except (IndexError, KeyError):
            L.warning("Could not find 'INDIA VIX' in instrument file. Falling back to static token 257281.")
            self.special_tokens["INDIA VIX"] = 257281

        L.info(f"Special tokens loaded: {self.special_tokens}")

    # --- NO CHANGES NEEDED BELOW ---
    # All other methods only read from self.df and do not
    # make any network or database calls. They are already safe.

    def get_token(self, trading_symbol: str) -> Optional[int]:
        try:
            return int(self.df_by_symbol.loc[trading_symbol, 'instrument_token'])
        except (KeyError, IndexError):
            return None

    def get_symbol(self, token: int) -> Optional[str]:
        try:
            return self.df_by_token.loc[token, 'tradingsymbol']
        except (KeyError, IndexError):
            return None

    def find_current_futures_contract(self, name: str) -> Optional[Dict]:
        today = date.today()
        df = self.df[(self.df["name"] == name) & (self.df["segment"] == "NFO-FUT") & (self.df["expiry"].dt.date >= today)]
        return df.sort_values(by="expiry").iloc[0].to_dict() if not df.empty else None

    def find_nearest_expiry_date(self, name: str) -> Optional[date]:
        today = date.today()
        df = self.df[(self.df["name"] == name) & (self.df["instrument_type"].isin(["CE", "PE"])) & (self.df["expiry"].dt.date >= today)]
        return df["expiry"].dt.date.min() if not df.empty else None

    def find_option(self, name: str, expiry: date, strike: float, otype: str) -> Optional[Dict]:
        df = self.df[(self.df["name"] == name) & (self.df["expiry"].dt.date == expiry) & (self.df["strike"] == float(strike)) & (self.df["instrument_type"] == otype.upper())]
        return df.iloc[-1].to_dict() if not df.empty else None

    def get_option_chain(self, name: str, expiry: date) -> pd.DataFrame:
        return self.df[(self.df["name"] == name) & (self.df["expiry"].dt.date == expiry) & (self.df["instrument_type"].isin(["CE", "PE"]))]

    def lot_size(self, name: str) -> Optional[int]:
        name_map = {"NIFTY 50": "NIFTY", "NIFTY BANK": "BANKNIFTY"}
        base_name = name_map.get(name, name)
        res = self.df[(self.df.name == base_name) & (self.df.instrument_type.isin(["CE", "PE"]))]
        return int(res.lot_size.iloc[0]) if not res.empty else None

    def step_size(self, name: str) -> int:
        return 100 if "BANKNIFTY" in name.upper() else 50

    def tick_size(self, tradingsymbol: str) -> float:
        try:
            return float(self.df_by_symbol.loc[tradingsymbol, 'tick_size'])
        except (KeyError, IndexError):
            return 0.05


class PriceBus:
    def __init__(self, kite: GovernedKite, access_token: str):
        self.kite_api_key = kite._kite.api_key
        self.access_token = access_token
        self.ws = KiteTicker(self.kite_api_key, self.access_token)
        self.lock = None
        self.tokens = set()
        self.last = {}
        self.full_ticks = {}
        self.lock = threading.RLock()

        self.on_connect_callbacks = []
        self.connected = threading.Event()
        self.ws_thread = None
        self.tick_queue = Queue()
        self.order_update_queue = Queue()

        self.last_tick_reception_time: Optional[datetime] = None
        self.last_tick_reception_time_per_token: Dict[int, datetime] = {}

        self.ws.on_connect = self._on_connect
        self.ws.on_ticks = self._on_ticks
        self.ws.on_order_update = self._on_order_update
        self.ws.on_close = self._on_close
        self.ws.on_error = self._on_error

    def start(self):
        self.ws_thread = threading.Thread(target=self.ws.connect, kwargs={"threaded": True}, daemon=True, name="PriceBusWS")
        self.ws_thread.start()

    def _on_ticks(self, ws, ticks):
        self.last_tick_reception_time = now_ist()
        for t in ticks:
            self.last_tick_reception_time_per_token[t['instrument_token']] = self.last_tick_reception_time
        self.tick_queue.put(ticks)

    def _on_order_update(self, ws, order):
        L.info(f"WS Order Update Received: {order.get('tradingsymbol')} {order.get('status')} ({order.get('order_id')})")
        self.order_update_queue.put(order)

    def _on_connect(self, ws, response):
        L.info("PriceBus WebSocket connected.")
        if G_WS_CONNECTED:
            G_WS_CONNECTED.set(1)
        self.connected.set()
        with self.lock:
            if self.tokens:
                ws.subscribe(list(self.tokens))
                ws.set_mode(ws.MODE_FULL, list(self.tokens))
            if not PAPER_TRADING:
                ws.subscribe_orders()
        for cb in self.on_connect_callbacks:
            try:
                cb()
            except Exception as e:
                L.error(f"Error in on_connect callback: {e}", exc_info=True)

    def _on_close(self, ws, code, reason):
        if G_WS_CONNECTED:
            G_WS_CONNECTED.set(0)
        self.connected.clear()
        L.warning(f"PriceBus WS closed: {code}-{reason}. Will attempt to reconnect automatically.")

    def _on_error(self, ws, code, reason):
        if G_WS_CONNECTED:
            G_WS_CONNECTED.set(0)
        L.error(f"PriceBus WS Error: {code} - {reason}")

    def subscribe(self, tokens: List[int]):
        with self.lock:
            new_tokens = [t for t in tokens if t not in self.tokens and t is not None]
            if new_tokens:
                self.tokens.update(new_tokens)
                L.info(f"Subscribing to new tokens: {new_tokens}")
                if self.connected.is_set():
                    self.ws.subscribe(new_tokens)
                    self.ws.set_mode(self.ws.MODE_FULL, new_tokens)

    def ltp(self, token: int) -> Optional[float]:
        with self.lock:
            return self.last.get(token)

    def get_full_tick(self, token: int) -> Optional[Dict]:
        with self.lock:
            return self.full_ticks.get(token)


class BarStore:
    def __init__(self, timeframes: List[int]):
        self.lock = None
        self.timeframes = sorted(timeframes)
        self.data: Dict[int, Dict[int, pd.DataFrame]] = {}
        self.lock = threading.RLock()

    def _ensure_token_data(self, token: int):
        if token not in self.data:
            self.data[token] = {tf: pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).astype(float) for tf in self.timeframes}
            for tf in self.timeframes:
                self.data[token][tf].index.name = "timestamp"

    def prime(self, token: int, hist_df: pd.DataFrame, append: bool = False):
        """
        Primes the bar store with historical data.
        If append=True, it concatenates new data with existing data,
        overwriting any duplicate timestamps with the new data.
        """
        with self.lock:
            if hist_df.empty:
                return
            
            # Ensure the token entry exists in the data structure
            self._ensure_token_data(token)

            # --- Data Preparation ---
            # Convert 'date' column to timezone-aware timestamp index
            ts_col = pd.to_datetime(hist_df['date'])
            if ts_col.dt.tz is None:
                hist_df['timestamp'] = ts_col.dt.tz_localize(IST)
            else:
                hist_df['timestamp'] = ts_col.dt.tz_convert(IST)
            
            hist_df = hist_df.set_index("timestamp").drop(columns=['date'])
            base_df = hist_df.copy()

            # --- Resample and Store for each Timeframe ---
            for tf in self.timeframes:
                resampled_df = None
                if tf == 1:
                    resampled_df = base_df
                else:
                    # Resample the 1-min base data for higher timeframes
                    resampled_df = base_df.resample(f'{tf}min').agg({
                        'open': 'first', 
                        'high': 'max', 
                        'low': 'min', 
                        'close': 'last', 
                        'volume': 'sum'
                    }).dropna()

                if resampled_df.empty:
                    continue

                # --- NEW: Append/Gap-Fill Logic ---
                if append and tf in self.data[token] and not self.data[token][tf].empty:
                    # Concatenate old data with new resampled data
                    combined_df = pd.concat([self.data[token][tf], resampled_df])
                    
                    # De-duplicate index: keep='last' ensures new data overwrites old
                    self.data[token][tf] = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
                else:
                    # Original behavior: complete overwrite
                    self.data[token][tf] = resampled_df
                # --- End NEW Logic ---

            # --- Modified Log Message ---
            L.info(f"Priming BarStore for {token} ({'APPEND' if append else 'FULL'}). {len(hist_df)} new bars. Total bars (1m): {len(self.data[token][1])}")

    def add_tick(self, tick: Dict) -> Optional[List[int]]:
        token = tick['instrument_token']
        ts = tick.get('exchange_timestamp')
        price = tick.get('last_price')
        qty = tick.get('last_traded_quantity')
        if not all([isinstance(ts, datetime), isinstance(price, (float, int)), isinstance(qty, int)]):
            return None
        updated_timeframes = []
        with self.lock:
            self._ensure_token_data(token)
            for tf in self.timeframes:
                df = self.data[token][tf]
                bar_ts = ts.replace(second=0, microsecond=0) - timedelta(minutes=ts.minute % tf)
                if bar_ts not in df.index:
                    if not df.empty:
                        updated_timeframes.append(tf)
                    new_row = pd.DataFrame([[price, price, price, price, float(qty)]], columns=['open', 'high', 'low', 'close', 'volume'], index=[bar_ts])
                    self.data[token][tf] = pd.concat([df, new_row])
                else:
                    df.loc[bar_ts, 'high'] = max(df.loc[bar_ts, 'high'], price)
                    df.loc[bar_ts, 'low'] = min(df.loc[bar_ts, 'low'], price)
                    df.loc[bar_ts, 'close'] = price
                    df.loc[bar_ts, 'volume'] += qty
        return list(set(updated_timeframes)) if updated_timeframes else None

    def get_ohlc(self, token: int, timeframe: int) -> pd.DataFrame:
        with self.lock:
            return self.data.get(token, {}).get(timeframe, pd.DataFrame()).copy()


# ==================================================================================================
# MARKET REGIME ANALYSIS
# ==================================================================================================
class RegimeClassifier:
    def __init__(self, engine: 'Engine', nifty_token: int, bn_token: int, vix_token: Optional[int], params: Dict):
        self.engine = engine
        self.prices = engine.prices
        self.nifty_token = nifty_token
        self.bn_token = bn_token
        self.vix_token = vix_token
        self.params = params
        self.tf = self.params["resample_minutes"]
        self.potential_regime: Optional[Tuple[Regime, int]] = None
        self.confirmation_count: int = 0
        self.confirmation_threshold: int = self.params.get("hysteresis_confirmation_count", 3)

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < self.params["bb_period"]:
            return df
        df.ta.adx(length=self.params["adx_period"], append=True)
        bbands = df.ta.bbands(length=self.params["bb_period"], append=True)
        df['bbw'] = (bbands[f'BBU_{self.params["bb_period"]}_2.0'] - bbands[f'BBL_{self.params["bb_period"]}_2.0']) / bbands[f'BBM_{self.params["bb_period"]}_2.0']

        df['atr'] = df.ta.atr(length=14)

        lookback = self.params.get("dynamic_regime_lookback", 200)
        df['bbw_pct_rank'] = df['bbw'].rolling(lookback).rank(pct=True) * 100
        df['adx_pct_rank'] = df[f'ADX_{self.params["adx_period"]}'].rolling(lookback).rank(pct=True) * 100

        df['atr_pct_rank'] = df['atr'].rolling(lookback).rank(pct=True) * 100

        df['ema_fast'] = df.ta.ema(length=20)
        df['ema_slow'] = df.ta.ema(length=50)
        return df

    def _get_scores(self, df: pd.DataFrame, current_regime_enum: Regime) -> Dict[str, int]:
        scores = {"trend_up": 0, "trend_down": 0, "chop": 0, "compression": 0}
        if len(df) < 60:
            return scores
        dmp_col, dmn_col = f'DMP_{self.params["adx_period"]}', f'DMN_{self.params["adx_period"]}'

        compression_threshold = self.params.get("compression_rank_threshold_pct", 10.0)
        adx_entry_threshold_pct = self.params.get("adx_trend_entry_percentile", 75.0)
        adx_exit_threshold_pct = self.params.get("adx_trend_exit_percentile", 60.0)

        if df['bbw_pct_rank'].iloc[-1] < compression_threshold:
            scores["compression"] += 2

        is_trending = current_regime_enum in [Regime.TRENDING_UP, Regime.TRENDING_DOWN]
        adx_threshold = adx_exit_threshold_pct if is_trending else adx_entry_threshold_pct
        adx_pct_rank_val = df['adx_pct_rank'].iloc[-1]

        atr_pct_rank_val = df['atr_pct_rank'].iloc[-1]
        atr_expansion_threshold = self.params.get("atr_expansion_percentile", 80.0)
        if atr_pct_rank_val > atr_expansion_threshold:
            if df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1]:
                scores["trend_up"] += 1  # Add 1 point for volatility expansion
            elif df['ema_fast'].iloc[-1] < df['ema_slow'].iloc[-1]:
                scores["trend_down"] += 1

        if adx_pct_rank_val < adx_exit_threshold_pct:
            scores["chop"] += 1

        if adx_pct_rank_val > adx_threshold:
            if df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1] and df[dmp_col].iloc[-1] > df[dmn_col].iloc[-1]:
                scores["trend_up"] += 2
            elif df['ema_fast'].iloc[-1] < df['ema_slow'].iloc[-1] and df[dmn_col].iloc[-1] > df[dmp_col].iloc[-1]:
                scores["trend_down"] += 2

        return scores

    # Insert this method inside the RegimeClassifier class in main1.py

def get_raw_classification(self, current_regime_enum: Regime) -> Tuple[Regime, Optional[int], float]:
    """
    Determines the raw market regime, the preferred trading instrument token,
    and a confidence score for the classification.

    Returns:
        Tuple[Regime, Optional[int], float]: (Predicted Regime, Active Token or None, Confidence Score 0.0-1.0)
    """
    try:
        # --- VIX Spike Check (Highest Priority Override) ---
        vix_ltp = self.prices.ltp(self.vix_token)
        if vix_ltp and self.vix_token:
            df_vix = self.engine.get_ohlc(self.vix_token, self.tf)
            spike_ma_period = self.params.get("vix_spike_ma_period", 20)
            if len(df_vix) > spike_ma_period:
                vix_ma = df_vix['close'].rolling(spike_ma_period).mean().iloc[-1]
                spike_multiplier = self.params.get("vix_spike_multiplier", 1.4)
                if not pd.isna(vix_ma) and vix_ltp > (vix_ma * spike_multiplier):
                    L.warning(f"VIX SPIKE DETECTED! LTP: {vix_ltp:.2f}, MA({spike_ma_period}): {vix_ma:.2f}. Overriding regime to CHAOS.")
                    # Return CHAOS, preferred token (BN), and high confidence
                    return (Regime.CHAOS, self.bn_token, 0.95) # Very high confidence for spike override

        # --- Data Availability Check ---
        df_bn = self.engine.get_ohlc(self.bn_token, self.tf)
        df_n = self.engine.get_ohlc(self.nifty_token, self.tf)
        min_bars_needed = max(60, self.params.get("dynamic_regime_lookback", 200)) # Ensure enough lookback
        if df_bn.empty or df_n.empty or len(df_bn) < min_bars_needed or len(df_n) < min_bars_needed:
            L.warning(f"Not enough data for regime classification. BN Bars: {len(df_bn)}, N Bars: {len(df_n)}. Min Needed: {min_bars_needed}")
            # Return UNCLEAR, no token, and low confidence
            return (Regime.UNCLEAR, None, 0.1) # Low confidence if not enough data

        # --- Calculate Indicators ---
        df_n = self._add_indicators(df_n)
        df_bn = self._add_indicators(df_bn)
        
        try:
            df_n_close = df_n['close']
            df_bn_close = df_bn['close']
            
            # Calculate the ratio
            ratio_series = df_n_close / df_bn_close
            
            # Get the lookback period from your config
            lookback = self.params.get("dynamic_regime_lookback", 200)
            
            # Calculate rolling mean and std dev
            ratio_mean = ratio_series.rolling(window=lookback).mean().iloc[-1]
            ratio_std = ratio_series.rolling(window=lookback).std().iloc[-1]
            current_ratio = ratio_series.iloc[-1]

            if ratio_std > 0:
                # Calculate the z-score (how many std devs from the mean)
                zscore = (current_ratio - ratio_mean) / ratio_std
                # Store this on the main engine object
                self.engine.nifty_bn_zscore = zscore 
                L.debug(f"StatArb Z-Score: {zscore:.2f} (Ratio: {current_ratio:.4f})")
            else:
                self.engine.nifty_bn_zscore = 0.0 # Not enough data or no volatility
                
        except Exception as e:
            L.warning(f"StatArb ratio calculation failed: {e}")
            self.engine.nifty_bn_zscore = None # Set to None on failure

        # --- Calculate Scores ---
        score_n = self._get_scores(df_n, current_regime_enum)
        score_bn = self._get_scores(df_bn, current_regime_enum)

        # --- VIX Threshold Check (Second Priority Override) ---
        if vix_ltp and vix_ltp > self.params.get("vix_chaos_threshold", 24.0):
            L.warning(f"VIX above threshold ({vix_ltp:.2f} > {self.params.get('vix_chaos_threshold', 24.0)}). Overriding regime to CHAOS.")
            # Return CHAOS, preferred token (BN), and moderate-high confidence
            return (Regime.CHAOS, self.bn_token, 0.85) # High confidence for VIX threshold

        # --- Combine Scores from Nifty & BankNifty ---
        # Sum scores for each potential regime state across both indices
        final_scores = {}
        possible_regime_keys = ["compression", "trend_up", "trend_down", "chop"] # Keys used in _get_scores
        for key in possible_regime_keys:
            # Map score key to Regime Enum if possible (e.g., "trend_up" -> Regime.TRENDING_UP)
            regime_enum = None
            if key == "trend_up": regime_enum = Regime.TRENDING_UP
            elif key == "trend_down": regime_enum = Regime.TRENDING_DOWN
            elif key == "compression": regime_enum = Regime.COMPRESSION
            elif key == "chop": regime_enum = Regime.CHOP

            if regime_enum:
                final_scores[regime_enum] = score_n.get(key, 0) + score_bn.get(key, 0)

        # --- Determine Winning Regime & Calculate Confidence ---
        winning_regime = Regime.UNCLEAR
        active_token = None
        confidence = 0.2 # Default low confidence

        if final_scores and max(final_scores.values()) > 0:
            # Sort regimes by score, descending
            sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
            winning_regime = sorted_scores[0][0]
            top_score = sorted_scores[0][1]
            next_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

            # Confidence Calculation:
            # Factors:
            # 1. Magnitude of the top score (higher score = more confidence)
            # 2. Separation from the next best score (larger gap = more confidence)
            # Max possible score per index is ~2-3, so combined max is ~4-6. Let's normalize against ~5.
            magnitude_factor = min(1.0, top_score / 5.0)
            separation_factor = (top_score - next_score) / (top_score + 1e-6) # Avoid division by zero

            # Combine factors (adjust weighting as needed)
            confidence = max(0.1, min(0.95, (magnitude_factor * 0.6) + (separation_factor * 0.4)))

            # --- Determine Active Token based on Winning Regime ---
            # Use relative strength for trends, lowest volatility rank for compression
            try:
                if winning_regime == Regime.TRENDING_UP:
                    n_perf = df_n['close'].pct_change(10).iloc[-1] if not pd.isna(df_n['close'].pct_change(10).iloc[-1]) else -float('inf')
                    bn_perf = df_bn['close'].pct_change(10).iloc[-1] if not pd.isna(df_bn['close'].pct_change(10).iloc[-1]) else -float('inf')
                    active_token = self.bn_token if bn_perf > n_perf else self.nifty_token
                elif winning_regime == Regime.TRENDING_DOWN:
                    n_perf = df_n['close'].pct_change(10).iloc[-1] if not pd.isna(df_n['close'].pct_change(10).iloc[-1]) else float('inf')
                    bn_perf = df_bn['close'].pct_change(10).iloc[-1] if not pd.isna(df_bn['close'].pct_change(10).iloc[-1]) else float('inf')
                    active_token = self.bn_token if bn_perf < n_perf else self.nifty_token
                elif winning_regime == Regime.COMPRESSION:
                    n_bbw_rank = df_n['bbw_pct_rank'].iloc[-1] if not pd.isna(df_n['bbw_pct_rank'].iloc[-1]) else float('inf')
                    bn_bbw_rank = df_bn['bbw_pct_rank'].iloc[-1] if not pd.isna(df_bn['bbw_pct_rank'].iloc[-1]) else float('inf')
                    active_token = self.bn_token if bn_bbw_rank < n_bbw_rank else self.nifty_token
                elif winning_regime == Regime.CHOP:
                    # Check correlation as a tie-breaker or confirmation for CHOP
                    corr_series = df_n['close'].pct_change().rolling(self.params["correlation_period"]).corr(df_bn['close'].pct_change())
                    last_corr = corr_series.iloc[-1] if not corr_series.empty and not pd.isna(corr_series.iloc[-1]) else 0.5 # Default neutral
                    if last_corr < self.params["correlation_threshold"]:
                         confidence = min(0.95, confidence + 0.1) # Boost confidence if correlation confirms chop
                         L.debug(f"Low correlation ({last_corr:.2f}) confirms CHOP regime.")

                    active_token = self.bn_token # Default to BankNifty for CHOP for liquidity/volatility
                # CHAOS and UNCLEAR don't need active token logic here (handled by overrides or default)

            except Exception as token_ex:
                 L.warning(f"Error determining active token for regime {winning_regime.name}: {token_ex}")
                 active_token = self.bn_token # Fallback to BankNifty

        else:
            # If scores are zero or empty, it's UNCLEAR
            winning_regime = Regime.UNCLEAR
            active_token = None
            confidence = 0.2 # Low confidence

        # Final return
        return (winning_regime, active_token, confidence)

    except Exception as e:
        L.error(f"FATAL Error in regime classification: {e}", exc_info=True)
        # Return UNCLEAR, no token, and very low confidence on major error
        return (Regime.UNCLEAR, None, 0.05)

# ==================================================================================================
# TRADING EXECUTION LAYER
# ==================================================================================================
class AbstractTrader(ABC):
    def __init__(self,
                 engine: 'Engine',
                 book: InstrumentBook,
                 prices: PriceBus,
                 store: Store,
                 config: Dict,
                 perf_callback: Optional[Callable[[float], None]] = None):
        self.engine = engine
        self.book = book
        self.prices = prices
        self.store = store
        self.config = config
        self._update_performance = perf_callback or (lambda pnl: None)
        self.lock = None
        self.positions: Dict[str, Position] = {}
        self.daily_realized_pnl: float = 0.0
        self.max_concurrent = self.config["trading"]["max_concurrent_trades"]
        self.lock = threading.RLock()

    @abstractmethod
    def open_position(self, trade_params: Dict) -> Optional[Position]:
        pass

    @abstractmethod
    def close_position(self, p: Position, reason: str) -> bool:
        pass

    @abstractmethod
    def modify_sl(self, p: Position, new_trigger: float):
        pass

    @abstractmethod
    def scale_out(self, p: Position, qty_to_close: int) -> bool:
        pass

    @abstractmethod
    def place_bracket_orders(self, p: Position) -> bool:
        pass

    @abstractmethod
    def cancel_pending_entry(self, p: Position) -> bool:
        pass

    @abstractmethod
    def execute_simulated_sl(self, p: Position) -> bool:
        pass

    # In AbstractTrader class
def unrealized_pnl(self) -> float:
    """Calculates total unrealized PnL for all active positions."""
    pnl = 0.0
    with self.lock:
        # Loop through all positions the trader is currently managing
        for p in self.positions.values():
            
            # --- THE FIX ---
            # Calculate PnL for *any* position that is not fully closed 
            # or in a pre-submission state.
            # This NOW INCLUDES PENDING_CLOSURE and PENDING_SL_EXIT.
            if p.status not in [
                PositionStatus.CLOSED.value,
                PositionStatus.PENDING_SUBMISSION.value,
                PositionStatus.REJECTED.value
            ]:
                # PENDING_ENTRY positions have p.qty = 0, so they 
                # will correctly contribute 0.0 to the PnL.
                if p.qty <= 0:
                    continue

                ltp = self.prices.ltp(p.token)
                if ltp is None:
                    # If no ltp, use entry price to calculate zero PnL
                    # for this position, but don't crash.
                    ltp = p.entry_price 
                
                # Use p.qty, which is the *current* open quantity
                pnl += (ltp - p.entry_price) * p.qty 
    return pnl


class PaperTrader(AbstractTrader):
    def open_position(self, trade_params: Dict) -> Optional[Position]:
        """
        Simulates the *submission* of a new paper trade.
        This function now creates a PENDING_ENTRY position and mimics
        the first stage of adaptive entry. It does not fill the trade instantly.
        """
        with self.lock:
            try:
                opt = trade_params['opt']
                initial_lots = int(trade_params['lots'])
                lot_size = self.book.lot_size(_get_underlying(opt['tradingsymbol']))
                
                if not lot_size:
                    L.error(f"[PAPER] Could not determine lot size for {opt['tradingsymbol']}. Aborting.")
                    return None

                qty = int(initial_lots * lot_size)

                # --- Check constraints ---
                open_positions = [p for p in self.positions.values() if p.status not in [PositionStatus.CLOSED.value, PositionStatus.REJECTED.value]]
                if len(open_positions) >= self.max_concurrent:
                    L.warning(f"[PAPER] Trade rejected: max concurrent trades ({self.max_concurrent}) reached.")
                    return None

                if qty <= 0:
                    L.warning(f"[PAPER] Trade rejected: calculated quantity is {qty}.")
                    return None

                # --- NEW: Mimic Adaptive Entry Bid Placement ---
                full_tick = self.prices.get_full_tick(int(opt['instrument_token']))
                if not full_tick or not full_tick.get('depth') or not full_tick['depth'].get('buy'):
                    L.warning(f"[PAPER] Cannot get depth for {opt['tradingsymbol']}, cannot simulate entry.")
                    return None
                
                # Get the bid price, just like the live trader
                bid_price = full_tick['depth']['buy'][0]['price']
                
                pos_id = f"PAPER_{uuid.uuid4()}"

                # --- Create PENDING Position ---
                pos = Position(
                    id=pos_id,
                    tradingsymbol=opt['tradingsymbol'],
                    token=int(opt['instrument_token']),
                    option_type=opt['instrument_type'],
                    
                    # --- Entry state is PENDING ---
                    status=PositionStatus.PENDING_ENTRY.value,
                    qty=0,  # Not filled yet
                    initial_qty=qty, # The total quantity we *want* to fill
                    entry_price=0, # Not filled yet
                    
                    # --- Store trade parameters to be used *upon fill* ---
                    option_sl_points=float(trade_params['option_sl_points']),
                    option_tp_points=float(trade_params['option_tp_points']),
                    initial_risk_points=float(trade_params['option_sl_points']), # Same as option_sl_points
                    
                    # --- Adaptive Entry Simulation State ---
                    last_entry_modification=now_ist(),
                    entry_stage=1,
                    is_entry_order_open=True, # Simulates an open order
                    entry_order_id=pos_id, # Use pos_id as the mock order_id
                    
                    # --- Standard parameters ---
                    opened_at=now_ist(), # This is the *submission* time
                    strategy=trade_params['strategy'],
                    underlying_sl_level=float(trade_params['underlying_sl']) if trade_params['underlying_sl'] is not None else None,
                    market_regime_at_entry=trade_params['regime'],
                    scale_out_rules=self.config['trading']['scale_out_rules'],
                    max_trade_duration_minutes=trade_params.get('max_trade_duration_minutes', 90),
                    oi_profit_target=trade_params.get('oi_profit_target'),
                    intended_risk_rupees=trade_params.get('total_trade_risk', 0.0),

                    # --- Stash the target bid price for the simulation logic ---
                    # We use 'greeks' dict as a generic stash for paper trading
                    greeks={"target_bid": bid_price},
                    
                    # --- Fields to be set on fill ---
                    initial_sl_price=0,
                    sl_price=0,
                    tp_price=0,
                    high_price_since_entry=0
                )

                self.positions[pos.id] = pos
                self.store_actor.q.put({"type": "upsert_position", "pos": pos})
                self.prices.subscribe([pos.token])
                
                send_alert(f"⏳ [PAPER] {pos.strategy} PENDING ENTRY {pos.tradingsymbol} Qty={qty} @ Target {bid_price:.2f}")

                return pos

            except Exception as e:
                L.critical(f"FATAL EXCEPTION in PaperTrader.open_position: {e}", exc_info=True)
                return None

    def close_position(self, p: Position, reason: str) -> bool:
        with self.lock:
            if p.status == PositionStatus.CLOSED.value:
                return True

            tick_size = self.book.tick_size(p.tradingsymbol)
            slippage = tick_size * self.config['trading'].get('paper_trade_slippage_ticks', 1)

            if reason == "SL_HIT_PAPER":
                exit_price = p.sl_price - slippage
            elif reason == "TP_HIT_PAPER":
                exit_price = p.tp_price - slippage
            else:
                current_ltp = self.prices.ltp(p.token) or p.entry_price
                exit_price = current_ltp - slippage

            pnl = (exit_price - p.entry_price) * p.initial_qty
            self.daily_realized_pnl += pnl
            p.status = PositionStatus.CLOSED.value
            p.exit_reason = reason
            p.exit_price = exit_price
            self.store_actor.q.put({"type": "upsert_position", "pos": p})
            self.store_actor.q.put({
                "type": "log_closed_trade", 
                "pos": p, "price": exit_price, "reason": reason
            })
            self.store_actor.q.put({
                "type": "log_strategy_performance", 
                "name": p.strategy, "pnl": pnl
            })

            self._update_performance(pnl)

            self.positions.pop(p.id, None)
            send_alert(f"❌ [PAPER] CLOSED {p.tradingsymbol} @ {exit_price:.2f} ({reason}). Final PnL: {pnl:.2f}. Daily PnL: {self.daily_realized_pnl:.2f}")
            return True

    def cancel_pending_entry(self, p: Position) -> bool:
        with self.lock:
            if p.status != PositionStatus.PENDING_ENTRY.value:
                return False
            p.status = PositionStatus.CLOSED.value
            p.exit_reason = "ENTRY_CANCELLED_PAPER"
            self.store_actor.q.put({"type": "upsert_position", "pos": p})
            self.positions.pop(p.id, None)
            L.info(f"[PAPER] Cancelled pending entry for {p.tradingsymbol}")
            return True

    def modify_sl(self, p: Position, new_trigger: float):
        with self.lock:
            if new_trigger > p.sl_price:
                L.info(f"[PAPER] Trailing SL for {p.tradingsymbol} from {p.sl_price:.2f} to {new_trigger:.2f}")
                p.sl_price = new_trigger
                self.store_actor.q.put({"type": "upsert_position", "pos": p})
    
    # In PaperTrader class, add new method
    # Inside PaperTrader class
    # In PaperTrader class
    def _manage_pending_entries(self, now: datetime):
        with self.lock:
            # Get a copy of IDs to iterate over, allowing removal from original dict
            pending_ids = [pid for pid, p in self.positions.items() if p.status == PositionStatus.PENDING_ENTRY.value]

        if not pending_ids: return # No pending entries to manage

        for p_id in pending_ids:
             with self.lock: # Lock needed if accessing/modifying self.positions
                 p = self.positions.get(p_id)
                 if not p or p.status != PositionStatus.PENDING_ENTRY.value: continue # Position gone or status changed

                 # Check if essential info is missing
                 if not p.last_entry_modification:
                      L.warning(f"Paper Entry {p.id} missing last_entry_modification timestamp. Cannot manage.")
                      continue

                 time_since_mod = (now - p.last_entry_modification).total_seconds()

                 full_tick = self.prices.get_full_tick(p.token)
                 if not full_tick:
                      L.debug(f"Paper Entry {p.id}: No full tick data available for {p.tradingsymbol}.")
                      continue

                 ltp = full_tick.get('last_price')
                 depth = full_tick.get('depth')
                 if not ltp or not depth or not depth.get('buy') or not depth.get('sell'):
                      L.debug(f"Paper Entry {p.id}: Missing LTP or depth for {p.tradingsymbol}.")
                      continue

                 bid_price = depth['buy'][0]['price']
                 ask_price = depth['sell'][0]['price']
                 # Ensure tick size rounding happens correctly
                 tick_size = self.book.tick_size(p.tradingsymbol)
                 mid_price = round(((bid_price + ask_price) / 2.0) / tick_size) * tick_size

                 target_price = -1.0
                 current_stage = p.entry_stage

                 # --- State Machine for Entry Target Price ---
                 if current_stage == 1:
                     target_price = p.greeks.get("target_bid", bid_price) # Target is BID
                     # Check if time to move to next stage
                     if time_since_mod > self.config['trading']['adaptive_entry_stage2_ms'] / 1000.0:
                         p.entry_stage = 2
                         p.last_entry_modification = now
                         p.greeks["target_mid"] = mid_price # Store new MID target
                         L.info(f"[PAPER] Entry {p.id} -> Stage 2 (Target Mid @ {mid_price:.2f})")
                         self.store_actor.q.put({"type": "upsert_position", "pos": p}) # Save stage change
                         target_price = mid_price # Update target price for this cycle's fill check

                 elif current_stage == 2:
                     target_price = p.greeks.get("target_mid", mid_price) # Target is MID
                     # Check if time to move to next stage
                     if time_since_mod > self.config['trading']['adaptive_entry_stage3_ms'] / 1000.0:
                         p.entry_stage = 3
                         p.last_entry_modification = now
                         p.greeks["target_ask"] = ask_price # Store new ASK target
                         L.info(f"[PAPER] Entry {p.id} -> Stage 3 (Target Ask @ {ask_price:.2f})")
                         self.store_actor.q.put({"type": "upsert_position", "pos": p}) # Save stage change
                         target_price = ask_price # Update target price for this cycle's fill check

                 elif current_stage == 3:
                     target_price = p.greeks.get("target_ask", ask_price) # Target is ASK
                     # Check for total entry timeout ONLY in the final stage
                     entry_timeout_config = (self.config['trading']['adaptive_entry_stage2_ms'] +
                                            self.config['trading']['adaptive_entry_stage3_ms'] +
                                            3000) # Total time + buffer in ms
                     entry_timeout_sec = entry_timeout_config / 1000.0

                     time_since_open = (now - p.opened_at).total_seconds() # Time since SUBMISSION
                     if time_since_open > entry_timeout_sec:
                         L.warning(f"[PAPER] Entry {p.id} ({p.tradingsymbol}) TIMED OUT after {time_since_open:.1f}s. Cancelling.")
                         self.cancel_pending_entry(p) # This removes from self.positions
                         continue # Move to next ID
                 
                 # --- REFACTORED FILL SIMULATION (NO RANDOMNESS) ---
                 fill_price = None

                 if current_stage == 3:
                     # Stage 3: Aggressive (target is ASK).
                     # We are crossing the spread. This should fill *immediately* at the ask price.
                     fill_price = target_price # target_price *is* the ask_price we set
                     L.info(f"✅ [PAPER] Stage 3 (ASK) FILL @ {fill_price:.2f} (Simulated crossing spread to target ASK {target_price:.2f}, LTP: {ltp:.2f})")

                 elif target_price > 0 and ltp <= target_price:
                     # Stage 1 (BID) or Stage 2 (MID): Passive
                     # The market price (LTP) has traded *at or below* our resting order's price.
                     # This simulates a real LIMIT order fill.
                     fill_price = target_price # We get filled at our limit price
                     L.info(f"✅ [PAPER] Stage {current_stage} ({'BID' if current_stage == 1 else 'MID'}) FILL @ {fill_price:.2f} (LTP {ltp:.2f} touched/crossed target {target_price:.2f})")
                 
                 # --- END REFACTORED FILL SIMULATION ---

                 # --- If Filled ---
                 if fill_price is not None:
                    p.entry_price = round(fill_price / tick_size) * tick_size # Ensure fill respects tick size
                    p.qty = p.initial_qty
                    p.opened_at = now # Set open time to fill time
                    p.initial_sl_price = p.entry_price - p.option_sl_points
                    p.sl_price = p.initial_sl_price
                    p.tp_price = p.entry_price + p.option_tp_points
                    p.high_price_since_entry = p.entry_price
                    p.status = PositionStatus.ACTIVE.value # Set to ACTIVE
                    p.greeks = {} # Clear the stashed targets
                    p.is_entry_order_open = False # Order is filled
                    p.entry_order_id = None # Clear mock order id
                    p.last_entry_modification = now # Update modification time on fill

                    send_alert(f"✅ [PAPER] {p.strategy} OPENED {p.tradingsymbol} Qty={p.qty} @ {p.entry_price:.2f}, SL={p.sl_price:.2f}, TP={p.tp_price:.2f}")
                    self.store_actor.q.put({"type": "upsert_position", "pos": p}) # Update DB immediately after fill
                    continue # Move to the next pending ID

def scale_out(self, p: Position, qty_to_close: int) -> bool:
    with self.lock:
        if p.status not in [PositionStatus.ACTIVE.value, PositionStatus.PARTIALLY_CLOSED.value] or qty_to_close <= 0 or qty_to_close > p.qty:
            # Added check for qty_to_close > p.qty
            if qty_to_close > p.qty:
                 L.warning(f"[PAPER] Scale out failed: Cannot close {qty_to_close} Qty, only {p.qty} remaining.")
            return False

        current_ltp = self.prices.ltp(p.token) or p.entry_price
        tick_size = self.book.tick_size(p.tradingsymbol)
        slippage = tick_size * self.config['trading'].get('paper_trade_slippage_ticks', 1)
        exit_price = round((current_ltp - slippage) / tick_size) * tick_size # Ensure exit price respects tick size

        pnl = (exit_price - p.entry_price) * qty_to_close
        self.daily_realized_pnl += pnl

        # --- REFACTOR START ---
        
        # 1. Log the partial trade directly using StoreActor
        #    (Replaces the call to _log_partial_trade which doesn't exist in PaperTrader)
        partial_trade_id = f"{p.id}_PARTIAL_SCALE_OUT_PAPER_{uuid.uuid4()}" 
        # Create a temporary 'partial position' object for logging, reflecting the closed quantity
        partial_pos_log_data = dataclasses.replace(p, id=partial_trade_id, initial_qty=qty_to_close, qty=qty_to_close)
        
        self.store_actor.q.put({
            "type": "log_closed_trade", 
            "pos": partial_pos_log_data, # Use the temporary object for logging
            "price": exit_price, 
            "reason": "PARTIAL_SCALE_OUT_PAPER"
        })

        # 2. Log strategy performance using StoreActor
        self.store_actor.q.put({
            "type": "log_strategy_performance", 
            "name": p.strategy, 
            "pnl": pnl
        })
        
        # --- REFACTOR END ---

        self._update_performance(pnl) # This callback is fine
        
        # Construct alert message before modifying position state
        alert_msg = f"💰 [PAPER] SCALED OUT {qty_to_close} of {p.tradingsymbol} @ {exit_price:.2f}. PnL: {pnl:.2f}. Daily PnL: {self.daily_realized_pnl:.2f}" # Corrected self.trader access

        L.info(f"[PAPER] Scaling out {qty_to_close} of {p.tradingsymbol}")
        
        # Update position state *after* logging and constructing messages
        p.qty -= qty_to_close
        p.scaled_out_qty += qty_to_close
        new_status = PositionStatus.PARTIALLY_CLOSED.value if p.qty > 0 else PositionStatus.CLOSED.value
        
        if p.status != new_status:
            L.info(f"[PAPER] {p.tradingsymbol} status changed: {p.status} -> {new_status}")
            p.status = new_status

        # 3. Upsert the final position state using StoreActor (already correct)
        self.store_actor.q.put({"type": "upsert_position", "pos": p})
        
        # Send alert now
        send_alert(alert_msg)

        if p.status == PositionStatus.CLOSED.value:
            L.info(f"[PAPER] Position {p.id} fully closed via scale out.")
            self.positions.pop(p.id, None) # Remove from active dict

        return True

    def place_bracket_orders(self, p: Position) -> bool:
        p.status = PositionStatus.ACTIVE.value
        self.store_actor.q.put({"type": "upsert_position", "pos": p})
        return True

    def execute_simulated_sl(self, p: Position) -> bool:
        L.warning(f"[PAPER] VIRTUAL SL HIT for {p.tradingsymbol}. Closing position.")
        return self.close_position(p, "SL_HIT_PAPER")


class Trader(AbstractTrader):
    def __init__(self,
                 engine: 'Engine',
                 store: Store,               # Pass original store for startup read ONLY
                 store_actor: StoreActor,    # Pass actor for runtime writes
                 order_actor: OrderActor,    # Pass actor for runtime I/O
                 book: InstrumentBook,
                 prices: PriceBus,
                 config: Dict,
                 perf_callback: Optional[Callable[[float], None]] = None):
        
        # Pass None for store to super(), as runtime writes use store_actor
        super().__init__(engine, book, prices, None, config, perf_callback) 
        
        self.order_actor = order_actor
        self.store_actor = store_actor
        
        # This is the *only* place we use the original store object directly.
        # This is safe because it's a blocking read *before* the engine starts.
        # It MUST use the passed `store` object, not the actor.
        try:
            L.info("Trader loading initial open positions from store...")
            self.positions = store.load_open_positions() 
            L.info(f"Loaded {len(self.positions)} open positions.")
        except Exception as e:
            L.critical(f"FATAL: Failed to load open positions on startup: {e}", exc_info=True)
            # Depending on severity, you might want to raise SystemExit here
            self.positions = {} # Start with empty positions if load fails

        # --- Startup Order Reconciliation (Needs direct API access via OrderActor) ---
        if self.positions:
            L.info("Reconciling open orders for existing positions on startup...")
            reply_q = queue.Queue()
            self.order_actor.q.put({"type": "orders", "reply_q": reply_q}) # Assuming OrderActor handles 'orders' type
            
            open_orders = None
            try:
                resp = reply_q.get(timeout=15.0) # Longer timeout for potentially large order book
                if resp['ok']:
                    open_orders = resp['res']
                else:
                    L.error(f"Failed to fetch open orders during startup reconciliation: {resp.get('error')}")
            except queue.Empty:
                L.error("Timeout fetching open orders during startup reconciliation.")
            except Exception as e:
                L.error(f"Exception fetching open orders during startup reconciliation: {e}", exc_info=True)

            if open_orders is not None:
                open_order_ids = {str(o['order_id']) for o in open_orders if o.get('status') == 'OPEN'}
                L.info(f"Found {len(open_order_ids)} open orders at broker.")
                
                needs_db_update = False
                for p in list(self.positions.values()):
                    if p.tp_order_id and str(p.tp_order_id) not in open_order_ids:
                        L.warning(f"Startup Reconcile: TP order {p.tp_order_id} for {p.tradingsymbol} not found open. Clearing.")
                        p.tp_order_id = None
                        needs_db_update = True
                    if p.slm_order_id and str(p.slm_order_id) not in open_order_ids:
                        L.warning(f"Startup Reconcile: SLM order {p.slm_order_id} for {p.tradingsymbol} not found open. Clearing.")
                        p.slm_order_id = None
                        needs_db_update = True
                    # Check pending entry orders too
                    if p.entry_order_id and p.status == PositionStatus.PENDING_ENTRY.value and str(p.entry_order_id) not in open_order_ids:
                         L.warning(f"Startup Reconcile: Pending Entry order {p.entry_order_id} for {p.tradingsymbol} not found open. Marking as REJECTED.")
                         p.status = PositionStatus.REJECTED.value
                         p.exit_reason = "STARTUP_RECONCILE_ENTRY_MISSING"
                         p.entry_order_id = None
                         p.is_entry_order_open = False
                         needs_db_update = True

                    if needs_db_update:
                        # Send non-blocking update to StoreActor
                        self.store_actor.q.put({"type": "upsert_position", "pos": p})
                        needs_db_update = False # Reset for next position
            else:
                 L.error("Could not retrieve open orders from broker. Reconciliation incomplete.")


        # Subscribe to ticks for loaded positions
        for p in self.positions.values():
            if p.token: # Ensure token exists
                self.prices.subscribe([p.token])

    # --- Order History Fetch (Refactored) ---
    def _get_order_avg_price(self, oid: str) -> Optional[float]:
        """Fetches order history via OrderActor and calculates the average fill price."""
        
        L.debug(f"Requesting order history for {oid} via OrderActor.")
        reply_q = queue.Queue()
        self.order_actor.q.put({
            "type": "order_history",
            "params": {"order_id": oid},
            "reply_q": reply_q
        })

        history = None
        try:
            # Block waiting for the actor's response (holds no locks)
            resp = reply_q.get(timeout=10.0) # Increased timeout for potentially slow API
            if resp['ok'] and resp['res'] is not None:
                history = resp['res']
                L.debug(f"Received history for {oid}: {len(history)} entries.")
            else:
                L.error(f"Failed to get order history for {oid} from OrderActor: {resp.get('error')}")
                return None
        except queue.Empty:
            L.error(f"Timeout waiting for order history response for {oid} from OrderActor.")
            return None
        except Exception as e:
            L.error(f"Unexpected error getting order history for {oid}: {e}", exc_info=True)
            return None

        if not history:
            L.warning(f"Order history for {oid} was empty after fetch.")
            return None
            
        trades = [t for t in history if t.get('status') == 'COMPLETE' and t.get('filled_quantity', 0) > 0]
        if not trades:
            L.warning(f"No 'COMPLETE' trades with filled quantity found in history for order {oid}.")
            return None
            
        total_qty = sum(t['filled_quantity'] for t in trades)
        # Use average_price if available (more accurate), fallback to price
        weighted_sum = sum(t.get('average_price', t.get('price', 0)) * t['filled_quantity'] for t in trades) 

        if total_qty > 0:
            avg_price = weighted_sum / total_qty
            L.info(f"Calculated avg price for order {oid}: {avg_price:.2f} (Qty: {total_qty})")
            return avg_price
        else:
            L.warning(f"Total filled quantity for order {oid} is zero after filtering trades.")
            return None

    # --- Order Cancellation (Refactored) ---
    def _cancel_all_open_orders_for_pos(self, p: Position, cancel_entry: bool = False):
        """
        Cancels all associated open orders for a position using the OrderActor (non-blocking).
        """
        # Define constants locally or globally if self.k is removed
        VARIETY_REGULAR = "regular" 
        
        orders_to_cancel = []
        if p.tp_order_id: orders_to_cancel.append(p.tp_order_id)
        if p.slm_order_id: orders_to_cancel.append(p.slm_order_id)

        if cancel_entry and p.entry_order_id:
            orders_to_cancel.append(p.entry_order_id)

        if not orders_to_cancel:
            L.debug(f"No orders to cancel for {p.tradingsymbol}.")
            return

        L.debug(f"Requesting cancellation for orders associated with {p.tradingsymbol}: {orders_to_cancel}")
        
        orders_sent_for_cancel = [] 
        for oid in orders_to_cancel:
            if oid: # Ensure oid is not None
                oid_str = str(oid)
                # Send a non-blocking ("fire-and-forget") message
                self.order_actor.q.put({
                    "type": "cancel_order",
                    "params": {"variety": VARIETY_REGULAR, "order_id": oid_str}, 
                    "reply_q": None # No need to wait
                })
                L.info(f"Sent cancel request for order {oid_str} via OrderActor.")
                orders_sent_for_cancel.append(oid_str) 
                
        # Immediately clear the local state
        if p.tp_order_id and str(p.tp_order_id) in orders_sent_for_cancel:
            p.tp_order_id = None
        if p.slm_order_id and str(p.slm_order_id) in orders_sent_for_cancel:
            p.slm_order_id = None
        if cancel_entry and p.entry_order_id and str(p.entry_order_id) in orders_sent_for_cancel:
            p.entry_order_id = None
            p.is_entry_order_open = False 
            
        L.debug(f"Cleared local order IDs for {p.tradingsymbol} after sending cancel requests.")
        # Caller function is responsible for upserting the position state if needed.

    # --- Open Position (Refactored) ---
    def open_position(self, trade_params: Dict) -> Optional[Position]:
        """
        Submits an adaptive entry order via the OrderActor and saves state via StoreActor.
        """
        # Define constants locally or globally
        VARIETY_REGULAR = "regular"
        TRANSACTION_TYPE_BUY = "BUY"
        PRODUCT_MIS = "MIS"
        ORDER_TYPE_LIMIT = "LIMIT"
        
        with self.lock:
            # --- 1. Pre-Trade Checks ---
            open_pos_count = sum(1 for p in self.positions.values() if p.status not in [PositionStatus.CLOSED.value, PositionStatus.REJECTED.value])
            opt, ts = trade_params['opt'], trade_params['opt']['tradingsymbol']
            lot_size = self.book.lot_size(_get_underlying(ts))

            if open_pos_count >= self.max_concurrent or not lot_size:
                L.warning(f"Trade rejected: Max concurrent ({self.max_concurrent}) or no lot size.")
                return None

            lots = int(trade_params['lots'])
            qty = int(lots * lot_size)
            if qty <= 0:
                L.warning(f"Trade rejected: Calculated quantity is {qty}.")
                return None

            # --- 2. Create Initial Position Object ---
            temp_id = f"TEMP_{uuid.uuid4()}"
            pos = Position(
                id=temp_id,
                status=PositionStatus.PENDING_SUBMISSION.value, 
                tradingsymbol=ts, token=int(opt['instrument_token']), option_type=opt['instrument_type'],
                qty=0, initial_qty=qty, entry_price=0, initial_sl_price=0, sl_price=0, tp_price=0,
                opened_at=now_ist(), strategy=trade_params['strategy'], market_regime_at_entry=trade_params['regime'],
                underlying_sl_level=trade_params['underlying_sl'],
                option_sl_points=trade_params['option_sl_points'],
                option_tp_points=trade_params['option_tp_points'],
                initial_risk_points=trade_params['option_sl_points'],
                scale_out_rules=self.config['trading']['scale_out_rules'],
                max_trade_duration_minutes=trade_params.get('max_trade_duration_minutes', 90),
                oi_profit_target=trade_params.get('oi_profit_target'),
                intended_risk_rupees=trade_params.get('total_trade_risk', 0.0),
                last_entry_modification=now_ist()
            )

            # --- 3. Save Initial State via StoreActor ---
            self.store_actor.q.put({"type": "upsert_position", "pos": pos})
            self.positions[pos.id] = pos # Add to local dict immediately

            # --- 4. Get Depth for Entry Price ---
            full_tick = self.prices.get_full_tick(int(opt['instrument_token']))
            if not full_tick or not full_tick.get('depth') or not full_tick['depth'].get('buy'):
                L.warning(f"Cannot get depth for {ts}, cannot place adaptive entry.")
                pos.status = PositionStatus.REJECTED.value
                pos.exit_reason = "NO_DEPTH_FOR_ENTRY"
                self.store_actor.q.put({"type": "upsert_position", "pos": pos})
                self.positions.pop(pos.id, None) 
                return None

            bid_price = full_tick['depth']['buy'][0]['price']

            # --- 5. Place Order via OrderActor ---
            place_params = {
                "variety": VARIETY_REGULAR, "exchange": "NFO", "tradingsymbol": ts,
                "transaction_type": TRANSACTION_TYPE_BUY, "quantity": qty,
                "product": PRODUCT_MIS,
                "order_type": ORDER_TYPE_LIMIT, "price": bid_price
            }

            L.info(f"Sending place order request for {ts} @ {bid_price} (Qty: {qty}) via OrderActor.")
            reply_q = queue.Queue()
            self.order_actor.q.put({
                "type": "place_order",
                "params": place_params,
                "reply_q": reply_q
            })

            oid = None
            try:
                resp = reply_q.get(timeout=10.0) 
                if resp['ok'] and resp['res'] and 'order_id' in resp['res']:
                    oid = resp['res']['order_id']
                    L.info(f"OrderActor successfully placed order {oid} for {ts}.")
                else:
                    raise Exception(resp.get('error', 'Unknown error from OrderActor'))
            except queue.Empty:
                L.error(f"Timeout waiting for place order response for {ts} from OrderActor.")
            except Exception as e:
                L.error(f"Adaptive Entry Stage 1 order placement failed for {ts}. Error: {e}")

            # --- 6. Handle Order Placement Result ---
            if oid is None:
                pos.status = PositionStatus.REJECTED.value
                pos.exit_reason = "BROKER_API_FAILURE"
                self.store_actor.q.put({"type": "upsert_position", "pos": pos})
                self.positions.pop(pos.id, None) 
                return None
            else:
                L.info(f"Placed Adaptive Entry Stage 1 (Passive) order {oid} for {ts} @ {bid_price}")
                self.positions.pop(temp_id, None) # Remove temp ID

                pos.id = f"LIVE_{oid}"
                pos.entry_order_id = str(oid)
                pos.status = PositionStatus.PENDING_ENTRY.value
                pos.entry_stage = 1
                pos.is_entry_order_open = True 

                self.store_actor.q.put({"type": "upsert_position", "pos": pos})
                self.positions[pos.id] = pos # Add back with correct ID
                self.prices.subscribe([pos.token]) 
                return pos

    # --- Place Brackets (Refactored) ---
    def place_bracket_orders(self, p: Position) -> bool:
        """Places or modifies TP and SL orders via OrderActor after entry fill."""
        # Define constants locally or globally
        VARIETY_REGULAR = "regular"
        TRANSACTION_TYPE_SELL = "SELL"
        PRODUCT_MIS = "MIS"
        ORDER_TYPE_LIMIT = "LIMIT"
        ORDER_TYPE_SLM = "SL-M"
        
        if p.status != PositionStatus.OPEN_AWAITING_BRACKETS.value:
            L.warning(f"place_bracket_orders called for {p.tradingsymbol} but status is {p.status}. Skipping.")
            return False
            
        if p.qty <= 0:
            L.error(f"Cannot place brackets for {p.tradingsymbol} with zero quantity ({p.qty}).")
            p.status = PositionStatus.CLOSED.value 
            p.exit_reason = "ZERO_QTY_ON_BRACKET"
            self.store_actor.q.put({"type": "upsert_position", "pos": p})
            self.positions.pop(p.id, None) # Remove if closed due to zero qty
            return False

        tick_size = self.book.tick_size(p.tradingsymbol)
        
        tp_success = False
        slm_success = False
        tp_oid_attempted = None
        slm_oid_attempted = None
        
        # --- 1. Place/Modify TP Order ---
        if p.tp_price > 0:
            tp_limit = round(p.tp_price / tick_size) * tick_size
            if not p.tp_order_id: # Place new TP
                place_params = {
                    "variety": VARIETY_REGULAR, "exchange": "NFO", 
                    "tradingsymbol": p.tradingsymbol, "transaction_type": TRANSACTION_TYPE_SELL, 
                    "quantity": p.qty, "product": PRODUCT_MIS, 
                    "order_type": ORDER_TYPE_LIMIT, "price": tp_limit
                }
                L.info(f"Sending request to place TP order @ {tp_limit} for {p.qty} of {p.tradingsymbol}...")
                reply_q = queue.Queue()
                self.order_actor.q.put({"type": "place_order", "params": place_params, "reply_q": reply_q})
                try:
                    resp = reply_q.get(timeout=10.0)
                    if resp['ok'] and resp['res'] and 'order_id' in resp['res']:
                        tp_oid_attempted = str(resp['res']['order_id'])
                        p.tp_order_id = tp_oid_attempted # Store ID
                        tp_success = True
                        L.info(f"Successfully placed TP order {p.tp_order_id}.")
                    else: raise Exception(resp.get('error', 'Unknown TP placement error'))
                except Exception as e: L.error(f"Failed to place TP order for {p.tradingsymbol}: {e}")
            
            else: # Modify existing TP quantity
                modify_params = {"variety": VARIETY_REGULAR, "order_id": p.tp_order_id, "quantity": p.qty}
                L.info(f"Sending request to modify TP order {p.tp_order_id} qty to {p.qty}...")
                reply_q = queue.Queue()
                self.order_actor.q.put({"type": "modify_order", "params": modify_params, "reply_q": reply_q})
                try:
                    resp = reply_q.get(timeout=10.0)
                    if resp['ok']:
                        tp_oid_attempted = p.tp_order_id 
                        tp_success = True
                        L.info(f"Successfully modified TP order {p.tp_order_id} quantity.")
                    else: raise Exception(resp.get('error', 'Unknown TP modification error'))
                except Exception as e: L.error(f"Failed to modify TP order {p.tp_order_id}: {e}")
        else: tp_success = True # No TP needed

        # --- 2. Place/Modify SL-M Order ---
        if p.sl_price > 0:
            sl_trigger = round(p.sl_price / tick_size) * tick_size
            if not p.slm_order_id: # Place new SLM
                place_params = {
                    "variety": VARIETY_REGULAR, "exchange": "NFO", "tradingsymbol": p.tradingsymbol,
                    "transaction_type": TRANSACTION_TYPE_SELL, "quantity": p.qty, 
                    "product": PRODUCT_MIS, "order_type": ORDER_TYPE_SLM, 
                    "trigger_price": sl_trigger
                }
                L.info(f"Sending request to place SL-M order @ trigger {sl_trigger} for {p.qty} of {p.tradingsymbol}...")
                reply_q = queue.Queue()
                self.order_actor.q.put({"type": "place_order", "params": place_params, "reply_q": reply_q})
                try:
                    resp = reply_q.get(timeout=10.0)
                    if resp['ok'] and resp['res'] and 'order_id' in resp['res']:
                        slm_oid_attempted = str(resp['res']['order_id'])
                        p.slm_order_id = slm_oid_attempted # Store ID
                        slm_success = True
                        L.info(f"Successfully placed SL-M order {p.slm_order_id}.")
                    else: raise Exception(resp.get('error', 'Unknown SLM placement error'))
                except Exception as e: L.error(f"Failed to place SL-M order for {p.tradingsymbol}: {e}")

            else: # Modify existing SLM quantity
                modify_params = {"variety": VARIETY_REGULAR, "order_id": p.slm_order_id, "quantity": p.qty}
                L.info(f"Sending request to modify SL-M order {p.slm_order_id} qty to {p.qty}...")
                reply_q = queue.Queue()
                self.order_actor.q.put({"type": "modify_order", "params": modify_params, "reply_q": reply_q})
                try:
                    resp = reply_q.get(timeout=10.0)
                    if resp['ok']:
                        slm_oid_attempted = p.slm_order_id 
                        slm_success = True
                        L.info(f"Successfully modified SL-M order {p.slm_order_id} quantity.")
                    else: raise Exception(resp.get('error', 'Unknown SLM modification error'))
                except Exception as e: L.error(f"Failed to modify SL-M order {p.slm_order_id}: {e}")
        else: slm_success = True # No SL needed (unlikely)

        # --- 3. Finalize State Based on Success ---
        if tp_success and slm_success:
            p.status = PositionStatus.ACTIVE.value
            self.store_actor.q.put({"type": "upsert_position", "pos": p}) 
            L.info(f"Position {p.tradingsymbol} ACTIVE. SL: {p.slm_order_id} @ {p.sl_price:.2f}, TP: {p.tp_order_id} @ {p.tp_price:.2f}.")
            return True
        else:
            # Cleanup logic: If one succeeded, cancel it
            L.error(f"Bracket placement failed for {p.tradingsymbol}. TP Success: {tp_success}, SLM Success: {slm_success}. Cleaning up.")
            orders_to_clean = []
            if tp_success and tp_oid_attempted: orders_to_clean.append(tp_oid_attempted); p.tp_order_id = None
            if slm_success and slm_oid_attempted: orders_to_clean.append(slm_oid_attempted); p.slm_order_id = None
                 
            for oid_clean in orders_to_clean:
                 L.warning(f"Cleaning up orphaned order: {oid_clean}")
                 self.order_actor.q.put({ # Fire-and-forget cancel
                     "type": "cancel_order", "params": {"variety": VARIETY_REGULAR, "order_id": oid_clean},
                     "reply_q": None })
                 
            # Trigger immediate close as placing brackets failed
            send_alert(f"CRITICAL: Failed bracket placement for {p.tradingsymbol}. Closing position!", "critical")
            self.close_position(p, "BRACKET_PLACEMENT_FAILURE") 
            return False

    # --- Close Position (Refactored) ---
    def close_position(self, p: Position, reason: str) -> bool:
        """Closes a position by cancelling brackets and placing a market order via Actors."""
        # Define constants locally or globally
        VARIETY_REGULAR = "regular"
        TRANSACTION_TYPE_SELL = "SELL"
        PRODUCT_MIS = "MIS"
        ORDER_TYPE_MARKET = "MARKET"
        
        with self.lock:
            if p.status in [PositionStatus.PENDING_CLOSURE.value, PositionStatus.CLOSED.value, PositionStatus.PENDING_SL_EXIT.value]:
                L.debug(f"close_position called for {p.tradingsymbol} ({p.id}) but already closing/closed ({p.status}). Skipping.")
                return True 

            original_status = p.status
            p.status = PositionStatus.PENDING_CLOSURE.value
            p.exit_reason = reason
            
            self.store_actor.q.put({"type": "upsert_position", "pos": p})
            L.info(f"Marked {p.tradingsymbol} ({p.id}) as PENDING_CLOSURE. Reason: {reason}.")

            self._cancel_all_open_orders_for_pos(p, cancel_entry=(original_status == PositionStatus.PENDING_ENTRY.value))

            if p.qty <= 0:
                L.warning(f"Attempted to close {p.tradingsymbol} ({p.id}) with zero/negative quantity ({p.qty}). Marking closed.")
                p.status = PositionStatus.CLOSED.value
                p.exit_reason = reason + "_ZERO_QTY"
                self.store_actor.q.put({"type": "upsert_position", "pos": p})
                self.positions.pop(p.id, None) 
                return True

            # --- Place Market Exit Order via OrderActor ---
            place_params = {
                "variety": VARIETY_REGULAR, "exchange": "NFO", "tradingsymbol": p.tradingsymbol, 
                "transaction_type": TRANSACTION_TYPE_SELL, "quantity": p.qty, 
                "product": PRODUCT_MIS, "order_type": ORDER_TYPE_MARKET
            }
            
            L.info(f"Sending MARKET exit order request for {p.qty} of {p.tradingsymbol} ({p.id}) via OrderActor. Reason: {reason}.")
            reply_q = queue.Queue()
            self.order_actor.q.put({"type": "place_order", "params": place_params, "reply_q": reply_q})

            oid = None
            try:
                resp = reply_q.get(timeout=10.0) 
                if resp['ok'] and resp['res'] and 'order_id' in resp['res']:
                    oid = resp['res']['order_id']
                    L.info(f"Market exit order {oid} placed successfully via OrderActor for {p.tradingsymbol}.")
                else: raise Exception(resp.get('error', 'Unknown error placing market exit'))
            except queue.Empty: L.critical(f"Timeout waiting for MARKET exit order response for {p.tradingsymbol}!")
            except Exception as e: L.critical(f"MARKET EXIT ORDER FAILED for {p.tradingsymbol}. Error: {e}")

            if oid is None:
                send_alert(f"🔥 CRITICAL: FAILED TO PLACE MARKET EXIT for {p.tradingsymbol} ({p.id}). POSITION IS STILL OPEN. Manual intervention required!", "critical")
                p.exit_reason = reason + "_EXIT_API_FAIL" # Keep PENDING_CLOSURE
                self.store_actor.q.put({"type": "upsert_position", "pos": p})
                return False 
            else:
                p.exit_order_id = str(oid)
                # Status remains PENDING_CLOSURE until fill confirmation
                self.store_actor.q.put({"type": "upsert_position", "pos": p})
                return True 

    # --- Cancel Pending Entry (Refactored) ---
    def cancel_pending_entry(self, p: Position) -> bool:
        """Cancels a pending entry order via OrderActor."""
        with self.lock:
            if p.status != PositionStatus.PENDING_ENTRY.value or not p.entry_order_id:
                L.debug(f"cancel_pending_entry called for {p.tradingsymbol} but not PENDING_ENTRY or no entry_order_id.")
                return False

            L.warning(f"Cancelling pending entry order {p.entry_order_id} for {p.tradingsymbol}.")
            self._cancel_all_open_orders_for_pos(p, cancel_entry=True) # Uses actor
            
            p.status = PositionStatus.CLOSED.value
            p.exit_reason = "ENTRY_TIMEOUT_CANCELLED" 
            self.store_actor.q.put({"type": "upsert_position", "pos": p}) # Uses actor
            self.positions.pop(p.id, None) 
            return True

    # --- Scale Out (Refactored) ---
    def scale_out(self, p: Position, qty_to_close: int) -> bool:
        """Scales out by cancelling brackets and placing market order via Actors."""
        # Define constants locally or globally
        VARIETY_REGULAR = "regular"
        TRANSACTION_TYPE_SELL = "SELL"
        PRODUCT_MIS = "MIS"
        ORDER_TYPE_MARKET = "MARKET"
        
        with self.lock:
            if p.status not in [PositionStatus.ACTIVE.value, PositionStatus.PARTIALLY_CLOSED.value] or qty_to_close <= 0 or qty_to_close > p.qty:
                 if qty_to_close > p.qty: L.warning(f"Scale out failed for {p.tradingsymbol}: Cannot close {qty_to_close}, only {p.qty} left.")
                 return False

            L.info(f"Scaling out {qty_to_close} of {p.tradingsymbol}. Cancelling brackets.")
            self._cancel_all_open_orders_for_pos(p, cancel_entry=False) # Uses actor

            # Update status immediately
            p.status = PositionStatus.OPEN_AWAITING_BRACKETS 
            self.store_actor.q.put({"type": "upsert_position", "pos": p}) # Uses actor

            # --- Place Scale-Out Market Order via OrderActor ---
            place_params = {
                "variety": VARIETY_REGULAR, "exchange": "NFO", "tradingsymbol": p.tradingsymbol, 
                "transaction_type": TRANSACTION_TYPE_SELL, "quantity": qty_to_close, 
                "product": PRODUCT_MIS, "order_type": ORDER_TYPE_MARKET
            }
            
            L.info(f"Sending scale-out MARKET order for {qty_to_close} of {p.tradingsymbol} via OrderActor.")
            reply_q = queue.Queue()
            self.order_actor.q.put({"type": "place_order", "params": place_params, "reply_q": reply_q})

            oid = None
            try:
                resp = reply_q.get(timeout=10.0) 
                if resp['ok'] and resp['res'] and 'order_id' in resp['res']:
                    oid = resp['res']['order_id']
                    L.info(f"Scale-out market order {oid} placed successfully.")
                else: raise Exception(resp.get('error', 'Unknown error placing scale-out order'))
            except queue.Empty: L.critical(f"Timeout waiting for scale-out MARKET order response for {p.tradingsymbol}!")
            except Exception as e: L.critical(f"SCALE OUT MARKET ORDER FAILED for {p.tradingsymbol}. Error: {e}")

            if oid is None:
                send_alert(f"CRITICAL: SCALE OUT MARKET ORDER FAILED for {p.tradingsymbol}. Closing full position!", "critical")
                self.close_position(p, "SCALE_OUT_FAILURE_FULL_EXIT") # Uses actor
                return False 
            else:
                p.partial_exit_order_ids.append(str(oid))
                # Status remains OPEN_AWAITING_BRACKETS
                self.store_actor.q.put({"type": "upsert_position", "pos": p}) # Uses actor
                return True 

    # --- Modify SL (Refactored - Copied from previous correct version) ---
    def modify_sl(self, p: Position, new_trigger: float):
        """Modifies SL using OrderActor (Cancel-Modify-Replace logic)."""
        # Define constants locally or globally
        VARIETY_REGULAR = "regular"
        TRANSACTION_TYPE_SELL = "SELL"
        PRODUCT_MIS = "MIS"
        ORDER_TYPE_SLM = "SL-M"
        
        tick_size = self.book.tick_size(p.tradingsymbol)
        new_trigger_rounded = round(new_trigger / tick_size) * tick_size

        if new_trigger_rounded <= p.sl_price or abs(new_trigger_rounded - p.sl_price) < tick_size:
             return # No change or change too small
        
        old_sl = p.sl_price
        old_slm_id = p.slm_order_id

        # --- 1. Place new SL-M order first ---
        L.info(f"Attempting to trail SL for {p.tradingsymbol} to {new_trigger_rounded:.2f}. Placing new order...")
        place_params = {
            "variety": VARIETY_REGULAR, "exchange": "NFO", "tradingsymbol": p.tradingsymbol,
            "transaction_type": TRANSACTION_TYPE_SELL, "quantity": p.qty, 
            "product": PRODUCT_MIS, "order_type": ORDER_TYPE_SLM, 
            "trigger_price": new_trigger_rounded
        }
        
        reply_q = queue.Queue()
        self.order_actor.q.put({"type": "place_order", "params": place_params, "reply_q": reply_q})
        
        new_slm_id = None
        try:
            resp = reply_q.get(timeout=10.0) 
            if resp['ok'] and resp['res'] and 'order_id' in resp['res']:
                new_slm_id = str(resp['res']['order_id'])
            else: raise Exception(resp.get('error', 'Unknown error'))
        except Exception as e:
            L.error(f"Failed to place NEW SL-M order for {p.tradingsymbol} trail. Aborting trail. Old SL {old_slm_id} active. Error: {e}")
            return # Old SL remains active

        L.info(f"Successfully placed new SL order {new_slm_id} for {p.tradingsymbol}. Cancelling old order {old_slm_id}.")

        # --- 2. Cancel the old SL order ---
        cancel_success = True
        if old_slm_id:
            cancel_reply_q = queue.Queue()
            self.order_actor.q.put({
                "type": "cancel_order",
                "params": {"variety": VARIETY_REGULAR, "order_id": old_slm_id},
                "reply_q": cancel_reply_q
            })
            try:
                cancel_resp = cancel_reply_q.get(timeout=10.0)
                if not cancel_resp['ok']: raise Exception(cancel_resp.get('error', 'Unknown error'))
                L.info(f"Successfully cancelled old SL order {old_slm_id}.")
            except Exception as e:
                # --- !! CRITICAL FAILURE !! ---
                L.critical(f"!! DOUBLE SL DANGER for {p.tradingsymbol} !! Failed to cancel old SL {old_slm_id} after placing new SL {new_slm_id}. Error: {e}")
                send_alert(f"🔥 CRITICAL: DOUBLE SL {p.tradingsymbol}. Old SL {old_slm_id} FAILED TO CANCEL. New SL {new_slm_id} active.", "critical")
                cancel_success = False

                # Put position into SAFE MODE
                p.status = PositionStatus.PENDING_CLOSURE.value 
                p.exit_reason = "SAFE_MODE_DOUBLE_SL"
                p.sl_price = new_trigger_rounded # Store the *new* SL info
                p.slm_order_id = new_slm_id
                self.store_actor.q.put({"type": "upsert_position", "pos": p}) 
                return # Stop processing this position

        # --- 3. Success (or old order cancel failed but we proceed with new SL) ---
        if cancel_success: # Only update state if cancel succeeded or wasn't needed
            p.sl_price = new_trigger_rounded
            p.slm_order_id = new_slm_id
            self.store_actor.q.put({"type": "upsert_position", "pos": p})
            L.info(f"Trailed SL for {p.tradingsymbol} from {old_sl:.2f} to {new_trigger_rounded:.2f} (New ID: {new_slm_id})")

    # --- Execute Simulated SL (No Change - Only for PaperTrader) ---
    def execute_simulated_sl(self, p: Position) -> bool:
        L.critical("execute_simulated_sl called on LIVE trader. This should not happen.")
        return False

    # --- Handle Partial Exit Fill (Refactored) ---
    def _handle_partial_exit_fill(self, pos: Position, order: Dict):
        """Handles the logic for a partial scale-out order fill (called by order update worker)."""
        filled_qty = order.get('filled_quantity', 0)
        oid = str(order.get('order_id')) 

        # Remove the OID immediately upon receiving the update, regardless of fill qty
        removed_oid = False
        if oid in pos.partial_exit_order_ids:
            pos.partial_exit_order_ids.remove(oid)
            L.debug(f"Removed partial exit OID {oid} from position {pos.id} list: {pos.partial_exit_order_ids}")
            removed_oid = True
        else:
             L.warning(f"Partial exit OID {oid} (status: {order.get('status')}) not found in pos.partial_exit_order_ids for {pos.id}. Current list: {pos.partial_exit_order_ids}")

        if filled_qty == 0:
            L.warning(f"Partial exit order {oid} completed with 0 fills for {pos.tradingsymbol}. Ignoring PnL.")
            # If OID was removed and no more partials pending, try re-bracketing
            if removed_oid and not pos.partial_exit_order_ids and pos.qty > 0:
                 L.info(f"Zero-fill partial exit {oid} complete. Re-attempting bracket placement for remaining {pos.qty} qty.")
                 if pos.status == PositionStatus.OPEN_AWAITING_BRACKETS: # Check status before placing
                     if not self.place_bracket_orders(pos): 
                         send_alert(f"CRITICAL: FAILED to re-place brackets after zero-fill scale out for {pos.tradingsymbol}. Closing position.", "critical")
                         self.close_position(pos, "BRACKET_REPLACE_FAILURE_POST_SCALE")
                 else:
                      L.warning(f"Cannot re-place brackets after zero-fill scale out for {pos.id}, status is {pos.status}")
            # Ensure DB is updated even if no fill, as OID list changed
            elif removed_oid:
                 self.store_actor.q.put({"type": "upsert_position", "pos": pos})
            return

        L.info(f"💰 Partial exit (scale-out) fill received for {pos.tradingsymbol}. Order: {oid}, Qty: {filled_qty}.")

        avg_price = self._get_order_avg_price(oid) # Uses OrderActor
        if not avg_price:
            L.error(f"Could not get avg price for partial exit {oid}. PnL inaccurate. Using LTP fallback.")
            avg_price = self.prices.ltp(pos.token) or pos.entry_price 

        pnl = (avg_price - pos.entry_price) * filled_qty
        self.daily_realized_pnl += pnl
        
        self.store_actor.q.put({"type": "log_strategy_performance", "name": pos.strategy, "pnl": pnl})

        # --- Update Position State ---
        pos.qty -= filled_qty
        pos.scaled_out_qty += filled_qty

        # --- Determine Next Step ---
        if pos.qty <= 0:
            L.info(f"Scale-out {oid} ({filled_qty} qty) resulted in zero/negative qty ({pos.qty}) for {pos.tradingsymbol}. Closing fully.")
            pos.qty = 0 
            pos.exit_reason = "SCALE_OUT_FULL"
            self._handle_exit_fill(pos, order) # Uses actors
        else:
             if pos.partial_exit_order_ids: # More partials pending?
                  L.info(f"Partial exit {oid} complete. Awaiting {pos.partial_exit_order_ids}. Status remains {pos.status}.")
                  self.store_actor.q.put({"type": "upsert_position", "pos": pos}) # Update DB
             else: # Last partial exit, re-place brackets
                  L.info(f"Final partial exit {oid} complete. Re-placing brackets for remaining {pos.qty} qty.")
                  pos.status = PositionStatus.OPEN_AWAITING_BRACKETS 
                  self.store_actor.q.put({"type": "upsert_position", "pos": pos}) # Save before placing
                  
                  if not self.place_bracket_orders(pos): # Uses actors
                      send_alert(f"CRITICAL: FAILED to re-place brackets after scale out for {pos.tradingsymbol}. Closing remaining position!", "critical")
                      self.close_position(pos, "BRACKET_REPLACE_FAILURE_POST_SCALE")
                  
             self._log_partial_trade_actor(pos, avg_price, filled_qty, f"PARTIAL_SCALE_OUT_{oid}") 
             
             self._update_performance(pnl) 
             send_alert(f"💰 SCALED OUT {filled_qty} of {pos.tradingsymbol} @ {avg_price:.2f}. PnL: {pnl:.2f}. Daily PnL: {self.daily_realized_pnl:.2f}")

    # --- Log Partial Trade via Actor (Refactored) ---
    def _log_partial_trade_actor(self, p: Position, exit_price: float, qty: int, reason: str):
        """Sends a message to the StoreActor to log a partial trade."""
        pnl = (exit_price - p.entry_price) * qty
        trade_id = f"{p.id}_{reason}_{uuid.uuid4()}" 
        
        # Create a temporary 'partial position' object for logging
        try:
             # Use dataclasses.replace if Position is a dataclass
             partial_pos_log_data = dataclasses.replace(p, id=trade_id, initial_qty=qty, qty=qty) 
        except TypeError:
             # Manual creation if not a dataclass (adjust fields as needed)
             L.warning("Position object might not be a dataclass. Creating log data manually.")
             partial_pos_log_data = Position(
                id=trade_id, tradingsymbol=p.tradingsymbol, token=p.token, 
                option_type=p.option_type, qty=qty, initial_qty=qty, # Key change: use closed qty
                entry_price=p.entry_price, opened_at=p.opened_at, strategy=p.strategy, 
                market_regime_at_entry=p.market_regime_at_entry, 
                # Include other necessary fields, default others
                initial_sl_price=0, sl_price=0, tp_price=0, status="LOGGED_PARTIAL" 
             )

        self.store_actor.q.put({
            "type": "log_closed_trade",
            "pos": partial_pos_log_data, 
            "price": exit_price,
            "reason": reason 
        })
        L.debug(f"Sent partial trade log request to StoreActor for {trade_id} ({qty} qty).")

# ==================================================================================================
# MODULARIZED LOGIC (NEW CLASSES)
# ==================================================================================================

class RiskManager:
    """Handles all logic related to P&L, drawdown, position sizing, and portfolio greeks."""
    def __init__(self,
                 engine: 'Engine',
                 trader: AbstractTrader,
                 book: InstrumentBook,
                 prices: PriceBus,
                 store_actor: StoreActor,
                 config: Dict):
        self.strategy_weights: Dict[str, float] = {}
        self.engine = engine
        self.trader = trader
        self.book = book
        self.prices = prices
        self.store_actor = store_actor
        self.trading_config = config["trading"]
        self.technical_config = config["technical"]
        self.timings_config = config["timings"]
        self.lock = None
        self.lock = threading.RLock()

        self.portfolio_greeks: Dict[str, float] = {"net_delta": 0.0, "net_vega": 0.0, "net_gamma": 0.0, "net_theta": 0.0}
        
        self.consecutive_losses, self.risk_factor = 0, 1.0
        self.dynamic_account_equity = float(os.environ.get("ACCOUNT_EQUITY", self.trading_config.get("account_equity", 100000.0)))
        self.daily_high_water_mark = self.dynamic_account_equity
        self.performance_score = 0
        self.weekly_high_water_mark = self.dynamic_account_equity
        self.in_weekly_drawdown_lock = False
        self.last_unrealized_pnl = 0.0
        self.max_daily_drawdown_pct = self.trading_config["max_daily_drawdown_pct"]
        self.account_equity_base = self.dynamic_account_equity
        self.portfolio_limits = self.trading_config.get("portfolio_limits", {})
        
        self.strategy_weights: Dict[str, float] = {} # strategy_name -> weight (e.g., 1.0)
        self.strategy_perf_lookback_days = self.trading_config.get("strategy_perf_lookback_days", 7)
        self.strategy_weight_min = self.trading_config.get("strategy_weight_min", 0.5)
        self.strategy_weight_max = self.trading_config.get("strategy_weight_max", 1.5)

    def reset_daily_state(self, last_trading_day: Optional[date]):
        L.info("RiskManager resetting daily state.")
        now = now_ist()
        if last_trading_day and last_trading_day.weekday() > now.date().weekday():
            L.info("New week detected. Resetting weekly high water mark and drawdown lock.")
            self.weekly_high_water_mark = self.dynamic_account_equity
            self.in_weekly_drawdown_lock = False
            
            # --- FIXED: Use StoreActor ---
            self.store_actor.q.put({
                "type": "set_kv",
                "key": "weekly_hwm",
                "value": str(self.weekly_high_water_mark)
            })
            # --- END FIX ---

        if last_trading_day:
            # --- FIXED: Use StoreActor ---
            self.store_actor.q.put({
                "type": "set_kv",
                "key": f"daily_pnl_{last_trading_day}",
                "value": str(self.trader.daily_realized_pnl)
            })
            # --- END FIX ---

        self.update_dynamic_equity() # This function is now fixed to use actors
        self.trader.daily_realized_pnl = 0.0

        with self.lock:
            self.portfolio_greeks = {"net_delta": 0.0, "net_vega": 0.0, "net_gamma": 0.0, "net_theta": 0.0}

        self.consecutive_losses = 0
        self.risk_factor = 1.0
        self.daily_high_water_mark = self.dynamic_account_equity
        self.performance_score = 0

        L.info(f"RiskManager state reset. Equity: ₹{self.dynamic_account_equity:,.2f}, Daily HWM: {self.daily_high_water_mark:,.2f}")

    def load_persistent_state(self, trading_day: date):
        # --- FIXED: Use StoreActor (Blocking Read) ---
        weekly_hwm_str = str(self.dynamic_account_equity)
        pnl_str = "0.0"
        
        try:
            reply_q_hwm = queue.Queue()
            self.store_actor.q.put({
                "type": "get_kv",
                "key": "weekly_hwm",
                "default": str(self.dynamic_account_equity),
                "reply_q": reply_q_hwm
            })
            resp_hwm = reply_q_hwm.get(timeout=10.0)
            if resp_hwm['ok']:
                weekly_hwm_str = resp_hwm['res']

            reply_q_pnl = queue.Queue()
            self.store_actor.q.put({
                "type": "get_kv",
                "key": f"daily_pnl_{trading_day}",
                "default": "0.0",
                "reply_q": reply_q_pnl
            })
            resp_pnl = reply_q_pnl.get(timeout=10.0)
            if resp_pnl['ok']:
                pnl_str = resp_pnl['res']

        except queue.Empty:
            L.error("Timeout loading persistent state from StoreActor.")
        except Exception as e:
            L.error(f"Error loading persistent state from StoreActor: {e}")
        
        self.weekly_high_water_mark = float(weekly_hwm_str)
        self.trader.daily_realized_pnl = float(pnl_str)
        # --- END FIX ---

        self.daily_high_water_mark = self.dynamic_account_equity + self.trader.daily_realized_pnl
        self.weekly_high_water_mark = max(self.weekly_high_water_mark, self.daily_high_water_mark)
        L.info(f"RiskManager state loaded. Daily PnL: {self.trader.daily_realized_pnl}. Daily HWM: {self.daily_high_water_mark}. Weekly HWM: {self.weekly_high_water_mark}")
        
        
    def _update_strategy_weights(self):
        L.info("Updating dynamic strategy weights...")
        
        # --- FIXED: Use StoreActor (Blocking Read) ---
        df = pd.DataFrame() # Default to empty df
        try:
            reply_q = queue.Queue()
            self.store_actor.q.put({
                "type": "get_strategy_performance",
                "lookback_days": self.strategy_perf_lookback_days,
                "reply_q": reply_q
            })
            resp = reply_q.get(timeout=10.0)
            if resp['ok']:
                df = resp['res']
            else:
                L.error(f"Failed to get strategy performance from StoreActor: {resp.get('error')}")
        except queue.Empty:
            L.error("Timeout getting strategy performance from StoreActor.")
        except Exception as e:
            L.error(f"Error getting strategy performance from StoreActor: {e}")
        # --- END FIX ---

        if df.empty:
            L.warning("No strategy performance data found. Using default weights.")
            # Reset all known strategies to 1.0
            all_strategy_names = [name.value for name in StrategyName]
            self.strategy_weights = {name: 1.0 for name in all_strategy_names}
            return

        # Calculate sum of PnL per strategy
        strategy_pnl = df.groupby('strategy_name')['pnl'].sum()

        # Simple weighting: normalize PnL
        # You can make this much more complex (e.g., Sharpe, profit factor)
        
        # Avoid division by zero if all PnL is zero
        total_pnl_magnitude = abs(strategy_pnl).sum()
        if total_pnl_magnitude == 0:
            return # Keep existing weights

        # Scale weights based on PnL contribution.
        # This is a simple example; you might want a more robust metric.
        # This example scales from 0 to 1 based on PnL.
        pnl_min = strategy_pnl.min()
        pnl_max = strategy_pnl.max()
        
        if pnl_max == pnl_min:
            # All strategies have same PnL, set all to 1.0
            weights = pd.Series(1.0, index=strategy_pnl.index)
        else:
            # Normalize PnL from 0 to 1
            normalized_pnl = (strategy_pnl - pnl_min) / (pnl_max - pnl_min)
            # Scale from min_weight to max_weight
            weights = self.strategy_weight_min + (normalized_pnl * (self.strategy_weight_max - self.strategy_weight_min))

        self.strategy_weights = weights.to_dict()
        L.info(f"New strategy weights: {self.strategy_weights}")

    def calculate_position_size(self, underlying_token: int, risk_per_lot: float, vega_per_lot: float,confidence_score: float,strategy_name: str) -> int:
        if risk_per_lot <= 0:
            return 0
        risk_tiers = self.trading_config["risk_tiers"]

        is_expiry_day = self.book.find_nearest_expiry_date(_get_underlying(self.book.get_symbol(underlying_token))) == date.today()
        expiry_day_risk_factor = 1.0
        if is_expiry_day and self.trading_config.get("expiry_day_protocol_active", True):
            expiry_day_risk_factor = self.trading_config.get("expiry_day_risk_reduction_factor", 0.5)
            L.warning(f"EXPIRY DAY PROTOCOL: Applying risk reduction factor of {expiry_day_risk_factor}.")

        if self.in_weekly_drawdown_lock:
            active_risk_pct = risk_tiers["defensive"]
            L.warning("Weekly drawdown lock is active. Using DEFENSIVE risk tier.")
        elif self.performance_score <= -2:
            active_risk_pct = risk_tiers["defensive"]
        elif self.performance_score >= 2:
            active_risk_pct = risk_tiers["aggressive"]
        else:
            active_risk_pct = risk_tiers["standard"]

        df_1m = self.engine.get_ohlc(underlying_token, 1)
        if len(df_1m) < 21:
            return 1

        atr = df_1m.ta.atr(20).iloc[-1]
        spot = df_1m.iloc[-1]['close']
        vol_pct = atr / spot if spot > 0 else 0
        base_vol = 0.005
        vol_adjustment = min(1.5, max(0.5, base_vol / vol_pct if vol_pct > 0 else 1.0))

        vix_ltp = self.prices.ltp(self.engine.vix_token)
        vix_params = self.trading_config["vix_adjustment"]
        vix_risk_factor = 1.0
        if vix_ltp:
            if vix_ltp > vix_params["high_threshold"]:
                vix_risk_factor = vix_params["high_factor"]
            elif vix_ltp < vix_params["low_threshold"]:
                vix_risk_factor = vix_params["low_factor"]

        time_of_day_multiplier = self._get_time_of_day_risk_multiplier()

        chaos_risk_factor = 1.0
        is_agnostic = strategy_name in [s.name.value for s in self.engine.strategies.get("AGNOSTIC", [])]

        if self.engine.regime == Regime.CHAOS:
            if not is_agnostic:
                chaos_risk_factor = self.trading_config.get("chaos_risk_reduction_factor", 0.25)
                L.warning(f"CHAOS regime active. Applying risk reduction factor of {chaos_risk_factor} to non-agnostic strategy.")
            else:
                L.info(f"Sizing: Agnostic strategy ({strategy_name}) detected. Skipping CHAOS risk penalty.")

        final_risk_factor = self.risk_factor * vol_adjustment * vix_risk_factor * expiry_day_risk_factor * time_of_day_multiplier * chaos_risk_factor
        allowed_risk = self.dynamic_account_equity * (active_risk_pct / 100.0) * final_risk_factor

        slippage_factor = self.trading_config.get("sl_slippage_factor_pct", 20.0) / 100.0
        
        # --- NEW: Confluence Score Sizing (from config) ---
        standard_score = self.trading_config.get("standard_trade_score", 2.0)
        sizing_floor = self.trading_config.get("regime_confidence_sizing_floor", 0.5)
        sizing_cap = self.trading_config.get("regime_confidence_sizing_cap", 1.5)
        
        # Calculate raw multiplier based on signal's confluence score
        raw_multiplier = confidence_score / standard_score
        
        # Clamp the multiplier using the floor and cap from config.json
        bet_sizing_multiplier = max(sizing_floor, min(sizing_cap, raw_multiplier))
        
        L.info(f"Confluence Sizing: Score={confidence_score:.2f}, Raw Multi={raw_multiplier:.2f}, "
               f"Clamped Multi={bet_sizing_multiplier:.2f} (Floor: {sizing_floor}, Cap: {sizing_cap})")

        # --- NEW: Dynamic Strategy Weighting ---
        # (This implements Change #3)
        strategy_weight = self.strategy_weights.get(strategy_name, 1.0) # Default to 1.0
        L.info(f"Strategy Weighting: Name={strategy_name}, Weight={strategy_weight:.2f}")

        # --- Apply new multipliers to final_risk_factor ---
        final_risk_factor = (
            self.risk_factor * vol_adjustment * vix_risk_factor * expiry_day_risk_factor * time_of_day_multiplier * chaos_risk_factor *
            bet_sizing_multiplier * # <-- APPLIED
            strategy_weight          # <-- APPLIED
        )
        
        allowed_risk = self.dynamic_account_equity * (active_risk_pct / 100.0) * final_risk_factor
        actual_expected_risk_per_lot = risk_per_lot * (1.0 + slippage_factor)
        if actual_expected_risk_per_lot <= 0:
            L.warning("Position Size Calc: Actual expected risk per lot is zero or negative. Aborting.")
            return 0

        calculated_lots = int(math.floor(allowed_risk / actual_expected_risk_per_lot))
        L.info(f"Position Size Calc: PerfScore={self.performance_score}, RiskTier={active_risk_pct}%, AllowedRisk={allowed_risk:.2f}, Risk/Lot (adj): {actual_expected_risk_per_lot:.2f}, Lots={calculated_lots}")
        return max(0, min(calculated_lots, self.trading_config['max_lots_per_trade']))

    def risk_ok(self, hypothetical_params: Dict) -> bool:
        with self.engine.master_lock:
            if self.engine.master_halt:
                return False

            lot_size = self.book.lot_size(_get_underlying(hypothetical_params['opt']['tradingsymbol']))
            if not lot_size:
                return False

            with self.lock:
                hypothetical_qty = hypothetical_params['lots'] * lot_size
                post_trade_delta = self.portfolio_greeks["net_delta"] + (hypothetical_params['greeks']['delta'] * hypothetical_qty)
                post_trade_vega = self.portfolio_greeks["net_vega"] + (hypothetical_params['greeks']['vega'] * hypothetical_qty)
                max_delta = self.portfolio_limits.get("max_portfolio_net_delta")
                max_vega = self.portfolio_limits.get("max_portfolio_net_vega")

                if max_delta and abs(post_trade_delta) > max_delta:
                    L.warning(f"Trade REJECTED: Breach max delta. Post-trade: {post_trade_delta:.0f}, Limit: {max_delta}")
                    return False
                if max_vega and post_trade_vega > max_vega:
                    L.warning(f"Trade REJECTED: Breach max vega. Post-trade: {post_trade_vega:.0f}, Limit: {max_vega}")
                    return False
                post_trade_theta = self.portfolio_greeks["net_theta"] + (hypothetical_params['greeks']['theta'] * hypothetical_qty)
                max_neg_theta = self.portfolio_limits.get("max_portfolio_negative_theta")
                
                if max_neg_theta and post_trade_theta < -max_neg_theta:
                    L.warning(f"Trade REJECTED: Breach max negative theta. Post-trade: {post_trade_theta:.0f}, Limit: {-max_neg_theta}")
                    return False

            current_equity = self.dynamic_account_equity + self.trader.daily_realized_pnl + self.last_unrealized_pnl

            # Weekly Drawdown Check
            weekly_dd_limit_pct = self.trading_config.get("weekly_drawdown_pct_limit", 5.0)
            weekly_dd_limit_abs = self.weekly_high_water_mark * (weekly_dd_limit_pct / 100.0)
            weekly_floor = self.weekly_high_water_mark - weekly_dd_limit_abs

            if current_equity < weekly_floor:
                if not self.in_weekly_drawdown_lock:
                    send_alert(f"⛔ WEEKLY DD LIMIT HIT. Peak: {self.weekly_high_water_mark:.2f}, Current: {current_equity:.2f} (Floor: {weekly_floor:.2f}). Entering DEFENSIVE mode.", "critical")
                    self.in_weekly_drawdown_lock = True

            # Daily Drawdown Check
            realized_pnl = self.trader.daily_realized_pnl
            profit_lock_floor = -float('inf')

            profit_lock_config = self.trading_config.get('profit_lock_in', {})
            if realized_pnl > profit_lock_config.get('min_profit_trigger', float('inf')):
                profit_at_risk_pct = profit_lock_config.get("pct_to_risk", 40.0) / 100.0
                profit_at_risk = realized_pnl * profit_at_risk_pct
                profit_lock_floor = self.daily_high_water_mark - profit_at_risk

            static_dd_limit = self.daily_high_water_mark * (self.max_daily_drawdown_pct / 100.0)
            static_floor = self.daily_high_water_mark - static_dd_limit

            final_floor = max(static_floor, profit_lock_floor)

            daily_dd_pct = (self.daily_high_water_mark - current_equity) / self.daily_high_water_mark * 100 if self.daily_high_water_mark > 0 else 0
            if G_DAILY_DRAWDOWN_PCT:
                G_DAILY_DRAWDOWN_PCT.set(daily_dd_pct)

            if current_equity < final_floor:
                send_alert(f"⛔ DD LIMIT HIT. HALTING. Peak Equity: {self.daily_high_water_mark:.2f}, Current: {current_equity:.2f} (Floor: {final_floor:.2f})", "critical")
                self.engine.master_halt = True
                if G_HALTED_STATUS:
                    G_HALTED_STATUS.set(1)

                with self.trader.lock:
                    positions_to_close = [p for p in list(self.trader.positions.values()) if p.status not in [PositionStatus.CLOSED.value, PositionStatus.PENDING_CLOSURE.value]]
                for p in positions_to_close:
                    self.trader.close_position(p, "DD_LIMIT_HIT")
                return False
            return True

    def update_performance_metrics(self, pnl: float):
        with self.engine.master_lock:
            current_equity = self.dynamic_account_equity + self.trader.daily_realized_pnl + self.last_unrealized_pnl
            self.daily_high_water_mark = max(self.daily_high_water_mark, current_equity)
            self.weekly_high_water_mark = max(self.weekly_high_water_mark, self.daily_high_water_mark)

            if pnl > 0:
                self.performance_score = min(4, self.performance_score + 1)
            else:
                self.performance_score = max(-4, self.performance_score - 2)

            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

            loss_streak_config = self.trading_config["consecutive_loss_adjustment"]
            self.risk_factor = max(loss_streak_config["min_factor"], 1.0 - loss_streak_config["reduction_per_loss"] * self.consecutive_losses)

            L.info(f"PnL: {pnl:.2f}, PerfScore: {self.performance_score}, ConsecLosses: {self.consecutive_losses}, RiskFactor: {self.risk_factor:.2f}")
            self.store_actor.q.put({
                "type": "set_kv", 
                "key": f"daily_pnl_{self.engine.last_trading_day}", 
                "value": str(self.trader.daily_realized_pnl)
            })
            self.store_actor.q.put({
                "type": "set_kv", 
                "key": "weekly_hwm", 
                "value": str(self.weekly_high_water_mark)
            })

    def update_dynamic_equity(self):
        if PAPER_TRADING:
            self.dynamic_account_equity = self.account_equity_base + self.trader.daily_realized_pnl
            L.info(f"Paper equity updated to: {self.dynamic_account_equity:,.2f}")
            return
        
        # --- FIXED: Use OrderActor (Blocking Read) ---
        try:
            reply_q = queue.Queue()
            # We access the order_actor via engine.trader
            self.engine.trader.order_actor.q.put({
                "type": "margins",
                "reply_q": reply_q
            })
            
            margins = None
            resp = reply_q.get(timeout=10.0)
            if resp['ok']:
                margins = resp['res']
            else:
                L.error(f"Failed to get margins from OrderActor: {resp.get('error')}")

            if margins and 'equity' in margins and margins['equity'].get('net'):
                self.dynamic_account_equity = float(margins['equity']['net'])
                L.info(f"Dynamic account equity updated to: {self.dynamic_account_equity:,.2f}")
        
        except queue.Empty:
            L.warning("Timeout updating dynamic account equity.")
        except Exception as e:
            L.warning(f"Could not update dynamic account equity: {e}")
        # --- END FIX ---

    def reconcile_broker_pnl(self):
        if PAPER_TRADING:
            return
        
        # --- FIXED: Use OrderActor (Blocking Read) ---
        try:
            reply_q = queue.Queue()
            self.engine.trader.order_actor.q.put({
                "type": "positions",
                "reply_q": reply_q
            })

            broker_positions = None
            resp = reply_q.get(timeout=10.0)
            if resp['ok']:
                broker_positions = resp['res']
            else:
                L.error(f"Failed to get positions from OrderActor: {resp.get('error')}")
                return

            if not broker_positions or 'net' not in broker_positions:
                return

            broker_unrealized_pnl = sum(pos.get('unrealised', 0) for pos in broker_positions['net'] if pos.get('product') == 'MIS')
            bot_unrealized_pnl = self.trader.unrealized_pnl()

            discrepancy = abs(broker_unrealized_pnl - bot_unrealized_pnl)
            threshold = self.trading_config.get('pnl_discrepancy_alert_threshold_rupees', 250.0)
            critical_threshold = self.trading_config.get('pnl_discrepancy_critical_threshold', 1000.0)

            
            if discrepancy > critical_threshold:
                send_alert(f"🔥🔥 FATAL P&L DISCREPANCY: ₹{discrepancy:.2f}. Halting bot.", "critical")
                self.engine.fatal_error_event.set() # Trigger engine shutdown
            elif discrepancy > threshold:
                send_alert(
                    f"⚠️ P&L DISCREPANCY DETECTED! "
                    f"Broker Unrealized: ₹{broker_unrealized_pnl:.2f}, "
                    f"Bot Unrealized: ₹{bot_unrealized_pnl:.2f}, "
                    f"Difference: ₹{discrepancy:.2f}",
                    "warning"
                )
        except queue.Empty:
            L.warning("Timeout reconciling broker P&L.") 
        except Exception as e:
            L.warning(f"Could not reconcile broker P&L: {e}")

    def update_pnl_metrics(self):
        self.last_unrealized_pnl = self.trader.unrealized_pnl()

    def initialize_position_greeks(self, p: Position):
        with self.lock:
            if not p.greeks:
                L.warning(f"Could not find pre-calculated greeks for {p.tradingsymbol}. This should not happen.")
                return

            self.portfolio_greeks["net_delta"] += p.greeks.get("delta", 0.0) * p.initial_qty
            self.portfolio_greeks["net_vega"] += p.greeks.get("vega", 0.0) * p.initial_qty
            self.portfolio_greeks["net_gamma"] += p.greeks.get("gamma", 0.0) * p.initial_qty
            self.portfolio_greeks["net_theta"] += p.greeks.get("theta", 0.0) * p.initial_qty
            L.info(f"Initialized greeks for {p.tradingsymbol}: {p.greeks}. Portfolio totals: {self.portfolio_greeks}")

    def update_position_greeks(self, p: Position, ltp: float):
        with self.lock:
            old_greeks = p.greeks.copy()

            underlying_name = _get_underlying(p.tradingsymbol)
            underlying_token = self.engine.bn_token if "BANKNIFTY" in underlying_name else self.engine.nifty_token
            spot = self.prices.ltp(underlying_token)
            if not spot:
                return

            opt_details = self.book.df_by_token.loc[p.token]
            T = _calculate_time_to_expiry(opt_details['expiry'].date(), now_ist(), self.timings_config["market_close"])

            hv = calculate_historical_volatility(self.engine.get_ohlc(underlying_token, 1)['close'])
            cached_iv = self.engine.atm_iv_cache.get(underlying_name, p.greeks.get('iv', 0.5))
            if not cached_iv:
                cached_iv = hv or 0.5 # Use HV or default if cache is still empty
                
            is_call = (p.option_type == 'CE') # Add this line
            iv = calculate_iv(ltp, spot, opt_details['strike'], T, 0.05, is_call, initial_guess=cached_iv, hv_fallback=hv)
            p.greeks = calculate_greeks(spot, opt_details['strike'], T, 0.05, iv, is_call)
            p.greeks['iv'] = iv  # Store calculated IV for next iteration's guess

            for key in ["delta", "vega", "gamma", "theta"]:
                change = p.greeks.get(key, 0.0) - old_greeks.get(key, 0.0)
                self.portfolio_greeks[f"net_{key}"] += change * p.qty

    def verify_position_risk(self, p: Position):
        L.info(f"Post-fill verification for {p.tradingsymbol}...")

        underlying_name = _get_underlying(p.tradingsymbol)
        underlying_token = self.engine.bn_token if "BANKNIFTY" in underlying_name else self.engine.nifty_token
        spot = self.prices.ltp(underlying_token)
        if not spot:
            return

        opt_details = self.book.df_by_token.loc[p.token]
        T = _calculate_time_to_expiry(opt_details['expiry'].date(), now_ist(), self.timings_config["market_close"])
        hv = calculate_historical_volatility(self.engine.get_ohlc(underlying_token, 1)['close'])
        iv = calculate_iv(p.entry_price, spot, opt_details['strike'], T, 0.05, p.option_type == 'CE', hv_fallback=hv)
        greeks = calculate_greeks(spot, opt_details['strike'], T, 0.05, iv, p.option_type == 'CE')

        actual_risk_rupees = p.option_sl_points * p.initial_qty
        risk_tolerance_factor = 1.4

        if actual_risk_rupees > (p.intended_risk_rupees * risk_tolerance_factor):
            L.critical(f"RISK BREACH POST-FILL on {p.tradingsymbol}! "
                       f"Intended Risk: ₹{p.intended_risk_rupees:.2f}, "
                       f"Actual Risk: ₹{actual_risk_rupees:.2f}. EXITING POSITION.")
            send_alert(f"🔥 RISK BREACH POST-FILL: {p.tradingsymbol}. Exiting immediately.", "critical")
            self.trader.close_position(p, "POST_FILL_RISK_BREACH")

    def _get_time_of_day_risk_multiplier(self) -> float:
        now_time = now_ist().time()
        time_config = self.trading_config.get("time_of_day_risk", {})

        open_start, open_end = dtime.fromisoformat(time_config.get("opening_range", "09:15:00")), dtime.fromisoformat(time_config.get("opening_end", "10:30:00"))
        midday_start, midday_end = dtime.fromisoformat(time_config.get("midday_range", "11:30:00")), dtime.fromisoformat(time_config.get("midday_end", "13:30:00"))

        if open_start <= now_time < open_end:
            return time_config.get("opening_multiplier", 1.25)
        elif midday_start <= now_time < midday_end:
            return time_config.get("midday_multiplier", 0.5)
        else:
            return time_config.get("default_multiplier", 1.0)


class MicrostructureMonitor:
    """Handles TFI, OBI, and other order book/tick-level analysis."""
    def __init__(self, prices: PriceBus, config: Dict):
        self.prices = prices
        self.technical_config = config["technical"]
        self.micro_config = config["technical"].get("microstructure", {})
        self.lock = None
        self.lock = threading.RLock()

        self.tfi_window = self.technical_config.get("tfi_window", 50)
        self.persistence_window = self.micro_config.get("persistence_window", 5)

        self.recent_trades: Dict[int, deque] = {}
        self.tfi_scores: Dict[int, float] = {}
        self.tfi_score_history: Dict[int, deque] = {}
        self.obi_ratio_history: Dict[int, deque] = {}

    def update_tfi_score(self, tick: dict):
        token = tick.get('instrument_token')
        price = tick.get('last_price')
        qty = tick.get('last_traded_quantity')
        depth = tick.get('depth')

        if not all([token, price, qty, depth]):
            return
        if not depth.get('buy') or not depth.get('sell'):
            return

        if token not in self.recent_trades:
            self.recent_trades[token] = deque(maxlen=self.tfi_window)
            self.tfi_scores[token] = 0.0
            self.tfi_score_history[token] = deque(maxlen=self.persistence_window)

        bid_price = depth['buy'][0]['price']
        ask_price = depth['sell'][0]['price']
        trade_value = 0

        if price >= ask_price:
            trade_value = qty
        elif price <= bid_price:
            trade_value = -qty

        if trade_value != 0:
            self.recent_trades[token].append(trade_value)
            self.tfi_scores[token] = sum(self.recent_trades[token])
            self.tfi_score_history[token].append(self.tfi_scores[token])

    def check_tfi(self, token: int, side: OrderSide) -> int:
        """Checks for persistent TFI pressure.
        Returns: 1 (Confirm), 0 (Neutral), -1 (Conflict)
        """
        history = self.tfi_score_history.get(token)

        if not history or len(history) < self.persistence_window:
            return 0  # Not enough data for a persistent signal

        mean_score = np.mean(list(history))
        is_increasing = history[-1] >= history[0]  # Pressure is stable or increasing

        threshold = self.micro_config.get("tfi_persistence_mean_threshold", 400)

        if side == OrderSide.BUY:
            if mean_score > threshold and is_increasing:
                L.info(f"Persistent TFI confirmed BUY for {token}. Mean Score: {mean_score:.0f}")
                return 1
            if mean_score < -threshold:
                L.warning(f"Persistent TFI CONFLICTS (SELL) for {token} on BUY signal.")
                return -1

        if side == OrderSide.SELL:
            if mean_score < -threshold and is_increasing:
                L.info(f"Persistent TFI confirmed SELL for {token}. Mean Score: {mean_score:.0f}")
                return 1
            if mean_score > threshold:
                L.warning(f"Persistent TFI CONFLICTS (BUY) for {token} on SELL signal.")
                return -1

        return 0  # Neutral

    def check_order_book_imbalance(self, token: int, side: OrderSide) -> int:
        """Checks for persistent Order Book Imbalance.
        Returns: 1 (Confirm), 0 (Neutral), -1 (Conflict)
        """
        full_tick = self.prices.get_full_tick(token)
        if not full_tick or not full_tick.get('depth'):
            return 0  # Neutral

        depth = full_tick['depth']
        bids = depth.get('buy', [])
        asks = depth.get('sell', [])

        if not bids or not asks:
            return 0  # Neutral

        total_bid_qty = sum(item['quantity'] for item in bids[:20])
        total_ask_qty = sum(item['quantity'] for item in asks[:20])

        if (total_bid_qty + total_ask_qty) == 0:
            return 0  # Neutral

        obi_ratio = total_bid_qty / (total_bid_qty + total_ask_qty)

        if token not in self.obi_ratio_history:
            self.obi_ratio_history[token] = deque(maxlen=self.persistence_window)
        self.obi_ratio_history[token].append(obi_ratio)

        history = self.obi_ratio_history.get(token)
        if not history or len(history) < self.persistence_window:
            return 0  # Neutral

        mean_ratio = np.mean(list(history))
        is_increasing = history[-1] >= history[0]

        threshold = self.micro_config.get("obi_persistence_threshold_pct", 65.0) / 100.0

        if side == OrderSide.BUY:
            if mean_ratio > threshold and is_increasing:
                L.info(f"Persistent OBI confirmed BUY for {token}. Mean Ratio: {mean_ratio:.2f}")
                return 1
            if (1 - mean_ratio) > threshold:
                L.warning(f"Persistent OBI CONFLICTS (SELL) for {token} on BUY signal.")
                return -1

        if side == OrderSide.SELL:
            if (1 - mean_ratio) > threshold and is_increasing:
                L.info(f"Persistent OBI confirmed SELL for {token}. (Ask dominance: {1-mean_ratio:.2f})")
                return 1
            if mean_ratio > threshold:
                L.warning(f"Persistent OBI CONFLICTS (BUY) for {token} on SELL signal.")
                return -1

        return 0  # Neutral


import queue # <--- MAKE SURE THIS IMPORT IS AT THE TOP OF YOUR FILE

# ... (other classes) ...

class PositionManager:
    """Handles the high-frequency loop for managing all active positions."""
    def __init__(self,
                 engine: 'Engine',
                 trader: AbstractTrader,
                 book: InstrumentBook,
                 prices: PriceBus,
                 store_actor: StoreActor, # Now correctly injected
                 risk_manager: RiskManager,
                 config: Dict):
        self.engine = engine
        self.trader = trader
        self.book = book
        self.prices = prices
        self.store_actor = store_actor # Store the actor
        self.risk_manager = risk_manager
        self.trading_config = config["trading"]
        self.timings_config = config["timings"]
        self.lock = None
        self.lock = threading.RLock()

        self.position_price_history: Dict[str, deque] = {}
        self.last_greeks_update_per_pos: Dict[str, datetime] = {}
        self.trailing_sl_config = self.trading_config.get("trailing_sl", {})
        self.trade_mgmt_config = self.trading_config.get("trade_management", {})

    def manage_positions(self):
        with self.trader.lock:
            # Create a copy to avoid modification during iteration issues
            active_positions = list(self.trader.positions.values())

        now = now_ist()
        if PAPER_TRADING:
            # PaperTrader._manage_pending_entries is safe, uses store_actor
            self.trader._manage_pending_entries(now)

        for p in active_positions:
            # --- Check if position still exists locally (might be closed by another thread) ---
            with self.trader.lock:
                if p.id not in self.trader.positions:
                    L.debug(f"Position {p.id} ({p.tradingsymbol}) no longer in active list, skipping management.")
                    continue

            # --- Handle Pending Entry ---
            if p.status == PositionStatus.PENDING_ENTRY.value:
                # Calls _manage_adaptive_entry which is now fixed
                self._manage_adaptive_entry(p, now)
                continue # Move to next position

            # --- Handle Placing Brackets ---
            if p.status == PositionStatus.OPEN_AWAITING_BRACKETS.value:
                # trader.place_bracket_orders is already refactored
                if not self.trader.place_bracket_orders(p):
                    send_alert(f"CRITICAL: FAILED to place brackets for {p.tradingsymbol}. Closing position.", "critical")
                    # trader.close_position is already refactored
                    if not self.trader.close_position(p, "BRACKET_PLACEMENT_FAILURE"):
                        send_alert(f"🔥 FATAL: UNABLE TO CLOSE {p.tradingsymbol}. HALTING ALL TRADING.", "critical")
                        self.engine.fatal_error_event.set()
                continue # Move to next position

            # --- Handle Pending SL-L Fallback ---
            if p.status == PositionStatus.PENDING_SL_EXIT.value:
                ltp = self.prices.ltp(p.token)
                if ltp and ltp < (p.sl_price * 0.99):
                    L.warning(f"SL-L for {p.tradingsymbol} likely missed (LTP: {ltp}, SL:{p.sl_price}). Firing MARKET order.")
                    # trader.close_position is already refactored
                    self.trader.close_position(p, "SL_L_MISSED_MK_FALLBACK")
                continue # Move to next position

            # --- Skip non-active positions ---
            if p.status not in [PositionStatus.ACTIVE.value, PositionStatus.PARTIALLY_CLOSED.value]:
                continue

            # --- Main Active Position Management ---
            ltp = self.prices.ltp(p.token)
            if not ltp:
                continue # Skip if no LTP available

            # Throttled Greek Update (safe, no I/O)
            GREEKS_UPDATE_INTERVAL_SECONDS = 10
            last_update = self.last_greeks_update_per_pos.get(p.id)
            if not last_update or (now - last_update).total_seconds() > GREEKS_UPDATE_INTERVAL_SECONDS:
                L.debug(f"Updating greeks for {p.tradingsymbol}")
                # risk_manager.update_position_greeks is safe, uses prices/book/engine.get_ohlc
                self.risk_manager.update_position_greeks(p, ltp)
                self.last_greeks_update_per_pos[p.id] = now

            # Price History for Velocity (safe, no I/O)
            if p.id not in self.position_price_history:
                self.position_price_history[p.id] = deque(maxlen=20)
            self.position_price_history[p.id].append((now, ltp))

            # Velocity Trigger Check (safe, no I/O)
            if self._check_velocity_trigger(p, now):
                send_alert(f"⛔ VELOCITY TRIGGER on {p.tradingsymbol}! Pre-emptive exit.", "warning")
                # trader.close_position is already refactored
                self.trader.close_position(p, "VELOCITY_TRIGGER_EXIT")
                continue # Move to next position

            # Time Stop Check (safe, no I/O)
            if (now - p.opened_at).total_seconds() / 60 > p.max_trade_duration_minutes:
                L.info(f"Position {p.tradingsymbol} hit time-stop of {p.max_trade_duration_minutes} mins. Closing.")
                # trader.close_position is already refactored
                self.trader.close_position(p, "TIME_STOP_EXIT")
                continue # Move to next position

            # Underlying SL Check (safe, uses prices)
            underlying_name = _get_underlying(p.tradingsymbol)
            underlying_token = self.engine.bn_token if "BANKNIFTY" in underlying_name else self.engine.nifty_token
            underlying_price = self.prices.ltp(underlying_token)
            if underlying_price and p.underlying_sl_level:
                if (p.option_type == 'CE' and underlying_price <= p.underlying_sl_level) or \
                   (p.option_type == 'PE' and underlying_price >= p.underlying_sl_level):
                    L.warning(f"UNDERLYING SL HIT for {p.tradingsymbol}. Underlying: {underlying_price:.2f}, SL: {p.underlying_sl_level:.2f}. Closing.")
                    # trader.close_position is already refactored
                    self.trader.close_position(p, "UNDERLYING_SL_HIT")
                    continue # Move to next position

            # Option Price SL Check (local backup)
            if ltp <= p.sl_price:
                if PAPER_TRADING:
                    # trader.execute_simulated_sl uses close_position which uses actors
                    self.trader.execute_simulated_sl(p)
                else:
                    L.warning(f"OPTION PRICE SL HIT for {p.tradingsymbol} (LTP: {ltp}, SL: {p.sl_price}). "
                              f"Broker SL-M {p.slm_order_id} should execute.")
                continue # Move to next position

            # Update High Water Mark (safe, local state)
            with self.trader.lock:
                 # Need lock here if trader might modify pos outside this loop
                 p.high_price_since_entry = max(p.high_price_since_entry, ltp)

            # Scale Out Logic (safe, trader.scale_out is refactored)
            profit_points = ltp - p.entry_price
            scaled_out_this_cycle = False
            with self.trader.lock: # Lock needed if trader might modify pos outside this loop
                for rule in p.scale_out_rules:
                    target = rule['rr_target']
                    if target not in p.triggered_scale_out_targets and profit_points >= p.initial_risk_points * target:
                        qty_to_close = int(p.initial_qty * (rule['pct_to_close'] / 100.0))
                        # trader.scale_out uses actors internally
                        if self.trader.scale_out(p, qty_to_close):
                            p.triggered_scale_out_targets.append(target)
                            scaled_out_this_cycle = True
                            break # Only one scale-out per cycle

            if scaled_out_this_cycle:
                continue # Let next cycle handle TSL after scale-out state settles

            # Trailing Stop Loss Logic (safe, trader.modify_sl is refactored)
            with self.trader.lock: # Lock needed if trader might modify pos outside this loop
                current_rr = profit_points / p.initial_risk_points if p.initial_risk_points > 0 else 0
                highest_sl_floor = p.initial_sl_price # Start with initial SL

                # --- Static RR-based Trailing ---
                trailing_stages = self.trade_mgmt_config.get("trailing_stop_stages", [])
                for stage in trailing_stages:
                    if current_rr >= stage['rr_target']:
                        new_floor = p.entry_price + (p.initial_risk_points * stage['trail_behind_rr'])
                        highest_sl_floor = max(highest_sl_floor, new_floor)

                final_new_sl = highest_sl_floor

                # --- Dynamic Chandelier Trailing ---
                # Check if TSL is armed
                if not p.trailing_sl_armed and profit_points >= p.initial_risk_points * self.trading_config['trailing_sl_activation_rr']:
                    p.trailing_sl_armed = True
                    L.info(f"Chandelier Trailing SL armed for {p.tradingsymbol} after reaching {self.trading_config['trailing_sl_activation_rr']}R.")

                # Calculate Chandelier SL if armed
                if p.trailing_sl_armed:
                    # _calculate_trailing_stop is safe, uses engine.get_ohlc
                    if calculated_chandelier_sl := self._calculate_trailing_stop(p):
                        final_new_sl = max(final_new_sl, calculated_chandelier_sl) # Take the higher of static or dynamic

                # --- Apply the Trail ---
                if final_new_sl > p.sl_price:
                    is_now_risk_free = final_new_sl >= p.entry_price

                    # Cancel static TP if TSL becomes profitable (uses actor)
                    if is_now_risk_free and p.tp_order_id and not PAPER_TRADING:
                        L.info(f"TSL for {p.tradingsymbol} is now profitable. Cancelling static TP {p.tp_order_id}.")
                        # --- FIXED: Use OrderActor (Non-blocking) ---
                        # Use engine.trader to access the actor
                        self.engine.trader.order_actor.q.put({
                            "type": "cancel_order",
                            "params": {"variety": "regular", "order_id": str(p.tp_order_id)},
                            "reply_q": None # Fire-and-forget
                        })
                        p.tp_order_id = None # Clear local ID immediately
                        # --- END FIX ---

                    # trader.modify_sl uses actors internally
                    self.trader.modify_sl(p, final_new_sl)

            # --- Final DB Update ---
            # Update position in DB via actor (non-blocking)
            self.store_actor.q.put({"type": "upsert_position", "pos": p})

    def _calculate_trailing_stop(self, p: Position) -> Optional[float]:
        # This function is safe. Uses engine.get_ohlc which reads from memory.
        try:
            trail_params = self.trailing_sl_config
            trail_tf = trail_params["timeframe_scaled_out"] if p.triggered_scale_out_targets else trail_params["timeframe"]
            df = self.engine.get_ohlc(p.token, trail_tf)

            period = trail_params["chandelier_period"]
            if len(df) < period:
                return None

            atr = df.ta.atr(length=period).iloc[-1]
            if pd.isna(atr):
                return None

            multiplier = trail_params["chandelier_multiplier_scaled_out"] if p.triggered_scale_out_targets else trail_params["chandelier_multiplier"]

            high_over_period = df['high'].rolling(period).max().iloc[-1]
            new_sl_price = high_over_period - atr * multiplier

            return new_sl_price
        except Exception as e:
            L.warning(f"Could not calculate trailing stop for {p.tradingsymbol}: {e}")
            return None

    def _manage_adaptive_entry(self, p: Position, now: datetime):
        # This function needs fixing as it uses self.trader.k and self.store
        if not p.last_entry_modification:
            return

        time_since_mod = (now - p.last_entry_modification).total_seconds()

        full_tick = self.prices.get_full_tick(p.token)
        if not full_tick or not full_tick.get('depth'):
            return

        depth = full_tick['depth']
        if not depth.get('buy') or not depth.get('sell'):
            return

        bid_price = depth['buy'][0]['price']
        ask_price = depth['sell'][0]['price']
        mid_price = (bid_price + ask_price) / 2.0
        tick_size = self.book.tick_size(p.tradingsymbol)
        mid_price = round(mid_price / tick_size) * tick_size

        new_price = -1.0
        target_stage = p.entry_stage # Track target stage

        # Determine if stage needs to change
        if p.entry_stage == 1 and time_since_mod > self.trading_config['adaptive_entry_stage2_ms'] / 1000.0:
            L.info(f"Adaptive Entry Stage 2 (Neutral) for {p.tradingsymbol}")
            new_price = mid_price
            target_stage = 2
        elif p.entry_stage == 2 and time_since_mod > self.trading_config['adaptive_entry_stage3_ms'] / 1000.0:
            L.info(f"Adaptive Entry Stage 3 (Aggressive) for {p.tradingsymbol}")
            new_price = ask_price
            target_stage = 3

        # Check for timeout ONLY in the final stage
        elif p.entry_stage == 3:
             entry_timeout_config = (self.trading_config['adaptive_entry_stage2_ms'] +
                                     self.trading_config['adaptive_entry_stage3_ms'] +
                                     3000) # Total time + buffer in ms
             entry_timeout_sec = entry_timeout_config / 1000.0
             time_since_open = (now - p.opened_at).total_seconds() # Time since SUBMISSION

             if time_since_open > entry_timeout_sec:
                 L.warning(f"Adaptive Entry TIMEOUT for {p.entry_order_id} ({p.tradingsymbol}) after {time_since_open:.1f}s. Cancelling.")
                 # trader.cancel_pending_entry uses actors
                 self.trader.cancel_pending_entry(p)
                 return # Stop processing this position

        # If a stage change requires a modification
        if new_price > 0:
            # --- FIXED: Use OrderActor (Blocking Modify) ---
            modify_params = {
                "variety": "regular", # Assuming regular variety
                "order_id": str(p.entry_order_id),
                "price": new_price
            }
            reply_q = queue.Queue()
            # Use engine.trader to access actor
            self.engine.trader.order_actor.q.put({
                "type": "modify_order",
                "params": modify_params,
                "reply_q": reply_q
            })

            modify_success = False
            try:
                resp = reply_q.get(timeout=10.0)
                if resp['ok']:
                    modify_success = True
                else:
                    L.error(f"Failed to modify entry order {p.entry_order_id}: {resp.get('error')}")
            except queue.Empty:
                L.error(f"Timeout modifying entry order {p.entry_order_id}.")
            # --- END FIX ---

            if modify_success:
                p.last_entry_modification = now
                p.entry_stage = target_stage # Update stage only on success
                # --- FIXED: Use StoreActor (Non-blocking Update) ---
                self.store_actor.q.put({"type": "upsert_position", "pos": p})
                # --- END FIX ---
                L.info(f"Successfully modified entry order {p.entry_order_id} to price {new_price} (Stage {p.entry_stage})")
            else:
                L.error(f"Failed to modify entry order {p.entry_order_id} for adaptive entry. Order may be stuck.")
                # Consider if you need error handling here, e.g., cancelling the order

    def _check_velocity_trigger(self, p: Position, now: datetime) -> bool:
        # This function is safe. Reads from memory. No I/O.
        history = self.position_price_history.get(p.id)
        velo_cfg = self.trade_mgmt_config.get("velocity_trigger", {})
        lookback_seconds = velo_cfg.get("lookback_seconds", 5)

        if not history or len(history) < 3:
            return False

        oldest_point = None
        for point in history:
            if (now - point[0]).total_seconds() <= lookback_seconds:
                oldest_point = point
                break

        if oldest_point is None:
            return False

        current_time, current_price = history[-1]
        oldest_time, oldest_price = oldest_point

        time_delta = (current_time - oldest_time).total_seconds()
        if time_delta < 1:
            return False

        price_delta = current_price - oldest_price
        velocity_points_per_sec = price_delta / time_delta

        if velocity_points_per_sec >= 0:
            return False

        distance_to_sl = current_price - p.sl_price
        if distance_to_sl <= 0:
            return False

        try:
            time_to_sl_hit = distance_to_sl / abs(velocity_points_per_sec)
        except ZeroDivisionError:
            return False

        threshold = velo_cfg.get("time_to_sl_threshold_seconds", 2.0)

        if time_to_sl_hit < threshold:
            L.warning(f"VELOCITY TRIGGER: Price moving towards SL on {p.tradingsymbol} at {abs(velocity_points_per_sec):.2f} pts/sec. "
                      f"Est. time to impact: {time_to_sl_hit:.2f}s (Threshold: {threshold}s).")
            return True

        return False


# ==================================================================================================
# STRATEGY DEFINITIONS
# ==================================================================================================
class BaseStrategy(ABC):
    def __init__(self, name: StrategyName, engine: 'Engine', params: Dict):
        self.name = name
        self.engine = engine
        self.params = params
        self.is_agnostic: bool = False

    @abstractmethod
    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]:
        pass

    @abstractmethod
    def get_risk_params(self, token: int, side: OrderSide, current_time: datetime) -> Tuple[float, float]:
        pass

    def evaluate(self, token: int, regime: Regime, current_time: datetime) -> Optional[TradeSignal]:
        if side := self.check_signal(token, regime, current_time):
            risk_points, reward_points = self.get_risk_params(token, side, current_time)
            if risk_points > 0 and reward_points > 0:
                return TradeSignal(self.name, side, risk_points, reward_points)
        return None
# Add this class definition to main1.py with the other strategy classes
class OpeningRangeBreakout(BaseStrategy):
    def __init__(self, name: StrategyName, engine: 'Engine', params: Dict):
        super().__init__(name, engine, params)
        self.orb_high = None
        self.orb_low = None
        try:
            # Load times from config, ensuring they are time objects
            self.orb_set_time = dtime.fromisoformat(self.params["orb_set_time"])
            self.entry_window_end = dtime.fromisoformat(self.params["entry_window_end"])
        except (KeyError, ValueError) as e:
            L.critical(f"FATAL: Invalid ORB time parameters in config for {name.value}: {e}")
            raise SystemExit(f"FATAL: {name.value} strategy config error.")

        self.trades_taken_today = set() # Stores tuples: (token, OrderSide)
        self._orb_set_date = None # Track the date range was set for

    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]:
        now_time = current_time.time()
        today = current_time.date()

        # Reset daily state if market is closed or before settling time
        # Or if the date changed since last check (covers overnight reset)
        if (now_time > self.engine.timings_config["market_close"] or
                now_time < self.engine.timings_config["market_settling_time"] or
                (self.orb_high is not None and self._orb_set_date != today)):

            if self.orb_high is not None or self.trades_taken_today: # Only log reset if needed
                L.info(f"ORB: Resetting daily state for {self.engine.book.get_symbol(token)}.")
            self.orb_high = None
            self.orb_low = None
            self.trades_taken_today.clear()
            self._orb_set_date = None # Clear the date tracker
            return None

        # Don't check before range is set
        if now_time < self.orb_set_time:
            return None

        df_1m = self.engine.get_ohlc(token, 1)
        # Ensure we have today's data and it's not empty
        if df_1m.empty or df_1m.index[-1].date() != today:
            L.warning(f"ORB: No valid 1m data for token {token} at {current_time}")
            return None

        # Set the ORB range exactly once per day
        if self.orb_high is None:
            try:
                # Use pandas `between_time` which handles tz-aware index correctly
                # include_end=False ensures the setting candle itself isn't part of the range
                orb_range_df = df_1m.between_time(self.engine.timings_config["market_open"], self.orb_set_time, include_end=False)

                # Check if the range actually contains today's data and is not empty
                if orb_range_df.empty or orb_range_df.index[-1].date() != today:
                    L.warning(f"ORB: 1m data empty or stale for range {self.engine.timings_config['market_open']} - {self.orb_set_time} on {today} for token {token}. Cannot set ORB.")
                    # Don't try setting again today if data is bad
                    self._orb_set_date = today # Mark as checked for today
                    return None
                self.orb_high = orb_range_df['high'].max()
                self.orb_low = orb_range_df['low'].min()
                self._orb_set_date = today # Mark the date range was set
                L.info(f"ORB Set for {self.engine.book.get_symbol(token)}: H={self.orb_high:.2f}, L={self.orb_low:.2f}")
            except Exception as e:
                L.error(f"ORB: Error setting range for token {token}: {e}")
                self._orb_set_date = today # Prevent retrying if error occurs
                return None

        # Check if ORB range failed to set or is invalid (e.g., high <= low)
        if self.orb_high is None or self.orb_low is None or self.orb_high <= self.orb_low:
             if self.orb_high is not None: # Only log if it was set then became invalid
                 L.warning(f"ORB: Invalid range calculated H={self.orb_high}, L={self.orb_low}. Skipping checks for {token} today.")
             # Mark as checked to prevent repeated invalid calculations
             self._orb_set_date = today
             self.orb_high = None # Ensure it stays None
             self.orb_low = None
             return None

        # Check if outside entry window
        if now_time >= self.entry_window_end:
            return None

        # Check for breakout only if range is valid and within window
        # Access last close safely
        try:
            last_close = df_1m.iloc[-1]['close']
        except IndexError:
             L.warning(f"ORB: Could not get last close for {token}.")
             return None

        # Check BUY signal (only once per day per token per side)
        if last_close > self.orb_high and (token, OrderSide.BUY) not in self.trades_taken_today:
            L.info(f"ORB BUY signal for {self.engine.book.get_symbol(token)} at {last_close:.2f} (ORB High: {self.orb_high:.2f})")
            self.trades_taken_today.add((token, OrderSide.BUY))
            return OrderSide.BUY

        # Check SELL signal (only once per day per token per side)
        if last_close < self.orb_low and (token, OrderSide.SELL) not in self.trades_taken_today:
            L.info(f"ORB SELL signal for {self.engine.book.get_symbol(token)} at {last_close:.2f} (ORB Low: {self.orb_low:.2f})")
            self.trades_taken_today.add((token, OrderSide.SELL))
            return OrderSide.SELL

        return None # No breakout or already traded this side

    def get_risk_params(self, token: int, side: OrderSide, current_time: datetime) -> Tuple[float, float]:
        if self.orb_high is None or self.orb_low is None:
            L.error(f"ORB: get_risk_params called for {self.engine.book.get_symbol(token)} but ORB range not set or invalid.")
            return 0.0, 0.0

        risk_points = self.orb_high - self.orb_low
        # Ensure risk is positive (should be caught earlier, but double-check)
        if risk_points <= 0:
            L.warning(f"ORB: Calculated 0 or negative risk points: {risk_points} for {self.engine.book.get_symbol(token)}. Returning 0.")
            return 0.0, 0.0

        reward_points = risk_points * self.params.get("rr_multiplier", 1.5)
        return risk_points, reward_points


class MomentumBreakoutStrategy(BaseStrategy):
    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]:
        df = self.engine.get_ohlc(token, self.params["resample_minutes"])
        if len(df) < self.params["squeeze_period"]:
            return None

        df.ta.bbands(length=self.params["bb_period"], append=True)
        df.ta.adx(length=14, append=True)
        df['bbw'] = (df[f'BBU_{self.params["bb_period"]}_2.0'] - df[f'BBL_{self.params["bb_period"]}_2.0']) / df[f'BBM_{self.params["bb_period"]}_2.0']
        df['vol_ma'] = df['volume'].rolling(self.params["bb_period"]).mean()
        last = df.iloc[-2]
        is_in_squeeze = last['bbw'] < df['bbw'].rolling(self.params["squeeze_period"]).mean().iloc[-2] * self.params["squeeze_factor"]
        adx_was_low = (df['ADX_14'].iloc[-10:-2] < 20).any()
        adx_is_rising = df['ADX_14'].iloc[-1] > df['ADX_14'].iloc[-2]
        if is_in_squeeze and adx_was_low and adx_is_rising and last['volume'] > self.params["volume_factor"] * last['vol_ma']:
            if df.iloc[-1]['close'] > last[f'BBU_{self.params["bb_period"]}_2.0']:
                return OrderSide.BUY
            if df.iloc[-1]['close'] < last[f'BBL_{self.params["bb_period"]}_2.0']:
                return OrderSide.SELL
        return None

    def get_risk_params(self, token: int, side: OrderSide, current_time: datetime) -> Tuple[float, float]:
        df_1m = self.engine.get_ohlc(token, 1)
        return calculate_dynamic_risk_params(
            df_1m,
            self.params["atr_sl_multiplier"],
            self.params["atr_tp_multiplier"]
        )


class TrendPullbackStrategy(BaseStrategy):
    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]:
        side = OrderSide.BUY if regime == Regime.TRENDING_UP else OrderSide.SELL
        df_primary = self.engine.get_ohlc(token, self.params["primary_tf"])
        df_confirm = self.engine.get_ohlc(token, self.params["confirm_tf"])
        if len(df_primary) < self.params["ema_period"] + 2 or len(df_confirm) < 21:
            return None
        df_confirm['ema_slow'] = df_confirm.ta.ema(length=21)
        is_htf_bullish = df_confirm['close'].iloc[-1] > df_confirm['ema_slow'].iloc[-1]
        is_htf_bearish = df_confirm['close'].iloc[-1] < df_confirm['ema_slow'].iloc[-1]
        df_primary['ema'] = df_primary.ta.ema(length=self.params["ema_period"])
        last, current = df_primary.iloc[-2], df_primary.iloc[-1]
        if side == OrderSide.BUY and is_htf_bullish and last['low'] <= last['ema'] and current['close'] > last['ema']:
            candle_range = current['high'] - current['low']
            if candle_range > 0 and (current['close'] - current['low']) / candle_range >= 0.75:
                return OrderSide.BUY
        if side == OrderSide.SELL and is_htf_bearish and last['high'] >= last['ema'] and current['close'] < last['ema']:
            candle_range = current['high'] - current['low']
            if candle_range > 0 and (current['high'] - current['close']) / candle_range >= 0.75:
                return OrderSide.SELL
        return None

    def get_risk_params(self, token: int, side: OrderSide, current_time: datetime) -> Tuple[float, float]:
        df_1m = self.engine.get_ohlc(token, 1)
        return calculate_dynamic_risk_params(
            df_1m,
            self.params["atr_sl_multiplier"],
            self.params["atr_tp_multiplier"]
        )


class MeanReversionStrategy(BaseStrategy):
    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]:
        df = self.engine.get_ohlc(token, self.params["resample_minutes"])
        if len(df) < self.params["bb_period"] + 2:
            return None

        df.ta.bbands(length=self.params["bb_period"], append=True)
        last_bar, current_bar = df.iloc[-2], df.iloc[-1]

        upper_band_col = f'BBU_{self.params["bb_period"]}_2.0'
        lower_band_col = f'BBL_{self.params["bb_period"]}_2.0'

        last_upper_band = df[upper_band_col].iloc[-2]
        last_lower_band = df[lower_band_col].iloc[-2]
        current_upper_band = df[upper_band_col].iloc[-1]
        current_lower_band = df[lower_band_col].iloc[-1]

        if last_bar['close'] > last_upper_band and current_bar['close'] < current_upper_band:
            return OrderSide.SELL
        if last_bar['close'] < last_lower_band and current_bar['close'] > current_lower_band:
            return OrderSide.BUY

        return None

    def get_risk_params(self, token: int, side: OrderSide, current_time: datetime) -> Tuple[float, float]:
        df_1m = self.engine.get_ohlc(token, 1)
        if df_1m.empty:
            return 0.0, 0.0
        atr = df_1m.ta.atr(length=14).iloc[-1]
        if pd.isna(atr):
            return 0.0, 0.0
        risk = atr * self.params["atr_sl_multiplier"]

        df_resampled = self.engine.get_ohlc(token, self.params["resample_minutes"])
        if df_resampled.empty:
            return 0.0, 0.0
        df_resampled.ta.bbands(length=self.params["bb_period"], append=True)
        if df_resampled.empty:
            return 0.0, 0.0

        middle_band = df_resampled[f'BBM_{self.params["bb_period"]}_2.0'].iloc[-1]
        current_price = df_resampled['close'].iloc[-1]

        reward = abs(current_price - middle_band)
        return (risk, reward) if reward > 0 else (risk, risk * 1.5)


class VolatilityMeanReversionStrategy(BaseStrategy):
    """A strategy for the CHAOS regime, looking for extreme moves to fade."""
    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]:
        if regime != Regime.CHAOS:
            return None

        df = self.engine.get_ohlc(token, self.params["resample_minutes"])
        if len(df) < self.params["ema_period"]:
            return None

        df['ema'] = df.ta.ema(length=self.params["ema_period"])
        last_close = df['close'].iloc[-1]
        last_ema = df['ema'].iloc[-1]
        deviation_pct = ((last_close - last_ema) / last_ema) * 100

        trigger_pct = self.params["deviation_pct_trigger"]

        if deviation_pct > trigger_pct:
            return OrderSide.SELL

        if deviation_pct < -trigger_pct:
            return OrderSide.BUY

        return None

    def get_risk_params(self, token: int, side: OrderSide, current_time: datetime) -> Tuple[float, float]:
        df = self.engine.get_ohlc(token, self.params["resample_minutes"])
        if len(df) < 14:
            return 0.0, 0.0

        atr = df.ta.atr(length=14).iloc[-1]
        if pd.isna(atr):
            return 0.0, 0.0

        risk_points = atr * self.params["atr_sl_multiplier"]
        reward_points = atr * self.params["atr_tp_multiplier"]

        return risk_points, reward_points


# ==================================================================================================
# MAIN TRADING ENGINE
# ==================================================================================================
class Engine:
    def __init__(self,
                 store_actor: StoreActor,
                 book: InstrumentBook,
                 prices: PriceBus,
                 config: Dict):
        # self.k = kite  <--- DELETED
        self.store_actor = store_actor
        self.book = book
        self.prices = prices
        self.config = config
        self.timings_config = config["timings"]
        self.trading_config = config["trading"]
        self.technical_config = config["technical"]
        self.market_breadth = 0 
        self.nifty_50_tokens = []

        self.trader: Optional[AbstractTrader] = None
        self.risk_manager: Optional[RiskManager] = None
        self.micro_monitor: Optional[MicrostructureMonitor] = None
        self.pos_manager: Optional[PositionManager] = None
        
        self.tick_thread: Optional[threading.Thread] = None
        self.order_thread: Optional[threading.Thread] = None
        self.trade_executor_thread: Optional[threading.Thread] = None
        self.scheduler_threads: Dict[str, threading.Thread] = {}

        self.master_lock = threading.RLock()
        self.bars = BarStore(timeframes=[1, 3, 5, 15])
        
        self.atm_iv_cache = {} # Key: "NIFTY"/"BANKNIFTY", Value: float
        self.nifty_token = self.book.special_tokens.get("NIFTY")
        self.bn_token = self.book.special_tokens.get("BANKNIFTY")
        self.vix_token = self.book.special_tokens.get("INDIA VIX")
        if not all([self.nifty_token, self.bn_token]):
            raise SystemExit("FATAL: Could not find NIFTY/BANKNIFTY futures contracts from instrument file.")

        self.running = threading.Event()
        self.master_halt = False      
        self.regime_halt = False
        self.regime = Regime.UNCLEAR
        self.regime_confidence: float = 0.0
        self.regime_change_history = deque(maxlen=100) 
        self.last_trade_timestamp: Optional[datetime] = None
        self.last_regime_change_time: Optional[datetime] = None
        self.potential_regime: Optional[Regime] = None
        self.potential_regime_count: int = 0
        self.regime_confirmation_threshold: int = self.config["strategies"]["regime_classifier"].get("hysteresis_confirmation_count", 3)
        self.nifty_bn_zscore: Optional[float] = None

        self.last_trading_day: Optional[date] = None
        self.eod_flatten_triggered = False
        self.eod_report_sent = False

        self.classifier = RegimeClassifier(self, self.nifty_token, self.bn_token, self.vix_token, self.config["strategies"]["regime_classifier"])
        self.strategies: Dict[Regime, List[BaseStrategy]] = self._load_strategies()
        

        self.last_known_prices: Dict[int, float] = {}
        self.sanity_check_pct = self.technical_config.get("insane_tick_pct", 5.0) / 100.0

        self.trade_signal_queue = Queue()
        self.fatal_error_event = threading.Event()
        self.underlying_cooldown: Dict[int, datetime] = {}

        self.nse_calendar = mcal.get_calendar('NSE')
        self.vix_long_history_df = pd.DataFrame()
        self.scheduler = self._setup_scheduler()

        self.historical_avg_iv: Dict[int, pd.Series] = {}

    def set_dependencies(self, trader: AbstractTrader, risk_manager: RiskManager, micro_monitor: MicrostructureMonitor, pos_manager: PositionManager):
        self.trader = trader
        self.risk_manager = risk_manager
        self.micro_monitor = micro_monitor
        self.pos_manager = pos_manager
        
        
        if not PAPER_TRADING:
            self.prices.on_connect_callbacks.append(self.reconcile)
        L.info("All dependencies injected into Engine.")

    def get_ohlc(self, token: int, timeframe: int) -> pd.DataFrame:
        return self.bars.get_ohlc(token, timeframe)
       
    def _update_market_breadth(self):
        """Scheduled task to calculate and update Nifty 50 Advance/Decline breadth."""
        if not self.nifty_50_tokens:
            return
        try:
            # --- FIXED: Use OrderActor ---
            L.debug("Requesting Nifty 50 quotes via OrderActor...")
            reply_q = queue.Queue()
            self.trader.order_actor.q.put({
                "type": "quote",
                "params": {"instrument_tokens": self.nifty_50_tokens},
                "reply_q": reply_q
            })
            
            quotes = None
            try:
                resp = reply_q.get(timeout=10.0)
                if resp['ok']:
                    quotes = resp['res']
                else:
                    raise Exception(resp.get('error', 'Failed to get quotes from OrderActor'))
            except queue.Empty:
                L.error("Timeout waiting for OrderActor quote reply for market breadth.")
                return
            # --- END FIX ---

            if not quotes:
                L.warning("Market breadth quote fetch returned no data.")
                return

            advances = 0
            declines = 0
            for token_str, data in quotes.items():
                # Use ohlc.close (previous day's close) for change
                prev_close = data.get('ohlc', {}).get('close', 0)
                ltp = data.get('last_price')

                if ltp and prev_close > 0:
                    if ltp > prev_close:
                        advances += 1
                    elif ltp < prev_close:
                        declines += 1

            self.market_breadth = advances - declines
            L.info(f"Market breadth updated: A={advances}, D={declines}, Net={self.market_breadth}")

        except Exception as e:
            L.error(f"Failed to update market breadth: {e}", exc_info=True)

    def _is_tick_sane(self, tick: Dict) -> bool:
        now_time = now_ist().time()
        if now_time < self.timings_config["market_settling_time"]:
            return True

        token = tick["instrument_token"]
        price = tick.get("last_price")
        if price is None or price <= 0:
            return False
        last_price = self.last_known_prices.get(token)
        if last_price is None:
            self.last_known_prices[token] = price
            return True
        price_change_pct = abs(price - last_price) / last_price
        if price_change_pct > self.sanity_check_pct:
            L.warning(f"INSANE TICK DETECTED for token {token}. New: {price}, Old: {last_price}. Discarding.")
            return False
        self.last_known_prices[token] = price
        return True
    
    def _calculate_atm_iv(self, underlying_token: int, spot: float, expiry: date, T: float, hv: float) -> Optional[float]:
        """Helper to calculate ATM IV for a given underlying."""
        underlying_name = _get_underlying(self.book.get_symbol(underlying_token))
        tep = self.book.step_size(underlying_name)
        atm_strike = round(spot / tep) * tep

        atm_call = self.book.find_option(underlying_name, expiry, atm_strike, "CE")
        atm_put = self.book.find_option(underlying_name, expiry, atm_strike, "PE")

        ivs = []
        for opt in [atm_call, atm_put]:
            if not opt:
                continue
            tick = self.prices.get_full_tick(int(opt['instrument_token']))
            if tick and tick.get('last_price'):
                iv = calculate_iv(tick['last_price'], spot, atm_strike, T, 0.05, opt['instrument_type'] == 'CE', hv_fallback=hv)
                ivs.append(iv)

        if ivs:
            return sum(ivs) / len(ivs)
        return None
    
    
    def _update_atm_iv_cache(self):
        """Scheduled task to update the ATM IV cache."""
        now = now_ist()
        market_close_time = self.timings_config["market_close"]

        for token in [self.nifty_token, self.bn_token]:
            if not token:
                continue
            spot = self.prices.ltp(token)
            if not spot:
                continue

            underlying_name = _get_underlying(self.book.get_symbol(token))
            expiry = self.book.find_nearest_expiry_date(underlying_name)
            if not expiry:
                continue

            T = _calculate_time_to_expiry(expiry, now, market_close_time)
            hv = calculate_historical_volatility(self.get_ohlc(token, 1)['close'])
            if not hv:
                hv = 0.3 # Default fallback

            atm_iv = self._calculate_atm_iv(token, spot, expiry, T, hv)
            if atm_iv:
                L.debug(f"Updating ATM IV cache for {underlying_name}: {atm_iv:.4f}")
                self.atm_iv_cache[underlying_name] = atm_iv
                
                
    def _tick_processor_worker(self):
        L.info("Tick processor worker started.")
        while self.running.is_set():
            try:
                ticks = self.prices.tick_queue.get(timeout=1)
                sane_ticks = [t for t in ticks if self._is_tick_sane(t)]
                if not sane_ticks:
                    continue

                with self.prices.lock:
                    for t in sane_ticks:
                        self.prices.last[t["instrument_token"]] = t.get("last_price")
                        self.prices.full_ticks[t["instrument_token"]] = t
                        self.micro_monitor.update_tfi_score(t)

                self.process_ticks(sane_ticks)
            except Empty:
                continue
            except Exception as e:
                L.error(f"FATAL Error in tick processor worker: {e}", exc_info=True)


    # Inside Engine class
    def _score_and_size_trade(self, signal: TradeSignal, token: int) -> Optional[Dict]:
        """
        Scores a signal based on multiple confluence factors and returns a package
        containing the score and PRELIMINARY trade parameters (for option selection).
        FINAL sizing is done later in the planner.

        Returns:
             Dict: {"score": float, "params": Dict, "is_agnostic": bool} if score meets threshold, else None.
                   'params' here contains option details but NOT the final calculated lot size.
        """
        score = 1.0 # Base score for a valid signal trigger
        score_log = [f"Base Signal ({signal.strategy_name.value}): {score:.1f}"]
        is_agnostic_signal = False

        # --- 1. Regime Confluence & Agnostic Check ---
        current_regime = self.regime
        strategy_name = signal.strategy_name
        # Check if the strategy exists in the "AGNOSTIC" list
        if "AGNOSTIC" in self.strategies and strategy_name in [s.name for s in self.strategies.get("AGNOSTIC", [])]:
             is_agnostic_signal = True
             score_log.append("Regime Agnostic: ±0.0")
             # Optional penalty if regime is clear and confident
             if self.regime != Regime.UNCLEAR and self.regime_confidence > 0.7:
                 score -= 0.2
                 score_log.append(f"Clear Regime Penalty: -0.2")
        else:
             # Apply standard regime confluence scoring for non-agnostic strategies
             good_combos = {
                 Regime.TRENDING_UP: [StrategyName.TREND_PULLBACK],
                 Regime.TRENDING_DOWN: [StrategyName.TREND_PULLBACK],
                 Regime.COMPRESSION: [StrategyName.MOMENTUM_BREAKOUT],
                 # MeanReversion might work in CHOP, but we penalize below, so don't list here
                 # Regime.CHOP: [StrategyName.MEAN_REVERSION],
                 Regime.CHAOS: [StrategyName.VOLATILITY_MEAN_REVERSION]
             }
             bad_combos = {
                 Regime.CHOP: [StrategyName.TREND_PULLBACK, StrategyName.MOMENTUM_BREAKOUT],
                 Regime.TRENDING_UP: [StrategyName.MEAN_REVERSION],
                 Regime.TRENDING_DOWN: [StrategyName.MEAN_REVERSION]
             }
             if strategy_name in good_combos.get(current_regime, []):
                 score += 1.0
                 score_log.append(f"Regime Confluence ({current_regime.name}): +1.0")
             elif strategy_name in bad_combos.get(current_regime, []):
                 score -= 1.0
                 score_log.append(f"Regime Conflict ({current_regime.name}): -1.0")

        # --- 2. Specific Strategy Penalties (MeanReversion in CHOP) ---
        if current_regime == Regime.CHOP and strategy_name == StrategyName.MEAN_REVERSION:
            iv_rank = self._get_iv_rank()
            # Use the same threshold as the Theta Filter for consistency
            low_iv_rank_threshold = self.trading_config.get("theta_filter_iv_rank_threshold", 25.0)
            if iv_rank is not None and iv_rank < low_iv_rank_threshold:
                 # Heavy penalty for buying options in low IV chop
                 score -= 1.5
                 score_log.append(f"Penalty: MeanReversion in Low IV CHOP ({iv_rank:.1f}%): -1.5")
            else:
                 # Smaller penalty if IV Rank is higher but still CHOP
                 score -= 0.5
                 score_log.append(f"Penalty: MeanReversion in CHOP: -0.5")

        # --- 3. Preliminary Parameter Check (Finds Option, Uses Dummy Size) ---
        # This call is crucial to get the 'opt' details for subsequent checks
        prelim_params = self.get_trade_params(
            token=token, side=signal.side,
            risk_points_on_underlying=signal.risk_points,
            reward_points_on_underlying=signal.reward_points,
            strategy=signal.strategy_name.value, regime=self.regime,
            confidence_score=1.0 # DUMMY score - size calculated here is ignored
        )
        # If no valid option contract is found based on filters, drop the signal
        if not prelim_params:
            L.debug(f"Signal dropped: No valid option contract found for {signal.strategy_name.value} on {self.book.get_symbol(token)}.")
            return None
        option_symbol = prelim_params['opt']['tradingsymbol'] # Get for logging

        # --- 4. Microstructure Score (TFI/OBI on the specific option) ---
        option_token = prelim_params['opt']['instrument_token']
        obi_score = self.micro_monitor.check_order_book_imbalance(option_token, signal.side)
        tfi_score = self.micro_monitor.check_tfi(option_token, signal.side)
        if obi_score == 1: score += 0.5; score_log.append("OBI Confirm: +0.5")
        if tfi_score == 1: score += 0.5; score_log.append("TFI Confirm: +0.5")
        if obi_score == -1: score -= 1.0; score_log.append("OBI Conflict: -1.0") # Increased penalty
        if tfi_score == -1: score -= 1.0; score_log.append("TFI Conflict: -1.0") # Increased penalty
        
        # --- NEW: 5. StatArb Relative Value Score ---
        if self.nifty_bn_zscore is not None:
            zscore = self.nifty_bn_zscore
            z_threshold = 1.5 # How many std devs to consider "expensive" or "cheap"

            if token == self.nifty_token: # We are trading NIFTY
                if signal.side == OrderSide.BUY:
                    if zscore < -z_threshold: # Nifty is "cheap"
                        score += 1.0; score_log.append(f"StatArb Confirm (Nifty Cheap): +1.0")
                    elif zscore > z_threshold: # Nifty is "expensive"
                        score -= 1.0; score_log.append(f"StatArb Conflict (Nifty Expensive): -1.0")
                
                elif signal.side == OrderSide.SELL:
                    if zscore > z_threshold: # Nifty is "expensive"
                        score += 1.0; score_log.append(f"StatArb Confirm (Nifty Expensive): +1.0")
                    elif zscore < -z_threshold: # Nifty is "cheap"
                        score -= 1.0; score_log.append(f"StatArb Conflict (Nifty Cheap): -1.0")

            elif token == self.bn_token: # We are trading BANKNIFTY (logic is inverted)
                if signal.side == OrderSide.BUY:
                    if zscore > z_threshold: # Nifty is "expensive" -> BN is "cheap"
                        score += 1.0; score_log.append(f"StatArb Confirm (BN Cheap): +1.0")
                    elif zscore < -z_threshold: # Nifty is "cheap" -> BN is "expensive"
                        score -= 1.0; score_log.append(f"StatArb Conflict (BN Expensive): -1.0")
                
                elif signal.side == OrderSide.SELL:
                    if zscore < -z_threshold: # Nifty is "cheap" -> BN is "expensive"
                        score += 1.0; score_log.append(f"StatArb Confirm (BN Expensive): +1.0")
                    elif zscore > z_threshold: # Nifty is "expensive" -> BN is "cheap"
                        score -= 1.0; score_log.append(f"StatArb Conflict (BN Cheap): -1.0")

        # --- 6. Market Breadth Score ---
        if self.market_breadth != 0:
            breadth_threshold = self.technical_config.get("breadth_score_threshold", 10)
            if (signal.side == OrderSide.BUY and self.market_breadth > breadth_threshold) or \
               (signal.side == OrderSide.SELL and self.market_breadth < -breadth_threshold):
                 score += 1.0; score_log.append(f"Breadth Confirm (Net {self.market_breadth}): +1.0")
            elif (signal.side == OrderSide.BUY and self.market_breadth < -breadth_threshold) or \
                 (signal.side == OrderSide.SELL and self.market_breadth > breadth_threshold):
                 score -= 1.0; score_log.append(f"Breadth Conflict (Net {self.market_breadth}): -1.0")

        # --- 7. Volatility Structure Score (IV vs HV) ---
        vol_config = self.technical_config.get("volatility_filter", {})
        hv_period = vol_config.get("hv_period", 20)
        underlying_ohlc = self.get_ohlc(token, 1)
        if not underlying_ohlc.empty:
            hv = calculate_historical_volatility(underlying_ohlc['close'], window=hv_period)
            underlying_name = _get_underlying(self.book.get_symbol(token))
            atm_iv = self.atm_iv_cache.get(underlying_name)
            if hv and atm_iv:
                vol_spread = atm_iv - hv
                spread_threshold_low = vol_config.get("iv_hv_spread_threshold_low", -0.05)
                spread_threshold_high = vol_config.get("iv_hv_spread_threshold_high", 0.10)
                if vol_spread < spread_threshold_low:
                     score += 1.0; score_log.append(f"Vol 'Coiled Spring' (Spread {vol_spread:.3f}): +1.0")
                elif vol_spread > spread_threshold_high:
                     if signal.side == OrderSide.BUY: # Penalize buying expensive premium
                         score -= 0.5; score_log.append(f"Vol 'Expensive Premium' (Spread {vol_spread:.3f}): -0.5")

        # NOTE: Regime Confidence multiplier is NOT applied to the score itself.
        # It affects the final *sizing* in the planner via RiskManager.

        # --- 8. Final Score Calculation & Minimum Threshold Check ---
        final_score = max(0, score) # Clamp score at 0 minimum

        # Log BEFORE threshold check for debugging visibility
        # L.info(f"Preliminary Score for {option_symbol}: {final_score:.1f}. Log: {', '.join(score_log)}") # Moved logging to planner

        min_score = self.trading_config.get("min_trade_score", 1.0)
        if final_score < min_score:
            # L.debug(f"Signal score {final_score:.1f} < min threshold {min_score} for {option_symbol}. Dropped.") # Moved logging
            return None

        # --- 9. Return Package with Score and Preliminary Params ---
        # The 'params' dict still holds the option details ('opt') and greeks,
        # but the 'lots' and 'total_trade_risk' are based on the dummy score=1.0 and will be recalculated.
        return {
            "score": final_score,
            "params": prelim_params,
            "is_agnostic": is_agnostic_signal
        }
        
    def _order_processor_worker(self):
        L.info("Order processor worker started.")
        while self.running.is_set():
            try:
                order = self.prices.order_update_queue.get(timeout=1)
                self._handle_order_update_from_queue(order)
            except Empty:
                continue
            except Exception as e:
                L.error(f"Error in order processor worker: {e}", exc_info=True)

    def start(self):
        if not all([self.trader, self.risk_manager, self.micro_monitor, self.pos_manager]):
            raise SystemExit("FATAL: Dependencies not set. Call engine.set_dependencies() before start().")

        self.warm_up() # This is now fixed to use actors
        self.prices.start()
        if not self.prices.connected.wait(10):
            raise SystemExit("FATAL: PriceBus WebSocket could not connect.")

        tokens_to_subscribe = [self.nifty_token, self.bn_token]
        if self.vix_token:
            tokens_to_subscribe.append(self.vix_token)

        self.prices.subscribe(tokens_to_subscribe)

        self.running.set()

        L.info("Starting TickProcessor thread...")
        self.tick_thread = threading.Thread(target=self._tick_processor_worker, name="TickProcessor", daemon=True)
        self.tick_thread.start()

        L.info("Starting OrderProcessor thread...")
        self.order_thread = threading.Thread(target=self._order_processor_worker, name="OrderProcessor", daemon=True)
        self.order_thread.start()

        L.info("Starting TradeExecutor thread...")
        self.trade_executor_thread = threading.Thread(target=self._trade_executor_worker, name="TradeExecutor", daemon=True)
        self.trade_executor_thread.start()

        L.info("Starting scheduler threads...")
        for name, (func, interval) in self.scheduler.items():
            thread = threading.Thread(target=self._run_task_in_loop, args=(func, interval, name), name=name, daemon=True)
            thread.start()
            self.scheduler_threads[name] = thread 
            L.info(f"Started scheduler thread for '{name}' with {interval}s interval.")

        self.loop() # Start the main loop

    def _run_task_in_loop(self, func: Callable, interval: int, name: str):
        while self.running.is_set():
            try:
                is_halted_check = False
                if name in ["strategic_planner"]:
                    with self.master_lock:
                        is_halted_check = self.master_halt

                if not is_halted_check:
                    func()

            except Exception as e:
                L.error(f"Error in scheduled task '{name}': {e}", exc_info=True)
            time.sleep(interval)

    
    def _send_eod_report(self):
        now_time = now_ist().time()
        if now_time > self.timings_config["market_close"] and not self.eod_report_sent:
            L.info("Sending End-of-Day report...")
            
            # --- FIXED: Use StoreActor ---
            L.debug("Requesting EOD stats from StoreActor...")
            reply_q = queue.Queue()
            self.store_actor.q.put({
                "type": "get_todays_trades_stats",
                "reply_q": reply_q
            })
            
            wins, losses = 0, 0
            try:
                resp = reply_q.get(timeout=10.0)
                if resp['ok']:
                    wins, losses = resp['res']
                else:
                    L.error(f"Could not get EOD stats from StoreActor: {resp.get('error')}")
            except queue.Empty:
                L.error("Timeout getting EOD stats from StoreActor")
            # --- END FIX ---
                
            total_trades = wins + losses
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            report = (f"📊 **--- End of Day Report ---** 📊\n\n"
                      f"**Net Realized PnL:** ₹{self.trader.daily_realized_pnl:,.2f}\n\n"
                      f"**Total Trades:** {total_trades}\n"
                      f"**Winning Trades:** {wins}\n"
                      f"**Losing Trades:** {losses}\n"
                      f"**Win Rate:** {win_rate:.2f}%\n")
            send_alert(report)
            self.eod_report_sent = True

    def loop(self):
        send_alert("🛡️ SENTINEL PRIME PROTOCOL ENGAGED. Awaiting market open...")
        while now_ist().time() < self.timings_config["market_open"] and self.running.is_set():
            L.info(f"Pre-Market state. Waiting for market open at {self.timings_config['market_open']}...")
            time.sleep(60)
        if not self.running.is_set():
            self.stop()
            return

        send_alert("🔔 Market is OPEN. Trading logic is now active.")
        L.info("Market Open state. Trading logic is active.")
        try:
            while self.running.is_set():
                now = now_ist()
                now_time = now.time()
                if now_time >= self.timings_config["market_close"]:
                    L.info("Market is now CLOSED. Transitioning to post-market state.")
                    break

                if self.fatal_error_event.is_set():
                    send_alert("🔥 FATAL ERROR EVENT RECEIVED. HALTING ALL TRADING.", "critical")
                    with self.master_lock:
                        self.master_halt = True
                    self.fatal_error_event.clear()

                with self.master_lock:
                    if os.path.exists(KILL_SWITCH_FILE):
                        if not self.master_halt:
                            send_alert("⛔ KILL SWITCH DETECTED. HALTING ALL NEW TRADES. ⛔", "critical")
                            self.master_halt = True
                            if G_HALTED_STATUS:
                                G_HALTED_STATUS.set(1)

                    if not self.eod_flatten_triggered and now_time >= self.timings_config["eod_flatten_time"]:
                        L.warning("EOD flatten time reached. Halting new trades and closing all positions.")
                        self.master_halt = True
                        if G_HALTED_STATUS:
                            G_HALTED_STATUS.set(1)
                        self.eod_flatten_triggered = True

                        with self.trader.lock:
                            positions_to_close = [p for p in self.trader.positions.values() if p.status not in [PositionStatus.CLOSED.value, PositionStatus.PENDING_CLOSURE.value]]

                        if not positions_to_close:
                            L.info("EOD flatten: No open positions to close.")
                        else:
                            for p in positions_to_close:
                                self.trader.close_position(p, "EOD_FLATTEN")
                time.sleep(10)
        except KeyboardInterrupt:
            L.warning("KeyboardInterrupt detected in main loop.")
            self.stop()
            return

        L.info("Post-Market state. Running final tasks for the day.")
        if not self.eod_report_sent:
            self._send_eod_report()
            
        # --- FIXED: Use StoreActor ---
        if self.last_trading_day:
            L.debug("Sending final daily PnL to StoreActor...")
            self.store_actor.q.put({
                "type": "set_kv",
                "key": f"daily_pnl_{self.last_trading_day}",
                "value": str(self.trader.daily_realized_pnl)
            })
        # --- END FIX ---
            
        send_alert(f"💤 Sentinel shutting down for the day. Final Realized PnL: ₹{self.trader.daily_realized_pnl:,.2f}")
        L.info("Daily tasks complete. Bot will now stop.")
        self.stop()

    # Inside Engine class
    def _load_strategies(self) -> Dict[Union[Regime, str], List[BaseStrategy]]: # Allow str key for agnostic
        strategies: Dict[Union[Regime, str], List[BaseStrategy]] = {
            # --- Regime Specific ---
            Regime.COMPRESSION: [],
            Regime.TRENDING_UP: [],
            Regime.TRENDING_DOWN: [],
            Regime.CHOP: [],
            Regime.CHAOS: [],
            # --- Always Run (or under specific conditions like UNCLEAR) ---
            "AGNOSTIC": []
        }

        # Load strategies based on config, appending to the correct list
        if "MomentumBreakout" in self.config["strategies"]:
            strategies[Regime.COMPRESSION].append(
                MomentumBreakoutStrategy(StrategyName.MOMENTUM_BREAKOUT, self, self.config['strategies']['MomentumBreakout'])
            )
        if "TrendPullback" in self.config["strategies"]:
            tp_params = self.config['strategies']['TrendPullback']
            strategies[Regime.TRENDING_UP].append(TrendPullbackStrategy(StrategyName.TREND_PULLBACK, self, tp_params))
            strategies[Regime.TRENDING_DOWN].append(TrendPullbackStrategy(StrategyName.TREND_PULLBACK, self, tp_params))
        if "MeanReversion" in self.config["strategies"]:
            strategies[Regime.CHOP].append(
                MeanReversionStrategy(StrategyName.MEAN_REVERSION, self, self.config['strategies']['MeanReversion'])
            )
        if "VolatilityMeanReversion" in self.config["strategies"]:
             strategies[Regime.CHAOS].append(
                 VolatilityMeanReversionStrategy(StrategyName.VOLATILITY_MEAN_REVERSION, self, self.config['strategies']['VolatilityMeanReversion'])
             )

        # Load Agnostic strategies
        if "OpeningRangeBreakout" in self.config["strategies"]:
             orb_params = self.config['strategies']['OpeningRangeBreakout']
             # Ensure the Enum was updated
             if hasattr(StrategyName, 'OPENING_RANGE_BREAKOUT'):
                 orb_strategy = OpeningRangeBreakout(StrategyName.OPENING_RANGE_BREAKOUT, self, orb_params)
                 orb_strategy.is_agnostic = True  # <-- ADD THIS LINE
                 strategies["AGNOSTIC"].append(orb_strategy)
                 L.info("Loaded OpeningRangeBreakout strategy (Regime Agnostic)")
             else:
                  L.error("OpeningRangeBreakout strategy configured but Enum not updated!")

        # Log loaded strategies
        for key, strat_list in strategies.items():
            if strat_list:
                key_name = key.name if isinstance(key, Regime) else key
                strat_names = [s.name.value for s in strat_list]
                L.info(f"Strategies loaded for {key_name}: {', '.join(strat_names)}")

        return strategies

    # Inside Engine class
    def _setup_scheduler(self) -> Dict[str, Tuple[Callable, int]]:
        tasks = {
            "strategic_planner": (self._run_strategic_planner, 2), # Correctly set to 2 seconds
            "position_management": (self.pos_manager.manage_positions, 1),
            "pnl_updater": (self.risk_manager.update_pnl_metrics, 2),
            "reconciliation": (self.reconcile, 300),
            "health_check": (self.health_check, 60),
            "strategy_weighting": (self.risk_manager._update_strategy_weights, 3600), # 1 hour
            "eod_report": (self._send_eod_report, 300), # 5 mins (runs near EOD)
            "equity_update": (self.risk_manager.update_dynamic_equity, 120), # 2 mins
            "data_persistence": (self._persist_bar_data, 3600), # 1 hour
            "bar_reconciliation": (self._reconcile_bars, 900), # 15 mins
            "pnl_reconciliation": (self.risk_manager.reconcile_broker_pnl, 900), # 15 mins
            "atm_iv_cache": (self._update_atm_iv_cache, 10), # 10 seconds
            "market_breadth": (self._update_market_breadth, 300) # 5 mins
        }
        if METRICS_APP:
            tasks["prometheus_metrics"] = (self._update_prometheus_metrics, 15) # 15 seconds
        return tasks

    def _reset_daily_state(self):
        L.info("Resetting daily state for new trading day.")
        # This now uses store_actor internally
        self.risk_manager.reset_daily_state(self.last_trading_day) 

        with self.master_lock:
            self.master_halt = False
            self.regime_halt = False
            if os.path.exists(KILL_SWITCH_FILE):
                try:
                    os.remove(KILL_SWITCH_FILE)
                    L.warning("Kill switch file removed for the new trading day.")
                except OSError as e:
                    L.error(f"Could not remove kill switch file: {e}")

            if G_HALTED_STATUS:
                G_HALTED_STATUS.set(0)
            self.last_trade_timestamp = None

        self.eod_flatten_triggered = False
        self.eod_report_sent = False
        self.last_trading_day = now_ist().date()
        send_alert(f"☀️ New Trading Day: {self.last_trading_day}. Equity: ₹{self.risk_manager.dynamic_account_equity:,.2f}")

    def warm_up(self):
        L.info("Warming up... Priming historical data.")
        self.last_trading_day = now_ist().date()
        # This now uses store_actor internally
        self.risk_manager.load_persistent_state(self.last_trading_day) 

        to_date, from_date = self.last_trading_day, self.last_trading_day - timedelta(days=self.technical_config["warmup_days"])

        tokens_to_prime = [self.nifty_token, self.bn_token]
        
        try:
            token_file_path = self.technical_config.get("nifty_50_constituents_file")
            if token_file_path and os.path.exists(token_file_path):
                with open(token_file_path, 'r') as f:
                    self.nifty_50_tokens = json.load(f)
                    L.info(f"Loaded {len(self.nifty_50_tokens)} Nifty 50 constituent tokens.")
            else:
                L.error("Nifty 50 constituents file not found or not configured. Market Breadth filter disabled.")
        except Exception as e:
            L.error(f"Failed to load Nifty 50 constituents: {e}")

        for token in tokens_to_prime:
            if token is None:
                continue
            symbol = self.book.get_symbol(token) or f"Token {token}"
            L.info(f"Priming historical data for: {symbol}")
            
            # --- FIXED: Use OrderActor ---
            hist = None
            L.debug(f"Requesting hist data for {symbol} via OrderActor...")
            reply_q = queue.Queue()
            params = {
                "instrument_token": token,
                "from_date": from_date,
                "to_date": to_date,
                "interval": "minute"
            }
            self.trader.order_actor.q.put({
                "type": "historical_data",
                "params": params,
                "reply_q": reply_q
            })
            try:
                resp = reply_q.get(timeout=30.0) # Longer timeout for historical
                if resp['ok']:
                    hist = resp['res']
                else:
                    L.error(f"Failed to get hist data from OrderActor: {resp.get('error')}")
            except queue.Empty:
                L.error(f"Timeout getting hist data for {symbol} from OrderActor")
            # --- END FIX ---

            if hist:
                df_hist=pd.DataFrame(hist)
                self.bars.prime(token, df_hist, append=False)
                L.info(f"Successfully primed {len(hist)} bars for {symbol}.")
                try:
                    last_bar_ts = pd.to_datetime(df_hist['date'].iloc[-1]).tz_localize(IST)
                    now = now_ist()
                    gap_minutes = (now - last_bar_ts).total_seconds() / 60.0
                    
                    if gap_minutes > 2 and gap_minutes < 375: # Gap is > 2 mins but < 1 day
                        L.warning(f"Detected {gap_minutes:.0f} min data gap for {symbol}. Backfilling...")
                        
                        # --- FIXED: Use OrderActor ---
                        gap_data = None
                        gap_params = {
                            "instrument_token": token,
                            "from_date": last_bar_ts + timedelta(minutes=1),
                            "to_date": now,
                            "interval": "minute"
                        }
                        gap_reply_q = queue.Queue()
                        self.trader.order_actor.q.put({
                            "type": "historical_data",
                            "params": gap_params,
                            "reply_q": gap_reply_q
                        })
                        try:
                            gap_resp = gap_reply_q.get(timeout=30.0)
                            if gap_resp['ok']:
                                gap_data = gap_resp['res']
                            else:
                                L.error(f"Gap-fill failed: {gap_resp.get('error')}")
                        except queue.Empty:
                            L.error("Timeout on gap-fill request")
                        # --- END FIX ---
                        
                        if gap_data:
                            self.bars.prime(token, pd.DataFrame(gap_data), append=True)
                            L.info(f"Successfully backfilled {len(gap_data)} bars for {symbol}.")
                except Exception as e:
                    L.error(f"Error during gap-fill for {symbol}: {e}")
            else:
                L.error(f"Failed to prime history for {symbol} after retries.")

        if self.vix_token:
            L.info("Priming long-term VIX history for IV Rank calculation...")
            vix_from_date = self.last_trading_day - timedelta(days=365)
            
            # --- FIXED: Use OrderActor ---
            vix_hist = None
            vix_params = {
                "instrument_token": self.vix_token,
                "from_date": vix_from_date,
                "to_date": to_date,
                "interval": "day"
            }
            vix_reply_q = queue.Queue()
            self.trader.order_actor.q.put({
                "type": "historical_data",
                "params": vix_params,
                "reply_q": vix_reply_q
            })
            try:
                vix_resp = vix_reply_q.get(timeout=30.0)
                if vix_resp['ok']:
                    vix_hist = vix_resp['res']
                else:
                    L.error(f"VIX hist failed: {vix_resp.get('error')}")
            except queue.Empty:
                L.error("Timeout on VIX hist request")
            # --- END FIX ---
            
            if vix_hist:
                self.vix_long_history_df = pd.DataFrame(vix_hist)
                L.info(f"Successfully primed {len(self.vix_long_history_df)} days of VIX data.")
            else:
                L.error("Failed to prime VIX history. IV Rank filter will be disabled.")

    def process_ticks(self, ticks: List[Dict]):
        for tick in ticks:
            self.bars.add_tick(tick)

    def _reconcile_bars(self):
        with self.bars.lock:
            try:
                now = now_ist()
                if not (self.timings_config["market_open"] < now.time() < self.timings_config["eod_flatten_time"]):
                    return

                with self.bars.lock:
                    for token in [self.nifty_token, self.bn_token]:
                        if not token:
                            continue
                        
                        # --- FIXED: Use OrderActor ---
                        hist_data = None
                        params = {
                            "instrument_token": token,
                            "from_date": now - timedelta(minutes=5),
                            "to_date": now,
                            "interval": "minute"
                        }
                        reply_q = queue.Queue()
                        self.trader.order_actor.q.put({
                            "type": "historical_data",
                            "params": params,
                            "reply_q": reply_q
                        })
                        try:
                            resp = reply_q.get(timeout=10.0)
                            if resp['ok']:
                                hist_data = resp['res']
                        except queue.Empty:
                            L.warning("Timeout reconciling bars")
                        # --- END FIX ---
                        
                        if not hist_data:
                            continue

                        hist_df = pd.DataFrame(hist_data)
                        bar_df = self.bars.data.get(token, {}).get(1)
                        if bar_df is None or bar_df.empty:
                            continue

                        hist_df['timestamp'] = pd.to_datetime(hist_df['date']).dt.tz_convert(IST)

                        current_bar_ts = now.replace(second=0, microsecond=0)
                        for _, row in hist_df.iterrows():
                            ts = row['timestamp']
                            if ts < current_bar_ts and ts in bar_df.index:
                                bar_df.loc[ts, ['open', 'high', 'low', 'close', 'volume']] = row[['open', 'high', 'low', 'close', 'volume']]
            except Exception as e:
                L.warning(f"Bar reconciliation failed: {e}")

    def _get_iv_rank(self) -> Optional[float]:
        if self.vix_long_history_df.empty or not self.vix_token:
            return None

        current_vix_ltp = self.prices.ltp(self.vix_token)
        if not current_vix_ltp:
            return None

        vix_history = self.vix_long_history_df['close']
        min_vix = vix_history.min()
        max_vix = vix_history.max()

        if max_vix == min_vix:
            return 50.0

        iv_rank = ((current_vix_ltp - min_vix) / (max_vix - min_vix)) * 100
        return iv_rank

    def _get_oi_barriers(self, underlying_name: str, expiry: date, spot: float) -> Tuple[Optional[float], Optional[float]]:
        try:
            chain = self.book.get_option_chain(underlying_name, expiry)
            if chain.empty:
                return None, None

            calls = chain[chain['instrument_type'] == 'CE']
            puts = chain[chain['instrument_type'] == 'PE']

            resistance_strike = calls[calls['strike'] > spot].sort_values('open_interest', ascending=False).head(1)
            support_strike = puts[puts['strike'] < spot].sort_values('open_interest', ascending=False).head(1)

            res = resistance_strike.iloc[0]['strike'] if not resistance_strike.empty else None
            sup = support_strike.iloc[0]['strike'] if not support_strike.empty else None
            return res, sup
        except Exception as e:
            L.warning(f"Could not get OI barriers: {e}")
            return None, None

    
    def _run_strategic_planner(self):
        """
        The Planner loop: Checks halts, scans strategies, scores signals,
        picks the best candidate, performs final sizing, checks risk, and queues the trade.
        Runs at a faster interval (e.g., 2 seconds).
        """
        start_time = time.perf_counter() # Timer starts *before* lock
            
        with self.master_lock:
            logic_start_time = time.perf_counter() # Timer for logic starts *after* lock
            now = now_ist()

            # --- 0. Check/Update Daily State ---
            is_trading_day = self.nse_calendar.valid_days(start_date=now.date(), end_date=now.date()).size > 0
            if self.last_trading_day is None or (now.date() >= self.last_trading_day and is_trading_day):
                 if self.last_trading_day is None or now.date() > self.last_trading_day:
                     self._reset_daily_state()
            elif not is_trading_day:
                 if self.last_trading_day is not None: self._reset_daily_state()
                 return # Not a trading day


            # --- 1. Theta Filter (Master Guard Clause) ---
            theta_filter_active = self.trading_config.get("activate_theta_filter", True)
            theta_filter_iv_rank_threshold = self.trading_config.get("theta_filter_iv_rank_threshold", 25.0)
            is_theta_halt_condition_met = False
            if theta_filter_active:
                iv_rank = self._get_iv_rank()
                if self.regime == Regime.CHOP and iv_rank is not None and iv_rank < theta_filter_iv_rank_threshold:
                    is_theta_halt_condition_met = True
                    if not getattr(self, '_theta_halt_active', False):
                        L.warning(f"THETA FILTER ENGAGED: Regime=CHOP, IV Rank ({iv_rank:.1f}%) < Threshold ({theta_filter_iv_rank_threshold}%). Setting REGIME halt.")
                        send_alert(f"⚠️ THETA FILTER ENGAGED: CHOP & Low IV Rank ({iv_rank:.1f}%). Halting regime-specific trades.")
                        # --- FIX: Use regime_halt ---
                        self.regime_halt = True
                        self._theta_halt_active = True
                        if G_HALTED_STATUS: G_HALTED_STATUS.set(1)

            # Check for resumption separately
            if getattr(self, '_theta_halt_active', False) and not is_theta_halt_condition_met:
                if not os.path.exists(KILL_SWITCH_FILE): 
                    L.info("Theta Filter conditions cleared. Resuming regime-specific trading.")
                    send_alert("✅ Theta Filter conditions cleared. Resuming.")
                    # --- FIX: Use regime_halt ---
                    self.regime_halt = False
                    self._theta_halt_active = False
                    # Only set global halt to 0 if master_halt is ALSO false
                    if G_HALTED_STATUS and not self.master_halt: G_HALTED_STATUS.set(0)
                else:
                    self._theta_halt_active = False
                    L.warning("Theta Filter cleared, but Kill Switch active. Trading remains halted.")

            # --- 2. Update Regime Classification (Needed for scoring even if halted) ---
            self.run_regime_classification(now)
            # --- 2a. Regime Stability/Confidence Check ---
            low_conf_thresh = self.trading_config.get("low_regime_confidence_threshold", 0.0)
            flip_rate_thresh = self.trading_config.get("high_regime_flip_rate_threshold", 99)
            
            flips_last_hour = 0
            if self.regime_change_history:
                one_hour_ago = now - timedelta(hours=1)
                flips_last_hour = sum(1 for ts in self.regime_change_history if ts > one_hour_ago)
            
            is_unstable = False
            if self.regime_confidence < low_conf_thresh:
                L.warning(f"REGIME STABILITY HALT: Confidence ({self.regime_confidence:.2f}) is below threshold ({low_conf_thresh}). Setting REGIME halt.")
                is_unstable = True
                
            if flips_last_hour > flip_rate_thresh:
                L.warning(f"REGIME STABILITY HALT: Flip rate ({flips_last_hour}/hr) is above threshold ({flip_rate_thresh}). Setting REGIME halt.")
                is_unstable = True

            if is_unstable:
                if not self.regime_halt: # Only log/alert on the transition
                     send_alert(f"⚠️ REGIME UNSTABLE: Confidence {self.regime_confidence:.2f}, Flip Rate {flips_last_hour}/hr. Halting regime-specific entries.", "warning")
                # --- FIX: Use regime_halt ---
                self.regime_halt = True
                if G_HALTED_STATUS: G_HALTED_STATUS.set(1)
            elif not is_theta_halt_condition_met: # Don't resume if theta filter is still on
                if self.regime_halt and not os.path.exists(KILL_SWITCH_FILE):
                    L.info("Regime has stabilized. Resuming regime-specific trading.")
                    send_alert("✅ Regime stabilized. Resuming regime-specific trades.")
                self.regime_halt = False
                if G_HALTED_STATUS and not self.master_halt: G_HALTED_STATUS.set(0)


            # --- 3. Master Halt Check (Blocks EVERYTHING) ---
            if self.master_halt:
                # L.debug("MASTER halt active. Skipping all strategy evaluation.")
                pass # Allow function to exit and run profiling
            else:
                # --- 4. Timing & Cooldown Guard Clauses ---
                now_time = now.time()
                final_entry_time = self.timings_config["final_entry_time"]
                is_expiry_day = self.book.find_nearest_expiry_date("NIFTY") == now.date()
                if is_expiry_day and self.timings_config.get("final_expiry_entry_time"):
                     final_entry_time = self.timings_config["final_expiry_entry_time"]

                if not (self.timings_config["market_settling_time"] <= now_time < final_entry_time):
                    pass # Allow function to exit
                else:
                    cooldown_ok = not self.last_trade_timestamp or (now - self.last_trade_timestamp) > timedelta(minutes=self.trading_config["trade_cooldown_minutes"])
                    if not cooldown_ok:
                        pass # Allow function to exit
                    else:
                        # --- 5. Universe Scan ---
                        strategies_to_run = []
                        if self.regime in self.strategies: strategies_to_run.extend(self.strategies[self.regime])
                        if "AGNOSTIC" in self.strategies: strategies_to_run.extend(self.strategies["AGNOSTIC"])
                        
                        if not strategies_to_run:
                            pass # Allow function to exit
                        else:
                            universe_tokens = [self.nifty_token, self.bn_token]
                            all_potential_signals = [] 

                            for token in universe_tokens:
                                if not token: continue
                                
                                cooldown_minutes = self.trading_config.get("trade_cooldown_minutes", 1)
                                last_trade_time = self.underlying_cooldown.get(token)
                                if last_trade_time and (now - last_trade_time) < timedelta(minutes=cooldown_minutes):
                                    continue
                                
                                for strategy in strategies_to_run:
                                
                                    # --- FIX 1: Apply Halt Logic ---
                                    # If regime_halt is on, only allow agnostic strategies
                                    if self.regime_halt and not strategy.is_agnostic:
                                        continue # Skip this regime-specific strategy
                                    # --- END FIX 1 ---

                                    try:
                                        if signal := strategy.evaluate(token, self.regime, now):
                                            all_potential_signals.append({"signal": signal, "token": token})
                                    except Exception as e:
                                         L.error(f"Error evaluating strategy {strategy.name.value} on token {token}: {e}", exc_info=True)


                            # --- REFACTOR START (FIX 4: Biased Signal Selection) ---
                            
                            if not all_potential_signals:
                                pass # No signals found
                            else:
                                # --- 6. Score All Signals (Gets Preliminary Params) ---
                                scored_trades = []
                                for potential in all_potential_signals:
                                    signal, token = potential["signal"], potential["token"]
                                    try:
                                        trade_package = self._score_and_size_trade(signal, token)
                                        if trade_package:
                                            scored_trades.append({
                                                "package": trade_package,
                                                "signal": signal,
                                                "token": token
                                            })
                                    except Exception as e:
                                        L.error(f"Error scoring signal {signal.strategy_name.value} on token {token}: {e}", exc_info=True)

                                if not scored_trades:
                                    pass # No signals passed min score
                                else:
                                    # --- 7. Perform Final Sizing for ALL Candidates ---
                                    fully_sized_candidates = []
                                    for candidate in scored_trades:
                                        signal = candidate["signal"]
                                        token = candidate["token"]
                                        score = candidate["package"]["score"]
                                        
                                        try:
                                            final_sized_params = self.get_trade_params(
                                                token=token,
                                                side=signal.side,
                                                risk_points_on_underlying=signal.risk_points,
                                                reward_points_on_underlying=signal.reward_points,
                                                strategy=signal.strategy_name.value,
                                                regime=self.regime,
                                                confidence_score=score # Use the REAL score
                                            )
                                            
                                            if final_sized_params and final_sized_params.get('lots', 0) > 0:
                                                fully_sized_candidates.append({
                                                    "score": score, 
                                                    "params": final_sized_params,
                                                    "is_agnostic": candidate["package"]["is_agnostic"]
                                                })
                                            else:
                                                L.debug(f"Signal {signal.strategy_name.value} dropped at final sizing (0 lots).")

                                        except Exception as e:
                                            L.error(f"Error during final sizing for {signal.strategy_name.value}: {e}", exc_info=True)

                                    if not fully_sized_candidates:
                                        pass # All candidates sized to 0 lots
                                    else:
                                        # --- 8. Final Risk Check on ALL Sized Candidates ---
                                        risk_approved_candidates = []
                                        for candidate in fully_sized_candidates:
                                            try:
                                                if self.risk_manager.risk_ok(hypothetical_params=candidate['params']):
                                                    risk_approved_candidates.append(candidate)
                                                else:
                                                    L.debug(f"Signal {candidate['params']['strategy']} dropped by master risk controls.")
                                            except Exception as e:
                                                L.error(f"Error during final risk check for {candidate['params']['strategy']}: {e}", exc_info=True)
                                    
                                    if not risk_approved_candidates:
                                        pass # All viable signals blocked by risk manager
                                    else:
                                        # --- 9. Pick the BEST from the final, approved list ---
                                        # Sort by score (desc), prefer non-agnostic in ties
                                        best_trade = max(risk_approved_candidates, key=lambda x: (x['score'], not x['is_agnostic']))
                                        
                                        # --- 10. Queue the single best trade ---
                                        L.info(f"==> Queuing trade for {best_trade['params']['strategy']} on {best_trade['params']['opt']['tradingsymbol']} ({best_trade['params']['lots']} lots, Score: {best_trade['score']:.2f})")
                                        self.trade_signal_queue.put(best_trade['params'])
                                        self.last_trade_timestamp = now
                                        self.underlying_cooldown[best_trade['params']['opt']['instrument_token']] = now
                            
                            # --- REFACTOR END ---
        
        # --- PROFILING LOGIC (runs outside the master_lock) ---
        logic_end_time = time.perf_counter()
        
        exec_time_ms = (logic_end_time - logic_start_time) * 1000.0
        lock_wait_ms = (logic_start_time - start_time) * 1000.0
        
        try:
            interval_s = self.scheduler["strategic_planner"][1]
            interval_ms = interval_s * 1000.0
            warn_threshold_ms = interval_ms * 0.5 # 50% threshold
        except (AttributeError, KeyError, IndexError, TypeError):
            interval_ms = 2000.0 # Fallback
            warn_threshold_ms = 1000.0 
        
        if exec_time_ms > warn_threshold_ms:
            L.warning(f"PERF WARN: _run_strategic_planner logic took {exec_time_ms:.2f}ms. (Threshold: {warn_threshold_ms:.2f}ms, Lock Wait: {lock_wait_ms:.2f}ms)")
        else:
            L.debug(f"Strategic planner executed. Logic: {exec_time_ms:.2f}ms, Lock Wait: {lock_wait_ms:.2f}ms.")

    def _trade_executor_worker(self):
        L.info("Trade executor worker started.")
        while self.running.is_set():
            try:
                trade_params = self.trade_signal_queue.get(timeout=1)

                if not self.risk_manager.risk_ok(hypothetical_params=trade_params):
                    L.warning(f"Trade for {trade_params['strategy']} rejected by final risk check just before execution.")
                    continue

                L.info(f"Executor received signal for {trade_params['strategy']}. Executing trade.")
                if self.trader.open_position(trade_params):
                    with self.master_lock:
                        self.last_trade_timestamp = now_ist()

            except Empty:
                continue
            except Exception as e:
                L.error(f"FATAL Error in trade executor worker: {e}", exc_info=True)

    def _handle_entry_fill(self, pos: Position, order: Dict):
        with self.master_lock:
            filled_qty = order.get('filled_quantity', 0)

        if order.get('status') == 'OPEN' and filled_qty > 0:
            pos.is_entry_order_open = True
        elif order.get('status') in ['COMPLETE', 'CANCELLED', 'REJECTED']:
            pos.is_entry_order_open = False

        if filled_qty > pos.qty:
            new_fills = filled_qty - pos.qty
            L.info(f"✅ Entry fill received for {pos.tradingsymbol}. Qty: {new_fills}. Total Filled: {filled_qty}.")

            if pos.status == PositionStatus.PENDING_ENTRY.value:
                # This method (trader._get_order_avg_price) is already refactored
                avg_price = self.trader._get_order_avg_price(pos.entry_order_id) 
                if not avg_price:
                    send_alert(f"CRITICAL: Could not get avg price for entry {pos.entry_order_id}. Closing position.", "critical")
                    self.trader.close_position(pos, "AVG_PRICE_FAILURE")
                    return

                pos.entry_price = avg_price
                pos.qty = filled_qty
                pos.high_price_since_entry = avg_price
                pos.opened_at = now_ist()
                pos.initial_sl_price = avg_price - pos.option_sl_points
                pos.sl_price = pos.initial_sl_price
                pos.tp_price = avg_price + pos.option_tp_points
                self.risk_manager.initialize_position_greeks(pos)

            pos.qty = filled_qty
            pos.status = PositionStatus.OPEN_AWAITING_BRACKETS
            
            # --- FIXED: Use StoreActor ---
            self.store_actor.q.put({"type": "upsert_position", "pos": pos})
            # --- END FIX ---

        if order.get('status') in ['COMPLETE', 'CANCELLED', 'REJECTED']:
            if pos.qty == 0:
                L.warning(f"Entry order {pos.entry_order_id} for {pos.tradingsymbol} {order.get('status')} with no fills. Removing position.")
                self.trader.positions.pop(pos.id, None)
            else:
                L.info(f"Entry order for {pos.tradingsymbol} is final. Total filled: {pos.qty}/{pos.initial_qty}.")
                pos.status = PositionStatus.OPEN_AWAITING_BRACKETS
                
                # --- FIXED: Use StoreActor ---
                self.store_actor.q.put({"type": "upsert_position", "pos": pos})
                # --- END FIX ---
                self.risk_manager.verify_position_risk(pos)

    def _handle_exit_fill(self, pos: Position, order: Dict):
        with self.master_lock:
            if pos.status == PositionStatus.CLOSED.value:
                L.info(f"Ignoring duplicate fill for already closed position {pos.tradingsymbol}")
                return

        reason = pos.exit_reason or "EXIT_FILL"
        L.info(f"Exit order {order.get('order_id')} ({reason}) complete for {pos.id}.")
        # This method (trader._cancel_all_open_orders_for_pos) is already refactored
        self.trader._cancel_all_open_orders_for_pos(pos, cancel_entry=pos.is_entry_order_open)

        # This method (trader._get_order_avg_price) is already refactored
        exit_price = self.trader._get_order_avg_price(order.get('order_id')) or self.prices.ltp(pos.token)
        if not exit_price and reason == "TP_HIT":
            exit_price = pos.tp_price
        if not exit_price:
            L.error(f"Could not determine exit price for {pos.tradingsymbol}!")
            return

        final_filled_qty = order.get('filled_quantity', 0)
        pnl_for_this_exit = 0.0 # Initialize PnL for this specific exit

        if final_filled_qty > 0:
            pnl_for_this_exit = (exit_price - pos.entry_price) * final_filled_qty
            self.trader.daily_realized_pnl += pnl_for_this_exit # Add only the PnL from this final chunk
            
            # --- FIXED: Use StoreActor ---
            self.store_actor.q.put({
                "type": "log_strategy_performance",
                "name": pos.strategy,
                "pnl": pnl_for_this_exit
            })
            # --- END FIX ---
            
            L.info(f"Final exit fill PnL contribution: {pnl_for_this_exit:.2f} for {final_filled_qty} qty.")
        else:
            L.warning(f"Final exit order {order.get('order_id')} for {pos.tradingsymbol} completed with 0 fills? PnL contribution is 0.")

        pos.status = PositionStatus.CLOSED.value
        pos.exit_price = exit_price # Store the final average exit price

        with self.risk_manager.lock:
            if final_filled_qty > 0 and pos.greeks: 
                delta_change = pos.greeks.get("delta", 0.0) * final_filled_qty
                vega_change = pos.greeks.get("vega", 0.0) * final_filled_qty
                gamma_change = pos.greeks.get("gamma", 0.0) * final_filled_qty
                theta_change = pos.greeks.get("theta", 0.0) * final_filled_qty

                self.risk_manager.portfolio_greeks["net_delta"] -= delta_change
                self.risk_manager.portfolio_greeks["net_vega"] -= vega_change
                self.risk_manager.portfolio_greeks["net_gamma"] -= gamma_change
                self.risk_manager.portfolio_greeks["net_theta"] -= theta_change
                L.info(f"Removed greeks for final {final_filled_qty} qty of {pos.tradingsymbol}. "
                    f"Δ: {-delta_change:.2f}, V: {-vega_change:.2f}, Γ: {-gamma_change:.4f}, Θ: {-theta_change:.2f}")
            elif final_filled_qty == 0:
                L.warning(f"Skipping greek removal for {pos.tradingsymbol} as final filled qty is 0.")
            elif not pos.greeks:
                L.warning(f"Skipping greek removal for {pos.tradingsymbol} as pos.greeks is empty.")

        pos.greeks = {}

        # --- FIXED: Use StoreActor ---
        self.store_actor.q.put({"type": "upsert_position", "pos": pos})
        self.store_actor.q.put({
            "type": "log_closed_trade",
            "pos": pos,
            "price": exit_price,
            "reason": reason
        })
        # --- END FIX ---
        
        self.risk_manager.update_performance_metrics(pnl_for_this_exit)
        self.trader.positions.pop(pos.id, None)
        send_alert(f"❌ CLOSED {pos.tradingsymbol} ({reason}). Final Piece PnL: {pnl_for_this_exit:.2f}. Daily PnL: {self.trader.daily_realized_pnl:.2f}")

    def _handle_order_update_from_queue(self, order: Dict):
        oid, status = str(order.get('order_id')), order.get('status')
        pos_id = f"LIVE_{oid}"

        with self.trader.lock:
            pos = self.trader.positions.get(pos_id)
            if not pos:
                pos = next((p for p in self.trader.positions.values() if oid in [p.tp_order_id, p.exit_order_id, p.slm_order_id] or oid in p.partial_exit_order_ids), None)
                if not pos:
                    L.debug(f"Received order update for {oid} but no matching position found.")
                    return

            if oid == pos.entry_order_id:
                self._handle_entry_fill(pos, order)

            elif (oid in [pos.tp_order_id, pos.exit_order_id, pos.slm_order_id] or oid in pos.partial_exit_order_ids) and status == 'COMPLETE':
                if oid in pos.partial_exit_order_ids:
                    # This method (trader._handle_partial_exit_fill) is already refactored
                    self.trader._handle_partial_exit_fill(pos, order)
                else:
                    if oid == pos.slm_order_id:
                        pos.exit_reason = "SL_HIT_BROKER"
                    elif oid == pos.tp_order_id:
                        pos.exit_reason = "TP_HIT_BROKER"
                    self._handle_exit_fill(pos, order)
            
            elif status == 'REJECTED':
                if oid == pos.entry_order_id:
                    L.error(f"Entry order {oid} for {pos.tradingsymbol} REJECTED. Reason: {order.get('status_message')}")
                    pos.status = PositionStatus.REJECTED.value
                    pos.exit_reason = f"ENTRY_REJECTED: {order.get('status_message')}"
                    
                    # --- FIXED: Use StoreActor ---
                    self.store_actor.q.put({"type": "upsert_position", "pos": pos})
                    # --- END FIX ---
                    
                    self.trader.positions.pop(pos.id, None)
                elif oid in [pos.tp_order_id, pos.slm_order_id]:
                    L.critical(f"Bracket order {oid} ({'TP' if oid == pos.tp_order_id else 'SL'}) for {pos.tradingsymbol} REJECTED: {order.get('status_message')}")
                    send_alert(f"🔥 CRITICAL: BRACKET ORDER REJECTED for {pos.tradingsymbol}. Closing position!", "critical")
                    self.trader.close_position(pos, "BRACKET_REJECTED")
            
            elif status == 'CANCELLED':
                L.info(f"Order {oid} for {pos.tradingsymbol} was CANCELLED.")
                # This is informational. The fill handlers manage state.
                pass

    # Insert this method inside the Engine class in main1.py
    def run_regime_classification(self, current_time: datetime):
            """
            Runs the classifier and applies hysteresis to prevent "flip-flopping".
            The official regime only changes after the new signal is confirmed
            for `regime_confirmation_threshold` consecutive cycles.
            """
            try:
                # 1. Get the RAW, unconfirmed signal from the classifier
                raw_regime, active_token, raw_confidence = self.classifier.get_raw_classification(self.regime)
                
                # 2. Apply Hysteresis (Confirmation) Logic
                if raw_regime == self.potential_regime:
                    # Signal is the same as last cycle, increment counter
                    self.potential_regime_count += 1
                else:
                    # Signal is new, reset the potential regime and counter
                    self.potential_regime = raw_regime
                    self.potential_regime_count = 1
                    L.debug(f"Regime Hysteresis: New potential regime {raw_regime.name}. Awaiting confirmation...")

                # 3. Check for Official Regime Change
                is_confirmed = self.potential_regime_count >= self.regime_confirmation_threshold
                is_new_regime = self.potential_regime != self.regime
                
                if is_confirmed and is_new_regime:
                    # --- This is the official change ---
                    old_regime_name = self.regime.name
                    self.regime = self.potential_regime
                    self.regime_confidence = raw_confidence # Use the latest confidence
                    
                    # Track change for flip rate
                    self.regime_change_history.append(current_time)
                    
                    # Update Prometheus Metric
                    if G_CURRENT_REGIME:
                        try:
                            G_CURRENT_REGIME.clear()
                            G_CURRENT_REGIME.labels(regime_name=self.regime.name).set(self.regime.value)
                        except Exception as e:
                            L.warning(f"Failed to set Prometheus regime gauge: {e}")

                    self.last_regime_change_time = current_time
                    log_msg = f"REGIME CHANGE CONFIRMED: {old_regime_name} -> {self.regime.name} (Conf: {self.regime_confidence:.2f})"
                    L.info(log_msg)
                    send_alert(log_msg)
                
                elif self.regime == self.potential_regime:
                    # Regime is stable, just update the confidence score
                    self.regime_confidence = raw_confidence

                # Handle the active token (this can update every cycle)
                if active_token:
                    self.active_underlying_token = active_token

            except Exception as e:
                L.error(f"FATAL Error in regime classification: {e}", exc_info=True)
                self.regime = Regime.UNCLEAR
                self.regime_confidence = 0.0

    def _find_best_option_contract(self, underlying_token: int, expiry: date, option_type: OptionType, strategy: str, regime: Regime) -> Optional[Dict]:
        spot = self.prices.ltp(underlying_token)
        if not spot:
            return None

        strategy_obj = next((s for s_list in self.strategies.values() for s in s_list if s.name.value == strategy), None)
        if not strategy_obj:
            return None

        now = now_ist()
        market_close_time = self.timings_config["market_close"]
        expiry_date = expiry.date() if isinstance(expiry, pd.Timestamp) else expiry
        T = _calculate_time_to_expiry(expiry_date, now, market_close_time)
        dte = (expiry_date - now.date()).days

        strike_cfg = self.trading_config['strike_selection']
        
        if dte <= 1: # Expiry day or day before
             target_delta = strike_cfg.get('expiry_delta', 0.65)
        elif dte <= 3: # Expiry week
             target_delta = strike_cfg.get('expiry_week_delta', 0.55)
        elif regime in [Regime.TRENDING_UP, Regime.TRENDING_DOWN]:
             target_delta = strike_cfg.get('trend_delta', 0.45)
        elif regime == Regime.CHOP:
             target_delta = strike_cfg.get('chop_delta', 0.60)
        else: # COMPRESSION
             target_delta = strike_cfg.get('compression_delta', 0.50)

        underlying_symbol = self.book.get_symbol(underlying_token)
        underlying_name = _get_underlying(underlying_symbol)

        step = self.book.step_size(underlying_name)
        num_strikes = 5

        otm_strikes_chain = self.book.get_option_chain(underlying_name, expiry)
        otm_calls = otm_strikes_chain[(otm_strikes_chain['instrument_type'] == 'CE') & (otm_strikes_chain['strike'] > spot)].sort_values('strike').head(num_strikes)
        otm_puts = otm_strikes_chain[(otm_strikes_chain['instrument_type'] == 'PE') & (otm_strikes_chain['strike'] < spot)].sort_values('strike', ascending=False).head(num_strikes)

        iv_analysis_chain = pd.concat([otm_calls, otm_puts])
        if iv_analysis_chain.empty:
            return None

        now = now_ist()
        market_close_time = self.timings_config["market_close"]
        underlying_bars = self.bars.get_ohlc(underlying_token, 1)
        hv = calculate_historical_volatility(underlying_bars['close'], timeframe_minutes=1) if not underlying_bars.empty else 0.3

        avg_iv = self._calculate_atm_iv(underlying_token, spot, expiry, T, hv)
        if not avg_iv:
    # Fallback if helper fails
            L.warning(f"ATM IV calculation failed for {underlying_name}. Using HV {hv} as fallback.")
            avg_iv = hv or 0.3

# Use the cached IV if it's available, otherwise use the one we just calculated
        avg_iv = self.atm_iv_cache.get(underlying_name, avg_iv)

        if underlying_token not in self.historical_avg_iv:
            self.historical_avg_iv[underlying_token] = pd.Series(dtype=float)

        s = self.historical_avg_iv[underlying_token]
        self.historical_avg_iv[underlying_token] = pd.concat([s, pd.Series([avg_iv], index=[now])])
        self.historical_avg_iv[underlying_token] = self.historical_avg_iv[underlying_token].last('4H')
        lookback = self.technical_config.get("iv_contraction_lookback", 120)
        iv_series = self.historical_avg_iv[underlying_token]

        is_in_contraction = False
        if len(iv_series) > lookback:
            iv_percentile = iv_series.rolling(lookback).rank(pct=True).iloc[-1]
            if iv_percentile < self.technical_config.get("iv_contraction_threshold_pct", 10) / 100.0:
                is_in_contraction = True
                L.info(f"VOLATILITY CONTRACTION DETECTED for {underlying_name}. Avg IV Pct Rank: {iv_percentile*100:.2f}%")

        if strategy == StrategyName.MOMENTUM_BREAKOUT.value and not is_in_contraction:
            L.info(f"MomentumBreakout signal for {underlying_name} skipped. Not in IV contraction phase.")
            return None

        full_chain = self.book.get_option_chain(underlying_name, expiry)
        trade_chain = full_chain[full_chain['instrument_type'] == option_type.value].copy()
        atm_strike = round(spot / step) * step
        search_range = 15 * step
        trade_chain = trade_chain[(trade_chain['strike'] >= atm_strike - search_range) & (trade_chain['strike'] <= atm_strike + search_range)]
        if trade_chain.empty:
            return None

        min_volume = self.trading_config['option_selection_filters']['min_option_volume']
        min_oi = self.trading_config['option_selection_filters']['min_option_oi']
        options_with_metrics = []
        for _, row in trade_chain.iterrows():
            token = int(row['instrument_token'])
            tick = self.prices.get_full_tick(token)
            if not tick:
                continue
            if tick.get('volume', 0) < min_volume or tick.get('open_interest', 0) < min_oi:
                continue
            ltp = tick.get('last_price')
            if not ltp or ltp < self.trading_config['min_option_price']:
                continue
            depth = tick.get('depth')
            if not depth or not depth.get('buy') or not depth.get('sell'):
                continue
            bid_price, ask_price = depth['buy'][0]['price'], depth['sell'][0]['price']
            spread = (ask_price - bid_price) / ask_price if ask_price > 0 else float('inf')
            if spread > self.trading_config['max_bid_ask_spread_pct'] / 100.0:
                continue

            iv = calculate_iv(ltp, spot, row['strike'], T, 0.05, option_type == OptionType.CE, hv_fallback=hv)
            greeks = calculate_greeks(spot, row['strike'], T, 0.05, iv, option_type == OptionType.CE)
            options_with_metrics.append({'delta_diff': abs(abs(greeks['delta']) - target_delta), 'opt': row.to_dict(), 'ltp': ltp, 'greeks': greeks})

        if not options_with_metrics:
            return None
        return min(options_with_metrics, key=lambda x: x['delta_diff'])

    def get_trade_params(
        self,
        token: int,
        side: OrderSide,
        risk_points_on_underlying: float,
        reward_points_on_underlying: float,
        strategy: str,
        regime: Regime,
        confidence_score: float
    ) -> Optional[Dict]:
        underlying_symbol = self.book.get_symbol(token)
        if not underlying_symbol:
            return None

        underlying_name = _get_underlying(underlying_symbol)
        lot_size = self.book.lot_size(underlying_name)
        expiry = self.book.find_nearest_expiry_date(underlying_name)

        if not all([lot_size, expiry]):
            return None
        option_type = OptionType.CE if side == OrderSide.BUY else OptionType.PE
        best_option_data = self._find_best_option_contract(underlying_token=token, expiry=expiry, option_type=option_type, strategy=strategy, regime=regime)
        if not best_option_data:
            return None

        option_contract, option_ltp, greeks = best_option_data['opt'], best_option_data['ltp'], best_option_data['greeks']

        estimated_delta = greeks['delta']
        spot_price = self.prices.ltp(token)
        if not spot_price:
            return None

        resistance, support = self._get_oi_barriers(underlying_name, expiry, spot_price)
        oi_profit_target = None
        min_dist_pct = self.trading_config['option_selection_filters'].get('min_dist_from_oi_wall_pct', 0.25)
        if side == OrderSide.BUY and resistance and (resistance - spot_price < (spot_price * (min_dist_pct / 100))):
            L.warning(f"Trade blocked. Too close to Call OI wall at {resistance}.")
            return None
        elif side == OrderSide.SELL and support and (spot_price - support < (spot_price * (min_dist_pct / 100))):
            L.warning(f"Trade blocked. Too close to Put OI wall at {support}.")
            return None
        oi_profit_target = resistance if side == OrderSide.BUY else support
        final_sl_points_on_option = risk_points_on_underlying * abs(estimated_delta)
        max_sl_pct = self.trading_config["max_sl_pct_of_premium"]
        if (option_ltp > 0) and (final_sl_points_on_option / option_ltp) > (max_sl_pct / 100.0):
            L.warning(f"Trade REJECTED: Calculated SL ({final_sl_points_on_option:.2f}) exceeds max {max_sl_pct}% of premium ({option_ltp}).")
            return None
        risk_per_lot = final_sl_points_on_option * lot_size
        number_of_lots = self.risk_manager.calculate_position_size(
            token, 
            risk_per_lot, 
            vega_per_lot=greeks['vega'] * lot_size,
            confidence_score=confidence_score,  # <-- PASS
            strategy_name=strategy                # <-- PASS
        )

        if number_of_lots <= 0:
            L.warning(f"Trade REJECTED: Calculated lot size is {number_of_lots}.")
            return None

        total_trade_risk = risk_per_lot * number_of_lots
        if total_trade_risk > (self.risk_manager.dynamic_account_equity * 0.05):
            L.critical(f"Trade REJECTED: Calculated risk ₹{total_trade_risk:.2f} exceeds 5% of equity. Risk logic error?")
            return None

        underlying_sl_level = (spot_price - risk_points_on_underlying) if side == OrderSide.BUY else (spot_price + risk_points_on_underlying)
        final_tp_points_on_option = reward_points_on_underlying * abs(estimated_delta)
        strategy_config = self.config['strategies'].get(strategy, {})

        return {
            "opt": option_contract, "ltp_opt": option_ltp, "lots": number_of_lots,
            "strategy": strategy, "regime": regime.name, "option_sl_points": final_sl_points_on_option,
            "option_tp_points": final_tp_points_on_option, "total_trade_risk": total_trade_risk,
            "underlying_sl": underlying_sl_level, "greeks": greeks,
            "max_trade_duration_minutes": strategy_config.get('max_duration_minutes', 90),
            "oi_profit_target": oi_profit_target, "intended_risk_rupees": total_trade_risk
        }

    def reconcile(self):
        if PAPER_TRADING:
            return
        L.info("--- Starting State Reconciliation with Broker ---")
        try:
            # --- FIXED: Use OrderActor ---
            L.debug("Reconcile: Fetching broker positions...")
            pos_reply_q = queue.Queue()
            self.trader.order_actor.q.put({"type": "positions", "reply_q": pos_reply_q})
            
            broker_positions_data = None
            try:
                pos_resp = pos_reply_q.get(timeout=10.0)
                if pos_resp['ok']:
                    broker_positions_data = pos_resp['res']
                else:
                    raise Exception(pos_resp.get('error'))
            except Exception as e:
                L.error(f"Reconcile: Failed to get broker positions from OrderActor: {e}")
                return
            # --- END FIX ---

            if not broker_positions_data:
                L.warning("Could not get broker positions for reconciliation.")
                return

            # --- FIXED: Use StoreActor ---
            L.debug("Reconcile: Fetching DB positions...")
            db_reply_q = queue.Queue()
            self.store_actor.q.put({"type": "load_open_positions", "reply_q": db_reply_q})
            
            db_positions = {}
            try:
                db_resp = db_reply_q.get(timeout=10.0)
                if db_resp['ok']:
                    db_positions = db_resp['res']
                else:
                    raise Exception(db_resp.get('error'))
            except Exception as e:
                L.error(f"Reconcile: Failed to get DB positions from StoreActor: {e}")
                return
            # --- END FIX ---

            broker_positions_raw = broker_positions_data.get('net', [])
            broker_positions_map = {pos['tradingsymbol']: pos for pos in broker_positions_raw if pos.get('product') == 'MIS' and abs(pos.get('quantity', 0)) > 0}
            broker_symbols, db_symbols = set(broker_positions_map.keys()), {p.tradingsymbol for p in db_positions.values()}

            for symbol in db_symbols - broker_symbols:
                pos = next((p for p in db_positions.values() if p.tradingsymbol == symbol), None)
                if pos:
                    send_alert(f"RECONCILE: DB has {symbol} but broker does not. Marking as closed.", "warning")
                    pos.status = PositionStatus.CLOSED.value
                    pos.exit_reason = "RECONCILE_GHOST_CLOSE"
                    
                    # --- FIXED: Use StoreActor ---
                    self.store_actor.q.put({"type": "upsert_position", "pos": pos})
                    # --- END FIX ---

            for symbol in broker_symbols - db_symbols:
                rogue_pos_data = broker_positions_map[symbol]
                qty = rogue_pos_data['quantity']
                send_alert(f"🔥 RECONCILE: Rogue position for {symbol} (Qty: {qty}) found at broker! Auto-flattening.", "critical")
                transaction_type = self.trader.TRANSACTION_TYPE_SELL if qty > 0 else self.trader.TRANSACTION_TYPE_BUY
                
                # --- FIXED: Use OrderActor ---
                L.info(f"Reconcile: Placing market order to flatten rogue {symbol}...")
                place_params = {
                    "variety": self.trader.VARIETY_REGULAR, "exchange": "NFO", "tradingsymbol": symbol,
                    "transaction_type": transaction_type, "quantity": abs(qty),
                    "product": self.trader.PRODUCT_MIS, "order_type": self.trader.ORDER_TYPE_MARKET
                }
                reply_q = queue.Queue()
                self.trader.order_actor.q.put({
                    "type": "place_order",
                    "params": place_params,
                    "reply_q": reply_q
                })
                
                oid = None
                try:
                    resp = reply_q.get(timeout=10.0)
                    if resp['ok'] and resp['res']:
                        oid = resp['res'].get('order_id')
                except queue.Empty:
                    L.error("Timeout placing flatten order")
                # --- END FIX ---

                if oid is None:
                    send_alert(f"🔥🔥 FATAL: FAILED to auto-flatten rogue position {symbol}. MANUAL INTERVENTION REQUIRED!", "critical")
                else:
                    L.info(f"Placed market order {oid} to flatten rogue position {symbol}.")

            L.info("--- Reconciliation Complete ---")
            
            # --- FIXED: Use StoreActor (to reload) ---
            L.debug("Reconcile: Reloading trader positions from DB...")
            reload_reply_q = queue.Queue()
            self.store_actor.q.put({"type": "load_open_positions", "reply_q": reload_reply_q})
            try:
                reload_resp = reload_reply_q.get(timeout=10.0)
                if reload_resp['ok']:
                    with self.trader.lock:
                        self.trader.positions = reload_resp['res']
                else:
                    raise Exception(reload_resp.get('error'))
            except Exception as e:
                L.error(f"Reconcile: Failed to reload trader positions: {e}")
            # --- END FIX ---

        except Exception as e:
            L.error(f"Reconciliation failed: {e}", exc_info=True)

    def health_check(self):
        if not self.prices.connected.is_set():
            send_alert("🔥 CRITICAL: PriceBus WebSocket is disconnected!", "critical")
            if G_WS_CONNECTED:
                G_WS_CONNECTED.set(0)
            return
        else:
            if G_WS_CONNECTED:
                G_WS_CONNECTED.set(1)

        now = now_ist()
        if self.timings_config["market_open"] < now.time() < self.timings_config["market_close"]:
            stale_feed = False
            for token in [self.nifty_token, self.bn_token]:
                if not token:
                    continue
                last_tick_time = self.prices.last_tick_reception_time_per_token.get(token)
                symbol_name = self.book.get_symbol(token) or f"Token {token}"
                if last_tick_time:
                    age = (now - last_tick_time).total_seconds()
                    if G_LAST_TICK_AGE_SECONDS:
                        G_LAST_TICK_AGE_SECONDS.set(age)
                    if age > 120:
                        send_alert(f"🔥 CRITICAL: Stale Feed for {symbol_name}! No ticks for {age:.0f}s.", "critical")
                        stale_feed = True
                else:
                    send_alert(f"🔥 CRITICAL: No ticks *ever* received for {symbol_name}.", "critical")
                    stale_feed = True

            with self.master_lock:
                if stale_feed:
                    if not self.master_halt:
                        L.warning("Stale data protocol activated. Halting new entries.")
                        send_alert("🔥 CRITICAL: Stale Feed! Halting new trades.", "critical")
                        self.master_halt = True
                        if G_HALTED_STATUS:
                            G_HALTED_STATUS.set(1)
                else:
                    # Feed is healthy
                    # Check master_halt, but only resume if regime_halt is ALSO false
                    if self.master_halt and not self.regime_halt and not os.path.exists(KILL_SWITCH_FILE):
                        L.info("Data feed is healthy. Resuming trading.")
                        send_alert("✅ Data feed healthy. Resuming new trades.", "info")
                        self.master_halt = False
                        if G_HALTED_STATUS:
                            G_HALTED_STATUS.set(0)
            if self.running.is_set():
                critical_threads_to_check = {
                "TickProcessor": self.tick_thread,
                "OrderProcessor": self.order_thread,
                "TradeExecutor": self.trade_executor_thread,
                "position_management": self.scheduler_threads.get("position_management"),
                "strategic_planner": self.scheduler_threads.get("strategic_planner")
            }

        for name, thread_obj in critical_threads_to_check.items():
            if thread_obj and not thread_obj.is_alive():
                with self.master_lock:
                    if self.running.is_set() and not self.fatal_error_event.is_set():
                        msg = f"🔥 CRITICAL: Worker thread '{name}' appears dead! Initiating shutdown."
                        L.critical(msg)
                        send_alert(msg, "critical")
                        self.fatal_error_event.set() 
                        return 

        L.debug("Worker thread liveness check passed.")

    # Insert this method inside the Engine class in main1.py

def _update_prometheus_metrics(self):
    """
    Updates all configured Prometheus gauges with the current system state.
    This function is intended to be called periodically by the scheduler.
    """
    # Check if Prometheus gauges were initialized (e.g., G_PNL_REALIZED is a good proxy)
    if not G_PNL_REALIZED:
        return # Skip if Prometheus is disabled

    try:
        # --- PnL Metrics ---
        G_PNL_REALIZED.set(self.trader.daily_realized_pnl)
        G_PNL_UNREALIZED.set(self.risk_manager.last_unrealized_pnl) # Assumes risk_manager updates this

        # --- Trading Status ---
        with self.master_lock:
            G_HALTED_STATUS.set(1 if self.master_halt or self.regime_halt else 0)

        # --- Regime Metrics ---
        # Ensure gauges exist before setting
        if G_REGIME_CONFIDENCE:
            G_REGIME_CONFIDENCE.set(self.regime_confidence)

        if G_REGIME_FLIP_RATE:
            flips_last_hour = 0
            if self.regime_change_history: # Check if the deque exists and is not empty
                now = now_ist()
                one_hour_ago = now - timedelta(hours=1)
                # Count timestamps within the last hour
                flips_last_hour = sum(1 for ts in self.regime_change_history if ts > one_hour_ago)
            G_REGIME_FLIP_RATE.set(flips_last_hour)

        # --- WebSocket & Tick Age ---
        # Assuming G_WS_CONNECTED and G_LAST_TICK_AGE_SECONDS are updated elsewhere
        # (e.g., PriceBus callbacks and health_check) - No direct update needed here unless logic changes.

        # --- Portfolio Greeks ---
        with self.risk_manager.lock:
            if G_PORTFOLIO_DELTA:
                G_PORTFOLIO_DELTA.set(self.risk_manager.portfolio_greeks.get("net_delta", 0.0))
            if G_PORTFOLIO_VEGA:
                G_PORTFOLIO_VEGA.set(self.risk_manager.portfolio_greeks.get("net_vega", 0.0))
            if G_PORTFOLIO_GAMMA:
                G_PORTFOLIO_GAMMA.set(self.risk_manager.portfolio_greeks.get("net_gamma", 0.0))
            if G_PORTFOLIO_THETA:
                G_PORTFOLIO_THETA.set(self.risk_manager.portfolio_greeks.get("net_theta", 0.0))

        L.debug("Prometheus metrics updated.")

    except Exception as e:
        L.warning(f"Failed to update Prometheus metrics: {e}", exc_info=True)

    def _persist_bar_data(self):
        try:
            L.info("Persisting in-memory 1-minute bars to disk...")
            os.makedirs(DATA_LOG_DIR, exist_ok=True)
            today_str = date.today().isoformat()

            tokens_to_log = {
                self.nifty_token: f"nifty_fut_1min_{today_str}.csv",
                self.bn_token: f"banknifty_fut_1min_{today_str}.csv",
                self.vix_token: f"india_vix_1min_{today_str}.csv"
            }

            for token, filename in tokens_to_log.items():
                if token is None:
                    continue
                bar_df = self.bars.get_ohlc(token, 1)
                if not bar_df.empty:
                    filepath = os.path.join(DATA_LOG_DIR, filename)
                    bar_df.to_csv(filepath)

            L.info("Bar data persistence complete.")
        except Exception as e:
            L.error(f"Failed to persist bar data: {e}", exc_info=True)

    def stop(self):
        if self.running.is_set():
            L.info("Disengaging Sentinel...")
            self.running.clear()
            if hasattr(self, 'prices') and self.prices.ws:
                self.prices.ws.close()

            L.info("Performing final data persistence before shutdown...")
            self._persist_bar_data()

            if self.last_trading_day:
                # --- FIXED: Use StoreActor ---
                self.store_actor.q.put({
                    "type": "set_kv",
                    "key": f"daily_pnl_{self.last_trading_day}",
                    "value": str(self.trader.daily_realized_pnl)
                })
                # --- END FIX ---
                
            send_alert("🛑 Sentinel PRIME disengaged.")
            L.info("Shutdown complete.")

# ==================================================================================================
# APPLICATION ENTRY POINT
# ==================================================================================================
def login_or_reuse(token_file: str = TOKEN_FILE_PATH) -> Tuple[KiteConnect, str]:
    """
    Handles Kite Connect login by reusing a stored token
    or prompting for a new one if invalid/missing.
    """
    api_key = os.environ.get("KITE_API_KEY")
    api_secret = os.environ.get("KITE_API_SECRET") # NOW CRITICAL for new token generation

    if not api_key:
        raise SystemExit("FATAL: KITE_API_KEY environment variable not set.")
    
    if not api_secret:
        L.warning("KITE_API_SECRET not set. Token reuse will work, but new token generation will fail.")

    kite = KiteConnect(api_key=api_key)
    access_token = None

    # --- 1. Try to reuse an existing token ---
    if os.path.exists(token_file):
        L.info(f"Access token file found at {token_file}. Attempting reuse.")
        try:
            with open(token_file, 'r') as f:
                token_data = json.load(f)
                access_token = token_data.get('access_token')

            if not access_token:
                raise ValueError("Access token not found in token file.")

            kite.set_access_token(access_token)
            
            # Validate the token by making a profile call
            profile = kite.profile()
            L.info(f"Successfully reused access token for user: {profile.get('user_id')}")
            return kite, access_token

        except (KiteException, ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            L.warning(f"Failed to reuse access token from {token_file}: {e}. "
                      f"Will attempt to generate a new one.")
            access_token = None # Ensure token is None to trigger generation
        except Exception as e:
            L.critical(f"FATAL: Unexpected error during token reuse: {e}.", exc_info=True)
            sys.exit(1)
            
    # --- 2. Generate a new token if reuse failed ---
    if access_token is None:
        if not api_secret:
            raise SystemExit("FATAL: KITE_API_SECRET environment variable not set. Cannot generate new token.")
            
        L.info("No valid access token found. Starting new login flow.")
        
        login_url = kite.login_url()
        print("\n" + "="*80)
        print(f"FIRST TIME LOGIN REQUIRED (or token expired):")
        print(f"1. Open this URL in your browser:\n")
        print(login_url)
        print(f"\n2. Log in, and you will be redirected to a blank page.")
        print(f"3. Copy the full URL from your browser's address bar.")
        print(f"   It will look like: https://your-redirect-url.com/?request_token=YOUR_TOKEN_HERE&action=...")
        print(f"4. Paste the *ENTIRE* redirected URL below and press Enter:")
        print("="*80)
        
        try:
            redirect_url = input("Paste redirected URL here: ")
            
            # Extract request_token from the URL
            parsed_url = requests.utils.urlparse(redirect_url)
            query_params = requests.utils.parse_qs(parsed_url.query)
            request_token = query_params.get('request_token', [None])[0]
            
            if not request_token:
                raise ValueError("Could not find 'request_token' in the pasted URL.")

            L.info("Request token received. Generating session...")
            
            # Generate the full session
            session_data = kite.generate_session(request_token, api_secret)
            access_token = session_data.get('access_token')
            
            if not access_token:
                raise ValueError("API did not return an access_token.")
                
            kite.set_access_token(access_token)
            
            # Verify the new token
            profile = kite.profile()
            L.info(f"Successfully generated new token for user: {profile.get('user_id')}")

            # Save the new token for future use
            # Ensure the directory exists
            os.makedirs(PERSIST_DIR, exist_ok=True)
            with open(token_file, 'w') as f:
                json.dump({'access_token': access_token}, f)
            L.info(f"New access token saved to {token_file} for future use.")
            
            return kite, access_token

        except (KiteException, ValueError) as e:
            L.critical(f"FATAL: Token generation failed: {e}")
            sys.exit(1)
        except Exception as e:
            L.critical(f"FATAL: An unexpected error occurred during token generation: {e}", exc_info=True)
            sys.exit(1)

    # This line should not be reachable, but as a fallback:
    raise SystemExit("FATAL: Login function failed to return a valid session.")

def handle_signal(sig, frame):
    """Gracefully shuts down the engine on receiving SIGINT or SIGTERM."""
    L.warning(f"Signal {sig} received. Initiating graceful shutdown...")
    if _engine_instance:
        _engine_instance.stop()
    sys.exit(0)


def main():
    global _engine_instance, PAPER_TRADING
    load_dotenv()
    config = load_config()

    PAPER_TRADING = config['trading'].get('paper_trading', True) or APP_ENV != "PRODUCTION"
    L.warning(f"--- PAPER TRADING MODE IS {'ENABLED' if PAPER_TRADING else 'DISABLED'} ---")
    if not PAPER_TRADING and APP_ENV != "PRODUCTION":
        L.warning("WARNING: Live trading enabled but APP_ENV is not PRODUCTION!")

    try:
        kite_raw, access_token = login_or_reuse()
    except SystemExit as e:
        L.critical(f"Login failed: {e}")
        return

    # --- REFACTOR START ---

    # 1. Create original objects
    store = Store()
    kite_gov = GovernedKite(kite_raw) # This is now ONLY for actors

    # 2. Create Actors (These will be the *only* things that use the objects above)
    order_actor = OrderActor(kite_gov)
    store_actor = StoreActor(store)
    
    # 3. Pass ACTORS as dependencies
    
    # FIXED: InstrumentBook now receives actors, not the original objects.
    # This forces it to use the actor model for all its DB/API calls.
    # NOTE: This requires you to update InstrumentBook's __init__
    book = InstrumentBook(store_actor, order_actor).load()
    
    # PriceBus is OK: It *only* manages the WebSocket, not state-changing API calls.
    prices = PriceBus(kite_gov, access_token)

    # Inject the ACTORS
    risk_manager = RiskManager(None, None, book, prices, store_actor, config)
    micro_monitor = Micromonitor(prices, config)
    pos_manager = PositionManager(None, None, book, prices, store_actor, risk_manager, config)

    # Initialize Trader based on mode, injecting actors (This part was already correct)
    if PAPER_TRADING:
        trader = PaperTrader(None, book, prices, store_actor, config, risk_manager.update_performance_metrics)
    else:
        # Pass original `store` for startup-read, and actors for runtime I/O
        trader = Trader(None, store, store_actor, order_actor, book, prices, config, risk_manager.update_performance_metrics)

    # Inject dependencies back into RiskManager and PositionManager
    risk_manager.trader = trader
    pos_manager.trader = trader

    # FIXED: Engine no longer receives the kite_gov object.
    # This forces it to use self.trader.order_actor for all API calls.
    # NOTE: This requires you to update Engine's __init__
    engine = Engine(store_actor, book, prices, config)
    
    # This injection is now CRITICAL, as Engine relies on it for all broker access.
    engine.set_dependencies(trader, risk_manager, micro_monitor, pos_manager)

    # Inject Engine dependency into modular components that need it
    risk_manager.engine = engine
    pos_manager.engine = engine
    trader.engine = engine # <-- Important injection

    # --- REFACTOR END ---

    _engine_instance = engine # Make engine accessible to signal handler

    # Setup signal handling for graceful shutdown
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if METRICS_APP:
        start_metrics_server(port=config.get("metrics_port", 9095))

    try:
        # 4. Start Actors BEFORE engine
        L.info("Starting StoreActor...")
        store_actor.start()
        if not PAPER_TRADING:
            L.info("Starting OrderActor...")
            order_actor.start()
        
        engine.start() # This blocks until engine stops or is interrupted
    
    except KeyboardInterrupt:
        L.info("Keyboard interrupt received in main. Stopping engine...")
    except SystemExit as e:
        L.warning(f"SystemExit caught in main: {e}")
    except Exception as e:
        L.critical(f"Unhandled FATAL exception in main execution: {e}", exc_info=True)
        send_alert(f"🔥 FATAL ERROR: Unhandled exception caused bot crash: {e}", "critical")
    finally:
        # 5. Add stop/join calls for actors in finally block for clean shutdown
        L.info("Shutting down... Stopping actors.")
        store_actor.stop()
        if not PAPER_TRADING:
            order_actor.stop()
        
        if _engine_instance: # Ensure engine exists before trying to stop
            _engine_instance.stop() 

        # Wait for actors to finish their queues
        L.info("Waiting for StoreActor to join...")
        store_actor.join(timeout=5.0)
        if not PAPER_TRADING:
            L.info("Waiting for OrderActor to join...")
            order_actor.join(timeout=5.0)
        L.info("Shutdown complete.")

if __name__ == "__main__":
    main()