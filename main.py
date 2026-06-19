from __future__ import annotations

import random
import os
import sys
import time
import math
import json
import queue
import dataclasses
import signal
import logging
import threading
import uuid
import gc
import sqlite3
import stat
from enum import Enum, auto
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import List, Dict, Optional, Tuple, Callable, Type
from datetime import datetime, timedelta, time as dtime, date
from abc import ABC, abstractmethod
from collections import deque
from numba import jit



from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pytz
import pandas_market_calendars as mcal
from scipy.optimize import newton
from scipy.stats import norm

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
# MATH UTILITIES
# ==================================================================================================
def _get_d1_d2(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, iv: float) -> Tuple[Optional[float], Optional[float]]:
    if time_to_expiry <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return None, None
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * iv ** 2) * time_to_expiry) / (iv * math.sqrt(time_to_expiry))
    d2 = d1 - iv * math.sqrt(time_to_expiry)
    return d1, d2

# ==================================================================================================
# ACTOR MODEL - FOR CONCURRENCY SAFETY
# ==================================================================================================
L_ACTORS = logging.getLogger("SENTINEL-PRIME.ACTORS")

class OrderActor(threading.Thread):
    """A thread-safe actor for all broker API I/O with internal rate limiting."""
    def __init__(self, kite_client: GovernedKite):
        super().__init__(daemon=True, name="OrderActor")
        self.q = queue.Queue()
        self.kite = kite_client
        self.running = True
        # Rate Limiter: 3 tokens/sec, max 10 tokens burst
        self.tokens = 10.0
        self.last_update = time.time()
        self.rate_lock = threading.Lock()

    def _wait_for_token(self):
        while True:
            with self.rate_lock:
                now = time.time()
                elapsed = now - self.last_update
                self.last_update = now
                # FIX: 2.5 tokens/sec instead of 3.0 to allow overhead
                self.tokens = min(10.0, self.tokens + (elapsed * 2.5)) 
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
            time.sleep(0.1)

    def run(self):
        while self.running:
            try:
                msg = self.q.get(timeout=1.0)
                if msg is None:
                    continue

                msg_type = msg['type']
                reply_q = msg.get('reply_q') 

                # --- RATE LIMITING ---
                # Only CANCEL operations are allowed to bypass the rate limiter.
                # place_order MUST consume tokens to prevent spam bans.
                if msg_type not in ["cancel_all_orders", "cancel_order"]:
                     self._wait_for_token()
                # ---------------------

                try:
                    res = None
                    # --- Write/Trade Methods ---
                    if msg_type == "place_order":
                        res = self.kite.place_order(**msg['params'])
                    elif msg_type == "modify_order":
                        try:
                            res = self.kite.modify_order(**msg['params'])
                        except KiteException as e:
                            # RACE CONDITION HANDLER
                            # If error is "NetworkException" or order id not found, it likely closed.
                            # We log it as a warning, NOT an error, and don't crash the thread.
                            err_str = str(e).lower()
                            if "unable to modify" in err_str or "does not exist" in err_str or "completed" in err_str:
                                L_ACTORS.warning(f"Race Condition caught: Tried to modify dead order. Ignoring. ({e})")
                                res = {"status": "RACE_CONDITION_IGNORED"}
                            else:
                                raise e # Real error, re-raise
                    elif msg_type == "cancel_order":
                        res = self.kite.cancel_order(**msg['params'])
                    
                    # --- Panic / Risk Management Methods ---
                    elif msg_type == "cancel_all_orders":
                        try:
                            orders = self.kite.orders()
                            count = 0
                            for o in orders:
                                if o['status'] == 'OPEN':
                                    try:
                                        self.kite.cancel_order(variety='regular', order_id=o['order_id'])
                                        count += 1
                                    except: pass
                            res = f"Cancelled {count} orders"
                        except Exception as e:
                            res = f"Cancel All Failed: {e}"

                    # --- Read/Data Methods ---
                    elif msg_type == "order_history":
                        res = self.kite.order_history(**msg['params'])
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
        self.q.put(None)
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
    
class LeakyQueue(queue.Queue):
    """A Queue that drops the oldest item when full (LIFO eviction for fresh data)."""
    def put(self, item, block=True, timeout=None):
        if self.full():
            try:
                self.get_nowait() # Discard oldest tick
            except queue.Empty:
                pass
        super().put(item, block, timeout)

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
    MOMENTUM_IGNITION = "MomentumIgnition" # Replaces MomentumBreakout
    SLINGSHOT = "Slingshot"     
    MEAN_REVERSION = "MeanReversion"
    VOLATILITY_MEAN_REVERSION = "VolatilityMeanReversion"
    OPENING_RANGE_BREAKOUT = "OpeningRangeBreakout"
    TREND_PULLBACK = "TrendPullback"
    GAMMA_BURST = "GammaBurst"


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
    underlying_entry_price: float = 0.0
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
                timeout=2
            )
            return
        except Exception as e:
            L.warning(f"Telegram alert failed on attempt {attempt+1}: {e}")
            time.sleep(0.5)

def calculate_trading_time_to_expiry(
    current_datetime: datetime, 
    expiry_date: date, 
    market_open_time: dtime, 
    market_close_time: dtime,
    nse_calendar
) -> float:
    """Calculates accurate trading time to expiry using market calendar."""
    total_session_minutes = (
        (market_close_time.hour * 60 + market_close_time.minute) -
        (market_open_time.hour * 60 + market_open_time.minute)
    )
    if total_session_minutes <= 0: total_session_minutes = 375
    TRADING_DAYS_PER_YEAR = 252.0
    MINUTES_PER_TRADING_YEAR = TRADING_DAYS_PER_YEAR * total_session_minutes
    current_date = current_datetime.date()
    current_time = current_datetime.time()

    if current_date > expiry_date or (current_date == expiry_date and current_time >= market_close_time):
        return 1e-9 

    if current_date == expiry_date:
        if current_time < market_open_time: minutes_remaining = total_session_minutes
        else:
            minutes_remaining = ((market_close_time.hour * 60 + market_close_time.minute) -
                                 (current_time.hour * 60 + current_time.minute))
        return max(1e-9, minutes_remaining / MINUTES_PER_TRADING_YEAR)

    try:
        trading_days = nse_calendar.valid_days(start_date=current_date, end_date=expiry_date)
        trading_days_left = len(trading_days)
        is_today_trading_day = current_date in trading_days
        if is_today_trading_day and current_time >= market_close_time: trading_days_left -= 1
        return max(1e-9, trading_days_left / TRADING_DAYS_PER_YEAR)
    except Exception as e:
        L.warning(f"Calendar calculation failed: {e}. Fallback to calendar days.")
        days_left = (expiry_date - current_date).days
        return max(1e-9, (days_left * (TRADING_DAYS_PER_YEAR / 365.25)) / TRADING_DAYS_PER_YEAR)

# --- NUMBA JIT ACCELERATED MATH CORE ---

@jit(nopython=True, cache=True, nogil=True)
def _numba_cdf(x):
    """Fast approximation of Cumulative Distribution Function"""
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))

@jit(nopython=True, cache=True, nogil=True)
def _numba_pdf(x):
    """Fast approximation of Probability Density Function"""
    return 0.3989422804014327 * math.exp(-0.5 * x * x)

@jit(nopython=True, cache=True, nogil=True)
def fast_greeks_jit(S, K, T, r, sigma, is_call):
    """
    Calculates Price and all Greeks in a single pass using machine code.
    Returns: (price, delta, vega, gamma, theta)
    """
    if T <= 1e-9 or sigma <= 1e-9 or S <= 0 or K <= 0:
        # Return intrinsics for safety
        intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
        d_int = 1.0 if (is_call and S > K) else -1.0 if (not is_call and S < K) else 0.0
        return intrinsic, d_int, 0.0, 0.0, 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    cdf_d1 = _numba_cdf(d1)
    cdf_d2 = _numba_cdf(d2)
    pdf_d1 = _numba_pdf(d1)
    
    if is_call:
        price = S * cdf_d1 - K * math.exp(-r * T) * cdf_d2
        delta = cdf_d1
        theta = (- (S * pdf_d1 * sigma) / (2 * sqrt_T) - r * K * math.exp(-r * T) * _numba_cdf(d2)) / 365.0
    else:
        cdf_neg_d1 = _numba_cdf(-d1)
        cdf_neg_d2 = _numba_cdf(-d2)
        price = K * math.exp(-r * T) * cdf_neg_d2 - S * cdf_neg_d1
        delta = cdf_d1 - 1.0
        theta = (- (S * pdf_d1 * sigma) / (2 * sqrt_T) + r * K * math.exp(-r * T) * _numba_cdf(-d2)) / 365.0
        
    vega = S * pdf_d1 * sqrt_T / 100.0
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    
    return price, delta, vega, gamma, theta

@jit(nopython=True, cache=True, nogil=True)
def implied_vol_jit(target_price, S, K, T, r, is_call):
    """Fast Newton-Raphson IV solver."""
    sigma = 0.5 # Initial guess
    for i in range(20):
        p, d, v, g, th = fast_greeks_jit(S, K, T, r, sigma, is_call)
        diff = target_price - p
        if abs(diff) < 1e-4:
            return sigma
        if v == 0:
            break
        sigma = sigma + diff / (v * 100.0) # Vega is scaled by 100 in fast_greeks
    return sigma

# --- WRAPPERS FOR COMPATIBILITY ---

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    p, _, _, _, _ = fast_greeks_jit(float(S), float(K), float(T), float(r), float(sigma), is_call)
    return p

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> Dict[str, float]:
    """Calculates primary greeks using Numba JIT backend."""
    _, delta, vega, gamma, theta = fast_greeks_jit(float(S), float(K), float(T), float(r), float(sigma), is_call)
    return {
        "delta": delta,
        "vega": vega,
        "gamma": gamma,
        "theta": theta
    }

def calculate_iv(target_price: float, S: float, K: float, T: float, r: float, is_call: bool, initial_guess: float = 0.5, hv_fallback: Optional[float] = None) -> float:
    """Calculates Implied Volatility using fast JIT solver."""
    try:
        iv = implied_vol_jit(float(target_price), float(S), float(K), float(T), float(r), is_call)
        if iv <= 0 or iv > 5.0 or math.isnan(iv): # Sanity check
             return hv_fallback or initial_guess
        return iv
    except:
        return hv_fallback or initial_guess

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
        # L.warning("Not enough 1m data for dynamic risk calculation.")
        return 0.0, 0.0

    atr_short = ohlc_df_1m.ta.atr(length=5).iloc[-1]
    atr_long = ohlc_df_1m.ta.atr(length=50).iloc[-1]

    if atr_long == 0 or pd.isna(atr_short) or pd.isna(atr_long):
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
            cursor.execute("PRAGMA journal_mode=WAL;")   # Enable non-blocking writes
            cursor.execute("PRAGMA synchronous=NORMAL;")
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
        self.bar_queue = queue.Queue(maxsize=10000)

        self.on_connect_callbacks = []
        self.connected = threading.Event()
        self.ws_thread = None
        self.tick_queue = LeakyQueue(maxsize=100)
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
        
        # 1. Fast Path (Signals)
        try:
            self.tick_queue.put_nowait(ticks)
        except queue.Full:
            pass # Drop oldest (Leaky)

        # 2. Reliable Path (Bars)
        try:
            self.bar_queue.put_nowait(ticks)
        except queue.Full:
            L.warning("⚠️ CRITICAL: Bar Queue FULL. Dropping market data to keep WS alive.")

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
    """
    Optimized In-Memory Store. 
    Uses deques for O(1) writes. Builds DataFrame only on read O(N).
    Removes the 'pd.concat' bottleneck from the hot path.
    """
    def __init__(self, timeframes: List[int]):
        self.lock = threading.RLock()
        self.timeframes = sorted(timeframes)
        # Structure: {token: {tf: deque([ {'ts':..., 'open':...}, ... ]) }}
        # Deque maxlen ensures automatic trimming of old bars
        self.buffer: Dict[int, Dict[int, deque]] = {} 
        # Active Bar Tracker
        self.active_bars: Dict[int, Dict[int, Dict]] = {}

    def _ensure_token_data(self, token: int):
        if token not in self.buffer:
            # Maxlen 2000 stores enough history for indicators without memory bloat
            self.buffer[token] = {tf: deque(maxlen=2000) for tf in self.timeframes}
            self.active_bars[token] = {tf: None for tf in self.timeframes}

    def prime(self, token: int, hist_df: pd.DataFrame, append: bool = False):
        """Loads initial historical data into the deque buffers."""
        with self.lock:
            if hist_df.empty: return
            self._ensure_token_data(token)
            
            ts_col = pd.to_datetime(hist_df['date'])
            hist_df['timestamp'] = ts_col.dt.tz_convert(IST) if ts_col.dt.tz else ts_col.dt.tz_localize(IST)
            base_df = hist_df.set_index("timestamp").drop(columns=['date'], errors='ignore')

            for tf in self.timeframes:
                # Resample historical data
                resampled = base_df if tf == 1 else base_df.resample(f'{tf}min').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna()
                
                if resampled.empty: continue
                
                # Convert to list of dicts for deque
                # reset_index to keep timestamp as a column for the dict
                records = resampled.reset_index().to_dict('records')
                
                # Rename timestamp column to 'ts' to match internal schema if needed, 
                # or just ensure get_ohlc handles it. Let's standardize on 'ts'.
                for r in records:
                    r['ts'] = r.pop('timestamp')
                
                if append:
                    self.buffer[token][tf].extend(records)
                else:
                    self.buffer[token][tf] = deque(records, maxlen=2000)
                    
            L.info(f"Primed BarStore for {token} with deque buffers.")

    def add_tick(self, tick: Dict) -> Optional[List[int]]:
        """
        Processes a new tick. 
        O(1) operation using deque append. Zero DataFrame allocations.
        """
        token, ts, price, qty = tick.get('instrument_token'), tick.get('exchange_timestamp'), tick.get('last_price'), tick.get('last_traded_quantity')
        if not all([token, ts, price, qty is not None]): return None
        
        updated_tfs = []
        with self.lock:
            self._ensure_token_data(token)
            for tf in self.timeframes:
                # Calculate bar start time
                bar_ts = ts.replace(second=0, microsecond=0) - timedelta(minutes=ts.minute % tf)
                current = self.active_bars[token][tf]
                
                # --- Condition 1: New Bar Started ---
                if current is None or current['ts'] != bar_ts:
                    if current is not None:
                        # Commit completed bar to deque (O(1))
                        self.buffer[token][tf].append(current)
                        updated_tfs.append(tf)
                    
                    # Initialize new active bar
                    self.active_bars[token][tf] = {
                        'ts': bar_ts, 
                        'open': price, 'high': price, 'low': price, 'close': price, 'volume': qty
                    }
                
                # --- Condition 2: Update Active Bar ---
                else:
                    if price > current['high']: current['high'] = price
                    elif price < current['low']: current['low'] = price
                    current['close'] = price
                    current['volume'] += qty
                    
        return updated_tfs

    def get_ohlc(self, token: int, timeframe: int) -> pd.DataFrame:
        """
        Returns DataFrame for analysis. 
        Builds DF on-demand (O(N)). Only called by Strategy Logic, not Tick Processor.
        """
        with self.lock:
            if token not in self.buffer or timeframe not in self.buffer[token]: 
                return pd.DataFrame()
            
            # 1. Copy deque to list (Fast)
            data = list(self.buffer[token][timeframe])
            
            # 2. Append active bar if exists (Real-time view)
            active = self.active_bars.get(token, {}).get(timeframe)
            if active:
                data.append(active)
                
            if not data: return pd.DataFrame()
            
            # 3. Create DataFrame (The only heavy op, but decoupled from tick stream)
            df = pd.DataFrame(data)
            df.set_index('ts', inplace=True)
            df.index.name = 'timestamp'
            return df


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
    def __init__(self, engine, book, prices, store_actor, config, perf_callback=None):
        super().__init__(engine, book, prices, None, config, perf_callback)
        self.store_actor = store_actor

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
            if dataclasses.is_dataclass(p):
                partial_pos_log_data = dataclasses.replace(p, id=partial_trade_id, initial_qty=qty_to_close, qty=qty_to_close)
            else:
                # Fallback for safety if Position is refactored to a standard class
                import copy
                partial_pos_log_data = copy.copy(p)
                partial_pos_log_data.id = partial_trade_id
                partial_pos_log_data.initial_qty = qty_to_close
                partial_pos_log_data.qty = qty_to_close
        
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
    # --- Open Position (Fixed: Respects FVG Limit Price) ---
    def open_position(self, trade_params: Dict) -> Optional[Position]:
        with self.lock:
            opt = trade_params['opt']
            ts = opt['tradingsymbol']
            strategy_name = trade_params['strategy']
            
            # 1. Pre-Trade Checks
            open_pos_count = sum(1 for p in self.positions.values() if p.status not in [PositionStatus.CLOSED.value, PositionStatus.REJECTED.value])
            lot_size = self.book.lot_size(_get_underlying(ts))

            if open_pos_count >= self.max_concurrent or not lot_size:
                L.warning(f"Trade rejected: Max concurrent ({self.max_concurrent}) or no lot size.")
                return None

            qty = int(int(trade_params['lots']) * lot_size)
            if qty <= 0: return None

            # 2. Create Position Object
            pos = Position(
                id=f"TEMP_{uuid.uuid4()}",
                status=PositionStatus.PENDING_SUBMISSION.value, 
                tradingsymbol=ts, token=int(opt['instrument_token']), option_type=opt['instrument_type'],
                qty=0, initial_qty=qty, entry_price=0, initial_sl_price=0, sl_price=0, tp_price=0,
                opened_at=now_ist(), strategy=strategy_name, market_regime_at_entry=trade_params['regime'],
                underlying_sl_level=trade_params['underlying_sl'],
                option_sl_points=trade_params['option_sl_points'],
                option_tp_points=trade_params['option_tp_points'],
                initial_risk_points=trade_params['option_sl_points'],
                scale_out_rules=self.config['trading']['scale_out_rules'],
                max_trade_duration_minutes=trade_params.get('max_trade_duration_minutes', 90),
                oi_profit_target=trade_params.get('oi_profit_target'),
                underlying_entry_price=trade_params.get('underlying_price', 0.0),
                intended_risk_rupees=trade_params.get('total_trade_risk', 0.0),
                last_entry_modification=now_ist()
            )
            self.positions[pos.id] = pos

            # --- 3. Execution Routing ---
            
            ltp = self.prices.ltp(pos.token) or 0
            if ltp == 0: 
                self.positions.pop(pos.id)
                return None
            
            tick_size = self.book.tick_size(ts) or 0.05
            place_params = {}

            # --- CHECK 1: FVG SNIPER (Specific Limit Price) ---
            # If the Strategy detected a Fair Value Gap, we use the pre-calculated limit.
            if trade_params.get('is_fvg_entry', False) and trade_params.get('ltp_opt', 0) > 0:
                target_limit = trade_params.get('ltp_opt')
                limit_price = round(target_limit / tick_size) * tick_size
                
                place_params = {
                    "variety": "regular", "exchange": "NFO", "tradingsymbol": ts,
                    "transaction_type": "BUY", "quantity": qty,
                    "product": "MIS", 
                    "order_type": "LIMIT", "price": limit_price,
                    "validity": "DAY" # Ensuring we wait for the pullback fill
                }
                L.info(f"🎯 FVG SNIPER EXECUTION: {ts} Limit @ {limit_price} (Waiting for Retest)")
                pos.entry_stage = 88 

            # --- CHECK 2: LEVIATHAN / MOMENTUM (Aggressive Limit) ---
            # Catches "Leviathan" (New), "MomentumIgnition" (Old), "Slingshot"
            elif strategy_name == "Leviathan" or strategy_name in [StrategyName.MOMENTUM_IGNITION.value, StrategyName.SLINGSHOT.value, "GammaBurst", "GAMMA_BURST_FLASH"]:
                
                # A. Calculate Base Price (Aggressive)
                full_tick = self.prices.get_full_tick(pos.token)
                if full_tick and full_tick.get('depth') and full_tick['depth'].get('sell'):
                    ask_price = full_tick['depth']['sell'][0]['price']
                else:
                    ask_price = ltp

                # Bid 0.5% ABOVE Ask. 
                # This acts like a Market Order (fills instantly) but protects against 10% freak spikes.
                limit_price = round((ask_price * 1.005) / tick_size) * tick_size
                
                # B. FREAK TRADE GUARD (The Safety Ceiling)
                # We calculate a dynamic ceiling using ATR. If limit > ceiling, we cap it.
                try:
                    df_atr = self.engine.get_ohlc(pos.token, 1)
                    # Use 14-period ATR or 1% of LTP if data is missing
                    current_atr = df_atr.ta.atr(14).iloc[-1] if (not df_atr.empty and len(df_atr) > 15) else (ltp * 0.01)
                    
                    # Ceiling = LTP + 2x ATR (A very wide band, but stops absolute madness)
                    sanity_ceiling = ltp + (current_atr * 2.0)
                    
                    if limit_price > sanity_ceiling:
                        L.warning(f"⚠️ FREAK GUARD ACTIVATED: Capping limit {limit_price} to {sanity_ceiling} for {ts}")
                        limit_price = round(sanity_ceiling / tick_size) * tick_size
                except Exception as e:
                    L.error(f"Freak Guard Calculation Failed: {e}")
                    # Hard Fallback: Max 2% above LTP
                    limit_price = min(limit_price, round((ltp * 1.02) / tick_size) * tick_size)

                # IDEMPOTENCY KEY: Prevents double-fills on network timeout
                # Take the UUID part of the ID: "LIVE_123e45..." -> "123e45..."
                raw_uuid = pos.id.split('_')[-1] 
                # Remove hyphens to fit more data into 20 chars
                clean_tag = raw_uuid.replace('-', '')
                # Take the last 20 characters. Kite limit is 20.
                order_tag = clean_tag[-20:] 

                place_params = {
                    "variety": "regular", "exchange": "NFO", "tradingsymbol": ts,
                    "transaction_type": "BUY", "quantity": qty,
                    "product": "MIS", 
                    "order_type": "LIMIT", "price": limit_price, 
                    "validity": "DAY", # <--- CRITICAL: Replaced IOC with DAY
                    "tag": order_tag 
                }
                L.info(f"🐋 LEVIATHAN EXECUTION: {ts} Aggressive Limit @ {limit_price}")
                pos.entry_stage = 99 # Skip adaptive logic, we have a hard limit

            # --- CHECK 3: STANDARD / ADAPTIVE (Trend Pullback) ---
            else:
                # 🐢 Passive Bid Entry (The Trapper)
                full_tick = self.prices.get_full_tick(pos.token)
                bid_price = full_tick['depth']['buy'][0]['price'] if full_tick and full_tick.get('depth') else ltp
                
                place_params = {
                    "variety": "regular", "exchange": "NFO", "tradingsymbol": ts,
                    "transaction_type": "BUY", "quantity": qty,
                    "product": "MIS", 
                    "order_type": "LIMIT", "price": bid_price
                }
                L.info(f"🐢 ADAPTIVE ENTRY ({strategy_name}): {ts} @ {bid_price}")
                pos.entry_stage = 1

            # 4. Submit to Actor
            self.store_actor.q.put({"type": "upsert_position", "pos": pos})
            
            reply_q = queue.Queue()
            self.order_actor.q.put({"type": "place_order", "params": place_params, "reply_q": reply_q})

            try:
                resp = reply_q.get(timeout=2.0)
                if resp['ok'] and resp['res'] and 'order_id' in resp['res']:
                    oid = str(resp['res']['order_id'])
                    self.positions.pop(pos.id) # Remove TEMP
                    
                    pos.id = f"LIVE_{oid}"
                    pos.entry_order_id = oid
                    pos.status = PositionStatus.PENDING_ENTRY.value
                    
                    self.positions[pos.id] = pos
                    self.store_actor.q.put({"type": "upsert_position", "pos": pos})
                    self.prices.subscribe([pos.token])
                    return pos
                else:
                    raise Exception(resp.get('error'))
            except Exception as e:
                L.error(f"Order Placement Failed: {e}")
                self.positions.pop(pos.id, None)
                return None
    
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
        """
        Closes position using a 'Market Protection' Limit Order.
        Places Limit at 5% adverse excursion to ensure fill but prevent freak-trade slippage.
        """
        with self.lock:
            if p.status in [PositionStatus.PENDING_CLOSURE.value, PositionStatus.CLOSED.value, PositionStatus.PENDING_SL_EXIT.value]:
                L.debug(f"close_position called for {p.tradingsymbol} ({p.id}) but already closing/closed ({p.status}). Skipping.")
                return True 

            L.info(f"Closing {p.tradingsymbol} ({reason})...")
            
            # 1. Mark status immediately to prevent double submission
            original_status = p.status
            p.status = PositionStatus.PENDING_CLOSURE.value
            p.exit_reason = reason
            self.store_actor.q.put({"type": "upsert_position", "pos": p})

            # 2. Cancel existing brackets (Atomic safety)
            self._cancel_all_open_orders_for_pos(p, cancel_entry=(original_status == PositionStatus.PENDING_ENTRY.value))

            if p.qty <= 0:
                L.warning(f"Attempted to close {p.tradingsymbol} ({p.id}) with zero/negative quantity ({p.qty}). Marking closed.")
                p.status = PositionStatus.CLOSED.value
                p.exit_reason = reason + "_ZERO_QTY"
                self.store_actor.q.put({"type": "upsert_position", "pos": p})
                self.positions.pop(p.id, None) 
                return True

            # Inside close_position...
    # REPLACE the "Calculate Protection Limit Price" block with this:

            ltp = self.prices.ltp(p.token) or p.entry_price
            tick_size = self.book.tick_size(p.tradingsymbol)
            
            # CHASE LOGIC: 
            # If buying to close (Short), Bid + 5%
            # If selling to close (Long), Ask - 5%
            # This guarantees a fill like a Market Order but prevents "Freak Trade" execution at zero.
            
            if p.qty > 0: # We are Long, need to Sell
                # Target 5% BELOW LTP to ensure fill
                limit_price = round((ltp * 0.95) / tick_size) * tick_size
                transaction_type = "SELL"
            else: # We are Short, need to Buy
                # Target 5% ABOVE LTP
                limit_price = round((ltp * 1.05) / tick_size) * tick_size
                transaction_type = "BUY"

            place_params = {
                "variety": "regular", "exchange": "NFO", "tradingsymbol": p.tradingsymbol, 
                "transaction_type": transaction_type, "quantity": abs(p.qty), 
                "product": "MIS", "order_type": "LIMIT", "price": limit_price
            }
            
            L.info(f"Sending PROTECTION LIMIT exit order for {p.qty} of {p.tradingsymbol} ({p.id}) via OrderActor. Reason: {reason}.")
            reply_q = queue.Queue()
            self.order_actor.q.put({"type": "place_order", "params": place_params, "reply_q": reply_q})

            oid = None
            try:
                resp = reply_q.get(timeout=10.0) 
                if resp['ok'] and resp['res'] and 'order_id' in resp['res']:
                    oid = resp['res']['order_id']
                    L.info(f"Exit order {oid} placed successfully via OrderActor for {p.tradingsymbol}.")
                else: raise Exception(resp.get('error', 'Unknown error placing exit'))
            except queue.Empty: L.critical(f"Timeout waiting for EXIT order response for {p.tradingsymbol}!")
            except Exception as e: L.critical(f"EXIT ORDER FAILED for {p.tradingsymbol}. Error: {e}")

            if oid is None:
                send_alert(f"🔥 CRITICAL: FAILED TO PLACE EXIT for {p.tradingsymbol} ({p.id}). POSITION IS STILL OPEN. Manual intervention required!", "critical")
                p.exit_reason = reason + "_EXIT_API_FAIL" # Keep PENDING_CLOSURE
                self.store_actor.q.put({"type": "upsert_position", "pos": p})
                return False 
            else:
                p.exit_order_id = str(oid)
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
            p.status = PositionStatus.OPEN_AWAITING_BRACKETS.value
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
        with self.lock:
            if not p.slm_order_id: return
            
            tick_size = self.book.tick_size(p.tradingsymbol) or 0.05
            new_trigger = round(new_trigger / tick_size) * tick_size
            
            if new_trigger <= p.sl_price: return

            # 1. Send Request with Reply Queue
            reply_q = queue.Queue()
            L.info(f"⚡ Modifying SL for {p.tradingsymbol} to {new_trigger}...")
            
            self.order_actor.q.put({
                "type": "modify_order",
                "params": {
                    "variety": "regular",
                    "order_id": str(p.slm_order_id),
                    "trigger_price": new_trigger
                },
                "reply_q": reply_q 
            })

            # 2. Wait for Acknowledgement (The Pessimistic Check)
            try:
                resp = reply_q.get(timeout=2.0) # Fast timeout
                if resp['ok']:
                    # ...
                    p.sl_price = new_trigger
                    self.store_actor.q.put({"type": "upsert_position", "pos": p})
                else:
                    L.error(f"❌ Broker REJECTED SL modify: {resp.get('error')}")
            except queue.Empty:
                L.error(f"❌ Timeout waiting for SL modify ack on {p.tradingsymbol}. Keeping old SL.")
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
    """Master Risk Manager: Async Checks, Circuit Breaker, Equity Curve, Calmar."""
    def __init__(self, engine: 'Engine', trader: AbstractTrader, book: InstrumentBook, prices: PriceBus, store_actor: StoreActor, config: Dict):
        self.engine = engine
        self.trader = trader
        self.book = book
        self.prices = prices
        self.store_actor = store_actor
        self.config = config
        self.trading_config = config["trading"]
        self.portfolio_limits = self.trading_config.get("portfolio_limits", {})
        
        self.dynamic_account_equity = self.trading_config.get("account_equity", 100000.0)
        self.daily_high_water_mark = self.dynamic_account_equity
        self.weekly_high_water_mark = self.dynamic_account_equity
        self.account_equity_base = self.dynamic_account_equity
        
        self.performance_score = 0
        self.consecutive_losses, self.risk_factor = 0, 1.0
        self.portfolio_greeks: Dict[str, float] = {"net_delta": 0.0, "net_vega": 0.0, "net_gamma": 0.0, "net_theta": 0.0}
        self.last_unrealized_pnl = 0.0
        
        self.last_equity_check = time.time()
        self.last_equity_value = self.dynamic_account_equity
        self.circuit_breaker_triggered = False
        self.lock = threading.RLock()
        
        self.strategy_weights: Dict[str, float] = {}
        self.strategy_perf_lookback_days = self.trading_config.get("strategy_perf_lookback_days", 7)
        self.strategy_weight_min = self.trading_config.get("strategy_weight_min", 0.5)
        self.strategy_weight_max = self.trading_config.get("strategy_weight_max", 1.5)
        
        # Start Async Worker
        threading.Thread(target=self._async_margin_worker, daemon=True, name="RiskMarginWorker").start()

    def _async_margin_worker(self):
        while True:
            if not self.engine or not hasattr(self.engine, 'trader') or not self.engine.trader:
                time.sleep(1); continue
            try:
                reply_q = queue.Queue()
                if self.engine.trader.order_actor:
                    self.engine.trader.order_actor.q.put({"type": "margins", "reply_q": reply_q}, timeout=1.0)
                    resp = reply_q.get(timeout=5.0)
                    if resp['ok'] and resp['res'] and 'equity' in resp['res']:
                        with self.lock: self.dynamic_account_equity = float(resp['res']['equity']['net'])
            except Exception: pass
            time.sleep(60)

    def check_circuit_breaker(self):
        if self.circuit_breaker_triggered: return
        with self.lock:
            now = time.time()
            if now - self.last_equity_check > 10:
                curr_eq = self.dynamic_account_equity + self.trader.unrealized_pnl()
                pct = (curr_eq - self.last_equity_value) / self.last_equity_value if self.last_equity_value else 0
                if pct < -0.02: 
                    L.critical(f"⚡ CIRCUIT BREAKER: Dropped {pct*100:.2f}% in 10s.")
                    self.trigger_panic_exit()
                self.last_equity_value = curr_eq; self.last_equity_check = now

    def trigger_panic_exit(self):
        self.circuit_breaker_triggered = True
        self.engine.master_halt = True
        if G_HALTED_STATUS: G_HALTED_STATUS.set(1)
        self.engine.trader.order_actor.q.put({"type": "cancel_all_orders", "params": {}})
        with self.trader.lock:
            for p in list(self.trader.positions.values()):
                if p.status not in ["CLOSED"]:
                    self.engine.trader.order_actor.q.put({
                        "type": "place_order", 
                        "params": {"variety": "regular", "exchange": "NFO", "tradingsymbol": p.tradingsymbol, "transaction_type": "SELL" if p.qty > 0 else "BUY", "quantity": abs(p.qty), "product": "MIS", "order_type": "MARKET"}
                    })

    def _calculate_calmar_ratio(self, pnl_series: pd.Series) -> float:
        if pnl_series.empty or len(pnl_series) < 2: return 0.0
        daily_ret = pnl_series / self.account_equity_base
        cum_ret = (1 + daily_ret).cumprod()
        dd = (cum_ret - cum_ret.cummax()).min()
        return ((cum_ret.iloc[-1] - 1) / abs(dd)) if dd != 0 else 0.0

    def _update_strategy_weights(self):
        try:
            reply_q = queue.Queue()
            self.store_actor.q.put({"type": "get_strategy_performance", "lookback_days": 15, "reply_q": reply_q})
            resp = reply_q.get(timeout=5)
            if resp['ok'] and not resp['res'].empty:
                df = resp['res']
                df['close_time'] = pd.to_datetime(df['close_time'])
                metrics = df.set_index('close_time').groupby('strategy_name')['pnl'].resample('D').sum().groupby('strategy_name').apply(self._calculate_calmar_ratio)
                metrics = metrics[metrics > 0]
                if not metrics.empty:
                    min_v, max_v = metrics.min(), metrics.max()
                    self.strategy_weights = {k: 0.5 + ((v - min_v)/(max_v - min_v) if max_v > min_v else 1.0) for k, v in metrics.items()}
                    L.info(f"Strategy Weights: {self.strategy_weights}")
        except Exception: pass

    def calculate_position_size(self, token: int, risk_per_lot: float, vega_per_lot: float, confidence_score: float, strategy_name: str) -> int:
        if risk_per_lot <= 0: return 0
        
        now_time = datetime.now(tz=IST).time()
        time_scalar = 1.0
        
        # Define Lunch Lull (10:30 AM to 12:45 PM IST)
        kill_zone_start = dtime(10, 30)
        kill_zone_end = dtime(12, 45)
        
        vix_val = self.prices.ltp(self.engine.vix_token) or 15.0
        
        # Logic: If in Lunch AND VIX is low (<18), reduce size by 75%. 
        # If VIX is high (>18), market is active, reduce by 20%.
        if kill_zone_start <= now_time <= kill_zone_end:
            if vix_val < 18.0: 
                time_scalar = 0.25 
                L.debug(f"💤 Kill Zone Active (Lunch + Low VIX). Scaling down 75%.")
            else:
                time_scalar = 0.8 
        
        # Equity Curve Adjustment
        equity_curve_mult = 1.25 if self.performance_score >= 3 else 0.5 if ((self.daily_high_water_mark - self.dynamic_account_equity)/self.daily_high_water_mark) > 0.015 else 1.0
        
        strat_weight = self.strategy_weights.get(strategy_name, 1.0)
        base_risk = self.dynamic_account_equity * (self.trading_config["risk_tiers"]["standard"] / 100.0)
        
        allowed_risk = base_risk * self.risk_factor * equity_curve_mult * strat_weight * (confidence_score / 2.0)
        return max(0, min(int(allowed_risk / (risk_per_lot * 1.2)), self.trading_config['max_lots_per_trade']))

    def risk_ok(self, hypothetical_params: Dict) -> bool:
        with self.lock:
            if self.circuit_breaker_triggered or self.engine.master_halt: return False
            if ((self.daily_high_water_mark - self.dynamic_account_equity)/self.daily_high_water_mark * 100) > self.trading_config["max_daily_drawdown_pct"]: return False
            return True
            
    def update_performance_metrics(self, pnl: float):
        with self.engine.master_lock:
            current_equity = self.dynamic_account_equity + self.trader.daily_realized_pnl + self.last_unrealized_pnl
            self.daily_high_water_mark = max(self.daily_high_water_mark, current_equity)
            self.weekly_high_water_mark = max(self.weekly_high_water_mark, self.daily_high_water_mark)

            if pnl > 0: self.performance_score = min(4, self.performance_score + 1)
            else: self.performance_score = max(-4, self.performance_score - 2)

            if pnl < 0: self.consecutive_losses += 1
            else: self.consecutive_losses = 0

            loss_streak_config = self.trading_config["consecutive_loss_adjustment"]
            self.risk_factor = max(loss_streak_config["min_factor"], 1.0 - loss_streak_config["reduction_per_loss"] * self.consecutive_losses)
            
            self.store_actor.q.put({"type": "set_kv", "key": f"daily_pnl_{self.engine.last_trading_day}", "value": str(self.trader.daily_realized_pnl)})
            self.store_actor.q.put({"type": "set_kv", "key": "weekly_hwm", "value": str(self.weekly_high_water_mark)})

    # In class RiskManager, update reconcile_broker_pnl

    def reconcile_broker_pnl(self):
        if PAPER_TRADING: return
        
        # Cost adjustment: Assume a conservative ₹50/lot round-trip fee
        COST_PER_LOT = self.trading_config.get("commission_per_lot", 50) 

        try:
            # 1. Fetch Broker Position PnL (Blocking call)
            reply_q = queue.Queue()
            self.engine.trader.order_actor.q.put({"type": "positions", "reply_q": reply_q}, timeout=10.0)
            
            try:
                resp = reply_q.get(timeout=10.0)
            except queue.Empty:
                L.warning("Reconcile: Timeout getting broker positions.")
                return

            if not resp['ok'] or 'net' not in resp['res']: return

            broker_unrealized_pnl = sum(pos.get('unrealised', 0) for pos in resp['res']['net'] if pos.get('product') == 'MIS')
            
            # 2. Adjust Bot PnL (Thread-Safe Calculation)
            bot_unrealized_pnl = self.trader.unrealized_pnl()
            
            # Inside reconcile_broker_pnl
            with self.trader.lock:
                open_lots = 0
                for p in self.trader.positions.values():
                    if p.status == PositionStatus.ACTIVE.value:
                        ls = self.book.lot_size(_get_underlying(p.tradingsymbol)) or 1 # Default to 1 if None
                        open_lots += abs(p.qty / ls)
            
            # Calculate Net PnL
            estimated_fees = open_lots * COST_PER_LOT * 2 
            bot_net_pnl_adjusted = bot_unrealized_pnl - estimated_fees
            
            # 3. Compare
            discrepancy = abs(broker_unrealized_pnl - bot_net_pnl_adjusted)
            critical_threshold = self.trading_config.get('pnl_discrepancy_critical_threshold', 1000.0)

            if discrepancy > critical_threshold:
                send_alert(f"🔥🔥 FATAL P&L DISCREPANCY: Bot: {bot_net_pnl_adjusted:.2f} vs Broker: {broker_unrealized_pnl:.2f}. Halting.", "critical")
                self.engine.fatal_error_event.set() 

        except Exception as e:
            L.warning(f"Could not reconcile broker P&L: {e}", exc_info=True)

    def update_pnl_metrics(self):
        self.last_unrealized_pnl = self.trader.unrealized_pnl()
        
    def initialize_position_greeks(self, p: Position):
        with self.lock:
            if p.greeks:
                self.portfolio_greeks["net_delta"] += p.greeks.get("delta", 0.0) * p.initial_qty
                self.portfolio_greeks["net_vega"] += p.greeks.get("vega", 0.0) * p.initial_qty
                self.portfolio_greeks["net_gamma"] += p.greeks.get("gamma", 0.0) * p.initial_qty
                self.portfolio_greeks["net_theta"] += p.greeks.get("theta", 0.0) * p.initial_qty

    def update_position_greeks(self, p: Position, ltp: float):
        with self.lock:
            old_greeks = p.greeks.copy()
            underlying_name = _get_underlying(p.tradingsymbol)
            underlying_token = self.engine.bn_token if "BANKNIFTY" in underlying_name else self.engine.nifty_token
            spot = self.prices.ltp(underlying_token)
            if not spot: return

            opt_details = self.book.df_by_token.loc[p.token]
            T = calculate_trading_time_to_expiry(datetime.combine(opt_details['expiry'].date(), self.engine.timings_config["market_close"]), now_ist(), self.engine.timings_config["market_open"], self.engine.timings_config["market_close"], self.engine.nse_calendar)
            
            hv = calculate_historical_volatility(self.engine.get_ohlc(underlying_token, 1)['close'])
            iv = calculate_iv(ltp, spot, opt_details['strike'], T, 0.05, p.option_type == 'CE', hv_fallback=hv)
            p.greeks = calculate_greeks(spot, opt_details['strike'], T, 0.05, iv, p.option_type == 'CE')
            p.greeks['iv'] = iv
            
            for key in ["delta", "vega", "gamma", "theta"]:
                change = p.greeks.get(key, 0.0) - old_greeks.get(key, 0.0)
                self.portfolio_greeks[f"net_{key}"] += change * p.qty

    def verify_position_risk(self, p: Position):
        underlying_name = _get_underlying(p.tradingsymbol)
        underlying_token = self.engine.bn_token if "BANKNIFTY" in underlying_name else self.engine.nifty_token
        spot = self.prices.ltp(underlying_token)
        if not spot: return
        
        actual_risk_rupees = p.option_sl_points * p.initial_qty
        if actual_risk_rupees > (p.intended_risk_rupees * 1.4):
            L.critical(f"RISK BREACH POST-FILL on {p.tradingsymbol}. Exiting.")
            self.trader.close_position(p, "POST_FILL_RISK_BREACH")

    def update_dynamic_equity(self):
         # Logic handled by _async_margin_worker now, but we can force a refresh
         pass

    def reset_daily_state(self, last_trading_day: Optional[date]):
        now = now_ist()
        if last_trading_day and last_trading_day.weekday() > now.date().weekday():
            self.weekly_high_water_mark = self.dynamic_account_equity
            self.in_weekly_drawdown_lock = False
            self.store_actor.q.put({"type": "set_kv", "key": "weekly_hwm", "value": str(self.weekly_high_water_mark)})

        if last_trading_day:
            self.store_actor.q.put({"type": "set_kv", "key": f"daily_pnl_{last_trading_day}", "value": str(self.trader.daily_realized_pnl)})

        self.trader.daily_realized_pnl = 0.0
        with self.lock: self.portfolio_greeks = {"net_delta": 0.0, "net_vega": 0.0, "net_gamma": 0.0, "net_theta": 0.0}
        self.consecutive_losses = 0
        self.risk_factor = 1.0
        self.daily_high_water_mark = self.dynamic_account_equity
        self.performance_score = 0
        self.circuit_breaker_triggered = False

    def load_persistent_state(self, trading_day: date):
        weekly_hwm_str = str(self.dynamic_account_equity)
        pnl_str = "0.0"
        try:
            reply_q_hwm = queue.Queue()
            self.store_actor.q.put({"type": "get_kv", "key": "weekly_hwm", "default": str(self.dynamic_account_equity), "reply_q": reply_q_hwm})
            resp_hwm = reply_q_hwm.get(timeout=5.0)
            if resp_hwm['ok']: weekly_hwm_str = resp_hwm['res']

            reply_q_pnl = queue.Queue()
            self.store_actor.q.put({"type": "get_kv", "key": f"daily_pnl_{trading_day}", "default": "0.0", "reply_q": reply_q_pnl})
            resp_pnl = reply_q_pnl.get(timeout=5.0)
            if resp_pnl['ok']: pnl_str = resp_pnl['res']
        except: pass
        
        
class ConstituentRadar:
    """Tracks real-time momentum of Nifty/BankNifty heavyweights (Kingmaker Filter)."""
    def __init__(self, book: InstrumentBook, prices: PriceBus, config: Dict, engine: 'Engine' = None):
        self.book = book
        self.prices = prices
        self.config = config
        self.engine = engine
        self.lock = threading.RLock()
        
        # Load Weights from Config (or Fallback)
        tech_cfg = self.config.get('technical', {}).get('constituent_weights', {})
        
        if 'NIFTY' in tech_cfg:
            self.nifty_weights = tech_cfg['NIFTY']
        else:
            L.warning("⚠️ ConstituentRadar: NIFTY weights missing in config. Using Hardcoded Fallback.")
            self.nifty_weights = {"HDFCBANK": 13.0, "RELIANCE": 10.0, "ICICIBANK": 8.0, "INFY": 6.0, "ITC": 3.5}

        if 'BANKNIFTY' in tech_cfg:
            self.bn_weights = tech_cfg['BANKNIFTY']
        else:
            L.warning("⚠️ ConstituentRadar: BANKNIFTY weights missing in config. Using Hardcoded Fallback.")
            self.bn_weights = {"HDFCBANK": 29.0, "ICICIBANK": 23.0, "SBIN": 11.0, "AXISBANK": 9.0, "KOTAKBANK": 9.0}

        self.tokens_map = {}
        self._init_tokens()
        
    # --- [PATCH 1] INSERT INTO CLASS ConstituentRadar ---
    def get_market_coherence(self, index_name: str) -> Dict:
        """
        Calculates the 'Leviathan Score' using Zero-Latency Memory Access.
        Returns: { 'score': float (-1.0 to 1.0), 'strength': float }
        """
        weights = self.bn_weights if "BANK" in index_name else self.nifty_weights
        
        active_tokens = []
        total_weight = 0.0
        weighted_velocity = 0.0
        
        with self.lock:
            for sym, w in weights.items():
                token = self.book.get_token(sym)
                if not token: continue
                
                # 1. Get Real-Time Price
                ltp = self.prices.ltp(token)
                if not ltp: continue
                
                # 2. ZERO-LATENCY REFERENCE CHECK
                # Read directly from BarStore memory (Active Candle Open)
                ref_price = ltp 
                try:
                    if token in self.engine.bars.active_bars:
                        active_bar = self.engine.bars.active_bars[token].get(1)
                        if active_bar: ref_price = active_bar['open']
                except Exception: pass

                if ref_price == 0: continue

                # 3. Calculate Flow
                pct_change = (ltp - ref_price) / ref_price
                
                # Weight the movement
                weighted_velocity += (pct_change * w)
                total_weight += w
                
                # Count Coherence (Ignore noise < 0.03%)
                if abs(pct_change) > 0.0003: 
                    active_tokens.append(np.sign(pct_change))

        if not active_tokens: return {'score': 0.0, 'strength': 0.0}
        
        net_direction = sum(active_tokens)
        coherence_score = net_direction / len(active_tokens)
        
        # Leviathan Strength (Normalized)
        strength = (weighted_velocity / (total_weight if total_weight > 0 else 1)) * 1000 
        
        return {'score': coherence_score, 'strength': strength}

    def _init_tokens(self):
        all_symbols = set(list(self.nifty_weights.keys()) + list(self.bn_weights.keys()))
        tokens_to_sub = []
        for sym in all_symbols:
            token = self.book.get_token(sym) 
            if token:
                self.tokens_map[token] = sym
                tokens_to_sub.append(token)
        if tokens_to_sub:
            self.prices.subscribe(tokens_to_sub)
            L.info(f"ConstituentRadar watching {len(tokens_to_sub)} heavyweights.")

    def get_weighted_momentum(self, index_name: str) -> float:
        weights = self.bn_weights if "BANK" in index_name else self.nifty_weights
        total_score, total_weight = 0.0, 0.0
        with self.lock:
            for sym, w in weights.items():
                token = self.book.get_token(sym)
                if not token: continue
                tick = self.prices.get_full_tick(token)
                if not tick or not tick.get('last_price') or not tick.get('ohlc'): continue
                pct_change = ((tick['last_price'] - tick['ohlc']['open']) / tick['ohlc']['open']) * 100
                score = 1 if pct_change > 0.15 else -1 if pct_change < -0.15 else 0
                total_score += (score * w)
                total_weight += w
        return (total_score / total_weight) * 100.0 if total_weight > 0 else 0.0

class MicrostructureMonitor:
    """
    SUPREME EDITION v3: TFI (Soft Split), Drift Reset, and Velocity Tracking.
    Optimized for Retail Data Fidelity (Kite Snapshots).
    """
    def __init__(self, prices: PriceBus, book: InstrumentBook, config: Dict):
        self.prices = prices
        self.book = book
        self.lock = threading.RLock()
        
        # --- CVD & Delta Streams ---
        # Raw stream of signed volume (Buy Vol - Sell Vol) for Delta-RSI
        self.delta_stream: Dict[int, List[float]] = {} 
        self.cvd: Dict[int, float] = {} 
        self.last_cvd_reset: Dict[int, int] = {} # Tracks minute of last reset to fix drift
        
        # --- Pivot Tracking (For Divergence) ---
        # Stores tuples of (Price, CVD) to detect absorption/exhaustion
        self.pivots: Dict[int, deque] = {} 
        
        # --- Velocity Tracking (For Adaptive Entry) ---
        self.tick_times: Dict[int, deque] = {} # Timestamps of recent ticks
        
        # --- Data Smoother ---
        self.last_tick_prices: Dict[int, float] = {}

    def update_tfi_score(self, tick: dict) -> Optional[Dict]:
        """
        Updates Microstructure state using 'Soft Split' logic to handle snapshot data.
        Returns None (Flash Signals are disabled per Phase 1 Purge).
        """
        token = tick.get('instrument_token')
        price = tick.get('last_price')
        qty = tick.get('last_traded_quantity', 0)
        
        if not all([token, price, qty is not None]): return None
        
        now = datetime.now(tz=IST)
        
        with self.lock:
            # 1. Velocity Tracking (Stores last 50 tick timestamps)
            if token not in self.tick_times: self.tick_times[token] = deque(maxlen=50)
            self.tick_times[token].append(now)

            # 2. Initialize Streams if new token
            if token not in self.delta_stream:
                self.delta_stream[token] = []
                self.cvd[token] = 0.0
                self.last_cvd_reset[token] = now.minute
                self.pivots[token] = deque(maxlen=50)
                self.last_tick_prices[token] = price
                return None # Need history to calc delta

            # 3. Soft Split CVD Logic (The "Drift Fix")
            # Standard Bid/Ask logic fails on snapshots. We use Price Change Probability.
            prev_price = self.last_tick_prices[token]
            price_change = price - prev_price
            self.last_tick_prices[token] = price # Update for next cycle

            buy_vol = 0.0
            
            if qty > 0:
                # 1. Robust Tick Calculation
                # Round to nearest 2 decimal places to kill floating point ghosts
                tick_size = self.book.tick_size(token) or 0.05
                price_change = round(price - prev_price, 2)
                
                # 2. Calculate "Displacement"
                # Use round to ensure 0.10 / 0.05 becomes 2.0, not 1.9999
                ticks_moved = round(abs(price_change) / tick_size)

                buy_ratio = 0.5 # Default: Neutral/Noise

                # 3. The "High-Pass Filter"
                if ticks_moved >= 2: # Strict 2-tick threshold
                    # Linear scaling: 2 ticks = 70%, 5 ticks = 100% (capped at 0.95)
                    confidence = min(0.95, 0.5 + (ticks_moved * 0.1))
                    
                    if price_change > 0:
                        buy_ratio = confidence
                    else:
                        buy_ratio = 1.0 - confidence

                # Apply the ratio to the volume
                buy_vol = qty * buy_ratio
                sell_vol = qty * (1.0 - buy_ratio)
                delta_val = buy_vol - sell_vol
                
                # Update Accumulators
                self.cvd[token] += delta_val
                self.delta_stream[token].append(delta_val)
                
                # Trim Delta Stream for RSI calc
                if len(self.delta_stream[token]) > 100: 
                    self.delta_stream[token].pop(0)

                # Pivot Detection (For Divergence)
                # We only record a "Pivot" if price moves > 0.05% to filter noise
                last_pivot = self.pivots[token][-1] if self.pivots[token] else (0,0)
                if abs(price - last_pivot[0]) > (price * 0.0005): 
                    self.pivots[token].append((price, self.cvd[token]))

            # 4. The Hard Reset (The "Drift Killer")
            # Every 30 minutes (at :00 and :30), reset CVD to 0.
            # This prevents the "Soft Split" estimation errors from accumulating to infinity.
            if now.minute % 30 == 0 and self.last_cvd_reset[token] != now.minute:
                 # L.info(f"🧹 CVD Hard Reset for {token} to clear accumulated drift.")
                 self.cvd[token] = 0.0
                 self.last_cvd_reset[token] = now.minute

            return None # Returns None because we killed Flash Signals (Phase 1)

    def get_tick_velocity(self, token: int) -> float:
        """
        Returns ticks per second over the last 10 seconds.
        Used by PositionManager to set Adaptive Entry Timeouts.
        """
        with self.lock:
            times = self.tick_times.get(token)
            if not times or len(times) < 2: return 0.0
            
            now = datetime.now(tz=IST)
            # Filter for timestamps within last 10 seconds
            recent = [t for t in times if (now - t).total_seconds() <= 10.0]
            
            if len(recent) < 2: return 0.0
            return len(recent) / 10.0

    def get_delta_rsi(self, token: int, period: int = 14) -> float:
        """Returns RSI of the Order Flow Delta (Exhaustion Indicator)."""
        with self.lock:
            if token not in self.delta_stream or len(self.delta_stream[token]) < period + 1:
                return 50.0
            
            # Convert to Pandas Series for efficient Calc
            deltas = pd.Series(self.delta_stream[token])
            rsi_val = ta.rsi(deltas, length=period)
            return rsi_val.iloc[-1] if not rsi_val.empty else 50.0

    def check_cvd_divergence(self, token: int, side: OrderSide) -> bool:
        """
        The "Lie Detector".
        Returns True if Price and Order Flow disagree (Absorption/Exhaustion).
        """
        with self.lock:
            history = self.pivots.get(token)
            if not history or len(history) < 5: return False
            
            # Compare Current Pivot vs Previous Pivot (Lookback 2 points)
            curr_p, curr_cvd = history[-1]
            prev_p, prev_cvd = history[-3] 
            
            if side == OrderSide.BUY:
                # BULLISH ABSORPTION (Buy Signal)
                # Price made Lower Low (or Equal), but CVD made Higher Low
                # Meaning: Sellers hit bid, but price refused to drop -> Limit Buyers Absorb.
                if curr_p <= prev_p and curr_cvd > prev_cvd:
                    return True 
                    
            elif side == OrderSide.SELL:
                # BEARISH EXHAUSTION (Sell Signal)
                # Price made Higher High (or Equal), but CVD made Lower High
                # Meaning: Buyers hit ask, but price refused to rise -> Limit Sellers Absorb.
                if curr_p >= prev_p and curr_cvd < prev_cvd:
                    return True
            
            return False
            
    
class PositionManager:
    """Handles the high-frequency loop for managing all active positions."""
    def __init__(self,
                 engine: 'Engine',
                 trader: AbstractTrader,
                 book: InstrumentBook,
                 prices: PriceBus,
                 store_actor: StoreActor,
                 risk_manager: RiskManager,
                 config: Dict):
        self.engine = engine
        self.trader = trader
        self.book = book
        self.prices = prices
        self.store_actor = store_actor 
        self.risk_manager = risk_manager
        self.trading_config = config["trading"]
        self.timings_config = config["timings"]
        self.lock = threading.RLock()

        self.position_price_history: Dict[str, deque] = {}
        underlying_entry_price: Optional[float] = None
        self.last_greeks_update_per_pos: Dict[str, datetime] = {}
        self.trailing_sl_config = self.trading_config.get("trailing_sl", {})
        self.trade_mgmt_config = self.trading_config.get("trade_management", {})

    def _check_microstructure_exit(self, p: Position, now: datetime) -> bool:
        """
        Checks if Order Flow has turned toxic (Microstructure Ejector).
        Returns True if position should be closed immediately.
        """
        # Give trade 30 seconds to breathe before checking flow
        if (now - p.opened_at).total_seconds() < 30: return False
        
        # Ensure Monitor is active
        if not self.engine.micro_monitor: return False

        # Logic: We are always LONG the Option (Call or Put).
        # We want buying pressure on the Option.
        # If there is Aggressive SELLING (TFI returns 1 for SELL), we must exit.
        
        # Check TFI (Trade Flow Imbalance) for Sell Pressure
        tfi_sell_pressure = self.engine.micro_monitor.check_tfi(p.token, OrderSide.SELL)
        
        if tfi_sell_pressure == 1:
            # Confirm with OBI (Order Book Imbalance) for safety
            # Check if the Book is also dominated by Sellers (Asks)
            obi_sell_dominance = self.engine.micro_monitor.check_order_book_imbalance(p.token, OrderSide.SELL)
            
            if obi_sell_dominance == 1:
                L.warning(f"🚀 EJECTOR: Toxic Sell Flow on {p.tradingsymbol}. TFI & OBI confirm reversal.")
                return True
                
        return False
    
    def _check_relative_strength_failure(self, p: Position, ltp: float, now: datetime) -> bool:
        # Give trade 60 seconds to stabilize
        if (now - p.opened_at).total_seconds() < 60: return False
        
        # Get Underlying Price
        u_name = _get_underlying(p.tradingsymbol)
        u_token = self.engine.bn_token if "BANKNIFTY" in u_name else self.engine.nifty_token
        u_ltp = self.prices.ltp(u_token)
        if not u_ltp: return False
        
        # Calculate Change since Entry
        u_change_pct = ((u_ltp - p.underlying_entry_price) / p.underlying_entry_price) * 100
        opt_change_pct = ((ltp - p.entry_price) / p.entry_price) * 100
        
        # LOGIC: If Underlying is moving favorably (> 0.15%), but Option is NEGATIVE
        if p.option_type == "CE":
            if u_change_pct > 0.15 and opt_change_pct < 0.0:
                L.warning(f"📉 RS FAIL on {p.tradingsymbol}: Nifty +{u_change_pct:.2f}% but Option {opt_change_pct:.2f}%. EJECTING.")
                return True
        elif p.option_type == "PE":
            if u_change_pct < -0.15 and opt_change_pct < 0.0:
                L.warning(f"📉 RS FAIL on {p.tradingsymbol}: Nifty {u_change_pct:.2f}% but Option {opt_change_pct:.2f}%. EJECTING.")
                return True
                
        return False
    
    def _check_rs_failure(self, p: Position, ltp: float, now: datetime) -> bool:
        """Returns True if Underlying moves in favor but Option fails to follow."""
        # 1. Grace Period: Give trade 45 seconds to stabilize/fill
        if (now - p.opened_at).total_seconds() < 45: return False
        
        # 2. Get Underlying Price
        u_name = _get_underlying(p.tradingsymbol)
        u_token = self.engine.bn_token if "BANKNIFTY" in u_name else self.engine.nifty_token
        u_ltp = self.prices.ltp(u_token)
        
        if not u_ltp or p.underlying_entry_price <= 0: return False
        
        # 3. Calculate Performance
        u_change_pct = ((u_ltp - p.underlying_entry_price) / p.underlying_entry_price) * 100
        opt_change_pct = ((ltp - p.entry_price) / p.entry_price) * 100
        
        # 4. The "Sucker" Check
        # If Nifty moved > +0.15% in favor, but Option is negative -> EJECT
        if p.option_type == "CE":
            if u_change_pct > 0.15 and opt_change_pct < 0.0:
                L.warning(f"📉 RS EJECT {p.tradingsymbol}: Spot +{u_change_pct:.2f}% but Opt {opt_change_pct:.2f}%")
                return True
        elif p.option_type == "PE":
            if u_change_pct < -0.15 and opt_change_pct < 0.0:
                L.warning(f"📉 RS EJECT {p.tradingsymbol}: Spot {u_change_pct:.2f}% but Opt {opt_change_pct:.2f}%")
                return True
                
        return False

    def manage_positions(self):
        """
        High-Frequency Position Management Loop.
        Handles Entry Logic, Risk Ejection, Trailing, and Scale-Outs.
        """
        with self.trader.lock:
            active_positions = list(self.trader.positions.values())

        now = now_ist()
        
        # Manage simulated fills for paper trading
        if PAPER_TRADING:
            self.trader._manage_pending_entries(now)

        for p in active_positions:
            with self.trader.lock:
                if p.id not in self.trader.positions: continue

            # --- 1. Handle Pending Entry (Adaptive Logic) ---
            if p.status == PositionStatus.PENDING_ENTRY.value:
                if not p.last_entry_modification or (now - p.last_entry_modification).total_seconds() > 1.0:
                    # Spawn thread to handle price chasing logic
                    threading.Thread(target=self._manage_adaptive_entry, args=(p, now), daemon=True).start()
                continue 

            # --- 2. Handle Placing Brackets (After Fill) ---
            if p.status == PositionStatus.OPEN_AWAITING_BRACKETS.value:
                if not self.trader.place_bracket_orders(p):
                    send_alert(f"CRITICAL: FAILED to place brackets for {p.tradingsymbol}. Closing.", "critical")
                    self.trader.close_position(p, "BRACKET_PLACEMENT_FAILURE")
                continue

            # --- 3. Handle Pending SL-L Fallback ---
            if p.status == PositionStatus.PENDING_SL_EXIT.value:
                ltp = self.prices.ltp(p.token)
                if ltp and ltp < (p.sl_price * 0.99):
                     self.trader.close_position(p, "SL_L_MISSED_MK_FALLBACK")
                continue

            # --- Skip non-active ---
            if p.status not in [PositionStatus.ACTIVE.value, PositionStatus.PARTIALLY_CLOSED.value]:
                continue

            # =========================================================
            # ACTIVE MANAGEMENT LOOP (The "Ejector Seat")
            # =========================================================
            ltp = self.prices.ltp(p.token)
            if not ltp: continue 
            
            # --- A. UPGRADE: STAGNATION KILL-SWITCH ---
            # If trade is flat/weak after 5 minutes, kill it to save Theta.
            time_in_trade_mins = (now - p.opened_at).total_seconds() / 60
            if 5.0 <= time_in_trade_mins <= 6.0:
                profit_pct = (ltp - p.entry_price) / p.entry_price
                # Need > 2% profit by minute 5 to justify holding
                if profit_pct < 0.02:
                    L.warning(f"💀 Stagnation Kill: {p.tradingsymbol} (+{profit_pct:.2%}) stalled. Ejecting.")
                    self.trader.close_position(p, "STAGNATION_KILL")
                    continue

            # --- B. UPGRADE: BREAKEVEN RATCHET ---
            # "Free Ride" Protocol: If +0.5R, move SL to Entry.
            if not p.trailing_sl_armed and p.sl_price < p.entry_price:
                current_r = (ltp - p.entry_price) / p.initial_risk_points if p.initial_risk_points > 0 else 0
                if current_r >= 0.5:
                    tick_sz = self.book.tick_size(p.tradingsymbol)
                    new_sl = p.entry_price + (tick_sz * 2) # Entry + Buffer
                    
                    if new_sl < ltp:
                        L.info(f"🛡️ Ratchet Triggered: {p.tradingsymbol} (+{current_r:.2f}R). SL -> Breakeven.")
                        p.sl_price = new_sl
                        p.trailing_sl_armed = True # Prevents re-firing
                        threading.Thread(target=self.trader.modify_sl, args=(p, new_sl), daemon=True).start()
                        self.store_actor.q.put({"type": "upsert_position", "pos": p})

            # --- C. EXISTING EJECTORS ---
            
            # 1. Relative Strength (Nifty moves, Option doesn't)
            if self._check_rs_failure(p, ltp, now):
                self.trader.close_position(p, "RS_FAILURE_EJECT")
                continue

            # 2. Microstructure (Toxic Flow Reversal)
            if self._check_microstructure_exit(p, now):
                self.trader.close_position(p, "MICRO_EJECT")
                continue 
            
            # 3. Velocity Trigger (Crash Protection)
            if p.id not in self.position_price_history:
                self.position_price_history[p.id] = deque(maxlen=20)
            self.position_price_history[p.id].append((now, ltp))
            
            if self._check_velocity_trigger(p, now):
                self.trader.close_position(p, "VELOCITY_EJECT")
                continue

            # 4. Time Stop (Hard Limit)
            if time_in_trade_mins > p.max_trade_duration_minutes:
                self.trader.close_position(p, "TIME_STOP")
                continue

            # 5. Underlying SL Check
            underlying_name = _get_underlying(p.tradingsymbol)
            u_token = self.engine.bn_token if "BANKNIFTY" in underlying_name else self.engine.nifty_token
            u_price = self.prices.ltp(u_token)
            
            if u_price and p.underlying_sl_level:
                hit_sl = (p.option_type == 'CE' and u_price <= p.underlying_sl_level) or \
                         (p.option_type == 'PE' and u_price >= p.underlying_sl_level)
                if hit_sl:
                    self.trader.close_position(p, "UNDERLYING_SL")
                    continue

            # --- D. Routine Updates ---
            
            # Greeks Update (Throttled 10s)
            if not self.last_greeks_update_per_pos.get(p.id) or (now - self.last_greeks_update_per_pos.get(p.id, datetime.min.replace(tzinfo=IST))).total_seconds() > 10:
                self.risk_manager.update_position_greeks(p, ltp)
                self.last_greeks_update_per_pos[p.id] = now

            # --- E. Scale Out & Trailing ---
            profit_points = ltp - p.entry_price
            
            # Scale Out Logic
            scaled = False
            with self.trader.lock:
                for rule in p.scale_out_rules:
                    target = rule['rr_target']
                    if target not in p.triggered_scale_out_targets and profit_points >= p.initial_risk_points * target:
                        qty_out = int(p.initial_qty * (rule['pct_to_close'] / 100.0))
                        p.triggered_scale_out_targets.append(target)
                        threading.Thread(target=self.trader.scale_out, args=(p, qty_out), daemon=True).start()
                        scaled = True
                        break
            if scaled: continue

            # Trailing SL Logic
            with self.trader.lock:
                current_rr = profit_points / p.initial_risk_points if p.initial_risk_points > 0 else 0
                new_sl = p.initial_sl_price

                # 1. Static RR Trail
                for stage in self.trade_mgmt_config.get("trailing_stop_stages", []):
                    if current_rr >= stage['rr_target']:
                        new_floor = p.entry_price + (p.initial_risk_points * stage['trail_behind_rr'])
                        new_sl = max(new_sl, new_floor)

                # 2. Chandelier Trail
                if not p.trailing_sl_armed and profit_points >= p.initial_risk_points * self.trading_config['trailing_sl_activation_rr']:
                    p.trailing_sl_armed = True
                    # L.info(f"Chandelier Trailing SL armed for {p.tradingsymbol}.")

                if p.trailing_sl_armed:
                    if c_sl := self._calculate_trailing_stop(p):
                        new_sl = max(new_sl, c_sl)

                # 3. Apply Trail
                if new_sl > p.sl_price:
                    is_risk_free = new_sl >= p.entry_price
                    # If risk-free, cancel static TP to let runners run (Optional optimization)
                    if is_risk_free and p.tp_order_id and not PAPER_TRADING:
                        self.engine.trader.order_actor.q.put({"type": "cancel_order", "params": {"variety": "regular", "order_id": str(p.tp_order_id)}, "reply_q": None})
                        p.tp_order_id = None
                    
                    p.sl_price = new_sl
                    threading.Thread(target=self.trader.modify_sl, args=(p, new_sl), daemon=True).start()

            # Final DB Update
            self.store_actor.q.put({"type": "upsert_position", "pos": p})

    def _calculate_trailing_stop(self, p: Position) -> Optional[float]:
        try:
            cfg = self.trailing_sl_config
            tf = cfg["timeframe_scaled_out"] if p.triggered_scale_out_targets else cfg["timeframe"]
            df = self.engine.get_ohlc(p.token, tf)
            if len(df) < cfg["chandelier_period"]: return None
            
            atr = df.ta.atr(length=cfg["chandelier_period"]).iloc[-1]
            if pd.isna(atr): return None
            
            mult = cfg["chandelier_multiplier_scaled_out"] if p.triggered_scale_out_targets else cfg["chandelier_multiplier"]
            high = df['high'].rolling(cfg["chandelier_period"]).max().iloc[-1]
            return high - (atr * mult)
        except: return None

    def _check_velocity_trigger(self, p: Position, now: datetime) -> bool:
        history = self.position_price_history.get(p.id)
        if not history or len(history) < 3: return False
        
        cfg = self.trade_mgmt_config.get("velocity_trigger", {})
        lookback = cfg.get("lookback_seconds", 5)
        
        old = next((x for x in history if (now - x[0]).total_seconds() <= lookback), None)
        if not old: return False
        
        t_delta = (history[-1][0] - old[0]).total_seconds()
        if t_delta < 1: return False
        
        p_delta = history[-1][1] - old[1]
        velo = p_delta / t_delta
        
        if velo >= 0: return False 
        
        dist = history[-1][1] - p.sl_price
        if dist <= 0: return False 
        
        tti = dist / abs(velo)
        return tti < cfg.get("time_to_sl_threshold_seconds", 2.0)

    def _manage_adaptive_entry(self, p, now):
        """
        Adaptive Execution Engine.
        - Uses Market Velocity to dynamically set timeouts.
        - Enforces 'No Chase' policy (Kills trade if Mid-Price doesn't fill).
        """
        if not p.last_entry_modification: return

        # 1. Calculate Time & Velocity
        time_since_mod = (now - p.last_entry_modification).total_seconds()
        
        velocity = 0.0
        if self.engine.micro_monitor:
            velocity = self.engine.micro_monitor.get_tick_velocity(p.token)
            
        # Default Config
        base_timeout = self.trading_config.get('adaptive_entry_stage2_ms', 800) / 1000.0
        
        # Dynamic Timeout Adjustment
        if velocity > 5.0:
            stage2_timeout = 0.5  # Fast Market: Hurry up (0.5s)
        elif velocity < 1.0:
            stage2_timeout = 5.0  # Slow Market: Be patient (5.0s)
        else:
            stage2_timeout = base_timeout

        # 2. Get Current Market Data
        full_tick = self.prices.get_full_tick(p.token)
        if not full_tick or not full_tick.get('depth'): return

        depth = full_tick['depth']
        bid_price = depth['buy'][0]['price']
        ask_price = depth['sell'][0]['price']
        tick_size = self.book.tick_size(p.tradingsymbol)
        
        # Target: Mid-Price (Aggressive but not crossing spread)
        mid_price = round(((bid_price + ask_price) / 2.0) / tick_size) * tick_size
        
        new_price = -1.0
        target_stage = p.entry_stage
        
        # 3. Gamma Burst Acceleration
        # If Gamma Burst, skip Stage 1 (Bid) and go straight to Stage 2 (Mid).
        # But do NOT skip logic to check if we need to modify.
        is_gamma_burst = p.strategy == "GammaBurst" or p.strategy == "GAMMA_BURST_FLASH"
        if is_gamma_burst and p.entry_stage == 1:
             L.info(f"🚀 Gamma Burst Acceleration: Skipping Bid, moving to Mid for {p.tradingsymbol}")
             # Force immediate transition logic below
             time_since_mod += 10.0 

        # 4. State Machine Logic
        if p.entry_stage == 1:
            # Stage 1: Sitting at BID.
            if time_since_mod > stage2_timeout:
                L.info(f"Adaptive Entry Stage 2 (Mid) for {p.tradingsymbol}. Velo={velocity:.1f}")
                new_price = mid_price
                target_stage = 2
                
        elif p.entry_stage == 2:
            # Stage 2: Sitting at MID.
            # UPGRADE: KILL STAGE 3 (NO CHASE POLICY)
            # If we are at Mid-Price and don't get filled within 2x timeout, the move is gone.
            # We do NOT cross the spread to the Ask. We Cancel.
            wait_limit = stage2_timeout * 2.0
            
            if time_since_mod > wait_limit:
                L.warning(f"🚫 Entry Timeout at Mid-Price for {p.tradingsymbol}. Cancelling (No Chase).")
                self.trader.cancel_pending_entry(p)
                return

        # 5. Execute Modification
        if new_price > 0:
            modify_params = {
                "variety": "regular", 
                "order_id": str(p.entry_order_id), 
                "price": new_price
            }
            # Fire-and-forget modify request
            self.engine.trader.order_actor.q.put({
                "type": "modify_order", 
                "params": modify_params, 
                "reply_q": None 
            })
            
            # Update State
            p.last_entry_modification = now
            p.entry_stage = target_stage
            self.store_actor.q.put({"type": "upsert_position", "pos": p})


# ==================================================================================================
# STRATEGY DEFINITIONS
# ==================================================================================================
# ==================================================================================================
# STRATEGY DEFINITIONS (MICROSTRUCTURE UPGRADED)
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


# --- [PATCH 2] REPLACE MomentumIgnitionStrategy WITH THIS ---
class LeviathanStrategy(BaseStrategy):
    """
    THE FINAL FORM.
    Integrates: Weighted Coherence (Kings), Singularity (GEX), and ILVP (Trap).
    Includes 'The King's Veto' to prevent False Coherence.
    """
    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]:
        # 0. Data Prep
        df = self.engine.get_ohlc(token, 5)
        if len(df) < 55: return None

        index_name = "BANKNIFTY" if token == self.engine.bn_token else "NIFTY"
        
        # ----------------------------------------------------------------------
        # 1. THE LEVIATHAN (Weighted Scoring) - NO MORE VETO
        # ----------------------------------------------------------------------
        metrics = self.engine.radar.get_market_coherence(index_name)
        coherence = metrics['score']
        strength = metrics['strength']
    
        # Base Score is the Coherence itself (-1.0 to 1.0)
        final_score = coherence
    
        # --- KINGS FACTOR (The Boost/Drag) ---
        # We check the Heavyweights. They don't Veto, they just adjust the score.
        kings_agree = False
    
        if index_name == "BANKNIFTY":
            hdfc_t = self.engine.book.special_tokens.get("HDFCBANK")
            icici_t = self.engine.book.special_tokens.get("ICICIBANK")
        
            h_df = self.engine.get_ohlc(hdfc_t, 1)
            i_df = self.engine.get_ohlc(icici_t, 1)
        
            h_dir = 0
            if not h_df.empty: h_dir = 1 if h_df['close'].iloc[-1] > h_df['open'].iloc[-1] else -1
        
            i_dir = 0
            if not i_df.empty: i_dir = 1 if i_df['close'].iloc[-1] > i_df['open'].iloc[-1] else -1
        
            # Agreement Logic
            if coherence > 0 and h_dir >= 0 and i_dir >= 0: kings_agree = True
            elif coherence < 0 and h_dir <= 0 and i_dir <= 0: kings_agree = True
        else:
            kings_agree = True # Simplified for Nifty
        
        # Apply Multipliers
        if kings_agree:
            final_score *= 1.2 # BOOST: Perfect setup
        else:
            final_score *= 0.8 # PENALTY: Imperfect, but still tradeable if coherence is huge
        
        # EXECUTE (Threshold > 0.7)
        # This allows a 0.9 coherence trade to pass even if Kings disagree (0.9 * 0.8 = 0.72)
        # But a 0.6 coherence trade needs Kings to agree (0.6 * 1.2 = 0.72)
        if final_score >= 0.7 and strength > 1.0:
            L.info(f"🐋 LEVIATHAN BUY: Score {final_score:.2f} (Coh: {coherence:.2f}, Kings: {kings_agree}).")
            return OrderSide.BUY
        if final_score <= -0.7 and strength < -1.0:
            L.info(f"🐋 LEVIATHAN SELL: Score {final_score:.2f} (Coh: {coherence:.2f}, Kings: {kings_agree}).")
            return OrderSide.SELL

        # ----------------------------------------------------------------------
        # 2. THE SINGULARITY (GEX + TRAP) - UNCHANGED
        # ----------------------------------------------------------------------
        swing_low = df['low'].iloc[-50:-1].min()
        swing_high = df['high'].iloc[-50:-1].max()
        curr_low, curr_high, curr_close = df['low'].iloc[-1], df['high'].iloc[-1], df['close'].iloc[-1]
    
        is_short_gamma = self.engine.gex_matrix.is_accelerating(token)
        has_bull_absorb = self.engine.micro_monitor.check_cvd_divergence(token, OrderSide.BUY)
        has_bear_absorb = self.engine.micro_monitor.check_cvd_divergence(token, OrderSide.SELL)

        if (curr_low < swing_low) and (curr_close > swing_low * 1.0001):
            # Kingmaker Divergence Check for Reversal
            hdfc_token = self.engine.book.special_tokens.get("HDFCBANK")
            if hdfc_token:
                h_df = self.engine.get_ohlc(hdfc_token, 5)
                if not h_df.empty and h_df['low'].iloc[-1] > h_df['low'].iloc[-50:-1].min():
                    # GEX is now just a "Nice to have", not a blocker. 
                    # But we still check it for the "God Mode" log.
                    if is_short_gamma or has_bull_absorb:
                        L.info(f"⚛️ GOD MODE BUY: Trap + HDFC Divergence + GEX/CVD.")
                        return OrderSide.BUY

        if (curr_high > swing_high) and (curr_close < swing_high * 0.9999):
            hdfc_token = self.engine.book.special_tokens.get("HDFCBANK")
            if hdfc_token:
                h_df = self.engine.get_ohlc(hdfc_token, 5)
                if not h_df.empty and h_df['high'].iloc[-1] < h_df['high'].iloc[-50:-1].max():
                    if is_short_gamma or has_bear_absorb:
                        L.info(f"⚛️ GOD MODE SELL: Trap + HDFC Divergence + GEX/CVD.")
                        return OrderSide.SELL

        return None

    def get_risk_params(self, token: int, side: OrderSide, current_time: datetime) -> Tuple[float, float]:
        df = self.engine.get_ohlc(token, 1)
        atr = df.ta.atr(14).iloc[-1] if not df.empty else 0
        if atr == 0: return 0.0, 0.0
        # TIGHTER STOP, BIGGER TARGET (Institutional R:R)
        sl_mult = self.params.get("atr_sl_multiplier", 1.0)
        tp_mult = self.params.get("atr_tp_multiplier", 4.5)
        return atr * sl_mult, atr * tp_mult

class SlingshotStrategy(BaseStrategy):
    """
    SUPREME EDITION: Uses CVD Divergence (The Lie Detector).
    """
    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]:
        if regime not in [Regime.CHOP, Regime.TRENDING_UP, Regime.TRENDING_DOWN]: return None

        df = self.engine.get_ohlc(token, 5)
        if len(df) < 22: return None

        swing_high = df['high'].rolling(20).max().shift(1).iloc[-1]
        swing_low = df['low'].rolling(20).min().shift(1).iloc[-1]
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 2. CVD Divergence Check
        has_bull_div = self.engine.micro_monitor.check_cvd_divergence(token, OrderSide.BUY)
        has_bear_div = self.engine.micro_monitor.check_cvd_divergence(token, OrderSide.SELL)

        # 3. Bullish Slingshot (Price reclaimed low + CVD Bullish Divergence)
        if prev['low'] < swing_low and current['close'] > swing_low:
            if has_bull_div: 
                L.info(f"🪝 CVD SLINGSHOT BUY {token}: Trap Reclaimed + Bullish CVD Divergence")
                return OrderSide.BUY

        # 4. Bearish Slingshot (Price failed high + CVD Bearish Divergence)
        if prev['high'] > swing_high and current['close'] < swing_high:
            if has_bear_div: 
                L.info(f"🪝 CVD SLINGSHOT SELL {token}: Trap Failed + Bearish CVD Divergence")
                return OrderSide.SELL


class SlingshotStrategy(BaseStrategy):
    """
    SUPREME EDITION: Uses CVD Divergence (The Lie Detector).
    """
    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]:
        if regime not in [Regime.CHOP, Regime.TRENDING_UP, Regime.TRENDING_DOWN]: return None

        df = self.engine.get_ohlc(token, 5)
        if len(df) < 22: return None

        swing_high = df['high'].rolling(20).max().shift(1).iloc[-1]
        swing_low = df['low'].rolling(20).min().shift(1).iloc[-1]
        current = df.iloc[-1]
        prev = df.iloc[-2]

        # 2. CVD Divergence Check
        has_bull_div = self.engine.micro_monitor.check_cvd_divergence(token, OrderSide.BUY)
        has_bear_div = self.engine.micro_monitor.check_cvd_divergence(token, OrderSide.SELL)

        # 3. Bullish Slingshot (Price reclaimed low + CVD Bullish Divergence)
        if prev['low'] < swing_low and current['close'] > swing_low:
            if has_bull_div: 
                L.info(f"🪝 CVD SLINGSHOT BUY {token}: Trap Reclaimed + Bullish CVD Divergence")
                return OrderSide.BUY

        # 4. Bearish Slingshot (Price failed high + CVD Bearish Divergence)
        if prev['high'] > swing_high and current['close'] < swing_high:
            if has_bear_div: 
                L.info(f"🪝 CVD SLINGSHOT SELL {token}: Trap Failed + Bearish CVD Divergence")
                return OrderSide.SELL

        return None

    def get_risk_params(self, token: int, side: OrderSide, current_time: datetime) -> Tuple[float, float]:
        df_1m = self.engine.get_ohlc(token, 1)
        atr = df_1m.ta.atr(14).iloc[-1]
        sl_mult = self.params.get("atr_sl_multiplier", 1.0)
        tp_mult = self.params.get("atr_tp_multiplier", 3.0)
        return atr * sl_mult, atr * tp_mult


class OpeningRangeBreakout(BaseStrategy):
    def __init__(self, name: StrategyName, engine: 'Engine', params: Dict):
        super().__init__(name, engine, params)
        self.orb_high = None
        self.orb_low = None
        self.is_agnostic = True 
        try:
            self.orb_set_time = dtime.fromisoformat(self.params["orb_set_time"])
            self.entry_window_end = dtime.fromisoformat(self.params["entry_window_end"])
        except (KeyError, ValueError):
            self.orb_set_time = dtime(9, 30)
            self.entry_window_end = dtime(9, 45)
        self.trades_taken_today = set()
        self._orb_set_date = None

    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]:
        now_time = current_time.time()
        today = current_time.date()
        if self._orb_set_date != today:
            self.orb_high = None
            self.orb_low = None
            self.trades_taken_today.clear()
            self._orb_set_date = today

        if now_time < self.orb_set_time: return None

        df_1m = self.engine.get_ohlc(token, 1)
        if self.orb_high is None:
            market_open = self.engine.timings_config["market_open"]
            relevant_bars = df_1m.between_time(market_open, self.orb_set_time)
            if not relevant_bars.empty:
                self.orb_high = relevant_bars['high'].max()
                self.orb_low = relevant_bars['low'].min()
                L.info(f"ORB Set for {token}: {self.orb_high}-{self.orb_low}")
            else: return None

        if now_time > self.entry_window_end: return None

        last_close = df_1m.iloc[-1]['close']
        last_vol = df_1m.iloc[-1]['volume']
        avg_vol = df_1m['volume'].rolling(10).mean().iloc[-1]

        if last_vol < avg_vol * 1.0: return None 

        if last_close > self.orb_high and (token, OrderSide.BUY) not in self.trades_taken_today:
            self.trades_taken_today.add((token, OrderSide.BUY))
            return OrderSide.BUY
            
        if last_close < self.orb_low and (token, OrderSide.SELL) not in self.trades_taken_today:
            self.trades_taken_today.add((token, OrderSide.SELL))
            return OrderSide.SELL
        return None

    def get_risk_params(self, token: int, side: OrderSide, current_time: datetime) -> Tuple[float, float]:
        if not self.orb_high or not self.orb_low: return 0.0, 0.0
        range_size = self.orb_high - self.orb_low
        risk = range_size * 0.5
        reward = risk * self.params.get("rr_multiplier", 1.5)
        return risk, reward


class TrendPullbackStrategy(BaseStrategy):
    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]:

        # --- EMA PULLBACK LOGIC (Restored) ---
        # 1. Trend Check: Price > EMA(50) for Uptrend, Price < EMA(50) for Downtrend
        df = self.engine.get_ohlc(token, 5)
        if len(df) < 55: return None

        df.ta.ema(length=50, append=True)
        df.ta.ema(length=20, append=True)

        current = df.iloc[-1]
        prev = df.iloc[-2]
        ema50 = current['EMA_50']
        ema20 = current['EMA_20']

        if not (ema50 > 0 and ema20 > 0): return None

        # UPTREND
        if current['close'] > ema50:
            # Pullback Condition: Low dipped near EMA20
            # Trigger: Green Candle closing above EMA20
            if prev['low'] <= ema20 * 1.001 and current['close'] > ema20 and current['close'] > current['open']:
                L.info(f"📈 TREND PULLBACK BUY {token}: Bounce off EMA20 in Uptrend.")
                return OrderSide.BUY

        # DOWNTREND
        elif current['close'] < ema50:
             # Pullback Condition: High rose near EMA20
             # Trigger: Red Candle closing below EMA20
             if prev['high'] >= ema20 * 0.999 and current['close'] < ema20 and current['close'] < current['open']:
                 L.info(f"📉 TREND PULLBACK SELL {token}: Rejection at EMA20 in Downtrend.")
                 return OrderSide.SELL

        return None

    def get_risk_params(self, token: int, side: OrderSide, current_time: datetime) -> Tuple[float, float]:
        df_1m = self.engine.get_ohlc(token, 1)
        return calculate_dynamic_risk_params(df_1m, self.params.get("atr_sl_multiplier", 1.2), self.params.get("atr_tp_multiplier", 2.0))


class VolatilityMeanReversionStrategy(BaseStrategy):
    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]:
        if regime != Regime.CHAOS: return None
        df = self.engine.get_ohlc(token, self.params["resample_minutes"])
        if len(df) < 20: return None
        df['ema'] = df.ta.ema(length=self.params["ema_period"])
        last_close = df['close'].iloc[-1]
        last_ema = df['ema'].iloc[-1]
        dev_pct = ((last_close - last_ema) / last_ema) * 100
        trigger = self.params.get("deviation_pct_trigger", 0.5)
        if dev_pct > trigger: return OrderSide.SELL
        if dev_pct < -trigger: return OrderSide.BUY
        return None

    def get_risk_params(self, token: int, side: OrderSide, current_time: datetime) -> Tuple[float, float]:
        df_1m = self.engine.get_ohlc(token, 1)
        return calculate_dynamic_risk_params(df_1m, 1.5, 1.5)

class GammaBurstStrategy(BaseStrategy):
    """
    The 'Nuclear' Option: Low IV + High VPIN + Breakout = Gamma Burst.
    Only trades when options are cheap enough to afford 10x returns.
    """
    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]:
        # 1. IV Check: Only active if IV Rank is low (Options are cheap)
        iv_rank = self.engine._get_iv_rank()
        max_iv = self.params.get("max_iv_percentile", 30.0)
        if iv_rank is None or iv_rank > max_iv: 
            return None # Too expensive to burst

        # 2. VPIN Toxic Flow Check (The "Smart Money" Trigger)
        # Note: Ensure engine has vpin_monitor initialized for this token
        if not self.engine.vpin_monitor.is_toxic():
            return None 

        # 3. Momentum Trigger (Breakout)
        df = self.engine.get_ohlc(token, 5) # 5-min timeframe
        if len(df) < 20: return None

        close = df['close'].iloc[-1]
        high_20 = df['high'].rolling(20).max().iloc[-2]
        low_20 = df['low'].rolling(20).min().iloc[-2]

        # Breakout of 20-candle High/Low
        if close > high_20: return OrderSide.BUY
        if close < low_20: return OrderSide.SELL

        return None

    def get_risk_params(self, token: int, side: OrderSide, current_time: datetime) -> Tuple[float, float]:
        # Very tight initial risk, massive reward target
        df_1m = self.engine.get_ohlc(token, 1)
        atr = df_1m.ta.atr(14).iloc[-1] if not df_1m.empty else 0
        if atr == 0: return 0.0, 0.0

        # Risk 0.5 ATR (Tight), Target 5.0 ATR (Home Run)
        return atr * 0.5, atr * 5.0
    
class VPINMonitor:
    """
    Volume-Synchronized Probability of Informed Trading (VPIN).
    Uses Bulk Volume Classification (BVC) to detect 'Toxic Flow'.
    """
    def __init__(self, volume_bucket_size: int = 5000, window_size: int = 50):
        self.bucket_size = volume_bucket_size
        self.window_size = window_size
        self.current_bucket_vol = 0
        self.buy_vol = 0
        self.sell_vol = 0
        self.buckets = deque(maxlen=window_size) # Stores (buy_vol, sell_vol)
        self.vpin_history = deque(maxlen=200)
        self.last_price = 0.0

    def update(self, tick: dict):
        """
        Updates volume buckets using BVC (price change heuristic).
        """
        price = tick.get('last_price')
        qty = tick.get('last_traded_quantity', 0)

        if not price or qty == 0: return

        # 1. Bulk Volume Classification (BVC)
        # If price rose, volume is aggressive buy. If fell, aggressive sell.
        # If neutral, split 50-50 (simplified for speed).
        buy_ratio = 0.5
        if self.last_price > 0:
            if price > self.last_price: buy_ratio = 1.0
            elif price < self.last_price: buy_ratio = 0.0

        self.last_price = price # Update for next tick

        v_buy = qty * buy_ratio
        v_sell = qty * (1 - buy_ratio)

        self.buy_vol += v_buy
        self.sell_vol += v_sell
        self.current_bucket_vol += qty

        # 2. Bucket Completion Check
        if self.current_bucket_vol >= self.bucket_size:
            # Store completed bucket
            self.buckets.append((self.buy_vol, self.sell_vol))
            self._calculate_vpin()
            
            # Reset accumulators
            self.current_bucket_vol = 0
            self.buy_vol = 0
            self.sell_vol = 0

    def _calculate_vpin(self):
        if len(self.buckets) < 10: return 0.0 # Need min data

        # VPIN = Sum(|Buy - Sell|) / Total Volume over window
        numerator = sum(abs(b[0] - b[1]) for b in self.buckets)
        denominator = sum(b[0] + b[1] for b in self.buckets)

        vpin = (numerator / denominator) if denominator > 0 else 0
        self.vpin_history.append(vpin)
        return vpin

    def is_toxic(self) -> bool:
        """Returns True if VPIN is in the top 10% of recent history (Toxic Flow)."""
        if len(self.vpin_history) < 20: return False
        # Dynamic threshold: 90th percentile of recent history
        threshold = np.percentile(list(self.vpin_history), 90)
        return self.vpin_history[-1] > threshold
    
class DealerGammaMatrix:
    """
    Calculates the 'Invisible Hand' of the market.
    Tracks Dealer Gamma Exposure (GEX) to predict Volatility Acceleration.
    """
    def __init__(self, book: InstrumentBook, prices: PriceBus, engine: 'Engine'):
        self.book = book
        self.prices = prices
        self.engine = engine
        self.lock = threading.RLock()
        self.gamma_profile: Dict[str, Dict] = {} # {Symbol: {'gex': float, 'timestamp': datetime}}

    def calculate_gex(self, token: int):
        try:
            u_sym = self.book.get_symbol(token)
            u_name = _get_underlying(u_sym)
            spot = self.prices.ltp(token)
            if not spot: return

            expiry = self.book.find_nearest_expiry_date(u_name)
            chain = self.book.get_option_chain(u_name, expiry)
            
            # --- OPTIMIZATION: KEY STRIKE FILTER ---
            # Only calculate Gamma for ATM +/- 10 strikes. Save CPU.
            step = self.book.step_size(u_name)
            atm_strike = round(spot / step) * step
            lower_bound = atm_strike - (10 * step)
            upper_bound = atm_strike + (10 * step)
            
            core_chain = chain[(chain['strike'] >= lower_bound) & (chain['strike'] <= upper_bound)]
            
            total_gamma = 0.0
            now = now_ist()
            close_time = self.engine.timings_config["market_close"]
            T = calculate_trading_time_to_expiry(now, expiry, self.engine.timings_config["market_open"], close_time, self.engine.nse_calendar)
            iv = 0.15 

            for _, row in core_chain.iterrows():
                d1, _ = _get_d1_d2(spot, row['strike'], T, 0.05, iv)
                if d1 is None: continue
                
                # Fallback Gamma Calc
                gamma = (norm.pdf(d1) / (spot * iv * math.sqrt(T)))
                gex_val = (gamma * row['open_interest'] * spot * 100)
                
                if row['instrument_type'] == 'CE': total_gamma += gex_val
                else: total_gamma -= gex_val

            with self.lock:
                self.gamma_profile[u_name] = {
                    'net_gex': total_gamma,
                    'timestamp': now,
                    'is_accelerating': total_gamma < 0, 
                    'is_suppressing': total_gamma > 0
                }
        except Exception as e:
            L.error(f"GEX Calc Error: {e}")

    def is_accelerating(self, token: int) -> bool:
        u_sym = self.book.get_symbol(token)
        if not u_sym: return False
        u_name = _get_underlying(u_sym)
        with self.lock:
            data = self.gamma_profile.get(u_name)
            if not data: return False 
            # If Net GEX is Negative, Dealers are hedging WITH the trend -> Acceleration
            return data['net_gex'] < 0 


# LeadLagPredictor REMOVED (Latency Trap)
    
# ==================================================================================================
# MAIN TRADING ENGINE
# ==================================================================================================

class Engine:
    def __init__(self,
    store_actor: StoreActor,
    book: InstrumentBook,
    prices: PriceBus,
    config: Dict):
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
        self.vpin_monitor = VPINMonitor()
        self.risk_manager: Optional[RiskManager] = None
        self.micro_monitor: Optional[MicrostructureMonitor] = None
        self.pos_manager: Optional[PositionManager] = None
        self.radar: Optional[ConstituentRadar] = None # Kingmaker Filter

        self.tick_thread: Optional[threading.Thread] = None
        self.order_thread: Optional[threading.Thread] = None
        self.trade_executor_thread: Optional[threading.Thread] = None
        self.scheduler_threads: Dict[str, threading.Thread] = {}

        self.heartbeats = {
        "SignalWorker": time.time(),
        "BarWorker": time.time(),
        "TickProcessor": time.time(),
        "OrderProcessor": time.time(),
        "TradeExecutor": time.time()
        }

        self.master_lock = threading.RLock()
        self.bars = BarStore(timeframes=[1, 3, 5, 15])

        self.atm_iv_cache = {}
        self.nifty_token = self.book.special_tokens.get("NIFTY")
        self.bn_token = self.book.special_tokens.get("BANKNIFTY")
        self.vix_token = self.book.special_tokens.get("INDIA VIX")
        if not all([self.nifty_token, self.bn_token]):
            raise SystemExit("FATAL: Could not find NIFTY/BANKNIFTY futures contracts from instrument file.")

        # Config Stale Warning (As requested)
        if 'constituent_weights' not in self.technical_config:
            L.warning("⚠️ ENGINE: 'constituent_weights' missing in technical config. Strategies may use stale hardcoded weights.")

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
        self.historical_avg_iv: Dict[int, pd.Series] = {}

        # --- HOT CHAMBER (Pre-computed Targets for Zero Latency) ---
        self.hot_chamber: Dict[str, Dict] = {
        "NIFTY_CE": None, "NIFTY_PE": None,
        "BANKNIFTY_CE": None, "BANKNIFTY_PE": None
        }
        self.hot_chamber_lock = threading.RLock()

        self.scheduler = self._setup_scheduler()

        self.gex_matrix = DealerGammaMatrix(book, prices, self)
        # LeadLagPredictor REMOVED (Latency Trap)
        # self.lead_lag = LeadLagPredictor(book, prices, self)

        # Add GEX calculation to scheduler manually or update _setup_scheduler
        self.scheduler["gex_calc"] = (self._update_gex, 60)

    def set_dependencies(self, trader: AbstractTrader, risk_manager: RiskManager, micro_monitor: MicrostructureMonitor, pos_manager: PositionManager, radar: ConstituentRadar):
        self.trader = trader
        self.risk_manager = risk_manager
        self.micro_monitor = micro_monitor
        self.pos_manager = pos_manager
        self.radar = radar
        if not PAPER_TRADING:
            self.prices.on_connect_callbacks.append(self.reconcile)
            L.info("All dependencies injected into Engine.")

    def get_ohlc(self, token: int, timeframe: int) -> pd.DataFrame:
        return self.bars.get_ohlc(token, timeframe)
    
    def get_skew_index(self, token: int) -> float:
        """
        Upgrade #1: Skew-Delta Divergence (The "Fear" Index).
        Calculates the difference between 25-Delta OTM Put IV and 25-Delta OTM Call IV.
    
        Formula: Skew = IV(Put_25d) - IV(Call_25d)
    
        Interpretation:
            - High Positive Skew: Puts are expensive (Fear of crash).
            - Rising Price + Rising Skew = TRAP (Market Makers selling calls but hedging with puts).
            """
        try:
            # 1. Basic Setup
            u_sym = self.book.get_symbol(token)
            if not u_sym: return 0.0
            u_name = _get_underlying(u_sym)
    
            spot = self.prices.ltp(token)
            if not spot: return 0.0
    
            # 2. Get Chain for Nearest Expiry
            expiry = self.book.find_nearest_expiry_date(u_name)
            if not expiry: return 0.0
    
            # Get Time to Expiry (T)
            now = now_ist()
            T = calculate_trading_time_to_expiry(now, expiry, self.timings_config["market_open"], self.timings_config["market_close"], self.nse_calendar)
    
            # Get Historical Volatility (HV) as fallback for IV calc
            # Using barstore directly is faster than self.get_ohlc copy
            bars = self.bars.get_ohlc(token, 1)
            hv = calculate_historical_volatility(bars['close']) if not bars.empty else 0.3

            chain = self.book.get_option_chain(u_name, expiry)
            if chain.empty: return 0.0

            # 3. Find 25-Delta Options
            # We scan OTM strikes: Calls > Spot, Puts < Spot
    
            target_delta = 0.25
            best_call_iv = None
            best_put_iv = None
            min_call_diff = 1.0
            min_put_diff = 1.0

            # Scan Loop
            for _, row in chain.iterrows():
                strike = row['strike']
                otype = row['instrument_type']
        
                # Optimization: Only check strikes within 10% of spot to save CPU
                if abs(strike - spot) / spot > 0.10: continue
        
                # Get Tick Data (Price is needed for IV)
                tick = self.prices.get_full_tick(int(row['instrument_token']))
                if not tick: continue
                ltp = tick.get('last_price', 0)
                if ltp <= 0: continue

                # Calculate IV & Greeks (JIT Accelerated)
                is_call = (otype == "CE")
                iv = implied_vol_jit(ltp, spot, strike, T, 0.05, is_call)
        
                # Filter bad IVs
                if iv <= 0.01 or iv > 5.0: iv = hv
        
                # Get Delta
                _, delta, _, _, _ = fast_greeks_jit(spot, strike, T, 0.05, iv, is_call)
        
                # Find match for 25 Delta
                # We want OTM 25 Delta.
                # Call Delta is roughly 0.25. Put Delta is roughly -0.25.
        
                if is_call and strike > spot: # OTM Call
                    diff = abs(delta - target_delta)
                    if diff < min_call_diff:
                        min_call_diff = diff
                        best_call_iv = iv
                    
                elif not is_call and strike < spot: # OTM Put
                    diff = abs(abs(delta) - target_delta)
                    if diff < min_put_diff:
                        min_put_diff = diff
                        best_put_iv = iv

            # 4. Calculate Skew
            if best_call_iv is not None and best_put_iv is not None:
                # Typical Skew: Puts (Downside) are usually more expensive than Calls (Upside).
                # Result is usually Positive.
                skew = best_put_iv - best_call_iv

                # L.debug(f"Skew Calc for {u_name}: PutIV({best_put_iv:.2f}) - CallIV({best_call_iv:.2f}) = {skew:.4f}")
                return skew
        
            return 0.0

        except Exception as e:
            L.warning(f"Error calculating Skew Index for {token}: {e}")
            return 0.0

    def _bar_worker(self):
        while self.running.is_set():
            self.heartbeats["BarWorker"] = time.time()
            try:
                ticks = self.prices.bar_queue.get(timeout=1)
                # ONLY update bars
                for t in ticks: self.bars.add_tick(t)
            except: pass
       
    def _update_market_breadth(self):
        if not self.nifty_50_tokens: return
        try:
            reply_q = queue.Queue()
            self.trader.order_actor.q.put({
                "type": "quote",
                "params": {"instrument_tokens": self.nifty_50_tokens},
                "reply_q": reply_q
            })
            quotes = None
            try:
                resp = reply_q.get(timeout=10.0)
                if resp['ok']: quotes = resp['res']
                else: raise Exception(resp.get('error'))
            except queue.Empty:
                L.error("Timeout waiting for OrderActor quote reply.")
                return

            if not quotes: return

            advances, declines = 0, 0
            for _, data in quotes.items():
                prev_close = data.get('ohlc', {}).get('close', 0)
                ltp = data.get('last_price')
                if ltp and prev_close > 0:
                    if ltp > prev_close: advances += 1
                    elif ltp < prev_close: declines += 1

            self.market_breadth = advances - declines
            L.info(f"Market breadth updated: A={advances}, D={declines}, Net={self.market_breadth}")
        except Exception as e:
            L.error(f"Failed to update market breadth: {e}")

    def _is_tick_sane(self, tick: Dict) -> bool:
        now_time = now_ist().time()
        if now_time < self.timings_config["market_settling_time"]: return True
        token, price = tick["instrument_token"], tick.get("last_price")
        if not price or price <= 0: return False

        last_price = self.last_known_prices.get(token)
        if last_price is None:
            self.last_known_prices[token] = price
            return True


        if abs(price - last_price) / last_price > self.sanity_check_pct:
            L.warning(f"INSANE TICK: {token}. New: {price}, Old: {last_price}. Discarding.")
            return False

        self.last_known_prices[token] = price
        return True
    
    def get_distance_to_oi_wall(self, token: int, side: OrderSide) -> float:
        """Returns percentage distance to the nearest major OI Wall."""
        try:
            u_sym = self.book.get_symbol(token)
            u_name = _get_underlying(u_sym)
            spot = self.prices.ltp(token)
            if not spot: return 999.0
    
            expiry = self.book.find_nearest_expiry_date(u_name)
            chain = self.book.get_option_chain(u_name, expiry)
            if chain.empty: return 999.0
    
            if side == OrderSide.BUY: # Look for Call Resistance (CE OI)
                ce_chain = chain[chain['instrument_type'] == 'CE']
                upper_chain = ce_chain[ce_chain['strike'] > spot]
                if upper_chain.empty: return 999.0
                # Find strike with Max OI
                wall_strike = upper_chain.loc[upper_chain['open_interest'].idxmax()]['strike']
                return (wall_strike - spot) / spot
        
            else: # Look for Put Support (PE OI)
                pe_chain = chain[chain['instrument_type'] == 'PE']
                lower_chain = pe_chain[pe_chain['strike'] < spot]
                if lower_chain.empty: return 999.0
                wall_strike = lower_chain.loc[lower_chain['open_interest'].idxmax()]['strike']
                return (spot - wall_strike) / spot
        except:
            return 999.0
    
    def _calculate_atm_iv(self, underlying_token: int, spot: float, expiry: date, T: float, hv: float) -> Optional[float]:
        underlying_name = _get_underlying(self.book.get_symbol(underlying_token))
        tep = self.book.step_size(underlying_name)
        atm_strike = round(spot / tep) * tep

        ivs = []
        for otype in ["CE", "PE"]:
            opt = self.book.find_option(underlying_name, expiry, atm_strike, otype)
            if not opt: continue
            tick = self.prices.get_full_tick(int(opt['instrument_token']))
            if tick and tick.get('last_price'):
                iv = calculate_iv(tick['last_price'], spot, atm_strike, T, 0.05, otype == "CE", hv_fallback=hv)
                ivs.append(iv)


        return sum(ivs) / len(ivs) if ivs else None
    
    def _update_atm_iv_cache(self):
        now, close_time = now_ist(), self.timings_config["market_close"]
        for token in [self.nifty_token, self.bn_token]:
            if not token: continue
            spot = self.prices.ltp(token)
            if not spot: continue
    
            name = _get_underlying(self.book.get_symbol(token))
            expiry = self.book.find_nearest_expiry_date(name)
            if not expiry: continue
    
            T = calculate_trading_time_to_expiry(now, expiry, self.timings_config["market_open"], close_time, self.nse_calendar)
            hv = calculate_historical_volatility(self.get_ohlc(token, 1)['close']) or 0.3
    
            if atm_iv := self._calculate_atm_iv(token, spot, expiry, T, hv):
                self.atm_iv_cache[name] = atm_iv

    def _update_hot_chamber(self):
        """Pre-selects best options for Flash Execution."""
        try:
            for token in [self.nifty_token, self.bn_token]:
                if not token: continue
                name = _get_underlying(self.book.get_symbol(token))
                expiry = self.book.find_nearest_expiry_date(name)
        
                ce = self._find_best_option_contract(token, expiry, OptionType.CE, StrategyName.MOMENTUM_IGNITION.value, self.regime, skip_filters=True)
                pe = self._find_best_option_contract(token, expiry, OptionType.PE, StrategyName.MOMENTUM_IGNITION.value, self.regime, skip_filters=True)
        
                with self.hot_chamber_lock:
                    if ce: self.hot_chamber[f"{name}_CE"] = ce
                    if pe: self.hot_chamber[f"{name}_PE"] = pe
        except Exception: pass

    def _signal_worker(self):
        L.info("Signal Worker started.")
        while self.running.is_set():
            try:
                self.heartbeats["SignalWorker"] = time.time()
                ticks = self.prices.tick_queue.get(timeout=1)
                sane = [t for t in ticks if self._is_tick_sane(t)]
        
                # Update VPIN (moved from TickProcessor)
                for t in sane:
                    self.vpin_monitor.update(t)

                # Update Microstructure
                if self.micro_monitor:
                    for t in sane:
                        try:
                            self.micro_monitor.update_tfi_score(t)
                        except Exception as e:
                            L.error(f"SignalWorker Logic Error: {e}", exc_info=True)
            except queue.Empty:
                continue
            except Exception as e:
                L.critical(f"🔥 FATAL SignalWorker Crash: {e}", exc_info=True)
                time.sleep(1)

    def health_check(self):
        """Checks if threads are alive (crashes) and heartbeats are recent (freezes)."""
        now = time.time()
        MAX_THREAD_SILENCE = 60.0

        # 1. Check Thread Liveness (Detects Crashes)
        threads = {
        "OrderProcessor": self.order_thread,
        "TradeExecutor": self.trade_executor_thread,
        "StoreActor": self.store_actor,
        "OrderActor": self.trader.order_actor
        }
        if hasattr(self, 'signal_thread'): threads["SignalWorker"] = self.signal_thread
        if hasattr(self, 'bar_thread'): threads["BarWorker"] = self.bar_thread

        for name, t in threads.items():
            if t is None: continue
            if not t.is_alive():
                L.critical(f"💀 THREAD DIED: {name}. HALTING BOT.")
                send_alert(f"💀 CRITICAL: {name} DIED (Crash). HALTING.", "critical")
                self.fatal_error_event.set()
                self.master_halt = True
                return


        # 2. Check Heartbeats (Detects Deadlocks)
        for name, last_beat in self.heartbeats.items():
            if name not in threads: continue
            if (now - last_beat) > MAX_THREAD_SILENCE:
                L.critical(f"\U0001f480 THREAD FROZEN: {name}. Last beat {now - last_beat:.1f}s ago. HALTING.")
                send_alert(f"\U0001f480 CRITICAL: {name} FROZEN (Deadlock). HALTING.", "critical")
                self.fatal_error_event.set()
                self.master_halt = True
                return

        # 3. Zombie Feed Check
        last_tick = self.prices.last_tick_reception_time
        if last_tick:
            silence = (now_ist() - last_tick).total_seconds()
            # If market is open and no data for 15s
            if silence > 15 and now > self.timings_config["market_open"]:
                L.critical(f"\U0001f480 ZOMBIE FEED: No ticks for {silence:.1f}s. RESTARTING WS.")
                self.prices.ws.close()  # Forces reconnect loop

    def _execute_flash_signal(self, flash):
        """Zero-latency execution path bypassing strategy loop."""
        u_name = "NIFTY" if flash['token'] == self.nifty_token else "BANKNIFTY" if flash['token'] == self.bn_token else None
        if not u_name: return

        # Grab pre-calc contract from Hot Chamber
        key = f"{u_name}_{'CE' if flash['side'] == OrderSide.BUY else 'PE'}"
        with self.hot_chamber_lock: pre = self.hot_chamber.get(key)
        if not pre: return

        # Cooldown Check
        now = now_ist()
        if (now - self.underlying_cooldown.get(flash['token'], datetime.min.replace(tzinfo=IST))).total_seconds() < 60: return

        L.info(f"⚡ FLASH TRIGGER: {u_name} TFI {flash['score']:.0f}. Firing {key} IOC.")

        # Direct Injection into Executor
        self.trade_signal_queue.put({
        "opt": pre['opt'], "lots": self.trading_config['max_lots_per_trade'],
        "strategy": "GAMMA_BURST_FLASH", "regime": self.regime.name,
        "option_sl_points": pre['ltp'] * 0.10, "option_tp_points": pre['ltp'] * 0.30,
        "total_trade_risk": 0.0, "underlying_sl": 0.0, "greeks": pre['greeks'],
        "max_trade_duration_minutes": 10, "oi_profit_target": None, "intended_risk_rupees": 0.0
        })
        self.underlying_cooldown[flash['token']] = now

    def check_oi_pressure(self, token: int) -> str:
        """
        Checks for 'Short Covering' (Bullish) or 'Long Unwinding' (Bearish).
        Returns: 'CALL_COVERING', 'PUT_UNWINDING', or None.
        """
        try:
            # Get underlying symbol name
            u_sym = self.book.get_symbol(token)
            u_name = _get_underlying(u_sym)
            spot = self.prices.ltp(token)
            if not spot: return None
    
            # Get Option Chain for nearest expiry
            expiry = self.book.find_nearest_expiry_date(u_name)
            chain = self.book.get_option_chain(u_name, expiry)
    
            # Filter for ATM strikes (Spot +/- 2%) where the "battle" is
            atm_chain = chain[
            (chain['strike'] >= spot * 0.98) &
            (chain['strike'] <= spot * 1.02)
            ]
    
            # Aggregated OI Change
            ce_oi_change = atm_chain[atm_chain['instrument_type'] == 'CE']['oi_change'].sum()
            pe_oi_change = atm_chain[atm_chain['instrument_type'] == 'PE']['oi_change'].sum()
    
            # Threshold: e.g., -200,000 contracts unwinding (Configurable)
            UNWIND_THRESHOLD = self.technical_config.get("oi_unwind_threshold", -200000)
    
            if ce_oi_change < UNWIND_THRESHOLD:
                L.info(f"\U0001f525 SHORT COVERING DETECTED on {u_name}! Call Writers fleeing.")
                return "CALL_COVERING"

            if pe_oi_change < UNWIND_THRESHOLD:
                L.info(f"\U0001f525 LONG UNWINDING DETECTED on {u_name}! Put Writers fleeing.")
                return "PUT_UNWINDING"

        except Exception as e:
            L.error(f"OI Pressure Check Failed: {e}")

        return None

    def _score_and_size_trade(self, signal: TradeSignal, token: int) -> Optional[Dict]:
        """
        The 'Brain' of the Engine.
        Applies Institutional Gatekeepers, Scores Confluence, and Sizes based on GEX.
        """
        score = 1.0
        is_agnostic = False
        score_log = ["Base: 1.0"]

        # --- 1. VIX CHAOS & VPIN FILTER (The Shield) ---
        vix = self.prices.ltp(self.vix_token) or 15.0
        chaos_reduction = 0.5 if vix > 30.0 and signal.strategy_name != StrategyName.VOLATILITY_MEAN_REVERSION else 1.0

        # VPIN: If Flow is Toxic, Disable Mean Reversion
        is_toxic = self.vpin_monitor.is_toxic(token) if hasattr(self, 'vpin_monitor') else False
        if is_toxic and signal.strategy_name in [StrategyName.VOLATILITY_MEAN_REVERSION, StrategyName.TREND_PULLBACK]:
            L.warning(f"🛑 Trade Blocked: Toxic Flow (VPIN) detected. Mean Reversion disabled.")
            return None

        # --- 2. VWAP FORTRESS (Strategy-Aware) ---
        df_5m = self.get_ohlc(token, 5)
        vwap = 0.0
        if not df_5m.empty:
            # Robust VWAP Calc: Sum(P*V) / Sum(V)
            cum_pv = (df_5m['close'] * df_5m['volume']).cumsum()
            cum_vol = df_5m['volume'].cumsum()
            vwap = (cum_pv / cum_vol).iloc[-1] if cum_vol.iloc[-1] > 0 else df_5m['close'].iloc[-1]
            curr_price = self.prices.ltp(token) or df_5m['close'].iloc[-1]

            # Define Strategies that are allowed to fade VWAP (Reversals)
            REVERSAL_STRATEGIES = [StrategyName.SLINGSHOT, StrategyName.VOLATILITY_MEAN_REVERSION, StrategyName.TREND_PULLBACK]

            if signal.strategy_name not in REVERSAL_STRATEGIES:
                # Rule: Never buy below VWAP, Never sell above VWAP for Trend Trades
                if signal.side == OrderSide.BUY and curr_price < vwap:
                    L.info(f"\U0001f6e1\ufe0f VWAP Block: Price {curr_price:.2f} < VWAP {vwap:.2f}. Longs denied for {signal.strategy_name}.")
                    return None
                if signal.side == OrderSide.SELL and curr_price > vwap:
                    L.info(f"\U0001f6e1\ufe0f VWAP Block: Price {curr_price:.2f} > VWAP {vwap:.2f}. Shorts denied for {signal.strategy_name}.")
                    return None
            else:
                # For Reversals, we just log it but don't block
                if (signal.side == OrderSide.BUY and curr_price < vwap) or (signal.side == OrderSide.SELL and curr_price > vwap):
                    score_log.append("Counter-VWAP Reversal (Allowed)")

        # --- 3. TWIN ENGINE & KINGPIN LOCK (The Confirmation) ---
        other_token = self.bn_token if token == self.nifty_token else self.nifty_token
        kingpin_token = 738561 if token == self.nifty_token else 341249

        # A. Twin Engine Check (Index Sync)
        other_df = self.get_ohlc(other_token, 1)
        other_ltp = self.prices.ltp(other_token)
        if not other_df.empty and other_ltp:
            other_open = other_df.iloc[-min(len(other_df), 2)]['open']
            other_change = (other_ltp - other_open) / other_open

            is_reversal = signal.strategy_name in [StrategyName.SLINGSHOT, StrategyName.VOLATILITY_MEAN_REVERSION]

            if signal.side == OrderSide.BUY and other_change < -0.0003:
                if is_reversal:
                    score -= 2.0; score_log.append("Twin Engine Divergence (-2.0)")
                else:
                    L.warning(f"\U0001f6d1 Twin Engine Fail: Target BUY, but Peer Index down {other_change*100:.3f}%")
                    return None
            elif signal.side == OrderSide.SELL and other_change > 0.0003:
                if is_reversal:
                    score -= 2.0; score_log.append("Twin Engine Divergence (-2.0)")
                else:
                    L.warning(f"\U0001f6d1 Twin Engine Fail: Target SELL, but Peer Index up {other_change*100:.3f}%")
                    return None
            else:
                score_log.append("Twin Engine: OK")

        # B. Kingpin Lock (Constituent Check)
        if kingpin_token:
            kp_df = self.get_ohlc(kingpin_token, 5)
            if not kp_df.empty:
                kp_vwap = ((kp_df['close'] * kp_df['volume']).cumsum() / kp_df['volume'].cumsum()).iloc[-1]
                kp_ltp = self.prices.ltp(kingpin_token)
                if kp_ltp and kp_vwap > 0:
                    if signal.side == OrderSide.BUY and kp_ltp < kp_vwap:
                        L.warning(f"\U0001f6d1 Kingpin Lock: General is WEAK (< VWAP). Cannot Buy Index.")
                        return None
                    if signal.side == OrderSide.SELL and kp_ltp > kp_vwap:
                        L.warning(f"\U0001f6d1 Kingpin Lock: General is STRONG (> VWAP). Cannot Sell Index.")
                        return None

        # --- 4. EFFORT vs RESULT (Volume Efficiency) ---
        if len(df_5m) > 2:
            curr_bar = df_5m.iloc[-1]
            vol, rng = curr_bar['volume'], curr_bar['high'] - curr_bar['low']
            if vol > 0:
                efficiency = rng / vol
                avg_eff = ((df_5m['high'] - df_5m['low']).rolling(10).sum() / df_5m['volume'].rolling(10).sum()).iloc[-1]

                # High Vol + Low Range = CHURN (Trap)
                if avg_eff > 0 and efficiency < (avg_eff * 0.5):
                    L.warning(f"\U0001f6d1 Churn Detected: High Vol / Low Range.")
                    return None
                score_log.append("Vol Eff: OK")

        # --- 5. CONFLUENCE SCORING ---

        # Regime Match
        if "AGNOSTIC" in self.strategies and signal.strategy_name in [s.name.value for s in self.strategies.get("AGNOSTIC", [])]:
            is_agnostic = True; score_log.append("Agnostic")
        else:
            good_combos = {
                Regime.TRENDING_UP: [StrategyName.TREND_PULLBACK, StrategyName.SLINGSHOT],
                Regime.TRENDING_DOWN: [StrategyName.TREND_PULLBACK, StrategyName.SLINGSHOT],
                Regime.COMPRESSION: [StrategyName.MOMENTUM_IGNITION],
                Regime.CHAOS: [StrategyName.VOLATILITY_MEAN_REVERSION]
            }
            if signal.strategy_name in good_combos.get(self.regime, []):
                score += 1.0; score_log.append("Regime Match: +1.0")

        # Radar (Kingmaker)
        if self.radar:
            idx = "BANKNIFTY" if token == self.bn_token else "NIFTY"
            radar_score = self.radar.get_weighted_momentum(idx)
            if (signal.side == OrderSide.BUY and radar_score > 20) or (signal.side == OrderSide.SELL and radar_score < -20):
                score += 2.0; score_log.append(f"Radar ({radar_score:.0f}): +2.0")
            elif (signal.side == OrderSide.BUY and radar_score < -10) or (signal.side == OrderSide.SELL and radar_score > 10):
                return None  # Block if constituents disagree

        # StatArb
        if self.nifty_bn_zscore is not None:
            z = self.nifty_bn_zscore
            if (token == self.nifty_token and ((signal.side==OrderSide.BUY and z < -1.5) or (signal.side==OrderSide.SELL and z > 1.5))) or \
               (token == self.bn_token and ((signal.side==OrderSide.BUY and z > 1.5) or (signal.side==OrderSide.SELL and z < -1.5))):
                score += 1.0; score_log.append("StatArb: +1.0")

        # Market Breadth
        if self.market_breadth != 0:
            thr = self.technical_config.get("breadth_score_threshold", 10)
            if (signal.side == OrderSide.BUY and self.market_breadth > thr) or (signal.side == OrderSide.SELL and self.market_breadth < -thr):
                score += 1.0; score_log.append("Breadth: +1.0")
            elif (signal.side == OrderSide.BUY and self.market_breadth < -thr) or (signal.side == OrderSide.SELL and self.market_breadth > thr):
                score -= 1.0

        # OI Pressure (The Decoupler)
        oi_signal = self.check_oi_pressure(token)
        if oi_signal:
            if (signal.side == OrderSide.BUY and oi_signal == "CALL_COVERING") or \
               (signal.side == OrderSide.SELL and oi_signal == "PUT_UNWINDING"):
                score += 3.0; score_log.append("\U0001f525 OI Pressure: +3.0")

        # --- 6. FINAL THRESHOLD CHECK ---
        if score < self.trading_config.get("min_trade_score", 1.0):
            return None

        # --- 7. PARAMETER CALCULATION (Rent-to-Speed Selection) ---
        prelim = self.get_trade_params(token, signal.side, signal.risk_points, signal.reward_points, signal.strategy_name.value, self.regime, score)
        if not prelim: return None

        # --- 8. GEX SIZING (The Multiplier) ---
        u_sym = self.book.get_symbol(token)
        gex_data = self.gex_matrix.gamma_profile.get(_get_underlying(u_sym))

        gex_multiplier = 1.0
        if gex_data:
            if gex_data['is_accelerating']:  # Negative GEX
                gex_multiplier = 2.0
                score_log.append("Neg GEX: Size x2")
            elif gex_data['is_suppressing']:  # Positive GEX
                gex_multiplier = 0.5
                score_log.append("Pos GEX: Size x0.5")

        prelim['lots'] = max(1, int(prelim['lots'] * gex_multiplier))
        prelim['intended_risk_rupees'] *= gex_multiplier
        prelim['chaos_reduction'] = chaos_reduction

        L.info(f"\u2705 Trade Scored: {score:.1f} | Size Mult: {gex_multiplier} | Logic: {', '.join(score_log)}")
        return {"score": max(0, score), "params": prelim, "is_agnostic": is_agnostic}
    
    def _handle_order_update_from_queue(self, order: Dict):
        oid, status = str(order.get('order_id')), order.get('status')
        pos_id = f"LIVE_{oid}"

        with self.trader.lock:
            # Try to find position by Entry ID first, then by other IDs
            pos = self.trader.positions.get(pos_id)
            if not pos:
                pos = next((p for p in self.trader.positions.values() if oid in [p.tp_order_id, p.exit_order_id, p.slm_order_id] or oid in p.partial_exit_order_ids), None)
                if not pos:
                    return

            # --- 1. ENTRY UPDATES ---
            if oid == pos.entry_order_id:
                if status == 'COMPLETE':
                    self._handle_entry_fill(pos, order)

                elif status == 'CANCELLED':
                    L.warning(f"\u26d4 Entry Order {oid} CANCELLED (IOC/Timeout). Killing Position {pos.tradingsymbol}.")
                    pos.status = PositionStatus.REJECTED.value
                    pos.exit_reason = "ENTRY_CANCELLED_IOC"
                    self.store_actor.q.put({"type": "upsert_position", "pos": pos})
                    self.trader.positions.pop(pos.id, None)

                elif status == 'REJECTED':
                    L.error(f"Entry order {oid} REJECTED. Reason: {order.get('status_message')}")
                    pos.status = PositionStatus.REJECTED.value
                    pos.exit_reason = f"ENTRY_REJECTED: {order.get('status_message')}"
                    self.store_actor.q.put({"type": "upsert_position", "pos": pos})
                    self.trader.positions.pop(pos.id, None)

            # --- 2. EXIT/BRACKET UPDATES ---
            elif oid in [pos.tp_order_id, pos.exit_order_id, pos.slm_order_id] or oid in pos.partial_exit_order_ids:
                if status == 'COMPLETE':
                    if oid in pos.partial_exit_order_ids:
                        self.trader._handle_partial_exit_fill(pos, order)
                    else:
                        if oid == pos.slm_order_id:
                            pos.exit_reason = "SL_HIT_BROKER"
                        elif oid == pos.tp_order_id:
                            pos.exit_reason = "TP_HIT_BROKER"
                        self._handle_exit_fill(pos, order)

                elif status == 'CANCELLED':
                    L.info(f"\u2139\ufe0f Exit/Bracket Order {oid} for {pos.tradingsymbol} was CANCELLED.")
                    if oid == pos.exit_order_id:
                        L.warning(f"\u26a0\ufe0f CRITICAL: Main Exit Order {oid} Cancelled! Position {pos.tradingsymbol} might be stuck.")
                        pos.status = PositionStatus.ACTIVE.value
                        self.store_actor.q.put({"type": "upsert_position", "pos": pos})

                elif status == 'REJECTED':
                    if oid in [pos.tp_order_id, pos.slm_order_id]:
                        L.critical(f"Bracket order {oid} REJECTED: {order.get('status_message')}")
                        send_alert(f"\U0001f525 CRITICAL: BRACKET ORDER REJECTED for {pos.tradingsymbol}. Closing position!", "critical")
                        self.trader.close_position(pos, "BRACKET_REJECTED")

    def _order_processor_worker(self):
        L.info("Order processor worker started.")
        while self.running.is_set():
            self.heartbeats["OrderProcessor"] = time.time()
            try:
                order = self.prices.order_update_queue.get(timeout=1)
                self._handle_order_update_from_queue(order)
            except Empty: continue
            except Exception as e: L.error(f"Error in order processor worker: {e}", exc_info=True)
    
    def _update_gex(self):
        """Background task to update Dealer Gamma Exposure."""
        for t in [self.nifty_token, self.bn_token]:
            if t: self.gex_matrix.calculate_gex(t)

    def start(self):
        if not all([self.trader, self.risk_manager, self.micro_monitor, self.pos_manager, self.radar]):
            raise SystemExit("FATAL: Dependencies not set.")

        self.warm_up()
        self.prices.start()
        if not self.prices.connected.wait(10): raise SystemExit("FATAL: PriceBus WebSocket failed.")

        self.signal_thread = threading.Thread(target=self._signal_worker, daemon=True, name="SignalWorker")
        self.signal_thread.start()

        self.bar_thread = threading.Thread(target=self._bar_worker, daemon=True, name="BarWorker")
        self.bar_thread.start()

        self.prices.subscribe([self.nifty_token, self.bn_token, self.vix_token])
        self.running.set()

        for t_name, t_func in [("OrderProcessor", self._order_processor_worker),
                               ("TradeExecutor", self._trade_executor_worker)]:
            t = threading.Thread(target=t_func, name=t_name, daemon=True)
            t.start()
            if t_name == "OrderProcessor": self.order_thread = t
            else: self.trade_executor_thread = t

        for name, (func, interval) in self.scheduler.items():
            t = threading.Thread(target=self._run_task_in_loop, args=(func, interval, name), name=name, daemon=True)
            t.start()
            self.scheduler_threads[name] = t
            L.info(f"Started scheduler thread for '{name}' with {interval}s interval.")

        self.loop()

    def _run_task_in_loop(self, func: Callable, interval: int, name: str):
        while self.running.is_set():
            try:
                if name == "strategic_planner":
                    with self.master_lock:
                        check = self.master_halt
                    if not check:
                        func()
                else:
                    func()
            except Exception as e: L.error(f"Error in task '{name}': {e}", exc_info=True)
            time.sleep(interval)

    def _send_eod_report(self):
        if now_ist().time() > self.timings_config["market_close"] and not self.eod_report_sent:
            L.info("Sending EOD report...")
            q = queue.Queue()
            self.store_actor.q.put({"type": "get_todays_trades_stats", "reply_q": q})
            try:
                resp = q.get(timeout=10)
                wins, losses = resp['res'] if resp['ok'] else (0, 0)
            except: wins, losses = 0, 0
    
            total = wins + losses
            rate = (wins/total*100) if total > 0 else 0
            msg = f"📊 EOD Report\nPnL: ₹{self.trader.daily_realized_pnl:,.2f}\nTrades: {total} (W:{wins} L:{losses})\nWin Rate: {rate:.1f}%"
            send_alert(msg)
            self.eod_report_sent = True

    def loop(self):
        send_alert("🛡️ SENTINEL PRIME ENGAGED.")
        while now_ist().time() < self.timings_config["market_open"] and self.running.is_set(): time.sleep(60)
        if not self.running.is_set(): self.stop(); return

        send_alert("🔔 Market OPEN.")
        try:
            while self.running.is_set():
                now = now_ist()
                if now.time() >= self.timings_config["market_close"]: break

                if self.fatal_error_event.is_set():
                    with self.master_lock: self.master_halt = True
                    send_alert("🔥 FATAL ERROR. HALTING.", "critical")
                    self.fatal_error_event.clear()

                with self.master_lock:
                    if os.path.exists(KILL_SWITCH_FILE) and not self.master_halt:
                        self.master_halt = True
                        if G_HALTED_STATUS: G_HALTED_STATUS.set(1)
                        send_alert("\u26d4 KILL SWITCH DETECTED.", "critical")

                if not self.eod_flatten_triggered and now.time() >= self.timings_config["eod_flatten_time"]:
                    self.master_halt = True; self.eod_flatten_triggered = True
                    if G_HALTED_STATUS: G_HALTED_STATUS.set(1)
                    with self.trader.lock:
                        for p in [pos for pos in self.trader.positions.values() if pos.status not in ["CLOSED", "PENDING_CLOSURE"]]:
                            self.trader.close_position(p, "EOD_FLATTEN")

                time.sleep(10)
        except KeyboardInterrupt: self.stop(); return

        if not self.eod_report_sent: self._send_eod_report()
        if self.last_trading_day:
            self.store_actor.q.put({"type": "set_kv", "key": f"daily_pnl_{self.last_trading_day}", "value": str(self.trader.daily_realized_pnl)})

        send_alert(f"\U0001f4a4 Shutdown. PnL: \u20b9{self.trader.daily_realized_pnl:,.2f}")
        self.stop()

    def _load_strategies(self):
        strats = {r: [] for r in Regime}
        strats["AGNOSTIC"] = []

        cfg = self.config["strategies"]

        if "Leviathan" in cfg:
            # Note: We map config key "Leviathan" to our new class
            lev = LeviathanStrategy(StrategyName.MOMENTUM_IGNITION, self, cfg['Leviathan'])
            # Leviathan is Agnostic (Trades Trend AND Reversal)
            strats["AGNOSTIC"].append(lev)

        # 2. TRENDING: TREND PULLBACK & SLINGSHOT (Bear/Bull Traps)
        if "TrendPullback" in cfg:
            tp = TrendPullbackStrategy(StrategyName.TREND_PULLBACK, self, cfg['TrendPullback'])
            strats[Regime.TRENDING_UP].append(tp)
            strats[Regime.TRENDING_DOWN].append(tp)

        # Slingshot catches reversals/traps in trends
        if "Slingshot" in cfg:
            slingshot = SlingshotStrategy(StrategyName.SLINGSHOT, self, cfg['Slingshot'])
            strats[Regime.TRENDING_UP].append(slingshot)
            strats[Regime.TRENDING_DOWN].append(slingshot)

            # 3. CHOP: SLINGSHOT ONLY (Replaces MeanReversion)
            strats[Regime.CHOP].append(slingshot)

        # 4. CHAOS: VOLATILITY MEAN REVERSION
        if "VolatilityMeanReversion" in cfg:
            vmr = VolatilityMeanReversionStrategy(StrategyName.VOLATILITY_MEAN_REVERSION, self, cfg['VolatilityMeanReversion'])
            strats[Regime.CHAOS].append(vmr)

        # 5. AGNOSTIC: ORB & GAMMA BURST
        if "OpeningRangeBreakout" in cfg:
            orb = OpeningRangeBreakout(StrategyName.OPENING_RANGE_BREAKOUT, self, cfg['OpeningRangeBreakout'])
            orb.is_agnostic = True
            strats["AGNOSTIC"].append(orb)

        if "GammaBurst" in cfg:
            gb = GammaBurstStrategy(StrategyName.GAMMA_BURST, self, cfg['GammaBurst'])
            strats["AGNOSTIC"].append(gb)

        return strats


    def _setup_scheduler(self):
        tasks = {
        "strategic_planner": (self._run_strategic_planner, 2),
        "hot_chamber_reload": (self._update_hot_chamber, 30),
        "position_management": (self.pos_manager.manage_positions, 1),
        "pnl_updater": (self.risk_manager.update_pnl_metrics, 2),
        "circuit_breaker": (self.risk_manager.check_circuit_breaker, 5),
        "reconciliation": (self.reconcile, 300),
        "health_check": (self.health_check, 60),
        "garbage_collector": (self._run_gc, 300),
        "strategy_weighting": (self.risk_manager._update_strategy_weights, 3600),
        "eod_report": (self._send_eod_report, 300),
        "data_persistence": (self._persist_bar_data, 3600),
        "bar_reconciliation": (self._reconcile_bars, 900),
        "pnl_reconciliation": (self.risk_manager.reconcile_broker_pnl, 900),
        "atm_iv_cache": (self._update_atm_iv_cache, 10),
        "market_breadth": (self._update_market_breadth, 300)
        }
        if METRICS_APP: tasks["prometheus_metrics"] = (self._update_prometheus_metrics, 15)
        return tasks
    
    def _run_gc(self):
        """Upgrade #8: Aggressive Memory Management."""
        # BarStore uses deques with maxlen=2000 (auto-trimming). No manual GC needed.


        # Force Python GC
        collected = gc.collect()
        L.info(f"Garbage Collector ran. Freed {collected} objects.")


    def _reconcile_bars(self):
        """Reconciles bar data to ensure consistency."""
        pass

    def _run_strategic_planner(self):
        """
        The 'Cortex' of the Engine.
        Continuously evaluates Market Regime and triggers Strategy Logic.
        """
        # 1. Update Regime (Every Cycle)
        try:
            regime, token, conf = self.classifier.get_raw_classification(self.regime)

            with self.master_lock:
                # Hysteresis Logic to prevent Regime Flickering
                if regime != self.regime:
                    if regime == self.potential_regime:
                        self.potential_regime_count += 1
                    else:
                        self.potential_regime = regime
                        self.potential_regime_count = 1

                    if self.potential_regime_count >= self.regime_confirmation_threshold:
                        L.info(f"\U0001f30d REGIME CHANGE: {self.regime.name} -> {regime.name} (Conf: {conf:.2f})")
                        self.regime = regime
                        self.regime_confidence = conf
                        self.last_regime_change_time = now_ist()
                        self.potential_regime_count = 0
                else:
                    self.potential_regime_count = 0
                    self.regime_confidence = conf
        except Exception as e:
            L.error(f"Regime Classification Failed: {e}")

        # 2. Evaluate Strategies
        if self.regime_halt or self.master_halt: return

        active_strats = self.strategies.get(self.regime, []) + self.strategies.get("AGNOSTIC", [])

        # Iterate through NIFTY and BANKNIFTY
        for token in [self.nifty_token, self.bn_token]:
            if not token: continue

            # Skip if data is stale
            if not self.prices.connected.is_set(): continue

            now = now_ist()
            for strategy in active_strats:
                try:
                    # Evaluate returns a TradeSignal object if a setup is found
                    signal = strategy.evaluate(token, self.regime, now)
                    if signal:
                        # Pass to the Gatekeeper (Score & Size)
                        trade_packet = self._score_and_size_trade(signal, token)
                        if trade_packet:
                            L.info(f"\U0001f680 SIGNAL ACCEPTED: {signal.strategy_name} on {token}. Score: {trade_packet['score']}")
                            self.trade_signal_queue.put(trade_packet['params'])
                except Exception as e:
                    L.error(f"Strategy {strategy.name} failed on {token}: {e}")


    def _reset_daily_state(self):
        L.info("Resetting daily state."); self.risk_manager.reset_daily_state(self.last_trading_day)
        with self.master_lock:
            self.master_halt = False; self.regime_halt = False; self.last_trade_timestamp = None
            if os.path.exists(KILL_SWITCH_FILE): os.remove(KILL_SWITCH_FILE)
            if G_HALTED_STATUS: G_HALTED_STATUS.set(0)
            self.eod_flatten_triggered = False; self.eod_report_sent = False; self.last_trading_day = now_ist().date()

    def warm_up(self):
        L.info("Warming up...")
        self.last_trading_day = now_ist().date()
        self.risk_manager.load_persistent_state(self.last_trading_day)

        try:
            path = self.technical_config.get("nifty_50_constituents_file")
            if path and os.path.exists(path):
                with open(path, 'r') as f: self.nifty_50_tokens = json.load(f)
        except: pass

        to_date = self.last_trading_day
        from_date = to_date - timedelta(days=self.technical_config["warmup_days"])

        for token in [self.nifty_token, self.bn_token]:
            if not token: continue
            q = queue.Queue()
            self.trader.order_actor.q.put({"type": "historical_data", "params": {"instrument_token": token, "from_date": from_date, "to_date": to_date, "interval": "minute"}, "reply_q": q})
            try:
                res = q.get(timeout=30)
                if res['ok'] and res['res']: self.bars.prime(token, pd.DataFrame(res['res']))
            except: L.error(f"Warmup failed for {token}")

        if self.vix_token:
            q = queue.Queue()
            self.trader.order_actor.q.put({"type": "historical_data", "params": {"instrument_token": self.vix_token, "from_date": to_date - timedelta(days=365), "to_date": to_date, "interval": "day"}, "reply_q": q})
            try:
                res = q.get(timeout=30)
                if res['ok'] and res['res']: self.vix_long_history_df = pd.DataFrame(res['res'])
            except: pass
    
    def reconcile(self):
        if PAPER_TRADING:
            return
        L.info("--- Starting State Reconciliation with Broker ---")
        try:
            # 1. Get Real Positions from Broker (OrderActor)
            pos_reply_q = queue.Queue()
            self.trader.order_actor.q.put({"type": "positions", "reply_q": pos_reply_q})

            broker_positions_data = None
            try:
                pos_resp = pos_reply_q.get(timeout=10.0)
                if pos_resp['ok']:
                    broker_positions_data = pos_resp['res']
            except queue.Empty:
                L.error("Reconcile: Timeout getting broker positions.")
                return

            if not broker_positions_data: return

            # 2. Get Bot's Internal Positions (StoreActor)
            db_reply_q = queue.Queue()
            self.store_actor.q.put({"type": "load_open_positions", "reply_q": db_reply_q})

            db_positions = {}
            try:
                db_resp = db_reply_q.get(timeout=10.0)
                if db_resp['ok']:
                    db_positions = db_resp['res']
            except queue.Empty: return

            # 3. Compare and Kill Rogues
            broker_positions_raw = broker_positions_data.get('net', [])
            # Filter for open MIS (Intraday) positions
            broker_positions_map = {
                pos['tradingsymbol']: pos
                for pos in broker_positions_raw
                if pos.get('product') == 'MIS' and abs(pos.get('quantity', 0)) > 0
            }

            broker_symbols = set(broker_positions_map.keys())
            db_symbols = {p.tradingsymbol for p in db_positions.values()}

            # CASE A: Bot thinks it's open, Broker says it's closed -> Mark Closed in DB
            for symbol in db_symbols - broker_symbols:
                pos = next((p for p in db_positions.values() if p.tradingsymbol == symbol), None)
                if pos:
                    L.warning(f"RECONCILE: Ghost position {symbol} detected. Marking CLOSED.")
                    pos.status = PositionStatus.CLOSED.value
                    pos.exit_reason = "RECONCILE_GHOST_CLOSE"
                    self.store_actor.q.put({"type": "upsert_position", "pos": pos})

            # CASE B: Broker has a position, Bot knows nothing -> FLATTEN IMMEDIATELY
            for symbol in broker_symbols - db_symbols:
                rogue_pos_data = broker_positions_map[symbol]
                qty = rogue_pos_data['quantity']
                send_alert(f"\U0001f525 ROGUE POSITION FOUND: {symbol} (Qty: {qty}). Auto-flattening!", "critical")

                transaction_type = "SELL" if qty > 0 else "BUY"

                L.info(f"Reconcile: Placing market order to flatten rogue {symbol}...")
                place_params = {
                    "variety": "regular",
                    "exchange": "NFO",
                    "tradingsymbol": symbol,
                    "transaction_type": transaction_type,
                    "quantity": abs(qty),
                    "product": "MIS",
                    "order_type": "MARKET"
                }
        except Exception as e:
            L.error(f"Reconciliation logic failed: {e}", exc_info=True)
    def _get_iv_rank(self) -> Optional[float]:
        if self.vix_long_history_df.empty or not self.vix_token: return None
        curr = self.prices.ltp(self.vix_token)
        if not curr: return None
        mn, mx = self.vix_long_history_df['close'].min(), self.vix_long_history_df['close'].max()
        return ((curr - mn) / (mx - mn)) * 100 if mx > mn else 50.0

    def _get_oi_barriers(self, name, expiry, spot):
        try:
            chain = self.book.get_option_chain(name, expiry)
            if chain.empty: return None, None
            calls = chain[chain['instrument_type'] == 'CE']
            puts = chain[chain['instrument_type'] == 'PE']
            res = calls[calls['strike'] > spot].sort_values('open_interest', ascending=False).head(1)
            sup = puts[puts['strike'] < spot].sort_values('open_interest', ascending=False).head(1)
            return (res.iloc[0]['strike'] if not res.empty else None, sup.iloc[0]['strike'] if not sup.empty else None)
        except: return None, None

    def _find_best_option_contract(
        self,
        underlying_token: int,
        expiry: date,
        option_type: OptionType,
        strategy: str,
        regime: Regime,
        skip_filters: bool = False
        ) -> Optional[Dict]:
            """
            UPGRADE: 'Rent-to-Speed' Optimizer + 'Delta Shifter' + 'Microstructure Gatekeeper'.
            Selects options based on Gamma Efficiency while respecting DTE physics and Order Book reality.
            """
            spot = self.prices.ltp(underlying_token)
            if not spot: return None

            now, close_time = now_ist(), self.timings_config["market_close"]
            T = calculate_trading_time_to_expiry(now, expiry, self.timings_config["market_open"], close_time, self.nse_calendar)
            dte = (expiry - now.date()).days

            # --- UPGRADE 1: DELTA SHIFTER ---
            # Adapts target Delta based on Days to Expiry (DTE)
            if dte > 1:
                # Mon-Wed: Buy ITM (Stock Replacement) to shield against Theta
                min_delta, max_delta = 0.55, 0.85
            elif dte == 1:
                # Wed: Transition Day
                min_delta, max_delta = 0.45, 0.75
            else:
                # Thu: Expiry Day. Max Gamma.
                min_delta, max_delta = 0.30, 0.65

            # Fetch Chain
            u_sym = self.book.get_symbol(underlying_token)
            chain = self.book.get_option_chain(_get_underlying(u_sym), expiry)

            # Filter Liquidity Zone (Spot +/- 3% to save CPU cycles)
            trade_chain = chain[
            (chain['instrument_type'] == option_type.value) &
            (chain['strike'] >= spot * 0.97) &
            (chain['strike'] <= spot * 1.03)
            ].copy()

            candidates = []
            is_call = (option_type == OptionType.CE)

            # Get HV for fallback if IV calculation fails
            bars = self.bars.get_ohlc(underlying_token, 1)
            hv = calculate_historical_volatility(bars['close']) if not bars.empty else 0.3

            for _, row in trade_chain.iterrows():
                tick = self.prices.get_full_tick(int(row['instrument_token']))
                if not tick: continue
 
                ltp = tick.get('last_price', 0)
                if ltp < 10.0: continue # Filter penny options

                # --- UPGRADE 2: MICROSTRUCTURE GATEKEEPER ---
                # Veto bad spreads and thin books before doing math
                depth = tick.get('depth')
                if depth:
                    bid = depth['buy'][0]['price']
                    ask = depth['sell'][0]['price']

                    if bid > 0 and ask > 0:
                        spread_pct = (ask - bid) / ltp
                        # REJECT if spread > 1.0% (Retail cannot afford this tax)
                        if spread_pct > 0.01:
                            continue

                        # REJECT if Best Ask Quantity is too low to fill us (Assumes < 500 is thin)
                        if depth['sell'][0]['quantity'] < 500:
                            continue

                # JIT Greeks Calculation
                iv = implied_vol_jit(ltp, spot, row['strike'], T, 0.05, is_call)
                if iv <= 0.01 or iv > 5.0: iv = hv
                _, delta, vega, gamma, theta = fast_greeks_jit(spot, row['strike'], T, 0.05, iv, is_call)

                # Delta Guardrails (The Shifter)
                if abs(delta) < min_delta or abs(delta) > max_delta:
                    continue

                # --- UPGRADE 3: RENT-TO-SPEED RATIO ---
                # Formula: (Gamma * 10000) / (Rent + epsilon)
                # We want maximum Acceleration (Gamma) for minimum Rent (Theta)
                rent_cost = abs(theta) + 1e-9
                efficiency_score = (gamma * 10000) / rent_cost

                # Liquidity Penalty: Don't pick efficient options if they have no volume
                if not skip_filters and tick.get('volume', 0) < 5000:
                    efficiency_score *= 0.1

                candidates.append({
                    'score': efficiency_score,
                    'opt': row.to_dict(),
                    'ltp': ltp,
                    'greeks': {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'iv': iv}
                })

            if not candidates: return None

            # Select Winner (Highest Rent-to-Speed Score)
            candidates.sort(key=lambda x: x['score'], reverse=True)
            return candidates[0]
    
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
            u_sym = self.book.get_symbol(token)
            if not u_sym: return None
            u_name = _get_underlying(u_sym)
            expiry = self.book.find_nearest_expiry_date(u_name)
            if not expiry: return None

            otype = OptionType.CE if side == OrderSide.BUY else OptionType.PE

            # UPGRADE #5: Uses the new Gamma-Efficiency logic internally
            best = self._find_best_option_contract(token, expiry, otype, strategy, regime)
            if not best: return None

            opt, ltp, greeks = best['opt'], best['ltp'], best['greeks']
            spot = self.prices.ltp(token)
            if not spot: return None

            # --- UPGRADE #2: FVG LIQUIDITY VACUUM SNIPER ---
            # Instead of blindly entering at market, check for a Fair Value Gap (FVG).
            # If an FVG exists, we calculate a specific LIMIT price to "snipe" the retest.

            df_5m = self.get_ohlc(token, 5)
            limit_underlying_price = 0.0

            if len(df_5m) >= 3:
                curr = df_5m.iloc[-1]
                prev_2 = df_5m.iloc[-3]

                if side == OrderSide.BUY:
                    # Bullish FVG: Gap exists between Candle 1 High and Candle 3 Low
                    if curr['low'] > prev_2['high']:
                        # We want to buy the retest of Candle 1 High
                        fvg_support = prev_2['high']
                        # Add 0.02% buffer to ensure fill
                        limit_underlying_price = fvg_support + (spot * 0.0002)
                        L.info(f"\U0001f3af FVG SNIPER (BUY): Detected gap. Targeting retest @ {limit_underlying_price:.2f} (Spot: {spot:.2f})")

                elif side == OrderSide.SELL:
                    # Bearish FVG: Gap exists between Candle 1 Low and Candle 3 High
                    if curr['high'] < prev_2['low']:
                        # We want to sell the retest of Candle 1 Low
                        fvg_resistance = prev_2['low']
                        # Subtract 0.02% buffer to ensure fill
                        limit_underlying_price = fvg_resistance - (spot * 0.0002)
                        L.info(f"\U0001f3af FVG SNIPER (SELL): Detected gap. Targeting retest @ {limit_underlying_price:.2f} (Spot: {spot:.2f})")

            # If FVG detected, use that limit price. Otherwise, use current Spot.
            underlying_ref_price = limit_underlying_price if limit_underlying_price > 0 else spot

            # --- Sizing & Risk Calculation ---
            delta = greeks['delta']

            # Calculate Stop Loss points on Option based on Underlying Risk points
            # Option_SL = Underlying_Points * Delta
            sl_pts_opt = risk_points_on_underlying * abs(delta)

            lot_size = self.book.lot_size(u_name)
            risk_per_lot = sl_pts_opt * lot_size

            # Final Sizing
            lots = self.risk_manager.calculate_position_size(
                token,
                risk_per_lot,
                vega_per_lot=greeks['vega'] * lot_size,
                confidence_score=confidence_score,
                strategy_name=strategy
            )

            if lots <= 0: return None

            total_trade_risk = risk_per_lot * lots
            if total_trade_risk > (self.risk_manager.dynamic_account_equity * 0.05):
                # Hard cap 5% equity risk per trade
                return None

            # Calculate Underlying SL Level
            u_sl = (underlying_ref_price - risk_points_on_underlying) if side == OrderSide.BUY else (underlying_ref_price + risk_points_on_underlying)

            # Calculate Option TP Points
            tp_pts_opt = reward_points_on_underlying * abs(delta)

            # --- Option Price Estimation ---
            option_entry_price = ltp
            if limit_underlying_price > 0:
                diff = limit_underlying_price - spot
                option_entry_price = ltp + (diff * delta)
                # Ensure positive price
                option_entry_price = max(0.05, option_entry_price)

            return {
                "opt": opt,
                "ltp_opt": option_entry_price,
                "lots": lots,
                "strategy": strategy,
                "regime": regime.name,
                "option_sl_points": sl_pts_opt,
                "option_tp_points": tp_pts_opt,
                "total_trade_risk": total_trade_risk,
                "underlying_sl": u_sl,
                "greeks": greeks,
                "max_trade_duration_minutes": 90,
                "oi_profit_target": None,
                "intended_risk_rupees": total_trade_risk,
                "underlying_price": underlying_ref_price,
                "is_fvg_entry": (limit_underlying_price > 0)
            }

    def _trade_executor_worker(self):
        while self.running.is_set():
            self.heartbeats["TradeExecutor"] = time.time()
            try:
                p = self.trade_signal_queue.get(timeout=1)
                if self.risk_manager.risk_ok(p):
                    L.info(f"Executor: Opening {p['strategy']} on {p['opt']['tradingsymbol']}")
                    if self.trader.open_position(p):
                        with self.master_lock: self.last_trade_timestamp = now_ist()
            except Empty: continue
            except Exception as e: L.error(f"Executor Error: {e}")

    def _persist_bar_data(self):
        try:
            os.makedirs(DATA_LOG_DIR, exist_ok=True)
            today = date.today().isoformat()
            for t, fname in {self.nifty_token: f"nifty_{today}.csv", self.bn_token: f"bn_{today}.csv", self.vix_token: f"vix_{today}.csv"}.items():
                if t:
                    df = self.bars.get_ohlc(t, 1)
                    if not df.empty: df.to_csv(os.path.join(DATA_LOG_DIR, fname))
                    L.info("Bar data persisted.")
        except: pass

    def _update_prometheus_metrics(self):
        if not G_PNL_REALIZED: return
        try:
            G_PNL_REALIZED.set(self.trader.daily_realized_pnl)
            with self.master_lock: G_HALTED_STATUS.set(1 if self.master_halt else 0)
            with self.risk_manager.lock:
                G_PORTFOLIO_DELTA.set(self.risk_manager.portfolio_greeks.get("net_delta", 0.0))
        except: pass

    def stop(self):
        self.running.clear()
        if hasattr(self, 'prices') and self.prices.ws: self.prices.ws.close()
        self._persist_bar_data()
    # ==================================================================================================
    # APPLICATION ENTRY POINT
    # ==================================================================================================
def login_or_reuse(token_file: str = TOKEN_FILE_PATH) -> Tuple[KiteConnect, str]:
    """
    Handles Kite Connect login by reusing a stored token
    or prompting for a new one if invalid/missing.
    """
    api_key = os.environ.get("KITE_API_KEY")
    api_secret = os.environ.get("KITE_API_SECRET")

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
            access_token = None
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

    try:
        kite_raw, access_token = login_or_reuse()
    except SystemExit as e:
        L.critical(f"Login failed: {e}")
        return

    store, kite_gov = Store(), GovernedKite(kite_raw)
    order_actor, store_actor = OrderActor(kite_gov), StoreActor(store)
    book = InstrumentBook(store_actor, order_actor).load()
    prices = PriceBus(kite_gov, access_token)
    engine = Engine(store_actor, book, prices, config) # Now has hot chamber

    risk_manager = RiskManager(engine, None, book, prices, store_actor, config)
    micro_monitor = MicrostructureMonitor(prices, book, config)
    radar = ConstituentRadar(book, prices, config, engine) # Kingmaker
    pos_manager = PositionManager(engine, None, book, prices, store_actor, risk_manager, config)

    trader = PaperTrader(engine, book, prices, store_actor, config, risk_manager.update_performance_metrics) if PAPER_TRADING else \
             Trader(engine, store, store_actor, order_actor, book, prices, config, risk_manager.update_performance_metrics)

    risk_manager.trader, pos_manager.trader, risk_manager.engine = trader, trader, engine
    engine.set_dependencies(trader, risk_manager, micro_monitor, pos_manager, radar)
    _engine_instance = engine 

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if METRICS_APP:
        start_metrics_server(port=config.get("metrics_port", 9095))

    try:
        L.info("Starting StoreActor...")
        store_actor.start()
        if not PAPER_TRADING:
            L.info("Starting OrderActor...")
            order_actor.start()
        
        engine.start() 
    
    except KeyboardInterrupt:
        L.info("Keyboard interrupt received in main. Stopping engine...")
    except SystemExit as e:
        L.warning(f"SystemExit caught in main: {e}")
    except Exception as e:
        L.critical(f"Unhandled FATAL exception in main execution: {e}", exc_info=True)
        send_alert(f"🔥 FATAL ERROR: Unhandled exception caused bot crash: {e}", "critical")
    finally:
        L.info("Shutting down... Stopping actors.")
        store_actor.stop()
        if not PAPER_TRADING:
            order_actor.stop()
        
        if _engine_instance: 
            _engine_instance.stop() 

        L.info("Waiting for actors to join...")
        store_actor.join(timeout=5.0)
        if not PAPER_TRADING:
            order_actor.join(timeout=5.0)
        L.info("Shutdown complete.")

if __name__ == "__main__":
    main()