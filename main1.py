from __future__ import annotations

import os
import sys
import time
import math
import json
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
try:
    from scipy.optimize import newton
    from scipy.stats import norm
except ImportError:
    raise RuntimeError("scipy library not found. Please install it: pip install scipy")


# ==================================================================================================
# CONFIGURATION LOADER
# ==================================================================================================
def load_config(path: str = "config.json") -> Dict:
    """Loads and validates the configuration from a JSON file."""
    L.info(f"Loading configuration from: {path}")
    try:
        with open(path, 'r') as f:
            config = json.load(f)

        timing_keys = ["market_open", "market_settling_time", "final_entry_time", "eod_flatten_time", "market_close", "final_expiry_entry_time"]
        for key in timing_keys:
            if key in config["timings"]:
                try:
                    config["timings"][key] = dtime.fromisoformat(config["timings"][key])
                except (ValueError, TypeError):
                    L.warning(f"Invalid or missing time format for '{key}'. Using default or ignoring.")

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


APP_ENV = os.environ.get("APP_ENV", "STAGING").upper()
CONFIG_FILE_PATH = "config.prod.json" if APP_ENV == "PRODUCTION" else "config.json"
CONFIG = load_config(CONFIG_FILE_PATH)

# --- Global Constants from Config ---
PAPER_TRADING = CONFIG["trading"]["paper_trading"]
if PAPER_TRADING and APP_ENV == "PRODUCTION":
    L.warning("Production environment (APP_ENV=PRODUCTION) is running in PAPER_TRADING mode.")
elif not PAPER_TRADING:
    L.critical("--- ☢️ LIVE TRADING MODE IS ACTIVE ☢️ ---")

ACCOUNT_EQUITY = float(os.environ.get("ACCOUNT_EQUITY", CONFIG["trading"]["account_equity"]))
MAX_DAILY_DRAWDOWN_PCT = CONFIG["trading"]["max_daily_drawdown_pct"]
MAX_CONCURRENT = CONFIG["trading"]["max_concurrent_trades"]
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
    # ### IMPROVEMENT ### State is now managed inside the Position object, not in config.
    # is_triggered is no longer part of the config-loaded dataclass.

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
    exit_reason: Optional[str] = None
    slm_order_id: Optional[str] = None # Will remain None for virtual SL
    tp_order_id: Optional[str] = None
    scaled_out_qty: int = 0
    breakeven_armed: bool = False
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
    # ### PILLAR 2 IMPLEMENTATION ### State for adaptive entry
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
    # ### IMPROVEMENT ### Added a sample pytest unit test as a comment.
    #
    # ### Pytest Example ###
    # def test_black_scholes_price():
    #     # Test a known deep ITM call option, should be close to intrinsic value
    #     price = black_scholes_price(S=110, K=100, T=0.1, r=0.05, sigma=0.2, is_call=True)
    #     assert price > 9.9 and price < 11.0
    #
    #     # Test a known deep OTM call option, should be close to zero
    #     price = black_scholes_price(S=90, K=100, T=0.1, r=0.05, sigma=0.2, is_call=True)
    #     assert price > 0 and price < 0.1
    #
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
    if d1 is None: return 0.0
    return S * norm.pdf(d1) * math.sqrt(T) / 100.0

def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, _ = _get_d1_d2(S, K, T, r, sigma)
    if d1 is None or S <= 0 or sigma <= 0 or T <= 0: return 0.0
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))

def bs_theta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    d1, d2 = _get_d1_d2(S, K, T, r, sigma)
    if d1 is None: return 0.0

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
    if len(price_series) < window: return None
    
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
        # ### IMPROVEMENT ### Use a smarter fallback (HV) if the solver fails.
        L.warning(f"IV calculation failed for S={S}, K={K}, P={target_price}. Falling back.")
        return hv_fallback or initial_guess

def _get_underlying(tradingsymbol: str) -> str:
    upper_symbol = tradingsymbol.upper()
    if "BANKNIFTY" in upper_symbol:
        return "BANKNIFTY"
    if "NIFTY" in upper_symbol:
        return "NIFTY"
    return "UNKNOWN"

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
            # ### IMPROVEMENT ### All migrations are now consolidated for brevity
            if db_version < 6:
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
                
                if db_version < 6: # ### PILLAR 2 ### DB migration for new fields
                    cursor.execute("ALTER TABLE positions ADD COLUMN entry_stage INTEGER DEFAULT 0")
                    cursor.execute("ALTER TABLE positions ADD COLUMN last_entry_modification TEXT")


                cursor.execute("PRAGMA user_version = 6")
                L.info("Database migrations complete. Now at version 6.")
    
    def upsert_position(self, p: Position):
        # This function is long due to the number of fields. It's a candidate for an ORM in the future.
        sql = """
            INSERT INTO positions (id, tradingsymbol, token, option_type, qty, initial_qty, entry_price, initial_sl_price, sl_price, tp_price, opened_at, strategy, market_regime_at_entry, underlying_sl_level, status, entry_order_id, slm_order_id, tp_order_id, scaled_out_qty, breakeven_armed, trailing_sl_armed, initial_risk_points, option_sl_points, option_tp_points, high_price_since_entry, scale_out_rules, exit_order_id, exit_reason, exit_price, greeks, max_trade_duration_minutes, oi_profit_target, intended_risk_rupees, is_entry_order_open, triggered_scale_out_targets, entry_stage, last_entry_modification)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                qty=excluded.qty, sl_price=excluded.sl_price, tp_price=excluded.tp_price, status=excluded.status, slm_order_id=excluded.slm_order_id, tp_order_id=excluded.tp_order_id, scaled_out_qty=excluded.scaled_out_qty, breakeven_armed=excluded.breakeven_armed, trailing_sl_armed=excluded.trailing_sl_armed, high_price_since_entry=excluded.high_price_since_entry, entry_price=excluded.entry_price, opened_at=excluded.opened_at, entry_order_id=excluded.entry_order_id, exit_order_id=excluded.exit_order_id, exit_reason=excluded.exit_reason, exit_price=excluded.exit_price, greeks=excluded.greeks, max_trade_duration_minutes=excluded.max_trade_duration_minutes, oi_profit_target=excluded.oi_profit_target, intended_risk_rupees=excluded.intended_risk_rupees, is_entry_order_open=excluded.is_entry_order_open, triggered_scale_out_targets=excluded.triggered_scale_out_targets, entry_stage=excluded.entry_stage, last_entry_modification=excluded.last_entry_modification
        """
        rules_json = json.dumps(p.scale_out_rules)
        greeks_json = json.dumps(p.greeks)
        triggered_json = json.dumps(p.triggered_scale_out_targets)
        last_mod_iso = p.last_entry_modification.isoformat() if p.last_entry_modification else None

        params = (p.id, p.tradingsymbol, p.token, p.option_type, p.qty, p.initial_qty, p.entry_price, p.initial_sl_price, p.sl_price, p.tp_price, p.opened_at.isoformat(), p.strategy, p.market_regime_at_entry, p.underlying_sl_level, p.status, p.entry_order_id, p.slm_order_id, p.tp_order_id, p.scaled_out_qty, int(p.breakeven_armed), int(p.trailing_sl_armed), p.initial_risk_points, p.option_sl_points, p.option_tp_points, p.high_price_since_entry, rules_json, p.exit_order_id, p.exit_reason, p.exit_price, greeks_json, p.max_trade_duration_minutes, p.oi_profit_target, p.intended_risk_rupees, int(p.is_entry_order_open), triggered_json, p.entry_stage, last_mod_iso)
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
            r_dict['breakeven_armed'] = bool(r_dict.get("breakeven_armed"))
            r_dict['trailing_sl_armed'] = bool(r_dict.get("trailing_sl_armed"))
            r_dict['is_entry_order_open'] = bool(r_dict.get("is_entry_order_open"))
            r_dict['opened_at'] = datetime.fromisoformat(r_dict["opened_at"])
            if r_dict.get("last_entry_modification"):
                r_dict['last_entry_modification'] = datetime.fromisoformat(r_dict["last_entry_modification"])
            
            # Remove keys that are not in the Position dataclass constructor
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
    def __init__(self, kite: GovernedKite, store: Store):
        self.kite = kite
        self.store = store
        self.df = None
        self.path = os.path.join(PERSIST_DIR, "instruments_nfo.csv")
        self.df_by_token = None
        self.df_by_symbol = None
        # ### IMPROVEMENT ### Cache for special tokens
        self.special_tokens: Dict[str, int] = {}

    def load(self):
        try:
            if not os.path.exists(self.path) or self.store.get_kv("instruments_refreshed") != str(date.today()):
                self.refresh()
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
        L.info("Refreshing NFO instrument list from broker...")
        try:
            instruments = self.kite.instruments("NFO")
            if instruments is None:
                raise KiteException("Failed to fetch NFO instruments from API after multiple retries.")
            df = pd.DataFrame(instruments)
            if "expiry" in df.columns:
                df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
            df.to_csv(self.path, index=False)
            self.df = df
            self.store.set_kv("instruments_refreshed", str(date.today()))
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
        """### IMPROVEMENT ### Dynamically find tokens for indices instead of hardcoding."""
        L.info("Loading special instrument tokens...")
        # Note: KiteConnect `instruments` call for "NFO" does not contain index futures like NIFTY 50.
        # This requires a full instrument dump from another source or a different API call if available.
        # For now, we will use a robust lookup on the NFO dataframe for futures.
        nifty_fut = self.find_current_futures_contract("NIFTY")
        bn_fut = self.find_current_futures_contract("BANKNIFTY")
        if nifty_fut: self.special_tokens["NIFTY"] = nifty_fut['instrument_token']
        if bn_fut: self.special_tokens["BANKNIFTY"] = bn_fut['instrument_token']
        
        # VIX is not in NFO dump, so it remains a "magic number" unless a different instrument source is used.
        # A full `instruments.csv` from Kite's website would be a better source.
        self.special_tokens["INDIA VIX"] = 257281 # This remains the only hardcoded value
        L.info(f"Special tokens loaded: {self.special_tokens}")


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
        self.lock = threading.Lock()
        self.tokens = set()
        self.last = {}
        self.full_ticks = {}
        self.on_order_update_callbacks = []
        self.on_connect_callbacks = []
        self.connected = threading.Event()
        self.ws_thread = None
        self.tick_queue = Queue()
        self.last_tick_reception_time: Optional[datetime] = None
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
        self.tick_queue.put(ticks)

    def _on_order_update(self, ws, order):
        L.info(f"WS Order Update: {order.get('tradingsymbol')} {order.get('status')} ({order.get('order_id')}) | Msg: {order.get('status_message')}")
        for cb in self.on_order_update_callbacks:
            try:
                cb(order)
            except Exception as e:
                L.error(f"Error in on_order_update callback: {e}", exc_info=True)

    def _on_connect(self, ws, response):
        L.info("PriceBus WebSocket connected.")
        if G_WS_CONNECTED: G_WS_CONNECTED.set(1)
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
        if G_WS_CONNECTED: G_WS_CONNECTED.set(0)
        self.connected.clear()
        L.warning(f"PriceBus WS closed: {code}-{reason}. Will attempt to reconnect automatically.")

    def _on_error(self, ws, code, reason):
        if G_WS_CONNECTED: G_WS_CONNECTED.set(0)
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
        return self.last.get(token)

    def get_full_tick(self, token: int) -> Optional[Dict]:
        return self.full_ticks.get(token)

class BarStore:
    def __init__(self, timeframes: List[int]):
        self.lock = threading.Lock()
        self.timeframes = sorted(timeframes)
        self.data: Dict[int, Dict[int, pd.DataFrame]] = {}

    def _ensure_token_data(self, token: int):
        if token not in self.data:
            self.data[token] = {tf: pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).astype(float) for tf in self.timeframes}
            for tf in self.timeframes:
                self.data[token][tf].index.name = "timestamp"

    def prime(self, token: int, hist_df: pd.DataFrame):
        with self.lock:
            if hist_df.empty:
                return
            L.info(f"Priming BarStore for {token} with {len(hist_df)} bars.")
            self._ensure_token_data(token)
            ts_col = pd.to_datetime(hist_df['date'])
            if ts_col.dt.tz is None:
                hist_df['timestamp'] = ts_col.dt.tz_localize(IST)
            else:
                hist_df['timestamp'] = ts_col.dt.tz_convert(IST)
            hist_df = hist_df.set_index("timestamp").drop(columns=['date'])
            base_df = hist_df.copy()
            for tf in self.timeframes:
                if tf == 1:
                    self.data[token][tf] = base_df
                else:
                    resampled_df = base_df.resample(f'{tf}min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
                    self.data[token][tf] = resampled_df

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
    def __init__(self, engine: 'Engine', nifty_token: int, bn_token: int, vix_token: Optional[int]):
        self.engine = engine
        self.prices = engine.prices
        self.nifty_token = nifty_token
        self.bn_token = bn_token
        self.vix_token = vix_token
        self.params = CONFIG["strategies"]["regime_classifier"]
        self.tf = self.params["resample_minutes"]
        self.potential_regime: Optional[Tuple[Regime, int]] = None
        self.confirmation_count: int = 0
        self.confirmation_threshold: int = self.params.get("hysteresis_confirmation_count", 3)

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < self.params["bb_period"]: return df
        df.ta.adx(length=self.params["adx_period"], append=True)
        bbands = df.ta.bbands(length=self.params["bb_period"], append=True)
        df['bbw'] = (bbands[f'BBU_{self.params["bb_period"]}_2.0'] - bbands[f'BBL_{self.params["bb_period"]}_2.0']) / bbands[f'BBM_{self.params["bb_period"]}_2.0']
        lookback = self.params.get("dynamic_regime_lookback", 200)
        df['bbw_pct_rank'] = df['bbw'].rolling(lookback).rank(pct=True) * 100
        df['adx_pct_rank'] = df[f'ADX_{self.params["adx_period"]}'].rolling(lookback).rank(pct=True) * 100
        df['ema_fast'] = df.ta.ema(length=20)
        df['ema_slow'] = df.ta.ema(length=50)
        return df

    def _get_scores(self, df: pd.DataFrame, current_regime_enum: Regime) -> Dict[str, int]:
        scores = {"trend_up": 0, "trend_down": 0, "chop": 0, "compression": 0}
        if len(df) < 60: return scores
        dmp_col, dmn_col = f'DMP_{self.params["adx_period"]}', f'DMN_{self.params["adx_period"]}'

        compression_threshold = self.params.get("compression_rank_threshold_pct", 10.0)
        adx_entry_threshold_pct = self.params.get("adx_trend_entry_percentile", 75.0)
        adx_exit_threshold_pct = self.params.get("adx_trend_exit_percentile", 60.0)

        if df['bbw_pct_rank'].iloc[-1] < compression_threshold:
            scores["compression"] += 2

        is_trending = current_regime_enum in [Regime.TRENDING_UP, Regime.TRENDING_DOWN]
        adx_threshold = adx_exit_threshold_pct if is_trending else adx_entry_threshold_pct
        adx_pct_rank_val = df['adx_pct_rank'].iloc[-1]

        if adx_pct_rank_val < adx_exit_threshold_pct:
            scores["chop"] += 1

        if adx_pct_rank_val > adx_threshold:
            if df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1] and df[dmp_col].iloc[-1] > df[dmn_col].iloc[-1]:
                scores["trend_up"] += 2
            elif df['ema_fast'].iloc[-1] < df['ema_slow'].iloc[-1] and df[dmn_col].iloc[-1] > df[dmp_col].iloc[-1]:
                scores["trend_down"] += 2
        
        # Market breadth logic is placeholder as NIFTY50 tokens are not fully loaded
        # breadth = self.engine.market_breadth["ratio"]
        # if scores["trend_up"] > 0 and breadth > 0.7: scores["trend_up"] += 1
        # if scores["trend_down"] > 0 and breadth < 0.3: scores["trend_down"] += 1
        # if 0.4 < breadth < 0.6: scores["chop"] += 1

        return scores

    def get_raw_classification(self, current_regime_enum: Regime) -> Tuple[Regime, Optional[int]]:
        try:
            df_bn = self.engine.get_ohlc(self.bn_token, self.tf)
            df_n = self.engine.get_ohlc(self.nifty_token, self.tf)
            if df_bn.empty or df_n.empty or len(df_bn) < 60 or len(df_n) < 60: return (Regime.UNCLEAR, None)
            df_n, df_bn = self._add_indicators(df_n), self._add_indicators(df_bn)
            score_n, score_bn = self._get_scores(df_n, current_regime_enum), self._get_scores(df_bn, current_regime_enum)

            vix_ltp = self.prices.ltp(self.vix_token)
            if vix_ltp and vix_ltp > self.params.get("vix_chaos_threshold", 24.0):
                return (Regime.CHAOS, self.bn_token)

            if score_n["trend_up"] >= 2 and score_bn["trend_up"] >= 2:
                active_token = self.bn_token if df_bn['close'].pct_change(10).iloc[-1] > df_n['close'].pct_change(10).iloc[-1] else self.nifty_token
                return (Regime.TRENDING_UP, active_token)

            if score_n["trend_down"] >= 2 and score_bn["trend_down"] >= 2:
                active_token = self.bn_token if df_bn['close'].pct_change(10).iloc[-1] < df_n['close'].pct_change(10).iloc[-1] else self.nifty_token
                return (Regime.TRENDING_DOWN, active_token)

            if score_n["compression"] >= 2 and score_bn["compression"] >= 2:
                active_token = self.bn_token if df_bn['bbw_pct_rank'].iloc[-1] < df_n['bbw_pct_rank'].iloc[-1] else self.nifty_token
                return (Regime.COMPRESSION, active_token)

            corr_series = df_n['close'].pct_change().rolling(self.params["correlation_period"]).corr(df_bn['close'].pct_change())
            if not corr_series.empty and not pd.isna(corr_series.iloc[-1]):
                if corr_series.iloc[-1] < self.params["correlation_threshold"]:
                    return (Regime.CHOP, self.bn_token)

            if score_n["chop"] >= 2 and score_bn["chop"] >= 2:
                return (Regime.CHOP, self.bn_token)

            return (Regime.UNCLEAR, None)

        except Exception as e:
            L.warning(f"Regime classification failed: {e}", exc_info=True)
            return (Regime.UNCLEAR, None)
# ==================================================================================================
# TRADING EXECUTION LAYER
# ==================================================================================================
class AbstractTrader(ABC):
    def __init__(self,
                 engine: 'Engine',
                 book: InstrumentBook,
                 prices: PriceBus,
                 store: Store,
                 perf_callback: Optional[Callable[[float], None]] = None):
        self.engine = engine
        self.book = book
        self.prices = prices
        self.store = store
        self._update_performance = perf_callback or (lambda pnl: None)
        self.lock = threading.Lock()
        self.positions: Dict[str, Position] = {}
        self.daily_realized_pnl: float = 0.0

    @abstractmethod
    def open_position(self, trade_params: Dict) -> Optional[Position]: pass
    @abstractmethod
    def close_position(self, p: Position, reason: str) -> bool: pass
    @abstractmethod
    def modify_sl(self, p: Position, new_trigger: float): pass
    @abstractmethod
    def scale_out(self, p: Position, qty_to_close: int) -> bool: pass
    @abstractmethod
    def place_bracket_orders(self, p: Position) -> bool: pass
    @abstractmethod
    def cancel_pending_entry(self, p: Position) -> bool: pass
    @abstractmethod
    def execute_simulated_sl(self, p: Position) -> bool: pass

    def unrealized_pnl(self) -> float:
        pnl = 0.0
        with self.lock:
            for p in self.positions.values():
                if p.status not in [PositionStatus.CLOSED.value, PositionStatus.PENDING_ENTRY.value, PositionStatus.PENDING_SUBMISSION.value, PositionStatus.REJECTED.value]:
                    ltp = self.prices.ltp(p.token)
                    if ltp is None:
                        ltp = p.entry_price
                    pnl += (ltp - p.entry_price) * p.qty
        return pnl

class PaperTrader(AbstractTrader):
    def open_position(self, trade_params: Dict) -> Optional[Position]:
        with self.lock:
            try:
                opt = trade_params['opt']
                initial_lots = int(trade_params['lots'])
                lot_size = self.book.lot_size(_get_underlying(opt['tradingsymbol']))
                if not lot_size:
                    L.error(f"Could not determine lot size for {opt['tradingsymbol']}. Aborting.")
                    return None

                qty = int(initial_lots * lot_size)

                simulated_entry_price = float(
                    trade_params['ltp_opt'] + (self.book.tick_size(opt['tradingsymbol']) * CONFIG['trading'].get('paper_trade_slippage_ticks', 1))
                )
                sl_points = float(trade_params['option_sl_points'])
                tp_points = float(trade_params['option_tp_points'])
                underlying_sl = float(trade_params['underlying_sl']) if trade_params['underlying_sl'] is not None else None

                open_positions = [p for p in self.positions.values() if p.status != PositionStatus.CLOSED.value]
                if len(open_positions) >= MAX_CONCURRENT:
                    L.warning(f"Trade rejected: max concurrent trades ({MAX_CONCURRENT}) reached.")
                    return None

                if qty <= 0:
                    L.warning(f"Trade rejected: calculated quantity is {qty}.")
                    return None
                
                # Dynamic correlation logic removed for brevity, assuming MAX_CONCURRENT=1 for simplicity.

                pos_id = f"PAPER_{uuid.uuid4()}"

                pos = Position(
                    id=pos_id,
                    tradingsymbol=opt['tradingsymbol'],
                    token=int(opt['instrument_token']),
                    option_type=opt['instrument_type'],
                    qty=qty,
                    initial_qty=qty,
                    entry_price=simulated_entry_price,
                    initial_sl_price=simulated_entry_price - sl_points,
                    sl_price=simulated_entry_price - sl_points,
                    tp_price=simulated_entry_price + tp_points,
                    opened_at=now_ist(),
                    strategy=trade_params['strategy'],
                    underlying_sl_level=underlying_sl,
                    market_regime_at_entry=trade_params['regime'],
                    initial_risk_points=sl_points,
                    status=PositionStatus.ACTIVE.value,
                    high_price_since_entry=simulated_entry_price,
                    scale_out_rules=CONFIG['trading']['scale_out_rules'],
                    option_sl_points=sl_points,
                    option_tp_points=tp_points,
                    entry_order_id=pos_id,
                    max_trade_duration_minutes=trade_params.get('max_trade_duration_minutes', 90),
                    oi_profit_target=trade_params.get('oi_profit_target'),
                    intended_risk_rupees=trade_params.get('total_trade_risk', 0.0)
                )

                self.positions[pos.id] = pos
                self.store.upsert_position(pos)
                self.prices.subscribe([pos.token])
                send_alert(f"✅ [PAPER] {pos.strategy} OPENED {pos.tradingsymbol} Qty={qty} @ {pos.entry_price:.2f}, SL={pos.sl_price:.2f}, TP={pos.tp_price:.2f}")

                return pos

            except Exception as e:
                L.critical(f"FATAL EXCEPTION in PaperTrader.open_position: {e}", exc_info=True)
                return None

    def close_position(self, p: Position, reason: str) -> bool:
        with self.lock:
            if p.status == PositionStatus.CLOSED.value: return True

            tick_size = self.book.tick_size(p.tradingsymbol)
            slippage = tick_size * CONFIG['trading'].get('paper_trade_slippage_ticks', 1)

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
            self.store.upsert_position(p)
            self.store.log_closed_trade(p, exit_price, reason)

            self._update_performance(pnl)

            self.positions.pop(p.id, None)
            send_alert(f"❌ [PAPER] CLOSED {p.tradingsymbol} @ {exit_price:.2f} ({reason}). Final PnL: {pnl:.2f}. Daily PnL: {self.daily_realized_pnl:.2f}")
            return True

    def cancel_pending_entry(self, p: Position) -> bool:
        """Paper trading version of cancelling a pending entry."""
        with self.lock:
            if p.status != PositionStatus.PENDING_ENTRY.value: return False
            p.status = PositionStatus.CLOSED.value
            p.exit_reason = "ENTRY_CANCELLED_PAPER"
            self.store.upsert_position(p)
            self.positions.pop(p.id, None)
            L.info(f"[PAPER] Cancelled pending entry for {p.tradingsymbol}")
            return True
            
    def modify_sl(self, p: Position, new_trigger: float):
        with self.lock:
            if new_trigger > p.sl_price:
                L.info(f"[PAPER] Trailing SL for {p.tradingsymbol} from {p.sl_price:.2f} to {new_trigger:.2f}")
                p.sl_price = new_trigger
                self.store.upsert_position(p)

    def scale_out(self, p: Position, qty_to_close: int) -> bool:
        with self.lock:
            if p.status not in [PositionStatus.ACTIVE.value, PositionStatus.PARTIALLY_CLOSED.value] or qty_to_close <= 0:
                return False
            L.info(f"[PAPER] Scaling out {qty_to_close} of {p.tradingsymbol}")
            p.qty -= qty_to_close
            p.scaled_out_qty += qty_to_close
            p.status = PositionStatus.PARTIALLY_CLOSED.value if p.qty > 0 else PositionStatus.CLOSED.value
            self.store.upsert_position(p)
            if p.status == PositionStatus.CLOSED.value:
                self.close_position(p, "SCALE_OUT_FULL")
            return True

    def place_bracket_orders(self, p: Position) -> bool:
        p.status = PositionStatus.ACTIVE.value
        self.store.upsert_position(p)
        return True

    def execute_simulated_sl(self, p: Position) -> bool:
        """Paper trading version of the simulated SL exit."""
        L.warning(f"[PAPER] VIRTUAL SL HIT for {p.tradingsymbol}. Closing position.")
        return self.close_position(p, "SL_HIT_PAPER")
class Trader(AbstractTrader):
    def __init__(self,
                 engine: 'Engine',
                 kite: GovernedKite,
                 store: Store,
                 book: InstrumentBook,
                 prices: PriceBus,
                 perf_callback: Optional[Callable[[float], None]] = None):
        super().__init__(engine, book, prices, store, perf_callback)
        self.k = kite
        self.positions = store.load_open_positions()

        if self.positions:
            L.info("Reconciling open orders for existing positions on startup...")
            try:
                open_orders = self.k.orders()
                if open_orders is not None:
                    open_order_ids = {str(o['order_id']) for o in open_orders if o['status'] == 'OPEN'}
                    for p in list(self.positions.values()):
                        if p.tp_order_id and str(p.tp_order_id) not in open_order_ids: 
                            L.warning(f"TP order {p.tp_order_id} for {p.tradingsymbol} not found open. Clearing."); 
                            p.tp_order_id = None
                        self.store.upsert_position(p)
            except Exception as e: L.error(f"Failed to reconcile open orders on startup: {e}")

        for p in self.positions.values(): self.prices.subscribe([p.token])

    def _get_order_avg_price(self, oid: str) -> Optional[float]:
        history = self.k.order_history(oid);
        if not history: return None
        trades = [t for t in history if t.get('status') == 'COMPLETE'];
        if not trades: return None
        qty = sum(t['filled_quantity'] for t in trades); return (sum(t['price'] * t['filled_quantity'] for t in trades) / qty) if qty > 0 else None

    def _cancel_all_open_orders_for_pos(self, p: Position, cancel_entry: bool = False):
        """Cancels all associated open orders for a position."""
        orders_to_cancel = [p.tp_order_id]
        if cancel_entry:
            orders_to_cancel.append(p.entry_order_id)

        for oid in orders_to_cancel:
            if oid:
                if self.k.cancel_order(self.k.VARIETY_REGULAR, str(oid)) is None:
                    L.warning(f"Could not cancel order {oid} after multiple retries.")
        
        p.tp_order_id = None
        if cancel_entry:
            p.entry_order_id = None


    def open_position(self, trade_params: Dict) -> Optional[Position]:
        with self.lock:
            open_pos_count = sum(1 for p in self.positions.values() if p.status not in [PositionStatus.CLOSED.value, PositionStatus.REJECTED.value])
            opt, ts = trade_params['opt'], trade_params['opt']['tradingsymbol']
            lot_size = self.book.lot_size(_get_underlying(ts))

            if open_pos_count >= MAX_CONCURRENT or not lot_size:
                L.warning(f"Trade rejected: Max concurrent ({MAX_CONCURRENT}) or no lot size.")
                return None

            lots = int(trade_params['lots'])
            qty = int(lots * lot_size)
            if qty <= 0: return None

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
                scale_out_rules=CONFIG['trading']['scale_out_rules'],
                max_trade_duration_minutes=trade_params.get('max_trade_duration_minutes', 90),
                oi_profit_target=trade_params.get('oi_profit_target'),
                intended_risk_rupees=trade_params.get('total_trade_risk', 0.0),
                last_entry_modification=now_ist() # ### PILLAR 2 ### Initialize mod time
            )
            self.store.upsert_position(pos)
            self.positions[pos.id] = pos
            
            # ### PILLAR 2 IMPLEMENTATION: ADAPTIVE ENTRY ###
            # Instead of placing an SL order, we start with a passive LIMIT order.
            # The management loop will make it more aggressive over time.
            full_tick = self.prices.get_full_tick(int(opt['instrument_token']))
            if not full_tick or not full_tick.get('depth'):
                L.warning(f"Cannot get depth for {ts}, cannot place adaptive entry.")
                pos.status = PositionStatus.REJECTED.value; pos.exit_reason = "NO_DEPTH_FOR_ENTRY"
                self.store.upsert_position(pos); return None

            bid_price = full_tick['depth']['buy'][0]['price']
            
            oid = self.k.place_order(
                variety=self.k.VARIETY_REGULAR, exchange="NFO", tradingsymbol=ts,
                transaction_type=self.k.TRANSACTION_TYPE_BUY, quantity=qty, product=self.k.PRODUCT_MIS,
                order_type=self.k.ORDER_TYPE_LIMIT, price=bid_price
            )

            if oid is None:
                L.error(f"Adaptive Entry Stage 1 order placement failed for {ts} after multiple retries.")
                pos.status = PositionStatus.REJECTED.value; pos.exit_reason = "BROKER_API_FAILURE"
                self.store.upsert_position(pos); return None

            L.info(f"Placed Adaptive Entry Stage 1 (Passive) order {oid} for {ts} @ {bid_price}")

            self.positions.pop(temp_id, None)
            pos.id = f"LIVE_{oid}"
            pos.entry_order_id = str(oid)
            pos.status = PositionStatus.PENDING_ENTRY.value
            pos.entry_stage = 1
            self.store.upsert_position(pos)
            self.positions[pos.id] = pos
            return pos

    def place_bracket_orders(self, p: Position) -> bool:
        if p.status != PositionStatus.OPEN_AWAITING_BRACKETS.value: return False
        tick_size = self.book.tick_size(p.tradingsymbol)
        tp_id = None
        try:
            # Place TP order if one doesn't already exist from a previous partial fill
            if p.tp_price > 0 and not p.tp_order_id:
                tp_limit = round(p.tp_price / tick_size) * tick_size
                tp_id = self.k.place_order(self.k.VARIETY_REGULAR, "NFO", p.tradingsymbol, self.k.TRANSACTION_TYPE_SELL, p.qty, self.k.PRODUCT_MIS, self.k.ORDER_TYPE_LIMIT, price=tp_limit)
                if tp_id is None: raise ValueError("TP order placement failed")
                p.tp_order_id = str(tp_id)
                L.info(f"Placed TP order {tp_id} @ {tp_limit} for {p.qty} of {p.tradingsymbol}")
            
            # Modify existing TP order quantity if it exists
            elif p.tp_order_id:
                if self.k.modify_order(self.k.VARIETY_REGULAR, p.tp_order_id, quantity=p.qty) is None:
                    raise ValueError(f"Failed to modify TP order {p.tp_order_id} quantity to {p.qty}")
                L.info(f"Modified TP order {p.tp_order_id} quantity to {p.qty} for {p.tradingsymbol}")


            p.status = PositionStatus.ACTIVE.value
            self.store.upsert_position(p)
            L.info(f"Position {p.tradingsymbol} is now ACTIVE with virtual SL @ {p.sl_price:.2f}.")
            return True
        except Exception as e:
            L.error(f"Failed to place/modify TP order for {p.tradingsymbol}: {e}")
            if tp_id: self.k.cancel_order(self.k.VARIETY_REGULAR, str(tp_id))
            return False

    def close_position(self, p: Position, reason: str) -> bool:
        with self.lock:
            if p.status in [PositionStatus.PENDING_CLOSURE.value, PositionStatus.CLOSED.value, PositionStatus.PENDING_SL_EXIT.value]:
                return True

            p.status = PositionStatus.PENDING_CLOSURE.value
            p.exit_reason = reason
            self.store.upsert_position(p)
            
            # Cancel any open entry or TP orders before placing the market exit
            self._cancel_all_open_orders_for_pos(p, cancel_entry=p.is_entry_order_open)

            if p.qty <= 0:
                L.warning(f"Attempted to close position {p.tradingsymbol} with zero quantity. Marking as closed.")
                p.status = PositionStatus.CLOSED.value
                self.store.upsert_position(p)
                return True

            oid = self.k.place_order(self.k.VARIETY_REGULAR, "NFO", p.tradingsymbol, self.k.TRANSACTION_TYPE_SELL, p.qty, self.k.PRODUCT_MIS, self.k.ORDER_TYPE_MARKET)
            if oid is None:
                L.critical(f"MARKET EXIT ORDER FAILED for {p.tradingsymbol}. Manual intervention required!");
                send_alert(f"🔥 CRITICAL: FAILED TO PLACE MARKET EXIT for {p.tradingsymbol}. POSITION IS STILL OPEN.", "critical");
                p.status = PositionStatus.ACTIVE.value # Revert status
                self.store.upsert_position(p)
                return False

            L.info(f"Market exit order {oid} placed for {p.tradingsymbol}. Reason: {reason}.");
            p.exit_order_id = str(oid);
            self.store.upsert_position(p);
            return True

    def cancel_pending_entry(self, p: Position) -> bool:
        """Cancels an entry order that has not been filled."""
        with self.lock:
            if p.status != PositionStatus.PENDING_ENTRY.value or not p.entry_order_id:
                return False
            
            L.warning(f"Cancelling pending entry order {p.entry_order_id} for {p.tradingsymbol}.")
            self._cancel_all_open_orders_for_pos(p, cancel_entry=True)
            p.status = PositionStatus.CLOSED.value
            p.exit_reason = "ENTRY_TIMEOUT_CANCELLED"
            self.store.upsert_position(p)
            self.positions.pop(p.id, None)
            return True

    def scale_out(self, p: Position, qty_to_close: int) -> bool:
        with self.lock:
            if p.status not in [PositionStatus.ACTIVE.value, PositionStatus.PARTIALLY_CLOSED.value] or qty_to_close <= 0 or qty_to_close > p.qty:
                return False

            L.info(f"Scaling out {qty_to_close} of {p.tradingsymbol}. Temporarily cancelling TP order.")
            # Only cancel the TP order, we will re-place/modify it after the scale-out fill
            self._cancel_all_open_orders_for_pos(p, cancel_entry=False)
            
            p.status = PositionStatus.OPEN_AWAITING_BRACKETS
            self.store.upsert_position(p)

            oid = self.k.place_order(
                self.k.VARIETY_REGULAR, "NFO", p.tradingsymbol, self.k.TRANSACTION_TYPE_SELL,
                qty_to_close, self.k.PRODUCT_MIS, self.k.ORDER_TYPE_MARKET
            )

            if oid is None:
                send_alert(f"CRITICAL: SCALE OUT MARKET ORDER FAILED for {p.tradingsymbol}. Closing full position!", "critical")
                self.close_position(p, "SCALE_OUT_FAILURE")
                return False
            
            # The on_order_update handler will confirm the fill, update quantities,
            # and then the `manage_positions` loop will call `place_bracket_orders`.
            return True

    def modify_sl(self, p: Position, new_trigger: float):
        """Modifies the virtual SL price in the database."""
        tick_size = self.book.tick_size(p.tradingsymbol)
        new_trigger_rounded = round(new_trigger / tick_size) * tick_size
        if new_trigger_rounded > p.sl_price and abs(new_trigger_rounded - p.sl_price) >= tick_size:
            old_sl = p.sl_price
            p.sl_price = new_trigger_rounded
            self.store.upsert_position(p)
            L.info(f"Trailing VIRTUAL SL for {p.tradingsymbol} from {old_sl:.2f} to {new_trigger_rounded:.2f}")

    def execute_simulated_sl(self, p: Position) -> bool:
        """Places the initial SL-L order and sets the state for the engine to monitor."""
        with self.lock:
            if p.status not in [PositionStatus.ACTIVE.value, PositionStatus.PARTIALLY_CLOSED.value]:
                return False

            self._cancel_all_open_orders_for_pos(p)
            
            p.status = PositionStatus.PENDING_SL_EXIT.value
            p.exit_reason = "SIMULATED_SL_HIT_INIT"
            
            tick_size = self.book.tick_size(p.tradingsymbol)
            limit_price = round((p.sl_price - (tick_size * 2)) / tick_size) * tick_size

            oid = self.k.place_order(
                variety=self.k.VARIETY_REGULAR, exchange="NFO", tradingsymbol=p.tradingsymbol,
                transaction_type=self.k.TRANSACTION_TYPE_SELL, quantity=p.qty,
                product=self.k.PRODUCT_MIS, order_type=self.k.ORDER_TYPE_LIMIT, price=limit_price
            )

            if oid is None:
                L.critical(f"Initial SIMULATED SL-L order FAILED for {p.tradingsymbol}. Falling back to MARKET exit.", "critical")
                return self.close_position(p, "SL_L_FAIL_MK_EXIT")

            L.info(f"Simulated SL-L exit order {oid} placed for {p.tradingsymbol} @ {limit_price:.2f}. Monitoring for fill...")
            p.exit_order_id = str(oid)
            self.store.upsert_position(p)
            return True
# ==================================================================================================
# STRATEGY DEFINITIONS
# ==================================================================================================
class BaseStrategy(ABC):
    def __init__(self, name: StrategyName, engine: 'Engine', params: Dict):
        self.name = name
        self.engine = engine
        self.params = params

    @abstractmethod
    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]: pass

    @abstractmethod
    def get_risk_params(self, token: int, side: OrderSide, current_time: datetime) -> Tuple[float, float]: pass

    def evaluate(self, token: int, regime: Regime, current_time: datetime) -> Optional[TradeSignal]:
        if side := self.check_signal(token, regime, current_time):
            risk_points, reward_points = self.get_risk_params(token, side, current_time)
            if risk_points > 0 and reward_points > 0:
                return TradeSignal(self.name, side, risk_points, reward_points)
        return None

class MomentumBreakoutStrategy(BaseStrategy):
    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]:
        df = self.engine.get_ohlc(token, self.params["resample_minutes"])
        if len(df) < self.params["squeeze_period"]: return None
        
        # ### PILLAR 1.3 IMPLEMENTATION: Check for Volatility Contraction ###
        # This check is now done in the `_find_best_option_contract` function which is called before this.
        # We will assume that if this strategy is called, the vol contraction is already confirmed.
        
        df.ta.bbands(length=self.params["bb_period"], append=True)
        df.ta.adx(length=14, append=True)
        df['bbw'] = (df[f'BBU_{self.params["bb_period"]}_2.0'] - df[f'BBL_{self.params["bb_period"]}_2.0']) / df[f'BBM_{self.params["bb_period"]}_2.0']
        df['vol_ma'] = df['volume'].rolling(self.params["bb_period"]).mean()
        last = df.iloc[-2]
        is_in_squeeze = last['bbw'] < df['bbw'].rolling(self.params["squeeze_period"]).mean().iloc[-2] * self.params["squeeze_factor"]
        adx_was_low = (df['ADX_14'].iloc[-10:-2] < 20).any()
        adx_is_rising = df['ADX_14'].iloc[-1] > df['ADX_14'].iloc[-2]
        if is_in_squeeze and adx_was_low and adx_is_rising and last['volume'] > self.params["volume_factor"] * last['vol_ma']:
            if df.iloc[-1]['close'] > last[f'BBU_{self.params["bb_period"]}_2.0']: return OrderSide.BUY
            if df.iloc[-1]['close'] < last[f'BBL_{self.params["bb_period"]}_2.0']: return OrderSide.SELL
        return None

    def get_risk_params(self, token: int, side: OrderSide, current_time: datetime) -> Tuple[float, float]:
        df_1m = self.engine.get_ohlc(token, 1)
        if len(df_1m) < 50: return 0.0, 0.0
        atr_short = df_1m.ta.atr(length=5).iloc[-1]
        atr_long = df_1m.ta.atr(length=50).iloc[-1]
        if atr_long == 0 or pd.isna(atr_short) or pd.isna(atr_long): return 0.0, 0.0
        vol_ratio = atr_short / atr_long
        base_sl_multiplier = self.params["atr_sl_multiplier"]; base_tp_multiplier = self.params["atr_tp_multiplier"]
        if vol_ratio > 1.5:
            final_sl_multiplier = base_sl_multiplier * 1.25
            final_tp_multiplier = base_tp_multiplier * 0.75
        elif vol_ratio < 0.7:
            final_sl_multiplier = base_sl_multiplier * 0.80
            final_tp_multiplier = base_tp_multiplier * 1.20
        else:
            final_sl_multiplier = base_sl_multiplier
            final_tp_multiplier = base_tp_multiplier
        current_atr = df_1m.ta.atr(14).iloc[-1]
        if pd.isna(current_atr): return 0.0, 0.0
        risk_points = current_atr * final_sl_multiplier
        reward_points = current_atr * final_tp_multiplier
        return risk_points, reward_points

class TrendPullbackStrategy(BaseStrategy):
    def check_signal(self, token: int, regime: Regime, current_time: datetime) -> Optional[OrderSide]:
        side = OrderSide.BUY if regime == Regime.TRENDING_UP else OrderSide.SELL
        df_primary = self.engine.get_ohlc(token, self.params["primary_tf"])
        df_confirm = self.engine.get_ohlc(token, self.params["confirm_tf"])
        if len(df_primary) < self.params["ema_period"] + 2 or len(df_confirm) < 21: return None
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
        if len(df_1m) < 50: return 0.0, 0.0
        atr_short = df_1m.ta.atr(length=5).iloc[-1]; atr_long = df_1m.ta.atr(length=50).iloc[-1]
        if atr_long == 0 or pd.isna(atr_short) or pd.isna(atr_long): return 0.0, 0.0
        vol_ratio = atr_short / atr_long
        base_sl_multiplier = self.params["atr_sl_multiplier"]; base_tp_multiplier = self.params["atr_tp_multiplier"]
        if vol_ratio > 1.5:
            final_sl_multiplier = base_sl_multiplier * 1.25
            final_tp_multiplier = base_tp_multiplier * 0.75
        elif vol_ratio < 0.7:
            final_sl_multiplier = base_sl_multiplier * 0.80
            final_tp_multiplier = base_tp_multiplier * 1.20
        else:
            final_sl_multiplier = base_sl_multiplier
            final_tp_multiplier = base_tp_multiplier
        current_atr = df_1m.ta.atr(14).iloc[-1]
        if pd.isna(current_atr): return 0.0, 0.0
        risk_points = current_atr * final_sl_multiplier
        reward_points = current_atr * final_tp_multiplier
        return risk_points, reward_points

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
        if df_1m.empty: return 0.0, 0.0
        atr = df_1m.ta.atr(length=14).iloc[-1]
        if pd.isna(atr): return 0.0, 0.0
        risk = atr * self.params["atr_sl_multiplier"]

        df_resampled = self.engine.get_ohlc(token, self.params["resample_minutes"])
        if df_resampled.empty: return 0.0, 0.0
        df_resampled.ta.bbands(length=self.params["bb_period"], append=True)
        if df_resampled.empty: return 0.0, 0.0

        middle_band = df_resampled[f'BBM_{self.params["bb_period"]}_2.0'].iloc[-1]
        current_price = df_resampled['close'].iloc[-1]

        reward = abs(current_price - middle_band)
        return (risk, reward) if reward > 0 else (risk, risk * 1.5)
# ==================================================================================================
# MAIN TRADING ENGINE
# ==================================================================================================
class Engine:
    def __init__(self,
                 kite: GovernedKite,
                 store: Store,
                 book: InstrumentBook,
                 prices: PriceBus):
        self.k = kite
        self.store = store
        self.book = book
        self.prices = prices
        self.trader: Optional[AbstractTrader] = None
        self.engine_lock = threading.RLock()

        self.bars = BarStore(timeframes=[1, 3, 5, 15])
        
        # ### IMPROVEMENT ### Tokens are now loaded from the InstrumentBook's special token cache.
        self.nifty_token = self.book.special_tokens.get("NIFTY")
        self.bn_token = self.book.special_tokens.get("BANKNIFTY")
        self.vix_token = self.book.special_tokens.get("INDIA VIX")
        if not all([self.nifty_token, self.bn_token]):
            raise SystemExit("FATAL: Could not find NIFTY/BANKNIFTY futures contracts from instrument file.")


        self.running = threading.Event()

        self.halt_trading = False
        self.regime = Regime.UNCLEAR
        self.last_trade_timestamp: Optional[datetime] = None
        self.last_regime_change_time: Optional[datetime] = None

        self.last_trading_day: Optional[date] = None
        self.consecutive_losses, self.risk_factor = 0, 1.0
        self.dynamic_account_equity = ACCOUNT_EQUITY
        self.daily_high_water_mark = self.dynamic_account_equity
        self.performance_score = 0
        self.weekly_high_water_mark = self.dynamic_account_equity
        self.in_weekly_drawdown_lock = False
        self.eod_flatten_triggered = False
        self.eod_report_sent = False

        self.classifier = RegimeClassifier(self, self.nifty_token, self.bn_token, self.vix_token)
        self.strategies: Dict[Regime, List[BaseStrategy]] = self._load_strategies()

        self.last_known_prices: Dict[int, float] = {}
        self.config = CONFIG
        self.sanity_check_pct = self.config["trading"].get("insane_tick_pct", 5.0) / 100.0

        self.trade_signal_queue = Queue()
        self.last_unrealized_pnl = 0.0
        self.fatal_error_event = threading.Event()

        self.portfolio_greeks: Dict[str, float] = {"net_delta": 0.0, "net_vega": 0.0, "net_gamma": 0.0, "net_theta": 0.0}
        self.portfolio_greeks_lock = threading.Lock()

        self.nse_calendar = mcal.get_calendar('NSE')
        self.vix_long_history_df = pd.DataFrame()
        self.scheduler = self._setup_scheduler()

        # ### PILLAR 1 IMPLEMENTATION ### Data structures for microstructure analysis
        self.tfi_scores: Dict[int, float] = {} # Trade Flow Imbalance score
        self.recent_trades: Dict[int, deque] = {} # Stores recent trades for TFI calc
        self.historical_avg_iv: Dict[int, pd.Series] = {} # For Volatility Contraction

    def set_dependencies(self, trader: AbstractTrader):
        self.trader = trader
        if not PAPER_TRADING:
            self.prices.on_order_update_callbacks.append(self.handle_order_update)
            self.prices.on_connect_callbacks.append(self.reconcile)
        L.info("Trader dependency injected into Engine.")

    def get_ohlc(self, token: int, timeframe: int) -> pd.DataFrame:
        return self.bars.get_ohlc(token, timeframe)

    def get_dynamic_correlation(self, period: int = 50) -> float:
        """Calculates the dynamic rolling correlation between NIFTY and BANKNIFTY."""
        try:
            df_n = self.get_ohlc(self.nifty_token, 5)
            df_bn = self.get_ohlc(self.bn_token, 5)
            if len(df_n) < period or len(df_bn) < period:
                return self.config["trading"].get("fallback_correlation_factor", 0.6)

            returns_n = df_n['close'].pct_change()
            returns_bn = df_bn['close'].pct_change()

            correlation = returns_n.rolling(window=period).corr(returns_bn).iloc[-1]
            return correlation if not pd.isna(correlation) else self.config["trading"].get("fallback_correlation_factor", 0.6)
        except Exception:
            return self.config["trading"].get("fallback_correlation_factor", 0.6)

    def _is_tick_sane(self, tick: Dict) -> bool:
        now_time = now_ist().time()
        if now_time < self.config["timings"]["market_settling_time"]:
            return True

        token = tick["instrument_token"]; price = tick.get("last_price")
        if price is None or price <= 0: return False
        last_price = self.last_known_prices.get(token)
        if last_price is None: self.last_known_prices[token] = price; return True
        price_change_pct = abs(price - last_price) / last_price
        if price_change_pct > self.sanity_check_pct:
            L.warning(f"INSANE TICK DETECTED for token {token}. New: {price}, Old: {last_price}. Discarding.")
            return False
        self.last_known_prices[token] = price
        return True

    def _tick_processor_worker(self):
        L.info("Tick processor worker started.")
        while self.running.is_set():
            try:
                ticks = self.prices.tick_queue.get(timeout=1)
                sane_ticks = [t for t in ticks if self._is_tick_sane(t)]
                if not sane_ticks: continue

                with self.trader.lock:
                    open_positions_by_token = {p.token: p for p in self.trader.positions.values() if p.status == PositionStatus.ACTIVE.value}

                for t in sane_ticks:
                    self.prices.last[t["instrument_token"]] = t.get("last_price")
                    self.prices.full_ticks[t["instrument_token"]] = t
                    
                    # ### PILLAR 1.2 IMPLEMENTATION: Trade Flow Imbalance (TFI) ###
                    self._update_tfi_score(t)

                    if t["instrument_token"] in open_positions_by_token:
                        self._update_position_greeks(open_positions_by_token[t["instrument_token"]], t)

                self.process_ticks(sane_ticks)
            except Empty: continue
            except Exception as e:
                L.error(f"FATAL Error in tick processor worker: {e}", exc_info=True)

    def start(self):
        if not self.trader:
            raise SystemExit("FATAL: Trader dependency not set. Call engine.set_dependencies(trader) before start().")

        self.warm_up()
        self.prices.start()
        if not self.prices.connected.wait(10):
            raise SystemExit("FATAL: PriceBus WebSocket could not connect.")

        tokens_to_subscribe = [self.nifty_token, self.bn_token]
        if self.vix_token: tokens_to_subscribe.append(self.vix_token)
        
        self.prices.subscribe(tokens_to_subscribe)

        self.running.set()

        tick_thread = threading.Thread(target=self._tick_processor_worker, name="TickProcessor", daemon=True)
        tick_thread.start()

        trade_executor_thread = threading.Thread(target=self._trade_executor_worker, name="TradeExecutor", daemon=True)
        trade_executor_thread.start()

        for name, (func, interval) in self.scheduler.items():
            thread = threading.Thread(target=self._run_task_in_loop, args=(func, interval, name), name=name, daemon=True)
            thread.start()
            L.info(f"Started scheduler for '{name}' with {interval}s interval.")

        self.loop()

    def _run_task_in_loop(self, func: Callable, interval: int, name: str):
        while self.running.is_set():
            try:
                is_halted_check = False
                if name in ["strategic_planner"]:
                    with self.engine_lock:
                        is_halted_check = self.halt_trading

                if not is_halted_check:
                    func()

            except Exception as e:
                L.error(f"Error in scheduled task '{name}': {e}", exc_info=True)
            time.sleep(interval)

    def _send_eod_report(self):
        now_time = now_ist().time()
        if now_time > CONFIG["timings"]["market_close"] and not self.eod_report_sent:
            L.info("Sending End-of-Day report...")
            wins, losses = self.store.get_todays_trades_stats(); total_trades = wins + losses; win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            report = (f"📊 **--- End of Day Report ---** 📊\n\n"
                      f"**Net Realized PnL:** ₹{self.trader.daily_realized_pnl:,.2f}\n\n"
                      f"**Total Trades:** {total_trades}\n"
                      f"**Winning Trades:** {wins}\n"
                      f"**Losing Trades:** {losses}\n"
                      f"**Win Rate:** {win_rate:.2f}%\n")
            send_alert(report); self.eod_report_sent = True

    def loop(self):
        send_alert("🛡️ SENTINEL PRIME PROTOCOL ENGAGED. Awaiting market open...")
        while now_ist().time() < CONFIG["timings"]["market_open"] and self.running.is_set():
            L.info(f"Pre-Market state. Waiting for market open at {CONFIG['timings']['market_open']}...")
            time.sleep(60)
        if not self.running.is_set(): self.stop(); return

        send_alert("🔔 Market is OPEN. Trading logic is now active.")
        L.info("Market Open state. Trading logic is active.")
        try:
            while self.running.is_set():
                now = now_ist(); now_time = now.time()
                if now_time >= CONFIG["timings"]["market_close"]:
                    L.info("Market is now CLOSED. Transitioning to post-market state.")
                    break

                if self.fatal_error_event.is_set():
                    send_alert("🔥 FATAL ERROR EVENT RECEIVED. HALTING ALL TRADING.", "critical")
                    with self.engine_lock:
                        self.halt_trading = True
                    self.fatal_error_event.clear()

                with self.engine_lock:
                    # ### IMPROVEMENT ### Added kill switch check to the main loop.
                    if os.path.exists(KILL_SWITCH_FILE):
                        if not self.halt_trading:
                            send_alert("⛔ KILL SWITCH DETECTED. HALTING ALL NEW TRADES. ⛔", "critical")
                            self.halt_trading = True
                            if G_HALTED_STATUS: G_HALTED_STATUS.set(1)
                    
                    if not self.eod_flatten_triggered and now_time >= CONFIG["timings"]["eod_flatten_time"]:
                        L.warning("EOD flatten time reached. Halting new trades and closing all positions.")
                        self.halt_trading = True
                        if G_HALTED_STATUS: G_HALTED_STATUS.set(1)
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
            L.warning("KeyboardInterrupt detected in main loop.");
            self.stop()
            return

        L.info("Post-Market state. Running final tasks for the day.")
        if not self.eod_report_sent: self._send_eod_report()
        if self.last_trading_day: self.store.set_kv(f"daily_pnl_{self.last_trading_day}", str(self.trader.daily_realized_pnl))
        send_alert(f"💤 Sentinel shutting down for the day. Final Realized PnL: ₹{self.trader.daily_realized_pnl:,.2f}")
        L.info("Daily tasks complete. Bot will now stop."); self.stop()

    def _load_strategies(self) -> Dict[Regime, List[BaseStrategy]]:
        return {
            Regime.COMPRESSION: [MomentumBreakoutStrategy(StrategyName.MOMENTUM_BREAKOUT, self, CONFIG['strategies']['MomentumBreakout'])],
            Regime.TRENDING_UP: [TrendPullbackStrategy(StrategyName.TREND_PULLBACK, self, CONFIG['strategies']['TrendPullback'])],
            Regime.TRENDING_DOWN: [TrendPullbackStrategy(StrategyName.TREND_PULLBACK, self, CONFIG['strategies']['TrendPullback'])],
            Regime.CHOP: [MeanReversionStrategy(StrategyName.MEAN_REVERSION, self, CONFIG['strategies']['MeanReversion'])]
        }
    def _setup_scheduler(self) -> Dict[str, Tuple[Callable, int]]:
        tasks = {
            "strategic_planner": (self._run_strategic_planner, 5),
            "position_management": (self.manage_positions, 1), # Faster loop for adaptive entry
            "pnl_updater": (self._update_pnl_metrics, 2),
            "reconciliation": (self.reconcile, 300),
            "health_check": (self.health_check, 60),
            "eod_report": (self._send_eod_report, 300),
            "equity_update": (self._update_dynamic_equity, 600),
            "data_persistence": (self._persist_bar_data, 3600),
            "bar_reconciliation": (self._reconcile_bars, 60),
            # ### IMPROVEMENT ### Added new scheduled task for P&L reconciliation
            "pnl_reconciliation": (self._reconcile_broker_pnl, 900)
        }
        if METRICS_APP:
            tasks["prometheus_metrics"] = (self._update_prometheus_metrics, 15)
        return tasks

    def _reset_daily_state(self):
        L.info("Resetting daily state for new trading day.")
        now = now_ist()
        if self.last_trading_day and self.last_trading_day.weekday() > now.date().weekday():
            L.info("New week detected. Resetting weekly high water mark and drawdown lock.")
            self.weekly_high_water_mark = self.dynamic_account_equity
            self.in_weekly_drawdown_lock = False

        if self.last_trading_day:
            self.store.set_kv(f"daily_pnl_{self.last_trading_day}", str(self.trader.daily_realized_pnl))

        self._update_dynamic_equity()
        self.trader.daily_realized_pnl = 0.0

        with self.engine_lock:
            self.halt_trading = False
            # ### IMPROVEMENT ### Remove kill switch file at the start of a new day.
            if os.path.exists(KILL_SWITCH_FILE):
                try:
                    os.remove(KILL_SWITCH_FILE)
                    L.warning("Kill switch file removed for the new trading day.")
                except OSError as e:
                    L.error(f"Could not remove kill switch file: {e}")
            
            if G_HALTED_STATUS: G_HALTED_STATUS.set(0)
            self.last_trade_timestamp = None
            with self.portfolio_greeks_lock:
                self.portfolio_greeks = {"net_delta": 0.0, "net_vega": 0.0, "net_gamma": 0.0, "net_theta": 0.0}

        self.eod_flatten_triggered = False; self.eod_report_sent = False
        self.last_trading_day = now.date()
        self.consecutive_losses = 0; self.risk_factor = 1.0
        self.daily_high_water_mark = self.dynamic_account_equity
        self.performance_score = 0
        send_alert(f"☀️ New Trading Day: {self.last_trading_day}. Equity: ₹{self.dynamic_account_equity:,.2f}")

    def warm_up(self):
        L.info("Warming up... Priming historical data."); self._reset_daily_state();
        to_date, from_date = self.last_trading_day, self.last_trading_day - timedelta(days=CONFIG["technical"]["warmup_days"])

        tokens_to_prime = [self.nifty_token, self.bn_token]

        for token in tokens_to_prime:
            if token is None: continue
            symbol = self.book.get_symbol(token) or f"Token {token}"
            L.info(f"Priming historical data for: {symbol}")
            hist = self.k.historical_data(token, from_date, to_date, "minute")
            if hist:
                self.bars.prime(token, pd.DataFrame(hist))
                L.info(f"Successfully primed {len(hist)} bars for {symbol}.")
            else:
                L.error(f"Failed to prime history for {symbol} after retries.")

        if self.vix_token:
            L.info("Priming long-term VIX history for IV Rank calculation...")
            vix_from_date = self.last_trading_day - timedelta(days=365)
            vix_hist = self.k.historical_data(self.vix_token, vix_from_date, to_date, "day")
            if vix_hist:
                self.vix_long_history_df = pd.DataFrame(vix_hist)
                L.info(f"Successfully primed {len(self.vix_long_history_df)} days of VIX data.")
            else:
                L.error("Failed to prime VIX history. IV Rank filter will be disabled.")

        pnl_str = self.store.get_kv(f"daily_pnl_{self.last_trading_day}", "0.0");
        self.trader.daily_realized_pnl = float(pnl_str)
        self.daily_high_water_mark = self.dynamic_account_equity + self.trader.daily_realized_pnl
        self.weekly_high_water_mark = max(self.weekly_high_water_mark, self.daily_high_water_mark)
        L.info(f"State loaded. Daily PnL restored to: {self.trader.daily_realized_pnl}. Daily HWM: {self.daily_high_water_mark}")
    
    def process_ticks(self, ticks: List[Dict]):
        for tick in ticks:
            self.bars.add_tick(tick)

    def _reconcile_bars(self):
        try:
            now = now_ist()
            if not (self.config["timings"]["market_open"] < now.time() < self.config["timings"]["eod_flatten_time"]):
                return

            with self.bars.lock:
                for token in [self.nifty_token, self.bn_token]:
                    if not token: continue
                    hist_data = self.k.historical_data(token, now - timedelta(minutes=5), now, "minute")
                    if not hist_data: continue

                    hist_df = pd.DataFrame(hist_data)
                    bar_df = self.bars.data.get(token, {}).get(1)
                    if bar_df is None: continue
                    
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
            if chain.empty: return None, None

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

    def _verify_position_risk(self, p: Position):
        L.info(f"Post-fill verification for {p.tradingsymbol}...")
        
        underlying_name = _get_underlying(p.tradingsymbol)
        underlying_token = self.bn_token if "BANKNIFTY" in underlying_name else self.nifty_token
        spot = self.prices.ltp(underlying_token)
        if not spot: return
        
        opt_details = self.book.df_by_token.loc[p.token]
        T = _calculate_time_to_expiry(opt_details['expiry'].date(), now_ist(), self.config["timings"]["market_close"])
        hv = calculate_historical_volatility(self.get_ohlc(underlying_token, 1)['close'])
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

    def _run_strategic_planner(self):
        with self.engine_lock:
            now = now_ist()
            
            is_trading_day = self.nse_calendar.valid_days(start_date=now.date(), end_date=now.date()).size > 0

            if now.date() > self.last_trading_day and is_trading_day:
                self._reset_daily_state()

            final_entry_time = self.config["timings"]["final_entry_time"]
            if now.weekday() in [2, 3] and self.config["timings"].get("final_expiry_entry_time"):
                final_entry_time = self.config["timings"]["final_expiry_entry_time"]

            if (self.halt_trading or
                not (self.config["timings"]["market_settling_time"] <= now.time() < final_entry_time)):
                return

            max_iv_rank = self.config["trading"].get("max_iv_rank_entry", 101.0)
            current_iv_rank = self._get_iv_rank()
            if current_iv_rank and current_iv_rank > max_iv_rank:
                L.info(f"Trade generation skipped. IV Rank {current_iv_rank:.1f}% > threshold {max_iv_rank}%.")
                return

            cooldown_ok = not self.last_trade_timestamp or (now - self.last_trade_timestamp) > timedelta(minutes=self.config["trading"]["trade_cooldown_minutes"])
            if not cooldown_ok:
                return

            self.run_regime_classification(now)

            strategies_for_regime = self.strategies.get(self.regime)
            if not strategies_for_regime:
                return

            # Simplified to one signal for clarity
            for token in [self.nifty_token, self.bn_token]:
                for strategy in strategies_for_regime:
                    if signal := strategy.evaluate(token, self.regime, now):
                        trade_params = self.get_trade_params(
                            token=token, side=signal.side,
                            risk_points_on_underlying=signal.risk_points,
                            reward_points_on_underlying=signal.reward_points,
                            strategy=signal.strategy_name.value, regime=self.regime
                        )
                        if not trade_params or trade_params['lots'] <= 0: continue
                        if not self.risk_ok(hypothetical_params=trade_params):
                            L.warning(f"--- Trade blocked by master risk controls for {signal.strategy_name.value}. ---")
                            continue
                        L.info(f"Planner approved signal for {signal.strategy_name.value}. Placing on execution queue.")
                        self.trade_signal_queue.put(trade_params)
                        return # Only take the first valid signal
        
    def _trade_executor_worker(self):
        L.info("Trade executor worker started.")
        while self.running.is_set():
            try:
                trade_params = self.trade_signal_queue.get(timeout=1)
                
                # Double-check risk just before execution
                if not self.risk_ok(hypothetical_params=trade_params):
                    L.warning(f"Trade for {trade_params['strategy']} rejected by final risk check just before execution.")
                    continue

                L.info(f"Executor received signal for {trade_params['strategy']}. Executing trade.")
                if self.trader.open_position(trade_params):
                    with self.engine_lock:
                        self.last_trade_timestamp = now_ist()

            except Empty:
                continue
            except Exception as e:
                L.error(f"FATAL Error in trade executor worker: {e}", exc_info=True)

    def _handle_entry_fill(self, pos: Position, order: Dict):
        """Helper to process fills for entry orders."""
        filled_qty = order.get('filled_quantity', 0)
        
        # Determine if the order is still open with partial fills
        if order.get('status') == 'OPEN' and filled_qty > 0:
            pos.is_entry_order_open = True
        elif order.get('status') in ['COMPLETE', 'CANCELLED', 'REJECTED']:
            pos.is_entry_order_open = False
        
        if filled_qty > pos.qty: # New fills have occurred
            new_fills = filled_qty - pos.qty
            L.info(f"✅ Entry fill received for {pos.tradingsymbol}. Qty: {new_fills}. Total Filled: {filled_qty}.")
            
            # On first fill, set up the position parameters
            if pos.status == PositionStatus.PENDING_ENTRY.value:
                avg_price = self._get_order_avg_price(pos.entry_order_id)
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
                self._initialize_position_greeks(pos)
            
            pos.qty = filled_qty
            pos.status = PositionStatus.OPEN_AWAITING_BRACKETS
            self.store.upsert_position(pos)
        
        if order.get('status') in ['COMPLETE', 'CANCELLED', 'REJECTED']:
            if pos.qty == 0:
                L.warning(f"Entry order {pos.entry_order_id} for {pos.tradingsymbol} {order.get('status')} with no fills. Removing position.")
                self.trader.positions.pop(pos.id, None)
            else:
                L.info(f"Entry order for {pos.tradingsymbol} is final. Total filled: {pos.qty}/{pos.initial_qty}.")
                pos.status = PositionStatus.OPEN_AWAITING_BRACKETS
                self.store.upsert_position(pos)
                self._verify_position_risk(pos)

    def _handle_exit_fill(self, pos: Position, order: Dict):
        """Helper to process fills for exit orders."""
        if pos.status == PositionStatus.CLOSED.value:
            L.info(f"Ignoring duplicate fill for already closed position {pos.tradingsymbol}")
            return
        
        reason = pos.exit_reason or "EXIT_FILL"
        L.info(f"Exit order {order.get('order_id')} ({reason}) complete for {pos.id}.")
        self._cancel_all_open_orders_for_pos(pos, cancel_entry=pos.is_entry_order_open)
        
        exit_price = self._get_order_avg_price(order.get('order_id')) or self.prices.ltp(pos.token)
        if not exit_price and reason == "TP_HIT": exit_price = pos.tp_price
        if not exit_price: L.error(f"Could not determine exit price for {pos.tradingsymbol}!"); return

        pnl = (exit_price - pos.entry_price) * pos.initial_qty
        self.trader.daily_realized_pnl += pnl
        pos.status = PositionStatus.CLOSED.value
        pos.exit_price = exit_price

        with self.portfolio_greeks_lock:
            self.portfolio_greeks["net_delta"] -= pos.greeks.get("delta", 0.0) * pos.initial_qty
            # ... and other greeks
        pos.greeks = {}

        self.store.upsert_position(pos)
        self.store.log_closed_trade(pos, exit_price, reason)
        self._update_performance_metrics(pnl)
        self.trader.positions.pop(pos.id, None)
        send_alert(f"❌ CLOSED {pos.tradingsymbol} ({reason}). Final PnL: {pnl:.2f}. Daily PnL: {self.trader.daily_realized_pnl:.2f}")

    def handle_order_update(self, order: Dict):
        oid, status = str(order.get('order_id')), order.get('status')
        pos_id = f"LIVE_{oid}"

        with self.trader.lock:
            pos = self.trader.positions.get(pos_id)
            if not pos:
                # Find position by TP or exit order ID
                pos = next((p for p in self.trader.positions.values() if oid in [p.tp_order_id, p.exit_order_id]), None)
                if not pos: return

            if oid == pos.entry_order_id:
                self._handle_entry_fill(pos, order)
            
            elif oid in [pos.tp_order_id, pos.exit_order_id] and status == 'COMPLETE':
                self._handle_exit_fill(pos, order)

    def run_regime_classification(self, current_time: datetime):
        raw_regime, active_token = self.classifier.get_raw_classification(self.regime)

        if self.classifier.potential_regime and raw_regime == self.classifier.potential_regime[0]:
            self.classifier.confirmation_count += 1
        else:
            self.classifier.potential_regime = (raw_regime, active_token)
            self.classifier.confirmation_count = 1

        if self.classifier.confirmation_count >= self.classifier.confirmation_threshold:
            if self.regime != raw_regime:
                old_regime_name = self.regime.name
                self.regime = raw_regime

                if G_CURRENT_REGIME:
                    try:
                        G_CURRENT_REGIME.clear(); G_CURRENT_REGIME.labels(regime_name=self.regime.name).set(self.regime.value)
                    except Exception as e: L.warning(f"Failed to set Prometheus regime gauge: {e}")

                self.last_regime_change_time = current_time
                L.info(f"REGIME SHIFT CONFIRMED: {old_regime_name} -> {self.regime.name}")
                send_alert(f"REGIME SHIFT: {old_regime_name} -> {self.regime.name}")

    def _find_best_option_contract(self, underlying_token: int, expiry: date, option_type: OptionType, strategy: str, regime: Regime) -> Optional[Dict]:
        spot = self.prices.ltp(underlying_token)
        if not spot: return None

        strategy_obj = next((s for s_list in self.strategies.values() for s in s_list if s.name.value == strategy), None)
        if not strategy_obj: return None

        strike_cfg = self.config['trading']['strike_selection']
        if regime in [Regime.TRENDING_UP, Regime.TRENDING_DOWN]: target_delta = strike_cfg.get('trend_delta', 0.45)
        elif regime == Regime.CHOP: target_delta = strike_cfg.get('chop_delta', 0.60)
        else: target_delta = strike_cfg.get('compression_delta', 0.50)

        underlying_symbol = self.book.get_symbol(underlying_token); underlying_name = _get_underlying(underlying_symbol)
        
        # ### PILLAR 1.3 IMPLEMENTATION: Volatility Contraction Breakout Logic ###
        step = self.book.step_size(underlying_name)
        num_strikes = 5 # Monitor 5 nearest OTM strikes
        
        # Get the chain of options to analyze for IV
        otm_strikes_chain = self.book.get_option_chain(underlying_name, expiry)
        otm_calls = otm_strikes_chain[(otm_strikes_chain['instrument_type'] == 'CE') & (otm_strikes_chain['strike'] > spot)].sort_values('strike').head(num_strikes)
        otm_puts = otm_strikes_chain[(otm_strikes_chain['instrument_type'] == 'PE') & (otm_strikes_chain['strike'] < spot)].sort_values('strike', ascending=False).head(num_strikes)
        
        iv_analysis_chain = pd.concat([otm_calls, otm_puts])
        if iv_analysis_chain.empty: return None

        now = now_ist(); market_close_time = self.config["timings"]["market_close"]
        T = _calculate_time_to_expiry(expiry.date() if isinstance(expiry, pd.Timestamp) else expiry, now, market_close_time)
        underlying_bars = self.bars.get_ohlc(underlying_token, 1)
        hv = calculate_historical_volatility(underlying_bars['close'], timeframe_minutes=1) if not underlying_bars.empty else 0.3
        
        ivs = []
        for _, row in iv_analysis_chain.iterrows():
            tick = self.prices.get_full_tick(int(row['instrument_token']))
            if tick and tick.get('last_price'):
                iv = calculate_iv(tick['last_price'], spot, row['strike'], T, 0.05, row['instrument_type'] == 'CE', hv_fallback=hv)
                ivs.append(iv)
        
        if not ivs: return None
        avg_iv = sum(ivs) / len(ivs)
        
        # Store and check for contraction
        if underlying_token not in self.historical_avg_iv:
            self.historical_avg_iv[underlying_token] = pd.Series(dtype=float)
        
        s = self.historical_avg_iv[underlying_token]
        # Using concat instead of append for modern pandas
        self.historical_avg_iv[underlying_token] = pd.concat([s, pd.Series([avg_iv], index=[now])])

        # Trim the series to keep it manageable (e.g., last 4 hours of data)
        self.historical_avg_iv[underlying_token] = self.historical_avg_iv[underlying_token].last('4H')

        lookback = CONFIG["technical"].get("iv_contraction_lookback", 120) # 120 5-sec checks = 10 mins
        iv_series = self.historical_avg_iv[underlying_token]
        
        is_in_contraction = False
        if len(iv_series) > lookback:
            iv_percentile = iv_series.rolling(lookback).rank(pct=True).iloc[-1]
            if iv_percentile < CONFIG["technical"].get("iv_contraction_threshold_pct", 10) / 100.0:
                is_in_contraction = True
                L.info(f"VOLATILITY CONTRACTION DETECTED for {underlying_name}. Avg IV Pct Rank: {iv_percentile*100:.2f}%")

        if strategy == StrategyName.MOMENTUM_BREAKOUT.value and not is_in_contraction:
             L.info(f"MomentumBreakout signal for {underlying_name} skipped. Not in IV contraction phase.")
             return None

        # Now, find the single best contract to trade based on delta
        full_chain = self.book.get_option_chain(underlying_name, expiry); 
        trade_chain = full_chain[full_chain['instrument_type'] == option_type.value].copy()
        atm_strike = round(spot / step) * step
        search_range = 15 * step; trade_chain = trade_chain[(trade_chain['strike'] >= atm_strike - search_range) & (trade_chain['strike'] <= atm_strike + search_range)]
        if trade_chain.empty: return None

        min_volume = self.config['trading']['min_option_volume']; min_oi = self.config['trading']['min_option_oi']
        options_with_metrics = []
        for _, row in trade_chain.iterrows():
            token = int(row['instrument_token']); tick = self.prices.get_full_tick(token)
            if not tick: continue
            if tick.get('volume', 0) < min_volume or tick.get('open_interest', 0) < min_oi: continue
            ltp = tick.get('last_price');
            if not ltp or ltp < CONFIG['trading']['min_option_price']: continue
            depth = tick.get('depth');
            if not depth or not depth.get('buy') or not depth.get('sell'): continue
            bid_price, ask_price = depth['buy'][0]['price'], depth['sell'][0]['price']; spread = (ask_price - bid_price) / ask_price if ask_price > 0 else float('inf')
            if spread > CONFIG['trading']['max_bid_ask_spread_pct'] / 100.0: continue
            
            iv = calculate_iv(ltp, spot, row['strike'], T, 0.05, option_type == OptionType.CE, hv_fallback=hv)
            greeks = calculate_greeks(spot, row['strike'], T, 0.05, iv, option_type == OptionType.CE)
            options_with_metrics.append({'delta_diff': abs(abs(greeks['delta']) - target_delta), 'opt': row.to_dict(), 'ltp': ltp, 'greeks': greeks})

        if not options_with_metrics: return None
        return min(options_with_metrics, key=lambda x: x['delta_diff'])


    def get_trade_params(
        self,
        token: int,
        side: OrderSide,
        risk_points_on_underlying: float,
        reward_points_on_underlying: float,
        strategy: str,
        regime: Regime
    ) -> Optional[Dict]:
        underlying_symbol = self.book.get_symbol(token)
        if not underlying_symbol: return None

        underlying_name = _get_underlying(underlying_symbol)
        lot_size = self.book.lot_size(underlying_name)
        expiry = self.book.find_nearest_expiry_date(underlying_name)

        if not all([lot_size, expiry]): return None
        option_type = OptionType.CE if side == OrderSide.BUY else OptionType.PE
        best_option_data = self._find_best_option_contract(underlying_token=token, expiry=expiry, option_type=option_type, strategy=strategy, regime=regime)
        if not best_option_data: return None

        option_contract, option_ltp, greeks = best_option_data['opt'], best_option_data['ltp'], best_option_data['greeks']
        
        # ### PILLAR 1.1 & 1.2 IMPLEMENTATION: Microstructure Filters ###
        if not self._check_order_book_imbalance(option_contract['instrument_token'], side):
            L.warning(f"Trade for {option_contract['tradingsymbol']} REJECTED by OBI filter.")
            return None
        if not self._check_tfi(option_contract['instrument_token'], side):
            L.warning(f"Trade for {option_contract['tradingsymbol']} REJECTED by TFI filter.")
            return None
        
        estimated_delta = greeks['delta']
        spot_price = self.prices.ltp(token)
        if not spot_price: return None

        resistance, support = self._get_oi_barriers(underlying_name, expiry, spot_price)
        oi_profit_target = None
        min_dist_pct = self.config['trading'].get('min_dist_from_oi_wall_pct', 0.25)
        if side == OrderSide.BUY and resistance and (resistance - spot_price < (spot_price * (min_dist_pct / 100))):
            L.warning(f"Trade blocked. Too close to Call OI wall at {resistance}."); return None
        elif side == OrderSide.SELL and support and (spot_price - support < (spot_price * (min_dist_pct / 100))):
            L.warning(f"Trade blocked. Too close to Put OI wall at {support}."); return None
        oi_profit_target = resistance if side == OrderSide.BUY else support

        final_sl_points_on_option = risk_points_on_underlying * abs(estimated_delta)
        max_sl_pct = CONFIG["trading"]["max_sl_pct_of_premium"]
        if (option_ltp > 0) and (final_sl_points_on_option / option_ltp) > (max_sl_pct / 100.0):
            L.warning(f"Trade REJECTED: Calculated SL ({final_sl_points_on_option:.2f}) exceeds max {max_sl_pct}% of premium ({option_ltp}).")
            return None

        risk_per_lot = final_sl_points_on_option * lot_size
        number_of_lots = self.calculate_position_size(token, risk_per_lot, vega_per_lot=greeks['vega'] * lot_size)
        if number_of_lots <= 0: return None

        underlying_sl_level = (spot_price - risk_points_on_underlying) if side == OrderSide.BUY else (spot_price + risk_points_on_underlying)
        final_tp_points_on_option = reward_points_on_underlying * abs(estimated_delta)
        total_trade_risk = risk_per_lot * number_of_lots
        strategy_config = self.config['strategies'].get(strategy, {})

        return {
            "opt": option_contract, "ltp_opt": option_ltp, "lots": number_of_lots,
            "strategy": strategy, "regime": regime.name, "option_sl_points": final_sl_points_on_option,
            "option_tp_points": final_tp_points_on_option, "total_trade_risk": total_trade_risk,
            "underlying_sl": underlying_sl_level, "greeks": greeks,
            "max_trade_duration_minutes": strategy_config.get('max_duration_minutes', 90),
            "oi_profit_target": oi_profit_target, "intended_risk_rupees": total_trade_risk
        }

    def calculate_position_size(self, underlying_token: int, risk_per_lot: float, vega_per_lot: float) -> int:
        if risk_per_lot <= 0: return 0
        risk_tiers = CONFIG["trading"]["risk_tiers"]

        is_expiry_day = self.book.find_nearest_expiry_date(_get_underlying(self.book.get_symbol(underlying_token))) == date.today()
        expiry_day_risk_factor = 1.0
        if is_expiry_day and CONFIG["trading"].get("expiry_day_protocol_active", True):
            expiry_day_risk_factor = CONFIG["trading"].get("expiry_day_risk_reduction_factor", 0.5)
            L.warning(f"EXPIRY DAY PROTOCOL: Applying risk reduction factor of {expiry_day_risk_factor}.")

        if self.in_weekly_drawdown_lock:
            active_risk_pct = risk_tiers["defensive"]; L.warning("Weekly drawdown lock is active. Using DEFENSIVE risk tier.")
        elif self.performance_score <= -2: active_risk_pct = risk_tiers["defensive"]
        elif self.performance_score >= 2: active_risk_pct = risk_tiers["aggressive"]
        else: active_risk_pct = risk_tiers["standard"]

        df_1m = self.bars.get_ohlc(underlying_token, 1)
        if len(df_1m) < 21: return 1

        atr = df_1m.ta.atr(20).iloc[-1]; spot = df_1m.iloc[-1]['close']
        vol_pct = atr / spot if spot > 0 else 0; base_vol = 0.005
        vol_adjustment = min(1.5, max(0.5, base_vol / vol_pct if vol_pct > 0 else 1.0))

        vix_ltp = self.prices.ltp(self.vix_token); vix_params = CONFIG["trading"]["vix_adjustment"]; vix_risk_factor = 1.0
        if vix_ltp:
            if vix_ltp > vix_params["high_threshold"]: vix_risk_factor = vix_params["high_factor"]
            elif vix_ltp < vix_params["low_threshold"]: vix_risk_factor = vix_params["low_factor"]

        # ### PILLAR 3.2 IMPLEMENTATION: Time-of-Day Risk Scaling ###
        time_of_day_multiplier = self._get_time_of_day_risk_multiplier()

        final_risk_factor = self.risk_factor * vol_adjustment * vix_risk_factor * expiry_day_risk_factor * time_of_day_multiplier
        allowed_risk = self.dynamic_account_equity * (active_risk_pct / 100.0) * final_risk_factor

        calculated_lots = int(math.floor(allowed_risk / risk_per_lot)) if risk_per_lot > 0 else 0
        L.info(f"Position Size Calc: PerfScore={self.performance_score}, RiskTier={active_risk_pct}%, AllowedRisk={allowed_risk:.2f}, Lots={calculated_lots}")
        return max(0, min(calculated_lots, CONFIG['trading']['max_lots_per_trade']))

    def _update_performance_metrics(self, pnl: float):
        with self.engine_lock:
            current_equity = self.dynamic_account_equity + self.trader.daily_realized_pnl + self.last_unrealized_pnl
            self.daily_high_water_mark = max(self.daily_high_water_mark, current_equity)
            self.weekly_high_water_mark = max(self.weekly_high_water_mark, current_equity)

            if pnl > 0: self.performance_score = min(4, self.performance_score + 1)
            else: self.performance_score = max(-4, self.performance_score - 2)

            if pnl < 0: self.consecutive_losses += 1
            else: self.consecutive_losses = 0

            loss_streak_config = CONFIG["trading"]["consecutive_loss_adjustment"]
            self.risk_factor = max(loss_streak_config["min_factor"], 1.0 - loss_streak_config["reduction_per_loss"] * self.consecutive_losses)

            L.info(f"PnL: {pnl:.2f}, PerfScore: {self.performance_score}, ConsecLosses: {self.consecutive_losses}, RiskFactor: {self.risk_factor:.2f}")
            self.store.set_kv(f"daily_pnl_{self.last_trading_day}", str(self.trader.daily_realized_pnl))

    def manage_positions(self):
        with self.trader.lock:
            active_positions = list(self.trader.positions.values())

        now = now_ist()
        for p in active_positions:
            # ### PILLAR 2 IMPLEMENTATION: Adaptive Entry Management ###
            if p.status == PositionStatus.PENDING_ENTRY.value:
                self._manage_adaptive_entry(p, now)
                continue
            
            # ### IMPROVEMENT ### Ensure brackets are always placed if a position is open
            if p.status == PositionStatus.OPEN_AWAITING_BRACKETS.value:
                if not self.trader.place_bracket_orders(p):
                    send_alert(f"CRITICAL: FAILED to place brackets for {p.tradingsymbol}. Closing position.", "critical")
                    if not self.trader.close_position(p, "BRACKET_PLACEMENT_FAILURE"):
                        send_alert(f"🔥 FATAL: UNABLE TO CLOSE {p.tradingsymbol}. HALTING ALL TRADING.", "critical")
                        self.fatal_error_event.set()
                continue
            
            if p.status == PositionStatus.PENDING_SL_EXIT.value:
                ltp = self.prices.ltp(p.token)
                if ltp and ltp < (p.sl_price * 0.99): # Price gapped below SL-Limit
                    L.warning(f"SL-L for {p.tradingsymbol} likely missed (LTP: {ltp}, SL:{p.sl_price}). Firing MARKET order.")
                    self.trader.close_position(p, "SL_L_MISSED_MK_FALLBACK")
                continue

            if p.status not in [PositionStatus.ACTIVE.value, PositionStatus.PARTIALLY_CLOSED.value]:
                continue

            ltp = self.prices.ltp(p.token)
            if not ltp: continue

            if (now - p.opened_at).total_seconds() / 60 > p.max_trade_duration_minutes:
                L.info(f"Position {p.tradingsymbol} hit time-stop of {p.max_trade_duration_minutes} mins. Closing.")
                self.trader.close_position(p, "TIME_STOP_EXIT")
                continue

            if ltp <= p.sl_price:
                self.trader.execute_simulated_sl(p)
                continue

            with self.trader.lock:
                p.high_price_since_entry = max(p.high_price_since_entry, ltp)

            profit_points = ltp - p.entry_price
            
            # Handle Scale Outs
            with self.trader.lock:
                for rule in p.scale_out_rules:
                    target = rule['rr_target']
                    if target not in p.triggered_scale_out_targets and profit_points >= p.initial_risk_points * target:
                        qty_to_close = int(p.initial_qty * (rule['pct_to_close'] / 100.0))
                        if self.trader.scale_out(p, qty_to_close):
                            send_alert(f"💰 Scaled Out {qty_to_close} of {p.tradingsymbol} at {ltp:.2f} (RR: {target}x)")
                            p.triggered_scale_out_targets.append(target)
                            break # Only one scale-out per cycle
            if p.status != PositionStatus.ACTIVE.value: continue # State may have changed after scale-out

            # Handle Trailing Logic
            with self.trader.lock:
                if not p.breakeven_armed and profit_points >= p.initial_risk_points * CONFIG['trading']['breakeven_trigger_rr']:
                    L.info(f"Arming Breakeven for {p.tradingsymbol}");
                    self.trader.modify_sl(p, p.entry_price + self.book.tick_size(p.tradingsymbol));
                    p.breakeven_armed = True

                if not p.trailing_sl_armed and profit_points >= p.initial_risk_points * CONFIG['trading']['trailing_sl_activation_rr']:
                    p.trailing_sl_armed = True;
                    L.info(f"Trailing SL armed for {p.tradingsymbol} after reaching {CONFIG['trading']['trailing_sl_activation_rr']}R.")

                if p.trailing_sl_armed:
                    if calculated_new_sl := self._calculate_trailing_stop(p):
                        if calculated_new_sl > p.sl_price:
                            self.trader.modify_sl(p, calculated_new_sl)
            
            self.store.upsert_position(p)

    def _calculate_trailing_stop(self, p: Position) -> Optional[float]:
        try:
            trail_tf = CONFIG["trading"]["trailing_sl_timeframe_scaled_out"] if p.triggered_scale_out_targets else CONFIG["trading"]["trailing_sl_timeframe"]
            df = self.get_ohlc(p.token, trail_tf)
            if len(df) < CONFIG["trading"]["trailing_sl_chandelier_period"]: return None

            atr = df.ta.atr(length=CONFIG["trading"]["trailing_sl_chandelier_period"]).iloc[-1]
            if pd.isna(atr): return None

            chandelier_multiplier = CONFIG["trading"]["trailing_sl_chandelier_multiplier_scaled_out"] if p.triggered_scale_out_targets else CONFIG["trading"]["trailing_sl_chandelier_multiplier"]
            
            # Chandelier Exits use the highest high (for longs) or lowest low (for shorts) over the period
            high_over_period = df['high'].rolling(CONFIG["trading"]["trailing_sl_chandelier_period"]).max().iloc[-1]
            
            new_sl_price = high_over_period - atr * chandelier_multiplier
            
            new_sl_target = max(p.sl_price, new_sl_price)
            if p.breakeven_armed:
                new_sl_target = max(new_sl_target, p.entry_price + self.book.tick_size(p.tradingsymbol))
            return new_sl_target
        except Exception as e:
            L.warning(f"Could not calculate trailing stop for {p.tradingsymbol}: {e}")
            return None

    def risk_ok(self, hypothetical_params: Dict) -> bool:
        with self.engine_lock:
            if self.halt_trading: return False

            lot_size = self.book.lot_size(_get_underlying(hypothetical_params['opt']['tradingsymbol']))
            if not lot_size: return False

            with self.portfolio_greeks_lock:
                hypothetical_qty = hypothetical_params['lots'] * lot_size
                post_trade_delta = self.portfolio_greeks["net_delta"] + (hypothetical_params['greeks']['delta'] * hypothetical_qty)
                post_trade_vega = self.portfolio_greeks["net_vega"] + (hypothetical_params['greeks']['vega'] * hypothetical_qty)
                max_delta = self.config["trading"].get("max_portfolio_net_delta")
                max_vega = self.config["trading"].get("max_portfolio_net_vega")

                if max_delta and abs(post_trade_delta) > max_delta:
                    L.warning(f"Trade REJECTED: Breach max delta. Post-trade: {post_trade_delta:.0f}, Limit: {max_delta}")
                    return False
                if max_vega and post_trade_vega > max_vega:
                    L.warning(f"Trade REJECTED: Breach max vega. Post-trade: {post_trade_vega:.0f}, Limit: {max_vega}")
                    return False

            current_equity = self.dynamic_account_equity + self.trader.daily_realized_pnl + self.last_unrealized_pnl
            
            # ### PILLAR 3.1 IMPLEMENTATION: Profit Lock-in Drawdown ###
            realized_pnl = self.trader.daily_realized_pnl
            profit_lock_floor = -float('inf')
            
            if realized_pnl > 0:
                profit_at_risk_pct = self.config["trading"].get("profit_lockin_pct_at_risk", 40.0) / 100.0
                profit_at_risk = realized_pnl * profit_at_risk_pct
                profit_lock_floor = self.daily_high_water_mark - profit_at_risk

            static_dd_limit = self.daily_high_water_mark * (MAX_DAILY_DRAWDOWN_PCT / 100.0)
            static_floor = self.daily_high_water_mark - static_dd_limit
            
            final_floor = max(static_floor, profit_lock_floor)
            
            daily_dd_pct = (self.daily_high_water_mark - current_equity) / self.daily_high_water_mark * 100 if self.daily_high_water_mark > 0 else 0
            if G_DAILY_DRAWDOWN_PCT: G_DAILY_DRAWDOWN_PCT.set(daily_dd_pct)

            if current_equity < final_floor:
                send_alert(f"⛔ DD LIMIT HIT. HALTING. Peak Equity: {self.daily_high_water_mark:.2f}, Current: {current_equity:.2f} (Floor: {final_floor:.2f})", "critical")
                self.halt_trading = True
                if G_HALTED_STATUS: G_HALTED_STATUS.set(1)

                with self.trader.lock:
                    positions_to_close = [p for p in list(self.trader.positions.values()) if p.status not in [PositionStatus.CLOSED.value, PositionStatus.PENDING_CLOSURE.value]]
                for p in positions_to_close: self.trader.close_position(p, "DD_LIMIT_HIT")
                return False
            return True

    def reconcile(self):
        if PAPER_TRADING: return
        L.info("--- Starting State Reconciliation with Broker ---")
        try:
            broker_positions_data = self.k.positions()
            if not broker_positions_data: L.warning("Could not get broker positions for reconciliation."); return

            broker_positions_raw = broker_positions_data.get('net', []);
            db_positions = self.store.load_open_positions();
            broker_positions_map = { pos['tradingsymbol']: pos for pos in broker_positions_raw if pos.get('product') == 'MIS' and abs(pos.get('quantity', 0)) > 0 };
            broker_symbols, db_symbols = set(broker_positions_map.keys()), {p.tradingsymbol for p in db_positions.values()}

            for symbol in db_symbols - broker_symbols:
                pos = next((p for p in db_positions.values() if p.tradingsymbol == symbol), None)
                if pos:
                    send_alert(f"RECONCILE: DB has {symbol} but broker does not. Marking as closed.", "warning");
                    pos.status = PositionStatus.CLOSED.value; pos.exit_reason = "RECONCILE_GHOST_CLOSE";
                    self.store.upsert_position(pos)

            for symbol in broker_symbols - db_symbols:
                rogue_pos_data = broker_positions_map[symbol]; qty = rogue_pos_data['quantity'];
                send_alert(f"🔥 RECONCILE: Rogue position for {symbol} (Qty: {qty}) found at broker! Auto-flattening.", "critical");
                transaction_type = self.k.TRANSACTION_TYPE_SELL if qty > 0 else self.k.TRANSACTION_TYPE_BUY
                oid = self.k.place_order( variety=self.k.VARIETY_REGULAR, exchange="NFO", tradingsymbol=symbol,
                                         transaction_type=transaction_type, quantity=abs(qty),
                                         product=self.k.PRODUCT_MIS, order_type=self.k.ORDER_TYPE_MARKET )
                if oid is None: send_alert(f"🔥🔥 FATAL: FAILED to auto-flatten rogue position {symbol}. MANUAL INTERVENTION REQUIRED!", "critical")
                else: L.info(f"Placed market order {oid} to flatten rogue position {symbol}.")

            L.info("--- Reconciliation Complete ---")
            with self.trader.lock: self.trader.positions = self.store.load_open_positions()
        except Exception as e: L.error(f"Reconciliation failed: {e}", exc_info=True)

    def health_check(self):
        if not self.prices.connected.is_set():
            send_alert("🔥 CRITICAL: PriceBus WebSocket is disconnected!", "critical")
            if G_WS_CONNECTED: G_WS_CONNECTED.set(0)
        else:
            if G_WS_CONNECTED: G_WS_CONNECTED.set(1)

        if self.prices.last_tick_reception_time:
            age = (now_ist() - self.prices.last_tick_reception_time).total_seconds()
            if G_LAST_TICK_AGE_SECONDS: G_LAST_TICK_AGE_SECONDS.set(age)
            if age > 120 and self.config["timings"]["market_open"] < now_ist().time() < self.config["timings"]["market_close"]:
                send_alert(f"🔥 CRITICAL: Stale Feed! No ticks received for {age:.0f} seconds.", "critical")
        elif self.config["timings"]["market_open"] < now_ist().time() < self.config["timings"]["market_close"]:
                 if G_LAST_TICK_AGE_SECONDS: G_LAST_TICK_AGE_SECONDS.set(999)

    def _update_dynamic_equity(self):
        if PAPER_TRADING:
            self.dynamic_account_equity = ACCOUNT_EQUITY + self.trader.daily_realized_pnl
            L.info(f"Paper equity updated to: {self.dynamic_account_equity:,.2f}")
            return
        try:
            margins = self.k.margins()
            if margins and 'equity' in margins and margins['equity'].get('net'):
                self.dynamic_account_equity = float(margins['equity']['net'])
                L.info(f"Dynamic account equity updated to: {self.dynamic_account_equity:,.2f}")
        except Exception as e:
            L.warning(f"Could not update dynamic account equity: {e}")

    def _reconcile_broker_pnl(self):
        """### IMPROVEMENT ### New function to check for P&L drift."""
        if PAPER_TRADING: return
        try:
            broker_positions = self.k.positions()
            if not broker_positions or 'net' not in broker_positions: return

            broker_unrealized_pnl = sum(pos.get('unrealised', 0) for pos in broker_positions['net'] if pos.get('product') == 'MIS')
            bot_unrealized_pnl = self.trader.unrealized_pnl()
            
            discrepancy = abs(broker_unrealized_pnl - bot_unrealized_pnl)
            threshold = self.config['trading'].get('pnl_discrepancy_alert_threshold_rupees', 250.0)

            if discrepancy > threshold:
                send_alert(
                    f"⚠️ P&L DISCREPANCY DETECTED! "
                    f"Broker Unrealized: ₹{broker_unrealized_pnl:.2f}, "
                    f"Bot Unrealized: ₹{bot_unrealized_pnl:.2f}, "
                    f"Difference: ₹{discrepancy:.2f}",
                    "warning"
                )
        except Exception as e:
            L.warning(f"Could not reconcile broker P&L: {e}")


    def _update_pnl_metrics(self):
        self.last_unrealized_pnl = self.trader.unrealized_pnl()

    def _update_prometheus_metrics(self):
        if not G_PNL_REALIZED: return
        try:
            G_PNL_REALIZED.set(self.trader.daily_realized_pnl)
            G_PNL_UNREALIZED.set(self.last_unrealized_pnl)
            with self.engine_lock:
                G_HALTED_STATUS.set(1 if self.halt_trading else 0)
            with self.portfolio_greeks_lock:
                if G_PORTFOLIO_DELTA: G_PORTFOLIO_DELTA.set(self.portfolio_greeks["net_delta"])
                if G_PORTFOLIO_VEGA: G_PORTFOLIO_VEGA.set(self.portfolio_greeks["net_vega"])
                if G_PORTFOLIO_GAMMA: G_PORTFOLIO_GAMMA.set(self.portfolio_greeks["net_gamma"])
                if G_PORTFOLIO_THETA: G_PORTFOLIO_THETA.set(self.portfolio_greeks["net_theta"])
        except Exception as e:
            L.warning(f"Failed to update Prometheus metrics: {e}")

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
                if token is None: continue
                bar_df = self.bars.get_ohlc(token, 1)
                if not bar_df.empty:
                    filepath = os.path.join(DATA_LOG_DIR, filename)
                    bar_df.to_csv(filepath)

            L.info("Bar data persistence complete.")
        except Exception as e:
            L.error(f"Failed to persist bar data: {e}", exc_info=True)

    def _initialize_position_greeks(self, p: Position):
        """Calculates initial greeks for a position once it's filled."""
        # Greeks should already be attached from trade_params, this function just commits them to the portfolio
        with self.portfolio_greeks_lock:
            if not p.greeks:
                L.warning(f"Could not find pre-calculated greeks for {p.tradingsymbol}. This should not happen.")
                return

            self.portfolio_greeks["net_delta"] += p.greeks.get("delta", 0.0) * p.initial_qty
            self.portfolio_greeks["net_vega"] += p.greeks.get("vega", 0.0) * p.initial_qty
            self.portfolio_greeks["net_gamma"] += p.greeks.get("gamma", 0.0) * p.initial_qty
            self.portfolio_greeks["net_theta"] += p.greeks.get("theta", 0.0) * p.initial_qty
            L.info(f"Initialized greeks for {p.tradingsymbol}: {p.greeks}. Portfolio totals: {self.portfolio_greeks}")

    def _update_position_greeks(self, p: Position, tick: Dict):
        """Continuously updates greeks for a position with each new tick."""
        with self.trader.lock, self.portfolio_greeks_lock:
            old_greeks = p.greeks.copy()

            underlying_name = _get_underlying(p.tradingsymbol)
            underlying_token = self.bn_token if "BANKNIFTY" in underlying_name else self.nifty_token
            spot = self.prices.ltp(underlying_token)
            if not spot: return

            opt_details = self.book.df_by_token.loc[p.token]
            T = _calculate_time_to_expiry(opt_details['expiry'].date(), now_ist(), self.config["timings"]["market_close"])

            hv = calculate_historical_volatility(self.get_ohlc(underlying_token, 1)['close'])
            iv = calculate_iv(tick['last_price'], spot, opt_details['strike'], T, 0.05, p.option_type == 'CE', initial_guess=p.greeks.get('iv', 0.5), hv_fallback=hv)
            p.greeks = calculate_greeks(spot, opt_details['strike'], T, 0.05, iv, p.option_type == 'CE')

            for key in ["delta", "vega", "gamma", "theta"]:
                change = p.greeks.get(key, 0.0) - old_greeks.get(key, 0.0)
                self.portfolio_greeks[f"net_{key}"] += change * p.qty

    # ### PILLAR 1 & 2 & 3 - NEW HELPER FUNCTIONS ###
    def _get_time_of_day_risk_multiplier(self) -> float:
        """### PILLAR 3.2 IMPLEMENTATION ###
        Returns a risk multiplier based on the time of day.
        """
        now_time = now_ist().time()
        time_config = self.config["trading"].get("time_of_day_risk", {})
        
        open_start, open_end = dtime.fromisoformat(time_config.get("opening_range", "09:15:00")), dtime.fromisoformat(time_config.get("opening_end", "10:30:00"))
        midday_start, midday_end = dtime.fromisoformat(time_config.get("midday_range", "11:30:00")), dtime.fromisoformat(time_config.get("midday_end", "13:30:00"))
        
        if open_start <= now_time < open_end:
            return time_config.get("opening_multiplier", 1.25)
        elif midday_start <= now_time < midday_end:
            return time_config.get("midday_multiplier", 0.5)
        else:
            return time_config.get("default_multiplier", 1.0)
            
    def _update_tfi_score(self, tick: dict):
        """### PILLAR 1.2 IMPLEMENTATION ###
        Analyzes a single tick to update the Trade Flow Imbalance score.
        """
        token = tick.get('instrument_token')
        price = tick.get('last_price')
        qty = tick.get('last_traded_quantity')
        depth = tick.get('depth')

        if not all([token, price, qty, depth]): return
        if not depth.get('buy') or not depth.get('sell'): return

        if token not in self.recent_trades:
            window = self.config["technical"].get("tfi_window", 50)
            self.recent_trades[token] = deque(maxlen=window)
            self.tfi_scores[token] = 0.0

        bid_price = depth['buy'][0]['price']
        ask_price = depth['sell'][0]['price']
        trade_value = 0

        if price >= ask_price: # Aggressive Buy
            trade_value = qty
        elif price <= bid_price: # Aggressive Sell
            trade_value = -qty
        
        if trade_value != 0:
            self.recent_trades[token].append(trade_value)
            self.tfi_scores[token] = sum(self.recent_trades[token])
            
    def _check_tfi(self, token: int, side: OrderSide) -> bool:
        """### PILLAR 1.2 IMPLEMENTATION ###
        Checks if the TFI score confirms a trade signal.
        """
        score = self.tfi_scores.get(token, 0.0)
        threshold = self.config["technical"].get("tfi_confirmation_threshold", 500)
        
        if side == OrderSide.BUY and score > threshold:
            L.info(f"TFI confirmed BUY for {token}. Score: {score}")
            return True
        if side == OrderSide.SELL and score < -threshold:
            L.info(f"TFI confirmed SELL for {token}. Score: {score}")
            return True
        
        return False

    def _check_order_book_imbalance(self, token: int, side: OrderSide) -> bool:
        """### PILLAR 1.1 IMPLEMENTATION ###
        Analyzes the order book for imbalances to confirm a signal.
        """
        full_tick = self.prices.get_full_tick(token)
        if not full_tick or not full_tick.get('depth'):
            return False

        depth = full_tick['depth']
        bids = depth.get('buy', [])
        asks = depth.get('sell', [])
        
        if not bids or not asks: return False
        
        # Analyze top 20 levels as requested
        total_bid_qty = sum(item['quantity'] for item in bids[:20])
        total_ask_qty = sum(item['quantity'] for item in asks[:20])
        
        if (total_bid_qty + total_ask_qty) == 0: return False
        
        obi_ratio = total_bid_qty / (total_bid_qty + total_ask_qty)
        
        threshold = self.config["technical"].get("obi_confirmation_threshold_pct", 75.0) / 100.0
        
        if side == OrderSide.BUY and obi_ratio > threshold:
            L.info(f"OBI confirmed BUY for {token}. Ratio: {obi_ratio:.2f}")
            return True
        if side == OrderSide.SELL and (1 - obi_ratio) > threshold:
            L.info(f"OBI confirmed SELL for {token}. Ratio: {obi_ratio:.2f} (Ask dominance: {1-obi_ratio:.2f})")
            return True
            
        return False
        
    def _manage_adaptive_entry(self, p: Position, now: datetime):
        """### PILLAR 2 IMPLEMENTATION ###
        Handles the logic for the multi-stage adaptive entry.
        """
        if not p.last_entry_modification: return
        
        time_since_mod = (now - p.last_entry_modification).total_seconds()
        
        full_tick = self.prices.get_full_tick(p.token)
        if not full_tick or not full_tick.get('depth'): return

        depth = full_tick['depth']
        if not depth.get('buy') or not depth.get('sell'): return

        bid_price = depth['buy'][0]['price']
        ask_price = depth['sell'][0]['price']
        mid_price = (bid_price + ask_price) / 2.0
        tick_size = self.book.tick_size(p.tradingsymbol)
        mid_price = round(mid_price / tick_size) * tick_size # Round to nearest tick

        new_price = -1.0

        if p.entry_stage == 1 and time_since_mod > self.config['trading']['adaptive_entry_stage2_ms']/1000.0:
            L.info(f"Adaptive Entry Stage 2 (Neutral) for {p.tradingsymbol}")
            new_price = mid_price
            p.entry_stage = 2
        elif p.entry_stage == 2 and time_since_mod > self.config['trading']['adaptive_entry_stage3_ms']/1000.0:
            L.info(f"Adaptive Entry Stage 3 (Aggressive) for {p.tradingsymbol}")
            new_price = ask_price
            p.entry_stage = 3
        
        if new_price > 0:
            if self.trader.k.modify_order(
                variety=self.trader.k.VARIETY_REGULAR,
                order_id=p.entry_order_id,
                price=new_price
            ):
                p.last_entry_modification = now
                self.store.upsert_position(p)
                L.info(f"Modified entry order {p.entry_order_id} to price {new_price}")
            else:
                L.error(f"Failed to modify entry order {p.entry_order_id} for adaptive entry.")

    def stop(self):
        if self.running.is_set():
            L.info("Disengaging Sentinel..."); self.running.clear()
            if hasattr(self, 'prices') and self.prices.ws:
                self.prices.ws.close()

            L.info("Performing final data persistence before shutdown...")
            self._persist_bar_data()

            if self.last_trading_day:
                self.store.set_kv(f"daily_pnl_{self.last_trading_day}", str(self.trader.daily_realized_pnl))
            send_alert("🛑 Sentinel PRIME disengaged.");
            L.info("Shutdown complete.")

# ==================================================================================================
# APPLICATION ENTRY POINT
# ==================================================================================================
def login_or_reuse(token_path: str) -> Tuple[KiteConnect, str]:
    api_key = os.environ.get("KITE_API_KEY"); api_secret = os.environ.get("KITE_API_SECRET")
    if not api_key or not api_secret:
        raise SystemExit("FATAL: KITE_API_KEY and KITE_API_SECRET must be set in .env file.")

    kite = KiteConnect(api_key=api_key)
    try:
        with open(token_path, 'r') as f:
            access_token = json.load(f)['access_token']
        kite.set_access_token(access_token);
        kite.profile();
        L.info("Reusing existing access token.");
        return kite, access_token
    except (KiteException, FileNotFoundError, KeyError):
        L.info("Starting new login flow (token invalid or not found).");
        print("Login URL:", kite.login_url());
        req_token = input("Paste request_token: ").strip()
        try:
            session = kite.generate_session(req_token, api_secret=api_secret);
            access_token = session["access_token"];
            kite.set_access_token(access_token);
            L.info("New session generated successfully.")

            with open(token_path, 'w') as f:
                json.dump({"access_token": access_token}, f)

            try:
                os.chmod(token_path, 0o600)
                L.info(f"Set secure (600) permissions on {token_path}")
            except Exception as e:
                L.warning(f"Could not set secure file permissions on {token_path}. This is normal on Windows. Error: {e}")

            return kite, access_token
        except KiteException as e:
            raise SystemExit(f"FATAL: Login failed. Could not generate session. Reason: {e}")

def main():
    load_dotenv()
    send_alert(f"Booting Sentinel PRIME Protocol... (Mode: {APP_ENV})")

    if METRICS_APP:
        start_metrics_server(port=CONFIG["trading"].get("metrics_port", 9095))

    engine_instance = None

    def handle_sigint(sig, frame):
        L.warning("SIGINT received. Initiating graceful shutdown...")
        if not engine_instance:
            sys.exit(0)

        with engine_instance.engine_lock:
            engine_instance.halt_trading = True

        with engine_instance.trader.lock:
            active_positions = list(engine_instance.trader.positions.values())

        positions_to_close = [
            p for p in active_positions
            if p.status not in [PositionStatus.CLOSED.value, PositionStatus.PENDING_CLOSURE.value, PositionStatus.REJECTED.value]
        ]

        if positions_to_close:
            L.info(f"Closing {len(positions_to_close)} open position(s) before shutdown...")
            for p in positions_to_close:
                engine_instance.trader.close_position(p, "GRACEFUL_SHUTDOWN")

            L.info("Waiting up to 15 seconds for exit orders to complete...")
            for i in range(15):
                with engine_instance.trader.lock:
                    still_open = any(
                        p.status not in [PositionStatus.CLOSED.value, PositionStatus.REJECTED.value]
                        for p in engine_instance.trader.positions.values()
                        if p.id in [pos.id for pos in positions_to_close]
                    )
                if not still_open:
                    L.info(f"All positions confirmed closed after {i+1} seconds.")
                    break
                time.sleep(1)
            else:
                L.warning("Timeout reached. Some positions may not have confirmed closure.")
        else:
            L.info("No open positions to close.")

        engine_instance.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        try:
            store = Store(DB_PATH)
        except Exception as e:
            L.critical(f"FATAL: Database is unreadable or corrupt on init: {e}. Cannot continue without state. Exiting.")
            send_alert(f"🔥 SENTINEL CRITICAL STARTUP FAILURE: Cannot read state DB. Bot is HALTED.", "critical")
            sys.exit(1)

        token_path = os.path.join(PERSIST_DIR, "access_token.json")
        kite_session, access_token = login_or_reuse(token_path)
        governed_kite = GovernedKite(kite_session)
        book = InstrumentBook(governed_kite, store).load()
        prices = PriceBus(governed_kite, access_token)

        engine_instance = Engine(
            kite=governed_kite,
            store=store,
            book=book,
            prices=prices
        )

        perf_callback = engine_instance._update_performance_metrics

        if PAPER_TRADING:
            trader = PaperTrader(engine_instance, book, prices, store, perf_callback=perf_callback)
        else:
            trader = Trader(engine_instance, governed_kite, store, book, prices, perf_callback=perf_callback)

        engine_instance.set_dependencies(trader)
        engine_instance.start()

    except SystemExit as e:
        send_alert(f"🔥 SENTINEL PRIME STARTUP FAILED: {e}", "critical")
        L.critical(str(e))
        sys.exit(1)
    except Exception as e:
        L.exception("A fatal unhandled error occurred.")
        send_alert(f"🔥 SENTINEL PRIME CRASHED (Unhandled Main Exception): {e}", "critical")
        if engine_instance: engine_instance.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()
