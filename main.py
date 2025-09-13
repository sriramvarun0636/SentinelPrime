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

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pytz

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
    PrometheusMetrics = None
    Gauge = None
    serve = None

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
        
        regime_params = config["strategies"]["regime_classifier"]
        if "adx_trend_entry_threshold" not in regime_params or "adx_trend_exit_threshold" not in regime_params:
            raise ValueError("Config is missing 'adx_trend_entry_threshold' or 'adx_trend_exit_threshold' in regime_classifier")
        
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
    METRICS_APP = Flask(__name__)
    metrics = PrometheusMetrics(METRICS_APP, export_defaults=False)
else:
    G_PNL_REALIZED = G_PNL_UNREALIZED = G_DAILY_DRAWDOWN_PCT = G_HALTED_STATUS = G_WS_CONNECTED = G_LAST_TICK_AGE_SECONDS = G_CURRENT_REGIME = None
    METRICS_APP = None

def start_metrics_server(port: int = 9095):
    """Runs the Waitress server for Flask/Prometheus in a separate daemon thread."""
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
INDEX_TR_SYMBOL = {"BANKNIFTY": "NIFTY BANK", "NIFTY": "NIFTY 50", "INDIA VIX": "INDIA VIX"}

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
    PARTIALLY_FILLED_NO_BRACKETS = "PARTIALLY_FILLED_NO_BRACKETS"
    OPEN_NO_BRACKETS = "OPEN_NO_BRACKETS"
    ACTIVE = "ACTIVE"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    PENDING_CLOSURE = "PENDING_CLOSURE"
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
    is_triggered: bool = False

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
    slm_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    scaled_out_qty: int = 0
    breakeven_armed: bool = False
    trailing_sl_armed: bool = False
    initial_risk_points: float = 0.0
    option_sl_points: float = 0.0
    option_tp_points: float = 0.0
    high_price_since_entry: float = 0.0
    exit_price: Optional[float] = None
    scale_out_rules: List[ScaleOut] = field(default_factory=list)

    def __post_init__(self):
        """Sanitizes numeric types to prevent silent crashes from NumPy types."""
        numeric_fields = [
            'qty', 'initial_qty', 'entry_price', 'initial_sl_price', 
            'sl_price', 'tp_price', 'underlying_sl_level', 'scaled_out_qty',
            'initial_risk_points', 'option_sl_points', 'option_tp_points',
            'high_price_since_entry', 'exit_price'
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

# ==================================================================================================
# UTILITIES & API GOVERNOR
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

### MODIFICATION START ###
# ADDED a more precise time-to-expiry calculation for better greeks.
def _calculate_time_to_expiry(expiry_date: date, current_datetime: datetime, market_close_time: dtime) -> float:
    """Calculates a precise time to expiry in years for intraday greek calculations."""
    if expiry_date < current_datetime.date():
        return 1e-9

    trading_days_per_year = 252.0
    
    # Calculate full trading days remaining
    full_days_left = np.busday_count(current_datetime.date(), expiry_date)
    
    # Calculate fraction of the current day remaining
    market_open_dt = current_datetime.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close_dt = current_datetime.replace(hour=market_close_time.hour, minute=market_close_time.minute, second=0, microsecond=0)
    
    if current_datetime < market_open_dt:
        # Market not yet open, full day is ahead
        day_fraction = 1.0
    elif current_datetime >= market_close_dt:
        # Market closed, no time left in this day
        day_fraction = 0.0
        # If market is closed, busday_count might include today, which it shouldn't
        if expiry_date >= current_datetime.date():
             full_days_left = max(0, full_days_left -1)
    else:
        # Market is open
        total_trading_seconds = (market_close_dt - market_open_dt).total_seconds()
        seconds_left = (market_close_dt - current_datetime).total_seconds()
        day_fraction = max(0.0, seconds_left / total_trading_seconds)
    
    # If today is the expiry day, the day_fraction is the total time
    if full_days_left == 0 and expiry_date == current_datetime.date():
        return max(1e-9, day_fraction / trading_days_per_year)

    # Total time is full days + fraction of today
    total_days = full_days_left + day_fraction
    
    return max(1e-9, total_days / trading_days_per_year)

### MODIFICATION END ###

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 1e-9 or sigma <= 1e-9 or S <= 0 or K <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
    except (ValueError, ZeroDivisionError):
        return max(0.0, S - K) if is_call else max(0.0, K - S)
        
    N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    if is_call:
        price = S * N(d1) - K * math.exp(-r * T) * N(d2)
    else:
        price = K * math.exp(-r * T) * N(-d2) - S * N(-d1)
    return price

def bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 1e-6 or sigma <= 1e-6 or S <= 0 or K <= 0:
        return (1.0 if S > K else 0.0) if is_call else (-1.0 if S < K else 0.0)
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return (0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))) if is_call else (0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0))) - 1.0)
    except Exception:
        return (1.0 if S > K else 0.0) if is_call else (-1.0 if S < K else 0.0)

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
        """Initializes and validates the database with migration and a write/read check."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA user_version")
            db_version = cursor.fetchone()[0]
            L.info(f"Database schema version: {db_version}")
            
            if db_version < 1:
                L.info("Applying migration to version 1...")
                cursor.execute("""
                CREATE TABLE positions (
                    id TEXT PRIMARY KEY, tradingsymbol TEXT, token INTEGER, option_type TEXT, qty INTEGER, initial_qty INTEGER,
                    entry_price REAL, initial_sl_price REAL, sl_price REAL, tp_price REAL, opened_at TEXT,
                    strategy TEXT, market_regime_at_entry TEXT, underlying_sl_level REAL, status TEXT,
                    entry_order_id TEXT, slm_order_id TEXT, tp_order_id TEXT, scaled_out_qty INTEGER DEFAULT 0,
                    breakeven_armed INTEGER DEFAULT 0, trailing_sl_armed INTEGER DEFAULT 0,
                    initial_risk_points REAL DEFAULT 0.0, option_sl_points REAL DEFAULT 0.0, option_tp_points REAL DEFAULT 0.0,
                    high_price_since_entry REAL DEFAULT 0.0, scale_out_rules TEXT DEFAULT '[]'
                )""")
                cursor.execute("""
                    CREATE TABLE trade_log (
                        id TEXT PRIMARY KEY, tradingsymbol TEXT, strategy TEXT, entry_time TEXT, exit_time TEXT,
                        entry_price REAL, exit_price REAL, qty INTEGER, pnl REAL, exit_reason TEXT, market_regime_at_entry TEXT
                    )""")
                cursor.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
                cursor.execute("PRAGMA user_version = 1")
                db_version = 1
                L.info("Migration to version 1 complete.")
            
            if db_version < 2:
                L.info("Applying migration to version 2...")
                try:
                    cursor.execute("ALTER TABLE positions ADD COLUMN exit_order_id TEXT")
                    cursor.execute("ALTER TABLE positions ADD COLUMN exit_reason TEXT")
                    cursor.execute("ALTER TABLE positions ADD COLUMN exit_price REAL")
                except sqlite3.OperationalError as e:
                    L.warning(f"Could not add columns in migration 2, they may already exist: {e}")
                cursor.execute("PRAGMA user_version = 2")
                db_version = 2
                L.info("Migration to version 2 complete.")
            
            try:
                validation_time = datetime.now(IST).isoformat()
                cursor.execute("REPLACE INTO meta (key, value) VALUES ('db_startup_validation', ?)", (validation_time,))
                conn.commit()
                cursor.execute("SELECT value FROM meta WHERE key='db_startup_validation'")
                check = cursor.fetchone()
                if not check or check[0] != validation_time:
                    raise sqlite3.DatabaseError("Failed post-migration validation write/read check.")
                L.info("Database initialization and write/read validation check complete.")
            except Exception as e:
                L.critical(f"FATAL: Database validation check failed: {e}")
                raise

    def upsert_position(self, p: Position):
        sql = """
            INSERT INTO positions (id, tradingsymbol, token, option_type, qty, initial_qty, entry_price, initial_sl_price, sl_price, tp_price, opened_at, strategy, market_regime_at_entry, underlying_sl_level, status, entry_order_id, slm_order_id, tp_order_id, scaled_out_qty, breakeven_armed, trailing_sl_armed, initial_risk_points, option_sl_points, option_tp_points, high_price_since_entry, scale_out_rules, exit_order_id, exit_reason, exit_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                qty=excluded.qty, sl_price=excluded.sl_price, tp_price=excluded.tp_price, status=excluded.status, slm_order_id=excluded.slm_order_id, tp_order_id=excluded.tp_order_id, scaled_out_qty=excluded.scaled_out_qty, breakeven_armed=excluded.breakeven_armed, trailing_sl_armed=excluded.trailing_sl_armed, high_price_since_entry=excluded.high_price_since_entry, scale_out_rules=excluded.scale_out_rules, entry_price=excluded.entry_price, opened_at=excluded.opened_at, entry_order_id=excluded.entry_order_id, exit_order_id=excluded.exit_order_id, exit_reason=excluded.exit_reason, exit_price=excluded.exit_price
        """
        rules_json = json.dumps([s.__dict__ for s in p.scale_out_rules])
        params = (p.id, p.tradingsymbol, p.token, p.option_type, p.qty, p.initial_qty, p.entry_price, p.initial_sl_price, p.sl_price, p.tp_price, p.opened_at.isoformat(), p.strategy, p.market_regime_at_entry, p.underlying_sl_level, p.status, p.entry_order_id, p.slm_order_id, p.tp_order_id, p.scaled_out_qty, int(p.breakeven_armed), int(p.trailing_sl_armed), p.initial_risk_points, p.option_sl_points, p.option_tp_points, p.high_price_since_entry, rules_json, p.exit_order_id, p.exit_reason, p.exit_price)
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
            scale_rules_data = json.loads(r_dict.get("scale_out_rules", '[]'))
            scale_rules = [ScaleOut(**data) for data in scale_rules_data]
            pos = Position(id=r_dict["id"], tradingsymbol=r_dict["tradingsymbol"], token=r_dict["token"], option_type=r_dict["option_type"], qty=r_dict["qty"], initial_qty=r_dict["initial_qty"], entry_price=r_dict["entry_price"], initial_sl_price=r_dict["initial_sl_price"], sl_price=r_dict["sl_price"], tp_price=r_dict["tp_price"], opened_at=datetime.fromisoformat(r_dict["opened_at"]), strategy=r_dict["strategy"], market_regime_at_entry=r_dict["market_regime_at_entry"], underlying_sl_level=r_dict["underlying_sl_level"], status=r_dict["status"], entry_order_id=r_dict["entry_order_id"], slm_order_id=r_dict["slm_order_id"], tp_order_id=r_dict["tp_order_id"], scaled_out_qty=r_dict["scaled_out_qty"], breakeven_armed=bool(r_dict.get("breakeven_armed")), trailing_sl_armed=bool(r_dict.get("trailing_sl_armed")), initial_risk_points=r_dict["initial_risk_points"], option_sl_points=r_dict["option_sl_points"], option_tp_points=r_dict["option_tp_points"], high_price_since_entry=r_dict["high_price_since_entry"], scale_out_rules=scale_rules, exit_order_id=r_dict.get("exit_order_id"), exit_reason=r_dict.get("exit_reason"), exit_price=r_dict.get("exit_price"))
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
        self.path = os.path.join(PERSIST_DIR, "instruments_all.csv")
        self.df_by_token = None
        self.df_by_symbol = None

    def load(self):
        try:
            if not os.path.exists(self.path) or self.store.get_kv("instruments_refreshed") != str(date.today()):
                self.refresh()
            else:
                L.info("Loading instruments from local cache.")
                self.df = pd.read_csv(self.path, parse_dates=["expiry"])
        except Exception as e:
            L.warning(f"Could not load instrument file from cache, attempting refresh: {e}")
            self.refresh()
        
        if self.df is None:
            L.critical("Instrument data failed to load from both API and cache.")
            raise SystemExit("System cannot run without instrument data.")

        self.df_by_token = self.df.set_index('instrument_token')
        self.df_by_symbol = self.df.set_index('tradingsymbol')
        return self

    def refresh(self):
        L.info("Refreshing full instrument list from broker...")
        try:
            instruments = self.kite.instruments()
            if instruments is None:
                raise KiteException("Failed to fetch instruments from API after multiple retries.")
            df = pd.DataFrame(instruments)
            if "expiry" in df.columns:
                df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
            df.to_csv(self.path, index=False)
            self.df = df
            self.store.set_kv("instruments_refreshed", str(date.today()))
            L.info(f"Instrument list refreshed and saved. Rows: {len(df)}")
        except Exception as e:
            L.error(f"API fetch for instruments failed: {e}")
            if os.path.exists(self.path):
                L.warning("Loading instruments from local cache as a fallback.")
                self.df = pd.read_csv(self.path, parse_dates=["expiry"])
            else:
                L.critical("API fetch failed and no instrument cache is available. Cannot continue.")
                self.df = None

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
        df = self.df[(self.df["name"] == name) & (self.df["segment"] == "NFO-OPT") & (self.df["expiry"].dt.date >= today)]
        return df["expiry"].dt.date.min() if not df.empty else None

    def find_option(self, name: str, expiry: date, strike: float, otype: str) -> Optional[Dict]:
        df = self.df[(self.df["name"] == name) & (self.df["expiry"].dt.date == expiry) & (self.df["strike"] == float(strike)) & (self.df["instrument_type"] == otype.upper())]
        return df.iloc[-1].to_dict() if not df.empty else None

    def get_option_chain(self, name: str, expiry: date) -> pd.DataFrame:
        return self.df[(self.df["name"] == name) & (self.df["expiry"].dt.date == expiry) & (self.df["segment"] == "NFO-OPT")]

    def lot_size(self, name: str) -> Optional[int]:
        name_map = {"NIFTY 50": "NIFTY", "NIFTY BANK": "BANKNIFTY"}
        base_name = name_map.get(name, name)
        res = self.df[(self.df.name == base_name) & (self.df.segment == "NFO-OPT")]
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
    def __init__(self, bars: 'Engine', prices: PriceBus, nifty_token: int, bn_token: int, vix_token: Optional[int]):
        self.bars = bars
        self.prices = prices
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
        df['bbw_rank'] = df['bbw'].rolling(self.params["bbw_rank_period"]).rank(pct=True) * 100
        df['ema_fast'] = df.ta.ema(length=20)
        df['ema_slow'] = df.ta.ema(length=50)
        return df

    def _get_scores(self, df: pd.DataFrame, current_regime_enum: Regime) -> Dict[str, int]:
        scores = {"trend_up": 0, "trend_down": 0, "chop": 0, "compression": 0}
        if len(df) < 60: return scores
        adx_col, dmp_col, dmn_col = f'ADX_{self.params["adx_period"]}', f'DMP_{self.params["adx_period"]}', f'DMN_{self.params["adx_period"]}'
        if df['bbw_rank'].iloc[-1] < self.params["compression_rank_threshold"]: scores["compression"] += 2
        if df['bbw_rank'].iloc[-1] > 70: scores["chop"] += 1
        adx_val = df[adx_col].iloc[-1]
        if adx_val < self.params["adx_trend_exit_threshold"]: scores["chop"] += 1
        is_trending = current_regime_enum in [Regime.TRENDING_UP, Regime.TRENDING_DOWN]
        adx_is_trending = adx_val > (self.params["adx_trend_exit_threshold"] if is_trending else self.params["adx_trend_entry_threshold"])
        if adx_is_trending:
            if df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1] and df[dmp_col].iloc[-1] > df[dmn_col].iloc[-1]: scores["trend_up"] += 2
            elif df['ema_fast'].iloc[-1] < df['ema_slow'].iloc[-1] and df[dmn_col].iloc[-1] > df[dmp_col].iloc[-1]: scores["trend_down"] += 2
        return scores

    def get_raw_classification(self, current_regime_enum: Regime) -> Tuple[Regime, Optional[int]]:
        try:
            df_bn = self.bars.get_ohlc(self.bn_token, self.tf)
            df_n = self.bars.get_ohlc(self.nifty_token, self.tf)
            if df_bn.empty or df_n.empty or len(df_bn) < 60 or len(df_n) < 60: return (Regime.UNCLEAR, None)
            df_n, df_bn = self._add_indicators(df_n), self._add_indicators(df_bn)
            score_n, score_bn = self._get_scores(df_n, current_regime_enum), self._get_scores(df_bn, current_regime_enum)
            
            vix_ltp = self.prices.ltp(self.vix_token)
            if vix_ltp and vix_ltp > self.params["vix_chaos_threshold"]: 
                return (Regime.CHAOS, self.bn_token)
                
            if score_n["trend_up"] >= 2 and score_bn["trend_up"] >= 2:
                active_token = self.bn_token if df_bn['close'].pct_change(10).iloc[-1] > df_n['close'].pct_change(10).iloc[-1] else self.nifty_token
                return (Regime.TRENDING_UP, active_token)
                
            if score_n["trend_down"] >= 2 and score_bn["trend_down"] >= 2:
                active_token = self.bn_token if df_bn['close'].pct_change(10).iloc[-1] < df_n['close'].pct_change(10).iloc[-1] else self.nifty_token
                return (Regime.TRENDING_DOWN, active_token)
                
            if score_n["compression"] >= 2 and score_bn["compression"] >= 2:
                active_token = self.bn_token if df_bn['bbw_rank'].iloc[-1] < df_n['bbw_rank'].iloc[-1] else self.nifty_token
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
                 book: InstrumentBook, 
                 prices: PriceBus, 
                 store: Store, 
                 perf_callback: Optional[Callable[[float], None]] = None):
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
    def modify_slm(self, p: Position, new_trigger: float): pass
    @abstractmethod
    def scale_out(self, p: Position, qty_to_close: int) -> bool: pass
    @abstractmethod
    def place_bracket_orders(self, p: Position) -> bool: pass
    @abstractmethod
    def modify_bracket_quantity(self, p: Position, new_qty: int) -> bool: pass

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
        """Calculates and opens a simulated (paper) trade."""
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

                if len(open_positions) == 1 and _get_underlying(opt['tradingsymbol']) != _get_underlying(open_positions[0].tradingsymbol):
                    correlated_lots = max(1, round(initial_lots * CONFIG["trading"]["correlated_risk_reduction_factor"]))
                    qty = int(correlated_lots * lot_size)
                    L.info(f"Applying correlated risk reduction. New quantity: {qty} ({correlated_lots} lots).")

                scale_rules = [ScaleOut(**r) for r in CONFIG['trading']['scale_out_rules']]
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
                    scale_out_rules=scale_rules,
                    option_sl_points=sl_points,
                    option_tp_points=tp_points,
                    entry_order_id=pos_id
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
            
            self._update_performance(pnl) # Call the injected callback

            self.positions.pop(p.id, None)
            send_alert(f"❌ [PAPER] CLOSED {p.tradingsymbol} @ {exit_price:.2f} ({reason}). Final PnL: {pnl:.2f}. Daily PnL: {self.daily_realized_pnl:.2f}")
            return True

    def modify_slm(self, p: Position, new_trigger: float):
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

    def modify_bracket_quantity(self, p: Position, new_qty: int) -> bool:
        L.info(f"[PAPER] Modifying bracket quantity for {p.tradingsymbol} to {new_qty}")
        return True

class Trader(AbstractTrader):
    """Live Trader implementation."""
    def __init__(self, 
                 kite: GovernedKite, 
                 store: Store, 
                 book: InstrumentBook, 
                 prices: PriceBus, 
                 perf_callback: Optional[Callable[[float], None]] = None):
        super().__init__(book, prices, store, perf_callback) 
        self.k = kite
        self.positions = store.load_open_positions()
        
        if self.positions:
            L.info("Reconciling open orders for existing positions on startup...")
            try:
                open_orders = self.k.orders()
                if open_orders is not None:
                    open_order_ids = {str(o['order_id']) for o in open_orders if o['status'] == 'OPEN'}
                    for p in list(self.positions.values()):
                        if p.slm_order_id and str(p.slm_order_id) not in open_order_ids: L.warning(f"SL order {p.slm_order_id} for {p.tradingsymbol} not found open. Clearing."); p.slm_order_id = None
                        if p.tp_order_id and str(p.tp_order_id) not in open_order_ids: L.warning(f"TP order {p.tp_order_id} for {p.tradingsymbol} not found open. Clearing."); p.tp_order_id = None
                        self.store.upsert_position(p)
            except Exception as e: L.error(f"Failed to reconcile open orders on startup: {e}")
        
        for p in self.positions.values(): self.prices.subscribe([p.token])
        
    def _get_order_avg_price(self, oid: str) -> Optional[float]:
        history = self.k.order_history(oid);
        if not history: return None
        trades = [t for t in history if t.get('status') == 'COMPLETE'];
        if not trades: return None
        qty = sum(t['filled_quantity'] for t in trades); return (sum(t['price'] * t['filled_quantity'] for t in trades) / qty) if qty > 0 else None
        
    def _cancel_all_open_orders_for_pos(self, p: Position):
        for oid in [p.slm_order_id, p.tp_order_id]:
            if oid and self.k.cancel_order(self.k.VARIETY_REGULAR, str(oid)) is None: L.warning(f"Could not cancel order {oid} after multiple retries.")
        p.slm_order_id, p.tp_order_id = None, None
        
    def open_position(self, trade_params: Dict) -> Optional[Position]:
        """ "Persist Then Act" implementation for live trading. """
        with self.lock:
            open_pos_count = sum(1 for p in self.positions.values() if p.status not in [PositionStatus.CLOSED.value, PositionStatus.REJECTED.value])
            opt, ts = trade_params['opt'], trade_params['opt']['tradingsymbol']
            lot_size = self.book.lot_size(_get_underlying(ts))

            if open_pos_count >= MAX_CONCURRENT or not lot_size:
                L.warning(f"Trade rejected: Max concurrent ({MAX_CONCURRENT}) or no lot size.")
                return None

            lots = int(trade_params['lots'])
            open_pos = [p for p in self.positions.values() if p.status not in [PositionStatus.CLOSED.value, PositionStatus.REJECTED.value]]
            if len(open_pos) == 1 and _get_underlying(ts) != _get_underlying(open_pos[0].tradingsymbol):
                lots = max(1, round(lots * CONFIG["trading"]["correlated_risk_reduction_factor"]))
            
            qty = int(lots * lot_size) 
            if qty <= 0:
                return None

            temp_id = f"TEMP_{uuid.uuid4()}"
            scale_rules = [ScaleOut(**r) for r in CONFIG['trading']['scale_out_rules']]
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
                scale_out_rules=scale_rules
            )
            self.store.upsert_position(pos)
            self.positions[pos.id] = pos

            ltp = self.prices.ltp(int(opt['instrument_token']))
            if not ltp:
                L.warning(f"Cannot get LTP for {ts}, cannot place entry order.")
                pos.status = PositionStatus.REJECTED.value
                pos.exit_reason = "NO_LTP_FOR_ENTRY"
                self.store.upsert_position(pos)
                return None

            tick_size = self.book.tick_size(ts)
            trigger_price = round((ltp + 2 * tick_size) / tick_size) * tick_size
            order_price = round((trigger_price + 5 * tick_size) / tick_size) * tick_size
            
            oid = self.k.place_order(
                variety=self.k.VARIETY_REGULAR, exchange="NFO", tradingsymbol=ts,
                transaction_type=self.k.TRANSACTION_TYPE_BUY, quantity=qty, product=self.k.PRODUCT_MIS,
                order_type=self.k.ORDER_TYPE_SL, price=order_price, trigger_price=trigger_price
            )

            if oid is None:
                L.error(f"Entry order placement failed for {ts} after multiple retries.")
                pos.status = PositionStatus.REJECTED.value
                pos.exit_reason = "BROKER_API_FAILURE"
                self.store.upsert_position(pos)
                return None
            
            L.info(f"Placed entry STOPLOSS order {oid} for {ts} with trigger @ {trigger_price}")
            
            self.positions.pop(temp_id, None) 
            pos.id = f"LIVE_{oid}" 
            pos.entry_order_id = str(oid)
            pos.status = PositionStatus.PENDING_ENTRY.value
            self.store.upsert_position(pos) 
            self.positions[pos.id] = pos
            return pos

    def place_bracket_orders(self, p: Position) -> bool:
        if p.status not in [PositionStatus.OPEN_NO_BRACKETS.value, PositionStatus.PARTIALLY_FILLED_NO_BRACKETS.value]: return False
        tick_size = self.book.tick_size(p.tradingsymbol); slm_id, tp_id = None, None
        try:
            sl_trigger = round(p.sl_price / tick_size) * tick_size
            slm_id = self.k.place_order(self.k.VARIETY_REGULAR, "NFO", p.tradingsymbol, self.k.TRANSACTION_TYPE_SELL, p.qty, self.k.PRODUCT_MIS, self.k.ORDER_TYPE_SLM, trigger_price=sl_trigger)
            if slm_id is None: raise ValueError("SLM order placement failed")
            p.slm_order_id = str(slm_id)
            
            if p.tp_price > 0:
                tp_limit = round(p.tp_price / tick_size) * tick_size
                tp_id = self.k.place_order(self.k.VARIETY_REGULAR, "NFO", p.tradingsymbol, self.k.TRANSACTION_TYPE_SELL, p.qty, self.k.PRODUCT_MIS, self.k.ORDER_TYPE_LIMIT, price=tp_limit)
                if tp_id is None: raise ValueError("TP order placement failed")
                p.tp_order_id = str(tp_id)
                
            p.status = PositionStatus.ACTIVE.value; self.store.upsert_position(p); L.info(f"Placed Brackets for {p.qty} qty: SL {slm_id} @ {sl_trigger} and TP {tp_id} @ {p.tp_price} for {p.tradingsymbol}"); return True
        except Exception as e:
            L.error(f"Failed to place bracket orders for {p.tradingsymbol}: {e}")
            if slm_id: self.k.cancel_order(self.k.VARIETY_REGULAR, str(slm_id))
            if tp_id: self.k.cancel_order(self.k.VARIETY_REGULAR, str(tp_id))
            return False
            
    def modify_bracket_quantity(self, p: Position, new_qty: int) -> bool:
        L.info(f"Modifying bracket quantity for {p.tradingsymbol} to {new_qty}."); success = True
        if p.slm_order_id:
            if self.k.modify_order(self.k.VARIETY_REGULAR, p.slm_order_id, quantity=new_qty) is None: 
                L.error(f"Failed to modify SLM order {p.slm_order_id} quantity to {new_qty}."); success = False
        if p.tp_order_id:
            if self.k.modify_order(self.k.VARIETY_REGULAR, p.tp_order_id, quantity=new_qty) is None: 
                L.error(f"Failed to modify TP order {p.tp_order_id} quantity to {new_qty}."); success = False
        if not success: 
            send_alert(f"CRITICAL: Failed to modify bracket quantities for {p.tradingsymbol}. Closing position!", "critical"); self.close_position(p, "BRACKET_MODIFY_FAILURE")
        return success
        
    def close_position(self, p: Position, reason: str) -> bool:
        with self.lock:
            if p.status in [PositionStatus.CLOSED.value, PositionStatus.PENDING_CLOSURE.value]: return True
            self._cancel_all_open_orders_for_pos(p)
            oid = self.k.place_order(self.k.VARIETY_REGULAR, "NFO", p.tradingsymbol, self.k.TRANSACTION_TYPE_SELL, p.qty, self.k.PRODUCT_MIS, self.k.ORDER_TYPE_MARKET)
            if oid is None: 
                L.critical(f"MARKET EXIT ORDER FAILED for {p.tradingsymbol}. Manual intervention required!"); 
                send_alert(f"🔥 CRITICAL: FAILED TO PLACE MARKET EXIT for {p.tradingsymbol}. POSITION IS STILL OPEN.", "critical"); 
                return False
            L.info(f"Market exit order {oid} placed for {p.tradingsymbol}. Reason: {reason}. Awaiting confirmation."); 
            p.status = PositionStatus.PENDING_CLOSURE.value; 
            p.exit_order_id = str(oid); 
            p.exit_reason = reason; 
            self.store.upsert_position(p); 
            return True
            
    def scale_out(self, p: Position, qty_to_close: int) -> bool:
        with self.lock:
            if p.status not in [PositionStatus.ACTIVE.value, PositionStatus.PARTIALLY_CLOSED.value] or qty_to_close <= 0 or qty_to_close > p.qty:
                return False

            L.info(f"Scaling out {qty_to_close} of {p.tradingsymbol}")
            
            oid = self.k.place_order(
                self.k.VARIETY_REGULAR, "NFO", p.tradingsymbol, self.k.TRANSACTION_TYPE_SELL,
                qty_to_close, self.k.PRODUCT_MIS, self.k.ORDER_TYPE_MARKET
            )

            if oid is None:
                send_alert(f"CRITICAL: SCALE OUT MARKET ORDER FAILED for {p.tradingsymbol}. Closing full position!", "critical")
                self.close_position(p, "SCALE_OUT_FAILURE")
                return False

            new_qty = p.qty - qty_to_close
            
            if new_qty > 0:
                L.info(f"Partial exit order placed. Modifying bracket orders to new quantity: {new_qty}")
                self.modify_bracket_quantity(p, new_qty)
                p.status = PositionStatus.PARTIALLY_CLOSED.value
            else:
                L.info(f"Position {p.tradingsymbol} will be fully closed via scale-out. Cancelling remaining brackets.")
                self._cancel_all_open_orders_for_pos(p)
                p.status = PositionStatus.PENDING_CLOSURE.value

            p.qty = new_qty
            p.scaled_out_qty += qty_to_close
            self.store.upsert_position(p)
            return True

    def modify_slm(self, p: Position, new_trigger: float):
        tick_size = self.book.tick_size(p.tradingsymbol); new_trigger_rounded = round(new_trigger / tick_size) * tick_size
        if p.slm_order_id and new_trigger_rounded > p.sl_price and abs(new_trigger_rounded - p.sl_price) >= tick_size:
            result = self.k.modify_order(self.k.VARIETY_REGULAR, str(p.slm_order_id), trigger_price=new_trigger_rounded)
            if result: 
                old_sl = p.sl_price; p.sl_price = new_trigger_rounded; self.store.upsert_position(p); 
                L.info(f"Trailing SL for {p.tradingsymbol} from {old_sl:.2f} to {new_trigger_rounded:.2f}")
            else: 
                L.warning(f"Could not modify SL for {p.tradingsymbol} after retries.")

# ==================================================================================================
# STRATEGY DEFINITIONS
# ==================================================================================================
class BaseStrategy(ABC):
    def __init__(self, name: StrategyName, engine: 'Engine', params: Dict):
        self.name = name; self.engine = engine; self.params = params
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
        if vol_ratio > 1.5: final_sl_multiplier = base_sl_multiplier * 1.25; final_tp_multiplier = base_tp_multiplier * 0.75
        elif vol_ratio < 0.7: final_sl_multiplier = base_sl_multiplier * 0.80; final_tp_multiplier = base_tp_multiplier * 1.20
        else: final_sl_multiplier = base_sl_multiplier; final_tp_multiplier = base_tp_multiplier
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
        if vol_ratio > 1.5: final_sl_multiplier = base_sl_multiplier * 1.25; final_tp_multiplier = base_tp_multiplier * 0.75
        elif vol_ratio < 0.7: final_sl_multiplier = base_sl_multiplier * 0.80; final_tp_multiplier = base_tp_multiplier * 1.20
        else: final_sl_multiplier = base_sl_multiplier; final_tp_multiplier = base_tp_multiplier
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
        self.nifty_fut = self.book.find_current_futures_contract("NIFTY")
        self.banknifty_fut = self.book.find_current_futures_contract("BANKNIFTY")
        if not all([self.nifty_fut, self.banknifty_fut]): 
            raise SystemExit("FATAL: Could not find NIFTY/BANKNIFTY futures contracts.")
        
        self.nifty_token = int(self.nifty_fut['instrument_token'])
        self.bn_token = int(self.banknifty_fut['instrument_token'])
        self.vix_token = self.book.get_token(INDEX_TR_SYMBOL["INDIA VIX"])
        
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
        
        self.classifier = RegimeClassifier(self, self.prices, self.nifty_token, self.bn_token, self.vix_token)
        self.strategies: Dict[Regime, List[BaseStrategy]] = self._load_strategies()
        
        self.last_known_prices: Dict[int, float] = {}
        self.config = CONFIG
        self.sanity_check_pct = self.config["trading"].get("insane_tick_pct", 5.0) / 100.0

        ### MODIFICATION START ###
        # DECOUPLING & DEADLOCK FIX: Producer-Consumer Queue and state variables
        self.trade_signal_queue = Queue()
        # This variable is updated by a dedicated task and read by the planner.
        # This prevents the planner from needing to acquire the trader lock.
        self.last_unrealized_pnl = 0.0
        # This event allows the position manager to signal a fatal error without
        # needing to acquire the engine lock, preventing deadlock.
        self.fatal_error_event = threading.Event()
        self.scheduler = self._setup_scheduler()
        ### MODIFICATION END ###

    def set_dependencies(self, trader: AbstractTrader):
        """Injects the trader dependency post-init to resolve circular dependency."""
        self.trader = trader
        if not PAPER_TRADING:
            self.prices.on_order_update_callbacks.append(self.handle_order_update)
            self.prices.on_connect_callbacks.append(self.reconcile)
        L.info("Trader dependency injected into Engine.")

    def get_ohlc(self, token: int, timeframe: int) -> pd.DataFrame:
        return self.bars.get_ohlc(token, timeframe)

    def get_relative_strength_status(self, current_time: datetime) -> str:
        if hasattr(self, '_rs_cache') and hasattr(self, '_rs_cache_time') and (current_time - self._rs_cache_time) < timedelta(minutes=1):
            return self._rs_cache
        try:
            ma_period = self.config["trading"]["intermarket_analysis"]["ratio_ma_period"]
            df_n = self.get_ohlc(self.nifty_token, 1)
            df_bn = self.get_ohlc(self.bn_token, 1)
            if len(df_n) < ma_period or len(df_bn) < ma_period: return "NEUTRAL"
            aligned_n, aligned_bn = df_n['close'].align(df_bn['close'], join='inner')
            if aligned_n.empty: return "NEUTRAL"
            ratio = aligned_bn / aligned_n
            ratio_ma = ratio.rolling(window=ma_period).mean()
            if ratio.iloc[-1] > ratio_ma.iloc[-1] * 1.001: status = "BNF_OUTPERFORMING"
            elif ratio.iloc[-1] < ratio_ma.iloc[-1] * 0.999: status = "NIFTY_OUTPERFORMING"
            else: status = "NEUTRAL"
            self._rs_cache = status; self._rs_cache_time = current_time
            return status
        except Exception as e:
            L.warning(f"Could not calculate relative strength: {e}")
            return "NEUTRAL"

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
                for t in sane_ticks: 
                    self.prices.last[t["instrument_token"]] = t.get("last_price")
                    self.prices.full_ticks[t["instrument_token"]] = t
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
        
        tokens = [self.nifty_token, self.bn_token]
        if self.vix_token: tokens.append(self.vix_token)
        self.prices.subscribe(tokens)
        
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
                # The strategic planner is the main logic, it respects the trading halt.
                # Other tasks like health checks and reconciliation should run regardless.
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

                ### MODIFICATION START ###
                # DEADLOCK FIX: Check for fatal error signal from other threads
                if self.fatal_error_event.is_set():
                    send_alert("🔥 FATAL ERROR EVENT RECEIVED. HALTING ALL TRADING.", "critical")
                    with self.engine_lock:
                        self.halt_trading = True
                    # The event is consumed, logic in risk_ok() will handle closing positions.
                    self.fatal_error_event.clear()
                ### MODIFICATION END ###

                with self.engine_lock:
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
            Regime.COMPRESSION: [MomentumBreakoutStrategy(StrategyName.MOMENTUM_BREAKOUT, self, CONFIG['strategies']['momentum_breakout'])], 
            Regime.TRENDING_UP: [TrendPullbackStrategy(StrategyName.TREND_PULLBACK, self, CONFIG['strategies']['trend_pullback'])], 
            Regime.TRENDING_DOWN: [TrendPullbackStrategy(StrategyName.TREND_PULLBACK, self, CONFIG['strategies']['trend_pullback'])], 
            Regime.CHOP: [MeanReversionStrategy(StrategyName.MEAN_REVERSION, self, CONFIG['strategies']['mean_reversion'])] 
        }

    def _setup_scheduler(self) -> Dict[str, Tuple[Callable, int]]:
        tasks = {
            "strategic_planner": (self._run_strategic_planner, 5),
            "position_management": (self.manage_positions, 10),
            ### MODIFICATION START ###
            # DEADLOCK FIX: New task to safely update PnL for the planner
            "pnl_updater": (self._update_pnl_metrics, 2), # Frequent, lightweight PnL update
            ### MODIFICATION END ###
            "reconciliation": (self.reconcile, 300),
            "health_check": (self.health_check, 60),
            "eod_report": (self._send_eod_report, 300),
            "equity_update": (self._update_dynamic_equity, 600),
            "data_persistence": (self._persist_bar_data, 3600)
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
            if G_HALTED_STATUS: G_HALTED_STATUS.set(0)
            self.last_trade_timestamp = None
            self.active_token = None
            self.active_strategy = None
        
        self.eod_flatten_triggered = False; self.eod_report_sent = False
        self.last_trading_day = now.date()
        self.consecutive_losses = 0; self.risk_factor = 1.0
        self.daily_high_water_mark = self.dynamic_account_equity
        self.performance_score = 0
        send_alert(f"☀️ New Trading Day: {self.last_trading_day}. Equity: ₹{self.dynamic_account_equity:,.2f}")
        
    def warm_up(self):
        L.info("Warming up... Priming historical data."); self._reset_daily_state(); 
        to_date, from_date = self.last_trading_day, self.last_trading_day - timedelta(days=CONFIG["technical"]["warmup_days"])
        
        tokens_to_prime = [self.nifty_token, self.bn_token, self.vix_token]
        
        for token in tokens_to_prime:
            if token is None: 
                continue
            
            symbol = self.book.get_symbol(token) or f"Token {token}"
            L.info(f"Priming historical data for: {symbol}")
            
            hist = self.k.historical_data(token, from_date, to_date, "minute")
            if hist: 
                self.bars.prime(token, pd.DataFrame(hist))
                L.info(f"Successfully primed {len(hist)} bars for {symbol}.")
            else: 
                L.error(f"Failed to prime history for {symbol} after retries.")
        
        pnl_str = self.store.get_kv(f"daily_pnl_{self.last_trading_day}", "0.0"); 
        self.trader.daily_realized_pnl = float(pnl_str)
        self.daily_high_water_mark = self.dynamic_account_equity + self.trader.daily_realized_pnl
        self.weekly_high_water_mark = max(self.weekly_high_water_mark, self.daily_high_water_mark)
        L.info(f"State loaded. Daily PnL restored to: {self.trader.daily_realized_pnl}. Daily HWM: {self.daily_high_water_mark}")
        
    def process_ticks(self, ticks: List[Dict]):
        for tick in ticks:
            # This check is now redundant since the planner runs on a timer, but harmless
            if self.bars.add_tick(tick):
                pass
                
    def _run_strategic_planner(self):
        """The 'thinking' part of the bot. Finds signals and puts them on the queue."""
        with self.engine_lock: # Acquires engine_lock (A) and nothing else.
            now = now_ist()
            
            if now.date() > self.last_trading_day:
                self._reset_daily_state()

            final_entry_time = self.config["timings"]["final_entry_time"]
            if now.weekday() in [2, 3] and self.config["timings"].get("final_expiry_entry_time"):
                final_entry_time = self.config["timings"]["final_expiry_entry_time"]
            
            if (self.halt_trading or 
                not (self.config["timings"]["market_settling_time"] <= now.time() < final_entry_time)):
                return
            
            cooldown_ok = not self.last_trade_timestamp or (now - self.last_trade_timestamp) > timedelta(minutes=self.config["trading"]["trade_cooldown_minutes"])
            if not cooldown_ok:
                return

            self.run_regime_classification(now)

            strategies_for_regime = self.strategies.get(self.regime)
            if not strategies_for_regime:
                return

            all_valid_signals: List[Tuple[int, TradeSignal]] = []
            for token in [self.nifty_token, self.bn_token]:
                for strategy in strategies_for_regime:
                    if signal := strategy.evaluate(token, self.regime, now):
                        all_valid_signals.append((token, signal))
            
            if not all_valid_signals:
                return
            
            filtered_signals = self._filter_signals(all_valid_signals, now)
            if not filtered_signals:
                return
            
            best_signal_tuple = filtered_signals[0]

            if best_signal_tuple:
                best_token, best_signal = best_signal_tuple
                
                # risk_ok is now safe to call from within this lock
                if not self.risk_ok():
                    L.warning(f"--- Trade blocked by master risk controls for {best_signal.strategy_name.value}. ---")
                    return
                
                strategy_obj = next((s for s_list in self.strategies.values() for s in s_list if s.name == best_signal.strategy_name), None)
                if not strategy_obj: 
                    L.error(f"Could not find strategy object for {best_signal.strategy_name}. Aborting trade.")
                    return

                trade_params = self.get_trade_params(
                    token=best_token, side=best_signal.side,
                    risk_points_on_underlying=best_signal.risk_points,
                    reward_points_on_underlying=best_signal.reward_points,
                    strategy=best_signal.strategy_name.value, regime=self.regime.name,
                    target_delta=strategy_obj.params["target_delta"]
                )

                if trade_params and trade_params['lots'] > 0:
                    L.info(f"Planner approved signal for {best_signal.strategy_name.value}. Placing on execution queue.")
                    self.trade_signal_queue.put((best_token, best_signal, trade_params))

    def _trade_executor_worker(self):
        """Waits for a validated trade signal and executes it."""
        L.info("Trade executor worker started.")
        while self.running.is_set():
            try:
                # This call blocks until a signal is available
                token, signal, trade_params = self.trade_signal_queue.get(timeout=1)
                
                L.info(f"Executor received signal for {signal.strategy_name.value}. Executing trade.")
                # This worker acquires trader.lock (B) and nothing else, preventing deadlock.
                if self.trader.open_position(trade_params):
                    with self.engine_lock: # Briefly lock engine state ONLY to update timestamp
                        self.last_trade_timestamp = now_ist()

            except Empty:
                continue
            except Exception as e:
                L.error(f"FATAL Error in trade executor worker: {e}", exc_info=True)

    def _filter_signals(self, signals: List[Tuple[int, TradeSignal]], current_time: datetime) -> List[Tuple[int, TradeSignal]]:
        if not CONFIG["trading"]["intermarket_analysis"]["enabled"]: return signals
        rs_status = self.get_relative_strength_status(current_time)
        final_signals = []
        for token, signal in signals:
            underlying_name = _get_underlying(self.book.get_symbol(token))
            if underlying_name == "BANKNIFTY" and signal.side == OrderSide.BUY and rs_status == "NIFTY_OUTPERFORMING": continue
            elif underlying_name == "NIFTY" and signal.side == OrderSide.BUY and rs_status == "BNF_OUTPERFORMING": continue
            final_signals.append((token, signal))
        if len(signals) > 0 and len(final_signals) == 0: L.info("All signals filtered by Inter-Market Analysis.")
        return final_signals

    def handle_order_update(self, order: Dict):
        """Callback for WebSocket order updates (Live Trading Only)."""
        oid, status = order.get('order_id'), order.get('status')
        pos_id = f"LIVE_{oid}"
        with self.trader.lock:
            pos = self.trader.positions.get(pos_id)
            if not pos:
                pos = next((p for p in self.trader.positions.values() if oid in [p.slm_order_id, p.tp_order_id, p.exit_order_id]), None)
                if not pos: return 

            # Entry Order Logic
            if oid == pos.entry_order_id and pos.status in [PositionStatus.PENDING_ENTRY.value, PositionStatus.PARTIALLY_FILLED_NO_BRACKETS.value]:
                filled_qty = order.get('filled_quantity', 0)
                if filled_qty > 0 and pos.status == PositionStatus.PENDING_ENTRY.value:
                    L.info(f"✅ FIRST PARTIAL FILL for {pos.tradingsymbol}. Qty: {filled_qty}."); 
                    avg_price = self._get_order_avg_price(oid)
                    if not avg_price: 
                        send_alert(f"Entry order {oid} for {pos.tradingsymbol} has no avg price. Removing.", "critical"); 
                        self.trader.positions.pop(pos.id, None); return
                    
                    pos.entry_price = avg_price; pos.qty = filled_qty; pos.high_price_since_entry = avg_price; pos.opened_at = now_ist(); 
                    pos.initial_sl_price = avg_price - pos.option_sl_points; pos.sl_price = pos.initial_sl_price; pos.tp_price = avg_price + pos.option_tp_points; 
                    pos.status = PositionStatus.PARTIALLY_FILLED_NO_BRACKETS.value; self.store.upsert_position(pos)
                    
                    if not self.trader.place_bracket_orders(pos): 
                        self.trader.close_position(pos, "BRACKET_PLACEMENT_FAILURE_ON_FILL")
                
                elif filled_qty > pos.qty:
                    L.info(f"✅ SUBSEQUENT FILL for {pos.tradingsymbol}. New total Qty: {filled_qty}."); 
                    self.trader.modify_bracket_quantity(pos, new_qty=filled_qty); 
                    pos.qty = filled_qty; self.store.upsert_position(pos)
                
                if status in ['COMPLETE', 'CANCELLED', 'REJECTED']:
                    if pos.qty == 0: L.warning(f"Entry order {oid} for {pos.tradingsymbol} {status} with no fills. Removing."); self.trader.positions.pop(pos.id, None)
                    else: L.info(f"Entry order for {pos.tradingsymbol} is final. Total filled: {pos.qty}/{pos.initial_qty}."); pos.status = PositionStatus.ACTIVE.value; self.store.upsert_position(pos)
            
            # Exit Order Logic
            elif (oid in [pos.slm_order_id, pos.tp_order_id, pos.exit_order_id]) and (status == 'COMPLETE') and (pos.status != PositionStatus.CLOSED.value):
                reason = pos.exit_reason or ("SL_HIT_WS" if oid == pos.slm_order_id else "TP_HIT_WS"); 
                L.info(f"Exit order {oid} ({reason}) complete. Cancelling all other open orders for {pos.id}.")
                self.trader._cancel_all_open_orders_for_pos(pos)
                exit_price = self._get_order_avg_price(oid) or self.prices.ltp(pos.token) or pos.sl_price
                pnl = (exit_price - pos.entry_price) * pos.initial_qty; 
                self.trader.daily_realized_pnl += pnl; 
                pos.status = PositionStatus.CLOSED.value
                pos.exit_price = exit_price
                self.store.upsert_position(pos); 
                self.store.log_closed_trade(pos, exit_price, reason); 
                self._update_performance_metrics(pnl) 
                self.trader.positions.pop(pos.id, None)
                send_alert(f"❌ CLOSED {pos.tradingsymbol} ({reason}). Final PnL: {pnl:.2f}. Daily PnL: {self.trader.daily_realized_pnl:.2f}")

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
                self.active_token = active_token
                
                if G_CURRENT_REGIME:
                    try:
                        G_CURRENT_REGIME.clear(); G_CURRENT_REGIME.labels(regime_name=self.regime.name).set(self.regime.value)
                    except Exception as e: L.warning(f"Failed to set Prometheus regime gauge: {e}")
                
                self.last_regime_change_time = current_time
                L.info(f"REGIME SHIFT CONFIRMED: {old_regime_name} -> {self.regime.name} | Active Token: {self.active_token}")
                send_alert(f"REGIME SHIFT: {old_regime_name} -> {self.regime.name}")

    def _find_best_option_contract(self, underlying_token: int, expiry: date, option_type: OptionType, target_delta: float, strategy: str) -> Optional[Dict]:
        spot = self.prices.ltp(underlying_token)
        if not spot: return None
        underlying_symbol = self.book.get_symbol(underlying_token); underlying_name = _get_underlying(underlying_symbol)
        chain = self.book.get_option_chain(underlying_name, expiry); chain = chain[chain['instrument_type'] == option_type.value].copy()
        step = self.book.step_size(underlying_name); atm_strike = round(spot / step) * step
        search_range = 15 * step; chain = chain[(chain['strike'] >= atm_strike - search_range) & (chain['strike'] <= atm_strike + search_range)]
        if chain.empty: return None
        tokens_to_check = chain['instrument_token'].tolist(); quotes = self.k.quote(tokens_to_check)
        if not quotes: L.error(f"Failed to fetch quotes for option chain: {underlying_name}"); return None
        
        strategy_key_map = {
            StrategyName.MOMENTUM_BREAKOUT.value: "momentum_breakout",
            StrategyName.TREND_PULLBACK.value: "trend_pullback",
            StrategyName.MEAN_REVERSION.value: "mean_reversion"
        }
        strategy_config_key = strategy_key_map.get(strategy)
        
        max_iv_for_strategy = None
        if strategy_config_key:
            strategy_params = CONFIG['strategies'].get(strategy_config_key, {})
            max_iv_for_strategy = strategy_params.get('max_iv_entry') 

        ### MODIFICATION START ###
        # Using precise time-to-expiry calculation
        now = now_ist()
        market_close_time = self.config["timings"]["market_close"]
        T = _calculate_time_to_expiry(expiry.date() if isinstance(expiry, pd.Timestamp) else expiry, now, market_close_time)
        ### MODIFICATION END ###
        
        options_with_metrics = []
        for _, row in chain.iterrows():
            tick = quotes.get(str(row['instrument_token']));
            if not tick: continue
            ltp = tick.get('last_price');
            if not ltp or ltp < CONFIG['trading']['min_option_price']: continue
            depth = tick.get('depth');
            if not depth or not depth.get('buy') or not depth.get('sell'): continue
            bid_price, ask_price = depth['buy'][0]['price'], depth['sell'][0]['price']; spread = (ask_price - bid_price) / ask_price if ask_price > 0 else float('inf')
            if spread > CONFIG['trading']['max_bid_ask_spread_pct'] / 100.0: continue
            iv = tick.get('iv', 0.2);
            if iv <= 0: continue

            if max_iv_for_strategy is not None and (iv * 100) > max_iv_for_strategy:
                L.warning(f"Trade REJECTED ({strategy}) on {row['tradingsymbol']}. Current IV ({iv*100:.1f}%) exceeds strategy max ({max_iv_for_strategy:.1f}%)")
                continue

            delta = bs_delta(spot, row['strike'], T, 0.05, iv, option_type == OptionType.CE)
            options_with_metrics.append({'delta_diff': abs(abs(delta) - target_delta), 'opt': row.to_dict(), 'ltp': ltp, 'delta': delta})
        
        if not options_with_metrics: return None
        return min(options_with_metrics, key=lambda x: x['delta_diff'])
        
    def get_trade_params(
        self,
        token: int,
        side: OrderSide,
        risk_points_on_underlying: float,
        reward_points_on_underlying: float,
        strategy: str,
        regime: str,
        target_delta: float
    ) -> Optional[Dict]:
        underlying_symbol = self.book.get_symbol(token)
        if not underlying_symbol: return None
            
        underlying_name = _get_underlying(underlying_symbol)
        lot_size = self.book.lot_size(underlying_name)
        expiry = self.book.find_nearest_expiry_date(underlying_name)

        if not all([lot_size, expiry]): return None
        
        option_type = OptionType.CE if side == OrderSide.BUY else OptionType.PE
        
        best_option_data = self._find_best_option_contract(
            underlying_token=token, expiry=expiry, option_type=option_type,
            target_delta=target_delta, strategy=strategy
        )

        if not best_option_data: return None
        
        option_contract = best_option_data['opt']
        option_ltp = best_option_data['ltp']
        estimated_delta = best_option_data['delta']

        if expiry == date.today() and CONFIG["trading"].get("expiry_day_protocol_active", True):
             expiry_day_delta_assumption = CONFIG["trading"].get("expiry_day_delta_assumption", 0.85)
             L.warning(f"EXPIRY DAY PROTOCOL: Using fixed delta assumption ({expiry_day_delta_assumption}) instead of calculated ({estimated_delta:.2f}) for risk.")
             estimated_delta = expiry_day_delta_assumption if side == OrderSide.BUY else -expiry_day_delta_assumption

        final_sl_points_on_option = risk_points_on_underlying * abs(estimated_delta)
        max_sl_pct = CONFIG["trading"]["max_sl_pct_of_premium"]
        
        if (option_ltp > 0) and (final_sl_points_on_option / option_ltp) > (max_sl_pct / 100.0):
            L.warning(f"Trade REJECTED: Calculated SL ({final_sl_points_on_option:.2f}) exceeds max {max_sl_pct}% of premium ({option_ltp}).")
            return None
        
        risk_per_lot = final_sl_points_on_option * lot_size
        number_of_lots = self.calculate_position_size(token, risk_per_lot)

        if number_of_lots <= 0: return None
            
        spot_price = self.prices.ltp(token)
        if not spot_price: return None

        underlying_sl_level = (spot_price - risk_points_on_underlying) if side == OrderSide.BUY else (spot_price + risk_points_on_underlying)
        final_tp_points_on_option = reward_points_on_underlying * abs(estimated_delta)
        total_trade_risk = risk_per_lot * number_of_lots
        
        return {
            "opt": option_contract, "ltp_opt": option_ltp, "lots": number_of_lots,
            "strategy": strategy, "regime": regime, "option_sl_points": final_sl_points_on_option,
            "option_tp_points": final_tp_points_on_option, "total_trade_risk": total_trade_risk,
            "underlying_sl": underlying_sl_level
        }

    def calculate_position_size(self, underlying_token: int, risk_per_lot: float) -> int:
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
            
        final_risk_factor = self.risk_factor * vol_adjustment * vix_risk_factor * expiry_day_risk_factor
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
            # Create a copy to iterate over, preventing modification issues.
            active_positions = list(self.trader.positions.values())
        
        if not active_positions: 
            return
        
        for p in active_positions:
            if p.status in [PositionStatus.OPEN_NO_BRACKETS.value, PositionStatus.PARTIALLY_FILLED_NO_BRACKETS.value]:
                if not self.trader.place_bracket_orders(p):
                    send_alert(f"CRITICAL: FAILED to place brackets for {p.tradingsymbol}. Closing position.", "critical")
                    closed_successfully = False
                    for attempt in range(3):
                        if self.trader.close_position(p, "BRACKET_PLACEMENT_FAILURE"): 
                            closed_successfully = True; break
                        time.sleep(2)
                    
                    ### MODIFICATION START ###
                    # DEADLOCK FIX: Use a thread-safe event to signal fatal error
                    # instead of acquiring the engine lock directly.
                    if not closed_successfully:
                        send_alert(f"🔥 FATAL: UNABLE TO CLOSE {p.tradingsymbol}. HALTING ALL TRADING.", "critical")
                        self.fatal_error_event.set()
                    ### MODIFICATION END ###
                continue # Move to the next position

            if p.status not in [PositionStatus.ACTIVE.value, PositionStatus.PARTIALLY_CLOSED.value]: 
                continue
                
            ltp = self.prices.ltp(p.token)
            if not ltp: continue
            
            with self.trader.lock:
                p.high_price_since_entry = max(p.high_price_since_entry, ltp)
            
            underlying_name = _get_underlying(p.tradingsymbol)
            underlying_token = self.bn_token if "BANKNIFTY" in underlying_name else self.nifty_token
            underlying_ltp = self.prices.ltp(underlying_token)

            if underlying_ltp and p.underlying_sl_level:
                is_call = (p.option_type == 'CE')
                sl_breached = (is_call and underlying_ltp <= p.underlying_sl_level) or \
                              (not is_call and underlying_ltp >= p.underlying_sl_level)
                
                if sl_breached: 
                    L.warning(f"UNDERLYING SL HIT for {p.tradingsymbol} ({underlying_ltp=}, SL={p.underlying_sl_level}). Closing."); 
                    self.trader.close_position(p, "UNDERLYING_SL_HIT"); 
                    continue
            
            if isinstance(self.trader, PaperTrader):
                if ltp <= p.sl_price: self.trader.close_position(p, "SL_HIT_PAPER"); continue
                if p.tp_price > 0 and ltp >= p.tp_price: self.trader.close_position(p, "TP_HIT_PAPER"); continue
            
            trade_duration_mins = (now_ist() - p.opened_at).total_seconds() / 60
            if trade_duration_mins > CONFIG["trading"].get("max_trade_duration_minutes", 90): 
                L.info(f"Position {p.tradingsymbol} exceeded max duration. Closing."); 
                self.trader.close_position(p, "TIME_STOP_EXIT"); continue
            
            profit_points = ltp - p.entry_price
            
            should_scale_out = False
            qty_to_close = 0
            triggered_rule = None
            
            with self.trader.lock:
                for rule in p.scale_out_rules:
                    if not rule.is_triggered and profit_points >= p.initial_risk_points * rule.rr_target:
                        qty_to_close = int(p.initial_qty * (rule.pct_to_close / 100.0))
                        should_scale_out = True
                        triggered_rule = rule
                        break

            if should_scale_out:
                if self.trader.scale_out(p, qty_to_close):
                    send_alert(f"💰 Scaled Out {qty_to_close} of {p.tradingsymbol} at {ltp:.2f} (RR: {triggered_rule.rr_target}x)")
                    with self.trader.lock:
                       triggered_rule.is_triggered = True
                continue
            
            if p.status in [PositionStatus.CLOSED.value, PositionStatus.PENDING_CLOSURE.value]: continue
            
            with self.trader.lock:
                breakeven_armed = p.breakeven_armed
                trailing_sl_armed = p.trailing_sl_armed
                initial_risk_points = p.initial_risk_points
                sl_price = p.sl_price

            if not breakeven_armed and profit_points >= initial_risk_points * CONFIG['trading']['breakeven_trigger_rr']:
                L.info(f"Arming Breakeven for {p.tradingsymbol}"); 
                self.trader.modify_slm(p, p.entry_price + self.book.tick_size(p.tradingsymbol)); 
                with self.trader.lock:
                    p.breakeven_armed = True
            
            if not trailing_sl_armed and profit_points >= initial_risk_points * CONFIG['trading']['trailing_sl_activation_rr']:
                with self.trader.lock:
                    p.trailing_sl_armed = True; 
                L.info(f"Trailing SL armed for {p.tradingsymbol} after reaching {CONFIG['trading']['trailing_sl_activation_rr']}R.")
            
            if trailing_sl_armed:
                calculated_new_sl = self._calculate_trailing_stop(p)
                if calculated_new_sl > sl_price:
                    self.trader.modify_slm(p, calculated_new_sl)
            
            self.store.upsert_position(p)

    def _calculate_trailing_stop(self, p: Position) -> float:
        """Centralized logic for calculating the new theoretical trailing stop price."""
        underlying_name = _get_underlying(p.tradingsymbol)
        underlying_token = self.bn_token if "BANKNIFTY" in underlying_name else self.nifty_token
        df_underlying_full = self.bars.get_ohlc(underlying_token, 1)
        if df_underlying_full.empty: return p.sl_price
        
        try:
            entry_time_floor = p.opened_at.replace(second=0, microsecond=0)
            underlying_price_at_entry = df_underlying_full.loc[entry_time_floor, 'close']
            high_price_underlying_since_entry = df_underlying_full.loc[entry_time_floor:]['high'].max()
            low_price_underlying_since_entry = df_underlying_full.loc[entry_time_floor:]['low'].min()
        except KeyError: 
            return p.sl_price
        
        first_scale_out_triggered = any(rule.is_triggered for rule in p.scale_out_rules)
        trail_tf = CONFIG["trading"].get("trailing_sl_timeframe_scaled_out", 3) if first_scale_out_triggered else CONFIG["trading"]["trailing_sl_timeframe"]
        trail_multiplier = CONFIG["trading"].get("trailing_sl_chandelier_multiplier_scaled_out", 2.0) if first_scale_out_triggered else CONFIG["trading"]["trailing_sl_chandelier_multiplier"]
        
        df_trail_underlying = df_underlying_full.resample(f'{trail_tf}min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(df_trail_underlying) < CONFIG["trading"]["trailing_sl_chandelier_period"]:
            return p.sl_price
            
        atr_underlying = df_trail_underlying.ta.atr(length=CONFIG["trading"]["trailing_sl_chandelier_period"]).iloc[-1]
        if pd.isna(atr_underlying): return p.sl_price
        
        if p.option_type == 'CE':
            chandelier_exit_underlying = high_price_underlying_since_entry - (atr_underlying * trail_multiplier)
            underlying_move_since_entry = chandelier_exit_underlying - underlying_price_at_entry
        else: # PE
            chandelier_exit_underlying = low_price_underlying_since_entry + (atr_underlying * trail_multiplier)
            underlying_move_since_entry = underlying_price_at_entry - chandelier_exit_underlying

        current_delta_estimate = abs(p.option_sl_points / p.initial_risk_points) if p.initial_risk_points > 0 else 0.5
        
        new_sl_price_from_underlying = p.entry_price + (underlying_move_since_entry * current_delta_estimate)
        
        new_sl_target = max(p.sl_price, new_sl_price_from_underlying)
        if p.breakeven_armed:
            new_sl_target = max(new_sl_target, p.entry_price + self.book.tick_size(p.tradingsymbol))

        return new_sl_target
            
    def risk_ok(self) -> bool:
        """Master risk check. NOTE: This is called from within engine_lock context."""
        if self.halt_trading:
            return False
        
        ### MODIFICATION START ###
        # DEADLOCK FIX: Use the cached unrealized PnL value instead of calculating it here.
        current_equity = self.dynamic_account_equity + self.trader.daily_realized_pnl + self.last_unrealized_pnl
        ### MODIFICATION END ###

        weekly_drawdown_limit = self.weekly_high_water_mark * (CONFIG["trading"].get("weekly_drawdown_pct_limit", 8.0) / 100.0)
        if not self.in_weekly_drawdown_lock and current_equity < (self.weekly_high_water_mark - weekly_drawdown_limit):
            self.in_weekly_drawdown_lock = True
            send_alert(f"🔒 WEEKLY DD LOCK ENGAGED. Peak: {self.weekly_high_water_mark:.2f}, Current: {current_equity:.2f}. Switching to DEFENSIVE mode.", "warning")
        
        daily_drawdown_limit = self.daily_high_water_mark * (MAX_DAILY_DRAWDOWN_PCT / 100.0)
        daily_dd_pct = 0.0
        if self.daily_high_water_mark > 0:
             daily_dd_pct = (self.daily_high_water_mark - current_equity) / self.daily_high_water_mark * 100
        
        if G_DAILY_DRAWDOWN_PCT: 
            G_DAILY_DRAWDOWN_PCT.set(daily_dd_pct)

        if current_equity < (self.daily_high_water_mark - daily_drawdown_limit):
            send_alert(f"⛔ DD LIMIT HIT. HALTING. Peak Equity: {self.daily_high_water_mark:.2f}, Current: {current_equity:.2f} (DD: {daily_dd_pct:.2f}%)", "critical")
            self.halt_trading = True
            if G_HALTED_STATUS: G_HALTED_STATUS.set(1)
            
            with self.trader.lock:
                positions_to_close = [
                    p for p in list(self.trader.positions.values())
                    if p.status not in [PositionStatus.CLOSED.value, PositionStatus.PENDING_CLOSURE.value]
                ]
            for p in positions_to_close:
                self.trader.close_position(p, "DD_LIMIT_HIT")
            
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
        """Checks for both WS connection AND stale data feed."""
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

    ### MODIFICATION START ###
    # DEADLOCK FIX: New method to safely calculate and cache unrealized PnL.
    def _update_pnl_metrics(self):
        """
        Safely calculates unrealized PnL and caches it for the engine.
        This is the only task that should call trader.unrealized_pnl() directly.
        """
        self.last_unrealized_pnl = self.trader.unrealized_pnl()
    ### MODIFICATION END ###

    def _update_prometheus_metrics(self):
        """Task to update all Prometheus gauges periodically."""
        if not G_PNL_REALIZED: return 
        try:
            G_PNL_REALIZED.set(self.trader.daily_realized_pnl)
            # Use the cached value for consistency
            G_PNL_UNREALIZED.set(self.last_unrealized_pnl)
            with self.engine_lock:
                G_HALTED_STATUS.set(1 if self.halt_trading else 0)
        except Exception as e:
            L.warning(f"Failed to update Prometheus metrics: {e}")

    def _persist_bar_data(self):
        """Periodically saves the in-memory 1-min bars to daily log files."""
        try:
            L.info("Persisting in-memory 1-minute bars to disk...")
            os.makedirs(DATA_LOG_DIR, exist_ok=True)
            today_str = date.today().isoformat()

            tokens_to_log = {
                self.nifty_token: f"nifty_1min_{today_str}.csv",
                self.bn_token: f"banknifty_1min_{today_str}.csv",
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
    """Main composition root. Creates and injects all dependencies."""
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
            trader = PaperTrader(book, prices, store, perf_callback=perf_callback)
        else:
            trader = Trader(governed_kite, store, book, prices, perf_callback=perf_callback) 

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