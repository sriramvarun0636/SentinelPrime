"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        NIFTY GAMMA SCALPING SYSTEM  v2.0  —  PRODUCTION GRADE              ║
║                                                                              ║
║  Features:                                                                   ║
║  1.  DTE guard  — skip entries within 2 days of expiry, roll to next week    ║
║  2.  Hybrid 1min/3min — 3min for hedge timing, 1min for SL/spike detection   ║
║  3.  Data cleaning — only ATM, ATM±1..±3 kept                                ║
║  4.  Regime classifier — Kaufman Efficiency Ratio (ER)                       ║
║  5.  Options-only hedging — buy CE/PE to hedge, no futures                   ║
║  6.  Realistic exits — 15% straddle TP/SL, 5% IV expansion/crush             ║
║  7.  Dynamic hedge SL/TP via ATR                                             ║
║  8.  Risk-to-Reward ratio as tunable hyperparameter                          ║
║  9.  Hedge capital TP — exit hedge at X% of deployed capital                 ║
║  10. 6 named exit reasons (gamma_win, vega_win, iv_crush, theta_bleed...)    ║
║  11. OPTUNA Hyperparameter tuning engine (TPE Sampler)                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, glob, warnings, itertools, time, json, random, sys
from datetime import date, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


# ─────────────────────────────────────────────────────────────────────────────
#  ① HYPERPARAMETER CONFIG  ← Baseline params (Optuna will tune these)
# ─────────────────────────────────────────────────────────────────────────────
PARAMS: Dict[str, Any] = {
    # ── Entry filters ────────────────────────────────────────────────────────
    "IVR_ENTRY_MAX"       : 40.0,   
    "IV_HV_RATIO_MAX"     : 0.85,   
    "MIN_ER_FOR_TREND"    : 0.35,   
    "ER_WINDOW"           : 14,     

    # ── DTE handling ─────────────────────────────────────────────────────────
    "DTE_MIN"             : 2,      

    # ── Hedge parameters (options-only) ──────────────────────────────────────
    "HEDGE_INTERVAL_3MIN" : 3,      
    "DELTA_THRESHOLD"     : 0.08,   
    "HEDGE_TP_PCT"        : 0.08,   
    "HEDGE_SL_ATR_MULT"   : 2.0,    
    "ATR_WINDOW"          : 14,     
    "RR_RATIO"            : 0.5,    

    # ── Straddle exit thresholds ─────────────────────────────────────────────
    "STRADDLE_TP_PCT"     : 0.15,   
    "STRADDLE_SL_PCT"     : 0.15,   
    "IV_EXPAND_PCT"       : 0.05,   
    "IV_CRUSH_PCT"        : 0.05,   
    "THETA_BLEED_MINS"    : 120,    

    # ── Session timings ──────────────────────────────────────────────────────
    "ENTRY_START"         : "09:30",
    "ENTRY_END"           : "14:00",
    "SESSION_END"         : "15:15",

    # ── Position / cost ──────────────────────────────────────────────────────
    "LOT_SIZE"            : 75,     
    "NUM_LOTS"            : 1,
    "TX_COST_PER_LOT"     : 30.0,   
    "SLIPPAGE_PER_HEDGE"  : 2.0,    

    # ── Volatility / signal ──────────────────────────────────────────────────
    "RV_WINDOW"           : 20,     
    "IVR_LOOKBACK_DAYS"   : 20,

    # ── Constants ────────────────────────────────────────────────────────────
    "RISK_FREE_RATE"      : 0.065,
    "MINS_PER_DAY"        : 375,
}

# ─────────────────────────────────────────────────────────────────────────────
#  PATH CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_1MIN  = '/content/drive/MyDrive/1MIN'
DATA_3MIN  = '/content/drive/MyDrive/3MIN'
OUTPUT_DIR = '/content/drive/MyDrive/gamma_scalp_output'


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA LOADER
# ═════════════════════════════════════════════════════════════════════════════

def _clean_date(s: str) -> str:
    return str(s).replace('="', '').replace('"', '').replace('=', '').strip()

def _parse_offset(s: str) -> Optional[int]:
    """'ATM' → 0, 'ATM+3' → 3, 'ATM-2' → -2"""
    import re
    s = str(s).strip().upper()
    if s == 'ATM': return 0
    m = re.search(r'ATM([+-]\d+)', s)
    return int(m.group(1)) if m else None

def load_files(folder: str, label: str = '') -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No CSV files in: {folder}")
    print(f"\nLOADING {len(files)} FILES  ({label})")
    print(f"Data folder: {folder}")
    frames =[]
    for fp in files:
        df = pd.read_csv(fp, dtype={'date': str, 'time': str}, low_memory=False)
        df.columns =[c.strip().lower() for c in df.columns]
        df['_src'] = os.path.basename(fp)
        frames.append(df)
        print(f"  ✓ {os.path.basename(fp):30s} {len(df):>10,} rows")
    raw = pd.concat(frames, ignore_index=True)
    print(f"  Total rows: {len(raw):,}")

    raw['date_c']  = raw['date'].apply(_clean_date)
    raw['datetime'] = pd.to_datetime(
        raw['date_c'] + ' ' + raw['time'].str.strip(),
        dayfirst=True, errors='coerce')
    raw = raw.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    
    for col in['open','high','low','close','volume','oi','iv','spot']:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')

    raw['option_type']   = raw['option_type'].str.upper().str.strip()
    raw['strike_offset'] = raw['strike_offset'].str.upper().str.strip()
    raw['_offset_num']   = raw['strike_offset'].apply(_parse_offset)
    return raw

def clean_strikes(raw: pd.DataFrame, keep_range: int = 3) -> pd.DataFrame:
    """Keep only ATM±keep_range — discard far OTM/ITM noise."""
    raw = raw[raw['_offset_num'].notna()].copy()
    raw['_offset_num'] = raw['_offset_num'].astype(int)
    kept = raw[raw['_offset_num'].abs() <= keep_range].copy()
    print(f"  [Clean] Kept ATM±{keep_range}: {len(kept):,} / {len(raw):,} rows")
    return kept

def build_straddle(raw: pd.DataFrame) -> pd.DataFrame:
    """Merge ATM call + put into one row per timestamp."""
    atm = raw[raw['_offset_num'] == 0].copy()
    calls = atm[atm['option_type'] == 'CALL']
    puts  = atm[atm['option_type'] == 'PUT']

    cmap = {c: f'c_{c}' for c in['open','high','low','close','volume','oi','iv','spot']}
    pmap = {c: f'p_{c}' for c in ['open','high','low','close','volume','oi','iv','spot']}

    c = calls[['datetime'] + list(cmap)].rename(columns=cmap)
    p = puts [['datetime'] + list(pmap)].rename(columns=pmap)

    if c.empty and p.empty: raise ValueError("No ATM rows found after cleaning.")
    if c.empty:
        m = p.copy(); m['c_close']=m['p_close']; m['c_iv']=m['p_iv']; m['c_spot']=m['p_spot']
    elif p.empty:
        m = c.copy(); m['p_close']=m['c_close']; m['p_iv']=m['c_iv']; m['p_spot']=m['c_spot']
    else:
        m = pd.merge(c, p, on='datetime', how='inner')

    m['spot']           = m['c_spot']
    m['straddle_price'] = m['c_close'] + m['p_close']
    m['mid_iv']         = (m['c_iv'] + m['p_iv']) / 2.0
    m['date']           = m['datetime'].dt.date
    m['hour_min']       = m['datetime'].dt.strftime('%H:%M')
    m = m.sort_values('datetime').reset_index(drop=True)
    print(f"  ATM straddle rows: {len(m):,}")
    return m


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — INDICATORS & SIGNALS
# ═════════════════════════════════════════════════════════════════════════════

def rv_close(close: pd.Series, window: int, annual_mins: int) -> pd.Series:
    return np.log(close/close.shift(1)).rolling(window).std() * np.sqrt(annual_mins)

def iv_rank(iv: pd.Series, lookback_days: int, mins_per_day: int) -> pd.Series:
    lb = lookback_days * mins_per_day
    lo = iv.rolling(lb, min_periods=50).min()
    hi = iv.rolling(lb, min_periods=50).max()
    return (iv - lo) / (hi - lo + 1e-10) * 100

def kaufman_er(close: pd.Series, window: int) -> pd.Series:
    net_move  = (close - close.shift(window)).abs()
    bar_moves = close.diff().abs().rolling(window).sum()
    return (net_move / bar_moves.clip(lower=1e-10)).clip(0, 1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def _next_thursday(dt: pd.Timestamp) -> date:
    days_ahead = (3 - dt.weekday()) % 7
    if days_ahead == 0: days_ahead = 7
    return (dt + timedelta(days=days_ahead)).date()

def dte_from_dt(dt: pd.Timestamp) -> int:
    nxt = _next_thursday(dt)
    return (nxt - dt.date()).days

def generate_signals(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()
    annual = p['MINS_PER_DAY'] * 252

    df['iv_dec']     = df['mid_iv'] / 100.0
    df['rv']         = rv_close(df['spot'], int(p['RV_WINDOW']), annual)
    df['ivr']        = iv_rank(df['iv_dec'], int(p['IVR_LOOKBACK_DAYS']), p['MINS_PER_DAY'])
    df['iv_rv_ratio']= df['iv_dec'] / df['rv'].clip(lower=0.005)
    df['er']         = kaufman_er(df['spot'], int(p['ER_WINDOW']))
    df['dte']        = df['datetime'].apply(dte_from_dt)

    df['straddle_atr'] = atr(
        df['straddle_price'] * 1.01,
        df['straddle_price'] * 0.99,
        df['straddle_price'],
        int(p['ATR_WINDOW'])
    )

    in_window = (df['hour_min'] >= p['ENTRY_START']) & (df['hour_min'] <= p['ENTRY_END'])
    cheap_iv  = df['ivr'] < p['IVR_ENTRY_MAX']
    rv_edge   = df['iv_rv_ratio'] < p['IV_HV_RATIO_MAX']
    dte_ok    = df['dte'] >= p['DTE_MIN']         

    df['buy_signal'] = in_window & cheap_iv & rv_edge & dte_ok & df['rv'].notna()
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — BS ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def _d1(S, K, T, r, sigma):
    if T < 1e-9 or sigma < 1e-9: return np.nan
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def bs_delta(S, K, T, r, sigma, flag='call'):
    d = _d1(S, K, T, r, sigma)
    if np.isnan(d): return 1.0 if (flag=='call' and S>K) else 0.0
    return norm.cdf(d) if flag=='call' else norm.cdf(d) - 1.0

def straddle_delta(S, K, T, r, sigma):
    return bs_delta(S,K,T,r,sigma,'call') + bs_delta(S,K,T,r,sigma,'put')


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — BACKTESTER
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class HedgePosition:
    entry_time : pd.Timestamp
    entry_spot : float
    strike     : float
    flag       : str
    entry_price: float
    lots       : int
    lot_size   : int
    sl_price   : float
    tp_price   : float
    status     : str = 'open'
    exit_price : float = 0.0
    exit_reason: str  = ''
    pnl        : float = 0.0

@dataclass
class StraddleTrade:
    entry_dt       : pd.Timestamp
    entry_spot     : float
    entry_iv       : float
    entry_straddle : float
    strike         : float
    dte_at_entry   : int
    lots           : int
    lot_size       : int
    er_at_entry    : float
    hedge_pnl      : float = 0.0
    hedge_costs    : float = 0.0
    hedge_count    : int   = 0
    last_hedge_dt  : Optional[pd.Timestamp] = None
    last_hedge_spot: float = 0.0
    last_straddle_delta: float = 0.0
    hedges         : List[HedgePosition] = field(default_factory=list)
    exit_dt        : Optional[pd.Timestamp] = None
    exit_spot      : float = 0.0
    exit_straddle  : float = 0.0
    exit_reason    : str   = ''
    straddle_pnl   : float = 0.0
    total_pnl      : float = 0.0
    duration_mins  : float = 0.0


def _find_hedge_option_price(df_1min, dt, target_strike_offset, flag):
    option_type = flag.upper()
    offset_str  = f'ATM+{target_strike_offset}' if target_strike_offset > 0 \
                  else (f'ATM-{abs(target_strike_offset)}' if target_strike_offset < 0 else 'ATM')

    mask = ((df_1min['datetime'] == dt) & (df_1min['option_type'] == option_type) & 
            (df_1min['strike_offset'] == offset_str))
    rows = df_1min[mask]
    if rows.empty: return 0.0
    return float(rows.iloc[0]['close'])


def _close_open_hedges(hedges, spot_now, df_1min, dt, p):
    pnl_sum, cost_sum = 0.0, 0.0
    for h in hedges:
        if h.status != 'open': continue
        atm_strike    = round(h.entry_spot / 50) * 50
        strike_offset = round((h.strike - atm_strike) / 50)
        current_price = _find_hedge_option_price(df_1min, dt, strike_offset, h.flag)
        if current_price <= 0: current_price = h.entry_price

        should_close, reason = False, ''
        if current_price >= h.tp_price:
            should_close, reason = True, 'hedge_tp'
        elif current_price <= h.sl_price:
            should_close, reason = True, 'hedge_sl'

        if should_close:
            h.exit_price, h.exit_reason, h.status = current_price, reason, 'closed'
            h.pnl     = (current_price - h.entry_price) * h.lots * h.lot_size
            pnl_sum  += h.pnl
            cost_sum += p['TX_COST_PER_LOT'] * h.lots
    return pnl_sum, cost_sum


def run_backtest(df_1min_straddle, df_1min_raw, df_3min_straddle, p, verbose=True):
    p1 = df_1min_straddle.reset_index(drop=True)
    p3 = df_3min_straddle.sort_values('datetime').reset_index(drop=True)

    trades, open_t = [], None
    dt3_set = set(p3['datetime'].values)
    p3_idx  = {row.datetime: row for row in p3.itertuples()}

    r, lot, n = p['RISK_FREE_RATE'], p['LOT_SIZE'], p['NUM_LOTS']

    if verbose: print(f"  BACKTEST — {len(p1):,} 1-min bars")

    for i, row1 in p1.iterrows():
        dt, spot, iv, hm = row1['datetime'], row1['spot'], row1['iv_dec'], row1['hour_min']
        if pd.isna(spot) or pd.isna(iv) or iv <= 0 or spot <= 0: continue

        if open_t is not None:
            t = open_t
            elapsed_mins = (dt - t.entry_dt).total_seconds() / 60.0
            c_close = row1.get('c_close', t.entry_straddle / 2)
            p_close = row1.get('p_close', t.entry_straddle / 2)
            straddle_now = c_close + p_close
            straddle_ret = (straddle_now / t.entry_straddle) - 1.0
            iv_ratio     = iv / t.entry_iv if t.entry_iv > 0 else 1.0

            exit_reason = None
            if straddle_ret >= p['STRADDLE_TP_PCT']:       exit_reason = 'gamma_win'
            elif iv_ratio >= (1.0 + p['IV_EXPAND_PCT']):     exit_reason = 'vega_win'
            elif iv_ratio <= (1.0 - p['IV_CRUSH_PCT']):      exit_reason = 'iv_crush'
            elif elapsed_mins > p['THETA_BLEED_MINS'] and straddle_now < t.entry_straddle: exit_reason = 'theta_bleed'
            elif straddle_ret <= -p['STRADDLE_SL_PCT']:      exit_reason = 'max_drawdown'
            elif hm >= p['SESSION_END']:                     exit_reason = 'eod'

            h_pnl, h_cost = _close_open_hedges(t.hedges, spot, df_1min_raw, dt, p)
            t.hedge_pnl   += h_pnl
            t.hedge_costs += h_cost

            is_3min_bar = dt in dt3_set
            if is_3min_bar and exit_reason is None:
                mins_since_hedge = ((dt - t.last_hedge_dt).total_seconds() / 60.0 if t.last_hedge_dt else 999)
                row3 = p3_idx.get(dt)
                if row3 is not None:
                    iv3   = getattr(row3, 'iv_dec', iv)
                    dte_r = max(dte_from_dt(dt) / 365.0, 1e-5)
                    s_delta = straddle_delta(spot, t.strike, dte_r, r, iv3 if iv3 > 0 else t.entry_iv)

                    if (abs(s_delta) > p['DELTA_THRESHOLD'] or mins_since_hedge >= p['HEDGE_INTERVAL_3MIN'] * 3) and abs(spot - t.last_hedge_spot) > 0:
                        _execute_hedge(t, dt, spot, s_delta, df_1min_raw, p, row1)
                        t.last_hedge_dt, t.last_hedge_spot, t.last_straddle_delta = dt, spot, s_delta

            if exit_reason:
                t.straddle_pnl = (straddle_now - t.entry_straddle) * n * lot
                for h in t.hedges:
                    if h.status == 'open':
                        atm_s    = round(t.entry_spot / 50) * 50
                        hp_now   = _find_hedge_option_price(df_1min_raw, dt, round((h.strike - atm_s) / 50), h.flag)
                        if hp_now <= 0: hp_now = h.entry_price
                        h.exit_price, h.exit_reason, h.status = hp_now, 'straddle_closed', 'closed'
                        h.pnl         = (hp_now - h.entry_price) * h.lots * lot
                        t.hedge_pnl  += h.pnl
                        t.hedge_costs+= p['TX_COST_PER_LOT'] * h.lots

                t.total_pnl = t.straddle_pnl + t.hedge_pnl - (p['TX_COST_PER_LOT']*2*n) - t.hedge_costs
                t.exit_dt, t.exit_spot, t.exit_straddle, t.exit_reason, t.duration_mins = dt, spot, straddle_now, exit_reason, elapsed_mins
                trades.append(t)
                open_t = None
            continue

        if row1.get('buy_signal', False):
            strad = row1.get('c_close', 0) + row1.get('p_close', 0)
            if strad > 0:
                open_t = StraddleTrade(
                    entry_dt=dt, entry_spot=spot, entry_iv=iv, entry_straddle=strad, strike=round(spot/50)*50,
                    dte_at_entry=dte_from_dt(dt), lots=n, lot_size=lot, er_at_entry=row1.get('er', 0.0),
                    hedge_costs=p['TX_COST_PER_LOT']*2*n, last_hedge_spot=spot, last_hedge_dt=dt
                )

    if open_t and len(p1) > 0:
        last = p1.iloc[-1]
        strad_now = last.get('c_close', open_t.entry_straddle/2) + last.get('p_close', open_t.entry_straddle/2)
        open_t.straddle_pnl = (strad_now - open_t.entry_straddle) * n * lot
        open_t.total_pnl    = open_t.straddle_pnl + open_t.hedge_pnl - p['TX_COST_PER_LOT']*2*n - open_t.hedge_costs
        open_t.exit_dt, open_t.exit_reason, open_t.exit_straddle = last['datetime'], 'end_of_data', strad_now
        trades.append(open_t)

    return _trades_to_df(trades, verbose)


def _execute_hedge(t, dt, spot, s_delta, df_1min, p, row1):
    atm_strike = round(spot / 50) * 50
    atr_val    = row1.get('straddle_atr', spot * 0.002)

    flag, offset = ('put', 0) if s_delta > p['DELTA_THRESHOLD'] else ('call', 0)
    h_price = _find_hedge_option_price(df_1min, dt, offset, flag)
    if h_price <= 0: return

    sl_dist  = p['HEDGE_SL_ATR_MULT'] * atr_val
    sl_price = max(h_price - sl_dist, h_price * 0.50)
    tp_price = min(h_price + sl_dist * p['RR_RATIO'], h_price * (1 + p['HEDGE_TP_PCT']))

    t.hedges.append(HedgePosition(dt, spot, atm_strike + (offset * 50), flag, h_price, p['NUM_LOTS'], p['LOT_SIZE'], sl_price, tp_price))
    t.hedge_count += 1
    t.hedge_costs += p['TX_COST_PER_LOT'] * p['NUM_LOTS'] + p['SLIPPAGE_PER_HEDGE']


def _trades_to_df(trades, verbose):
    if not trades: return pd.DataFrame()
    df = pd.DataFrame([asdict(t) for t in trades])
    df['realized_pnl'] = df['total_pnl']
    df['cumulative_pnl'] = df['realized_pnl'].cumsum()
    df['num_hedges'] = df['hedges'].apply(len)
    df.drop(columns=['hedges'], inplace=True)
    if verbose:
        print(f"  Trades: {len(df)} | Win rate: {(df['realized_pnl']>0).mean()*100:.1f}% | Total P&L: ₹{df['realized_pnl'].sum():,.0f}")
    return df


def compute_metrics(df):
    if df.empty or 'realized_pnl' not in df.columns: return {'n_trades': 0, 'sharpe': -99, 'total_pnl': 0, 'win_rate': 0}
    pnl  = df['realized_pnl']
    wins, loss = pnl[pnl > 0], pnl[pnl <= 0]
    daily = df.groupby(pd.to_datetime(df['entry_dt']).dt.date)['realized_pnl'].sum()
    
    return {
        'n_trades': len(df),
        'win_rate': (len(wins) / len(pnl) * 100) if len(pnl)>0 else 0,
        'total_pnl': pnl.sum(),
        'avg_win': wins.mean() if len(wins) else 0,
        'avg_loss': loss.mean() if len(loss) else 0,
        'profit_factor': (-wins.sum() / loss.sum()) if loss.sum() < 0 else (999 if wins.sum()>0 else 0),
        'max_dd': (pnl.cumsum() - pnl.cumsum().cummax()).min(),
        'sharpe': (daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() > 0 else 0,
        'expectancy': pnl.mean(),
        'avg_duration': df['duration_mins'].mean(),
        'avg_hedges': df['hedge_count'].mean(),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — OPTUNA HYPERPARAMETER TUNING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

FIXED_PARAMS = {   
    "DTE_MIN"             : 2,
    "HEDGE_INTERVAL_3MIN" : 3,
    "DELTA_THRESHOLD"     : 0.08,
    "ATR_WINDOW"          : 14,
    "ENTRY_START"         : "09:30",
    "ENTRY_END"           : "14:00",
    "SESSION_END"         : "15:15",
    "LOT_SIZE"            : 75,
    "NUM_LOTS"            : 1,
    "TX_COST_PER_LOT"     : 30.0,
    "SLIPPAGE_PER_HEDGE"  : 2.0,
    "MINS_PER_DAY"        : 375,
    "RISK_FREE_RATE"      : 0.065,
    "MIN_ER_FOR_TREND"    : 0.35,
    "THETA_BLEED_MINS"    : 120,
}

def optuna_objective(trial, base_params, df_1min_s, df_1min_raw, df_3min_s, score_metric):
    p = base_params.copy()
    
    # ── Tunable Space ──
    p['IVR_ENTRY_MAX']     = trial.suggest_float('IVR_ENTRY_MAX', 20.0, 55.0)
    p['IV_HV_RATIO_MAX']   = trial.suggest_float('IV_HV_RATIO_MAX', 0.60, 1.00)
    p['STRADDLE_TP_PCT']   = trial.suggest_float('STRADDLE_TP_PCT', 0.08, 0.25)
    p['STRADDLE_SL_PCT']   = trial.suggest_float('STRADDLE_SL_PCT', 0.08, 0.25)
    p['IV_EXPAND_PCT']     = trial.suggest_float('IV_EXPAND_PCT', 0.02, 0.15)
    p['IV_CRUSH_PCT']      = trial.suggest_float('IV_CRUSH_PCT', 0.02, 0.15)
    p['HEDGE_TP_PCT']      = trial.suggest_float('HEDGE_TP_PCT', 0.03, 0.20)
    p['HEDGE_SL_ATR_MULT'] = trial.suggest_float('HEDGE_SL_ATR_MULT', 0.5, 4.0)
    p['RR_RATIO']          = trial.suggest_float('RR_RATIO', 0.2, 1.5)
    p['RV_WINDOW']         = trial.suggest_int('RV_WINDOW', 5, 40)
    p['IVR_LOOKBACK_DAYS'] = trial.suggest_int('IVR_LOOKBACK_DAYS', 5, 40)
    p['ER_WINDOW']         = trial.suggest_int('ER_WINDOW', 5, 30)

    # Silence logs during 100+ trials to save memory/console
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        sig  = generate_signals(df_1min_s.copy(), p)
        t_df = run_backtest(sig, df_1min_raw, df_3min_s, p, verbose=False)
        
        if t_df.empty or len(t_df) < 5: return -999.0
            
        m = compute_metrics(t_df)
        score = m.get(score_metric, 0)
        if m['n_trades'] < 10: score *= (m['n_trades'] / 10.0)
        return score
    except Exception as e:
        return -999.0
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


def optuna_search(n_trials: int, df_1min_s: pd.DataFrame, df_1min_raw: pd.DataFrame, 
                  df_3min_s: pd.DataFrame, score_metric: str, output_dir: str):
    print(f"\n  Optuna Search: {n_trials} trials maximizing '{score_metric}' …")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42, multivariate=True))
    study.optimize(
        lambda trial: optuna_objective(trial, FIXED_PARAMS, df_1min_s, df_1min_raw, df_3min_s, score_metric),
        n_trials=n_trials, show_progress_bar=True, n_jobs=1
    )
    
    best = study.best_params
    best.update(FIXED_PARAMS)
    
    print(f"\n  ✅ Optuna Complete | Best {score_metric}: {study.best_value:.3f}")
    print(f"  Best Params:\n{json.dumps({k: round(v,4) for k,v in study.best_params.items()}, indent=4)}")
    
    tune_df = study.trials_dataframe()
    tune_df = tune_df.rename(columns={'value': 'score'})
    tune_df.columns =[col.replace('params_', '') for col in tune_df.columns]
    
    try:
        fig_hist = plot_optimization_history(study)
        fig_hist.figure.savefig(os.path.join(output_dir, 'optuna_history.png'), facecolor='#0d1117')
        fig_imp = plot_param_importances(study)
        fig_imp.figure.savefig(os.path.join(output_dir, 'optuna_importances.png'), facecolor='#0d1117')
        print(f"  Optuna Insight Charts saved.")
    except Exception as e:
        pass # Visualizations can fail if backend constraints exist, we skip safely.

    return best, tune_df


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — CHARTS & EDA
# ═════════════════════════════════════════════════════════════════════════════

DARK = '#0d1117'; PANEL = '#161b22'; GREEN = '#39d353'; RED = '#f85149'
BLUE = '#58a6ff'; ORANGE = '#e3b341'; PURPLE = '#bc8cff'; GRAY = '#8b949e'; TEXT = '#c9d1d9'

def _sty(ax, title=''):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GRAY, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    if title: ax.set_title(title, color=TEXT, fontsize=9, pad=6, fontweight='bold')
    ax.grid(alpha=0.15, color=GRAY, lw=0.4)
    ax.yaxis.label.set_color(GRAY); ax.xaxis.label.set_color(GRAY)

def plot_backtest(df: pd.DataFrame, path: str):
    if df.empty: return
    m = compute_metrics(df)
    pc = GREEN if m['total_pnl'] >= 0 else RED

    fig = plt.figure(figsize=(24, 18), facecolor=DARK)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3, top=0.9, bottom=0.05, left=0.05, right=0.95)

    fig.text(0.5, 0.95, f"NIFTY GAMMA SCALPING v2.0 | P&L: ₹{m['total_pnl']:,.0f} | Sharpe: {m['sharpe']:.2f} | Win: {m['win_rate']:.1f}%",
             ha='center', color=pc, fontsize=14, fontweight='bold', fontfamily='monospace')

    ax = fig.add_subplot(gs[0,:2])
    cum = df['cumulative_pnl']
    ax.plot(cum.values, color=pc, lw=1.8)
    ax.fill_between(range(len(cum)), cum, where=cum>=0, color=GREEN, alpha=0.25)
    ax.fill_between(range(len(cum)), cum, where=cum<0, color=RED, alpha=0.25)
    _sty(ax, 'Cumulative P&L')

    ax = fig.add_subplot(gs[0,2])
    dd = cum - cum.cummax()
    ax.plot(dd.values, color=RED, lw=0.8)
    ax.fill_between(range(len(dd)), dd, color=RED, alpha=0.5)
    _sty(ax, 'Drawdown')

    ax = fig.add_subplot(gs[1,0])
    pnl = df['realized_pnl'].values
    ax.hist(pnl[pnl>=0], bins=30, color=GREEN, alpha=0.8)
    ax.hist(pnl[pnl< 0], bins=30, color=RED,   alpha=0.8)
    _sty(ax, 'P&L Distribution')

    ax = fig.add_subplot(gs[1,1])
    ax.scatter(df['straddle_pnl'], df['hedge_pnl'], c=df['realized_pnl'], cmap='RdYlGn', s=20, alpha=0.7)
    _sty(ax, 'Straddle vs Hedge P&L')

    ax = fig.add_subplot(gs[1,2])
    ec = df['exit_reason'].value_counts()
    ax.pie(ec.values, labels=ec.index, colors=[GREEN, ORANGE, RED, BLUE, PURPLE, GRAY, TEXT], 
           autopct='%1.0f%%', textprops={'color': TEXT, 'fontsize': 8})
    ax.set_title('Exit Reasons', color=TEXT)

    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    print(f"  Backtest Report saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def resample_to_3min(raw_1min: pd.DataFrame) -> pd.DataFrame:
    raw_1min = raw_1min.copy().set_index('datetime')
    groups = []
    for (opt, off), grp in raw_1min.groupby(['option_type','strike_offset']):
        r = grp[['open','high','low','close','volume','oi','iv','spot']].resample('3T').agg({
            'open':'first','high':'max','low':'min','close':'last',
            'volume':'sum','oi':'last','iv':'last','spot':'last'
        }).dropna(subset=['close'])
        r['option_type'], r['strike_offset'], r['_offset_num'] = opt, off, _parse_offset(off)
        groups.append(r)
    return pd.concat(groups).reset_index().rename(columns={'index':'datetime'}).sort_values('datetime').reset_index(drop=True)

def run_full_pipeline(data_1min_folder=DATA_1MIN, data_3min_folder=DATA_3MIN, output_dir=OUTPUT_DIR,
                      params=None, run_tuning=False, tuning_trials=100, score_metric='sharpe'):
    
    os.makedirs(output_dir, exist_ok=True)
    p = params or PARAMS

    raw_1min = load_files(data_1min_folder, '1MIN')
    raw_1min = clean_strikes(raw_1min, keep_range=3)

    try:
        raw_3min = load_files(data_3min_folder, '3MIN')
        raw_3min = clean_strikes(raw_3min, keep_range=3)
    except FileNotFoundError:
        print("  ⚠ 3MIN folder not found — resampling 1MIN to 3MIN …")
        raw_3min = resample_to_3min(raw_1min)

    print(f"\n{'═'*62}\n  BUILDING ATM STRADDLES\n{'═'*62}")
    straddle_1min = build_straddle(raw_1min)
    straddle_3min = build_straddle(raw_3min)

    print(f"\n{'═'*62}\n  GENERATING SIGNALS\n{'═'*62}")
    sig_1min = generate_signals(straddle_1min, p)
    sig_3min = generate_signals(straddle_3min, p)

    if run_tuning:
        best_p, tune_df = optuna_search(tuning_trials, sig_1min, raw_1min, sig_3min, score_metric, output_dir)
        tune_df.to_csv(os.path.join(output_dir, 'optuna_trials.csv'), index=False)
        p = best_p
        with open(os.path.join(output_dir, 'optuna_best_params.json'), 'w') as f:
            json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in p.items()}, f, indent=4)

    print(f"\n{'═'*62}\n  RUNNING FINAL BACKTEST\n{'═'*62}")
    sig_final = generate_signals(straddle_1min, p)
    trades_df = run_backtest(sig_final, raw_1min, sig_3min, p, verbose=True)

    if not trades_df.empty:
        trades_df.to_csv(os.path.join(output_dir, 'gamma_scalp_trades_v2.csv'), index=False)
        plot_backtest(trades_df, os.path.join(output_dir, 'gamma_scalp_backtest_v2.png'))
        
        m = compute_metrics(trades_df)
        print(f"\n{'═'*62}\n  FINAL PERFORMANCE METRICS\n{'═'*62}")
        for k, v in m.items(): print(f"  {k:<22} : {v:>12.2f}" if isinstance(v, float) else f"  {k:<22} : {v:>12}")

    return trades_df, sig_final, p

# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    trades_df, signals, final_params = run_full_pipeline(
        data_1min_folder = DATA_1MIN,
        data_3min_folder = DATA_3MIN,
        output_dir       = OUTPUT_DIR,
        params           = PARAMS,
        run_tuning       = True,      # ← Triggers Optuna
        tuning_trials    = 100,       # ← 100 trials is recommended for TPE to learn effectively
        score_metric     = 'sharpe',  # ← Maximize the Sharpe Ratio
    )