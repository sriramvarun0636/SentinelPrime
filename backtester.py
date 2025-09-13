from __future__ import annotations
import optuna
import pandas as pd
import pandas_ta as ta
import numpy as np
import pytz
import math
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, date, timedelta
from typing import List, Dict, Optional, Tuple
from scipy.interpolate import interp1d

try:
    from main import (
        Regime, OrderSide, StrategyName,
        MomentumBreakoutStrategy, TrendPullbackStrategy, MeanReversionStrategy,
        RegimeClassifier, TradeSignal, StrategyPerformanceTracker, HypotheticalTrade,
        _get_underlying, black_scholes_price
    )
except ImportError:
    print("FATAL: Could not import from 'sentinel_apex_v4_advanced_bot.py'.")
    print("Please ensure the required classes are present and the file is in the same directory.")
    import sys
    sys.exit(1)

# --- Constants & Configuration ---
L = logging.getLogger("BACKTESTER")
IST = pytz.timezone("Asia/Kolkata")
STARTING_EQUITY = 100000.0
TRANSACTION_COST_PCT = 0.0003
RISK_FREE_RATE = 0.05
EARLIEST_TIMESTAMP = IST.localize(datetime(2000, 1, 1))

# --- Mock Objects to Mimic Live Engine ---
class MockPriceBus:
    def __init__(self): self._ltps = {}
    def ltp(self, token: int) -> Optional[float]: return self._ltps.get(token)
    def update_ltp(self, token: int, price: float): self._ltps[token] = price

class MockBarStore:
    def __init__(self, full_data: pd.DataFrame):
        ts_col = pd.to_datetime(full_data['date'])
        if ts_col.dt.tz is None:
            self.full_df = full_data.set_index(ts_col.dt.tz_localize(IST))
        else:
            self.full_df = full_data.set_index(ts_col.dt.tz_convert(IST))

    def get_ohlc(self, timeframe: int, until_time: datetime) -> pd.DataFrame:
        sliced_1min_df = self.full_df.loc[self.full_df.index < until_time].copy()
        if timeframe == 1:
            return sliced_1min_df
        return sliced_1min_df.resample(f'{timeframe}min').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).dropna()

class MockEngine:
    # --- FIX #1: Update the __init__ method to accept a price_bus ---
    def __init__(self, nifty_bars: MockBarStore, bn_bars: MockBarStore, price_bus: MockPriceBus):
        self.bars = {1: nifty_bars, 2: bn_bars}
        self.book = {1: "NIFTY 50", 2: "NIFTY BANK"}
        self.prices = price_bus # Add the prices attribute

    def get_ohlc(self, token: int, timeframe: int, until_time: datetime) -> pd.DataFrame:
        return self.bars[token].get_ohlc(timeframe, until_time)

# --- Data Structures for Backtesting ---
@dataclass
class BacktestTrade:
    entry_time: datetime; exit_time: datetime; entry_price_underlying: float
    exit_price_underlying: float; qty: int; side: OrderSide; pnl: float
    exit_reason: str; strategy: str; regime: str
    entry_option_price: float; exit_option_price: float

@dataclass
class BacktestPosition:
    entry_time: datetime; side: OrderSide; qty: int; token: int;
    entry_price_underlying: float; sl_price_underlying: float; tp_price_underlying: float
    strategy: str; regime: str;
    strike_price: float; expiry_date: date; is_call: bool
    entry_option_price: float; favorable_price_since_entry: float = 0.0

# --- The Main Backtesting Engine ---
class BacktestEngine:
    def __init__(self, nifty_data: pd.DataFrame, bn_data: pd.DataFrame, vix_data: pd.DataFrame, strategy_params: Dict, config: Dict):
        nifty_bars, bn_bars = MockBarStore(nifty_data), MockBarStore(bn_data)
        self.vix_data = vix_data.copy()
        ts_col = pd.to_datetime(self.vix_data['date'])
        if ts_col.dt.tz is None: self.vix_data['timestamp'] = ts_col.dt.tz_localize(IST)
        else: self.vix_data['timestamp'] = ts_col.dt.tz_convert(IST)
        self.vix_data = self.vix_data.set_index('timestamp')

        self.mock_prices = MockPriceBus()
        # --- FIX #2: Pass the mock_prices object when creating MockEngine ---
        self.mock_engine = MockEngine(nifty_bars, bn_bars, self.mock_prices)

        self.config = config; self.params = strategy_params
        self.classifier = RegimeClassifier(self.mock_engine, self.mock_prices, 1, 2, 260105)
        self.strategies: Dict[Regime, List] = {
            Regime.COMPRESSION: [MomentumBreakoutStrategy(StrategyName.MOMENTUM_BREAKOUT, self.mock_engine, self.params["momentum_breakout"])],
            Regime.TRENDING_UP: [TrendPullbackStrategy(StrategyName.TREND_PULLBACK, self.mock_engine, self.params["trend_pullback"])],
            Regime.TRENDING_DOWN: [TrendPullbackStrategy(StrategyName.TREND_PULLBACK, self.mock_engine, self.params["trend_pullback"])],
            Regime.CHOP: [MeanReversionStrategy(StrategyName.MEAN_REVERSION, self.mock_engine, self.params["mean_reversion"])]
        }
        self.performance_tracker = StrategyPerformanceTracker(lookback=self.config["trading"]["meta_strategy"]["performance_lookback_trades"])
        self.open_hypothetical_trades: Dict[str, HypotheticalTrade] = {}
        self.trade_log: List[BacktestTrade] = []; self.equity = STARTING_EQUITY; self.equity_curve = []
        self.current_regime = Regime.UNCLEAR
        self.last_regime_change_time: Optional[datetime] = None
        self._load_vol_surfaces()

    def _load_vol_surfaces(self):
        self.vol_surfaces: Dict[str, Dict[date, Optional[interp1d]]] = {"NIFTY": {}, "BANKNIFTY": {}}
        surface_dir = os.path.join("data", "vol_surface")
        if not os.path.exists(surface_dir):
            L.warning(f"Volatility surface directory not found at '{surface_dir}'. Backtester will fallback to VIX.")
            return

        for file_name in os.listdir(surface_dir):
            try:
                parts = file_name.replace('.csv', '').split('_')
                name, dt_str = parts[0], parts[1]
                surface_date = datetime.strptime(dt_str, '%Y-%m-%d').date()
                df = pd.read_csv(os.path.join(surface_dir, file_name)).sort_values('strike').drop_duplicates(subset=['strike'])
                if len(df) < 4: continue
                self.vol_surfaces[name][surface_date] = interp1d(df['strike'], df['iv'], kind='cubic', fill_value="extrapolate")
            except Exception as e:
                L.warning(f"Could not load vol surface file {file_name}: {e}")

    def _get_iv_for_strike(self, token: int, strike: float, trade_date: date) -> Optional[float]:
        name = "BANKNIFTY" if token == 2 else "NIFTY"
        surface_func = self.vol_surfaces[name].get(trade_date)
        if surface_func is not None:
            try:
                return float(surface_func(strike))
            except Exception:
                L.warning(f"Could not extrapolate IV for strike {strike} on {trade_date}. Falling back to VIX.")

        L.debug(f"No vol surface for {name} on {trade_date}, falling back to VIX.")
        try:
            vix_for_day = self.vix_data.loc[self.vix_data.index.date == trade_date]
            return vix_for_day.iloc[0]['close'] / 100 if not vix_for_day.empty else None
        except IndexError:
            return None

    def run(self):
        idx1 = self.mock_engine.bars[1].full_df.index
        idx2 = self.mock_engine.bars[2].full_df.index
        idx3 = self.vix_data.index
        common_index = idx1.intersection(idx2).intersection(idx3).sort_values()

        position: Optional[BacktestPosition] = None
        cooldown_until = EARLIEST_TIMESTAMP
        L.info(f"Starting backtest on {len(common_index)} synchronized bars...")

        if len(common_index) == 0:
            L.error("No common data found across Nifty, BankNifty, and VIX files. Aborting backtest.")
            return self.generate_report()

        for bar_time in common_index:
            self.mock_prices.update_ltp(1, self.mock_engine.bars[1].full_df.loc[bar_time, 'close'])
            self.mock_prices.update_ltp(2, self.mock_engine.bars[2].full_df.loc[bar_time, 'close'])
            self.mock_prices.update_ltp(260105, self.vix_data.loc[bar_time, 'close'])

            self._update_regime_state(bar_time)
            self._manage_hypothetical_trades(bar_time)

            if position:
                row = self.mock_engine.bars[position.token].full_df.loc[bar_time]
                position = self._manage_open_position(position, row)

            if not position and bar_time > cooldown_until:
                pos, cooldown_until_new = self._check_for_new_entry(bar_time)
                if pos:
                    position = pos
                    cooldown_until = cooldown_until_new

            self.equity_curve.append({'timestamp': bar_time, 'equity': self.equity})

        L.info("Backtest finished.")
        return self.generate_report()

    def _update_regime_state(self, current_time: datetime):
        old_regime = self.current_regime
        raw_regime, _ = self.classifier.get_raw_classification(self.current_regime, current_time)
        if self.classifier.potential_regime and raw_regime == self.classifier.potential_regime[0]:
            self.classifier.confirmation_count += 1
        else:
            self.classifier.potential_regime = (raw_regime, None)
            self.classifier.confirmation_count = 1
        
        if self.classifier.confirmation_count >= self.classifier.confirmation_threshold:
            confirmed_regime, _ = self.classifier.potential_regime
            if self.current_regime != confirmed_regime:
                self.current_regime = confirmed_regime
                self.last_regime_change_time = current_time
                L.info(f"REGIME SHIFT at {current_time}: {old_regime.name} -> {self.current_regime.name}")

    def _manage_open_position(self, pos: BacktestPosition, row: pd.Series) -> Optional[BacktestPosition]:
        exit_price_underlying, exit_reason = None, None
        
        if pos.side == OrderSide.BUY:
            if row['low'] <= pos.sl_price_underlying:
                exit_price_underlying, exit_reason = pos.sl_price_underlying, "SL_HIT"
            elif pos.tp_price_underlying > 0 and row['high'] >= pos.tp_price_underlying:
                exit_price_underlying, exit_reason = pos.tp_price_underlying, "TP_HIT"
        else: # SELL
            if row['high'] >= pos.sl_price_underlying:
                exit_price_underlying, exit_reason = pos.sl_price_underlying, "SL_HIT"
            elif pos.tp_price_underlying > 0 and row['low'] <= pos.tp_price_underlying:
                exit_price_underlying, exit_reason = pos.tp_price_underlying, "TP_HIT"
        
        if exit_reason:
            self._close_position(pos, row.name, exit_price_underlying, exit_reason)
            return None
        return pos

    def _check_for_new_entry(self, bar_time: datetime) -> Tuple[Optional[BacktestPosition], datetime]:
        cooldown_time = bar_time + pd.Timedelta(minutes=self.config["trading"]["trade_cooldown_minutes"])
        now_time = bar_time.time()
        settling_time = dtime.fromisoformat(self.config["timings"]["market_settling_time"])
        final_entry = dtime.fromisoformat(self.config["timings"]["final_entry_time"])

        if not (settling_time <= now_time < final_entry): return None, cooldown_time
        if self.last_regime_change_time and (bar_time - self.last_regime_change_time) < timedelta(minutes=15): return None, cooldown_time

        all_valid_signals: List[Tuple[int, TradeSignal]] = []
        for token in [1, 2]:
            if strategies_for_regime := self.strategies.get(self.current_regime):
                for strategy in strategies_for_regime:
                    if signal := strategy.evaluate(token, self.current_regime, bar_time):
                        L.info(f">>> SIGNAL DETECTED at {bar_time} by {signal.strategy_name.value} for token {token} <<<")
                        all_valid_signals.append((token, signal))
        
        if not all_valid_signals: return None, cooldown_time
        
        filtered_signals = self._filter_signals_by_rs(all_valid_signals, bar_time)
        if not filtered_signals: return None, cooldown_time
        best_signal_tuple = self.performance_tracker.get_best_strategy(filtered_signals) if self.config["trading"]["meta_strategy"]["enabled"] else filtered_signals[0]
        
        if best_signal_tuple:
            best_token, best_signal = best_signal_tuple
            row = self.mock_engine.bars[best_token].full_df.loc[bar_time]
            position = self._open_position(best_token, best_signal, row, bar_time)
            if position:
                for token, signal in all_valid_signals:
                    try:
                        entry_price_underlying = self.mock_engine.bars[token].full_df.loc[bar_time, 'close']
                        hypo_id = f"hypo_{signal.strategy_name.value}_{token}_{bar_time.timestamp()}"
                        self.open_hypothetical_trades[hypo_id] = HypotheticalTrade(
                            signal=signal, token=token, entry_time=bar_time,
                            entry_price_underlying=entry_price_underlying,
                            sl_price_underlying=entry_price_underlying - signal.risk_points if signal.side == OrderSide.BUY else entry_price_underlying + signal.risk_points,
                            tp_price_underlying=entry_price_underlying + signal.reward_points if signal.side == OrderSide.BUY else entry_price_underlying - signal.reward_points
                        )
                    except KeyError: continue
            return position, cooldown_time
        return None, cooldown_time

    def _filter_signals_by_rs(self, signals: List[Tuple[int, TradeSignal]], current_time: datetime) -> List[Tuple[int, TradeSignal]]:
        if not self.config["trading"]["intermarket_analysis"]["enabled"]: return signals
        rs_status = self._get_simulated_relative_strength(current_time)
        final_signals = []
        for token, signal in signals:
            underlying_name = _get_underlying(self.mock_engine.book[token])
            if signal.side == OrderSide.BUY:
                if underlying_name == "BANKNIFTY" and rs_status == "NIFTY_OUTPERFORMING": continue
                elif underlying_name == "NIFTY" and rs_status == "BNF_OUTPERFORMING": continue
            final_signals.append((token, signal))
        if len(signals) > 0 and len(final_signals) < len(signals): L.info(f"Signal filtered by Inter-Market Analysis at {current_time}.")
        return final_signals

    def _get_simulated_relative_strength(self, current_time: datetime) -> str:
        ma_period = self.config["trading"]["intermarket_analysis"]["ratio_ma_period"]
        df_n = self.mock_engine.get_ohlc(1, 1, current_time)
        df_bn = self.mock_engine.get_ohlc(2, 1, current_time)
        if len(df_n) < ma_period or len(df_bn) < ma_period: return "NEUTRAL"
        
        aligned_n, aligned_bn = df_n['close'].align(df_bn['close'], join='inner')
        if aligned_n.empty or len(aligned_n) < ma_period: return "NEUTRAL"
        
        ratio = aligned_bn / aligned_n
        ratio_ma = ratio.rolling(window=ma_period).mean()
        if pd.isna(ratio.iloc[-1]) or pd.isna(ratio_ma.iloc[-1]): return "NEUTRAL"
        
        if ratio.iloc[-1] > ratio_ma.iloc[-1] * 1.001: return "BNF_OUTPERFORMING"
        elif ratio.iloc[-1] < ratio_ma.iloc[-1] * 0.999: return "NIFTY_OUTPERFORMING"
        else: return "NEUTRAL"

    def _open_position(self, token: int, signal: TradeSignal, row: pd.Series, bar_time: datetime) -> Optional[BacktestPosition]:
        underlying_price = row['close']
        underlying_name = "BANKNIFTY" if token == 2 else "NIFTY"
        step_size = 100 if token == 2 else 50
        strike_price = round(underlying_price / step_size) * step_size
        is_call = (signal.side == OrderSide.BUY)

        today = bar_time.date()
        days_to_thursday = (3 - today.weekday() + 7) % 7
        expiry_date = today + timedelta(days=days_to_thursday if days_to_thursday > 0 else 7)
        time_to_expiry = max(1e-6, (datetime.combine(expiry_date, dtime(15, 30)) - bar_time.replace(tzinfo=None)).total_seconds() / (365 * 24 * 60 * 60))
        
        iv = self._get_iv_for_strike(token, strike_price, today)
        if iv is None: return None

        entry_option_price = black_scholes_price(S=underlying_price, K=strike_price, T=time_to_expiry, r=RISK_FREE_RATE, sigma=iv, is_call=is_call)
        sl_underlying = underlying_price - signal.risk_points if signal.side == OrderSide.BUY else underlying_price + signal.risk_points
        tp_underlying = underlying_price + signal.reward_points if signal.side == OrderSide.BUY else underlying_price - signal.reward_points
        sl_option_price = black_scholes_price(S=sl_underlying, K=strike_price, T=time_to_expiry, r=RISK_FREE_RATE, sigma=iv, is_call=is_call)
        risk_per_option = abs(entry_option_price - sl_option_price)
        
        max_sl_pct = self.config["trading"].get("max_sl_pct_of_premium", 100)
        sl_as_pct_of_premium = (risk_per_option / entry_option_price * 100) if entry_option_price > 0 else float('inf')
        
        if risk_per_option <= 0.1 or sl_as_pct_of_premium > max_sl_pct: return None
        
        lot_sizes = self.config["trading"]["lot_sizes"]
        lot_size = lot_sizes.get(underlying_name, 0)
        if lot_size == 0:
            L.error(f"Lot size for {underlying_name} not found in config.")
            return None

        risk_per_contract = risk_per_option * lot_size
        allowed_risk = self.equity * (self.config["trading"]["risk_tiers"]["standard"] / 100.0)
        qty = int(allowed_risk // risk_per_contract) * lot_size if risk_per_contract > 0 else 0
        
        if qty == 0: return None

        return BacktestPosition(
            entry_time=row.name, side=signal.side, qty=qty, token=token,
            entry_price_underlying=underlying_price, sl_price_underlying=sl_underlying,
            tp_price_underlying=tp_underlying if signal.reward_points > 0 else 0,
            strategy=signal.strategy_name.value, regime=self.current_regime.name, strike_price=strike_price, expiry_date=expiry_date,
            is_call=is_call, entry_option_price=entry_option_price, favorable_price_since_entry=underlying_price
        )

    def _close_position(self, pos: BacktestPosition, exit_time: datetime, exit_price_underlying: float, reason: str):
        time_to_expiry = max(1e-6, (datetime.combine(pos.expiry_date, dtime(15, 30)) - exit_time.replace(tzinfo=None)).total_seconds() / (365 * 24 * 60 * 60))
        
        iv = self._get_iv_for_strike(pos.token, pos.strike_price, exit_time.date())
        if iv is None:
            try: iv = self.vix_data.loc[exit_time, 'close'] / 100
            except KeyError: iv = self.vix_data.loc[self.vix_data.index < exit_time].iloc[-1]['close'] / 100
        
        exit_option_price = black_scholes_price(S=exit_price_underlying, K=pos.strike_price, T=time_to_expiry, r=RISK_FREE_RATE, sigma=iv, is_call=pos.is_call)
        
        gross_pnl = (exit_option_price - pos.entry_option_price) * pos.qty
        
        entry_turnover = pos.entry_option_price * pos.qty
        exit_turnover = exit_option_price * pos.qty
        total_turnover = entry_turnover + exit_turnover
        costs = total_turnover * TRANSACTION_COST_PCT
        
        net_pnl = gross_pnl - costs
        
        self.equity += net_pnl
        trade = BacktestTrade(
            entry_time=pos.entry_time, exit_time=exit_time, entry_price_underlying=pos.entry_price_underlying,
            exit_price_underlying=exit_price_underlying, qty=pos.qty, side=pos.side, pnl=net_pnl, 
            exit_reason=reason, strategy=pos.strategy, regime=pos.regime,
            entry_option_price=pos.entry_option_price, exit_option_price=exit_option_price
        )
        self.trade_log.append(trade)

    def _manage_hypothetical_trades(self, bar_time: datetime):
        if not self.open_hypothetical_trades: return
        closed_hypo_ids = []
        for hypo_id, hypo in self.open_hypothetical_trades.items():
            if hypo.entry_time.date() != bar_time.date():
                closed_hypo_ids.append(hypo_id); continue

            df_underlying = self.mock_engine.bars[hypo.token].full_df
            try:
                trade_bars = df_underlying.loc[(df_underlying.index > hypo.entry_time) & (df_underlying.index <= bar_time)]
                if trade_bars.empty: continue
            except KeyError: continue
            
            exit_reason, exit_price = None, None
            if hypo.signal.side == OrderSide.BUY:
                sl_hit = (trade_bars['low'] <= hypo.sl_price_underlying).any()
                tp_hit = hypo.tp_price_underlying > 0 and (trade_bars['high'] >= hypo.tp_price_underlying).any()
            else: # SELL
                sl_hit = (trade_bars['high'] >= hypo.sl_price_underlying).any()
                tp_hit = hypo.tp_price_underlying > 0 and (trade_bars['low'] <= hypo.tp_price_underlying).any()
            
            if sl_hit: exit_reason, exit_price = "SL_HIT", hypo.sl_price_underlying
            elif tp_hit: exit_reason, exit_price = "TP_HIT", hypo.tp_price_underlying
            
            if exit_reason:
                pnl_points = exit_price - hypo.entry_price_underlying
                if hypo.signal.side == OrderSide.SELL: pnl_points *= -1
                self.performance_tracker.add_hypothetical_result(hypo.signal, pnl_points)
                closed_hypo_ids.append(hypo_id)

        for hypo_id in closed_hypo_ids:
            self.open_hypothetical_trades.pop(hypo_id, None)
    
    def generate_report(self) -> Dict:
        if not self.trade_log: return {"error": "No trades were executed."}
        
        df = pd.DataFrame(self.trade_log)
        df['duration_mins'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
        df['win'] = df['pnl'] > 0
        
        total_trades = len(df)
        wins = df['win'].sum()
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = df['pnl'].sum()
        
        winning_trades = df[df['win']]
        losing_trades = df[~df['win']]
        
        avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        
        reward_risk_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if not losing_trades.empty and losing_trades['pnl'].sum() != 0 else float('inf')
        
        equity_df = pd.DataFrame(self.equity_curve).set_index('timestamp')
        returns = equity_df['equity'].pct_change().dropna()
        
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * (6.5*60)) if returns.std() > 0 else 0.0
        max_drawdown = (equity_df['equity'] / equity_df['equity'].cummax() - 1).min()
        
        return {
            "Total PNL": round(total_pnl, 2),
            "Profit Factor": round(profit_factor, 2),
            "Sharpe Ratio": round(sharpe_ratio, 2),
            "Max Drawdown Pct": round(max_drawdown * 100, 2),
            "Total Trades": total_trades,
            "Win Rate": round(win_rate, 2),
            "Avg Win": round(avg_win, 2),
            "Avg Loss": round(avg_loss, 2),
            "Reward Risk Ratio": round(reward_risk_ratio, 2),
            "Avg Trade Duration Mins": round(df['duration_mins'].mean(), 2)
        }

def objective(trial: 'optuna.Trial', nifty_df: pd.DataFrame, bn_df: pd.DataFrame, vix_df: pd.DataFrame) -> float:
    backtest_config = {
        "trading": { "lot_sizes": {"NIFTY": 25, "BANKNIFTY": 15}, "risk_tiers": {"standard": 1.5}, "scale_out_rules": [{"rr_target": trial.suggest_float("rr_target", 1.5, 2.5), "pct_to_close": 50}], "trade_cooldown_minutes": trial.suggest_int("cooldown", 5, 15), "meta_strategy": {"enabled": True, "performance_lookback_trades": 5}, "intermarket_analysis": {"enabled": True, "ratio_ma_period": 20} },
        "timings": { "market_settling_time": "09:30:00", "final_entry_time": "14:45:00" }
    }
    strategy_params = {
        "momentum_breakout": { "target_delta": 0.55, "resample_minutes": 5, "bb_period": 20, "squeeze_period": 50, "squeeze_factor": 0.8, "volume_factor": 1.75, "atr_sl_multiplier": trial.suggest_float("m_atr_sl", 1.5, 2.2), "atr_tp_multiplier": trial.suggest_float("m_atr_tp", 2.5, 3.5), "max_iv_entry": trial.suggest_float("m_max_iv", 25.0, 32.0) },
        "trend_pullback": { "target_delta": 0.6, "primary_tf": 5, "confirm_tf": 15, "ema_period": trial.suggest_int("t_ema_period", 18, 25), "atr_sl_multiplier": trial.suggest_float("t_atr_sl", 1.8, 2.5), "atr_tp_multiplier": trial.suggest_float("t_atr_tp", 3.0, 4.5) },
        "mean_reversion": { "target_delta": 0.45, "resample_minutes": 5, "bb_period": 20, "atr_sl_multiplier": trial.suggest_float("mr_atr_sl", 1.2, 1.8), "atr_tp_multiplier": 1.5 }
    }
    engine = BacktestEngine(nifty_df, bn_df, vix_df, strategy_params, backtest_config)
    report = engine.run()
    if "error" in report or report["total_trades"] < 15: return -999.0
    
    sharpe = report["Sharpe Ratio"]
    drawdown = report["Max Drawdown Pct"]
    profit_factor = report["Profit Factor"]
    pnl = report["Total PNL"]
    
    drawdown_penalty = 1.0 if drawdown > -25.0 else 0.1
    fitness_score = ((sharpe * 0.6) + (profit_factor * 0.4)) * drawdown_penalty
    
    if pnl < 0: fitness_score = -abs(fitness_score)
    return fitness_score

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    default_params = {
        "momentum_breakout": { "target_delta": 0.55, "resample_minutes": 5, "bb_period": 20, "squeeze_period": 50, "squeeze_factor": 0.8, "volume_factor": 1.75, "atr_sl_multiplier": 1.8, "atr_tp_multiplier": 3.0, "max_iv_entry": 28.0 },
        "trend_pullback": { "target_delta": 0.6, "primary_tf": 5, "confirm_tf": 15, "ema_period": 21, "atr_sl_multiplier": 2.0, "atr_tp_multiplier": 3.5 },
        "mean_reversion": { "target_delta": 0.45, "resample_minutes": 5, "bb_period": 20, "atr_sl_multiplier": 1.5, "atr_tp_multiplier": 1.5 }
    }
    default_config = {
        "trading": { "lot_sizes": {"NIFTY": 25, "BANKNIFTY": 15}, "risk_tiers": {"standard": 1.5}, "scale_out_rules": [{"rr_target": 1.8, "pct_to_close": 50}], "trade_cooldown_minutes": 10, "meta_strategy": {"enabled": True, "performance_lookback_trades": 5}, "intermarket_analysis": {"enabled": True, "ratio_ma_period": 20}, "max_sl_pct_of_premium": 50 },
        "timings": { "market_settling_time": "09:30:00", "final_entry_time": "14:45:00" }
    }
    
    try:
        nifty_df = pd.read_csv("data/nifty_1min_data.csv")
        bn_df = pd.read_csv("data/banknifty_1min_data.csv")
        vix_df = pd.read_csv("data/india_vix_1min_data.csv")
        
        backtest_engine = BacktestEngine(nifty_df, bn_df, vix_df, default_params, default_config)
        final_report = backtest_engine.run()
        
        print("\n--- Backtest Performance Report ---")
        for key, value in final_report.items():
            print(f"{key:<25}: {value}")
        print("---------------------------------")
        
    except FileNotFoundError as e:
        print(f"\nERROR: Data file not found: {e.filename}. Please run 'python download_data.py' first.")