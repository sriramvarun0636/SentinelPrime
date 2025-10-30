import optuna
import logging
import copy
import sys
import os
from datetime import datetime

# --- Import your modified backtester and its dependencies ---
try:
    from backtester import BacktestEngine
    from main1 import load_config, PERSIST_DIR
except ImportError as e:
    print(f"FATAL: Could not import BacktestEngine from backtester.py.")
    print("Please ensure backtester.py is in the same directory.")
    print(f"Error: {e}")
    sys.exit(1)

# --- Configuration ---
CONFIG_FILE = "config.json"
HISTORICAL_DATA_FILE = "historical_data.csv"
INSTRUMENT_FILE = os.path.join(PERSIST_DIR, "instruments_nfo.csv")

N_TRIALS = 100      # Number of backtests to run
N_JOBS = 1          # Number of CPU cores to use. Set to -1 to use all cores.
MIN_TRADES_PER_TRIAL = 10 # Prune trials with too few trades

# --- Setup Logging ---
# Silence the backtester's INFO logs to keep the console clean
logging.getLogger("SENTINEL-PRIME-BACKTESTER").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.INFO)

# --- Load Base Config ONCE ---
L_OPT = logging.getLogger("OPTIMIZER")
L_OPT.setLevel(logging.INFO)
if not L_OPT.handlers:
    L_OPT.addHandler(logging.StreamHandler())

try:
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    if not os.path.exists(HISTORICAL_DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {HISTORICAL_DATA_FILE}")
    if not os.path.exists(INSTRUMENT_FILE):
         raise FileNotFoundError(f"Instrument file not found: {INSTRUMENT_FILE}. Run main1.py once to create it.")
         
    BASE_CONFIG = load_config(CONFIG_FILE)
    L_OPT.info(f"Base config '{CONFIG_FILE}' loaded successfully.")
except FileNotFoundError as e:
    L_OPT.critical(f"FATAL: Prerequisite file missing. {e}")
    sys.exit(1)
except Exception as e:
    L_OPT.critical(f"FATAL: Error loading '{CONFIG_FILE}': {e}")
    sys.exit(1)


def objective(trial: optuna.trial.Trial) -> float:
    """
    This is the main function for Optuna.
    It creates, runs, and scores one full backtest.
    """
    
    # 1. Create a deep copy of the base config for this trial
    config = copy.deepcopy(BASE_CONFIG)

    # 2. Define the "Search Space"
    # This is where we optimize the high-leverage parameters
    # we discussed earlier.

    # --- Group 1: RegimeClassifier Parameters ---
    config['strategies']['regime_classifier']['hysteresis_confirmation_count'] = trial.suggest_int(
        "reg_hysteresis_count", 2, 5
    )
    config['strategies']['regime_classifier']['compression_rank_threshold_pct'] = trial.suggest_float(
        "reg_compression_thresh", 5.0, 25.0
    )
    config['strategies']['regime_classifier']['adx_trend_entry_percentile'] = trial.suggest_float(
        "reg_adx_entry", 60.0, 90.0
    )

    # --- Group 2: Strategy Entry Parameters ---
    
    # MomentumBreakout
    config['strategies']['MomentumBreakout']['squeeze_factor'] = trial.suggest_float(
        "mb_squeeze_factor", 0.6, 1.2
    )
    
    # TrendPullback
    config['strategies']['TrendPullback']['ema_period'] = trial.suggest_int(
        "tp_ema_period", 8, 21
    )
    
    # MeanReversion
    config['strategies']['MeanReversion']['bb_period'] = trial.suggest_int(
        "mr_bb_period", 10, 30
    )

    # --- Group 3: Strategy Risk/Exit Parameters (R:R Ratios) ---
    
    # MomentumBreakout R:R
    config['strategies']['MomentumBreakout']['atr_sl_multiplier'] = trial.suggest_float(
        "mb_atr_sl", 1.0, 3.5
    )
    config['strategies']['MomentumBreakout']['atr_tp_multiplier'] = trial.suggest_float(
        "mb_atr_tp", 2.0, 6.0
    )
    
    # TrendPullback R:R
    config['strategies']['TrendPullback']['atr_sl_multiplier'] = trial.suggest_float(
        "tp_atr_sl", 1.0, 3.5
    )
    config['strategies']['TrendPullback']['atr_tp_multiplier'] = trial.suggest_float(
        "tp_atr_tp", 2.0, 6.0
    )
    
    # MeanReversion R:R
    config['strategies']['MeanReversion']['atr_sl_multiplier'] = trial.suggest_float(
        "mr_atr_sl", 1.0, 3.0
    )
    config['strategies']['MeanReversion']['atr_tp_multiplier'] = trial.suggest_float(
        "mr_atr_tp", 1.5, 4.0
    )

    # --- Group 4: Global Risk/Exit Parameters ---
    config['trading']['trailing_sl_activation_rr'] = trial.suggest_float(
        "tsl_activation_rr", 0.8, 2.5
    )
    config['trading']['trailing_sl']['chandelier_multiplier'] = trial.suggest_float(
        "tsl_chandelier_mult", 1.5, 4.0
    )
    
    # --- NOTE: We DO NOT optimize 'microstructure' parameters ---
    # As the backtester is bar-based, the MicrostructureMonitor is blind.
    # Optimizing its parameters would be meaningless.

    L_OPT.info(f"--- Starting Trial {trial.number} ---")

    try:
        # 3. Initialize the BacktestEngine with the MODIFIED config
        backtester = BacktestEngine(
            config_dict=config,
            historical_data_path=HISTORICAL_DATA_FILE
        )
        
        # 4. Run the backtest
        backtester.run_backtest()
        
        # 5. Get the metrics dictionary
        metrics = backtester.generate_report(return_metrics=True)

        # 6. Pruning: Discard trials that are bad
        if not metrics or metrics['total_trades'] < MIN_TRADES_PER_TRIAL:
            L_OPT.warning(f"Trial {trial.number} pruned: Too few trades ({metrics.get('total_trades', 0)}).")
            raise optuna.exceptions.TrialPruned()
            
        # 7. Return the score to Optuna
        # We want to maximize the Profit Factor
        score = metrics.get('profit_factor', 0.0)
        
        L_OPT.info(f"--- Trial {trial.number} Finished ---")
        L_OPT.info(f"Trades: {metrics['total_trades']}, PnL: {metrics['total_pnl']:.2f}, PF: {score:.2f}")
        
        return score

    except optuna.exceptions.TrialPruned as e:
        # Re-raise to let Optuna handle it
        raise e
    except Exception as e:
        L_OPT.error(f"Error in trial {trial.number}: {e}", exc_info=True)
        # Prune this trial by raising the exception
        raise optuna.exceptions.TrialPruned()

# ==================================================================================================
# --- 6. MAIN EXECUTION SCRIPT ---
# ==================================================================================================

if __name__ == "__main__":
    
    # 1. Check for dependencies
    try:
        import pandas_ta
    except ImportError:
        L_OPT.critical("FATAL: `pandas-ta` library not found.")
        L_OPT.critical("Please install it: pip install pandas-ta")
        sys.exit(1)
        
    L_OPT.info("Starting SENTINEL-PRIME Parameter Optimization...")
    L_OPT.info(f"Database will be saved to: sentinel_study.db")
    L_OPT.info(f"Running {N_TRIALS} trials using {N_JOBS} parallel job(s)...")

    # We use a database (SQLite) to store results, so you can resume
    study_name = "sentinel_prime_v1"
    storage_name = "sqlite:///sentinel_study.db"

    # 2. Create or load the study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize" # We want to MAXIMIZE the Profit Factor
    )

    # 3. Run the optimization
    start_time = datetime.now()
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=N_JOBS, # Set to -1 to use all CPU cores
        show_progress_bar=True # Requires `tqdm` (pip install tqdm)
    )
    end_time = datetime.now()

    # 4. Print the final results
    L_OPT.info("--- OPTIMIZATION COMPLETE ---")
    L_OPT.info(f"Total time: {end_time - start_time}")
    
    try:
        best_trial = study.best_trial
        L_OPT.info("Best trial found:")
        L_OPT.info(f"  Value (Profit Factor): {best_trial.value:.4f}")
        L_OPT.info("  Best Parameters:")
        for key, value in best_trial.params.items():
            L_OPT.info(f"    {key}: {value}")
            
    except ValueError:
        L_OPT.warning("No valid trials were completed. Could not find best parameters.")
        
    L_OPT.info("You can visualize results by running:")
    L_OPT.info(f"$ optuna-dashboard {storage_name}")