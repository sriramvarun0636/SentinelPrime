import optuna
import pandas as pd
import logging
import json
import sys
import logging
from datetime import datetime
import os

try:
    from backtester import objective
except ImportError:
    print("FATAL: Could not import 'objective' from 'backtester.py'.")
    print("Please ensure 'backtester.py' is in the same directory as this script.")
    sys.exit(1)


# --- CONFIGURATION ---
N_TRIALS = 100  # Number of different parameter combinations to test. 100-300 is a good range.
OUTPUT_CONFIG_FILE = "config.json"
CONFIG_BACKUP_DIR = "config_backups"
# --- END CONFIGURATION ---

def backup_config(source_path: str, backup_dir: str):
    """Creates a timestamped backup of the config file before overwriting."""
    if not os.path.exists(source_path):
        print(f"Warning: No existing config file found at '{source_path}' to back up.")
        return
    
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_path = os.path.join(backup_dir, f"config_{timestamp}.json")
    
    try:
        with open(source_path, 'r') as f_source:
            with open(backup_path, 'w') as f_backup:
                f_backup.write(f_source.read())
        print(f"Backed up current config to '{backup_path}'")
    except Exception as e:
        print(f"ERROR: Could not back up config file: {e}")

if __name__ == "__main__":
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    print("--- Starting Weekly Strategy Optimization ---")
    
    print("1. Loading historical data...")
    try:
        nifty_df = pd.read_csv("data/nifty_1min_data.csv")
        bn_df = pd.read_csv("data/banknifty_1min_data.csv")
        vix_df = pd.read_csv("data/india_vix_1min_data.csv")
        print(f"   Loaded {len(bn_df)} bars of BankNifty, Nifty, and VIX data.")
    except FileNotFoundError as e:
        print(f"\nERROR: Data file not found: {e.filename}. Please run 'python download_data.py' first.")
        sys.exit(1)

    # 2. Creating and running the Optuna study
    study = optuna.create_study(direction="maximize")
    
    print(f"\n2. Starting optimization with {N_TRIALS} trials. This may take some time...")
    
    try:
        # Pass all three dataframes to the objective function
        study.optimize(lambda trial: objective(trial, nifty_df, bn_df, vix_df), n_trials=N_TRIALS)
    except Exception as e:
        print(f"\nFATAL ERROR during optimization process: {e}")
        print("The process will be aborted.")
        sys.exit(1)

    # 3. Processing and displaying the results
    print("\n--- Optimization Finished ---")
    
    if not study.best_trial or study.best_value <= -999.0:
        print("Optimization did not find any profitable parameter sets that met the minimum trade criteria.")
        print("Consider widening parameter ranges in backtester.py or running more trials.")
        sys.exit(0)

    best = study.best_trial
    print(f"Best Fitness Score Achieved: {best.value:.4f}")
    print("\nOptimal Parameters Found:")
    
    best_params_dict = best.params
    for key, value in best_params_dict.items():
        print(f"  {key}: {value}")

    # 4. Surgically updating the live config.json file
    print(f"\n4. Preparing to update live configuration file: '{OUTPUT_CONFIG_FILE}'")
    
    backup_config(OUTPUT_CONFIG_FILE, CONFIG_BACKUP_DIR)
    
    try:
        with open(OUTPUT_CONFIG_FILE, 'r') as f:
            live_config = json.load(f)

        # Update strategy parameters
        live_config["strategies"]["momentum_breakout"]["atr_sl_multiplier"] = best_params_dict["m_atr_sl"]
        live_config["strategies"]["momentum_breakout"]["atr_tp_multiplier"] = best_params_dict["m_atr_tp"]
        live_config["strategies"]["momentum_breakout"]["max_iv_entry"] = best_params_dict["m_max_iv"]
        
        live_config["strategies"]["trend_pullback"]["ema_period"] = best_params_dict["t_ema_period"]
        live_config["strategies"]["trend_pullback"]["atr_sl_multiplier"] = best_params_dict["t_atr_sl"]
        live_config["strategies"]["trend_pullback"]["atr_tp_multiplier"] = best_params_dict["t_atr_tp"]

        live_config["strategies"]["mean_reversion"]["atr_sl_multiplier"] = best_params_dict["mr_atr_sl"]
        
        # Update trading rules
        live_config["trading"]["scale_out_rules"][0]["rr_target"] = best_params_dict["rr_target"]
        live_config["trading"]["trade_cooldown_minutes"] = best_params_dict["cooldown"]
        
        with open(OUTPUT_CONFIG_FILE, 'w') as f:
            json.dump(live_config, f, indent=2)
            
        print("\nSuccessfully updated config file. The system is calibrated for the new week.")

    except FileNotFoundError:
        print(f"\nERROR: Live config file '{OUTPUT_CONFIG_FILE}' not found. Please create it or check the path.")
        print("Optimal parameters were not deployed.")
    except Exception as e:
        print(f"\nERROR: Failed to update config file: {e}")
        print("Please update it manually with the parameters printed above.")