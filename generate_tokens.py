import json
import os
import sys
import logging
import pandas as pd
import requests
from io import StringIO
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from kiteconnect import KiteConnect

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
L = logging.getLogger("TokenGenerator")

# --- Define Constants ---
# This is the official NSE source for the Nifty 50 constituents
NSE_NIFTY50_URL = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"

# File paths from your main script
PERSIST_DIR = os.environ.get("PERSIST_DIR", "./persist_sentinel_prime")
TOKEN_FILE_PATH = os.path.join(PERSIST_DIR, "kite_token.json")
OUTPUT_JSON_FILE = "nifty50_tokens.json" # The file your config needs

# --- Re-use your login function ---
# (This is a simplified version of the one you just added to main1.py)
# In generate_token_file.py

def login_or_reuse(token_file: str = TOKEN_FILE_PATH) -> KiteConnect:
    """
    Handles Kite Connect login by reusing a stored token
    or prompting for a new one if invalid/missing.
    (This is the ROBUST version)
    """
    api_key = os.environ.get("KITE_API_KEY")
    api_secret = os.environ.get("KITE_API_SECRET")

    if not api_key:
        raise SystemExit("FATAL: KITE_API_KEY environment variable not set. (Did you create the .env file?)")
    
    kite = KiteConnect(api_key=api_key)
    access_token = None

    # 1. Try to reuse
    if os.path.exists(token_file):
        L.info(f"Access token file found at {token_file}. Attempting reuse.")
        try:
            with open(token_file, 'r') as f:
                access_token = json.load(f).get('access_token')
            if not access_token:
                raise ValueError("Access token not found in token file.")

            kite.set_access_token(access_token)
            profile = kite.profile()
            L.info(f"Successfully reused access token for user: {profile.get('user_id')}")
            return kite
        except Exception as e:
            L.warning(f"Failed to reuse access token: {e}. Will attempt new login.")
            access_token = None
            
    # 2. Generate new
    if access_token is None:
        if not api_secret:
            raise SystemExit("FATAL: KITE_API_SECRET not set. Cannot generate new token. (Check .env file)")
            
        L.info("Starting new login flow.")
        login_url = kite.login_url()
        print("\n" + "="*80)
        print(f"LOGIN REQUIRED:")
        print(f"1. Open this URL in your browser:\n\n{login_url}")
        print(f"\n2. Log in, and copy the full URL you are redirected to.")
        print(f"3. Paste the *ENTIRE* redirected URL below and press Enter:")
        print("="*80)
        
        try:
            redirect_url = input("Paste redirected URL here: ")
            
            # --- THIS IS THE ROBUST FIX ---
            # It properly parses the URL instead of just splitting the string
            from urllib.parse import urlparse, parse_qs
            parsed_url = urlparse(redirect_url)
            query_params = parse_qs(parsed_url.query)
            request_token = query_params.get('request_token', [None])[0]

            if not request_token:
                raise ValueError("Could not find 'request_token' in the pasted URL. Please try again.")
            # --- END OF FIX ---

            L.info("Request token received. Generating session...")
            session_data = kite.generate_session(request_token, api_secret)
            access_token = session_data.get('access_token')
            
            if not access_token:
                raise ValueError("API did not return an access_token.")
                
            kite.set_access_token(access_token)
            profile = kite.profile()
            L.info(f"Successfully generated new token for user: {profile.get('user_id')}")

            os.makedirs(PERSIST_DIR, exist_ok=True)
            with open(token_file, 'w') as f:
                json.dump({'access_token': access_token}, f)
            L.info(f"New access token saved to {token_file} for future use.")
            
            return kite
        except Exception as e:
            L.critical(f"FATAL: Token generation failed: {e}", exc_info=True)
            sys.exit(1)

def fetch_nifty50_symbols() -> list:
    """
    Fetches the Nifty 50 constituent list from the NSE website.
    """
    L.info(f"Fetching Nifty 50 constituent list from NSE...")
    try:
        # NSE website requires browser-like headers to respond
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Referer': 'https://www.nseindia.com/market-data/live-equity-market',
        }
        
        # Use a session to handle cookies
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5) # Hit main page to get cookies
        
        response = session.get(NSE_NIFTY50_URL, headers=headers, timeout=10)
        response.raise_for_status() # Raise an error for bad responses
        
        data = response.json()
        symbols = [item['symbol'] for item in data['marketDeptData']]
        
        L.info(f"Successfully fetched {len(symbols)} symbols from NSE.")
        return symbols
    except Exception as e:
        L.critical(f"FATAL: Could not fetch Nifty 50 list from NSE: {e}")
        L.critical("This might be due to a change in the NSE website URL or API.")
        L.critical("Please check the 'NSE_NIFTY50_URL' constant in this script.")
        return []

def main():
    load_dotenv()
    
    try: # <-- Start of the block
        # 1. Log in to Kite
        L.info("Logging in to Kite...")
        kite = login_or_reuse()
        
        # 2. Get Nifty 50 Symbols from NSE
        nifty50_symbols = fetch_nifty50_symbols()
        if not nifty50_symbols:
            sys.exit("Could not proceed without Nifty 50 symbol list.")
            
        # 3. Get all NSE instruments from Kite
        L.info("Fetching full NSE instrument list from Kite... (This may take a moment)")
        try: # This nested try/except is fine
            all_instruments = kite.instruments(exchange='NSE')
            L.info(f"Fetched {len(all_instruments)} total instruments from Kite.")
        except Exception as e:
            L.critical(f"FATAL: Could not fetch instruments from Kite: {e}")
            sys.exit(1)
            
        # 4. Create a DataFrame for easy searching
        inst_df = pd.DataFrame(all_instruments)
        
        # Filter for "EQ" (Equity) segment only
        eq_df = inst_df[inst_df['segment'] == 'NSE-EQ'].set_index('tradingsymbol')
        
        # 5. Cross-reference to find tokens
        nifty50_tokens = []
        missing_symbols = []
        
        L.info("Matching NSE symbols to Kite instrument tokens...")
        for symbol in nifty50_symbols:
            try:
                # Find the stock in the 'tradingsymbol' index
                token = eq_df.loc[symbol, 'instrument_token']
                nifty50_tokens.append(int(token))
            except KeyError:
                L.warning(f"Could not find instrument token for symbol: {symbol}")
                missing_symbols.append(symbol)
            except Exception as e:
                L.error(f"Error processing symbol {symbol}: {e}")
                
        L.info(f"Successfully found {len(nifty50_tokens)} tokens.")
        if missing_symbols:
            L.warning(f"Could not find tokens for: {', '.join(missing_symbols)}")
            
        # 6. Save the final list to JSON
        if nifty50_tokens:
            with open(OUTPUT_JSON_FILE, 'w') as f:
                json.dump(nifty50_tokens, f, indent=2)
            L.info(f"SUCCESS! Created '{OUTPUT_JSON_FILE}' with {len(nifty50_tokens)} tokens.")
        else:
            L.error("No tokens were found. File was not created.")

    # --- THIS IS THE FIX ---
    # Add these 'except' blocks to match the 'try' at the top
    except SystemExit as e:
        # This will catch the sys.exit() calls cleanly
        L.error(f"Script halted: {e}")
    except Exception as e:
        # This catches any other unexpected error
        L.critical(f"An unexpected error occurred in main: {e}", exc_info=True)
    # --- END OF FIX ---

if __name__ == "__main__":
    main()