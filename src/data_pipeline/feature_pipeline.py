# src/data_pipeline/feature_pipeline.py
import pandas as pd
import sqlite3
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_feature_store(db_path: str):
    """Initializes the SQLite database and creates the feature tables."""
    logging.info(f"Initializing feature store at {db_path}...")
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Create user_features table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS user_features (
            user_id INTEGER PRIMARY KEY,
            total_views INTEGER,
            total_addtocarts INTEGER,
            total_transactions INTEGER,
            last_seen_ts TIMESTAMP
        )
    ''')

    # Create item_features table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS item_features (
            item_id INTEGER PRIMARY KEY,
            total_views INTEGER,
            total_addtocarts INTEGER,
            total_transactions INTEGER
        )
    ''')
    
    con.commit()
    con.close()
    logging.info("Feature store initialized successfully.")

def run_feature_pipeline(processed_data_path: str, db_path: str):
    """
    Computes and loads user and item features into the SQLite feature store.

    Args:
        processed_data_path (str): Path to the processed Parquet file from ETL.
        db_path (str): Path to the SQLite database file.
    """
    logging.info("Starting feature pipeline...")
    
    processed_path = Path(processed_data_path)
    if not processed_path.exists():
        logging.error(f"Processed data file not found at {processed_path}. Run ETL first.")
        return

    # Initialize DB if it doesn't exist
    if not Path(db_path).exists():
        create_feature_store(db_path)
        
    df = pd.read_parquet(processed_path)

    # --- Feature Engineering for Users ---
    logging.info("Engineering user features...")
    user_features = df.groupby('user_id').agg(
        total_views=('event', lambda x: (x == 'view').sum()),
        total_addtocarts=('event', lambda x: (x == 'addtocart').sum()),
        total_transactions=('event', lambda x: (x == 'transaction').sum()),
        last_seen_ts=('timestamp_dt', 'max')
    ).reset_index()
    
    # --- Feature Engineering for Items ---
    logging.info("Engineering item features...")
    item_features = df.groupby('item_id').agg(
        total_views=('event', lambda x: (x == 'view').sum()),
        total_addtocarts=('event', lambda x: (x == 'addtocart').sum()),
        total_transactions=('event', lambda x: (x == 'transaction').sum())
    ).reset_index()

    # --- Load Features into SQLite ---
    logging.info(f"Loading features into {db_path}...")
    con = sqlite3.connect(db_path)
    
    # Use 'replace' to update existing records or insert new ones
    user_features.to_sql('user_features', con, if_exists='replace', index=False)
    item_features.to_sql('item_features', con, if_exists='replace', index=False)
    
    con.commit()
    con.close()
    logging.info("Feature pipeline completed successfully.")

if __name__ == '__main__':
    PROCESSED_DATA_PATH = 'data/processed/processed_events.parquet'
    DB_PATH = 'feature_store.db'
    run_feature_pipeline(PROCESSED_DATA_PATH, DB_PATH)