# src/data_pipeline/etl.py

import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_etl(raw_data_path: str, processed_data_path: str):
    """
    Runs the ETL process to clean and prepare the Retailrocket events data.

    Args:
        raw_data_path (str): Path to the raw events.csv file.
        processed_data_path (str): Path to save the processed Parquet file.
    """
    logging.info("Starting ETL process...")

    # Define paths
    raw_path = Path(raw_data_path)
    processed_path = Path(processed_data_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # --- EXTRACT ---
    logging.info(f"Extracting data from {raw_path}...")
    try:
        events_df = pd.read_csv(raw_path)
    except FileNotFoundError:
        logging.error(f"Raw data file not found at {raw_path}. Please download the dataset.")
        return

    logging.info(f"Initial dataset shape: {events_df.shape}")

    # --- TRANSFORM ---
    logging.info("Transforming data...")

    # 1. Rename columns for clarity
    events_df.rename(columns={'visitorid': 'user_id', 'itemid': 'item_id'}, inplace=True)

    # 2. Convert timestamp to datetime
    events_df['timestamp_dt'] = pd.to_datetime(events_df['timestamp'], unit='ms')

    # 3. Create a confidence score based on event type
    # This quantifies user intent: transaction > addtocart > view
    event_strength = {
        'view': 1.0,
        'addtocart': 2.0,
        'transaction': 4.0,
    }
    events_df['event_strength'] = events_df['event'].map(event_strength)

    # 4. Filter out noise: users and items with few interactions
    # This is a critical step for collaborative filtering models
    min_user_interactions = 5
    min_item_interactions = 5

    while True:
        # Count interactions for users and items
        user_counts = events_df['user_id'].value_counts()
        item_counts = events_df['item_id'].value_counts()

        # Identify users and items to keep
        users_to_keep = user_counts[user_counts >= min_user_interactions].index
        items_to_keep = item_counts[item_counts >= min_item_interactions].index

        # Store original shape for comparison
        original_shape = events_df.shape

        # Filter the DataFrame
        events_df = events_df[events_df['user_id'].isin(users_to_keep)]
        events_df = events_df[events_df['item_id'].isin(items_to_keep)]
        
        logging.info(f"Filtering... Shape changed from {original_shape} to {events_df.shape}")

        # If no more rows are being removed, break the loop
        if original_shape == events_df.shape:
            break

    logging.info(f"Filtered dataset shape: {events_df.shape}")
    logging.info(f"Number of unique users: {events_df['user_id'].nunique()}")
    logging.info(f"Number of unique items: {events_df['item_id'].nunique()}")

    # --- LOAD ---
    logging.info(f"Loading processed data to {processed_path}...")
    events_df.to_parquet(processed_path, index=False)
    logging.info("ETL process completed successfully.")

if __name__ == '__main__':
    # This allows the script to be run directly
    # In a real pipeline, these paths would be configured or passed as arguments
    RAW_DATA_PATH = 'data/raw/events.csv'
    PROCESSED_DATA_PATH = 'data/processed/processed_events.parquet'
    run_etl(RAW_DATA_PATH, PROCESSED_DATA_PATH)