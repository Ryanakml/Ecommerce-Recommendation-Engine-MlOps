# src/data_pipeline/etl.py

import os
import kagglehub
import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def run_etl(raw_data_path: str, processed_data_path: str):
    """
    Runs the ETL process to clean and prepare the Retailrocket events data.

    Args:
        raw_data_path (str): Path to the raw events.csv file.
        processed_data_path (str): Path to save the processed Parquet file.
    """
    logging.info("ETL PROCESS STARTED")

    # Define paths
    raw_path = Path(raw_data_path)
    processed_path = Path(processed_data_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # EXTRACT
    logging.info(f"[EXTRACT] Reading data from: {raw_path}")
    try:
        events_df = pd.read_csv(raw_path)
    except FileNotFoundError:
        logging.error(f"File not found at {raw_path}. Please download the dataset.")
        return

    logging.info(f"[EXTRACT] Dataset loaded. Shape: {events_df.shape}, Columns: {list(events_df.columns)}")

    # TRANSFORM
    logging.info("[TRANSFORM] Starting data transformation...")

    # 1. Rename columns
    logging.info("[TRANSFORM] Renaming columns for clarity...")
    events_df.rename(columns={'visitorid': 'user_id', 'itemid': 'item_id'}, inplace=True)

    # 2. Convert timestamp
    logging.info("[TRANSFORM] Converting 'timestamp' to datetime...")
    events_df['timestamp_dt'] = pd.to_datetime(events_df['timestamp'], unit='ms')

    # 3. Drop mostly-null column
    logging.info("[TRANSFORM] Dropping 'transactionid' column cause mostly null...")
    if 'transactionid' in events_df.columns:
        null_ratio = events_df['transactionid'].isna().mean()
        logging.info(f"[TRANSFORM] Dropping 'transactionid' (null ratio: {null_ratio:.2%})...")
        events_df.drop(columns=['transactionid'], inplace=True)

    # 4. Add event strength
    logging.info("[TRANSFORM] Adding 'event_strength' feature...")
    event_strength = {
        'view': 1.0,
        'addtocart': 2.0,
        'transaction': 4.0,
    }
    events_df['event_strength'] = events_df['event'].map(event_strength)

    # 5. Filter users/items with few interactions
    logging.info("[TRANSFORM] Filtering users and items with fewer than 5 interactions...")
    min_user_interactions = 5
    min_item_interactions = 5

    while True:
        user_counts = events_df['user_id'].value_counts()
        item_counts = events_df['item_id'].value_counts()

        users_to_keep = user_counts[user_counts >= min_user_interactions].index
        items_to_keep = item_counts[item_counts >= min_item_interactions].index

        original_shape = events_df.shape
        events_df = events_df[events_df['user_id'].isin(users_to_keep)]
        events_df = events_df[events_df['item_id'].isin(items_to_keep)]

        if original_shape == events_df.shape:
            break

    logging.info(f"[TRANSFORM] Final dataset shape: {events_df.shape}")
    logging.info(f"[TRANSFORM] Unique users: {events_df['user_id'].nunique()}")
    logging.info(f"[TRANSFORM] Unique items: {events_df['item_id'].nunique()}")

    # LOAD
    logging.info(f"[LOAD] Saving processed data to {processed_path}...")
    events_df.to_parquet(processed_path, index=False)

    logging.info("ETL PROCESS COMPLETED SUCCESSFULLY")

if __name__ == '__main__':
    # Download dataset from kaggle
    path = kagglehub.dataset_download("retailrocket/ecommerce-dataset")
    print("Path to dataset files:", path)

    RAW_DATA_PATH = os.path.join(path, 'events.csv')
    PROCESSED_DATA_PATH = 'data/processed/processed_events.parquet'
    run_etl(RAW_DATA_PATH, PROCESSED_DATA_PATH)