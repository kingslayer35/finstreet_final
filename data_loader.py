"""
Data Preparation: Convert hourly IRCON data to daily OHLCV
"""
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_hourly_data(filepath: str) -> pd.DataFrame:
    """Load the raw hourly CSV data."""
    df = pd.read_csv(filepath)
    
    # Parse datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['date'] = df['Datetime'].dt.date
    
    # Standardize column names
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    logger.info(f"Loaded {len(df)} hourly rows from {filepath}")
    return df


def convert_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly data to daily OHLCV.
    - Open: first of the day
    - High: max of the day
    - Low: min of the day  
    - Close: last of the day
    - Volume: sum of the day
    """
    daily = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Converted to {len(daily)} daily rows")
    return daily


def prepare_data(raw_path: str, output_path: str) -> pd.DataFrame:
    """Main function to prepare daily data from hourly."""
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load and convert
    df_hourly = load_hourly_data(raw_path)
    df_daily = convert_to_daily(df_hourly)
    
    # Save
    df_daily.to_csv(output_path, index=False)
    logger.info(f"Saved daily data to {output_path}")
    
    return df_daily


if __name__ == "__main__":
    from config import RAW_DATA_PATH, DAILY_DATA_PATH
    prepare_data(str(RAW_DATA_PATH), str(DAILY_DATA_PATH))
