import pandas as pd
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml."""
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def load_data(data_path):
    """Load and preprocess the data."""
    try:
        df = pd.read_csv(data_path)
        logger.info("Data loaded successfully.")
        return df
        print(df.shape)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

print(df.shape)