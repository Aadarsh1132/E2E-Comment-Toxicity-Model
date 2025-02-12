import mlflow
import logging
from src.data_loading import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Set up MLflow tracking."""
    try:
        config = load_config()
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        logger.info("MLflow tracking set up successfully.")
    except Exception as e:
        logger.error(f"Error setting up MLflow: {e}")
        raise