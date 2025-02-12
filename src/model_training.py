import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, LSTM, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
import mlflow
import mlflow.tensorflow
import logging
import yaml
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    """Load configuration from config.yaml."""
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

# Set up MLflow
def setup_mlflow():
    """Set up MLflow tracking."""
    try:
        config = load_config()
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
        if experiment is None:
            mlflow.create_experiment(config['mlflow']['experiment_name'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        logger.info("MLflow tracking set up successfully.")
    except Exception as e:
        logger.error(f"Error setting up MLflow: {e}")
        raise

# Prepare dataset
def prepare_dataset(df, max_features, output_sequence_length, batch_size, train_split, val_split, test_split):
    """Prepare the TensorFlow dataset."""
    try:
        X = df['comment_text']
        y = df.iloc[:, 2:8].values  # Extract target labels (last 6 columns)

        # Debug: Check shape of y
        print("Shape of y:", y.shape)
        print("First 5 rows of y:\n", y[:5])

        vectorizer = TextVectorization(max_tokens=max_features, output_sequence_length=output_sequence_length, output_mode='int')
        vectorizer.adapt(X.values)
        vectorized_text = vectorizer(X.values)

        dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
        dataset = dataset.cache().shuffle(160000).batch(batch_size).prefetch(8)

        train_size = int(len(dataset) * train_split)
        val_size = int(len(dataset) * val_split)
        test_size = int(len(dataset) * test_split)

        train = dataset.take(train_size)
        val = dataset.skip(train_size).take(val_size)
        test = dataset.skip(train_size + val_size).take(test_size)

        logger.info("Dataset prepared successfully.")
        return train, val, test, vectorizer
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        raise

# Build model
def build_model(max_features):
    """Build and compile the model."""
    try:
        model = Sequential([
            Embedding(max_features + 1, 32),
            Bidirectional(LSTM(32, activation='tanh')),
            Dense(128, activation="relu"),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(6, activation='sigmoid')  # Output layer for 6 classes
        ])

        model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])
        logger.info("Model built and compiled successfully.")
        return model
    except Exception as e:
        logger.error(f"Error building model: {e}")
        raise

# Train model
def train_model(model, train, val, epochs, model_save_path):
    """Train the model and save it."""
    try:
        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            print("✅ Starting model training...")
            history = model.fit(train, epochs=epochs, validation_data=val, verbose=1)
            print("✅ Training completed.")
            model.save(model_save_path)
            mlflow.log_artifact(model_save_path)
            logger.info("Model trained and saved successfully.")
        return history
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

# Main function
def main():
    """Main function to run the pipeline."""
    try:
        # Set up MLflow
        setup_mlflow()

        # Load configuration
        config = load_config()

        # Load data
        df = pd.read_csv(config['data']['data_path'])
        logger.info("Data loaded successfully.")

        # Prepare dataset
        train, val, test, vectorizer = prepare_dataset(
            df,
            config['model']['max_features'],
            config['model']['output_sequence_length'],
            config['model']['batch_size'],
            config['model']['train_split'],
            config['model']['val_split'],
            config['model']['test_split']
        )

        # Build model
        model = build_model(config['model']['max_features'])

        # Train model
        history = train_model(model, train, val, config['model']['epochs'], config['model']['model_save_path'])

        # Evaluate model
        evaluate_model(model, test)

    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")

# Evaluate model
def evaluate_model(model, test):
    """Evaluate the model."""
    try:
        pre = Precision()
        re = Recall()
        acc = CategoricalAccuracy()

        for batch in test.as_numpy_iterator():
            X_true, y_true = batch
            yhat = model.predict(X_true)
            
            y_true = y_true.flatten()
            yhat = yhat.flatten()
            
            pre.update_state(y_true, yhat)
            re.update_state(y_true, yhat)
            acc.update_state(y_true, yhat)

        logger.info(f"Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}")
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

if __name__ == "__main__":
    main()