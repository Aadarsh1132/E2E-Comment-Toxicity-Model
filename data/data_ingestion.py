import os
import pandas as pd

def load_data(file_path):
    """Load the dataset."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the dataset."""
    X = df['comment_text']
    y = df[df.columns[2:]].values
    return X, y