import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    """Download the Parkinson's Disease dataset from Kaggle."""
    print("Downloading dataset from Kaggle...")
    
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Create a directory for the dataset
    os.makedirs('kaggle_data', exist_ok=True)
    
    # Download the dataset
    api.dataset_download_files(
        'vikasukani/parkinsons-disease-data-set',
        path='kaggle_data',
        unzip=True
    )
    print("Dataset downloaded successfully!")

def process_dataset():
    """Process the downloaded dataset."""
    # Read the dataset
    dataset_path = 'kaggle_data/parkinsons.data'
    print(f"Reading dataset from: {dataset_path}")
    
    # Read the dataset
    df = pd.read_csv(dataset_path)
    
    # Print dataset info
    print("\nDataset Info:")
    print(f"Shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())
    print("\nSample data:")
    print(df.head())
    
    # Separate features and target
    X = df.drop(['name', 'status'], axis=1)
    y = df['status']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save processed data
    os.makedirs('data', exist_ok=True)
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Positive cases (PD): {sum(y)}")
    print(f"Negative cases (No PD): {len(y) - sum(y)}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Download dataset
    download_dataset()
    
    # Process dataset
    X_train, X_test, y_train, y_test = process_dataset() 