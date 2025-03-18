"""
Main script for molecular property prediction pipeline.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from src.features.molecular_descriptors import generate_features_df
from src.models.train_model import train_test_split_data, train_random_forest, evaluate_model, save_model
from src.models.predict_model import predict_solubility
from src.visualization.visualize import plot_feature_importance, plot_actual_vs_predicted, plot_distributions

# Define paths
DATA_RAW = os.path.join('data', 'raw')
DATA_PROCESSED = os.path.join('data', 'processed')
MODELS_DIR = 'models'

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)


def download_dataset():
    """
    Use the local Delaney solubility dataset.
    """
    print("The Delaney dataset should be located at data/raw/delaney.csv")
    print("If it's not there, please copy it from: /Users/ariharrison/Downloads/delaney-processed.csv")
    print("Using command: cp /Users/ariharrison/Downloads/delaney-processed.csv data/raw/delaney.csv")
    
    # Check if the file exists locally
    file_path = os.path.join(DATA_RAW, 'delaney.csv')
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    else:
        print(f"Error: Dataset not found at {file_path}")
        return None


def process_data(df=None):
    """
    Process the dataset and generate molecular features.
    """
    print("Processing data and generating molecular features...")
    
    if df is None:
        # Load the dataset if not provided
        df_path = os.path.join(DATA_RAW, 'delaney.csv')
        if not os.path.exists(df_path):
            df = download_dataset()
        else:
            df = pd.read_csv(df_path)
    
    if df is None:
        print("Error: Could not load dataset.")
        return None
    
    # Get SMILES and experimental values
    smiles_list = df['smiles'].tolist()
    logS_exp = df['measured log solubility in mols per litre'].values
    
    # Generate molecular features
    features_df = generate_features_df(smiles_list)
    
    # Merge with experimental values
    features_df['logS_exp'] = logS_exp
    
    # Save processed data
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    features_df.to_csv(os.path.join(DATA_PROCESSED, 'molecular_features.csv'), index=False)
    print(f"Processed data saved to {os.path.join(DATA_PROCESSED, 'molecular_features.csv')}")
    
    return features_df


def train_model(df=None):
    """
    Train a machine learning model to predict solubility.
    """
    print("Training model...")
    
    if df is None:
        # Load processed data if not provided
        df_path = os.path.join(DATA_PROCESSED, 'molecular_features.csv')
        if not os.path.exists(df_path):
            print("Processed data not found. Processing data first...")
            df = process_data()
        else:
            df = pd.read_csv(df_path)
    
    if df is None:
        print("Error: Could not load processed data.")
        return None
    
    # Prepare data for modeling
    X = df.drop(['SMILES', 'logS_exp'], axis=1)
    y = df['logS_exp']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split_data(X_scaled, y)
    
    # Train model
    model = train_random_forest(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_model(model, os.path.join(MODELS_DIR, 'random_forest_model.pkl'))
    
    # Save scaler
    save_model(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    
    # Plot feature importance
    fig = plot_feature_importance(model, X.columns)
    fig.savefig(os.path.join('figures', 'feature_importance.png'), dpi=300, bbox_inches='tight')
    
    # Plot actual vs predicted
    y_pred = model.predict(X_test)
    fig = plot_actual_vs_predicted(y_test, y_pred)
    fig.savefig(os.path.join('figures', 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
    
    # Plot distributions
    fig = plot_distributions(y_test, y_pred)
    fig.savefig(os.path.join('figures', 'distributions.png'), dpi=300, bbox_inches='tight')
    
    return model, scaler, metrics


def main():
    """
    Run the full molecular property prediction pipeline.
    """
    print("Starting molecular property prediction pipeline...")
    
    # Create directories
    os.makedirs('figures', exist_ok=True)
    
    # Download and process data
    data = download_dataset()
    processed_data = process_data(data)
    
    # Train and evaluate model
    model, scaler, metrics = train_model(processed_data)
    
    print("Pipeline completed successfully!")
    print(f"Model saved to {os.path.join(MODELS_DIR, 'random_forest_model.pkl')}")
    print(f"Figures saved to 'figures' directory")


if __name__ == "__main__":
    main()