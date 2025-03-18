"""
Module for making predictions with trained models.
"""
import pandas as pd
import numpy as np
from rdkit import Chem

from src.features.molecular_descriptors import generate_features_df


def predict_solubility(model, smiles_list, scaler=None):
    """
    Predict solubility for a list of SMILES strings.
    
    Parameters:
    -----------
    model : object
        Trained model with predict method
    smiles_list : list
        List of SMILES strings
    scaler : object, optional
        Fitted scaler for feature normalization
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with SMILES and predicted solubility
    """
    # Generate features for input SMILES
    features_df = generate_features_df(smiles_list)
    
    # Store SMILES separately
    smiles = features_df['SMILES']
    
    # Drop SMILES column for prediction
    X = features_df.drop('SMILES', axis=1)
    
    # Apply scaling if provided
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'SMILES': smiles,
        'Predicted_Solubility': predictions
    })
    
    return results


def batch_predict(model, input_file, output_file, smiles_col='SMILES', scaler=None):
    """
    Predict solubility for all compounds in an input file.
    
    Parameters:
    -----------
    model : object
        Trained model with predict method
    input_file : str
        Path to input CSV file with SMILES
    output_file : str
        Path to save output predictions
    smiles_col : str, optional
        Name of column containing SMILES strings
    scaler : object, optional
        Fitted scaler for feature normalization
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with SMILES and predicted solubility
    """
    # Read input file
    input_df = pd.read_csv(input_file)
    
    # Get SMILES list
    smiles_list = input_df[smiles_col].tolist()
    
    # Make predictions
    results = predict_solubility(model, smiles_list, scaler)
    
    # Save predictions to file
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return results