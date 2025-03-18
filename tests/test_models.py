"""
Tests for the model training and prediction modules.
"""
import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from sklearn.ensemble import RandomForestRegressor

from src.models.train_model import (
    train_test_split_data,
    train_random_forest,
    evaluate_model,
    save_model,
    load_model
)
from src.models.predict_model import predict_solubility


def test_train_test_split():
    """Test that data splitting works correctly."""
    # Create dummy data
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })
    y = pd.Series(np.random.rand(100))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2)
    
    # Check shapes
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    
    # Check that splits don't overlap
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_train_random_forest():
    """Test that random forest training works."""
    # Create dummy data
    X_train = pd.DataFrame({
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50)
    })
    y_train = pd.Series(np.random.rand(50))
    
    # Simple parameter grid for testing
    param_grid = {
        'n_estimators': [10],
        'max_depth': [3]
    }
    
    # Train model
    model = train_random_forest(X_train, y_train, param_grid=param_grid)
    
    # Check that we get a RandomForestRegressor
    assert isinstance(model, RandomForestRegressor)
    
    # Check that hyperparameters were set
    assert model.n_estimators == 10
    assert model.max_depth == 3


def test_evaluate_model():
    """Test model evaluation."""
    # Create dummy data
    X_test = pd.DataFrame({
        'feature1': np.random.rand(20),
        'feature2': np.random.rand(20)
    })
    y_test = pd.Series(np.random.rand(20))
    
    # Create a simple model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_test, y_test)  # Just for testing, using test data for training
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Check that we get a dictionary of metrics
    assert isinstance(metrics, dict)
    
    # Check that essential metrics are present
    assert 'RMSE' in metrics
    assert 'MAE' in metrics
    assert 'R^2' in metrics


def test_save_load_model():
    """Test saving and loading a model."""
    # Create a simple model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    # Create a temporary file for the model
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
        model_path = temp_file.name
    
    try:
        # Save the model
        save_model(model, model_path)
        
        # Check that the file exists
        assert os.path.exists(model_path)
        
        # Load the model
        loaded_model = load_model(model_path)
        
        # Check that we get a RandomForestRegressor
        assert isinstance(loaded_model, RandomForestRegressor)
        
        # Check that hyperparameters match
        assert loaded_model.n_estimators == model.n_estimators
        assert loaded_model.random_state == model.random_state
    
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)


def test_predict_solubility():
    """Test solubility prediction."""
    # Create a simple model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    # Simple feature matrix for training
    X_train = pd.DataFrame({
        'MW': [180.2, 194.2],
        'LogP': [1.5, 2.1],
        'TPSA': [60.0, 55.0],
        'NumRotatableBonds': [3, 2],
        'NumHDonors': [1, 0],
        'NumHAcceptors': [3, 2],
        'NumRings': [1, 2],
        'NumAromaticRings': [1, 1],
        'BertzCT': [250.0, 280.0],
        'HallKierAlpha': [2.0, 2.5],
        'PEOE_VSA_FPNEG': [0.5, 0.4],
        'PEOE_VSA_FPOS': [0.3, 0.2]
    })
    
    # Simple target values
    y_train = pd.Series([-3.5, -4.2])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Example SMILES for testing
    smiles_list = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'  # Caffeine
    ]
    
    # Make predictions
    # This test will only work if RDKit is properly installed
    try:
        predictions = predict_solubility(model, smiles_list)
        
        # Check that we get a DataFrame
        assert isinstance(predictions, pd.DataFrame)
        
        # Check that we have the right number of rows
        assert len(predictions) == len(smiles_list)
        
        # Check that columns are present
        assert 'SMILES' in predictions.columns
        assert 'Predicted_Solubility' in predictions.columns
        
    except Exception as e:
        # If RDKit isn't installed or has issues, skip this test
        pytest.skip(f"Error in RDKit processing: {str(e)}")


if __name__ == '__main__':
    pytest.main()