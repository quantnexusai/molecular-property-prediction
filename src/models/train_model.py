"""
Module for training machine learning models.
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target values
    test_size : float, optional
        Proportion of data to use for testing
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_random_forest(X_train, y_train, param_grid=None):
    """
    Train a Random Forest model with optional hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    y_train : pandas.Series
        Training target values
    param_grid : dict, optional
        Parameter grid for GridSearchCV
        
    Returns:
    --------
    sklearn.ensemble.RandomForestRegressor
        Trained model
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    
    # Create base model
    rf = RandomForestRegressor(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=1
    )
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    # Print best parameters and score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    # Return the best model
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : object
        Trained model with predict method
    X_test : pandas.DataFrame
        Test feature matrix
    y_test : pandas.Series
        Test target values
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Return metrics as a dictionary
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R^2': r2
    }
    
    print(f"Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics


def save_model(model, filename):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : object
        Trained model
    filename : str
        Path to save the model
        
    Returns:
    --------
    None
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")


def load_model(filename):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    filename : str
        Path to the saved model
        
    Returns:
    --------
    object
        Trained model
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model