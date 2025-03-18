"""
Module for visualizing molecular data and model results.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw


def plot_feature_importance(model, feature_names, top_n=15, figsize=(12, 8)):
    """
    Plot feature importance for a trained model.
    
    Parameters:
    -----------
    model : object
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int, optional
        Number of top features to display
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame of features and importances
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Take top N features
    top_features = feature_imp.head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis', ax=ax)
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    
    return fig


def plot_actual_vs_predicted(y_true, y_pred, figsize=(10, 8)):
    """
    Plot actual vs. predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Calculate metrics
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add labels and title
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Actual vs. Predicted Values\nCorrelation: {corr:.4f}, RMSE: {rmse:.4f}')
    
    # Add text box with metrics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, f'Correlation: {corr:.4f}\nRMSE: {rmse:.4f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    return fig


def plot_distributions(y_true, y_pred, figsize=(12, 6)):
    """
    Plot distributions of actual and predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot histograms
    sns.histplot(y_true, kde=True, ax=ax1, color='blue', label='Actual')
    ax1.set_title('Distribution of Actual Values')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    
    sns.histplot(y_pred, kde=True, ax=ax2, color='green', label='Predicted')
    ax2.set_title('Distribution of Predicted Values')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    
    return fig


def plot_molecule_grid(smiles_list, labels=None, molsPerRow=4, figsize=(12, 10)):
    """
    Plot a grid of molecules.
    
    Parameters:
    -----------
    smiles_list : list
        List of SMILES strings
    labels : list, optional
        List of labels for each molecule
    molsPerRow : int, optional
        Number of molecules per row
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Convert SMILES to RDKit molecules
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list if Chem.MolFromSmiles(smiles) is not None]
    
    # Create image
    if labels is not None:
        img = Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(300, 300), legends=labels)
    else:
        img = Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(300, 300))
    
    # Convert to matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis('off')
    
    return fig