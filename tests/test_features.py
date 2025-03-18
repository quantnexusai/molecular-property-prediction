"""
Tests for the molecular descriptors feature generation.
"""
import pytest
import pandas as pd
from rdkit import Chem
from src.features.molecular_descriptors import generate_features, generate_features_df


def test_generate_features():
    """Test that feature generation works for a valid SMILES."""
    # Aspirin
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    features = generate_features(smiles)
    
    # Check that we get a dictionary of features
    assert isinstance(features, dict)
    
    # Check that essential features are present
    assert 'MW' in features
    assert 'LogP' in features
    assert 'TPSA' in features
    
    # Check that values are reasonable
    assert 150 < features['MW'] < 190  # Aspirin MW is around 180
    assert isinstance(features['NumRings'], int)
    assert features['NumRings'] >= 1


def test_generate_features_invalid_smiles():
    """Test that feature generation returns None for invalid SMILES."""
    # Invalid SMILES
    smiles = 'invalid_smiles_string'
    features = generate_features(smiles)
    
    # Should return None for invalid SMILES
    assert features is None


def test_generate_features_df():
    """Test feature generation for a list of SMILES."""
    # List of valid SMILES
    smiles_list = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CCC1(CC)C(=O)NC(=O)N(C)C1=O'  # Phenobarbital
    ]
    
    # Generate features DataFrame
    df = generate_features_df(smiles_list)
    
    # Check that we get a DataFrame
    assert isinstance(df, pd.DataFrame)
    
    # Check that we have the right number of rows
    assert len(df) == len(smiles_list)
    
    # Check that SMILES column is present
    assert 'SMILES' in df.columns
    
    # Check that essential feature columns are present
    assert 'MW' in df.columns
    assert 'LogP' in df.columns
    assert 'TPSA' in df.columns


def test_generate_features_df_with_invalid_smiles():
    """Test feature generation with a mix of valid and invalid SMILES."""
    # Mix of valid and invalid SMILES
    smiles_list = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'invalid_smiles_1',
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'invalid_smiles_2'
    ]
    
    # Generate features DataFrame
    df = generate_features_df(smiles_list)
    
    # Check that we only have rows for valid SMILES
    assert len(df) == 2
    
    # Check that SMILES column contains only valid SMILES
    assert all(Chem.MolFromSmiles(smiles) is not None for smiles in df['SMILES'])


if __name__ == '__main__':
    pytest.main()