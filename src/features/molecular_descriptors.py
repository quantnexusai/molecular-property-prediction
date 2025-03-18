"""
Module for calculating molecular descriptors using RDKit.
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf


def generate_features(smiles):
    """
    Generate molecular descriptors for a given SMILES string.
    
    Parameters:
    -----------
    smiles : str
        SMILES representation of a molecule
        
    Returns:
    --------
    dict
        Dictionary of molecular descriptors
    """
    # Create RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    features = {}
    
    # Basic descriptors
    features['MW'] = Descriptors.MolWt(mol)
    features['LogP'] = Descriptors.MolLogP(mol)
    features['TPSA'] = Descriptors.TPSA(mol)
    features['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    features['NumHDonors'] = Lipinski.NumHDonors(mol)
    features['NumHAcceptors'] = Lipinski.NumHAcceptors(mol)
    
    # Ring information
    features['NumRings'] = mol.GetRingInfo().NumRings()
    features['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
    
    # Topological properties
    features['BertzCT'] = Descriptors.BertzCT(mol)
    features['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)
    
    # Surface area-related descriptors (more widely available)
    features['LabuteASA'] = Descriptors.LabuteASA(mol)
    features['TPSA'] = MolSurf.TPSA(mol)
    
    return features
    
    return features


def generate_features_df(smiles_list):
    """
    Generate molecular descriptors for a list of SMILES strings.
    
    Parameters:
    -----------
    smiles_list : list
        List of SMILES strings
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame of molecular descriptors
    """
    feature_dicts = []
    valid_smiles = []
    
    for smiles in smiles_list:
        features = generate_features(smiles)
        if features is not None:
            feature_dicts.append(features)
            valid_smiles.append(smiles)
        else:
            print(f"Warning: Invalid SMILES string: {smiles}")
    
    # Create DataFrame from list of feature dictionaries
    df = pd.DataFrame(feature_dicts)
    
    # Add SMILES as a column
    df['SMILES'] = valid_smiles
    
    return df