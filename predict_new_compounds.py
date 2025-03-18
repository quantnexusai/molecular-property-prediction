import pandas as pd
from src.models.train_model import load_model
from src.models.predict_model import predict_solubility

# Load the trained model and scaler
model = load_model('models/random_forest_model.pkl')
scaler = load_model('models/scaler.pkl')

# SMILES strings for new compounds
new_smiles = [
    'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    'OC1=CC=CC=C1',  # Phenol
    # Add more compounds of interest
]

# Make predictions
predictions = predict_solubility(model, new_smiles, scaler=scaler)
print(predictions)