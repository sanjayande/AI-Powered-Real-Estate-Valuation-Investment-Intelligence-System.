import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RealEstateFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for Domain-Specific Real Estate Feature Engineering.
    Strictly operates on predictor variables without target leakage.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # Area ratios & differences
        if 'Carpet_Area_SqFt' in X_out.columns and 'Super_Area_SqFt' in X_out.columns:
            X_out['Carpet_Super_Ratio'] = X_out['Carpet_Area_SqFt'] / (X_out['Super_Area_SqFt'] + 1e-5)
            X_out['Area_Diff_SqFt'] = X_out['Super_Area_SqFt'] - X_out['Carpet_Area_SqFt']
        
        # Room ratios & metrics
        if 'Super_Area_SqFt' in X_out.columns and 'BHK' in X_out.columns:
            X_out['Area_per_BHK'] = X_out['Super_Area_SqFt'] / (X_out['BHK'] + 1e-5)
        
        if 'Super_Area_SqFt' in X_out.columns and 'Bathrooms' in X_out.columns:
            X_out['Area_per_Bathroom'] = X_out['Super_Area_SqFt'] / (X_out['Bathrooms'] + 1e-5)
            
        if 'BHK' in X_out.columns and 'Bathrooms' in X_out.columns:
            X_out['Bath_per_BHK'] = X_out['Bathrooms'] / (X_out['BHK'] + 1e-5)
            X_out['Total_Rooms'] = X_out['BHK'] + X_out['Bathrooms']
            
        # Floor metrics
        if 'Floor_Number' in X_out.columns and 'Total_Floors' in X_out.columns:
            X_out['Floor_Ratio'] = X_out['Floor_Number'] / (X_out['Total_Floors'] + 1e-5)
            X_out['Is_Ground_Floor'] = (X_out['Floor_Number'] == 0).astype(int)
            X_out['Is_Top_Floor'] = (X_out['Floor_Number'] == X_out['Total_Floors']).astype(int)

        # Accessibility & Location composites
        if 'Distance_to_Metro_km' in X_out.columns and 'Distance_to_City_Center_km' in X_out.columns:
            X_out['Total_Distance_km'] = X_out['Distance_to_Metro_km'] + X_out['Distance_to_City_Center_km']
            X_out['Distance_Ratio'] = X_out['Distance_to_Metro_km'] / (X_out['Distance_to_City_Center_km'] + 1e-5)

        # Amenity Score
        amenity_cols = [c for c in ['Parking', 'Lift_Available', 'Gated_Community'] if c in X_out.columns]
        if amenity_cols:
            X_out['Amenity_Score'] = X_out[amenity_cols].sum(axis=1)

        # Age Groups
        if 'Age_of_Property' in X_out.columns:
            X_out['Age_Group'] = pd.cut(
                X_out['Age_of_Property'],
                bins=[-1, 5, 15, 25, 100],
                labels=['New_0-5', 'Modern_5-15', 'Established_15-25', 'Older_25+']
            ).astype(str)

        return X_out
