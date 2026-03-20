"""
Model Training Module
Handles training of carbon emission forecasting models
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Tuple, Dict
import pickle


class EmissionModel:
    """
    Carbon Emission Forecasting Model using Linear Regression
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        self.metrics = {}
        
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the forecasting model on historical emission data
        
        Args:
            df: DataFrame with 'Year' and 'Emission' columns
            
        Returns:
            Dictionary containing training metrics
        """
        # Prepare features and target
        X = df[['Year']].values
        y = df['Emission'].values
        
        # Train the model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate metrics
        y_pred = self.model.predict(X)
        self.metrics = {
            'r2_score': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'training_samples': len(df)
        }
        
        return self.metrics
    
    def get_model_params(self) -> Dict[str, float]:
        """
        Get model parameters
        
        Returns:
            Dictionary with slope and intercept
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        return {
            'slope': float(self.model.coef_[0]),
            'intercept': float(self.model.intercept_)
        }
    
    def save_model(self, filepath: str):
        """
        Save trained model to file
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath: str):
        """
        Load trained model from file
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True


def train_emission_model(df: pd.DataFrame) -> EmissionModel:
    """
    Convenience function to train a new emission model
    
    Args:
        df: DataFrame with historical emission data
        
    Returns:
        Trained EmissionModel instance
    """
    model = EmissionModel()
    model.train(df)
    return model
