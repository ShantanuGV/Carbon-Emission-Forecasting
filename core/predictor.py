"""
Prediction Module
Handles emission forecasting for future years
"""

import pandas as pd
import numpy as np
from typing import Optional
from .model import EmissionModel


class EmissionPredictor:
    """
    Predicts future carbon emissions using trained model
    """
    
    def __init__(self, model: EmissionModel):
        """
        Initialize predictor with trained model
        
        Args:
            model: Trained EmissionModel instance
        """
        if not model.is_trained:
            raise ValueError("Model must be trained before prediction")
        self.model = model
    
    def predict_future(self, 
                       last_year: int, 
                       years_ahead: int = 10) -> pd.DataFrame:
        """
        Predict emissions for future years
        
        Args:
            last_year: Last year in historical data
            years_ahead: Number of years to forecast (default: 10)
            
        Returns:
            DataFrame with Year and Predicted_Emission columns
        """
        # Generate future years
        future_years = np.arange(last_year + 1, last_year + years_ahead + 1)
        
        # Predict emissions
        X_future = future_years.reshape(-1, 1)
        predictions = self.model.model.predict(X_future)
        
        # Create DataFrame
        forecast_df = pd.DataFrame({
            'Year': future_years,
            'Predicted_Emission': predictions
        })
        
        return forecast_df
    
    def predict_specific_year(self, year: int) -> float:
        """
        Predict emission for a specific year
        
        Args:
            year: Target year for prediction
            
        Returns:
            Predicted emission value
        """
        X = np.array([[year]])
        prediction = self.model.model.predict(X)[0]
        return float(prediction)
    
    def get_baseline_forecast(self, 
                             historical_df: pd.DataFrame, 
                             years_ahead: int = 10) -> pd.DataFrame:
        """
        Get complete baseline forecast including historical and future data
        
        Args:
            historical_df: Historical emission data
            years_ahead: Number of years to forecast
            
        Returns:
            DataFrame with both historical and predicted emissions
        """
        last_year = int(historical_df['Year'].max())
        
        # Get future predictions
        future_df = self.predict_future(last_year, years_ahead)
        
        # Combine historical and future
        historical_renamed = historical_df.copy()
        historical_renamed.columns = ['Year', 'Predicted_Emission']
        
        baseline_df = pd.concat([
            historical_renamed,
            future_df
        ], ignore_index=True)
        
        return baseline_df


def create_predictor(model: EmissionModel) -> EmissionPredictor:
    """
    Convenience function to create a predictor
    
    Args:
        model: Trained EmissionModel
        
    Returns:
        EmissionPredictor instance
    """
    return EmissionPredictor(model)


def quick_forecast(historical_df: pd.DataFrame, 
                   model: EmissionModel, 
                   years_ahead: int = 10) -> pd.DataFrame:
    """
    Quick forecast function combining predictor creation and prediction
    
    Args:
        historical_df: Historical emission data
        model: Trained model
        years_ahead: Years to forecast
        
    Returns:
        Complete forecast DataFrame
    """
    predictor = EmissionPredictor(model)
    return predictor.get_baseline_forecast(historical_df, years_ahead)
