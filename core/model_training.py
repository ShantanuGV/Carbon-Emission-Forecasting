"""
Enhanced Model Training Module
Supports multiple ML algorithms for multi-factor emission forecasting
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from typing import Dict, Tuple, Optional, List
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MultiFactorEmissionModel:
    """
    Advanced emission forecasting model supporting multiple algorithms
    """
    
    SUPPORTED_MODELS = {
        'linear': Ridge,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor
    }
    
    def __init__(self, model_type: str = 'random_forest', **model_params):
        """
        Initialize model
        
        Args:
            model_type: Type of model ('linear', 'random_forest', 'gradient_boosting')
            **model_params: Additional parameters for the model
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type must be one of {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_type = model_type
        self.model_params = model_params
        
        # Initialize model with default parameters
        if model_type == 'linear':
            default_params = {'alpha': 100.0} # Strong regularization to prevent runaway forecasts
            default_params.update(model_params)
            self.model = Ridge(**default_params)
        elif model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': 1
            }
            default_params.update(model_params)
            self.model = RandomForestRegressor(**default_params)
        elif model_type == 'gradient_boosting':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }
            default_params.update(model_params)
            self.model = GradientBoostingRegressor(**default_params)
        
        self.is_trained = False
        self.metrics = {}
        self.feature_names = []
        self.feature_importance = None
        self.scaler = StandardScaler()
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the model on multi-factor data
        
        Args:
            X: Feature DataFrame
            y: Target emissions
            
        Returns:
            Dictionary of training metrics
        """
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features internally
        X_processed = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_processed, y)
        
        # --- Scientific Sanity Check for Linear Model ---
        if self.model_type == 'linear':
            # Identify indices
            for i, name in enumerate(self.feature_names):
                # Green/Sink factors must NOT increase emissions
                if any(k in name for k in ['Renewable', 'Forest', 'Sink']):
                    self.model.coef_[i] = -abs(self.model.coef_[i])
                
                # Growth/Consumption factors must NOT decrease emissions
                if any(k in name for k in ['Population', 'Urbanization', 'Transport', 'Industrial', 'Energy', 'Fossil', 'Dependency']):
                    self.model.coef_[i] = abs(self.model.coef_[i])
            
            # Re-calculate intercept to maintain fit at the average point of the SCALED features
            # Scaled features mean is 0 by definition of StandardScaler
            self.model.intercept_ = y.mean()
        
        self.is_trained = True
        
        # Make predictions
        y_pred = self.model.predict(X_processed)
        
        # Calculate metrics
        self.metrics = {
            'r2_score': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mape': mean_absolute_percentage_error(y, y_pred) * 100,
            'training_samples': len(X),
            'n_features': X.shape[1]
        }
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        elif hasattr(self.model, 'coef_'):
            # For linear regression, use absolute coefficients
            self.feature_importance = dict(zip(
                self.feature_names,
                np.abs(self.model.coef_)
            ))
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_processed = self.scaler.transform(X)
        return self.model.predict(X_processed)
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Args:
            top_n: Return only top N features (optional)
            
        Returns:
            Dictionary of feature importances
        """
        if self.feature_importance is None:
            return {}
        
        # Sort by importance
        sorted_importance = dict(
            sorted(self.feature_importance.items(), 
                   key=lambda x: x[1], 
                   reverse=True)
        )
        
        if top_n:
            sorted_importance = dict(list(sorted_importance.items())[:top_n])
        
        return sorted_importance
    
    def get_model_params(self) -> Dict:
        """
        Get model parameters
        
        Returns:
            Dictionary with model information
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        params = {
            'model_type': self.model_type,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names
        }
        
        # Add model-specific parameters
        if self.model_type == 'linear':
            params['coefficients'] = dict(zip(self.feature_names, self.model.coef_))
            params['intercept'] = float(self.model.intercept_)
        elif self.model_type == 'random_forest':
            params['n_estimators'] = self.model.n_estimators
            params['max_depth'] = self.model.max_depth
        elif self.model_type == 'gradient_boosting':
            params['n_estimators'] = self.model.n_estimators
            params['learning_rate'] = self.model.learning_rate
        
        return params
    
    def save_model(self, filepath: str):
        """
        Save trained model
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'scaler': self.scaler
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """
        Load trained model
        
        Args:
            filepath: Path to saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data['metrics']
        self.feature_importance = model_data.get('feature_importance')
        self.scaler = model_data.get('scaler', StandardScaler())
        self.is_trained = True


class ModelComparison:
    """
    Compare multiple models and select the best
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_multiple_models(self, 
                             X: pd.DataFrame, 
                             y: pd.Series,
                             model_types: Optional[List[str]] = None) -> Dict:
        """
        Train multiple model types and compare
        
        Args:
            X: Features
            y: Target
            model_types: List of model types to train (default: all)
            
        Returns:
            Dictionary of results for each model
        """
        if model_types is None:
            model_types = ['linear', 'random_forest', 'gradient_boosting']
        
        for model_type in model_types:
            print(f"Training {model_type} model...")
            
            model = MultiFactorEmissionModel(model_type=model_type)
            metrics = model.train(X, y)
            
            self.models[model_type] = model
            self.results[model_type] = metrics
        
        return self.results
    
    def get_best_model(self, metric: str = 'r2_score') -> Tuple[str, MultiFactorEmissionModel]:
        """
        Get the best performing model
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_type, model)
        """
        if not self.results:
            raise ValueError("Must train models first")
        
        # For R2, higher is better; for MAE/RMSE, lower is better
        reverse = metric in ['r2_score']
        
        best_model_type = max(
            self.results.keys(),
            key=lambda k: self.results[k][metric] if reverse else -self.results[k][metric]
        )
        
        return best_model_type, self.models[best_model_type]
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Get comparison table of all models
        
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df.index.name = 'Model Type'
        
        return comparison_df


def train_multifactor_model(X: pd.DataFrame, 
                            y: pd.Series,
                            model_type: str = 'random_forest') -> MultiFactorEmissionModel:
    """
    Convenience function to train a multi-factor emission model
    
    Args:
        X: Feature DataFrame
        y: Target emissions
        model_type: Type of model to train
        
    Returns:
        Trained model
    """
    model = MultiFactorEmissionModel(model_type=model_type)
    model.train(X, y)
    return model


def auto_select_best_model(X: pd.DataFrame, 
                           y: pd.Series) -> Tuple[str, MultiFactorEmissionModel, pd.DataFrame]:
    """
    Automatically train and select best model
    
    Args:
        X: Features
        y: Target
        
    Returns:
        Tuple of (best_model_type, best_model, comparison_table)
    """
    comparison = ModelComparison()
    comparison.train_multiple_models(X, y)
    best_type, best_model = comparison.get_best_model()
    comparison_table = comparison.get_comparison_table()
    
    return best_type, best_model, comparison_table
