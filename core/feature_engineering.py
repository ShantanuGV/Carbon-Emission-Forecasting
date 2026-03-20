"""
Feature Engineering Module
Combines structural and policy factors for multi-factor emission modeling
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """
    Handles feature engineering for multi-factor emission modeling
    """
    
    # Define feature categories
    POLICY_FEATURES = [
        'Renewable_Percent',
        'Fossil_Percent'
    ]
    
    STRUCTURAL_FEATURES = [
        'Population_Million',
        'Urbanization_Rate',
        'Forest_Cover_Percent',
        'Energy_Demand_Index',
        'Transport_Index',
        'Industrial_Production_Index'
    ]
    
    GROWTH_FEATURES = [
        'Industrial_Growth'
    ]
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
    
    def load_multifactor_data(self, filepath: str) -> pd.DataFrame:
        """
        Load multi-factor emission dataset
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with all factors
        """
        try:
            df = pd.read_csv(filepath)
            
            # Validate required columns
            required_cols = ['Year', 'Emission'] + self.POLICY_FEATURES + self.STRUCTURAL_FEATURES
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Sort by year
            df = df.sort_values('Year').reset_index(drop=True)
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading multi-factor data: {str(e)}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using interpolation and forward fill
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        # Interpolate numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                # Linear interpolation
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                # Forward fill any remaining
                df[col] = df[col].ffill()
                # Backward fill any remaining
                df[col] = df[col].bfill()
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features for better modeling
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with additional interaction features
        """
        df = df.copy()
        
        # Energy intensity (energy demand per capita)
        if 'Energy_Demand_Index' in df.columns and 'Population_Million' in df.columns:
            df['Energy_Intensity'] = df['Energy_Demand_Index'] / (df['Population_Million'] / 1000)
        
        # Urbanization-transport interaction
        if 'Urbanization_Rate' in df.columns and 'Transport_Index' in df.columns:
            df['Urban_Transport_Factor'] = df['Urbanization_Rate'] * df['Transport_Index'] / 100
        
        # Fossil dependency index
        if 'Fossil_Percent' in df.columns and 'Industrial_Production_Index' in df.columns:
            df['Fossil_Dependency'] = df['Fossil_Percent'] * df['Industrial_Production_Index'] / 100
        
        # Carbon sink capacity (forest cover effect)
        if 'Forest_Cover_Percent' in df.columns:
            df['Carbon_Sink_Capacity'] = df['Forest_Cover_Percent'] * 1.5  # Amplify forest effect
        
        # Renewable penetration rate
        if 'Renewable_Percent' in df.columns and 'Energy_Demand_Index' in df.columns:
            df['Renewable_Penetration'] = df['Renewable_Percent'] * df['Energy_Demand_Index'] / 100
        
        return df
    
    def prepare_features(self, 
                        df: pd.DataFrame, 
                        include_interactions: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for model training
        
        Args:
            df: Raw data DataFrame
            include_interactions: Whether to include interaction features
            
        Returns:
            Tuple of (feature DataFrame, feature column names)
        """
        df = df.copy()
        
        # Create interaction features if requested
        if include_interactions:
            df = self.create_interaction_features(df)
        
        # Select feature columns (Removed 'Year' to allow policy factors to drive causal projection)
        feature_cols = self.POLICY_FEATURES + self.STRUCTURAL_FEATURES + self.GROWTH_FEATURES
        
        # Add interaction features if they exist
        if include_interactions:
            interaction_cols = [
                'Energy_Intensity',
                'Urban_Transport_Factor',
                'Fossil_Dependency',
                'Carbon_Sink_Capacity',
                'Renewable_Penetration'
            ]
            feature_cols += [col for col in interaction_cols if col in df.columns]
        
        # Store feature columns
        self.feature_columns = feature_cols
        
        # Extract features
        X = df[feature_cols].copy()
        
        return X, feature_cols
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using StandardScaler
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            Scaled feature DataFrame
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transform")
            X_scaled = self.scaler.transform(X)
        
        # Convert back to DataFrame
        X_scaled_df = pd.DataFrame(
            X_scaled,
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled_df
    
    def project_structural_factors(self, 
                                   df: pd.DataFrame,
                                   years_ahead: int = 10,
                                   growth_rates: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Project structural factors into the future
        
        Args:
            df: Historical data
            years_ahead: Number of years to project
            growth_rates: Custom growth rates for factors (optional)
            
        Returns:
            DataFrame with projected structural factors
        """
        last_year = int(df['Year'].max())
        future_years = np.arange(last_year + 1, last_year + years_ahead + 1)
        
        # Default growth rates (annual %)
        default_growth_rates = {
            'Population_Million': 0.8,  # Slowing population growth
            'Urbanization_Rate': 0.5,  # Gradual urbanization
            'Forest_Cover_Percent': -0.2,  # Slight deforestation
            'Energy_Demand_Index': 1.5,  # Growing energy demand
            'Transport_Index': 1.2,  # Growing transport
            'Industrial_Production_Index': 1.8,  # Industrial growth
        }
        
        # Use custom rates if provided
        if growth_rates:
            default_growth_rates.update(growth_rates)
        
        # Get last known values
        last_values = df.iloc[-1].to_dict()
        
        # Project each factor
        projected_data = []
        for i, year in enumerate(future_years, start=1):
            row = {'Year': year}
            
            for factor, growth_rate in default_growth_rates.items():
                if factor in last_values:
                    # Compound growth
                    projected_value = last_values[factor] * ((1 + growth_rate / 100) ** i)
                    
                    # Apply constraints
                    if factor == 'Urbanization_Rate':
                        projected_value = min(projected_value, 95.0)  # Cap at 95%
                    elif factor == 'Forest_Cover_Percent':
                        projected_value = max(projected_value, 15.0)  # Floor at 15%
                    elif 'Percent' in factor:
                        projected_value = np.clip(projected_value, 0, 100)
                    
                    row[factor] = projected_value
            
            projected_data.append(row)
        
        projected_df = pd.DataFrame(projected_data)
        
        return projected_df
    
    def get_feature_importance_names(self) -> List[str]:
        """
        Get human-readable feature names for importance display
        
        Returns:
            List of feature names
        """
        name_mapping = {
            'Year': 'Time Trend',
            'Renewable_Percent': 'Renewable Energy %',
            'Fossil_Percent': 'Fossil Fuel %',
            'Industrial_Growth': 'Industrial Growth Rate',
            'Population_Million': 'Population',
            'Urbanization_Rate': 'Urbanization Rate',
            'Forest_Cover_Percent': 'Forest Cover',
            'Energy_Demand_Index': 'Energy Demand',
            'Transport_Index': 'Transport Activity',
            'Industrial_Production_Index': 'Industrial Production',
            'Energy_Intensity': 'Energy Intensity',
            'Urban_Transport_Factor': 'Urban Transport',
            'Fossil_Dependency': 'Fossil Dependency',
            'Carbon_Sink_Capacity': 'Carbon Sink Capacity',
            'Renewable_Penetration': 'Renewable Penetration'
        }
        
        return [name_mapping.get(col, col) for col in self.feature_columns]
    
    def get_structural_baseline(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get current structural factor baseline values
        
        Args:
            df: Historical data
            
        Returns:
            Dictionary of current structural values
        """
        latest = df.iloc[-1]
        
        baseline = {
            'population_million': latest.get('Population_Million', 0),
            'urbanization_rate': latest.get('Urbanization_Rate', 0),
            'forest_cover_percent': latest.get('Forest_Cover_Percent', 0),
            'energy_demand_index': latest.get('Energy_Demand_Index', 0),
            'transport_index': latest.get('Transport_Index', 0),
            'industrial_production_index': latest.get('Industrial_Production_Index', 0),
            'renewable_percent': latest.get('Renewable_Percent', 0),
            'fossil_percent': latest.get('Fossil_Percent', 0)
        }
        
        return baseline


def load_and_engineer_features(filepath: str, 
                               include_interactions: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Convenience function to load data and engineer features
    
    Args:
        filepath: Path to multi-factor CSV
        include_interactions: Whether to create interaction features
        
    Returns:
        Tuple of (full DataFrame, feature DataFrame, feature names)
    """
    engineer = FeatureEngineer()
    df = engineer.load_multifactor_data(filepath)
    X, feature_names = engineer.prepare_features(df, include_interactions)
    
    return df, X, feature_names
