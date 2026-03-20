"""
Data Loader Module
Handles loading and cleaning of carbon emission datasets
"""

import pandas as pd
import numpy as np
from typing import Optional


def load_emission_data(filepath: str) -> pd.DataFrame:
    """
    Load carbon emission dataset from CSV file
    
    Args:
        filepath: Path to the CSV file containing emission data
        
    Returns:
        Cleaned pandas DataFrame with Year and Emission columns
    """
    try:
        # Load the CSV file
        df = pd.read_csv(filepath)
        
        # Ensure required columns exist
        if 'Year' not in df.columns or 'Emission' not in df.columns:
            raise ValueError("Dataset must contain 'Year' and 'Emission' columns")
        
        # Keep only required columns
        df = df[['Year', 'Emission']].copy()
        
        # Clean missing values
        df = df.dropna()
        
        # Ensure Year is integer and Emission is float
        df['Year'] = df['Year'].astype(int)
        df['Emission'] = df['Emission'].astype(float)
        
        # Sort by year
        df = df.sort_values('Year').reset_index(drop=True)
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the emission data
    
    Args:
        df: DataFrame with emission data
        
    Returns:
        Dictionary containing summary statistics
    """
    return {
        'start_year': int(df['Year'].min()),
        'end_year': int(df['Year'].max()),
        'total_years': len(df),
        'min_emission': float(df['Emission'].min()),
        'max_emission': float(df['Emission'].max()),
        'avg_emission': float(df['Emission'].mean()),
        'current_emission': float(df.iloc[-1]['Emission'])
    }


def fill_missing_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in missing years with interpolated values
    
    Args:
        df: DataFrame with emission data
        
    Returns:
        DataFrame with all years filled
    """
    start_year = df['Year'].min()
    end_year = df['Year'].max()
    
    # Create complete year range
    all_years = pd.DataFrame({'Year': range(start_year, end_year + 1)})
    
    # Merge and interpolate
    df_complete = all_years.merge(df, on='Year', how='left')
    df_complete['Emission'] = df_complete['Emission'].interpolate(method='linear')
    
    return df_complete
