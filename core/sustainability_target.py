"""
Sustainability Target Module
Defines and calculates healthy/sustainable emission levels
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class SustainabilityTarget:
    """
    Defines sustainable emission targets
    """
    target_type: str  # 'historical_baseline', 'percentage_reduction', 'net_zero_pathway'
    target_value: float  # Sustainable emission level in MT
    target_year: int  # Year to achieve target
    baseline_year: int = 2005  # Reference year for calculations
    baseline_emission: float = 0.0  # Emission at baseline year
    
    def __post_init__(self):
        """Validate target parameters"""
        valid_types = ['historical_baseline', 'percentage_reduction', 'net_zero_pathway']
        if self.target_type not in valid_types:
            raise ValueError(f"Target type must be one of {valid_types}")


class SustainabilityCalculator:
    """
    Calculates sustainable emission targets and pathways
    """
    
    # Residual emissions floor (cannot go below this)
    # Based on essential activities: agriculture, aviation, some industry
    RESIDUAL_EMISSION_FLOOR_PERCENT = 10  # 10% of baseline
    
    def __init__(self):
        self.target = None
    
    def calculate_historical_baseline(self, 
                                     df: pd.DataFrame, 
                                     safe_year: int = 1990) -> float:
        """
        Calculate sustainable level based on historical safe baseline
        
        Args:
            df: Historical emission data
            safe_year: Year considered as safe baseline
            
        Returns:
            Sustainable emission level
        """
        if safe_year not in df['Year'].values:
            # Use earliest available year
            safe_year = int(df['Year'].min())
        
        baseline_emission = df[df['Year'] == safe_year]['Emission'].values[0]
        
        # Add 10% buffer for population growth since baseline
        sustainable_level = baseline_emission * 1.1
        
        return sustainable_level
    
    def calculate_percentage_reduction(self, 
                                      df: pd.DataFrame, 
                                      baseline_year: int = 2005,
                                      reduction_percent: float = 50.0) -> Tuple[float, float]:
        """
        Calculate target based on percentage reduction from baseline
        
        Args:
            df: Historical emission data
            baseline_year: Reference year
            reduction_percent: Target reduction percentage
            
        Returns:
            Tuple of (baseline_emission, target_emission)
        """
        if baseline_year not in df['Year'].values:
            baseline_year = int(df['Year'].max())
        
        baseline_emission = df[df['Year'] == baseline_year]['Emission'].values[0]
        target_emission = baseline_emission * (1 - reduction_percent / 100)
        
        # Ensure not below residual floor
        residual_floor = baseline_emission * (self.RESIDUAL_EMISSION_FLOOR_PERCENT / 100)
        target_emission = max(target_emission, residual_floor)
        
        return baseline_emission, target_emission
    
    def calculate_net_zero_pathway(self, 
                                  df: pd.DataFrame,
                                  target_year: int = 2050) -> float:
        """
        Calculate net-zero pathway target (with residual emissions)
        
        Args:
            df: Historical emission data
            target_year: Year to achieve net-zero
            
        Returns:
            Residual emission level (not absolute zero)
        """
        current_emission = df['Emission'].iloc[-1]
        
        # Net-zero means residual emissions only
        residual_emission = current_emission * (self.RESIDUAL_EMISSION_FLOOR_PERCENT / 100)
        
        return residual_emission
    
    def create_target(self, 
                     df: pd.DataFrame,
                     target_type: str = 'percentage_reduction',
                     **kwargs) -> SustainabilityTarget:
        """
        Create sustainability target based on type
        
        Args:
            df: Historical emission data
            target_type: Type of target calculation
            **kwargs: Additional parameters for specific target types
            
        Returns:
            SustainabilityTarget instance
        """
        if target_type == 'historical_baseline':
            safe_year = kwargs.get('safe_year', 1990)
            target_value = self.calculate_historical_baseline(df, safe_year)
            baseline_year = safe_year
            baseline_emission = target_value / 1.1
            
        elif target_type == 'percentage_reduction':
            baseline_year = kwargs.get('baseline_year', 2005)
            reduction_percent = kwargs.get('reduction_percent', 50.0)
            baseline_emission, target_value = self.calculate_percentage_reduction(
                df, baseline_year, reduction_percent
            )
            
        elif target_type == 'net_zero_pathway':
            target_year = kwargs.get('target_year', 2050)
            target_value = self.calculate_net_zero_pathway(df, target_year)
            baseline_year = int(df['Year'].max())
            baseline_emission = df['Emission'].iloc[-1]
            
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        self.target = SustainabilityTarget(
            target_type=target_type,
            target_value=target_value,
            target_year=kwargs.get('target_year', 2050),
            baseline_year=baseline_year,
            baseline_emission=baseline_emission
        )
        
        return self.target
    
    def generate_pathway(self, 
                        start_year: int,
                        end_year: int,
                        current_emission: float) -> pd.DataFrame:
        """
        Generate sustainable pathway from current to target
        
        Args:
            start_year: Starting year
            end_year: Target year
            current_emission: Current emission level
            
        Returns:
            DataFrame with sustainable pathway
        """
        if self.target is None:
            raise ValueError("Must create target first using create_target()")
        
        years = np.arange(start_year, end_year + 1)
        n_years = len(years)
        
        # Exponential decay toward target (more realistic than linear)
        decay_rate = -np.log((self.target.target_value / current_emission)) / n_years
        
        pathway_emissions = []
        for i, year in enumerate(years):
            emission = current_emission * np.exp(-decay_rate * i)
            # Ensure doesn't go below target
            emission = max(emission, self.target.target_value)
            pathway_emissions.append(emission)
        
        pathway_df = pd.DataFrame({
            'Year': years,
            'Sustainable_Pathway': pathway_emissions
        })
        
        return pathway_df
    
    def get_target_info(self) -> Dict:
        """
        Get information about current target
        
        Returns:
            Dictionary with target details
        """
        if self.target is None:
            return {}
        
        return {
            'type': self.target.target_type,
            'target_emission': self.target.target_value,
            'target_year': self.target.target_year,
            'baseline_year': self.target.baseline_year,
            'baseline_emission': self.target.baseline_emission,
            'reduction_from_baseline': (
                (self.target.baseline_emission - self.target.target_value) / 
                self.target.baseline_emission * 100
            ),
            'residual_floor_percent': self.RESIDUAL_EMISSION_FLOOR_PERCENT
        }
    
    def check_scenario_sustainability(self, 
                                     scenario_df: pd.DataFrame,
                                     target_year: int = 2050) -> Dict:
        """
        Check if scenario meets sustainability target
        
        Args:
            scenario_df: Scenario emission predictions
            target_year: Year to check
            
        Returns:
            Dictionary with sustainability assessment
        """
        if self.target is None:
            raise ValueError("Must create target first")
        
        if target_year not in scenario_df['Year'].values:
            target_year = int(scenario_df['Year'].max())
        
        scenario_emission = scenario_df[
            scenario_df['Year'] == target_year
        ]['Scenario_Emission'].values[0]
        
        gap = scenario_emission - self.target.target_value
        gap_percent = (gap / self.target.target_value) * 100
        
        meets_target = scenario_emission <= self.target.target_value
        
        return {
            'meets_target': meets_target,
            'scenario_emission': scenario_emission,
            'target_emission': self.target.target_value,
            'gap': gap,
            'gap_percent': gap_percent,
            'target_year': target_year
        }


def create_sustainability_target(df: pd.DataFrame,
                                 target_type: str = 'percentage_reduction',
                                 reduction_percent: float = 50.0,
                                 baseline_year: int = 2005,
                                 target_year: int = 2050) -> SustainabilityCalculator:
    """
    Convenience function to create sustainability calculator with target
    
    Args:
        df: Historical emission data
        target_type: Type of target
        reduction_percent: Reduction percentage (for percentage_reduction type)
        baseline_year: Baseline year
        target_year: Target achievement year
        
    Returns:
        SustainabilityCalculator with target set
    """
    calculator = SustainabilityCalculator()
    calculator.create_target(
        df,
        target_type=target_type,
        reduction_percent=reduction_percent,
        baseline_year=baseline_year,
        target_year=target_year
    )
    return calculator
