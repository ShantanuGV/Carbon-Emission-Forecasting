"""
Scenario Simulation Module
Handles policy-based emission adjustments and scenario analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class PolicyParameters:
    """
    Policy parameters for scenario simulation
    """
    renewable_growth_percent: float = 0.0
    fossil_reduction_percent: float = 0.0
    industrial_growth_percent: float = 0.0
    policy_start_year: int = 2025
    
    def __post_init__(self):
        """Validate parameters"""
        if not -100 <= self.renewable_growth_percent <= 100:
            raise ValueError("Renewable growth must be between -100 and 100")
        if not -100 <= self.fossil_reduction_percent <= 100:
            raise ValueError("Fossil reduction must be between -100 and 100")
        if not -100 <= self.industrial_growth_percent <= 100:
            raise ValueError("Industrial growth must be between -100 and 100")


class ScenarioSimulator:
    """
    Simulates emission scenarios based on policy interventions
    """
    
    # Impact weights for different policy factors
    RENEWABLE_IMPACT = 0.4
    FOSSIL_IMPACT = 0.35
    INDUSTRIAL_IMPACT = 0.25
    
    def __init__(self):
        self.scenarios = self._define_preset_scenarios()
    
    def _define_preset_scenarios(self) -> Dict[str, PolicyParameters]:
        """
        Define preset policy scenarios
        
        Returns:
            Dictionary of scenario names to PolicyParameters
        """
        return {
            'best_case': PolicyParameters(
                renewable_growth_percent=50.0,
                fossil_reduction_percent=60.0,
                industrial_growth_percent=-20.0
            ),
            'average_case': PolicyParameters(
                renewable_growth_percent=25.0,
                fossil_reduction_percent=30.0,
                industrial_growth_percent=10.0
            ),
            'worst_case': PolicyParameters(
                renewable_growth_percent=-10.0,
                fossil_reduction_percent=-20.0,
                industrial_growth_percent=40.0
            )
        }
    
    def calculate_impact_factor(self, params: PolicyParameters) -> float:
        """
        Calculate overall emission impact factor from policy parameters
        
        Args:
            params: Policy parameters
            
        Returns:
            Impact factor (negative = reduction, positive = increase)
        """
        # Convert percentages to decimal multipliers
        renewable_effect = -(params.renewable_growth_percent / 100) * self.RENEWABLE_IMPACT
        fossil_effect = -(params.fossil_reduction_percent / 100) * self.FOSSIL_IMPACT
        industrial_effect = (params.industrial_growth_percent / 100) * self.INDUSTRIAL_IMPACT
        
        # Combined impact factor
        total_impact = renewable_effect + fossil_effect + industrial_effect
        
        return total_impact
    
    def apply_scenario(self, 
                       baseline_df: pd.DataFrame, 
                       params: PolicyParameters) -> pd.DataFrame:
        """
        Apply policy scenario to baseline predictions
        
        Args:
            baseline_df: DataFrame with baseline predictions
            params: Policy parameters to apply
            
        Returns:
            DataFrame with scenario-adjusted emissions
        """
        df = baseline_df.copy()
        
        # Calculate impact factor
        impact_factor = self.calculate_impact_factor(params)
        
        # Apply impact only to years after policy start
        df['Scenario_Emission'] = df.apply(
            lambda row: self._adjust_emission(
                row['Predicted_Emission'],
                row['Year'],
                params.policy_start_year,
                impact_factor
            ),
            axis=1
        )
        
        return df
    
    def _adjust_emission(self, 
                        emission: float, 
                        year: int, 
                        start_year: int, 
                        impact_factor: float) -> float:
        """
        Adjust emission value based on policy impact
        
        Args:
            emission: Original emission value
            year: Year of emission
            start_year: Year policy starts
            impact_factor: Impact multiplier
            
        Returns:
            Adjusted emission value
        """
        if year < start_year:
            return emission
        
        # Calculate years since policy implementation
        years_active = year - start_year + 1
        
        # Progressive impact (compounds over time)
        cumulative_impact = 1 + (impact_factor * years_active * 0.1)
        
        # Ensure emission doesn't go negative
        adjusted = max(0, emission * cumulative_impact)
        
        return adjusted
    
    def get_preset_scenario(self, scenario_name: str) -> PolicyParameters:
        """
        Get preset scenario parameters
        
        Args:
            scenario_name: Name of preset scenario
            
        Returns:
            PolicyParameters for the scenario
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        return self.scenarios[scenario_name]
    
    def compare_scenarios(self, 
                         baseline_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare all preset scenarios
        
        Args:
            baseline_df: Baseline predictions
            
        Returns:
            DataFrame with all scenario comparisons
        """
        result = baseline_df[['Year', 'Predicted_Emission']].copy()
        result.rename(columns={'Predicted_Emission': 'Baseline'}, inplace=True)
        
        for name, params in self.scenarios.items():
            scenario_df = self.apply_scenario(baseline_df, params)
            result[name.replace('_', ' ').title()] = scenario_df['Scenario_Emission']
        
        return result
    
    def calculate_reduction_potential(self, 
                                     baseline_df: pd.DataFrame, 
                                     target_year: int = 2035) -> Dict[str, float]:
        """
        Calculate emission reduction potential for different scenarios
        
        Args:
            baseline_df: Baseline predictions
            target_year: Year to compare
            
        Returns:
            Dictionary of scenario reductions
        """
        # Check if target year exists in baseline
        if target_year not in baseline_df['Year'].values:
            # Use the last available year
            target_year = int(baseline_df['Year'].max())
        
        baseline_emission = baseline_df[baseline_df['Year'] == target_year]['Predicted_Emission'].values[0]
        
        reductions = {}
        for name, params in self.scenarios.items():
            scenario_df = self.apply_scenario(baseline_df, params)
            scenario_emission = scenario_df[scenario_df['Year'] == target_year]['Scenario_Emission'].values[0]
            reduction = baseline_emission - scenario_emission
            reduction_percent = (reduction / baseline_emission) * 100
            
            reductions[name] = {
                'absolute_reduction': reduction,
                'percent_reduction': reduction_percent,
                'final_emission': scenario_emission
            }
        
        return reductions


def simulate_custom_scenario(baseline_df: pd.DataFrame,
                            renewable_growth: float,
                            fossil_reduction: float,
                            industrial_growth: float,
                            start_year: int = 2025) -> pd.DataFrame:
    """
    Convenience function for custom scenario simulation
    
    Args:
        baseline_df: Baseline predictions
        renewable_growth: Renewable energy growth percentage
        fossil_reduction: Fossil fuel reduction percentage
        industrial_growth: Industrial growth percentage
        start_year: Year policy starts
        
    Returns:
        DataFrame with scenario results
    """
    params = PolicyParameters(
        renewable_growth_percent=renewable_growth,
        fossil_reduction_percent=fossil_reduction,
        industrial_growth_percent=industrial_growth,
        policy_start_year=start_year
    )
    
    simulator = ScenarioSimulator()
    return simulator.apply_scenario(baseline_df, params)
