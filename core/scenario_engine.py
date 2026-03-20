"""
Enhanced Scenario Engine Module
Integrates policy and structural factors for realistic emission scenarios
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from .feature_engineering import FeatureEngineer
from .model_training import MultiFactorEmissionModel


@dataclass
class EnhancedPolicyParameters:
    """
    Enhanced policy parameters including structural factors
    """
    # Policy factors (controllable)
    renewable_growth_percent: float = 0.0
    fossil_reduction_percent: float = 0.0
    industrial_growth_percent: float = 0.0
    forest_protection_percent: float = 0.0  # NEW: Forest conservation
    
    # Structural factor adjustments (slower-changing)
    population_growth_adjustment: float = 0.0  # Adjustment to baseline growth
    urbanization_rate_adjustment: float = 0.0
    energy_efficiency_improvement: float = 0.0  # % improvement in energy intensity
    
    # Policy timing
    policy_start_year: int = 2025
    
    def __post_init__(self):
        """Validate parameters"""
        if not -500 <= self.renewable_growth_percent <= 500:
            raise ValueError("Renewable growth must be between -500 and 500")
        if not -500 <= self.fossil_reduction_percent <= 500:
            raise ValueError("Fossil reduction must be between -500 and 500")


class EnhancedScenarioSimulator:
    """
    Advanced scenario simulator with structural and policy factors
    """
    
    def __init__(self, 
                 model: MultiFactorEmissionModel,
                 feature_engineer: FeatureEngineer,
                 historical_df: pd.DataFrame):
        """
        Initialize simulator
        
        Args:
            model: Trained multi-factor model
            feature_engineer: Feature engineering instance
            historical_df: Historical data with all factors
        """
        self.model = model
        self.engineer = feature_engineer
        self.historical_df = historical_df
        self.scenarios = self._define_enhanced_scenarios()
        
        # Residual emission floor (10% of current)
        self.residual_floor = historical_df['Emission'].iloc[-1] * 0.10
    
    def _define_enhanced_scenarios(self) -> Dict[str, EnhancedPolicyParameters]:
        """
        Define enhanced preset scenarios
        
        Returns:
            Dictionary of scenario parameters
        """
        return {
            'best_case': EnhancedPolicyParameters(
                renewable_growth_percent=150.0, # Highly ambitious target
                fossil_reduction_percent=120.0, # Radical fossil phase-out
                industrial_growth_percent=-20.0, # Strong green industry shift
                forest_protection_percent=40.0,  # Massive reforestation
                population_growth_adjustment=-0.5,
                urbanization_rate_adjustment=-0.3,
                energy_efficiency_improvement=50.0  # 50% efficiency gain
            ),
            'average_case': EnhancedPolicyParameters(
                renewable_growth_percent=80.0,
                fossil_reduction_percent=60.0,
                industrial_growth_percent=0.0,
                forest_protection_percent=15.0,
                population_growth_adjustment=-0.1,
                urbanization_rate_adjustment=-0.1,
                energy_efficiency_improvement=25.0
            ),
            'worst_case': EnhancedPolicyParameters(
                renewable_growth_percent=10.0, # Sluggish transition
                fossil_reduction_percent=-5.0, # Fossil increase
                industrial_growth_percent=40.0, # High emission growth
                forest_protection_percent=-15.0, # High deforestation
                population_growth_adjustment=0.3,
                urbanization_rate_adjustment=0.3,
                energy_efficiency_improvement=5.0
            )
        }
    
    def project_scenario_features(self,
                                  params: EnhancedPolicyParameters,
                                  years_ahead: int = 10) -> pd.DataFrame:
        """
        Project all features under a scenario
        
        Args:
            params: Policy parameters
            years_ahead: Number of years to project
            
        Returns:
            DataFrame with projected features
        """
        last_year = int(self.historical_df['Year'].max())
        future_years = np.arange(last_year + 1, last_year + years_ahead + 1)
        
        # Get last known values
        last_values = self.historical_df.iloc[-1].to_dict()
        
        projected_data = []
        
        for i, year in enumerate(future_years, start=1):
            row = {'Year': year}
            
            # Determine if policy is active
            policy_active = year >= params.policy_start_year
            years_since_policy = max(0, year - params.policy_start_year + 1)
            
            # Project renewable energy
            if policy_active:
                renewable_growth = params.renewable_growth_percent / 100
                renewable_value = last_values['Renewable_Percent'] * (1 + renewable_growth * years_since_policy * 0.05)
                renewable_value = np.clip(renewable_value, 0, 95)  # Cap at 95%
            else:
                # Baseline growth
                renewable_value = last_values['Renewable_Percent'] * (1.02 ** i)
                renewable_value = min(renewable_value, 95)
            
            row['Renewable_Percent'] = renewable_value
            
            # Project fossil fuel usage (inverse of renewable + residual)
            if policy_active:
                fossil_reduction = params.fossil_reduction_percent / 100
                fossil_value = last_values['Fossil_Percent'] * (1 - fossil_reduction * years_since_policy * 0.05)
                fossil_value = max(fossil_value, 5)  # Minimum 5% (residual)
            else:
                fossil_value = last_values['Fossil_Percent'] * (0.98 ** i)
                fossil_value = max(fossil_value, 5)
            
            row['Fossil_Percent'] = fossil_value
            
            # Project industrial growth
            baseline_ind_growth = 0.018
            if policy_active:
                # Slider provides a percentage change relative to baseline growth (e.g., +35% means base * 1.35)
                ind_growth = baseline_ind_growth * (1 + params.industrial_growth_percent / 100)
            else:
                ind_growth = baseline_ind_growth
                
            row['Industrial_Growth'] = ind_growth * 100
            
            # Project population
            base_pop_growth = 0.008 + (params.population_growth_adjustment / 100)
            pop_value = last_values['Population_Million'] * ((1 + base_pop_growth) ** i)
            row['Population_Million'] = pop_value
            
            # Project urbanization
            base_urban_growth = 0.005 + (params.urbanization_rate_adjustment / 100)
            urban_value = last_values['Urbanization_Rate'] * ((1 + base_urban_growth) ** i)
            urban_value = min(urban_value, 95)
            row['Urbanization_Rate'] = urban_value
            
            # Project forest cover
            if policy_active:
                forest_change = params.forest_protection_percent / 100
                forest_value = last_values['Forest_Cover_Percent'] * (1 + forest_change * years_since_policy * 0.03)
            else:
                forest_value = last_values['Forest_Cover_Percent'] * (0.998 ** i)  # Slight decline
            
            forest_value = np.clip(forest_value, 10, 50)
            row['Forest_Cover_Percent'] = forest_value
            
            # Project energy demand (with efficiency improvements)
            base_energy_growth = 0.015
            if policy_active:
                efficiency_factor = 1 - (params.energy_efficiency_improvement / 100) * 0.02 * years_since_policy
            else:
                efficiency_factor = 1.0
            
            energy_value = last_values['Energy_Demand_Index'] * ((1 + base_energy_growth) ** i) * efficiency_factor
            row['Energy_Demand_Index'] = energy_value
            
            # Project transport
            transport_growth = 0.012
            transport_value = last_values['Transport_Index'] * ((1 + transport_growth) ** i)
            row['Transport_Index'] = transport_value
            
            # Project industrial production
            ind_prod_growth = ind_growth
            ind_prod_value = last_values['Industrial_Production_Index'] * ((1 + ind_prod_growth) ** i)
            row['Industrial_Production_Index'] = ind_prod_value
            
            projected_data.append(row)
        
        projected_df = pd.DataFrame(projected_data)
        
        # Create interaction features
        projected_df = self.engineer.create_interaction_features(projected_df)
        
        return projected_df
    
    def simulate_scenario(self,
                         params: EnhancedPolicyParameters,
                         years_ahead: int = 10) -> pd.DataFrame:
        """
        Simulate emission scenario with all factors
        
        Args:
            params: Policy parameters
            years_ahead: Forecast horizon
            
        Returns:
            DataFrame with scenario emissions
        """
        # Project features
        future_features = self.project_scenario_features(params, years_ahead)
        
        # Prepare features for prediction
        X_future, _ = self.engineer.prepare_features(
            future_features,
            include_interactions=True
        )
        
        # Ensure all required features are present
        for col in self.model.feature_names:
            if col not in X_future.columns:
                X_future[col] = 0  # Fill missing with 0
        
        # Reorder columns to match training
        X_future = X_future[self.model.feature_names]
        
        # Predict emissions
        predictions = self.model.predict(X_future)
        
        # Apply residual floor
        predictions = np.maximum(predictions, self.residual_floor)
        
        # Combine with features and add realistic 'zik zak' volatility (approx 2% annual variation)
        result_df = future_features[['Year']].copy()
        
        # Consistent seed but unique per year to simulate economic cycles
        np.random.seed(42)
        volatility = 1.0 + (np.random.standard_normal(len(predictions)) * 0.025)
        result_df['Scenario_Emission'] = predictions * volatility
        
        # Add key factors for display
        result_df['Renewable_Percent'] = future_features['Renewable_Percent']
        result_df['Fossil_Percent'] = future_features['Fossil_Percent']
        result_df['Forest_Cover_Percent'] = future_features['Forest_Cover_Percent']
        
        # Prepend the last historical data point so that charts connect past and future seamlessly
        last_hist = self.historical_df.iloc[-1]
        last_hist_df = pd.DataFrame([{
            'Year': int(last_hist['Year']),
            'Scenario_Emission': last_hist['Emission'],
            'Renewable_Percent': last_hist.get('Renewable_Percent', 0),
            'Fossil_Percent': last_hist.get('Fossil_Percent', 0),
            'Forest_Cover_Percent': last_hist.get('Forest_Cover_Percent', 0)
        }])
        
        result_df = pd.concat([last_hist_df, result_df], ignore_index=True)
        
        return result_df
    
    def get_baseline_projection(self, years_ahead: int = 10) -> pd.DataFrame:
        """
        Get baseline projection (using historical average trends)
        
        Args:
            years_ahead: Forecast horizon
            
        Returns:
            DataFrame with baseline emissions
        """
        # Calculate recent historical trends to use as baseline (last 10 years)
        latest_10 = self.historical_df.tail(10)
        
        # We use sliders as % change relative to zero (for Baseline, we want the current real world trend)
        # For renewables/fossil/forest, the policy logic uses growth scalars on last value
        
        # Renewable: find annual growth %
        avg_ren_growth = latest_10['Renewable_Percent'].pct_change().mean() * 100
        # Fossil: find annual reduction % (positive = reduction)
        avg_foss_reduction = -(latest_10['Fossil_Percent'].pct_change().mean() * 100)
        # Forest: annual growth
        avg_forest_growth = latest_10['Forest_Cover_Percent'].pct_change().mean() * 100
        
        baseline_params = EnhancedPolicyParameters(
            renewable_growth_percent=float(np.nan_to_num(avg_ren_growth)),
            fossil_reduction_percent=float(np.nan_to_num(avg_foss_reduction)),
            industrial_growth_percent=0.0, # 0.0 means baseline 1.8% in our logic
            forest_protection_percent=float(np.nan_to_num(avg_forest_growth)),
            policy_start_year=int(self.historical_df['Year'].max()) + 1
        )
        
        baseline_df = self.simulate_scenario(baseline_params, years_ahead)
        baseline_df.rename(columns={'Scenario_Emission': 'Current Trend Forecast'}, inplace=True)
        
        return baseline_df
    
    def compare_scenarios(self, years_ahead: int = 10) -> pd.DataFrame:
        """
        Compare all preset scenarios
        
        Args:
            years_ahead: Forecast horizon
            
        Returns:
            DataFrame with all scenarios
        """
        # Get baseline
        result_df = self.get_baseline_projection(years_ahead)
        
        # Add preset scenarios
        for name, params in self.scenarios.items():
            scenario_df = self.simulate_scenario(params, years_ahead)
            result_df[name.replace('_', ' ').title()] = scenario_df['Scenario_Emission']
        
        return result_df[['Year', 'Current Trend Forecast', 'Best Case', 'Average Case', 'Worst Case']]
    
    def get_preset_scenario(self, scenario_name: str) -> EnhancedPolicyParameters:
        """
        Get preset scenario parameters
        
        Args:
            scenario_name: Name of scenario
            
        Returns:
            Policy parameters
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        return self.scenarios[scenario_name]
    
    def calculate_emission_gap(self,
                               scenario_df: pd.DataFrame,
                               target_emission: float,
                               target_year: int) -> Dict:
        """
        Calculate gap between scenario and target
        
        Args:
            scenario_df: Scenario results
            target_emission: Target emission level
            target_year: Target year
            
        Returns:
            Dictionary with gap analysis
        """
        if target_year not in scenario_df['Year'].values:
            target_year = int(scenario_df['Year'].max())
        
        scenario_emission = scenario_df[
            scenario_df['Year'] == target_year
        ]['Scenario_Emission'].values[0]
        
        gap = scenario_emission - target_emission
        gap_percent = (gap / target_emission) * 100
        
        return {
            'target_year': target_year,
            'scenario_emission': scenario_emission,
            'target_emission': target_emission,
            'gap': gap,
            'gap_percent': gap_percent,
            'meets_target': gap <= 0
        }


def create_enhanced_simulator(model: MultiFactorEmissionModel,
                              engineer: FeatureEngineer,
                              historical_df: pd.DataFrame) -> EnhancedScenarioSimulator:
    """
    Convenience function to create enhanced simulator
    
    Args:
        model: Trained model
        engineer: Feature engineer
        historical_df: Historical data
        
    Returns:
        EnhancedScenarioSimulator instance
    """
    return EnhancedScenarioSimulator(model, engineer, historical_df)
