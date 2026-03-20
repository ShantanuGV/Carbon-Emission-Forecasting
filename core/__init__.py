"""
Core Module Initialization
Enhanced with multi-factor modeling capabilities
"""

# Legacy modules (backward compatibility)
from .data_loader import load_emission_data, get_data_summary, fill_missing_years
from .model import EmissionModel, train_emission_model
from .predictor import EmissionPredictor, create_predictor, quick_forecast
from .scenario import ScenarioSimulator, PolicyParameters, simulate_custom_scenario

# New enhanced modules
from .feature_engineering import (
    FeatureEngineer,
    load_and_engineer_features
)
from .model_training import (
    MultiFactorEmissionModel,
    ModelComparison,
    train_multifactor_model,
    auto_select_best_model
)
from .scenario_engine import (
    EnhancedScenarioSimulator,
    EnhancedPolicyParameters,
    create_enhanced_simulator
)
from .sustainability_target import (
    SustainabilityCalculator,
    SustainabilityTarget,
    create_sustainability_target
)

__all__ = [
    # Legacy exports
    'load_emission_data',
    'get_data_summary',
    'fill_missing_years',
    'EmissionModel',
    'train_emission_model',
    'EmissionPredictor',
    'create_predictor',
    'quick_forecast',
    'ScenarioSimulator',
    'PolicyParameters',
    'simulate_custom_scenario',
    
    # Enhanced exports
    'FeatureEngineer',
    'load_and_engineer_features',
    'MultiFactorEmissionModel',
    'ModelComparison',
    'train_multifactor_model',
    'auto_select_best_model',
    'EnhancedScenarioSimulator',
    'EnhancedPolicyParameters',
    'create_enhanced_simulator',
    'SustainabilityCalculator',
    'SustainabilityTarget',
    'create_sustainability_target'
]
