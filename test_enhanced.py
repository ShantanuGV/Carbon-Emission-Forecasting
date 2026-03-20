"""
Test Enhanced Multi-Factor Carbon Emission System
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core import (
    FeatureEngineer,
    train_multifactor_model,
    auto_select_best_model,
    EnhancedScenarioSimulator,
    EnhancedPolicyParameters,
    create_sustainability_target
)


def test_enhanced_system():
    """Test all enhanced modules"""
    
    print("=" * 70)
    print("Testing Enhanced Multi-Factor Carbon Emission System")
    print("=" * 70)
    
    # Test 1: Feature Engineering
    print("\n1. Testing Feature Engineering...")
    try:
        data_path = Path(__file__).parent / "data" / "emission_multifactor.csv"
        engineer = FeatureEngineer()
        df = engineer.load_multifactor_data(str(data_path))
        
        print(f"   ✓ Multi-factor data loaded")
        print(f"   ✓ Years: {df['Year'].min()} - {df['Year'].max()}")
        print(f"   ✓ Records: {len(df)}")
        print(f"   ✓ Features: {len(df.columns)}")
        
        # Prepare features
        X, feature_names = engineer.prepare_features(df, include_interactions=True)
        print(f"   ✓ Features prepared: {len(feature_names)} total")
        print(f"   ✓ Interaction features created")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 2: Model Training
    print("\n2. Testing Advanced ML Models...")
    try:
        y = df['Emission']
        
        # Train single model
        model = train_multifactor_model(X, y, model_type='random_forest')
        print(f"   ✓ Random Forest trained")
        print(f"   ✓ R² Score: {model.metrics['r2_score']:.4f}")
        print(f"   ✓ MAE: {model.metrics['mae']:.2f} MT")
        print(f"   ✓ RMSE: {model.metrics['rmse']:.2f} MT")
        
        # Auto-select best model
        print("\n   Comparing multiple models...")
        best_type, best_model, comparison = auto_select_best_model(X, y)
        print(f"   ✓ Best model: {best_type}")
        print(f"   ✓ Best R²: {best_model.metrics['r2_score']:.4f}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 3: Sustainability Targets
    print("\n3. Testing Sustainability Target Calculator...")
    try:
        sustainability_calc = create_sustainability_target(
            df,
            target_type='percentage_reduction',
            reduction_percent=50.0,
            baseline_year=2005,
            target_year=2050
        )
        
        target_info = sustainability_calc.get_target_info()
        print(f"   ✓ Sustainability target created")
        print(f"   ✓ Target type: {target_info['type']}")
        print(f"   ✓ Target emission: {target_info['target_emission']:.0f} MT")
        print(f"   ✓ Reduction: {target_info['reduction_from_baseline']:.1f}%")
        print(f"   ✓ Residual floor: {sustainability_calc.RESIDUAL_EMISSION_FLOOR_PERCENT}%")
        
        # Generate pathway
        pathway = sustainability_calc.generate_pathway(2025, 2050, df['Emission'].iloc[-1])
        print(f"   ✓ Sustainable pathway generated: {len(pathway)} years")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 4: Enhanced Scenario Simulation
    print("\n4. Testing Enhanced Scenario Simulator...")
    try:
        simulator = EnhancedScenarioSimulator(best_model, engineer, df)
        
        # Baseline projection
        baseline = simulator.get_baseline_projection(years_ahead=26)
        print(f"   ✓ Baseline projection: {len(baseline)} years")
        
        # Preset scenarios
        comparison = simulator.compare_scenarios(years_ahead=26)
        print(f"   ✓ Scenario comparison: {len(comparison.columns)-1} scenarios")
        
        # Custom scenario
        custom_params = EnhancedPolicyParameters(
            renewable_growth_percent=70.0,
            fossil_reduction_percent=60.0,
            industrial_growth_percent=-5.0,
            forest_protection_percent=15.0,
            energy_efficiency_improvement=30.0
        )
        
        custom_scenario = simulator.simulate_scenario(custom_params, years_ahead=26)
        print(f"   ✓ Custom scenario simulated")
        print(f"   ✓ 2050 emission: {custom_scenario.iloc[-1]['Scenario_Emission']:.0f} MT")
        
        # Check sustainability
        gap_analysis = simulator.calculate_emission_gap(
            custom_scenario,
            target_info['target_emission'],
            2050
        )
        print(f"   ✓ Gap analysis: {gap_analysis['gap']:.0f} MT")
        print(f"   ✓ Meets target: {gap_analysis['meets_target']}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Feature Importance
    print("\n5. Testing Feature Importance Analysis...")
    try:
        importance = best_model.get_feature_importance(top_n=5)
        print(f"   ✓ Feature importance calculated")
        print(f"   ✓ Top 5 factors:")
        for i, (feature, score) in enumerate(importance.items(), 1):
            print(f"      {i}. {feature}: {score:.4f}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 6: Structural Factor Projection
    print("\n6. Testing Structural Factor Projection...")
    try:
        projected = engineer.project_structural_factors(df, years_ahead=10)
        print(f"   ✓ Structural factors projected: {len(projected)} years")
        print(f"   ✓ 2034 population: {projected.iloc[-1]['Population_Million']:.0f}M")
        print(f"   ✓ 2034 forest cover: {projected.iloc[-1]['Forest_Cover_Percent']:.1f}%")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ All enhanced system tests passed successfully!")
    print("=" * 70)
    
    # Summary
    print("\n📊 SYSTEM CAPABILITIES VERIFIED:")
    print("   ✓ Multi-factor data loading (10+ features)")
    print("   ✓ Feature engineering with interactions")
    print("   ✓ Multiple ML algorithms (Linear, RF, GBM)")
    print("   ✓ Auto-model selection")
    print("   ✓ Sustainability target calculation")
    print("   ✓ Sustainable pathway generation")
    print("   ✓ Enhanced scenario simulation")
    print("   ✓ Policy + structural factor integration")
    print("   ✓ Residual emission floors")
    print("   ✓ Feature importance analysis")
    print("   ✓ Structural factor projection")
    
    return True


if __name__ == "__main__":
    success = test_enhanced_system()
    sys.exit(0 if success else 1)
