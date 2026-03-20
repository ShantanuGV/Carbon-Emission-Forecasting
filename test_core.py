"""
Test script to verify core modules functionality
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from core import (
    load_emission_data,
    get_data_summary,
    train_emission_model,
    create_predictor,
    ScenarioSimulator,
    PolicyParameters,
    simulate_custom_scenario
)


def test_core_modules():
    """Test all core modules"""
    
    print("=" * 60)
    print("Testing Carbon Emission Forecasting System")
    print("=" * 60)
    
    # Test 1: Data Loading
    print("\n1. Testing Data Loader...")
    try:
        data_path = Path(__file__).parent / "data" / "emission.csv"
        df = load_emission_data(str(data_path))
        summary = get_data_summary(df)
        
        print(f"   ✓ Data loaded successfully")
        print(f"   ✓ Years: {summary['start_year']} - {summary['end_year']}")
        print(f"   ✓ Total records: {summary['total_years']}")
        print(f"   ✓ Current emission: {summary['current_emission']:.2f} MT")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 2: Model Training
    print("\n2. Testing Model Training...")
    try:
        model = train_emission_model(df)
        metrics = model.metrics
        params = model.get_model_params()
        
        print(f"   ✓ Model trained successfully")
        print(f"   ✓ R² Score: {metrics['r2_score']:.4f}")
        print(f"   ✓ MAE: {metrics['mae']:.2f} MT")
        print(f"   ✓ Slope: {params['slope']:.4f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 3: Prediction
    print("\n3. Testing Predictor...")
    try:
        predictor = create_predictor(model)
        baseline_df = predictor.get_baseline_forecast(df, years_ahead=10)
        
        last_year = df['Year'].max()
        future_2035 = predictor.predict_specific_year(2035)
        
        print(f"   ✓ Predictions generated successfully")
        print(f"   ✓ Forecast length: {len(baseline_df)} years")
        print(f"   ✓ 2035 Prediction: {future_2035:.2f} MT")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 4: Scenario Simulation
    print("\n4. Testing Scenario Simulator...")
    try:
        simulator = ScenarioSimulator()
        
        # Test preset scenarios
        best_case = simulator.get_preset_scenario('best_case')
        best_case_df = simulator.apply_scenario(baseline_df, best_case)
        
        # Test custom scenario
        custom_df = simulate_custom_scenario(
            baseline_df,
            renewable_growth=40.0,
            fossil_reduction=50.0,
            industrial_growth=10.0,
            start_year=2025
        )
        
        # Compare scenarios
        comparison_df = simulator.compare_scenarios(baseline_df)
        
        print(f"   ✓ Scenario simulation successful")
        print(f"   ✓ Preset scenarios: {len(simulator.scenarios)}")
        print(f"   ✓ Comparison scenarios: {len(comparison_df.columns) - 1}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 5: Reduction Potential
    print("\n5. Testing Reduction Analysis...")
    try:
        reductions = simulator.calculate_reduction_potential(baseline_df, target_year=2035)
        
        print(f"   ✓ Reduction analysis complete")
        for scenario, data in reductions.items():
            print(f"   ✓ {scenario.replace('_', ' ').title()}: {data['percent_reduction']:.1f}% reduction")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_core_modules()
    sys.exit(0 if success else 1)
