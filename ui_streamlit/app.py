"""
Carbon Emission Forecasting & Policy Scenario Simulator
Streamlit Dashboard Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add parent directory to path for core module imports
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    load_emission_data,
    get_data_summary,
    train_emission_model,
    create_predictor,
    ScenarioSimulator,
    PolicyParameters,
    simulate_custom_scenario
)

# Page configuration
st.set_page_config(
    page_title="Carbon Emission Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3241;
    }
    .stMetric label {
        color: #8b92a8 !important;
    }
    .stMetric .metric-value {
        color: #ffffff !important;
    }
    h1 {
        color: #4da6ff;
        font-weight: 700;
    }
    h2, h3 {
        color: #66b3ff;
    }
    .scenario-card {
        background: linear-gradient(135deg, #1e2130 0%, #2a2d3e 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #3a3d4e;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_prepare_data(data_path):
    """Load and prepare emission data"""
    df = load_emission_data(data_path)
    summary = get_data_summary(df)
    return df, summary


@st.cache_resource
def train_model(df):
    """Train emission forecasting model"""
    model = train_emission_model(df)
    return model


def create_baseline_chart(baseline_df, historical_df):
    """Create baseline forecast visualization"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_df['Year'],
        y=historical_df['Emission'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#4da6ff', width=3),
        marker=dict(size=6)
    ))
    
    # Future predictions
    future_df = baseline_df[baseline_df['Year'] > historical_df['Year'].max()]
    fig.add_trace(go.Scatter(
        x=future_df['Year'],
        y=future_df['Predicted_Emission'],
        mode='lines+markers',
        name='Baseline Forecast',
        line=dict(color='#ff6b6b', width=3, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Carbon Emission Baseline Forecast',
        xaxis_title='Year',
        yaxis_title='CO₂ Emissions (Million Tonnes)',
        template='plotly_dark',
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_scenario_comparison_chart(comparison_df):
    """Create scenario comparison visualization"""
    fig = go.Figure()
    
    colors = {
        'Baseline': '#8b92a8',
        'Best Case': '#51cf66',
        'Average Case': '#ffd43b',
        'Worst Case': '#ff6b6b'
    }
    
    for column in comparison_df.columns[1:]:
        fig.add_trace(go.Scatter(
            x=comparison_df['Year'],
            y=comparison_df[column],
            mode='lines',
            name=column,
            line=dict(color=colors.get(column, '#4da6ff'), width=3)
        ))
    
    fig.update_layout(
        title='Policy Scenario Comparison',
        xaxis_title='Year',
        yaxis_title='CO₂ Emissions (Million Tonnes)',
        template='plotly_dark',
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_custom_scenario_chart(baseline_df, scenario_df, historical_df):
    """Create custom scenario visualization"""
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=historical_df['Year'],
        y=historical_df['Emission'],
        mode='lines',
        name='Historical',
        line=dict(color='#4da6ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(77, 166, 255, 0.1)'
    ))
    
    # Baseline
    future_baseline = baseline_df[baseline_df['Year'] > historical_df['Year'].max()]
    fig.add_trace(go.Scatter(
        x=future_baseline['Year'],
        y=future_baseline['Predicted_Emission'],
        mode='lines',
        name='Baseline',
        line=dict(color='#ff6b6b', width=2, dash='dash')
    ))
    
    # Custom scenario
    future_scenario = scenario_df[scenario_df['Year'] > historical_df['Year'].max()]
    fig.add_trace(go.Scatter(
        x=future_scenario['Year'],
        y=future_scenario['Scenario_Emission'],
        mode='lines',
        name='Custom Scenario',
        line=dict(color='#51cf66', width=3),
        fill='tozeroy',
        fillcolor='rgba(81, 207, 102, 0.1)'
    ))
    
    fig.update_layout(
        title='Custom Policy Scenario Impact',
        xaxis_title='Year',
        yaxis_title='CO₂ Emissions (Million Tonnes)',
        template='plotly_dark',
        hovermode='x unified',
        height=500
    )
    
    return fig


def main():
    # Header
    st.title("🌍 Carbon Emission Forecasting & Policy Scenario Simulator")
    st.markdown("""
        **Interactive platform for predicting future CO₂ emissions and simulating climate policy impacts**
        
        This system uses machine learning to forecast carbon emissions and allows you to explore 
        different policy scenarios through interactive controls.
    """)
    
    # Data path
    data_path = Path(__file__).parent.parent / "data" / "emission.csv"
    
    try:
        # Load data
        with st.spinner("Loading emission data..."):
            df, summary = load_and_prepare_data(str(data_path))
        
        # Train model
        with st.spinner("Training forecasting model..."):
            model = train_model(df)
            predictor = create_predictor(model)
        
        # Sidebar controls
        st.sidebar.header("⚙️ Forecast Settings")
        
        years_ahead = st.sidebar.slider(
            "Forecast Horizon (years)",
            min_value=5,
            max_value=30,
            value=10,
            step=1
        )
        
        target_year = st.sidebar.selectbox(
            "Target Year for Analysis",
            options=list(range(2030, 2051, 5)),
            index=1
        )
        
        # Generate baseline forecast
        baseline_df = predictor.get_baseline_forecast(df, years_ahead)
        
        # KPI Metrics
        st.header("📊 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Emission",
                f"{summary['current_emission']:.2f} MT",
                delta=None
            )
        
        with col2:
            forecast_2035 = baseline_df[baseline_df['Year'] == target_year]['Predicted_Emission'].values[0]
            st.metric(
                f"{target_year} Forecast",
                f"{forecast_2035:.2f} MT",
                delta=f"{forecast_2035 - summary['current_emission']:.2f} MT"
            )
        
        # Calculate best case reduction
        simulator = ScenarioSimulator()
        best_case_params = simulator.get_preset_scenario('best_case')
        best_case_params.policy_start_year = df['Year'].max() + 1
        best_case_df = simulator.apply_scenario(baseline_df, best_case_params)
        best_case_emission = best_case_df[best_case_df['Year'] == target_year]['Scenario_Emission'].values[0]
        reduction = forecast_2035 - best_case_emission
        
        with col3:
            st.metric(
                "Best Case Reduction",
                f"{reduction:.2f} MT",
                delta=f"-{(reduction/forecast_2035)*100:.1f}%",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Data Coverage",
                f"{summary['total_years']} years",
                delta=f"{summary['start_year']}-{summary['end_year']}"
            )
        
        st.divider()
        
        # Baseline Forecast
        st.header("📈 Baseline Emission Forecast")
        baseline_chart = create_baseline_chart(baseline_df, df)
        st.plotly_chart(baseline_chart, use_container_width=True)
        
        st.divider()
        
        # Scenario Comparison
        st.header("🎯 Policy Scenario Comparison")
        
        st.markdown("""
            Compare three preset policy scenarios to understand potential emission trajectories:
            - **Best Case**: Aggressive renewable adoption + fossil fuel reduction
            - **Average Case**: Moderate policy implementation
            - **Worst Case**: Continued high fossil fuel use + industrial growth
        """)
        
        comparison_df = simulator.compare_scenarios(baseline_df)
        scenario_chart = create_scenario_comparison_chart(comparison_df)
        st.plotly_chart(scenario_chart, use_container_width=True)
        
        # Scenario details
        col1, col2, col3 = st.columns(3)
        
        scenarios_info = {
            'best_case': ('Best Case', col1, '#51cf66'),
            'average_case': ('Average Case', col2, '#ffd43b'),
            'worst_case': ('Worst Case', col3, '#ff6b6b')
        }
        
        for scenario_key, (name, col, color) in scenarios_info.items():
            params = simulator.get_preset_scenario(scenario_key)
            scenario_emission = comparison_df[comparison_df['Year'] == target_year][name].values[0]
            reduction_pct = ((forecast_2035 - scenario_emission) / forecast_2035) * 100
            
            with col:
                st.markdown(f"""
                    <div class="scenario-card">
                        <h3 style="color: {color};">{name}</h3>
                        <p><strong>Renewable Growth:</strong> {params.renewable_growth_percent:+.0f}%</p>
                        <p><strong>Fossil Reduction:</strong> {params.fossil_reduction_percent:+.0f}%</p>
                        <p><strong>Industrial Growth:</strong> {params.industrial_growth_percent:+.0f}%</p>
                        <p><strong>{target_year} Emission:</strong> {scenario_emission:.2f} MT</p>
                        <p><strong>vs Baseline:</strong> {reduction_pct:+.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Custom Scenario Simulator
        st.header("🎛️ Custom Policy Scenario Simulator")
        
        st.markdown("**Adjust the sliders to create your own policy scenario:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            renewable_growth = st.slider(
                "🌱 Renewable Energy Growth (%)",
                min_value=-50.0,
                max_value=100.0,
                value=30.0,
                step=5.0,
                help="Increase in renewable energy adoption"
            )
            
            fossil_reduction = st.slider(
                "⛽ Fossil Fuel Reduction (%)",
                min_value=-50.0,
                max_value=100.0,
                value=40.0,
                step=5.0,
                help="Reduction in fossil fuel usage"
            )
        
        with col2:
            industrial_growth = st.slider(
                "🏭 Industrial Emission Growth (%)",
                min_value=-50.0,
                max_value=100.0,
                value=5.0,
                step=5.0,
                help="Change in industrial emissions"
            )
            
            policy_start = st.slider(
                "📅 Policy Start Year",
                min_value=int(df['Year'].max()) + 1,
                max_value=int(df['Year'].max()) + 10,
                value=int(df['Year'].max()) + 1,
                step=1
            )
        
        # Generate custom scenario
        custom_scenario_df = simulate_custom_scenario(
            baseline_df,
            renewable_growth,
            fossil_reduction,
            industrial_growth,
            policy_start
        )
        
        # Custom scenario visualization
        custom_chart = create_custom_scenario_chart(baseline_df, custom_scenario_df, df)
        st.plotly_chart(custom_chart, use_container_width=True)
        
        # Custom scenario impact
        custom_emission = custom_scenario_df[custom_scenario_df['Year'] == target_year]['Scenario_Emission'].values[0]
        custom_reduction = forecast_2035 - custom_emission
        custom_reduction_pct = (custom_reduction / forecast_2035) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                f"Baseline {target_year}",
                f"{forecast_2035:.2f} MT"
            )
        
        with col2:
            st.metric(
                f"Custom Scenario {target_year}",
                f"{custom_emission:.2f} MT",
                delta=f"{custom_reduction:+.2f} MT"
            )
        
        with col3:
            st.metric(
                "Reduction Achieved",
                f"{abs(custom_reduction_pct):.1f}%",
                delta=f"{custom_reduction:.2f} MT",
                delta_color="inverse" if custom_reduction > 0 else "normal"
            )
        
        # Data table
        with st.expander("📋 View Detailed Forecast Data"):
            display_df = custom_scenario_df[['Year', 'Predicted_Emission', 'Scenario_Emission']].copy()
            display_df.columns = ['Year', 'Baseline Emission', 'Scenario Emission']
            display_df['Difference'] = display_df['Baseline Emission'] - display_df['Scenario Emission']
            st.dataframe(display_df, use_container_width=True)
        
        # Model information
        with st.expander("ℹ️ Model Information"):
            metrics = model.metrics
            params = model.get_model_params()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Performance:**")
                st.write(f"- R² Score: {metrics['r2_score']:.4f}")
                st.write(f"- MAE: {metrics['mae']:.2f} MT")
                st.write(f"- RMSE: {metrics['rmse']:.2f} MT")
                st.write(f"- Training Samples: {metrics['training_samples']}")
            
            with col2:
                st.markdown("**Model Parameters:**")
                st.write(f"- Slope: {params['slope']:.4f}")
                st.write(f"- Intercept: {params['intercept']:.2f}")
                st.write(f"- Algorithm: Linear Regression")
                st.write(f"- Forecast Horizon: {years_ahead} years")
        
    except FileNotFoundError:
        st.error(f"""
            ❌ **Data file not found!**
            
            Please ensure `emission.csv` exists in the `data/` directory.
            
            Expected path: `{data_path}`
        """)
        
        st.info("""
            **Sample Data Format:**
            
            The CSV file should contain two columns:
            - `Year`: Integer year values
            - `Emission`: CO₂ emission values in million tonnes
        """)
    
    except Exception as e:
        st.error(f"❌ **Error:** {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
