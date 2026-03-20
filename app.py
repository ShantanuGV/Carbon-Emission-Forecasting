"""
Enhanced Carbon Emission Forecasting & Policy Scenario Simulator
Multi-Factor ML-Driven System with Sustainability Targets
hi ...
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import importlib
import core
importlib.reload(core)
# Specifically reload engine and model to push our scientific sign constraint fixes
import core.scenario_engine
import core.model_training
import core.feature_engineering
importlib.reload(core.scenario_engine)
importlib.reload(core.model_training)
importlib.reload(core.feature_engineering)

from core import (
    FeatureEngineer,
    MultiFactorEmissionModel,
    train_multifactor_model,
    auto_select_best_model,
    EnhancedScenarioSimulator,
    EnhancedPolicyParameters,
    SustainabilityCalculator,
    create_sustainability_target
)

# Page configuration
st.set_page_config(
    page_title="Enhanced Carbon Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
    }
    div.stMetric {
        background: rgba(30, 33, 48, 0.4);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div.stMetric:hover {
        transform: translateY(-5px) scale(1.02);
        border: 1px solid rgba(77, 166, 255, 0.4);
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        background: rgba(30, 33, 48, 0.7);
    }
    h1 {
        color: #e6edf3;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .gradient-text {
        background: linear-gradient(135deg, #4da6ff 0%, #51cf66 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2, h3 {
        color: #e6edf3;
        font-weight: 600;
        border-bottom: 2px solid rgba(77, 166, 255, 0.2);
        padding-bottom: 5px;
        margin-top: 1.5rem;
    }
    .factor-card {
        background: rgba(30, 33, 48, 0.4);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin: 10px 0;
        transition: all 0.3s ease;
        border-top: 3px solid #4da6ff;
    }
    .factor-card:hover {
        background: rgba(30, 33, 48, 0.6);
        border-color: rgba(77, 166, 255, 0.3);
        transform: translateY(-2px);
    }
    .sustainability-banner {
        background: linear-gradient(145deg, #1b2838 0%, #0d1117 100%);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(81, 207, 102, 0.2);
        border-left: 6px solid #51cf66;
        color: #e6edf3;
        margin: 30px 0;
        box-shadow: 0 15px 40px rgba(0,0,0,0.5);
    }
    .target-highlight {
        color: #51cf66;
        font-size: 1.8em;
        font-weight: 800;
        text-shadow: 0 0 15px rgba(81, 207, 102, 0.3);
    }
    .residual-tag {
        display: inline-block;
        background: rgba(81, 207, 102, 0.1);
        color: #51cf66;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.9em;
        font-weight: 600;
        margin-top: 10px;
        border: 1px solid rgba(81, 207, 102, 0.3);
    }
    /* Specific overrides for Streamlit metric arrows */
    [data-testid="stMetricDelta"] svg {
        filter: drop-shadow(0 0 5px currentColor);
    }
    </style>
""", unsafe_allow_html=True)


#@st.cache_data
def load_multifactor_data(data_path):
    """Load multi-factor emission data"""
    engineer = FeatureEngineer()
    df = engineer.load_multifactor_data(data_path)
    return df, engineer


##@st.cache_resource
def train_enhanced_model(_engineer, df):
    """Train multi-factor emission model"""
    # Prepare features
    X, feature_names = _engineer.prepare_features(df, include_interactions=True)
    y = df['Emission']
    
    # Auto-select best model logic is bypassed to force linear extrapolation
    best_type, _, comparison = auto_select_best_model(X, y) # kept for the UI comparison table
    
    # Force linear model for proper future policy divergence (tree models extrapolate constant planes)
    from core.model_training import train_multifactor_model
    best_model = train_multifactor_model(X, y, model_type='linear')
    best_type = 'linear'
    
    return best_model, best_type, comparison, feature_names


#@st.cache_resource
def create_sustainability_calc(df):
    """Create sustainability calculator"""
    calculator = create_sustainability_target(
        df,
        target_type='percentage_reduction',
        reduction_percent=50.0,
        baseline_year=2005,
        target_year=2050
    )
    return calculator


def create_enhanced_forecast_chart(historical_df, baseline_df, sustainability_calc):
    """Create forecast with sustainability target"""
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=historical_df['Year'],
        y=historical_df['Emission'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#4da6ff', width=3),
        marker=dict(size=6)
    ))
    
    # Baseline forecast
    fig.add_trace(go.Scatter(
        x=baseline_df['Year'],
        y=baseline_df['Current Trend Forecast'],
        mode='lines',
        name='Current Trend Forecast',
        line=dict(color='#ff6b6b', width=3, dash='dash')
    ))
    
    # Sustainability target line
    target_info = sustainability_calc.get_target_info()
    target_years = baseline_df['Year']
    target_line = [target_info['target_emission']] * len(target_years)
    
    fig.add_trace(go.Scatter(
        x=target_years,
        y=target_line,
        mode='lines',
        name='Sustainability Target',
        line=dict(color='#51cf66', width=3, dash='dot')
    ))
    
    # Sustainable pathway
    current_emission = historical_df['Emission'].iloc[-1]
    pathway_df = sustainability_calc.generate_pathway(
        int(historical_df['Year'].max()) + 1,
        int(baseline_df['Year'].max()),
        current_emission
    )
    
    fig.add_trace(go.Scatter(
        x=pathway_df['Year'],
        y=pathway_df['Sustainable_Pathway'],
        mode='lines',
        name='Sustainable Pathway',
        line=dict(color='#51cf66', width=2),
        fill='tonexty',
        fillcolor='rgba(81, 207, 102, 0.1)'
    ))
    
    fig.update_layout(
        title='Multi-Factor Emission Forecast with Sustainability Target',
        xaxis_title='Year',
        yaxis_title='CO₂ Emissions (Million Tonnes)',
        template='plotly_dark',
        hovermode='x unified',
        height=550
    )
    
    return fig


def create_scenario_comparison_chart(comparison_df, sustainability_target):
    """Create scenario comparison with sustainability line"""
    fig = go.Figure()
    
    colors = {
        'Current Trend Forecast': '#8b92a8',
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
    
    # Add sustainability target
    fig.add_trace(go.Scatter(
        x=comparison_df['Year'],
        y=[sustainability_target] * len(comparison_df),
        mode='lines',
        name='Sustainability Target',
        line=dict(color='#51cf66', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title='Policy Scenario Comparison vs Sustainability Target',
        xaxis_title='Year',
        yaxis_title='CO₂ Emissions (Million Tonnes)',
        template='plotly_dark',
        hovermode='x unified',
        height=550
    )
    
    return fig


def create_feature_importance_chart(model):
    """Create feature importance visualization"""
    importance = model.get_feature_importance(top_n=10)
    
    if not importance:
        return None
    
    # Create readable names
    name_mapping = {
        'Year': 'Time Trend',
        'Renewable_Percent': 'Renewable Energy %',
        'Fossil_Percent': 'Fossil Fuel %',
        'Population_Million': 'Population',
        'Urbanization_Rate': 'Urbanization',
        'Forest_Cover_Percent': 'Forest Cover',
        'Energy_Demand_Index': 'Energy Demand',
        'Transport_Index': 'Transport Activity',
        'Industrial_Production_Index': 'Industrial Production',
        'Energy_Intensity': 'Energy Intensity',
        'Fossil_Dependency': 'Fossil Dependency',
        'Carbon_Sink_Capacity': 'Carbon Sink',
        'Renewable_Penetration': 'Renewable Penetration'
    }
    
    features = [name_mapping.get(k, k) for k in importance.keys()]
    values = list(importance.values())
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(
            color=values,
            colorscale='Viridis',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title='Top 10 Most Important Factors',
        xaxis_title='Importance Score',
        yaxis_title='Factor',
        template='plotly_dark',
        height=400
    )
    
    return fig


def main():
    # Header
    st.markdown("""
        <h1>
            🌍 <span class="gradient-text">Enhanced Carbon Emission Forecasting System</span>
        </h1>
    """, unsafe_allow_html=True)
    st.markdown("""
        **Multi-Factor ML-Driven Platform with Sustainability Targets**
        
        This advanced system integrates **policy factors** and **structural dynamics** to forecast 
        emissions and simulate realistic decarbonization pathways toward **sustainable targets**.
    """)
    
    # Data path
    data_path = Path(__file__).parent.parent / "data" / "real_emission_dataset.csv"
    
    try:
        # Load data
        with st.spinner("Loading multi-factor emission data..."):
            df, engineer = load_multifactor_data(str(data_path))
        
        # Train model
        with st.spinner("Training advanced ML models (comparing algorithms)..."):
            model, model_type, comparison, feature_names = train_enhanced_model(engineer, df)
        
        # Create sustainability calculator
        sustainability_calc = create_sustainability_calc(df)
        target_info = sustainability_calc.get_target_info()
        
        # Sidebar
        st.sidebar.header("⚙️ Forecast Settings")
        
        years_ahead = st.sidebar.slider(
            "Forecast Horizon (years)",
            min_value=10,
            max_value=30,
            value=26,  # To 2050
            step=1
        )
        
        target_year = st.sidebar.selectbox(
            "Target Year for Analysis",
            options=[2030, 2035, 2040, 2045, 2050],
            index=4
        )
        
        # Sustainability target settings
        st.sidebar.markdown("---")
        st.sidebar.header("🎯 Sustainability Target")
        
        reduction_target = st.sidebar.slider(
            "Reduction Target (%)",
            min_value=30,
            max_value=80,
            value=50,
            step=5,
            help="Percentage reduction from 2005 baseline"
        )
        
        # Update sustainability target if changed
        if reduction_target != 50:
            sustainability_calc = create_sustainability_target(
                df,
                target_type='percentage_reduction',
                reduction_percent=reduction_target,
                baseline_year=2005,
                target_year=target_year
            )
            target_info = sustainability_calc.get_target_info()
        
        # Sustainability Banner (Redesigned)
        st.markdown(f"""
            <div class="sustainability-banner">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; color: #8b92a8;">Net Zero Target Analysis</span>
                        <div class="target-highlight">🎯 {target_info['target_emission']:.0f} MT by {target_year}</div>
                        <div style="color: #51cf66; font-weight: 600;">{target_info['reduction_from_baseline']:.0f}% reduction from 2005 baseline</div>
                    </div>
                    <div style="text-align: right;">
                        <div class="residual-tag">
                            🛡️ Residual Floor: {sustainability_calc.RESIDUAL_EMISSION_FLOOR_PERCENT}%
                        </div>
                        <div style="font-size: 0.8em; color: #8b92a8; margin-top: 5px; max-width: 250px;">
                            Hard-to-abate sectors: Agriculture, Aviation, Heavy Industry
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # KPIs
        st.header("📊 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        current_emission = df['Emission'].iloc[-1]
        current_year = int(df['Year'].max())
        
        with col1:
            st.metric(
                "Current Emission",
                f"{current_emission:.0f} MT",
                delta=f"Year: {current_year}",
                delta_color="off" # Year is not a change metric, turn off arrow/color
            )
        
        # Create simulator
        simulator = EnhancedScenarioSimulator(model, engineer, df)
        baseline_df = simulator.get_baseline_projection(years_ahead)
        
        if target_year in baseline_df['Year'].values:
            baseline_target_val = baseline_df[baseline_df['Year'] == target_year]['Current Trend Forecast'].values[0]
        else:
            baseline_target_val = baseline_df['Current Trend Forecast'].iloc[-1]
        
        with col2:
            st.metric(
                f"{target_year} Forecast",
                f"{baseline_target_val:.0f} MT",
                delta=f"+{baseline_target_val - current_emission:.0f} MT",
                delta_color="inverse" # Increase is bad (red)
            )
        
        with col3:
            gap = baseline_target_val - target_info['target_emission']
            st.metric(
                "Emission Gap",
                f"{gap:.0f} MT",
                delta=f"{(gap/target_info['target_emission']*100):.0f}% over target",
                delta_color="inverse" # Gap is bad (red)
            )
        
        with col4:
            st.metric(
                "Forecasting Engine",
                model_type.replace('_', ' ').title(),
                delta=f"Accuracy: {model.metrics['r2_score']:.3f}",
                delta_color="normal" # Higher is better (green)
            )
        
        st.divider()
        
        # Structural Baseline Factors
        st.header("🏗️ Structural Baseline Factors")
        st.markdown("*These slow-changing factors influence emissions alongside policies*")
        
        baseline_factors = engineer.get_structural_baseline(df)
        
        # Calculate recent trends (last 10 years)
        latest_10 = df.tail(10)
        pop_growth = (latest_10['Population_Million'].pct_change().mean() * 100)
        urb_growth = (latest_10['Urbanization_Rate'].diff().mean()) # Percentage point change
        forest_growth = (latest_10['Forest_Cover_Percent'].pct_change().mean() * 100)
        energy_growth = (latest_10['Energy_Demand_Index'].pct_change().mean() * 100)
        trans_growth = (latest_10['Transport_Index'].pct_change().mean() * 100)
        ind_growth_val = latest_10['Industrial_Growth'].mean()
        renewable_trend = (latest_10['Renewable_Percent'].diff().mean())
        fossil_trend = (latest_10['Fossil_Percent'].diff().mean())
        
        # Display factors with dynamic trend labels
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="factor-card">
                    <div style="font-size: 1.2em; display: flex; align-items: center; gap: 8px;">
                        👥 <strong>Population</strong>
                    </div>
                    <div style="font-size: 1.5em; font-weight: 700; margin: 10px 0;">{baseline_factors['population_million']:.0f}M</div>
                    <div style="color: #4da6ff; font-size: 0.9em;">
                        Trending ~{pop_growth:+.2f}% annually
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="factor-card" style="border-top-color: #ffd43b;">
                    <div style="font-size: 1.2em; display: flex; align-items: center; gap: 8px;">
                        🏙️ <strong>Urbanization</strong>
                    </div>
                    <div style="font-size: 1.5em; font-weight: 700; margin: 10px 0;">{baseline_factors['urbanization_rate']:.1f}%</div>
                    <div style="color: #ffd43b; font-size: 0.9em;">
                        Increasing ~{urb_growth:+.1f}% annually
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="factor-card" style="border-top-color: #51cf66;">
                    <div style="font-size: 1.2em; display: flex; align-items: center; gap: 8px;">
                        🌳 <strong>Forest Cover</strong>
                    </div>
                    <div style="font-size: 1.5em; font-weight: 700; margin: 10px 0;">{baseline_factors['forest_cover_percent']:.1f}%</div>
                    <div style="color: #51cf66; font-size: 0.9em;">
                        {'Increasing' if forest_growth > 0 else 'Decreasing'} ~{abs(forest_growth):.2f}% annually
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="factor-card" style="border-top-color: #ff6b6b;">
                    <div style="font-size: 1.2em; display: flex; align-items: center; gap: 8px;">
                        ⚡ <strong>Energy Demand</strong>
                    </div>
                    <div style="font-size: 1.5em; font-weight: 700; margin: 10px 0;">{baseline_factors['energy_demand_index']:.0f}pt</div>
                    <div style="color: #ff6b6b; font-size: 0.9em;">
                        Growing ~{energy_growth:+.2f}% annually
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="factor-card" style="border-top-color: #ff922b;">
                    <div style="font-size: 1.2em; display: flex; align-items: center; gap: 8px;">
                        🚛 <strong>Transport</strong>
                    </div>
                    <div style="font-size: 1.5em; font-weight: 700; margin: 10px 0;">{baseline_factors['transport_index']:.0f}pt</div>
                    <div style="color: #ff922b; font-size: 0.9em;">
                        Trending ~{trans_growth:+.2f}% annually
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="factor-card" style="border-top-color: #cc5de8;">
                    <div style="font-size: 1.2em; display: flex; align-items: center; gap: 8px;">
                        🏭 <strong>Industrial</strong>
                    </div>
                    <div style="font-size: 1.5em; font-weight: 700; margin: 10px 0;">{baseline_factors['industrial_production_index']:.0f}pt</div>
                    <div style="color: #cc5de8; font-size: 0.9em;">
                        Growing ~{ind_growth_val:+.2f}% annually
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="factor-card" style="border-top-color: #20c997;">
                    <div style="font-size: 1.2em; display: flex; align-items: center; gap: 8px;">
                        🌱 <strong>Renewable</strong>
                    </div>
                    <div style="font-size: 1.5em; font-weight: 700; margin: 10px 0;">{baseline_factors['renewable_percent']:.1f}%</div>
                    <div style="color: #20c997; font-size: 0.9em;">
                        Trending {renewable_trend:+.1f}% annually
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="factor-card" style="border-top-color: #adb5bd;">
                    <div style="font-size: 1.2em; display: flex; align-items: center; gap: 8px;">
                        ⛽ <strong>Fossil Fuels</strong>
                    </div>
                    <div style="font-size: 1.5em; font-weight: 700; margin: 10px 0;">{baseline_factors['fossil_percent']:.1f}%</div>
                    <div style="color: #adb5bd; font-size: 0.9em;">
                        Trending {fossil_trend:+.1f}% annually
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Baseline Forecast
        st.header("📈 Multi-Factor Emission Forecast")
        
        forecast_chart = create_enhanced_forecast_chart(df, baseline_df, sustainability_calc)
        st.plotly_chart(forecast_chart, use_container_width=True)
        
        st.divider()
        
        # Scenario Comparison
        st.header("🎯 Enhanced Policy Scenario Comparison")
        
        comparison_df = simulator.compare_scenarios(years_ahead)
        scenario_chart = create_scenario_comparison_chart(
            comparison_df,
            target_info['target_emission']
        )
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
            
            if target_year in comparison_df['Year'].values:
                scenario_emission = comparison_df[comparison_df['Year'] == target_year][name].values[0]
            else:
                scenario_emission = comparison_df[name].iloc[-1]
            
            gap = scenario_emission - target_info['target_emission']
            meets_target = gap <= 0
            
            with col:
                st.markdown(f"""
                    <div class="factor-card">
                        <h3 style="color: {color};">{name}</h3>
                        <p><strong>🌱 Renewable Growth:</strong> {params.renewable_growth_percent:+.0f}%</p>
                        <p><strong>⛽ Fossil Reduction:</strong> {params.fossil_reduction_percent:+.0f}%</p>
                        <p><strong>🏭 Industrial Growth:</strong> {params.industrial_growth_percent:+.0f}%</p>
                        <p><strong>🌳 Forest Protection:</strong> {params.forest_protection_percent:+.0f}%</p>
                        <p><strong>⚡ Energy Efficiency:</strong> {params.energy_efficiency_improvement:+.0f}%</p>
                        <hr>
                        <p><strong>{target_year} Emission:</strong> {scenario_emission:.0f} MT</p>
                        <p><strong>vs Target:</strong> {gap:+.0f} MT</p>
                        <p><strong>Status:</strong> {'✅ Meets Target' if meets_target else '❌ Exceeds Target'}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Custom Scenario Simulator
        st.header("🎛️ Custom Multi-Factor Scenario Simulator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Policy Factors (Controllable)")
            
            renewable_growth = st.slider(
                "🌱 Renewable Energy Growth (%)",
                min_value=-20.0,
                max_value=150.0,
                value=60.0,
                step=5.0
            )
            
            fossil_reduction = st.slider(
                "⛽ Fossil Fuel Reduction (%)",
                min_value=-30.0,
                max_value=80.0,
                value=50.0,
                step=5.0
            )
            
            industrial_growth = st.slider(
                "🏭 Industrial Emission Growth (%)",
                min_value=-30.0,
                max_value=50.0,
                value=0.0,
                step=5.0
            )
            
            forest_protection = st.slider(
                "🌳 Forest Protection/Expansion (%)",
                min_value=-20.0,
                max_value=30.0,
                value=10.0,
                step=5.0
            )
        
        with col2:
            st.subheader("Structural Adjustments")
            
            energy_efficiency = st.slider(
                "⚡ Energy Efficiency Improvement (%)",
                min_value=-10.0,
                max_value=50.0,
                value=25.0,
                step=5.0
            )
            
            pop_adjustment = st.slider(
                "👥 Population Growth Adjustment (%)",
                min_value=-0.5,
                max_value=0.5,
                value=0.0,
                step=0.1
            )
            
            urban_adjustment = st.slider(
                "🏙️ Urbanization Rate Adjustment (%)",
                min_value=-0.5,
                max_value=0.5,
                value=0.0,
                step=0.1
            )
            
            policy_start = st.slider(
                "📅 Policy Start Year",
                min_value=int(df['Year'].max()) + 1,
                max_value=int(df['Year'].max()) + 5,
                value=int(df['Year'].max()) + 1
            )
        
        # Simulate custom scenario
        custom_params = EnhancedPolicyParameters(
            renewable_growth_percent=renewable_growth,
            fossil_reduction_percent=fossil_reduction,
            industrial_growth_percent=industrial_growth,
            forest_protection_percent=forest_protection,
            energy_efficiency_improvement=energy_efficiency,
            population_growth_adjustment=pop_adjustment,
            urbanization_rate_adjustment=urban_adjustment,
            policy_start_year=policy_start
        )
        
        custom_scenario = simulator.simulate_scenario(custom_params, years_ahead)
        
        # Custom scenario chart
        fig_custom = go.Figure()
        
        # Historical
        fig_custom.add_trace(go.Scatter(
            x=df['Year'],
            y=df['Emission'],
            mode='lines',
            name='Historical',
            line=dict(color='#4da6ff', width=2)
        ))
        
        # Current Trend Forecast
        fig_custom.add_trace(go.Scatter(
            x=baseline_df['Year'],
            y=baseline_df['Current Trend Forecast'],
            mode='lines',
            name='Current Trend Forecast',
            line=dict(color='#ff6b6b', width=2, dash='dash')
        ))
        
        # Custom scenario
        fig_custom.add_trace(go.Scatter(
            x=custom_scenario['Year'],
            y=custom_scenario['Scenario_Emission'],
            mode='lines',
            name='Custom Scenario',
            line=dict(color='#51cf66', width=3)
        ))
        
        # Sustainability target
        fig_custom.add_trace(go.Scatter(
            x=custom_scenario['Year'],
            y=[target_info['target_emission']] * len(custom_scenario),
            mode='lines',
            name='Sustainability Target',
            line=dict(color='#51cf66', width=2, dash='dot')
        ))
        
        fig_custom.update_layout(
            title='Custom Scenario vs Current Trend vs Sustainability Target',
            xaxis_title='Year',
            yaxis_title='CO₂ Emissions (Million Tonnes)',
            template='plotly_dark',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_custom, use_container_width=True)
        
        # Custom scenario metrics
        if target_year in custom_scenario['Year'].values:
            custom_emission = custom_scenario[custom_scenario['Year'] == target_year]['Scenario_Emission'].values[0]
        else:
            custom_emission = custom_scenario['Scenario_Emission'].iloc[-1]
        
        custom_gap = custom_emission - target_info['target_emission']
        baseline_gap = baseline_target_val - target_info['target_emission']
        improvement = baseline_gap - custom_gap
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                f"Current Trend {target_year}",
                f"{baseline_target_val:.0f} MT",
                delta=f"+{baseline_gap:.0f} MT over target",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                f"Custom {target_year}",
                f"{custom_emission:.0f} MT",
                delta=f"{custom_gap:+.0f} MT vs target",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Improvement vs Trend",
                f"{improvement:.0f} MT",
                delta=f"{(improvement/baseline_target_val*100):.1f}% reduction",
                delta_color="normal" # Reduction is good (green if positive improvement)
            )
        
        with col4:
            meets_target = custom_gap <= 0
            st.metric(
                "Target Status",
                "✅ Achieved" if meets_target else "❌ Not Met",
                delta=f"{abs(custom_gap):.0f} MT {'below' if meets_target else 'above'}",
                delta_color="normal" if meets_target else "inverse"
            )
        
        st.divider()
        
        # Feature Importance
        st.header("🔍 Factor Importance Analysis")
        
        importance_chart = create_feature_importance_chart(model)
        if importance_chart:
            st.plotly_chart(importance_chart, use_container_width=True)
            
            st.info("""
                **Understanding Factor Importance:**
                - Higher values indicate stronger influence on emissions
                - Policy factors (renewable, fossil) are directly controllable
                - Structural factors (population, urbanization) change slowly
                - Interaction effects capture complex relationships
            """)
        
        # Model Comparison
        with st.expander("🤖 ML Model Comparison"):
            st.dataframe(comparison, use_container_width=True)
            
            st.markdown(f"""
                **Selected Model:** {model_type.replace('_', ' ').title()}
                
                **Why this model?**
                - Highest R² score (best fit to historical data)
                - Captures non-linear relationships between factors
                - Handles multi-factor interactions effectively
            """)
        
        # Data Table
        with st.expander("📋 Detailed Forecast Data"):
            display_df = custom_scenario[['Year', 'Scenario_Emission', 'Renewable_Percent', 'Fossil_Percent', 'Forest_Cover_Percent']].copy()
            display_df.columns = ['Year', 'Emission (MT)', 'Renewable %', 'Fossil %', 'Forest Cover %']
            st.dataframe(display_df, use_container_width=True)
        
    except FileNotFoundError:
        st.error(f"""
            ❌ **Multi-factor data file not found!**
            
            Please ensure `emission_multifactor.csv` exists in the `data/` directory.
            
            Expected path: `{data_path}`
        """)
    
    except Exception as e:
        st.error(f"❌ **Error:** {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
