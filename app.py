import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import importlib

# Add project root to path for core module imports
sys.path.append(str(Path(__file__).parent))

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
    page_title="Carbon Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global Design Tokens
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap');

    /* BASE */
    .stApp { background-color: #FAFAF7 !important; font-family: 'Inter', sans-serif; }
    header, footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_multifactor_data(data_path):
    """Load multi-factor emission data"""
    engineer = FeatureEngineer()
    df = engineer.load_multifactor_data(data_path)
    return df, engineer


@st.cache_resource
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


@st.cache_resource
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
    
    _GRID  = 'rgba(159, 207, 184, 0.12)'
    _LINE  = 'rgba(159, 207, 184, 0.25)'
    _TICK  = '#9FCFB8'
    _AXIS  = '#DDA340'
    fig.update_layout(
        title=dict(
            text='Multi-Factor Emission Forecast with Sustainability Target',
            font=dict(family='Playfair Display, serif', size=24, color='#DDA340'),
            x=0.5, xanchor='center'
        ),
        xaxis_title='Year',
        yaxis_title='CO₂ Emissions (Million Tonnes)',
        template='plotly_dark',
        paper_bgcolor='#073831',
        plot_bgcolor='#052A25',
        font=dict(family='Space Grotesk, Inter, sans-serif', color='#9FCFB8', size=13),
        hovermode='x unified',
        height=550,
        margin=dict(l=60, r=30, t=70, b=60),
        legend=dict(
            bgcolor='rgba(5, 42, 37, 0.85)',
            bordercolor='rgba(221, 163, 64, 0.3)',
            borderwidth=1,
            font=dict(color='#E0F0E8', size=12)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=_GRID,
            linecolor=_LINE,
            tickfont=dict(color=_TICK, size=12),
            title_font=dict(color=_AXIS, size=13, family='Inter')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=_GRID,
            linecolor=_LINE,
            tickfont=dict(color=_TICK, size=12),
            title_font=dict(color=_AXIS, size=13, family='Inter')
        )
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
    
    _GRID  = 'rgba(159, 207, 184, 0.12)'
    _LINE  = 'rgba(159, 207, 184, 0.25)'
    _TICK  = '#9FCFB8'
    _AXIS  = '#DDA340'
    fig.update_layout(
        title=dict(
            text='Policy Scenario Comparison vs Sustainability Target',
            font=dict(family='Playfair Display, serif', size=24, color='#DDA340'),
            x=0.5, xanchor='center'
        ),
        xaxis_title='Year',
        yaxis_title='CO₂ Emissions (Million Tonnes)',
        template='plotly_dark',
        paper_bgcolor='#073831',
        plot_bgcolor='#052A25',
        font=dict(family='Space Grotesk, Inter, sans-serif', color='#9FCFB8', size=13),
        hovermode='x unified',
        height=550,
        margin=dict(l=60, r=30, t=70, b=60),
        legend=dict(
            bgcolor='rgba(5, 42, 37, 0.85)',
            bordercolor='rgba(221, 163, 64, 0.3)',
            borderwidth=1,
            font=dict(color='#E0F0E8', size=12)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=_GRID,
            linecolor=_LINE,
            tickfont=dict(color=_TICK, size=12),
            title_font=dict(color=_AXIS, size=13, family='Inter')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=_GRID,
            linecolor=_LINE,
            tickfont=dict(color=_TICK, size=12),
            title_font=dict(color=_AXIS, size=13, family='Inter')
        )
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
            colorscale='GnBu_r',
            showscale=False,
            line=dict(color='rgba(81, 207, 102, 0.3)', width=1)
        )
    ))
    
    _GRID  = 'rgba(159, 207, 184, 0.12)'
    _LINE  = 'rgba(159, 207, 184, 0.25)'
    _TICK  = '#9FCFB8'
    _AXIS  = '#DDA340'
    fig.update_layout(
        title=dict(
            text='🔍 Structural Factor Importance',
            font=dict(family='Playfair Display, serif', size=22, color='#DDA340'),
            x=0.5, xanchor='center'
        ),
        xaxis_title='Importance Score',
        yaxis_title='Factor',
        template='plotly_dark',
        paper_bgcolor='#073831',
        plot_bgcolor='#052A25',
        font=dict(family='Space Grotesk, Inter, sans-serif', color='#9FCFB8', size=13),
        height=450,
        margin=dict(l=160, r=30, t=70, b=60),
        xaxis=dict(
            showgrid=True,
            gridcolor=_GRID,
            linecolor=_LINE,
            tickfont=dict(color=_TICK, size=12),
            title_font=dict(color=_AXIS, size=13, family='Inter')
        ),
        yaxis=dict(
            showgrid=False,
            linecolor=_LINE,
            tickfont=dict(color=_TICK, size=12),
            title_font=dict(color=_AXIS, size=13, family='Inter')
        )
    )
    
    return fig


def render_hero():
    import base64
    from pathlib import Path

    bg_path = Path(__file__).parent / "bg.jpg"

    if bg_path.exists():
        with open(bg_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
    else:
        encoded_string = ""  # fallback

    st.session_state.encoded_hero_bg = encoded_string

    # Store encoded string in session state to use in global CSS outside this function
    st.session_state.encoded_hero_bg = encoded_string

    st.markdown(f"""
        <div id="home" class="hero-section">
            <h1 class="hero-title">CARBON FORECASTING</h1>
            <div class="hero-subtitle">Structural Modeling & Policy Simulation</div>
            <p class="hero-description">
                "Empowering global stewards with architectural-grade emission trajectories 
                and multi-factor policy stress-testing."
            </p>
            <div class="scroll-indicator">↓</div>
        </div>
    """, unsafe_allow_html=True)


def main():
    # Setup background image context
    bg_path = r"bg.jpg"
    import base64
    with open(bg_path, "rb") as image_file:
        encoded_hero = base64.b64encode(image_file.read()).decode()

    # CORE ARCHITECTURAL CSS ROOT — Light Theme
    st.markdown(f"""
        <style>
        /* ── 1. GLOBAL LAYOUT ──────────────────────────────── */
        [data-testid="stSidebar"] {{
            display: block !important;
            background-color: #052A25 !important;
        }}

        .main .block-container {{
            padding: 5rem 10% !important;
            padding-top: 0px !important;
            max-width: 100% !important;
            margin: 0 auto !important;
            background-color: #FAFAF7 !important;
            text-align: center !important;
        }}

        html, body, .stApp, .main, [data-testid="stAppViewContainer"] {{
            overflow: auto !important;
            height: auto !important;
            background-color: #FAFAF7 !important;
        }}

        /* ── 2. HERO SECTION ───────────────────────────────── */
        #home.hero-section {{
            position: relative;
            width: 120vw;
            height: 100vh;
            left: 0;
            margin-top: -15vh;
            margin-left: -15vw;
            background: url('data:image/png;base64,{encoded_hero}') repeat fixed 100% 100%;
            background-size: cover;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            z-index: 10;
        }}

        .hero-title {{
            font-family: 'Playfair Display', serif !important;
            font-size: 5.5rem !important;
            color: #DDA340 !important;
            font-weight: 900 !important;
            letter-spacing: 2px;
            text-shadow: 0 6px 32px rgba(0,0,0,0.55);
            margin: 0;
        }}

        .hero-subtitle {{
            font-family: 'Inter', sans-serif !important;
            font-size: 1.4rem !important;
            color: #FFFFFF !important;
            letter-spacing: 12px;
            margin-top: 16px !important;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .hero-description {{
            color: rgba(255,255,255,0.82) !important;
            font-size: 1.1rem !important;
            font-style: italic;
            margin-top: 30px !important;
            max-width: 680px;
            line-height: 1.7;
        }}

        @keyframes bounce {{
            0%,20%,50%,80%,100% {{ transform: translateY(0); }}
            40% {{ transform: translateY(-10px); }}
            60% {{ transform: translateY(-5px); }}
        }}
        .scroll-indicator {{
            position: absolute;
            bottom: 40px;
            font-size: 2rem;
            color: #DDA340;
            animation: bounce 2s infinite;
        }}

        /* ── 3. NAVIGATION BAR ─────────────────────────────── */
        .custom-navbar {{
            position: fixed;
            top: 0; left: 0;
            width: 100%;
            height: 70px;
            background: rgba(5, 42, 37, 0.97);
            backdrop-filter: blur(14px);
            z-index: 999;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 60px;
            border-bottom: 1px solid rgba(221, 163, 64, 0.18);
        }}
        .nav-brand {{
            font-family: 'Playfair Display', serif;
            font-size: 1.6rem;
            font-weight: 900;
            color: #DDA340 !important;
        }}
        .nav-links {{ display: flex; gap: 35px; }}
        .nav-link {{
            font-family: 'Inter', sans-serif;
            text-decoration: none !important;
            color: rgba(255,255,255,0.88) !important;
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: color 0.2s, border-bottom 0.2s;
            border-bottom: 2px solid transparent;
            padding-bottom: 4px;
            cursor: pointer;
        }}
        .nav-link:hover {{ color: #DDA340 !important; }}

        /* ── 4. DASHBOARD TYPOGRAPHY ───────────────────────── */
        .main h1, .main h2, .main h3,
        .main [data-testid="stMarkdownContainer"] h1,
        .main [data-testid="stMarkdownContainer"] h2,
        .main [data-testid="stMarkdownContainer"] h3 {{
            color: #073831 !important;
            font-family: 'Playfair Display', serif !important;
            text-align: center !important;
            margin-top: 2rem !important;
            width: 100% !important;
        }}

        /* Body text — dark amber-brown, readable on #FAFAF7 */
        .main p, .main span, .main label,
        .main [data-testid="stMarkdownContainer"] p,
        .main [data-testid="stMarkdownContainer"] span {{
            color: #7A5A1E !important;
            font-family: 'Inter', sans-serif !important;
            text-align: center !important;
            width: 100% !important;
            display: block !important;
        }}

        /* ── 5. METRIC CARDS ───────────────────────────────── */
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricLabel"] div {{
            color: #9FCFB8 !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            text-align: center !important;
            justify-content: center !important;
        }}

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] div {{
            color: #FFFFFF !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1.95rem !important;
            text-align: center !important;
            justify-content: center !important;
            letter-spacing: -0.5px !important;
        }}

        [data-testid="stMetricDelta"] {{ justify-content: center !important; }}
        [data-testid="stMetricDelta"] div {{ color: #DDA340 !important; }}

        div.stMetric {{
            background: #073831 !important;
            border: 1px solid rgba(221, 163, 64, 0.25) !important;
            border-top: 3px solid #DDA340 !important;
            border-radius: 6px !important;
            box-shadow: 0 4px 18px rgba(5, 42, 37, 0.18) !important;
            text-align: center !important;
            padding: 16px 12px !important;
        }}

        /* ── 6. FACTOR CARDS ───────────────────────────────── */
        .factor-card {{
            background: #073831 !important;
            border: 1px solid rgba(221, 163, 64, 0.2) !important;
            border-top: 3px solid #DDA340 !important;
            border-radius: 6px !important;
            box-shadow: 0 4px 16px rgba(5, 42, 37, 0.18) !important;
            text-align: center !important;
            padding: 16px 14px 14px !important;
            margin-bottom: 12px;
        }}
        .factor-card strong {{
            color: #DDA340 !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
        }}
        .factor-card > div:nth-child(2) {{
            color: #FFFFFF !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1.85em !important;
            letter-spacing: -0.5px !important;
        }}
        .factor-card > div:first-child {{
            justify-content: center !important;
            font-size: 1.05em !important;
            color: #9FCFB8 !important;
        }}

        /* ── 7. SUSTAINABILITY BANNER ──────────────────────── */
        .sustainability-banner {{
            background: #073831 !important;
            border: 1px solid rgba(221, 163, 64, 0.2) !important;
            border-left: 5px solid #DDA340 !important;
            border-radius: 6px !important;
            box-shadow: 0 4px 18px rgba(5, 42, 37, 0.18) !important;
            text-align: left !important;
            margin: 24px 0 !important;
        }}

        /* ── 8. SLIDERS ────────────────────────────────────── */
        [data-testid="stSlider"] label,
        [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {{
            color: #7A5A1E !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.93rem !important;
        }}
        [data-testid="stSlider"] [data-baseweb="slider"] [role="progressbar"] {{
            background-color: #073831 !important;
        }}
        [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {{
            background-color: rgba(7, 56, 49, 0.14) !important;
            height: 5px !important;
        }}
        [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
            background-color: #DDA340 !important;
            border-color: #073831 !important;
            box-shadow: 0 2px 8px rgba(7, 56, 49, 0.22) !important;
        }}
        [data-testid="stSlider"] [data-testid="stThumbValue"] {{
            color: #052A25 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 700 !important;
            background: #FFFFFF !important;
            border: 1px solid rgba(7,56,49,0.15) !important;
            border-radius: 4px !important;
            padding: 2px 6px !important;
        }}
        [data-testid="stSlider"] span {{
            color: rgba(7, 56, 49, 0.55) !important;
            font-family: 'Inter', sans-serif !important;
        }}

        /* ── 9. EXPANDER ───────────────────────────────────── */
        [data-testid="stExpander"] {{
            border: 1px solid rgba(7, 56, 49, 0.1) !important;
            border-radius: 6px !important;
            background: #FFFFFF !important;
        }}
        [data-testid="stExpander"] summary {{
            color: #073831 !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }}

        /* ── 10. MISC ──────────────────────────────────────── */
        hr {{ border-color: rgba(7, 56, 49, 0.1) !important; margin: 2.5rem 0 !important; }}
        .stSpinner > div > div {{ border-top-color: #DDA340 !important; }}
        header, footer {{ visibility: hidden; }}

        /* ── 11. FULL-SCREEN SECTIONS ──────────────────────── */
        /* Each .page-section fills the viewport and centers its content */
        .page-section {{
            min-height: 10vh;
            padding: 100px 0 60px 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            scroll-snap-align: start;
            scroll-margin-top: 70px;
        }}

        /* Alternating subtle section background tints */
        .page-section:nth-child(odd)  {{ background-color: #FAFAF7; }}
        .page-section:nth-child(even) {{ background-color: #F2F5F0; }}

        /* Snap-scroll on the main content wrapper */
        [data-testid="stAppViewContainer"] > section {{
            scroll-snap-type: y proximity;
        }}

        /* ── 12. SMOOTH SCROLL (global) ─────────────────────── */
        html {{ scroll-behavior: smooth; }}

        /* Active nav link highlight */
        .nav-link.active {{ color: #DDA340 !important; }}
        </style>
    """, unsafe_allow_html=True)

    # Smooth-scroll + active-link JS (injected once)
    st.markdown("""
        <script>
        (function() {{
            // Intercept all nav-link clicks and smooth-scroll to target
            document.addEventListener('click', function(e) {{
                var link = e.target.closest('a.nav-link');
                if (!link) return;
                var href = link.getAttribute('href');
                if (!href || !href.startsWith('#')) return;
                var target = document.getElementById(href.slice(1));
                if (!target) return;
                e.preventDefault();
                target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }});

            // Highlight active nav link on scroll
            var links = document.querySelectorAll('a.nav-link');
            var sections = Array.from(links)
                .map(l => document.getElementById((l.getAttribute('href') || '').slice(1)))
                .filter(Boolean);

            function onScroll() {{
                var scrollY = window.scrollY + 100;
                var current = sections[0];
                sections.forEach(function(s) {{
                    if (s.getBoundingClientRect().top + window.scrollY <= scrollY) current = s;
                }});
                links.forEach(function(l) {{
                    l.classList.toggle('active',
                        l.getAttribute('href') === '#' + (current && current.id));
                }});
            }}
            window.addEventListener('scroll', onScroll, {{ passive: true }});
            onScroll();
        }})();
        </script>
    """, unsafe_allow_html=True)

    # Render Hero Landing
    render_hero()

    # CUSTOM NAVIGATION BAR — smooth-scroll via JS
    st.markdown("""
        <div class="custom-navbar">
            <div class="nav-brand">🍃 CARBON FORECAST</div>
            <div class="nav-links">
                <a href="#home" class="nav-link">Home</a>
                <a href="#forecast" class="nav-link">Forecast</a>
                <a href="#scenarios" class="nav-link">Scenarios</a>
                <a href="#simulator" class="nav-link">Simulator</a>
                <a href="#importance" class="nav-link">Analysis</a>
            </div>
        </div>
        <script>
        function scrollTo(id) {
            var el = document.getElementById(id);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        // Active link tracking
        (function() {
            var ids = ['home','forecast','scenarios','simulator','importance'];
            var links = document.querySelectorAll('a.nav-link');
            function update() {
                var scrollY = window.scrollY + 120;
                var current = ids[0];
                ids.forEach(function(id) {
                    var s = document.getElementById(id);
                    if (s && s.getBoundingClientRect().top + window.scrollY <= scrollY) current = id;
                });
                links.forEach(function(l) {
                    var match = l.getAttribute('onclick') && l.getAttribute('onclick').includes("'" + current + "'");
                    l.style.color = match ? '#DDA340' : 'rgba(255,255,255,0.88)';
                    l.style.borderBottom = match ? '2px solid #DDA340' : '2px solid transparent';
                });
            }
            window.addEventListener('scroll', update, { passive: true });
            update();
        })();
        </script>
    """, unsafe_allow_html=True)

        
    # ── SECTION: FORECAST ──────────────────────────────────────
    st.markdown('<div id="forecast" class="page-section">', unsafe_allow_html=True)
    st.markdown("""
        <h1 style="font-family: 'Playfair Display', serif; text-align: center; color: rgba(5, 42, 37, 0.95); font-weight: 900; font-size: 3rem;">
            🌍 Carbon Emission Forecasting System
        </h1>
    """, unsafe_allow_html=True)
    st.markdown("""
        <p style="color: #7A5A1E; text-align: center; font-family: 'Inter', sans-serif; font-size: 1.1rem; line-height: 1.8; max-width: 800px; margin: 0 auto;">
            <strong>Multi-Factor ML-Driven Platform with Sustainability Targets</strong><br><br>
            This advanced system integrates <strong>policy factors</strong> and <strong>structural dynamics</strong> to forecast 
            emissions and simulate realistic decarbonization pathways toward <strong>sustainable targets</strong>.
        </p>
    """, unsafe_allow_html=True)

    
    # Data path (Correct for root)
    data_path = Path(__file__).parent / "data" / "real_emission_dataset.csv"
    
    try:
        # Load data
        with st.spinner("Loading multi-factor emission data..."):
            df, engineer = load_multifactor_data(str(data_path))
        
        # Train model
        with st.spinner("Training advanced ML models (comparing algorithms)..."):
            model, model_type, comparison, feature_names = train_enhanced_model(engineer, df)
        
        # FIXED FORECAST SETTINGS (As requested)
        years_ahead = 26 # Horizon 2050
        target_year = 2050

        # Define Reduction Target (from slider key if exists, else default 50)
        reduction_target = st.session_state.get('target_slider', 50)
        
        # Calculate sustainability target
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
            <div class="sustainability-banner" style = "padding: 20px; padding-right: -20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="text-align: left;">
                        <span style="font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; color: #9FCFB8;">Net Zero Target Analysis</span>
                        <h1 class="target-highlight" style="color: #FFFFFF; font-family: 'Playfair Display', serif">🎯 {target_info['target_emission']:.0f} MT by {target_year}</h1>
                        <div style="color: #DDA340; font-weight: 600;">{target_info['reduction_from_baseline']:.0f}% reduction from 2005 baseline</div>
                    </div>
                    <div style="text-align: right;">
                        <h3 class="residual-tag" style="color: #FFFFFF; font-family: 'Playfair Display', serif">
                            🛡️ Residual Floor: {sustainability_calc.RESIDUAL_EMISSION_FLOOR_PERCENT}%
                        </h3>
                        <div style="font-size: 0.8em; color: #9FCFB8; margin-top: 5px; max-width: 250px;">
                            Hard-to-abate sectors: Agriculture, Aviation, Heavy Industry
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # KPIs
        st.markdown("""
            <h2 style="font-family: 'Playfair Display', serif; text-align: center; color: rgba(5, 42, 37, 0.95); font-weight: 900; font-size: 2.2rem; margin-top: 2rem;">
                📊 Key Performance Indicators
            </h2>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        current_emission = df['Emission'].iloc[-1]
        current_year = int(df['Year'].max())
        
        with col1:
            st.metric(
                "Current Emission",
                f"{current_emission:.0f} MT",
                delta=f"Year: {current_year}",
                delta_color="off"
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
                delta_color="inverse"
            )
        
        with col3:
            gap = baseline_target_val - target_info['target_emission']
            st.metric(
                "Emission Gap",
                f"{gap:.0f} MT",
                delta=f"{(gap/target_info['target_emission']*100):.0f}% over target",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Forecasting Engine",
                model_type.replace('_', ' ').title(),
                delta=f"Accuracy: {model.metrics['r2_score']:.3f}",
                delta_color="normal"
            )
        
        st.divider()
        
        # Structural Baseline Factors
        st.markdown("""
            <h2 style="font-family: 'Playfair Display', serif; text-align: center; color: rgba(5, 42, 37, 0.95); font-weight: 900; font-size: 2.2rem; margin-top: 2rem;">
                🏗️ Structural Baseline Factors
            </h2>
            <p style="font-family: 'Inter', sans-serif; text-align: center; color: #7A5A1E; font-size: 1rem; margin-top: -0.5rem;">
                These slow-changing factors influence emissions alongside policies
            </p>
        """, unsafe_allow_html=True)
        
        baseline_factors = engineer.get_structural_baseline(df)
        
        # Calculate recent trends (last 10 years)
        latest_10 = df.tail(10)
        pop_growth = (latest_10['Population_Million'].pct_change().mean() * 100)
        urb_growth = (latest_10['Urbanization_Rate'].diff().mean())
        forest_growth = (latest_10['Forest_Cover_Percent'].pct_change().mean() * 100)
        energy_growth = (latest_10['Energy_Demand_Index'].pct_change().mean() * 100)
        trans_growth = (latest_10['Transport_Index'].pct_change().mean() * 100)
        ind_growth_val = latest_10['Industrial_Growth'].mean()
        renewable_trend = (latest_10['Renewable_Percent'].diff().mean())
        fossil_trend = (latest_10['Fossil_Percent'].diff().mean())
        
        col1, col2, col3, col4 = st.columns(4)
        st.markdown("""
            <br><br>
        """, unsafe_allow_html=True)
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
        
        # Forecast Chart
        st.markdown("""
            <h2 style="font-family: 'Playfair Display', serif; text-align: center; color: rgba(5, 42, 37, 0.95); font-weight: 900; font-size: 2.2rem; margin-top: 2rem;">
                📈 Multi-Factor Emission Forecast
            </h2>
        """, unsafe_allow_html=True)
        forecast_chart = create_enhanced_forecast_chart(df, baseline_df, sustainability_calc)
        st.plotly_chart(forecast_chart, use_container_width=True)
        
        # Interactive Target Control (Placed BELOW the graph as requested)
        st.markdown("<br>", unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
        with col_s2:
            st.slider(
                "🎯 Sustainability Target: Reduction Target (%)",
                min_value=30,
                max_value=80,
                value=50,
                step=5,
                key='target_slider',
                help="Adjust the desired percentage reduction from the 2005 baseline."
            )
        
        st.divider()
        st.markdown('</div>', unsafe_allow_html=True)  # close #forecast section

        # ── SECTION: SCENARIOS ─────────────────────────────────
        st.markdown('<div id="scenarios" class="page-section">', unsafe_allow_html=True)
        st.markdown("""
            <h2 style="font-family: 'Playfair Display', serif; text-align: center; color: rgba(5, 42, 37, 0.95); font-weight: 900; font-size: 2.2rem;">
                🎯 Enhanced Policy Scenario Comparison
            </h2>
        """, unsafe_allow_html=True)
        comparison_df = simulator.compare_scenarios(years_ahead)
        scenario_chart = create_scenario_comparison_chart(comparison_df, target_info['target_emission'])
        st.plotly_chart(scenario_chart, width='stretch')
        
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
                        <h3 style="color: {color}; margin-top: 0; font-size: 1.5rem;">{name}</h3>
                        <div style="font-size: 0.9rem; color: #9FCFB8; margin-bottom: 15px;">Target: {target_year} Analysis</div>
                        <p style="margin: 5px 0; color: #E0F0E8;"><strong>🌱 Renewable:</strong> {params.renewable_growth_percent:+.0f}%</p>
                        <p style="margin: 5px 0; color: #E0F0E8;"><strong>⛽ Fossil:</strong> {params.fossil_reduction_percent:+.0f}%</p>
                        <p style="margin: 5px 0; color: #E0F0E8;"><strong>🏭 Industrial:</strong> {params.industrial_growth_percent:+.0f}%</p>
                        <p style="margin: 5px 0; color: #E0F0E8;"><strong>🌳 Forest:</strong> {params.forest_protection_percent:+.0f}%</p>
                        <p style="margin: 5px 0; color: #E0F0E8;"><strong>⚡ Efficiency:</strong> {params.energy_efficiency_improvement:+.0f}%</p>
                        <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid rgba(221, 163, 64, 0.25);">
                            <div style="font-size: 1.1rem; font-weight: 600; color: #FFFFFF;">{scenario_emission:.0f} MT</div>
                            <div style="color: {'#51cf66' if meets_target else '#ff6b6b'}; font-size: 0.9rem;">
                                {'✅ Target Achieved' if meets_target else f'❌ {gap:+.0f} MT Gap'}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        st.markdown('</div>', unsafe_allow_html=True)  # close #scenarios section

        # ── SECTION: SIMULATOR ─────────────────────────────────
        st.markdown('<div id="simulator" class="page-section">', unsafe_allow_html=True)
        st.markdown("""
            <h2 style="font-family: 'Playfair Display', serif; text-align: center; color: rgba(5, 42, 37, 0.95); font-weight: 900; font-size: 2.2rem;">
                🎛️ Custom Multi-Factor Scenario Simulator
            </h2>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style=\"font-family: 'Playfair Display', serif; text-align: center; color: #A0722A; font-size: 1.3rem;\">Policy Factors (Controllable)</h3>", unsafe_allow_html=True)
            renewable_growth = st.slider("🌱 Renewable Energy Growth (%)", -20.0, 150.0, 60.0, 5.0)
            fossil_reduction = st.slider("⛽ Fossil Fuel Reduction (%)", -30.0, 80.0, 50.0, 5.0)
            industrial_growth = st.slider("🏭 Industrial Emission Growth (%)", -30.0, 50.0, 0.0, 5.0)
            forest_protection = st.slider("🌳 Forest Protection/Expansion (%)", -20.0, 30.0, 10.0, 5.0)
        
        with col2:
            st.markdown("<h3 style=\"font-family: 'Playfair Display', serif; text-align: center; color: #A0722A; font-size: 1.3rem;\">Structural Adjustments</h3>", unsafe_allow_html=True)
            energy_efficiency = st.slider("⚡ Energy Efficiency Improvement (%)", -10.0, 50.0, 25.0, 5.0)
            pop_adjustment = st.slider("👥 Population Growth Adjustment (%)", -0.5, 0.5, 0.0, 0.1)
            urban_adjustment = st.slider("🏙️ Urbanization Rate Adjustment (%)", -0.5, 0.5, 0.0, 0.1)
            policy_start = st.slider("📅 Policy Start Year", int(df['Year'].max()) + 1, int(df['Year'].max()) + 5, int(df['Year'].max()) + 1)
        
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
        
        # Custom Chart
        fig_custom = go.Figure()
        fig_custom.add_trace(go.Scatter(x=df['Year'], y=df['Emission'], mode='lines', name='Historical', line=dict(color='#4da6ff', width=2)))
        fig_custom.add_trace(go.Scatter(x=baseline_df['Year'], y=baseline_df['Current Trend Forecast'], mode='lines', name='Current Trend Forecast', line=dict(color='#ff6b6b', width=2, dash='dash')))
        fig_custom.add_trace(go.Scatter(x=custom_scenario['Year'], y=custom_scenario['Scenario_Emission'], mode='lines', name='Custom Scenario', line=dict(color='#51cf66', width=3)))
        fig_custom.add_trace(go.Scatter(x=custom_scenario['Year'], y=[target_info['target_emission']] * len(custom_scenario), mode='lines', name='Sustainability Target', line=dict(color='#51cf66', width=2, dash='dot')))
        
        fig_custom.update_layout(
            title=dict(
                text='Custom Scenario Comparison',
                font=dict(family='Playfair Display, serif', size=22, color='#DDA340'),
                x=0.5, xanchor='center'
            ),
            xaxis_title='Year',
            yaxis_title='CO₂ Emissions (MT)',
            template='plotly_dark',
            paper_bgcolor='#073831',
            plot_bgcolor='#052A25',
            font=dict(family='Space Grotesk, Inter, sans-serif', color='#9FCFB8', size=13),
            hovermode='x unified',
            height=500,
            margin=dict(l=60, r=30, t=70, b=60),
            legend=dict(
                bgcolor='rgba(5, 42, 37, 0.85)',
                bordercolor='rgba(221, 163, 64, 0.3)',
                borderwidth=1,
                font=dict(color='#E0F0E8', size=12)
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(159, 207, 184, 0.12)',
                linecolor='rgba(159, 207, 184, 0.25)',
                tickfont=dict(color='#9FCFB8', size=12),
                title_font=dict(color='#DDA340', size=13, family='Inter')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(159, 207, 184, 0.12)',
                linecolor='rgba(159, 207, 184, 0.25)',
                tickfont=dict(color='#9FCFB8', size=12),
                title_font=dict(color='#DDA340', size=13, family='Inter')
            )
        )
        st.plotly_chart(fig_custom, width='stretch')
        
        # Custom Metrics
        if target_year in custom_scenario['Year'].values:
            custom_emission = custom_scenario[custom_scenario['Year'] == target_year]['Scenario_Emission'].values[0]
        else:
            custom_emission = custom_scenario['Scenario_Emission'].iloc[-1]
            
        custom_gap = custom_emission - target_info['target_emission']
        baseline_gap = baseline_target_val - target_info['target_emission']
        improvement = baseline_gap - custom_gap
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric(f"Current Trend {target_year}", f"{baseline_target_val:.0f} MT", delta=f"+{baseline_gap:.0f} MT over target", delta_color="inverse")
        with col2: st.metric(f"Custom {target_year}", f"{custom_emission:.0f} MT", delta=f"{custom_gap:+.0f} MT vs target", delta_color="inverse")
        with col3: st.metric("Improvement", f"{improvement:.0f} MT", delta=f"{(improvement/baseline_target_val*100):.1f}% reduction", delta_color="normal")
        with col4: 
            meets = custom_gap <= 0
            st.metric("Target Status", "✅ Achieved" if meets else "❌ Not Met", delta=f"{abs(custom_gap):.0f} MT {'below' if meets else 'above'}", delta_color="normal" if meets else "inverse")
        
        st.divider()
        st.markdown('</div>', unsafe_allow_html=True)  # close #simulator section

        # ── SECTION: IMPORTANCE ────────────────────────────────
        st.markdown('<div id="importance" class="page-section">', unsafe_allow_html=True)
        st.markdown("""
            <h2 style="font-family: 'Playfair Display', serif; text-align: center; color: rgba(5, 42, 37, 0.95); font-weight: 900; font-size: 2.2rem;">
                🔍 Factor Importance Analysis
            </h2>
        """, unsafe_allow_html=True)
        importance_chart = create_feature_importance_chart(model)
        if importance_chart: st.plotly_chart(importance_chart, width='stretch')
        
        with st.expander("🤖 ML Comparison"): st.dataframe(comparison, width='stretch')
        with st.expander("📋 Forecast Data"): st.dataframe(custom_scenario, width='stretch')

        st.markdown('</div>', unsafe_allow_html=True)  # close #importance section
        
    except FileNotFoundError:
        st.error(f"❌ Data file not found at {data_path}!")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
