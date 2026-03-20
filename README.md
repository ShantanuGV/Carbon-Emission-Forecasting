# 🌍 Research-Grade Carbon Emission Forecasting System
### Multi-Factor ML-Driven Policy Scenario Simulator

An advanced carbon emission forecasting platform that goes beyond simple trends to model **structural socio-economic dynamics**, **environmental factors**, and **industrial lock-ins** using multi-variable machine learning.

## 📋 Project Overview (v2.0 Upgrade)

The system has been upgraded to a research-grade tool that simulates realistic decarbonization pathways:
- **Core Objective**: Convergence toward a "Sustainable/Healthy Emission Level" rather than an unrealistic zero-carbon target.
- **Multi-Factor Modeling**: Integrates policy factors (controllable) with structural factors (slow-changing).
- **Advanced ML**: Utilizes Multiple Linear Regression and Random Forest Regressors trained on multi-feature datasets.
- **Structural Integrity**: Accounts for population, urbanization, energy demand, and forest cover dynamics.

## 🏗️ Architecture

```
carbon_project/
│
├── core/                          # Modular ML & Data Science Engine
│   ├── __init__.py               # Exports public APIs
│   ├── feature_engineering.py    # Multi-factor data prep & interactions
│   ├── model_training.py         # Advanced ML (Linear, RF, GBM)
│   ├── scenario_engine.py        # Enhanced policy + structural simulator
│   └── sustainability_target.py  # Healthy threshold computation
│
├── ui_streamlit/                 # Frontend Layers
│   ├── app.py                    # Legacy basic dashboard
│   └── app_enhanced.py           # NEW: Research-grade dashboard
│
├── data/                         # Intelligent Datasets
│   ├── emission.csv              # Basic historical data
│   └── emission_multifactor.csv  # Enhanced multi-source dataset
│
├── requirements.txt              # Project dependencies
└── README.md                     # Enhanced documentation
```

## ✨ Enhanced Features

### 1. **Multi-Factor Forecasting**
- **Socio-Economic**: Population growth, urbanization rates, energy demand.
- **Environmental**: Forest cover change, carbon sink capacity.
- **Industrial**: Infrastructure lock-in, production indices, energy intensity.

### 2. **Sustainability Target Logic**
- Implements "Healthy Emission Thresholds" based on historical safe baselines or % reduction targets.
- Visualizes convergence pathways with a residual emission floor (non-zero).

### 3. **Sophisticated Scenario Engine**
- **Best Case**: Aggressive renewables + fossil reduction + forest expansion + population stabilization.
- **Average Case**: Current trend continuation + moderate policy adoption.
- **Worst Case**: Fossil surge + industrial growth + deforestation.

### 4. **ML Model Upgrade**
- Automatic comparison and selection of best models (Multiple Linear Regression, Random Forest).
- Feature importance analysis to identify key drivers of emissions.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Navigate to the project directory:
```bash
cd "C:\Users\hp\Desktop\Codes\Python\Carbon Emission"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Enhanced Dashboard
```bash
python -m streamlit run ui_streamlit/app_enhanced.py
```

## 📊 Core Modules

### feature_engineering.py
- Handles multi-source data loading and missing value imputation.
- Creates complex interaction features (e.g., Energy Intensity, Fossil Dependency).
- Projects structural factors into the future based on historical growth rates.

### model_training.py
- Trains multiple ML models (Linear, RF, GBM) on multi-feature inputs.
- Provides model persistence and performance metrics (R², MAE, RMSE).
- Auto-selects the best-performing algorithm for your dataset.

### scenario_engine.py
- Simulates scenarios by combining controllable policy sliders with "silently" projected structural factors.
- Calculates emission trajectories that respect realistic industrial floors.

### sustainability_target.py
- Computes sustainable targets (e.g., 50% below 2005 levels).
- Generates exponential decay pathways toward those targets.

## 🧠 Future Compatibility
The architecture is fully modular. The `core/` modules are frontend-agnostic and ready to be wrapped in a Flask/FastAPI backend for React-based implementations.

## 🎯 Scenario Simulation Logic

The system calculates emission adjustments using weighted impact factors:

```python
Impact = (Renewable × 0.4) + (Fossil × 0.35) + (Industrial × 0.25)
```

**Impact is applied progressively:**
- Compounds over time after policy start year
- Ensures emissions never go negative
- Reflects realistic policy implementation curves

## 🎨 Dashboard Features

### Main Sections

1. **KPI Dashboard**
   - Current emission levels
   - Target year forecasts
   - Best-case reduction potential
   - Historical data coverage

2. **Baseline Forecast**
   - Historical data visualization
   - Future emission predictions
   - Interactive hover details

3. **Scenario Comparison**
   - Side-by-side preset scenario analysis
   - Detailed parameter cards
   - Comparative emission trajectories

4. **Custom Simulator**
   - Real-time slider controls
   - Live chart updates
   - Impact metrics calculation

5. **Data Explorer**
   - Detailed forecast tables
   - Model performance metrics
   - Training information

## 🔮 Future Enhancements

### API Integration (Planned)
The modular architecture supports future API development:

```python
# Example Flask/FastAPI wrapper
from core import load_emission_data, train_emission_model, simulate_custom_scenario

@app.post("/api/forecast")
def forecast_emissions(params: ForecastRequest):
    df = load_emission_data("data/emission.csv")
    model = train_emission_model(df)
    # ... return predictions
```

### Potential Features
- [ ] React frontend integration
- [ ] REST API backend
- [ ] Multiple country support
- [ ] Advanced ML models (LSTM, Prophet)
- [ ] Real-time data integration
- [ ] Export reports (PDF, Excel)
- [ ] User authentication
- [ ] Scenario saving/loading

## 📈 Model Performance

The Linear Regression model provides:
- **R² Score**: Measures prediction accuracy
- **MAE**: Mean Absolute Error in MT
- **RMSE**: Root Mean Squared Error in MT

View detailed metrics in the dashboard's "Model Information" section.

## 🛠️ Customization

### Adding New Scenarios

Edit `core/scenario.py`:

```python
def _define_preset_scenarios(self):
    return {
        'custom_scenario': PolicyParameters(
            renewable_growth_percent=40.0,
            fossil_reduction_percent=50.0,
            industrial_growth_percent=0.0
        )
    }
```

### Adjusting Impact Weights

Modify in `core/scenario.py`:

```python
class ScenarioSimulator:
    RENEWABLE_IMPACT = 0.4   # Adjust these weights
    FOSSIL_IMPACT = 0.35
    INDUSTRIAL_IMPACT = 0.25
```

### Using Different ML Models

Replace in `core/model.py`:

```python
from sklearn.ensemble import RandomForestRegressor

class EmissionModel:
    def __init__(self):
        self.model = RandomForestRegressor()  # Instead of LinearRegression
```

## 📝 Usage Examples

### Programmatic Usage

```python
from core import (
    load_emission_data,
    train_emission_model,
    create_predictor,
    simulate_custom_scenario
)

# Load data
df = load_emission_data("data/emission.csv")

# Train model
model = train_emission_model(df)

# Generate forecast
predictor = create_predictor(model)
baseline = predictor.get_baseline_forecast(df, years_ahead=10)

# Simulate custom scenario
scenario = simulate_custom_scenario(
    baseline,
    renewable_growth=40.0,
    fossil_reduction=50.0,
    industrial_growth=10.0,
    start_year=2025
)

print(scenario)
```

## 🐛 Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# Ensure you're in the project root
cd "C:\Users\hp\Desktop\Codes\Python\Carbon Emission"
streamlit run ui_streamlit/app.py
```

**2. Data File Not Found**
- Verify `data/emission.csv` exists
- Check file path in error message
- Ensure CSV has correct column names

**3. Dependency Issues**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📄 License

This project is open-source and available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional ML models
- Enhanced visualizations
- API development
- Documentation
- Test coverage

## 📧 Contact

For questions or suggestions, please open an issue in the project repository.

---

**Built with ❤️ for climate research and policy analysis**
#   C a r b o n - E m i s s i o n - F o r e c a s t i n g  
 #   C a r b o n - E m i s s i o n - F o r e c a s t i n g  
 #   C a r b o n - E m i s s i o n - F o r e c a s t i n g  
 