# 🚀 Quick Start Guide (v2.0)
## Enhanced Carbon Forecasting System

---

## ⚡ INSTANT START

### The enhanced app is READY! 🎉
**Launch it with:**
```bash
python -m streamlit run ui_streamlit/app_enhanced.py
```
**Access in browser:** http://localhost:8501

---

## 🎮 DASHBOARD CONTROLS

### 1️⃣ Forecast & Sustainability Settings (Sidebar)
- **Forecast Horizon**: Predict up to 30 years ahead.
- **Reduction Target**: Set a % reduction goal from the 2005 baseline (e.g., 50%).
- **Target Year**: Define the year to achieve the sustainable level (e.g., 2050).

### 2️⃣ Custom Policy Sliders (Controllable)
- **🌱 Renewable Energy Growth**: Policy-driven adoption rate.
- **⛽ Fossil Fuel Reduction**: Targeted transition away from coal/gas.
- **🏭 Industrial Emission Growth**: Controls new industrial expansion.
- **🌳 Forest Protection**: Expansion of natural carbon sinks.

### 3️⃣ Structural Adjustments (Non-Controllable)
*Adjust how the simulation handles socio-economic trends:*
- **⚡ Energy Efficiency**: Improvement in efficiency per capita.
- **👥 Population Adjustment**: Tweak baseline demographic growth.
- **🏙️ Urbanization Adjustment**: Tweak baseline urban shift rates.

---

## 📊 READING THE NEW CHARTS

### 📈 Multi-Factor Forecast
- **Blue Line**: Historical emissions.
- **Red Dashed**: Baseline trend if no new policies are implemented.
- **Green Dotted**: The **Sustainability Target** threshold.
- **Green Area**: The **Sustainable Pathway** to reach the target.

### 🎯 Scenario Comparison
- Compare **Best**, **Average**, and **Worst** case trajectories against the **Sustainability Target**.
- See real-time status: ✅ **Meets Target** or ❌ **Exceeds Target**.

### 🔍 Factor Importance
- Identify which factors (e.g., Fossil Fuel %, Population) have the biggest mathematical impact on your forecast results.

---

## 🏗️ STRUCTURAL BASELINE FACTORS
The system now "silently" considers the following slow-changing factors:
- **Population Growth** (~0.8% annually)
- **Urbanization Rate** (~0.5% annually)
- **Forest Cover Change** (~0.2% decline without policy)
- **Energy Demand Trend** (~1.5% growth)

---

## 🧠 ML PERFORMANCE
The system automatically compares:
1. **Multiple Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting (GBM)**

It selects the one with the highest **R² Score** for the most accurate simulations.

---

## 💡 TIPS FOR REALISTIC SIMULATION
1. **Don't aim for zero**: Real economies have "residual emissions." The system floors at ~10% for essential activities.
2. **Combine Factors**: Forest protection + energy efficiency + renewables is more effective than any single factor.
3. **Start Year Matters**: Delaying policy starts significantly increases the gap to the target.

---

**Happy Forecasting! 🌍📊**

---

## 📥 EXPORTING DATA

### From the Dashboard
1. Expand "View Detailed Forecast Data"
2. Right-click the table
3. Copy or download as needed

### Programmatically
```python
from core import load_emission_data, train_emission_model, create_predictor

df = load_emission_data("data/emission.csv")
model = train_emission_model(df)
predictor = create_predictor(model)
baseline = predictor.get_baseline_forecast(df, years_ahead=10)

# Export to CSV
baseline.to_csv("forecast_output.csv", index=False)
```

---

## 🌐 SHARING YOUR DASHBOARD

### Option 1: Local Network
```bash
python -m streamlit run ui_streamlit/app.py --server.address 0.0.0.0
```
Others on your network can access via: `http://YOUR_IP:8501`

### Option 2: Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect your repository
4. Deploy!

### Option 3: Ngrok (Temporary Public URL)
```bash
# Install ngrok
# Run in separate terminal:
ngrok http 8501
```

---

## 📚 LEARNING RESOURCES

### Understanding the Model
- **Linear Regression**: Predicts trend based on historical pattern
- **R² Score**: How well model fits data (closer to 1 = better)
- **MAE**: Average prediction error in million tonnes
- **RMSE**: Root mean squared error (penalizes large errors)

### Scenario Simulation
- **Impact Factor**: Combined effect of all policies
- **Progressive Impact**: Effects increase over time
- **Cumulative Effect**: Year-over-year compounding

### Data Requirements
- Minimum 10 years of historical data recommended
- More data = better predictions
- Consistent measurement units required

---

## 🎯 COMMON QUESTIONS

**Q: Why do scenarios diverge over time?**
A: Policy impacts compound progressively, creating larger differences in later years.

**Q: Can emissions go negative?**
A: No, the model floors emissions at 0 (can't have negative pollution).

**Q: How accurate are the predictions?**
A: Check the R² score in "Model Information". Higher = more reliable.

**Q: Can I add more scenario presets?**
A: Yes! Edit `core/scenario.py` and add to `_define_preset_scenarios()`.

**Q: What if my target year isn't in the forecast?**
A: The system automatically uses the closest available year.

---

## 🚀 ADVANCED USAGE

### Batch Scenario Testing
```python
from core import *

df = load_emission_data("data/emission.csv")
model = train_emission_model(df)
predictor = create_predictor(model)
baseline = predictor.get_baseline_forecast(df, 20)

scenarios = [
    (50, 60, -20, "Aggressive"),
    (30, 40, 0, "Moderate"),
    (10, 20, 20, "Conservative")
]

for ren, fos, ind, name in scenarios:
    result = simulate_custom_scenario(baseline, ren, fos, ind, 2025)
    print(f"{name}: {result.iloc[-1]['Scenario_Emission']:.2f} MT")
```

### Custom Visualization
```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=baseline['Year'], y=baseline['Predicted_Emission']))
fig.show()
```

---

## 🎉 YOU'RE READY!

**Open http://localhost:8501 and start exploring!**

For detailed documentation, see `README.md`
For project overview, see `PROJECT_SUMMARY.md`

---

**Happy Forecasting! 🌍📊**
