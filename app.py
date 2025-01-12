import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Climate Prediction", page_icon="🌤️")


class ClimateModel(nn.Module):
    def __init__(self):
        super(ClimateModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.fc(x)


model = ClimateModel()
model.load_state_dict(torch.load("models/climate_model.pth"))
model.eval()
scaler = torch.load("models/scaler.pth")

st.sidebar.title("Main Menu")
option = st.sidebar.radio(
    "Select an option:",
    ("Home", "Predictions", "What-If Scenarios", "Data Visualizations", "Sensitivity Analysis")
)


st.title("Climate Prediction and Analysis")
st.caption("By Team Paradox")



prediction = None
prediction_adjusted = None

def simulate_temperature_rise(current_temp, years, rise_factor=2):
    predicted_temp = current_temp + rise_factor
    temperature_changes = [predicted_temp] * years
    return temperature_changes

def simulate_emissions_reduction(current_emissions, reduction_percentage=50):
    new_emissions = current_emissions * (1 - reduction_percentage / 100)
    return new_emissions

def simulate_sea_level_rise(current_sea_level, rise_factor=3):
    new_sea_level = current_sea_level + rise_factor
    return new_sea_level

def simulate_deforestation_stoppage(current_forest_area, deforestation_rate=0):
    new_forest_area = current_forest_area * (1 + deforestation_rate / 100)
    return new_forest_area

def simulate_el_nino_event(current_event_frequency, increase_factor=1.2):
    new_event_frequency = current_event_frequency * increase_factor
    return new_event_frequency

def simulate_geoengineering_impact(current_temperature, cooling_factor=0.5):
    new_temperature = current_temperature - cooling_factor
    return new_temperature

def plot_scenario(current_temperature, years, scenarios):
    plt.figure(figsize=(10, 6))
    for scenario_name, temperature_changes in scenarios:
        plt.plot(range(years), temperature_changes, label=scenario_name)
    plt.title('Climate Prediction under Various Scenarios')
    plt.xlabel('Years')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()

if option == "Home":
    st.write("""
        The Climate Prediction and Analysis System is a tool leveraging deep learning for forecasting critical
climate variables and generating actionable insights. Designed with scalability and
user-friendliness in mind, the project integrates advanced technologies for data-driven
decision-making in climate risk assessment and scenario modeling.""")


elif option == "Predictions":
    st.subheader("Climate Prediction Inputs")

    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=50.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    pressure = st.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1010.0)
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0)


    if st.button("Predict Temperature (Original Input)"):
       
        input_data = np.array([[rainfall, humidity, pressure, wind_speed]])
        input_data_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

      
        with torch.no_grad():
            prediction = model(input_tensor).item()
        st.success(f"Predicted Temperature: {prediction:.2f} °C")

elif option == "What-If Scenarios":
    st.subheader("What-If Scenarios: Climate Parameter Adjustments")

   
    current_temperature = st.number_input("Current Temperature (°C)", min_value=-50.0, max_value=50.0, value=15.0)
    years = st.slider("Number of Years to Simulate", min_value=10, max_value=100, value=50)

    
    st.subheader("Temperature Increase Simulation (2°C Rise)")
    temperature_changes = simulate_temperature_rise(current_temperature, years)
    st.write(f"Predicted Temperature Rise: {temperature_changes[-1]:.2f} °C after {years} years")
    st.line_chart(temperature_changes)

   
    st.subheader("CO2 Emissions Reduction (50% cut)")
    current_emissions = st.number_input("Current CO2 Emissions (gigatons/year)", min_value=0.0, max_value=100.0, value=40.0)
    new_emissions = simulate_emissions_reduction(current_emissions, reduction_percentage=50)
    st.write(f"Predicted New CO2 Emissions: {new_emissions:.2f} gigatons/year after 50% reduction")

    
    st.subheader("Sea Level Rise Simulation (3 meters)")
    current_sea_level = st.number_input("Current Sea Level Rise (meters)", min_value=0.0, max_value=10.0, value=0.5)
    new_sea_level = simulate_sea_level_rise(current_sea_level, rise_factor=3)
    st.write(f"Predicted Sea Level Rise: {new_sea_level:.2f} meters after 3-meter rise")

   
    st.subheader("Deforestation Stoppage Simulation")
    current_forest_area = st.number_input("Current Forest Area (million hectares)", min_value=0.0, value=1000.0)
    new_forest_area = simulate_deforestation_stoppage(current_forest_area, deforestation_rate=0)
    st.write(f"Predicted Forest Area: {new_forest_area:.2f} million hectares after stopping deforestation")


    st.subheader("El Niño/La Niña Event Frequency Simulation")
    current_event_frequency = st.number_input("Current Event Frequency (events per year)", min_value=0.0, max_value=10.0, value=1.5)
    new_event_frequency = simulate_el_nino_event(current_event_frequency, increase_factor=1.2)
    st.write(f"Predicted Event Frequency: {new_event_frequency:.2f} events per year after increase")

    
    st.subheader("Geoengineering Impact Simulation")
    new_temperature_geoengineering = simulate_geoengineering_impact(current_temperature, cooling_factor=0.5)
    st.write(f"Predicted Temperature After Geoengineering: {new_temperature_geoengineering:.2f} °C")



elif option == "Data Visualizations":
    
    sample_data = pd.read_csv("data/climate_data.csv")

   
    st.subheader("Correlation Heatmap of Climate Variables")
    corr_matrix = sample_data[['temperature', 'rainfall', 'humidity', 'pressure', 'wind_speed']].corr()
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot(fig)

    
    st.subheader("Scatter Plot of Temperature vs. Rainfall")
    fig = px.scatter(sample_data, x='rainfall', y='temperature', title="Temperature vs Rainfall")
    st.plotly_chart(fig)

    
    st.subheader("Histogram of Temperature Distribution")
    fig = px.histogram(sample_data, x="temperature", title="Temperature Distribution")
    st.plotly_chart(fig)

    
    st.subheader("Temperature Trends Over Time")
    fig = px.line(sample_data, x='date', y='temperature', title="Temperature Trends Over Time")
    st.plotly_chart(fig)

elif option == "Sensitivity Analysis":
    st.subheader("Sensitivity Analysis: How Changes in Climate Variables Affect Temperature")

    
    rainfall_range = st.slider("Rainfall (mm)", min_value=0.0, max_value=500.0, value=50.0)
    humidity_range = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    pressure_range = st.slider("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1010.0)
    wind_speed_range = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0)

    
    rain_values = np.linspace(0, 500, 10)
    humidity_values = np.linspace(0, 100, 10)
    pressure_values = np.linspace(900, 1100, 10)
    wind_speed_values = np.linspace(0, 100, 10)

    
    results = []
    for rain in rain_values:
        for hum in humidity_values:
            for press in pressure_values:
                for wind in wind_speed_values:
                    input_data = np.array([[rain, hum, press, wind]])
                    input_data_scaled = scaler.transform(input_data)
                    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
                    with torch.no_grad():
                        temp = model(input_tensor).item()
                    results.append([rain, hum, press, wind, temp])

   
    sensitivity_df = pd.DataFrame(results, columns=["Rainfall", "Humidity", "Pressure", "Wind Speed", "Temperature"])

    
    st.subheader("Sensitivity of Temperature to Climate Variables")
    fig = px.scatter_3d(sensitivity_df, x="Rainfall", y="Humidity", z="Temperature", color="Pressure", title="Sensitivity of Temperature")
    st.plotly_chart(fig)
