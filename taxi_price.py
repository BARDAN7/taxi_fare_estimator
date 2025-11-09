import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('xg_model.pt')

#Module to predict data
def data_predict(data):
    df=data.copy()
    
    #Encoding categorica; variables
    Time={'Morning':1,'Afternoon':2,'Evening':3,'Night':4}
    df['Time_of_Day']=df['Time_of_Day'].map(Time)
    Day={'Weekday':1,'Weekend':2}
    df['Day_of_Week']=df['Day_of_Week'].map(Day)
    Traffic={'Low':1,'High':2,'Medium':3}
    df['Traffic_Conditions']=df['Traffic_Conditions'].map(Traffic)
    Weather_={'Clear':1, 'Rain':2, 'Snow':3}
    df['Weather']=df['Weather'].map(Weather_)
    
    return df

# Page configuration
st.set_page_config(
    page_title="Taxi Fare Predictor",
    page_icon="🚕",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 15px;
    }
    h1 {
        color: #0072B5;
        text-align: center;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #0072B5;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #005b8e;
        color: #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🚖 Smart Taxi Fare Prediction App")
st.write("### Enter your trip details below to estimate your taxi fare 💰")

# Sidebar info
st.sidebar.header("ℹ️ About the App")
st.sidebar.write("""
This **Machine Learning** app predicts taxi fare prices based on:
- Distance  
- Duration  
- Traffic and weather  
- Time of day  
- Rate details  
---
Developed using **Python, Scikit-learn & Streamlit**.
""")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    trip_distance = st.number_input("🚗 Trip Distance (km)", min_value=0.0, step=0.1)
    time_of_day = st.selectbox("🕒 Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
    day_of_week = st.selectbox("📅 Day of Week", ["Weekday","Weekend"])
    passenger_count = st.number_input("👥 Passenger Count", min_value=1, step=1)
    traffic = st.selectbox("🚦 Traffic Conditions", ["Low", "High", "Medium"])

with col2:
    weather = st.selectbox("🌤️ Weather", ["Clear", "Rain", "Snow"])
    base_fare = st.number_input("💵 Base Fare (Rs)", min_value=0.0, step=1.0)
    per_km_rate = st.number_input("📍 Per Km Rate (Rs)", min_value=0.0, step=0.5)
    per_minute_rate = st.number_input("⏱️ Per Minute Rate (Rs)", min_value=0.0, step=0.5)
    duration = st.number_input("🕐 Trip Duration (Minutes)", min_value=0.0, step=1.0)

# Prediction button
if st.button("🔮 Predict Fare"):
    # Make input dataframe
    input = pd.DataFrame([{
        'Trip_Distance_km': trip_distance,
        'Time_of_Day': time_of_day,
        'Day_of_Week': day_of_week,
        'Passenger_Count': passenger_count,
        'Traffic_Conditions': traffic,
        'Weather': weather,
        'Base_Fare': base_fare,
        'Per_Km_Rate': per_km_rate,
        'Per_Minute_Rate': per_minute_rate,
        'Trip_Duration_Minutes': duration
    }])

    # Predict
    input_data=data_predict(input)
    predicted_price = model.predict(input_data)[0]

    st.markdown("---")
    st.subheader("🎯 **Estimated Fare**")
    st.success(f"💰 Rs. {predicted_price:.2f}")
    st.balloons()  # 🎉 little animation!

# Footer
st.markdown("""
            ---
            Designed and Developed by Bardan Kc""")
