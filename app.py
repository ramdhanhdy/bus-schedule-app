import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random

# Placeholder for ML model (replace this with your actual model)
def predict_duration(destination):
    return random.randint(5, 60)

# Function to estimate arrival time
def estimate_arrival_time(current_time, duration):
    return (current_time + timedelta(minutes=duration)).strftime("%I:%M %p")

# Set page config
st.set_page_config(page_title="Larkin Sentral Terminal", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #4285F4;
        margin-bottom: 0;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 20px;
        text-align: center;
    }
    .bus-stop {
        font-weight: bold;
        color: white;
        background-color: #5f9ea0;
        border-radius: 5px;
        padding: 5px 10px;
        display: inline-block;
        width: 60px;
        text-align: center;
    }
    .stDataFrame {
        font-size: 18px;
        margin: 0 auto;
    }
    .stDataFrame td:nth-child(3) {
        font-weight: bold;
        color: #4CAF50;
    }
    .stDataFrame th {
        background-color: #f0f2f6;
        font-weight: bold;
        color: #333;
    }
    .stDataFrame td {
        background-color: white;
    }
    .dataframe {
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Title and current time
st.markdown('<p class="title">Bus Sekolah</p>', unsafe_allow_html=True)
current_time = datetime.now()
st.markdown(f'<p class="subtitle">{current_time.strftime("%A, %d %B %Y, %I:%M:%S %p")}</p>', unsafe_allow_html=True)

# Create sample data
data = {
    "Route": ["10", "11"],
    "Destination": ["JB Sentral Bus Terminal", "Tmn Universiti Terminal", "Kulai Bus Terminal"]
}

df = pd.DataFrame(data)

# Predict durations and estimate arrival times
df["Duration"] = df["Destination"].apply(predict_duration)
df["Arrival Time"] = df.apply(lambda row: estimate_arrival_time(current_time, row["Duration"]), axis=1)

# Format bus numbers with HTML
df["Route"] = df["Route"].apply(lambda x: f'<span class="bus-stop">{x}</span>')
df["Destination"] = df["Destination"].apply(lambda x: f'<span class="bus-stop">{x}</span>')

# Display the schedule using Streamlit's dataframe
st.write(df.to_html(escape=False, index=False, classes='dataframe'), unsafe_allow_html=True)

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)