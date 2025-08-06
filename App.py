import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Download model from Hugging Face
model_path = hf_hub_download(
    repo_id="hugginfaceismyid/Flight_PP",  # Your Hugging Face repo
    filename="model.pkl"       # File name in the repo
)

# Load model
model = joblib.load(model_path)

# App title
st.title('Flight Price Predictor')

# Inputs
airline = st.selectbox("Airline", ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India'])
source_city = st.selectbox("Source City", ['Delhi', 'Mumbai', 'Kolkata', 'Hyderabad', 'Bangalore', 'Chennai'])
destination_city = st.selectbox("Destination City", ['Delhi', 'Mumbai', 'Kolkata', 'Hyderabad', 'Bangalore', 'Chennai'])
departure_time = st.selectbox("Departure Time", ['Morning', 'Afternoon', 'Evening', 'Early_Morning'])
arrival_time = st.selectbox("Arrival Time", ['Morning', 'Afternoon', 'Evening', 'Night', 'Early_Morning'])
stops = st.selectbox("Stops", ['zero', 'one'])
flight_class = st.selectbox("Class", ['Economy'])
duration = st.slider("Flight Duration (hrs)", min_value=1.0, max_value=20.0, step=0.25)
days_left = st.slider("Days Left to Departure", min_value=1, max_value=30)

# DataFrame for prediction
input_df = pd.DataFrame({
    'airline': [airline],
    'source_city': [source_city],
    'departure_time': [departure_time],
    'stops': [stops],
    'arrival_time': [arrival_time],
    'destination_city': [destination_city],
    'class': [flight_class],
    'duration': [duration],
    'days_left': [days_left]
})

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Flight Price: ₹{int(prediction):,}")

