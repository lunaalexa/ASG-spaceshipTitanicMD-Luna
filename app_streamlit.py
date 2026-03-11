"""
Streamlit App
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import mlflow
from pre_processing import feature_engineering 
# Load preprocessor and model
scaler = joblib.load(Path(__file__).parent/"artifacts/preprocessor.pkl")
model       = joblib.load(Path(__file__).parent/"artifacts/model.pkl")

def main():
    st.title("ASG 04 MD - Luna - Spaceship Titanic Model Deployment")

    # dibagi jadi 2 kolom
    col1, col2 = st.columns(2)

    with col1:
        passenger_id=st.text_input("PassengerId", value="0001_01")
        home_planet=st.selectbox("HomePlanet", ["Earth","Europa","Mars"], index=0)
        cryosleep=st.selectbox("CryoSleep", [True,False], index=1)
        cabin=st.text_input("Cabin (Deck/Num/Side)", value="B/0/P")
        destination=st.selectbox("Destination", ["TRAPPIST-1e","PSO J318.5-22","55 Cancri e"], index=0)
        age=st.number_input("Age", min_value=0, max_value=100, value=30)
        vip=st.selectbox("VIP Status", [True,False], index=1)
    with col2:
        room_service=st.number_input("RoomService bill", value=0.0)
        food_court=st.number_input("FoodCourt bill", value=0.0)
        shopping_mall=st.number_input("ShoppingMall bill", value=0.0)
        spa=st.number_input("Spa bill", value=0.0)
        vr_deck=st.number_input("VRDeck bill", value=0.0)
        name=st.text_input("Passenger Name", value="Joko")

    if st.button("Make Prediction"):
        features = pd.DataFrame([{
            'PassengerId': passenger_id, 'HomePlanet': home_planet, 'CryoSleep': cryosleep,
            'Cabin': cabin, 'Destination': destination, 'Age': age, 'VIP': vip,
            'RoomService': room_service, 'FoodCourt': food_court, 
            'ShoppingMall': shopping_mall, 'Spa': spa, 'VRDeck': vr_deck, 'Name': name
        }])
        
        result = make_prediction(features)   
        if result == "Transported":
            st.success(f"Predicted Result: {result}")
        else:
            st.error(f"Predicted Result: {result}")

def make_prediction(features):
    df_fe=feature_engineering(features)
    X=df_fe[scaler['feature_columns']]
    for col in scaler['categorical_features']:
        le=scaler['label_encoders'][col]
        X[col]=X[col].astype(str).map(lambda x: x if x in le.classes_ else 'Unknown')
        X[col]=le.transform(X[col])

    X_scaled = scaler['scaler'].transform(X)
    prediction = model.predict(X_scaled)
    return "Transported" if prediction[0] == 1 else "Not Transported"

if __name__ == "__main__":
    main()