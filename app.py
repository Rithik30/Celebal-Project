# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Creditworthiness Prediction", layout="centered")
st.title("💳 Creditworthiness Predictor App")
st.markdown("Using a trained **Random Forest Classifier** on the German Credit dataset.")

# Load model and encoders
model = joblib.load("credit_rf_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Load original dataset for reference
df_raw = pd.read_csv("german_credit_data.csv")

# Sidebar input
st.sidebar.header("Enter Customer Details")

user_input = {}
for col in feature_names:
    if df_raw[col].dtype == 'object':
        user_input[col] = st.sidebar.selectbox(col, sorted(df_raw[col].unique()))
    else:
        user_input[col] = st.sidebar.number_input(col,
                                                  min_value=int(df_raw[col].min()),
                                                  max_value=int(df_raw[col].max()),
                                                  value=int(df_raw[col].mean()))

# Convert input to dataframe
input_df = pd.DataFrame([user_input])

# Encode user input
for col in input_df.select_dtypes(include='object').columns:
    le = joblib.load('label_encoder.pkl')
    input_df[col] = le.fit(df_raw[col]).transform(input_df[col])

# Predict
if st.button("Predict Creditworthiness"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.success(f"✅ This person is likely **creditworthy** with {proba*100:.2f}% confidence.")
    else:
        st.error(f"❌ This person is likely **not creditworthy** with {proba*100:.2f}% confidence.")

st.markdown("---")
st.markdown("📊 Sample of dataset used:")
st.dataframe(df_raw.head(10))
