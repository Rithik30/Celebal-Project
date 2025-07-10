# 💳 Creditworthiness Prediction Web App
Streamlit web app to predict creditworthiness of individuals using a Random Forest model trained on the UCI German Credit dataset. Includes data pre-processing, model training, and interactive frontend.

A Streamlit-based web application that predicts whether a person is creditworthy or not using a machine learning model trained on the UCI German Credit dataset.

---

## 🚀 Project Overview

This project was developed during an internship and uses a **Random Forest Classifier** to analyze the credit risk of individuals based on their financial attributes. The model is trained on the [UCI German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) and deployed using **Streamlit** for interactive use.

---

## 📂 Project Structure
├── app.py # Streamlit web app
├── model_training.py # Model training and saving
├── convert_data.py # Convert raw .data to CSV
├── german.data # Raw dataset from UCI
├── german_credit_data.csv # Processed dataset
├── credit_rf_model.pkl # Trained Random Forest model
├── label_encoder.pkl # Encoded labels for categorical features
├── feature_names.pkl # Saved feature names used by the model
├── requirements.txt # All Python dependencies
└── README.md # This file
