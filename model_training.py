# model_training.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('german_credit_data.csv')

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Convert target values to binary (1 = good, 0 = bad)
df['Credit_risk'] = df['Credit_risk'].map({1: 1, 2: 0})

# Split into features and target
X = df.drop('Credit_risk', axis=1)
y = df['Credit_risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and label encoders
joblib.dump(model, 'credit_rf_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("Model and encoders saved successfully.")
