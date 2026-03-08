import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv("weatherHistory.csv")

# Handle missing values
df['Precip Type'].fillna(df['Precip Type'].mode()[0], inplace=True)

# Convert date
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], errors='coerce', utc=True)

# Extract date features
df['year'] = df['Formatted Date'].dt.year
df['month'] = df['Formatted Date'].dt.month
df['day'] = df['Formatted Date'].dt.day
df['hour'] = df['Formatted Date'].dt.hour

# Drop original column
df.drop('Formatted Date', axis=1, inplace=True)

# Encode categorical data
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop(['Temperature (C)', 'Apparent Temperature (C)'], axis=1)
y = df['Temperature (C)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

predictions = model.predict(X_test)

print("Training Score:", train_score)
print("Testing Score:", test_score)
print("R2 Score:", r2_score(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))

# Save model
import pickle

with open("weather_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature names
with open("features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Model and features saved successfully")