# B.Tech Weather Prediction Project

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset

try:
    df = pd.read_csv('weather.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset file not found. Make sure the file is in the same folder.")
    exit()

# Step 2: Preprocess the data
# Drop the date column as it's not needed for this simple model
df = df.drop('date', axis=1)

# Fill any missing numerical values with the column average (mean)
for col in ['precipitation', 'temp_max', 'temp_min', 'wind']:
    if df[col].isnull().any():
        df[col].fillna(df[col].mean(), inplace=True)

# Convert the 'weather' column from text to numbers
le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df['weather'])

# Prepare the final processed dataframe
df_processed = df.drop('weather', axis=1)

# Step 3: Define features (X) and target (y)
X = df_processed[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = df_processed['weather_encoded']

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and train the Random Forest model
print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# Step 7: Make a prediction on new data
print("\n--- Making a New Prediction ---")
# New data: precipitation=0, temp_max=15, temp_min=8, wind=5.0
new_data = [[0, 15, 8, 5.0]]
prediction_encoded = model.predict(new_data)
prediction_weather = le.inverse_transform(prediction_encoded)

print(f"Input Data: {new_data[0]}")
print(f"Predicted Weather: {prediction_weather[0]}")
