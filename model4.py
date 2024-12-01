import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load the dataset
file_path = 'eda_data.csv'  # Adjust the path to your local file if necessary
data = pd.read_csv(file_path)

# Define features and target variable
features = data[['Average_Temperature_C', 'CO2_Emissions_MT', 'Extreme_Weather_Events_0', 'Extreme_Weather_Events_1']]
target = data['Total_Climate_Impact']

# Preprocess the features (Handle missing values, scale)
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler())  # Scale the features
])

# Fit and transform the features
features_scaled = preprocessor.fit_transform(features)

# Save the preprocessor to a pickle file
preprocessor_filename = 'preprocessor4.pkl'
with open(preprocessor_filename, 'wb') as file:
    pickle.dump(preprocessor, file)
print(f"Preprocessor saved to {preprocessor_filename}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Build the DNN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer with 64 neurons
model.add(Dense(32, activation='relu'))  # Hidden layer with 32 neurons
model.add(Dense(1))  # Output layer (single continuous value)

# Compile the model with Adam optimizer and MSE loss
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Evaluate the model
predictions = model.predict(X_test)
predictions = predictions.flatten()  # Flatten to match y_test shape

# Calculate MSE and R2 score
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R² Score: {r2:.4f}')

# Save the trained model to a pickle file
model_filename = 'dnn_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved to {model_filename}")

# Load and Use Both Pickle Files (Model and Preprocessor)
with open(preprocessor_filename, 'rb') as file:
    loaded_preprocessor = pickle.load(file)

with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Example: Custom input data for prediction
custom_input_data = {
    'Average_Temperature_C': [34.36],
    'CO2_Emissions_MT': [13.11],
    'Extreme_Weather_Events_0': [0],
    'Extreme_Weather_Events_1': [0]
}

# Convert custom input to DataFrame
custom_input_df = pd.DataFrame(custom_input_data)

# Preprocess custom input using the loaded preprocessor
custom_input_transformed = loaded_preprocessor.transform(custom_input_df)

# Make prediction using the loaded model
custom_prediction = loaded_model.predict(custom_input_transformed).flatten()[0]
if custom_prediction > 0:
    print(f"Predicted Total Climate Impact for custom input: {custom_prediction:.2f}% rise in temperature")
elif custom_prediction < 0:
    print(f"Predicted Total Climate Impact for custom input: {custom_prediction:.2f}% dip in temperature")
else:
    print(f"Predicted Total Climate Impact for custom input is {custom_prediction:.2f}. No impact!")
