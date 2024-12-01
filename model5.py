import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# Load the dataset
file_path = 'eda_data.csv'  # Adjust the path to your local file if necessary
data = pd.read_csv(file_path)

# Define features and target variable
features = data[['Average_Temperature_C', 'Total_Precipitation_mm', 'CO2_Emissions_MT', 
                 'Country_Argentina', 'Country_Australia', 'Country_Brazil', 'Country_Canada', 
                 'Country_China', 'Country_France', 'Country_India', 'Country_Nigeria', 'Country_Russia', 
                 'Country_USA', 'Crop_Type_Barley', 'Crop_Type_Coffee', 'Crop_Type_Corn', 
                 'Crop_Type_Cotton', 'Crop_Type_Fruits', 'Crop_Type_Rice', 'Crop_Type_Soybeans', 
                 'Crop_Type_Sugarcane', 'Crop_Type_Vegetables', 'Crop_Type_Wheat', 
                 'Adaptation_Strategies_Crop Rotation', 'Adaptation_Strategies_Drought-resistant Crops', 
                 'Adaptation_Strategies_No Adaptation', 'Adaptation_Strategies_Organic Farming', 
                 'Adaptation_Strategies_Water Management']]

target = data['Total_Climate_Impact']

# Preprocess the features (Handle missing values, scale)
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler())  # Scale the features
])

# Fit and transform the features
features_scaled = preprocessor.fit_transform(features)

# Save the preprocessor to a pickle file
preprocessor_filename = 'preprocessor5.pkl'
with open(preprocessor_filename, 'wb') as file:
    pickle.dump(preprocessor, file)
print(f"Preprocessor saved to {preprocessor_filename}")

# Reshape the data to match the LSTM input format (samples, time_steps, features)
X_scaled = features_scaled.reshape(features_scaled.shape[0], 1, features_scaled.shape[1])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer=Adam(), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R² Score: {r2:.4f}')

# Save the trained model using TensorFlow's save format
model_filename = 'rnn_model.h5'
model.save(model_filename)
print(f"Model saved to {model_filename}")

# Load and use both pickle files (model and preprocessor)
with open(preprocessor_filename, 'rb') as file:
    loaded_preprocessor = pickle.load(file)

loaded_model = Sequential()
loaded_model = load_model(model_filename)

# Example: Custom input data for prediction
custom_input_data = {
    'Average_Temperature_C': [1.55], 'Total_Precipitation_mm': [447.06], 'CO2_Emissions_MT': [15.22],
    'Country_Argentina': [0], 'Country_Australia': [0], 'Country_Brazil': [0], 'Country_Canada': [0],
    'Country_China': [0], 'Country_France': [0], 'Country_India': [1], 'Country_Nigeria': [0], 
    'Country_Russia': [0], 'Country_USA': [0], 'Crop_Type_Barley': [0], 'Crop_Type_Coffee': [0], 
    'Crop_Type_Corn': [1], 'Crop_Type_Cotton': [0], 'Crop_Type_Fruits': [0], 'Crop_Type_Rice': [0], 
    'Crop_Type_Soybeans': [0], 'Crop_Type_Sugarcane': [0], 'Crop_Type_Vegetables': [0], 'Crop_Type_Wheat': [0],
    'Adaptation_Strategies_Crop Rotation': [0], 'Adaptation_Strategies_Drought-resistant Crops': [0], 
    'Adaptation_Strategies_No Adaptation': [0], 'Adaptation_Strategies_Organic Farming': [0], 
    'Adaptation_Strategies_Water Management': [1]
}

# Convert custom input to DataFrame
custom_input_df = pd.DataFrame(custom_input_data)

# Preprocess custom input using the loaded preprocessor
custom_input_transformed = loaded_preprocessor.transform(custom_input_df)

# Reshape the custom input data to match the LSTM input format (1, 1, 28)
custom_input_transformed = custom_input_transformed.reshape(custom_input_transformed.shape[0], 1, custom_input_transformed.shape[1])

# Make prediction using the loaded model
custom_prediction = loaded_model.predict(custom_input_transformed).flatten()[0]
if custom_prediction > 0:
    print(f"Predicted Total Climate Impact for custom input: {custom_prediction:.2f}% rise in temperature")
elif custom_prediction < 0:
    print(f"Predicted Total Climate Impact for custom input: {custom_prediction:.2f}% dip in temperature")
else:
    print(f"Predicted Total Climate Impact for custom input is {custom_prediction:.2f}. No impact!")
