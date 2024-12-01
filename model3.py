import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib

# Load the dataset
file_path = 'eda_data.csv'  # Adjust the path to your local file if necessary
data = pd.read_csv(file_path)

# Define adaptation strategy columns (features)
adaptation_columns = [
    'Adaptation_Strategies_Crop Rotation', 
    'Adaptation_Strategies_Drought-resistant Crops', 
    'Adaptation_Strategies_No Adaptation', 
    'Adaptation_Strategies_Organic Farming', 
    'Adaptation_Strategies_Water Management'
]

# Validate that the required columns exist in the dataset
missing_columns = set(adaptation_columns) - set(data.columns)
if missing_columns:
    raise ValueError(f"Missing required columns in the dataset: {missing_columns}")

# Prepare the features (adaptation strategies columns)
X = data[adaptation_columns]

# Create a preprocessor for the features
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler())  # Standardize numerical features
])

# Fit and transform the features
X_transformed = preprocessor.fit_transform(X)

# Select a random region column starting with 'Region_'
region_columns = [col for col in data.columns if col.startswith('Region_')]
if not region_columns:
    raise ValueError("No columns starting with 'Region_' found in the dataset.")

selected_region = random.choice(region_columns)

# Prepare the target (region column for classification)
y = data[selected_region].astype(int)  # Convert to binary (0 for no, 1 for yes)

# One-hot encode the target labels (binary classification)
y_encoded = to_categorical(y, num_classes=2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_encoded, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    Dense(64, activation='relu'),  # Hidden layer
    Dense(2, activation='softmax')  # Output layer (2 classes for binary classification)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Test Accuracy: {accuracy:.4f}")

# Save the model and preprocessor
model_filename = 'neural_network_model.h5'
preprocessor_filename = 'preprocessor.pkl'

model.save(model_filename)  # Save the trained model using TensorFlow's save function
joblib.dump(preprocessor, preprocessor_filename)  # Save the preprocessor

print(f"Model saved to {model_filename}")
print(f"Preprocessor saved to {preprocessor_filename}")

# Load the model and preprocessor
loaded_model = load_model(model_filename)
loaded_preprocessor = joblib.load(preprocessor_filename)

# Example: Make predictions on new data
new_data = pd.DataFrame({
    'Adaptation_Strategies_Crop Rotation': [1],
    'Adaptation_Strategies_Drought-resistant Crops': [0],
    'Adaptation_Strategies_No Adaptation': [0],
    'Adaptation_Strategies_Organic Farming': [0],
    'Adaptation_Strategies_Water Management': [0]
})

# Preprocess the new data
new_data_transformed = loaded_preprocessor.transform(new_data)

# Make predictions using the trained model
new_predictions = loaded_model.predict(new_data_transformed)

# Get the predicted class (index of max value in the softmax output)
new_predicted_classes = np.argmax(new_predictions, axis=1)
adaptation_strategies = [
    'Crop Rotation', 
    'Drought-resistant Crops', 
    'No Adaptation', 
    'Organic Farming', 
    'Water Management'
]
new_predicted_adaptation_strategy = adaptation_strategies[new_predicted_classes[0]]

print(f"Predicted Adaptation Strategy for {selected_region}: {new_predicted_adaptation_strategy}")
