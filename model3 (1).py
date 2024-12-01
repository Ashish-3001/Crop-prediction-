import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pickle

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
predictions = model.predict(X_test)

# Get the predicted class (index of max value in the softmax output)
predicted_classes = np.argmax(predictions, axis=1)

# Map the prediction to the corresponding adaptation strategy name
adaptation_strategies = [
    'Crop Rotation', 
    'Drought-resistant Crops', 
    'No Adaptation', 
    'Organic Farming', 
    'Water Management'
]

# Map the true classes from the test set to strategy names
true_classes = np.argmax(y_test, axis=1)
predicted_adaptation_strategies = [adaptation_strategies[i] for i in predicted_classes]
true_adaptation_strategies = [adaptation_strategies[i] for i in true_classes]

# Calculate accuracy based on strategy names
accuracy = np.sum(np.array(predicted_adaptation_strategies) == np.array(true_adaptation_strategies)) / len(true_adaptation_strategies)

print(f"Accuracy for predicting adaptation strategy: {accuracy:.4f}")

# Save the model and preprocessor as pickle files
model_filename = 'neural_network_model.pkl'
preprocessor_filename = 'preprocessor3.pkl'

# Save the model
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)  # Save the entire model using pickle

# Save the preprocessor
with open(preprocessor_filename, 'wb') as preprocessor_file:
    pickle.dump(preprocessor, preprocessor_file)  # Save the preprocessor

print(f"Model and preprocessor saved to {model_filename} and {preprocessor_filename}")

# Load the model and preprocessor from pickle files
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)  # Load the trained model from pickle file

with open(preprocessor_filename, 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)  # Load the preprocessor from pickle file

# Define a function that predicts adaptation strategy and returns selected region
def predict_adaptation_strategy(new_data):
    # Preprocess the new data
    new_data_transformed = preprocessor.transform(new_data)
    
    # Make predictions using the trained model
    new_predictions = model.predict(new_data_transformed)
    
    # Get the predicted class (index of max value in the softmax output)
    new_predicted_classes = np.argmax(new_predictions, axis=1)
    
    # Map the predicted class to adaptation strategy
    new_predicted_adaptation_strategy = adaptation_strategies[new_predicted_classes[0]]
    
    # Return the prediction along with the selected region
    return new_predicted_adaptation_strategy, selected_region

# Example: Make predictions on new data
new_data = pd.DataFrame({
    'Adaptation_Strategies_Crop Rotation': [1],
    'Adaptation_Strategies_Drought-resistant Crops': [0],
    'Adaptation_Strategies_No Adaptation': [0],
    'Adaptation_Strategies_Organic Farming': [0],
    'Adaptation_Strategies_Water Management': [0]
})

# Call the function to get the predicted adaptation strategy and the selected region
predicted_adaptation_strategy, region = predict_adaptation_strategy(new_data)

print(f"Predicted Adaptation Strategy for {region}: {predicted_adaptation_strategy}")
