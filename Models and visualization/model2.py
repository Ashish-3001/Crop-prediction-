import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

# Load the dataset
file_path = 'eda_data.csv' # Adjust the path to your local file if necessary
dataset = pd.read_csv(file_path)

# Define features and target variable

X = dataset[['Average_Temperature_C', 'CO2_Emissions_MT']] # Features
y = dataset['Economic_Impact_Million_USD']  # Target

numerical_columns = ['Average_Temperature_C', 'CO2_Emissions_MT']

# Create a preprocessor with numerical and categorical transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
            ('scaler', StandardScaler())  # Standardize numerical features
        ]), numerical_columns)
    ]
)

# Fit the preprocessor on the dataset
preprocessor.fit(X)

# Save the preprocessor to a pickle file
preprocessor_filename = 'preprocessor2.pkl'
with open(preprocessor_filename, 'wb') as file:
    pickle.dump(preprocessor, file)
    print(f"Preprocessor saved to {preprocessor_filename}")

# Transform the dataset
X_transformed = preprocessor.transform(X)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_transformed, y)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# # Initialize the Random Forest model
# random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

# # Train the model
# random_forest_model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_transformed)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Random Forest Model Results:")
print(f"  MSE: {mse:.4f}")
print(f"  R²: {r2:.4f}")

# Save the trained model to a pickle file
model_filename = 'random_forest_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved to {model_filename}")

# Load and Use Both Pickle Files
with open(preprocessor_filename, 'rb') as file:
    loaded_preprocessor = pickle.load(file)
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Use the loaded preprocessor and model for predictions
# # Example: Custom input data for prediction
custom_input_data = {
    'Average_Temperature_C': [34.36],
    'CO2_Emissions_MT': [13.11]
}

# Convert custom input to DataFrame
custom_input_df = pd.DataFrame(custom_input_data)

# Preprocess custom input using the loaded preprocessor
custom_input_transformed = loaded_preprocessor.transform(custom_input_df)

# Make prediction using the loaded model
custom_prediction = loaded_model.predict(custom_input_transformed)
print(f"Predicted Economic Impact for custom input: {custom_prediction[0]:.4f} Million USD")
