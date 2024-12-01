import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

# Load the dataset
file_path = 'data.csv'  # Adjust the path to your local file if necessary
dataset = pd.read_csv(file_path)

# Define features and target variable
X = dataset.drop(columns=['Crop_Yield_MT_per_HA'])  # Features
y = dataset['Crop_Yield_MT_per_HA']  # Target

# Identify categorical and numerical columns
categorical_columns = ['Country', 'Region', 'Crop_Type', 'Adaptation_Strategies']
numerical_columns = ['Average_Temperature_C', 'Total_Precipitation_mm',
                     'CO2_Emissions_MT', 'Extreme_Weather_Events',
                     'Irrigation_Access_%', 'Pesticide_Use_KG_per_HA',
                     'Fertilizer_Use_KG_per_HA', 'Soil_Health_Index',
                     'Economic_Impact_Million_USD']

# Create a preprocessor with numerical and categorical transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
            ('scaler', StandardScaler())  # Standardize numerical features
        ]), numerical_columns),
        ('cat', OneHotEncoder(drop='first'), categorical_columns)  # One-hot encode categorical features
    ]
)

# Fit the preprocessor on the dataset
preprocessor.fit(X)

# Save the preprocessor to a pickle file
preprocessor_filename = 'preprocessor.pkl'
with open(preprocessor_filename, 'wb') as file:
    pickle.dump(preprocessor, file)
print(f"Preprocessor saved to {preprocessor_filename}")

# Transform the dataset
X_transformed = preprocessor.transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting model
gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train the model
gradient_boosting_model.fit(X_train, y_train)

# Evaluate the model
y_pred = gradient_boosting_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Gradient Boosting Model Results:")
print(f"  MSE: {mse:.4f}")
print(f"  R²: {r2:.4f}")

# Save the trained model to a pickle file
model_filename = 'gradient_boosting_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(gradient_boosting_model, file)
print(f"Model saved to {model_filename}")

# Load and Use Both Pickle Files
with open(preprocessor_filename, 'rb') as file:
    loaded_preprocessor = pickle.load(file)
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Use the loaded preprocessor and model for predictions
# Example: Custom input data for prediction
custom_input_data = {
    'Country': ['India'],
    'Region': ['West Bengal'],
    'Crop_Type': ['Corn'],
    'Adaptation_Strategies': ['Water Management'],
    'Average_Temperature_C': [1.55],
    'Total_Precipitation_mm': [447.06],
    'CO2_Emissions_MT': [15.22],
    'Extreme_Weather_Events': [8],
    'Irrigation_Access_%': [14.54],
    'Pesticide_Use_KG_per_HA': [10.08],
    'Fertilizer_Use_KG_per_HA': [14.78],
    'Soil_Health_Index': [83.25],
    'Economic_Impact_Million_USD': [808.13]
}

# Convert custom input to DataFrame
custom_input_df = pd.DataFrame(custom_input_data)

# Preprocess custom input using the loaded preprocessor
custom_input_transformed = loaded_preprocessor.transform(custom_input_df)

# Make prediction using the loaded model
custom_prediction = loaded_model.predict(custom_input_transformed)
print(f"Predicted Crop Yield for custom input: {custom_prediction[0]:.4f} MT/HA")
