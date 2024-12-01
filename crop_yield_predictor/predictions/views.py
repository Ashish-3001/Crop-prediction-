import pandas as pd
import pickle
from django.shortcuts import render
from .forms import PredictionForm  # Import the form
from .forms import EconomicImpactForm, AdaptationPredictionForm, ClimateImpactForm, ClimateImpactLSTMForm
from tensorflow.keras.models import load_model
import numpy as np
import joblib

# Load the model and preprocessor once
with open('predictions/models/gradient_boosting_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('predictions/models/preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)


with open('predictions/models/preprocessor2.pkl', 'rb') as file:
    preprocessor1 = pickle.load(file)
with open('predictions/models/random_forest_model.pkl', 'rb') as file:
    model1 = pickle.load(file)


model2 = load_model('predictions/models/neural_network_model.h5')
preprocessor2 = joblib.load('predictions/models/preprocessor3.pkl')


# Load and Use Both Pickle Files (Model and Preprocessor)
with open('predictions/models/preprocessor4.pkl', 'rb') as file:
    loaded_preprocessor = pickle.load(file)
with open('predictions/models/dnn_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# Load and use both pickle files (model and preprocessor)
with open('predictions/models/preprocessor5.pkl', 'rb') as file:
    loaded_preprocessor1 = pickle.load(file)
loaded_model1 = load_model('predictions/models/rnn_model.h5')

with open('predictions/models/v1.pkl', 'rb') as f:
    visualizations1 = pickle.load(f)

with open('predictions/models/v2.pkl', 'rb') as f:
    visualizations2 = pickle.load(f)

with open('predictions/models/v3.pkl', 'rb') as f:
    visualizations3 = pickle.load(f)

with open('predictions/models/v4.pkl', 'rb') as f:
    visualizations4 = pickle.load(f)

with open('predictions/models/v5.pkl', 'rb') as f:
    visualizations5 = pickle.load(f)

def Home(request):
    return render(request, 'Home.html')

def predict_view(request):
    result = None
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Collect input data into a DataFrame
            input_data = pd.DataFrame([form.cleaned_data])

            # Ensure all expected columns are present
            missing_columns = set(preprocessor.feature_names_in_) - set(input_data.columns)
            for col in missing_columns:
                input_data[col] = 0  # Default value for missing columns

            # Preprocess the data
            transformed_data = preprocessor.transform(input_data)

            # Make prediction
            prediction = model.predict(transformed_data)
            
            print(prediction)
            # Format result
            result = f"Predicted Crop Yield: {prediction[0]:.2f} MT/HA"
    else:
        form = PredictionForm()
    print(visualizations1)
    return render(request, 'predict.html', {'form': form, 'result': result, 'visualizations': visualizations1})

def predict_view_1(request):

    result = None
    if request.method == 'POST':
        form = EconomicImpactForm(request.POST)
        if form.is_valid():
            # Collect input data into a DataFrame
            input_data = pd.DataFrame([form.cleaned_data])

            # Preprocess the data
            transformed_data = preprocessor1.transform(input_data)

            # Make prediction
            prediction = model1.predict(transformed_data)

            # Format result
            result = f"Predicted Economic Impact: {prediction[0]:.2f} Million USD"
    else:
        form = EconomicImpactForm()
    print(visualizations2)
    return render(request, 'predict_1.html', {'form': form, 'result': result, 'visualizations': visualizations2})

# List of strategies for mapping predictions
ADAPTATION_STRATEGIES = [
    'Crop Rotation',
    'Drought-resistant Crops',
    'No Adaptation',
    'Organic Farming',
    'Water Management'
]
def predict_view_2(request):
    prediction = None

    if request.method == 'POST':
        form = AdaptationPredictionForm(request.POST)
        if form.is_valid():
            # Get form data
            data = {
                'Adaptation_Strategies_Crop Rotation': [form.cleaned_data['crop_rotation']],
                'Adaptation_Strategies_Drought-resistant Crops': [form.cleaned_data['drought_resistant_crops']],
                'Adaptation_Strategies_No Adaptation': [form.cleaned_data['no_adaptation']],
                'Adaptation_Strategies_Organic Farming': [form.cleaned_data['organic_farming']],
                'Adaptation_Strategies_Water Management': [form.cleaned_data['water_management']],
            }

            # Convert data to DataFrame
            input_df = pd.DataFrame(data)

            # Preprocess the input (standardization)
            input_transformed = preprocessor2.transform(input_df)

            # Make prediction using the trained model
            predictions = model2.predict(input_transformed)

            # Get the predicted class (index of max value in the softmax output)
            print(predictions)
            predicted_class = np.argmax(predictions, axis=1)[0]
            prediction = ADAPTATION_STRATEGIES[predicted_class]

    else:
        form = AdaptationPredictionForm()

    return render(request, 'predict_adaptation.html', {
        'form': form,
        'prediction': prediction,
        'visualizations': visualizations3
    })

def predict_view_3(request):
    prediction = None
    form = ClimateImpactForm(request.POST or None)
    
    if form.is_valid():
        # Extract the form data
        avg_temp = form.cleaned_data['avg_temp']
        co2_emissions = form.cleaned_data['co2_emissions']
        extreme_weather_0 = form.cleaned_data['extreme_weather_0']
        extreme_weather_1 = form.cleaned_data['extreme_weather_1']

        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            'Average_Temperature_C': [avg_temp],
            'CO2_Emissions_MT': [co2_emissions],
            'Extreme_Weather_Events_0': [extreme_weather_0],
            'Extreme_Weather_Events_1': [extreme_weather_1]
        })

        # Preprocess the input data using the loaded preprocessor
        input_transformed = loaded_preprocessor.transform(input_data)

        # Make the prediction using the loaded model
        custom_prediction = loaded_model.predict(input_transformed).flatten()[0]

        # Format the prediction message
        if custom_prediction > 0:
            prediction = f"Predicted Total Climate Impact: {custom_prediction:.2f}% rise in temperature"
        elif custom_prediction < 0:
            prediction = f"Predicted Total Climate Impact: {custom_prediction:.2f}% dip in temperature"
        else:
            prediction = f"Predicted Total Climate Impact: {custom_prediction:.2f}. No impact!"
    
    return render(request, 'predict_form.html', {'form': form, 'prediction': prediction, 'visualizations': visualizations4})

def predict_view_4(request):
    prediction = None
    form = ClimateImpactLSTMForm(request.POST or None)
    
    if form.is_valid():
        # Extract form data
        avg_temp = form.cleaned_data['average_temperature']
        total_precip = form.cleaned_data['total_precipitation']
        co2_emissions = form.cleaned_data['co2_emissions']
        country = form.cleaned_data['country']
        crop_type = form.cleaned_data['crop_type']
        adaptation_strategy = form.cleaned_data['adaptation_strategy']
        
        # Create input data
        input_data = {
            'Average_Temperature_C': [avg_temp],
            'Total_Precipitation_mm': [total_precip],
            'CO2_Emissions_MT': [co2_emissions],
            'Country_Argentina': [1 if country == "Argentina" else 0],
            'Country_Australia': [1 if country == "Australia" else 0],
            'Country_Brazil': [1 if country == "Brazil" else 0],
            'Country_Canada': [1 if country == "Canada" else 0],
            'Country_China': [1 if country == "China" else 0],
            'Country_France': [1 if country == "France" else 0],
            'Country_India': [1 if country == "India" else 0],
            'Country_Nigeria': [1 if country == "Nigeria" else 0],
            'Country_Russia': [1 if country == "Russia" else 0],
            'Country_USA': [1 if country == "USA" else 0],
            'Crop_Type_Barley': [1 if crop_type == "Barley" else 0],
            'Crop_Type_Coffee': [1 if crop_type == "Coffee" else 0],
            'Crop_Type_Corn': [1 if crop_type == "Corn" else 0],
            'Crop_Type_Cotton': [1 if crop_type == "Cotton" else 0],
            'Crop_Type_Fruits': [1 if crop_type == "Fruits" else 0],
            'Crop_Type_Rice': [1 if crop_type == "Rice" else 0],
            'Crop_Type_Soybeans': [1 if crop_type == "Soybeans" else 0],
            'Crop_Type_Sugarcane': [1 if crop_type == "Sugarcane" else 0],
            'Crop_Type_Vegetables': [1 if crop_type == "Vegetables" else 0],
            'Crop_Type_Wheat': [1 if crop_type == "Wheat" else 0],
            'Adaptation_Strategies_Crop Rotation': [1 if adaptation_strategy == "Crop Rotation" else 0],
            'Adaptation_Strategies_Drought-resistant Crops': [1 if adaptation_strategy == "Drought-resistant Crops" else 0],
            'Adaptation_Strategies_No Adaptation': [1 if adaptation_strategy == "No Adaptation" else 0],
            'Adaptation_Strategies_Organic Farming': [1 if adaptation_strategy == "Organic Farming" else 0],
            'Adaptation_Strategies_Water Management': [1 if adaptation_strategy == "Water Management" else 0]
        }

        
        # Ensure all columns are present
        all_columns = [
            'Average_Temperature_C', 'Total_Precipitation_mm', 'CO2_Emissions_MT',
            'Country_Argentina', 'Country_Australia', 'Country_Brazil', 'Country_Canada',
            'Country_China', 'Country_France', 'Country_India', 'Country_Nigeria',
            'Country_Russia', 'Country_USA', 'Crop_Type_Barley', 'Crop_Type_Coffee',
            'Crop_Type_Corn', 'Crop_Type_Cotton', 'Crop_Type_Fruits', 'Crop_Type_Rice',
            'Crop_Type_Soybeans', 'Crop_Type_Sugarcane', 'Crop_Type_Vegetables',
            'Crop_Type_Wheat', 'Adaptation_Strategies_Crop Rotation',
            'Adaptation_Strategies_Drought-resistant Crops',
            'Adaptation_Strategies_No Adaptation',
            'Adaptation_Strategies_Organic Farming',
            'Adaptation_Strategies_Water Management'
        ]

        input_df = pd.DataFrame(input_data).reindex(columns=all_columns, fill_value=0)
        # Preprocess and reshape input
        transformed_input = loaded_preprocessor1.transform(input_df)
        transformed_input = transformed_input.reshape(1, 1, transformed_input.shape[1])

        # Make prediction
        custom_prediction = loaded_model1.predict(transformed_input).flatten()[0]
        if custom_prediction > 0:
            prediction = f"{custom_prediction:.2f}% rise in temperature"
        elif custom_prediction < 0:
            prediction = f"{custom_prediction:.2f}% dip in temperature"
        else:
            prediction = "No significant impact"

    return render(request, 'predict_lstm_form.html', {'form': form, 'prediction': prediction, 'visualizations': visualizations5})