from django import forms

class PredictionForm(forms.Form):
    Country = forms.CharField()
    Region = forms.CharField()
    Crop_Type = forms.CharField()
    Adaptation_Strategies = forms.CharField()
    Average_Temperature_C = forms.FloatField()
    Total_Precipitation_mm = forms.FloatField()
    CO2_Emissions_MT = forms.FloatField()
    Extreme_Weather_Events = forms.IntegerField()
    Irrigation_Access_ = forms.FloatField()  # Fixed naming here
    Pesticide_Use_KG_per_HA = forms.FloatField()
    Fertilizer_Use_KG_per_HA = forms.FloatField()
    Soil_Health_Index = forms.FloatField()
    Economic_Impact_Million_USD = forms.FloatField()


class EconomicImpactForm(forms.Form):
    Average_Temperature_C = forms.FloatField(label="Average Temperature (°C)")
    CO2_Emissions_MT = forms.FloatField(label="CO2 Emissions (MT)")


class AdaptationPredictionForm(forms.Form):
    crop_rotation = forms.FloatField(label="Crop Rotation", min_value=0, max_value=1)
    drought_resistant_crops = forms.FloatField(label="Drought-resistant Crops", min_value=0, max_value=1)
    no_adaptation = forms.FloatField(label="No Adaptation", min_value=0, max_value=1)
    organic_farming = forms.FloatField(label="Organic Farming", min_value=0, max_value=1)
    water_management = forms.FloatField(label="Water Management", min_value=0, max_value=1)

class ClimateImpactForm(forms.Form):
    avg_temp = forms.FloatField(label='Average Temperature (°C)', required=True)
    co2_emissions = forms.FloatField(label='CO2 Emissions (MT)', required=True)
    extreme_weather_0 = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], required=True)
    extreme_weather_1 = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], required=True)

class ClimateImpactLSTMForm(forms.Form):
    average_temperature = forms.FloatField(label='Average Temperature (°C)', required=True)
    total_precipitation = forms.FloatField(label='Total Precipitation (mm)', required=True)
    co2_emissions = forms.FloatField(label='CO2 Emissions (MT)', required=True)

    country = forms.ChoiceField(
        choices=[
            ('Argentina', 'Argentina'), ('Australia', 'Australia'), ('Brazil', 'Brazil'),
            ('Canada', 'Canada'), ('China', 'China'), ('France', 'France'),
            ('India', 'India'), ('Nigeria', 'Nigeria'), ('Russia', 'Russia'),
            ('USA', 'USA')
        ],
        label='Country', required=True
    )

    crop_type = forms.ChoiceField(
        choices=[
            ('Barley', 'Barley'), ('Coffee', 'Coffee'), ('Corn', 'Corn'),
            ('Cotton', 'Cotton'), ('Fruits', 'Fruits'), ('Rice', 'Rice'),
            ('Soybeans', 'Soybeans'), ('Sugarcane', 'Sugarcane'), ('Vegetables', 'Vegetables'),
            ('Wheat', 'Wheat')
        ],
        label='Crop Type', required=True
    )

    adaptation_strategy = forms.ChoiceField(
        choices=[
            ('Crop Rotation', 'Crop Rotation'), ('Drought-resistant Crops', 'Drought-resistant Crops'),
            ('No Adaptation', 'No Adaptation'), ('Organic Farming', 'Organic Farming'),
            ('Water Management', 'Water Management')
        ],
        label='Adaptation Strategy', required=True
    )