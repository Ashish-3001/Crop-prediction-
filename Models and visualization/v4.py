import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset
file_path = 'eda_data.csv'  # Adjust the path to your dataset file
data = pd.read_csv(file_path)

# Data Preprocessing: Dropping non-numeric columns and handling missing values if necessary
# For the sake of visualization, we'll focus on numeric columns related to climate impact
numeric_columns = [
    'Average_Temperature_C', 'CO2_Emissions_MT', 'Total_Climate_Impact', 'Extreme_Weather_Events_0',
    'Extreme_Weather_Events_1', 'Crop_Yield_MT_per_HA', 'Economic_Impact_Million_USD'
]

# Visualize the distribution of Total Climate Impact
plt.figure(figsize=(10, 6))
sns.histplot(data['Total_Climate_Impact'], kde=True, bins=30, color='blue')
plt.title('Distribution of Total Climate Impact')
plt.xlabel('Total Climate Impact')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('climate_impact_distribution.png')
plt.show()

# Visualize the relationship between Average Temperature and Total Climate Impact
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average_Temperature_C', y='Total_Climate_Impact', data=data, color='red')
plt.title('Average Temperature vs. Total Climate Impact')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Total Climate Impact')
plt.tight_layout()
plt.savefig('temperature_vs_climate_impact.png')
plt.show()

# Correlation heatmap between relevant numeric columns
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Climate Impact-related Variables')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()


# Visualize the boxplot for economic impact and climate impact
plt.figure(figsize=(10, 6))
sns.boxplot(x='Economic_Impact_Million_USD', y='Total_Climate_Impact', data=data, hue='Economic_Impact_Million_USD', palette='Blues', legend=False)
plt.title('Economic Impact vs. Total Climate Impact (Boxplot)')
plt.xlabel('Economic Impact (Million USD)')
plt.ylabel('Total Climate Impact')
plt.tight_layout()
plt.savefig('economic_impact_vs_climate_impact_boxplot.png')
plt.show()

visualizations = {
    'climate_impact_distribution': 'images/4_climate_impact_distribution.png',
    'temperature_vs_climate_impact': 'images/4_temperature_vs_climate_impact.png',
    'correlation_heatmap': 'images/4_correlation_heatmap.png',
    'economic_impact_vs_climate_impact_boxplot': 'images/4_economic_impact_vs_climate_impact_boxplot.png'
}

# Save the boxplot as a pickle file for later use
with open('v4.pkl', 'wb') as f:
    pickle.dump(visualizations, f)
