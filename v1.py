import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load the preprocessed dataset (eda_data.csv)
file_path = 'eda_data.csv'  # Update with the correct path
dataset = pd.read_csv(file_path)

# Set the style for the plots
sns.set(style="whitegrid")

# Visualize the standardized features (e.g., standardized crop yield and economic impact)
plt.figure(figsize=(10, 6))
sns.histplot(dataset['Crop_Yield_MT_per_HA_Standardized'], kde=True, color='orange', bins=20)
plt.title('Standardized Crop Yield Distribution')
plt.xlabel('Standardized Crop Yield')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('standardized_crop_yield_distribution.png')
plt.show()

# Visualize the relationship between standardized crop yield and economic impact
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Economic_Impact_Million_USD_Standardized', y='Crop_Yield_MT_per_HA_Standardized', data=dataset, color='red')
plt.title('Standardized Crop Yield vs Economic Impact')
plt.xlabel('Standardized Economic Impact')
plt.ylabel('Standardized Crop Yield')
plt.tight_layout()
plt.savefig('standardized_crop_yield_vs_economic_impact.png')
plt.show()

# Correlation heatmap of the standardized features
plt.figure(figsize=(12, 8))
standardized_features = dataset[['Average_Temperature_C_Standardized', 'Total_Precipitation_mm_Standardized', 
                                 'CO2_Emissions_MT_Standardized', 'Crop_Yield_MT_per_HA_Standardized', 
                                 'Economic_Impact_Million_USD_Standardized']]
correlation_matrix_standardized = standardized_features.corr()
sns.heatmap(correlation_matrix_standardized, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Standardized Features')
plt.tight_layout()
plt.savefig('standardized_features_correlation_heatmap.png')
plt.show()

# Store the visualization file paths in a pickle file for the frontend
visualizations = {
    'standardized_crop_yield_distribution': 'images/standardized_crop_yield_distribution.png',
    'standardized_crop_yield_vs_economic_impact': 'images/standardized_crop_yield_vs_economic_impact.png',
    'standardized_features_correlation_heatmap': 'images/standardized_features_correlation_heatmap.png'
}


# Save the pickle file with paths to visualizations
with open('v1.pkl', 'wb') as f:
    pickle.dump(visualizations, f)

print("Preprocessed visualizations saved and pickle file created.")
