# visualization_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset
file_path = 'eda_data.csv'  # Adjust the path if necessary
dataset = pd.read_csv(file_path)


# Extract features and target variable
X = dataset[['Average_Temperature_C', 'CO2_Emissions_MT']]  # Features
y = dataset['Economic_Impact_Million_USD']  # Target

# Prepare data for frontend use
frontend_data = {
    'features': ['Average_Temperature_C', 'CO2_Emissions_MT'],
    'mse': None,  # Will be calculated later if needed for metrics visualization
    'r2': None,  # Will be calculated later if needed for metrics visualization
    'plot_settings': {
        'feature_importance': {
            'title': 'Feature Importance for Economic Impact Prediction',
            'xlabel': 'Features',
            'ylabel': 'Importance'
        },
        'correlation_heatmap': {
            'title': 'Correlation Heatmap between Features and Economic Impact',
            'xlabel': 'Features',
            'ylabel': 'Economic Impact'
        },
        'actual_vs_predicted': {
            'title': 'Actual vs Predicted Economic Impact',
            'xlabel': 'Actual Economic Impact (Million USD)',
            'ylabel': 'Predicted Economic Impact (Million USD)'
        },
        'error_distribution': {
            'title': 'Distribution of Prediction Errors',
            'xlabel': 'Error (Actual - Predicted)',
            'ylabel': 'Frequency'
        }
    }
}

# Generate a correlation heatmap
correlation_matrix = dataset[['Average_Temperature_C', 'CO2_Emissions_MT', 'Economic_Impact_Million_USD']].corr()
frontend_data['correlation_matrix'] = correlation_matrix

# Generate Actual vs Predicted Plot (we'll use the same data here for simplicity)
y_pred = y  # Since we aren't using a model here, we'll assume actual == predicted for now

# Calculate the errors (we'll use this for the error distribution)
errors = y - y_pred
frontend_data['errors'] = errors


# Optionally, generate and save plots (for frontend display as images)

# 1. Feature Importance (Bar Plot)
plt.figure(figsize=(8, 6))
sns.barplot(x=frontend_data['features'], y=[0.6, 0.4])  # Simulate some feature importance values
plt.title(frontend_data['plot_settings']['feature_importance']['title'])
plt.xlabel(frontend_data['plot_settings']['feature_importance']['xlabel'])
plt.ylabel(frontend_data['plot_settings']['feature_importance']['ylabel'])
plt.tight_layout()
plt.savefig('feature_importance_plot.png')
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, fmt='.2f', linewidths=0.5)
plt.title(frontend_data['plot_settings']['correlation_heatmap']['title'])
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# 3. Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y, y=y_pred)
plt.plot([min(y), max(y)], [min(y), max(y)], linestyle='--', color='red')
plt.title(frontend_data['plot_settings']['actual_vs_predicted']['title'])
plt.xlabel(frontend_data['plot_settings']['actual_vs_predicted']['xlabel'])
plt.ylabel(frontend_data['plot_settings']['actual_vs_predicted']['ylabel'])
plt.tight_layout()
plt.savefig('actual_vs_predicted_plot.png')
plt.show()


# Store the visualization file paths in a pickle file for the frontend
visualizations = {
    'feature_importance_plot': 'images/feature_importance_plot.png',
    'correlation_heatmap': 'images/correlation_heatmap.png',
    'actual_vs_predicted_plot': 'images/actual_vs_predicted_plot.png'
}

# Save the pickle file with paths to visualizations
with open('v2.pkl', 'wb') as f:
    pickle.dump(visualizations, f)  
