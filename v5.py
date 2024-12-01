import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load the dataset
file_path = 'eda_data.csv'  # Adjust the path to your local file if necessary
data = pd.read_csv(file_path)

# Create a function to generate and store visualizations

sns.set(style="whitegrid")

# 1. Distribution of Total Climate Impact
plt.figure(figsize=(8, 6))
sns.histplot(data['Total_Climate_Impact'], bins=30, kde=True, color='blue')
plt.title('Distribution of Total Climate Impact')
plt.xlabel('Total Climate Impact')
plt.ylabel('Frequency')
plt.savefig('5_Distribution_of_Total_Climate_Impact.png')
plt.show()

# 2. Correlation heatmap between variables (excluding the target)
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig('5_Correlation_Heatmap.png')
plt.show()

# 3. Pie chart of Extreme Weather Events count
extreme_weather_counts = data[['Extreme_Weather_Events_0', 'Extreme_Weather_Events_1',
                                'Extreme_Weather_Events_2', 'Extreme_Weather_Events_3',
                                'Extreme_Weather_Events_4', 'Extreme_Weather_Events_5',
                                'Extreme_Weather_Events_6', 'Extreme_Weather_Events_7',
                                'Extreme_Weather_Events_8', 'Extreme_Weather_Events_9']].sum(axis=0)
plt.figure(figsize=(8, 6))
extreme_weather_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='Set3')
plt.title('Distribution of Extreme Weather Events')
plt.savefig('5_Distribution_of_Extreme_Weather_Events.png')
plt.show()

# 4. Boxplot of Total Climate Impact by Adaptation Strategy
plt.figure(figsize=(8, 6))
sns.boxplot(x='Adaptation_Strategies_Crop Rotation', y='Total_Climate_Impact', data=data, hue='Adaptation_Strategies_Crop Rotation', palette='Set2', legend=False)
plt.title('Boxplot of Total Climate Impact by Crop Rotation Adaptation Strategy')
plt.savefig('5_Boxplot_of_Total_Climate_Impact_by_Crop_Rotation_Adaptation_Strategy.png')
plt.show()

# 5. Boxplot of Total Climate Impact by Region
plt.figure(figsize=(12, 8))
sns.violinplot(x='Region_British Columbia', y='Total_Climate_Impact', data=data, hue='Region_British Columbia',  palette='viridis', legend=False)
plt.title('Boxplot of Total Climate Impact by Region: Eg: British Columbia')
plt.xlabel('Region')
plt.ylabel('Total Climate Impact')
plt.savefig('5_Violinplot_of_Total_Climate_Impact_by_Region.png')
plt.show()

# Store the plots in a dictionary
visualizations = {
    'Distribution_of_Total_Climate_Impact': 'images/5_Distribution_of_Total_Climate_Impact.png',
    'Correlation_Heatmap': 'images/5_Correlation_Heatmap.png',
    'Distribution_of_Extreme_Weather_Events': 'images/5_Distribution_of_Extreme_Weather_Events.png',
    'Boxplot_of_Total_Climate_Impact_by_Crop_Rotation_Adaptation_Strategy': 'images/5_Boxplot_of_Total_Climate_Impact_by_Crop_Rotation_Adaptation_Strategy.png',
    'Violinplot_of_Total_Climate_Impact_by_Region': 'images/5_Violinplot_of_Total_Climate_Impact_by_Region.png'
}

# Save the visualizations to a pickle file
with open('v5.pkl', 'wb') as f:
    pickle.dump(visualizations, f)
    print("Visualizations saved to v5.pkl pickle file.")
