import pandas as pd
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

# Preprocess features (imputation and scaling)
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler())  # Standardize numerical features
])

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
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Visualization 1: Plot training and validation accuracy over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_plot.png')  # Save the plot
plt.show()

# Visualize model weights for feature importance (not perfect for neural nets, but illustrative)
weights = model.layers[0].get_weights()[0]  # Weights of the first layer (input layer)
plt.figure(figsize=(10, 6))
sns.barplot(x=adaptation_columns, y=np.abs(weights.mean(axis=1)))
plt.title('Feature Importance (Input Layer Weights)')
plt.xlabel('Adaptation Strategy')
plt.ylabel('Mean Absolute Weight')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance.png')  # Save the plot
plt.show()

# Save the feature importance plot as a pickle file
visualizations = {
    'accuracy_plot': 'images/accuracy_plot.png',
    'feature_importance': 'images/feature_importance.png'
}

# Save the pickle file with paths to visualizations
with open('v3.pkl', 'wb') as f:
    pickle.dump(visualizations, f)
    print(f"Visualizations are now saved to v3.pkl pickle file.")