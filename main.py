import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 1. Data Loading and Exploration
print("Loading and exploring the dataset...")

# Reading the dataset - handling the custom format
# First read the data
with open('Static-Binary classification dataset.csv', 'r') as file:
    lines = file.readlines()

# Extract header - the first line contains all feature names separated by commas
header = lines[0].strip().split(',')

# Extract data rows
data_rows = []
for line in lines[1:]:
    values = line.strip().split(',')
    if len(values) >= len(header):  # Ensure the row has enough values
        data_rows.append(values)

# Create DataFrame
df = pd.DataFrame(data_rows, columns=header)

# Convert all columns except the last one to numeric
# for col in df.columns[:-1]:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# Rename the target column to 'Label' and convert to binary
df.rename(columns={df.columns[-1]: 'Label'}, inplace=True)
df['Label'] = df['Label'].map({'B': 0, 'M': 1})

# Display basic information
print("\nDataset Overview:")
print(f"- Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check class distribution
print("\nClass Distribution:")
print(df['Label'].value_counts())
print(f"Percentage of malware: {df['Label'].mean()*100:.2f}%")

# 2. Data Preprocessing
print("\n\nPerforming data preprocessing...")

# Separate features and target
X = df.drop('Label', axis=1)
y = df['Label']

# Check for features with high correlation
correlation_matrix = X.corr()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training and Evaluation
print("\n\nTraining and evaluating models...")

# Define the models with hyperparameters
models = {
    'Decision Tree': DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        min_samples_split=5
    ),
    'Random Forest': RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        algorithm='auto',
        leaf_size=30,
        p=2  # Euclidean distance
    )
}

# Evaluation metrics
results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

# Function to evaluate model performance
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Print results
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Add results to DataFrame
    return pd.DataFrame({
        'Model': [name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1]
    })

# Evaluate each model
for name, model in models.items():
    model_results = evaluate_model(name, model, X_train_scaled, X_test_scaled, y_train, y_test)
    results = pd.concat([results, model_results], ignore_index=True)


print("\nProject completed successfully!")