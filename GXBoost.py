import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import time


# Step 1: Load Data from CSV Files
x_data_path = 'MicroDataSets/MicroXData.csv'
y_data_path = 'MicroDataSets/MicroYData.csv'

# Read the datasets
x_data = pd.read_csv(x_data_path, header=None, names=["CodeStateID", "Code"])
y_data = pd.read_csv(y_data_path, header=None, names=["CodeStateID", "Code"])

# Step 2: Concatenate the Data into a Single DataFrame
# Create labels (X Data is label 0, Y Data is label 1)
x_data['label'] = 0
y_data['label'] = 1

# Combine both datasets
data = pd.concat([x_data, y_data], ignore_index=True)

# Step 3: Feature Extraction (Use TF-IDF to vectorize the function code)
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust the number of features as needed
X = vectorizer.fit_transform(data['Code']).toarray()

# Labels
y = data['label']

# Step 4: Split the data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training")
start_time = time.time()
# Step 5: Train XGBoost Model
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)
end_time = time.time()

# Printing out training time
elapsed_time = end_time - start_time
print(f"training time: {elapsed_time:.6f} seconds")

# Step 6: Predict and Evaluate the Model
y_pred = model.predict(X_test)

# Accuracy
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
