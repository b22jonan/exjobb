import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import time

# File paths
x_data_path = 'MicroDataSets/MicroXData.csv'
y_data_path = 'MicroDataSets/MicroYData.csv'
failed_x_path = 'ML_models/results/XGBoost_ChatGPT4o/Student.csv'
failed_y_path = 'ML_models/results/XGBoost_ChatGPT4o/LLM.csv'
metrics_path = 'ML_models/results/XGBoost_ChatGPT4o/metrics.txt'

# Load datasets
x_data = pd.read_csv(x_data_path, header=None, names=["CodeStateID", "Code"])
y_data = pd.read_csv(y_data_path, header=None, names=["CodeStateID", "Code", "Prompt"])

# Add labels (0 for X, 1 for Y)
x_data['label'] = 0
y_data['label'] = 1

# Store ExtraField separately and drop from y_data for training
y_extra_field = y_data[['CodeStateID', 'Prompt']]
y_data = y_data.drop(columns=['Prompt'])

# Combine datasets
data = pd.concat([x_data, y_data], ignore_index=True)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['Code']).toarray()
y = data['label']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training...")
start_time = time.time()

# Train XGBoost model
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

end_time = time.time()
print(f"Training time: {end_time - start_time:.6f} seconds")

# Make predictions
y_pred = model.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save metrics to a text file
with open(metrics_path, 'w') as f:
    f.write(f'Accuracy: {accuracy:.4f}\n')
    f.write(f'Precision: {precision:.4f}\n')
    f.write(f'Recall: {recall:.4f}\n')
    f.write(f'F1-score: {f1:.4f}\n')

# Identify misclassified entries
misclassified = data.iloc[y_test.index][y_test != y_pred]

# Separate misclassified entries for X and Y datasets
failed_x = misclassified[misclassified['label'] == 0].drop(columns=['label'])
failed_y = misclassified[misclassified['label'] == 1].drop(columns=['label'])

# Merge back the extra field for dataset Y
failed_y = failed_y.merge(y_extra_field, on="CodeStateID", how="left")

# Save misclassified entries to CSV
failed_x.to_csv(failed_x_path, index=False)
failed_y.to_csv(failed_y_path, index=False)

print(f'Results saved: {metrics_path}, {failed_x_path}, {failed_y_path}')