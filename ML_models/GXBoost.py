import pandas as pd 
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import time
import numpy as np
import subprocess
import sys

# File paths
x_data_path = 'prompting\Qwen\processed_responses.csv'
failed_x_path = 'ML_models/results/XGBoost_Qwen/LLM.csv'
failed_y_path = 'ML_models/results/XGBoost_Qwen/Student.csv'
metrics_path = 'ML_models/results/XGBoost_Qwen/metrics.txt'

# Initialize lists to store metrics and failed samples
accuracies = []
precisions = []
recalls = []
f1_scores = []
failed_x_all = pd.DataFrame()
failed_y_all = pd.DataFrame()

# Number of iterations for training
iterations = 50

# Start training loop
with open(metrics_path, 'w') as f:
    for i in range(iterations):
        print(f"Training iteration {i + 1}...")
        
        # Generate new dataset for each iteration
        os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin" 
        subprocess.run([sys.executable, "scripts/dataset_sampler.py"], check=True)

        # Load first dataset (LLM dataset)
        x_data = pd.read_csv(x_data_path, header=0, names=["CodeStateID", "Prompt", "Code"])
        
        # Load second dataset (student dataset)
        y_data = pd.read_csv("CSV_files\Sampled_CodeStates.csv", header=None, names=["CodeStateID", "Code"])
        
        # Add labels (0 for X, 1 for Y)
        x_data['label'] = 0
        y_data['label'] = 1

        # Store ExtraField separately and drop from y_data for training
        x_extra_field = x_data[['CodeStateID', 'Prompt']]
        x_data = x_data.drop(columns=['Prompt'])

        # Combine datasets
        data = pd.concat([x_data, y_data], ignore_index=True)

        # Feature extraction using TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(data['Code']).toarray()
        y = data['label']

        random_state = np.random.randint(0, 10000)  # Randomly generate random_state for each iteration
        
        # Split the data with a new random_state each time
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        model = xgb.XGBClassifier(eval_metric='logloss', random_state=random_state, max_depth=5)
        
        # Train XGBoost model
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        print(f"Training time for iteration {i + 1}: {end_time - start_time:.6f} seconds")

        # Make predictions
        y_pred = model.predict(X_test)

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Store metrics
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # Save individual iteration metrics
        f.write(f'Iteration{i + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}\n')

        # Identify misclassified entries
        misclassified = data.iloc[y_test.index][y_test != y_pred]

        # Separate misclassified entries for X and Y datasets
        failed_x = misclassified[misclassified['label'] == 0].drop(columns=['label'])
        failed_y = misclassified[misclassified['label'] == 1].drop(columns=['label'])

        # Merge back the extra field for dataset Y and X
        failed_x = failed_x.merge(x_extra_field, on="CodeStateID", how="left")

        # Append the misclassified data to the cumulative list
        failed_x_all = pd.concat([failed_x_all, failed_x], ignore_index=True)
        failed_y_all = pd.concat([failed_y_all, failed_y], ignore_index=True)

    # Calculate mean and standard deviation for each metric
    accuracy_mean = np.mean(accuracies)
    accuracy_std = np.std(accuracies)

    precision_mean = np.mean(precisions)
    precision_std = np.std(precisions)

    recall_mean = np.mean(recalls)
    recall_std = np.std(recalls)

    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)

    # Save average and standard deviation metrics
    f.write(f'\nOverall Metrics:\n')
    f.write(f'Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}\n')
    f.write(f'Precision: {precision_mean:.4f} ± {precision_std:.4f}\n')
    f.write(f'Recall: {recall_mean:.4f} ± {recall_std:.4f}\n')
    f.write(f'F1-score: {f1_mean:.4f} ± {f1_std:.4f}\n')

# Save misclassified entries to CSV
failed_x_all.to_csv(failed_x_path, index=False)
failed_y_all.to_csv(failed_y_path, index=False)

print(f'Results saved: {metrics_path}, {failed_x_path}, {failed_y_path}')