import pandas as pd
import numpy as np
import os
import subprocess
import sys
import random
import matplotlib.pyplot as plt
import graphviz
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add Graphviz to PATH for tree visualization (Windows, already installed) - Change path accordingly
os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"

# Run the dataset sampling script using the virtual environment Python
subprocess.run([sys.executable, "scripts/dataset_sampler.py"], check=True)

# Load first dataset (student dataset)
data1 = pd.read_csv("CSV_files/Sampled_CodeStates.csv", header=0, names=["CodeStateID", "Code"])
data1["label"] = 1  # Label for dataset A

# Load second dataset (LLM generated) and use only Extracted_Code
data2 = pd.read_csv("prompting/Qwen/processed_responses.csv", header=None, names=["ID", "Prompt", "Extracted_Code"])
data2 = data2[["ID", "Prompt", "Extracted_Code"]].rename(columns={"Extracted_Code": "Code"})
data2["label"] = 0  # Label for dataset B

# Combine datasets
data = pd.concat([data1, data2], ignore_index=True)

# Convert the 'Code' column (text data) into TF-IDF features
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 6), max_features=1000)
X = vectorizer.fit_transform(data["Code"])
y = data["label"]

# Perform multiple train-test splits with randomized random states
num_iterations = 5
random_states = random.sample(range(100, 10000), num_iterations)  # Generate random states dynamically
accuracy_list, precision_list, recall_list, f1_list = [], [], [], []

final_model = None  # Store the final trained model

for state in random_states:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=state)
    
    # Train a LightGBM model
    model = lgb.LGBMClassifier(learning_rate=0.05, num_leaves=31, random_state=state)
    model.fit(X_train, y_train)
    final_model = model  # Store the last trained model
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred))
    recall_list.append(recall_score(y_test, y_pred))
    f1_list.append(f1_score(y_test, y_pred))

# Compute mean and standard deviation
accuracy_mean, accuracy_std = np.mean(accuracy_list), np.std(accuracy_list)
precision_mean, precision_std = np.mean(precision_list), np.std(precision_list)
recall_mean, recall_std = np.mean(recall_list), np.std(recall_list)
f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)

# Save evaluation metrics
with open("ML_models/results/LightGBM_Qwen/results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_mean:.4f} (±{accuracy_std:.4f})\n")
    f.write(f"Precision: {precision_mean:.4f} (±{precision_std:.4f})\n")
    f.write(f"Recall: {recall_mean:.4f} (±{recall_std:.4f})\n")
    f.write(f"F1 Score: {f1_mean:.4f} (±{f1_std:.4f})\n")

print(f"Accuracy: {accuracy_mean:.4f} (±{accuracy_std:.4f})")
print(f"Precision: {precision_mean:.4f} (±{precision_std:.4f})")
print(f"Recall: {recall_mean:.4f} (±{recall_std:.4f})")
print(f"F1 Score: {f1_mean:.4f} (±{f1_std:.4f})")

# Identify misclassified cases
misclassified_indices = np.where(final_model.predict(X_test) != y_test)[0]
misclassified_cases = data.iloc[misclassified_indices].copy()
misclassified_cases["Predicted Label"] = final_model.predict(X_test)[misclassified_indices]

# Save misclassified cases separately
misclassified_A = misclassified_cases[misclassified_cases["label"] == 1]
misclassified_B = misclassified_cases[misclassified_cases["label"] == 0]

misclassified_A.to_csv("ML_models/results/LightGBM_Qwen/Student.csv", index=False)
misclassified_B.to_csv("ML_models/results/LightGBM_Qwen/LLM.csv", index=False, columns=["ID", "Prompt", "Code", "label", "Predicted Label"])

# Visualize the LightGBM tree
lgb.plot_tree(final_model, tree_index=0, figsize=(20, 10), show_info=['split_gain', 'internal_value', 'leaf_count'])
plt.title("LightGBM Decision Tree")
plt.savefig("ML_models/results/LightGBM_Qwen/tree_visualization.png")
plt.show()

# Plot feature importance
lgb.plot_importance(final_model, max_num_features=20, figsize=(10, 6))
plt.title("Feature Importance in LightGBM")
plt.savefig("ML_models/results/LightGBM_Qwen/feature_importance.png")
plt.show()
