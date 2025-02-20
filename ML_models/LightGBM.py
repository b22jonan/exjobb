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

# Perform multiple train-test splits with randomized random states
num_iterations = 50
random_states = random.sample(list(range(100, 10000)), num_iterations)  # Generate unique random states

results_df = pd.DataFrame(columns=["Iteration", "Accuracy", "Precision", "Recall", "F1 Score"])
final_model = None  # Store the final trained model

for i, state in enumerate(random_states):
    # Run the dataset sampling script using the virtual environment Python
    subprocess.run([sys.executable, "scripts/dataset_sampler.py"], check=True)

    # Load first dataset (student dataset)
    data1 = pd.read_csv("CSV_files/Sampled_CodeStates.csv", header=0, names=["CodeStateID", "Code"])
    data1["label"] = 1  # Label for dataset A

    # Load second dataset (LLM generated) and use only Extracted_Code
    data2 = pd.read_csv("prompting/ChatGPT4o/processed_responses.csv", header=None, names=["ID", "Prompt", "Extracted_Code"])
    data2 = data2[["ID", "Prompt", "Extracted_Code"]].rename(columns={"Extracted_Code": "Code"})
    data2["label"] = 0  # Label for dataset B

    # Combine datasets
    data = pd.concat([data1, data2], ignore_index=True)

    # Convert code to features using TF-IDF
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4,6), max_features=1000)
    X = vectorizer.fit_transform(data["Code"])
    y = data["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=state)
    
    # Train a LightGBM model
    model = lgb.LGBMClassifier(learning_rate=0.05, num_leaves=31, random_state=state)
    model.fit(X_train, y_train)
    final_model = model  # Store the last trained model
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Save iteration results
    results_df.loc[i] = [i+1, accuracy, precision, recall, f1]
    
    # Identify misclassified cases
    misclassified_indices = np.where(y_pred != y_test)[0]
    misclassified_cases = data.iloc[misclassified_indices].copy()
    misclassified_cases["Predicted Label"] = y_pred[misclassified_indices]

    # Save misclassified cases separately per iteration
    misclassified_A = misclassified_cases[misclassified_cases["label"] == 1]
    misclassified_B = misclassified_cases[misclassified_cases["label"] == 0]

    misclassified_A.to_csv(f"ML_models/results/LightGBM_ChatGPT4o/misclassified_Student_iter_{i+1}.csv", index=False)
    misclassified_B.to_csv(f"ML_models/results/LightGBM_ChatGPT4o/misclassified_LLM_iter_{i+1}.csv", index=False, columns=["ID", "Prompt", "Code", "label", "Predicted Label"])

# Save all iteration results
results_df.to_csv("ML_models/results/LightGBM_ChatGPT4o/all_iterations.csv", index=False)

# Compute mean and standard deviation
accuracy_mean, accuracy_std = results_df["Accuracy"].mean(), results_df["Accuracy"].std()
precision_mean, precision_std = results_df["Precision"].mean(), results_df["Precision"].std()
recall_mean, recall_std = results_df["Recall"].mean(), results_df["Recall"].std()
f1_mean, f1_std = results_df["F1 Score"].mean(), results_df["F1 Score"].std()

# Save evaluation metrics
with open("ML_models/results/LightGBM_ChatGPT4o/results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_mean:.4f} (±{accuracy_std:.4f})\n")
    f.write(f"Precision: {precision_mean:.4f} (±{precision_std:.4f})\n")
    f.write(f"Recall: {recall_mean:.4f} (±{recall_std:.4f})\n")
    f.write(f"F1 Score: {f1_mean:.4f} (±{f1_std:.4f})\n")

print(f"Accuracy: {accuracy_mean:.4f} (±{accuracy_std:.4f})")
print(f"Precision: {precision_mean:.4f} (±{precision_std:.4f})")
print(f"Recall: {recall_mean:.4f} (±{recall_std:.4f})")
print(f"F1 Score: {f1_mean:.4f} (±{f1_std:.4f})")

# Visualize the LightGBM tree
lgb.plot_tree(final_model, tree_index=0, figsize=(20, 10), show_info=['split_gain', 'internal_value', 'leaf_count'])
plt.title("LightGBM Decision Tree")
plt.savefig("ML_models/results/LightGBM_ChatGPT4o/tree_visualization.png")
plt.show()

# Plot feature importance
lgb.plot_importance(final_model, max_num_features=20, figsize=(10, 6))
plt.title("Feature Importance in LightGBM")
plt.savefig("ML_models/results/LightGBM_ChatGPT4o/feature_importance.png")
plt.show()
