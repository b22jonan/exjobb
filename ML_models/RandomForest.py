import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import os
import subprocess
import sys
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import export_graphviz

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

    # Load the sampled student dataset instead of the full one
    data1 = pd.read_csv("CSV_files/Sampled_CodeStates.csv", header=0, names=["CodeStateID", "Code"])
    data1["label"] = 1  # Label for dataset A

    # Load second dataset (LLM generated) and use only Extracted_Code
    data2 = pd.read_csv("prompting/Qwen/processed_responses.csv", header=None, names=["ID", "Prompt", "Extracted_Code"])
    data2 = data2[["ID", "Prompt", "Extracted_Code"]].rename(columns={"Extracted_Code": "Code"})
    data2["label"] = 0  # Label for dataset B

    # Combine datasets
    data = pd.concat([data1, data2], ignore_index=True)

    # Convert code to features using TF-IDF
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4,6), max_features=1000)
    X = vectorizer.fit_transform(data["Code"])
    y = data["label"]

    # Train-test split
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, data.index, test_size=0.3, random_state=state)
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=state
    )
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
    misclassified_indices = np.where(y_pred != y_test.values)[0]
    misclassified_cases = data.iloc[indices_test[misclassified_indices]].copy()
    misclassified_cases["Predicted Label"] = y_pred[misclassified_indices]


    # Save misclassified cases separately per iteration
    misclassified_A = misclassified_cases[misclassified_cases["label"] == 1].copy()
    misclassified_B = misclassified_cases[misclassified_cases["label"] == 0].copy()

    if misclassified_B.empty:
        print(f"Iteration {i+1}: No misclassified LLM cases found.")
    else:
        print(f"Iteration {i+1}: {len(misclassified_B)} misclassified LLM cases saved.")
        misclassified_B = misclassified_B[["ID", "Prompt", "Code", "label", "Predicted Label"]]
        misclassified_B.to_csv(f"ML_models/results/RandomForest_Qwen/misclassified_LLM_iter_{i+1}.csv", index=False)

    misclassified_A.to_csv(f"ML_models/results/RandomForest_Qwen/misclassified_Student_iter_{i+1}.csv", index=False)

# Save all iteration results
results_df.to_csv("ML_models/results/RandomForest_Qwen/all_iterations.csv", index=False)

# Compute mean and standard deviation
accuracy_mean, accuracy_std = results_df["Accuracy"].mean(), results_df["Accuracy"].std()
precision_mean, precision_std = results_df["Precision"].mean(), results_df["Precision"].std()
recall_mean, recall_std = results_df["Recall"].mean(), results_df["Recall"].std()
f1_mean, f1_std = results_df["F1 Score"].mean(), results_df["F1 Score"].std()

# Save evaluation metrics
with open("ML_models/results/RandomForest_Qwen/results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_mean:.4f} (±{accuracy_std:.4f})\n")
    f.write(f"Precision: {precision_mean:.4f} (±{precision_std:.4f})\n")
    f.write(f"Recall: {recall_mean:.4f} (±{recall_std:.4f})\n")
    f.write(f"F1 Score: {f1_mean:.4f} (±{f1_std:.4f})\n")

print(f"Accuracy: {accuracy_mean:.4f} (±{accuracy_std:.4f})")
print(f"Precision: {precision_mean:.4f} (±{precision_std:.4f})")
print(f"Recall: {recall_mean:.4f} (±{recall_std:.4f})")
print(f"F1 Score: {f1_mean:.4f} (±{f1_std:.4f})")
