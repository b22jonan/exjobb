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

# Run the dataset sampling script using the virtual environment Python
subprocess.run([sys.executable, "scripts/dataset_sampler.py"], check=True)

# Load the sampled student dataset instead of the full one
data1 = pd.read_csv("CSV_files/Sampled_CodeStates.csv", header=0, names=["CodeStateID", "Code"])
data1["label"] = 1  # Label for dataset A

# Load second dataset (llm generated) and use only Extracted_Code
data2 = pd.read_csv("prompting/ChatGPT/processed_responses.csv", header=None, names=["ID", "Prompt", "Extracted_Code"])
data2 = data2[["ID", "Prompt", "Extracted_Code"]].rename(columns={"Extracted_Code": "Code"})
data2["label"] = 0  # Label for dataset B

# Combine datasets
data = pd.concat([data1, data2], ignore_index=True)

# Convert code to features using TF-IDF
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4,6), max_features=1000)
X = vectorizer.fit_transform(data["Code"])
y = data["label"]

# Perform multiple train-test splits with randomized random states
num_iterations = 5
random_states = random.sample(range(100, 10000), num_iterations)  # Generate random states dynamically
accuracy_list, precision_list, recall_list, f1_list = [], [], [], []

final_model = None  # Store the final trained model

for state in random_states:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=state)
    
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
with open("ML_models/results/RandomForest_ChatGPT35/results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_mean:.4f} (±{accuracy_std:.4f})\n")
    f.write(f"Precision: {precision_mean:.4f} (±{precision_std:.4f})\n")
    f.write(f"Recall: {recall_mean:.4f} (±{recall_std:.4f})\n")
    f.write(f"F1 Score: {f1_mean:.4f} (±{f1_std:.4f})\n")

print(f"Accuracy: {accuracy_mean:.4f} (±{accuracy_std:.4f})")
print(f"Precision: {precision_mean:.4f} (±{precision_std:.4f})")
print(f"Recall: {recall_mean:.4f} (±{recall_std:.4f})")
print(f"F1 Score: {f1_mean:.4f} (±{f1_std:.4f})")

# Clean feature names for Graphviz
clean_feature_names = [name.replace(";", "").replace("\"", "").replace("'", "").replace("\\", "").replace("\n", " ").replace("\r", " ") for name in vectorizer.get_feature_names_out()]

# Visualize an individual decision tree
if final_model and hasattr(final_model, "estimators_") and len(final_model.estimators_) > 0:
    tree = final_model.estimators_[0]  # Take the first tree from the final trained model
    dot_data = export_graphviz(
        tree,
        feature_names=clean_feature_names,  # Use sanitized names
        class_names=["LLM", "Student"],
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=7  # Limit tree depth for better visualization
    )
    graph = graphviz.Source(dot_data)
    graph.render("ML_models/results/RandomForest_ChatGPT35/tree_visualization")
else:
    print("Warning: No trees found in the final trained model.")

# Plot feature importance
importances = final_model.feature_importances_
feature_names = vectorizer.get_feature_names_out()
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importance in Random Forest")
plt.bar(range(20), importances[indices][:20], align="center")
plt.xticks(range(20), np.array(feature_names)[indices][:20], rotation=90)
plt.xlabel("Feature (Character n-grams)")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("ML_models/results/RandomForest_ChatGPT35/feature_importance.png")
plt.show()

# Print tree depth and number of leaves for first three trees
for i, tree in enumerate(final_model.estimators_[:3]):
    print(f"Tree {i}: depth={tree.get_depth()}, leaves={tree.get_n_leaves()}")