import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import export_graphviz

# Add Graphviz to PATH for tree visualization (Windows, already installed) - Change path accordingly
os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"

# Load first dataset (student dataset)
data1 = pd.read_csv("CSV_files/CodeStates.csv", header=None, names=["CodeStateID", "Code"])
data1["label"] = 1  # Label for dataset A

# Load second dataset (llm generated) and use only Extracted_Code
data2 = pd.read_csv("prompting/ChatGPT/processed_responses.csv", header=None, names=["ID", "Prompt", "Extracted_Code"])
data2 = data2[["ID", "Prompt", "Extracted_Code"]].rename(columns={"Extracted_Code": "Code"})
data2["label"] = 0  # Label for dataset B

# Combine datasets
data = pd.concat([data1, data2], ignore_index=True)

# Convert code to features using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data["Code"])
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y, data.index, test_size=0.3, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

# Compare training vs test accuracy
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save evaluation metrics
with open("ML_models/results/RandomForest_ChatGPT35/results.txt", "w") as f:
    f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Cross-Validation Mean Accuracy: {cv_mean:.4f} (±{cv_std:.4f})\n")
    f.write(f"Final Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Cross-Validation Mean Accuracy: {cv_mean:.4f} (±{cv_std:.4f})")
print(f"Final Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Identify misclassified cases
misclassified_indices = indices_test[y_pred != y_test.values]
misclassified_cases = data.loc[misclassified_indices]
misclassified_cases["Predicted Label"] = y_pred[y_pred != y_test.values]  # Store predicted labels

# Save misclassified cases separately
misclassified_A = misclassified_cases[misclassified_cases["label"] == 1]  # From dataset A
misclassified_B = misclassified_cases[misclassified_cases["label"] == 0]  # From dataset B

misclassified_A.to_csv("ML_models/results/RandomForest_ChatGPT35/Student.csv", index=False)
misclassified_B.to_csv("ML_models/results/RandomForest_ChatGPT35/LLM.csv", index=False, columns=["ID", "Prompt", "Code", "label", "Predicted Label"])

# Visualize an individual decision tree
tree = model.estimators_[0]
dot_data = export_graphviz(
    tree,
    feature_names=vectorizer.get_feature_names_out(),
    class_names=["LLM", "Student"],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("ML_models/results/RandomForest_ChatGPT35/tree_visualization")

# Plot feature importance
importances = model.feature_importances_
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
for i, tree in enumerate(model.estimators_[:3]):
    print(f"Tree {i}: depth={tree.get_depth()}, leaves={tree.get_n_leaves()}")