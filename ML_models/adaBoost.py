import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

# Load first dataset (student dataset)
data1 = pd.read_csv("CSV_files/CodeStates.csv", header=None, names=["CodeStateID", "Code"])
data1["label"] = 1  # Label for dataset A

# Load second dataset (LLM generated) and use only Extracted_Code
data2 = pd.read_csv("prompting/ChatGPT/processed_responses.csv", header=None, names=["ID", "Prompt", "Extracted_Code"])
data2 = data2[["ID", "Prompt", "Extracted_Code"]].rename(columns={"Extracted_Code": "Code"})
data2["label"] = 0  # Label for dataset B

# Combine datasets
data = pd.concat([data1, data2], ignore_index=True)

# Convert code to features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data["Code"]).toarray()
y = data["label"].values

# Handle missing values by replacing NaNs with the median value of each column
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y, data.index, test_size=0.2, random_state=42
)

# Build the AdaBoost model using a decision tree as the base estimator
base_model = DecisionTreeClassifier(max_depth=3)
model = AdaBoostClassifier(base_model, n_estimators=50, learning_rate=1)

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save evaluation metrics
with open("ML_models/results/AdaBoost_ChatGPT35/results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Identify misclassified cases
misclassified_indices = indices_test[y_pred != y_test]
misclassified_cases = data.loc[misclassified_indices]
misclassified_cases["Predicted Label"] = y_pred[y_pred != y_test]  # Store predicted labels

# Save misclassified cases separately
misclassified_A = misclassified_cases[misclassified_cases["label"] == 1]  # From dataset A
misclassified_B = misclassified_cases[misclassified_cases["label"] == 0]  # From dataset B

misclassified_A.to_csv("ML_models/results/AdaBoost_ChatGPT35/Student.csv", index=False)
misclassified_B.to_csv("ML_models/results/AdaBoost_ChatGPT35/LLM.csv", index=False, columns=["ID", "Prompt", "Code", "label", "Predicted Label"])
