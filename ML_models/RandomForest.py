import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load first dataset (MicroXData)
data1 = pd.read_csv("CSV files/MicroXData.csv", header=None, names=["CodeStateID", "Code"])
data1["label"] = 1  # Label for dataset A

# Load second dataset (MicroYData) and use only Extracted_Code
data2 = pd.read_csv("CSV files/MicroYData.csv", header=None, names=["ID", "Prompt", "Extracted_Code"])
data2 = data2[["ID", "Extracted_Code"]].rename(columns={"Extracted_Code": "Code"})
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
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save evaluation metrics
with open("ML models/results/model_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

print(f"Accuracy: {accuracy:.4f}")
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

misclassified_A.to_csv("ML models/results/misclassified_A.csv", index=False)
misclassified_B.to_csv("ML models/results/misclassified_B.csv", index=False)
