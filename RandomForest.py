import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load datasets and add labels
data1 = pd.read_csv("MicroDataSets\MicroXData.csv", header=None, names=["CodeStateID", "Code"])
data2 = pd.read_csv("MicroDataSets\MicroYData.csv", header=None, names=["CodeStateID", "Code"])
data1["label"], data2["label"] = 1, 0

# Combine datasets
data = pd.concat([data1, data2], ignore_index=True)

# Convert code to features using TF-IDF
X = TfidfVectorizer(max_features=1000).fit_transform(data["Code"])
y = data["label"]

# Train-test split and Random Forest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")