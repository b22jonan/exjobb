import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_csv(file, label):
    df = pd.read_csv(file, encoding="utf-8")
    df["label"] = label  # Assign label (0 for dataset1, 1 for dataset2)
    return df[["CodeStateID", "Code", "label"]]  # Only return necessary columns

# Load datasets and assign labels (0 for dataset1, 1 for dataset2)
df = pd.concat([
    load_csv("MicroDataSets\MicroXData.csv", 0),  # Assign 0 as label to dataset1
    load_csv("MicroDataSets\MicroYData.csv", 1)   # Assign 1 as label to dataset2
], ignore_index=True)

# Convert the 'Code' column (text data) into TF-IDF features
X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6)).fit_transform(df["Code"])
y = df["label"]  # Labels are assigned based on dataset origin (0 or 1)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a LightGBM model
model = lgb.LGBMClassifier(learning_rate=0.05, num_leaves=31).fit(X_train, y_train)

# Evaluate the model
print(f"Test Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")