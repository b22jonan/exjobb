import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and prepare the datasets
df1 = pd.read_csv("dataset1.csv", header=None, names=["id", "code"]).assign(label=0)
df2 = pd.read_csv("dataset2.csv", header=None, names=["id", "code"]).assign(label=1)

# Combine datasets and split into features (X) and target (y)
df = pd.concat([df1, df2])
X, y = df["code"], df["label"]

# Vectorize the code using TF-IDF and split the data
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
    TfidfVectorizer(max_features=1000, ngram_range=(1, 2)).fit_transform(X), y, test_size=0.2, random_state=42)

# Train the model and evaluate its accuracy
model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=100)
model.fit(X_train_tfidf, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test_tfidf))
print(f"Accuracy: {accuracy * 100:.2f}%")