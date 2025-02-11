import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load and preprocess data
def load_and_preprocess(file_path, label):
    df = pd.read_csv(file_path, header=None, names=['ID', 'Code'])
    df['Label'] = label
    return df

# Paths to datasets
file_path_1 = 'MicroDataSets/MicroXData.csv'  # Replace with your path
file_path_2 = 'MicroDataSets/MicroYData.csv'  # Replace with your path

# Load datasets and combine
df1 = load_and_preprocess(file_path_1, 0)
df2 = load_and_preprocess(file_path_2, 1)
df = pd.concat([df1, df2])

# Tokenization of code data (convert code into numerical representations)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Code']).toarray()  # Transform code to a matrix of TF-IDF features
y = df['Label'].values

# Handle missing values by replacing NaNs with the median value of each column
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the AdaBoost model using a decision tree as the base estimator
base_model = DecisionTreeClassifier(max_depth=3)
model = AdaBoostClassifier(base_model, n_estimators=50, learning_rate=1)

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')