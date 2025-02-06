import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load and label datasets
df1 = pd.read_csv('MicroDataSets\MicroXData.csv', header=None, names=['CodeStateID', 'Code'])
df2 = pd.read_csv('MicroDataSets\MicroYData.csv', header=None, names=['CodeStateID', 'Code'])
df = pd.concat([df1.assign(Label=1), df2.assign(Label=0)])

# Feature extraction and train-test split
X = TfidfVectorizer(stop_words='english').fit_transform(df['Code'])
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM and print accuracy
model = SVC(kernel='linear').fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")