import pandas as pd
import subprocess
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

# File paths
data_x_path = "prompting/DeepSeek/processed_responses.csv"
failed_y_path = "ML_models/results/NN_DeepSeek/Student.csv"
failed_x_path = "ML_models/results/NN_DeepSeek/LLM.csv"
conf_matrix_path = "ML_models/results/NN_DeepSeek/confusion_matrices.csv"

# Variables to store misclassified samples
misclassified_df1 = pd.DataFrame()  # For Label 0 (failed_x_path)
misclassified_df2 = pd.DataFrame()  # For Label 1 (failed_y_path)

# Store confusion matrices
confusion_matrices = []

# Set the number of iterations
num_iterations = 50
iteration = 0

# Create Neural Network model
model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)

while iteration < num_iterations:
    subprocess.run([sys.executable, "scripts/dataset_sampler.py"], check=True)

    df1 = pd.read_csv("CSV_files/Sampled_CodeStates.csv", header=0, names=["ID", "Code"])
    df2 = pd.read_csv(data_x_path, header=0, names=['ID', 'Prompt', 'Code'])

    df2['Label'] = 1
    df1['Label'] = 0 

    x_extra_field = df2[['ID', 'Prompt']]
    df2 = df2.drop(columns=['Prompt'])

    df = pd.concat([df1, df2] , ignore_index=True)

    # Feature extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Code'])
    y = df['Label']

    random_state = np.random.randint(0, 10000)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    confusion_matrices.append([iteration + 1, tn, fp, fn, tp])

    # Identify misclassified samples
    misclassified = df.iloc[y_test.index][y_test != y_pred]

    # Ensure label column is included in the misclassified data
    misclassified = misclassified[['ID', 'Code', 'Label']]

    failed_x = misclassified[misclassified['Label'] == 1]  
    failed_y = misclassified[misclassified['Label'] == 0] 

    failed_x = failed_x.merge(x_extra_field, on="ID", how="left")

    failed_x = failed_x[['ID', 'Code', 'Prompt']]
    failed_y = failed_y[['ID', 'Code']]

    misclassified_df1 = pd.concat([misclassified_df1, failed_x], ignore_index=True)
    misclassified_df2 = pd.concat([misclassified_df2, failed_y], ignore_index=True)

    iteration += 1
    print(f"{iteration} ML model")

# Save confusion matrices
confusion_df = pd.DataFrame(confusion_matrices, columns=["Loopnr","TN", "FP", "FN", "TP"])
confusion_df.to_csv(conf_matrix_path, index=False)

# Save misclassified samples with labels
misclassified_df1.to_csv(failed_x_path, index=False)
misclassified_df2.to_csv(failed_y_path, index=False)

print(f"Results saved: {conf_matrix_path}, {failed_x_path}, {failed_y_path}")
