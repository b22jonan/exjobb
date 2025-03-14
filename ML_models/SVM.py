import pandas as pd
import subprocess
import sys
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

# File paths
data_y_path = "prompting/Qwen/processed_responses.csv"
failed_y_path = "ML_models/results/SVM_Qwen/Student.csv"
failed_x_path = "ML_models/results/SVM_Qwen/LLM.csv"
conf_matrix_path = "ML_models/results/SVM_Qwen/confusion_matrices.csv"

# Variables to store misclassified samples
misclassified_df1 = pd.DataFrame()  # For Label 1 (failed_x_path)
misclassified_df2 = pd.DataFrame()  # For Label 0 (failed_y_path)

# Store confusion matrices
conf_matrix_list = []

# Set the number of iterations
num_iterations = 50
iteration = 0

# Create SVM model
model = SVC(kernel='linear', C=1.0)

# Loop through iterations
while iteration < num_iterations:
    subprocess.run([sys.executable, "scripts/dataset_sampler.py"], check=True)

    df1 = pd.read_csv("CSV_files/Sampled_CodeStates.csv", header=0, names=["ID", "Code"])
    df2 = pd.read_csv(data_y_path, header=None, names=['ID', 'Prompt', 'Code'])

    # Store ExtraField separately and drop from y_data for training
    y_extra_field = df2[['ID', 'Prompt']]
    df2 = df2.drop(columns=['Prompt'])

    df1 = df1[['ID', 'Code']].assign(Label=0)  
    df2 = df2[['ID', 'Code']].assign(Label=1)

    df = pd.concat([df1, df2]).reset_index(drop=True)  # Combine datasets

    # Feature extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Code'])
    y = df['Label']

    random_state = np.random.randint(0, 10000)  # Different random state for each iteration

    # Train-test split for each iteration
    X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(
        X, y, df.index, test_size=0.2, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute confusion matrix (TN, FP, FN, TP)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    conf_matrix_list.append([iteration + 1, tn, fp, fn, tp])

    # Identify misclassified samples using .iloc to prevent index errors
    misclassified = df.iloc[test_index].copy()
    misclassified = misclassified[y_test.to_numpy() != y_pred]
    
    # Assign labels to misclassified entries
    misclassified["Label"] = misclassified["Label"].astype(int)

    failed_x = misclassified[misclassified['Label'] == 0]
    failed_y = misclassified[misclassified['Label'] == 1]

    failed_y = failed_y.merge(y_extra_field, on="ID", how="left")

    failed_y = failed_y[['ID', 'Code', 'Prompt']]
    failed_x = failed_x[['ID', 'Code']]

    # Append to misclassified DataFrames
    misclassified_df1 = pd.concat([misclassified_df1, failed_y], ignore_index=True)
    misclassified_df2 = pd.concat([misclassified_df2, failed_x], ignore_index=True)
    
    iteration += 1
    print(f"{iteration} ML model")

# Save misclassified samples with labels to CSV files
misclassified_df1.to_csv(failed_x_path, index=False)
misclassified_df2.to_csv(failed_y_path, index=False)

# Save confusion matrices to CSV file
conf_matrix_df = pd.DataFrame(conf_matrix_list, columns=["loopnr", "tn", "fp", "fn", "tp"])
conf_matrix_df.to_csv(conf_matrix_path, index=False)

# Print the final results
print(f"Results saved: {failed_x_path}, {failed_y_path}, {conf_matrix_path}")
