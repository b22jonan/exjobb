import pandas as pd
import subprocess
import sys
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

# File paths
data_x_path = "prompting/ChatGPT4o/processed_responses.csv"
failed_y_path = "ML_models/results/SVM_ChatGPT4o/Student.csv"
failed_x_path = "ML_models/results/SVM_ChatGPT4o/LLM.csv"
conf_matrix_path = "ML_models/results/SVM_ChatGPT4o/confusion_matrices.csv"

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
    df2 = pd.read_csv(data_x_path, header=0, names=['ID', 'Prompt', 'Code'])

    df2['Label'] = 1
    df1['Label'] = 0 

    # Store ExtraField separately and drop from y_data for training
    x_extra_field = df2[['ID', 'Prompt']]
    df2 = df2.drop(columns=['Prompt'])

    df = pd.concat([df1, df2], ignore_index=True)  # Combine datasets

    # Feature extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Code'])
    y = df['Label']

    random_state = np.random.randint(0, 10000)  # Different random state for each iteration

    # Train-test split for each iteration
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute confusion matrix (TN, FP, FN, TP)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    conf_matrix_list.append([iteration + 1, tn, fp, fn, tp])

    # Identify misclassified samples using .iloc to prevent index errors
    misclassified = df.iloc[y_test.index][y_test != y_pred]

    # Ensure label column is included in the misclassified data
    misclassified = misclassified[['ID', 'Code', 'Label']]
    
    failed_y = misclassified[misclassified['Label'] == 0]
    failed_x = misclassified[misclassified['Label'] == 1]

    failed_x = failed_x.merge(x_extra_field, on="ID", how="left")

    failed_x = failed_x[['ID', 'Code', 'Prompt']]
    failed_y = failed_y[['ID', 'Code']]

    # Save misclassified samples with labels to CSV files
    if iteration == 0 :
        failed_x.to_csv(failed_x_path, mode="w", index=False, header=True)
        failed_y.to_csv(failed_y_path, mode="w", index=False, header=True)
    else :
        failed_x.to_csv(failed_x_path, mode="a", index=False, header=False)
        failed_y.to_csv(failed_y_path, mode="a", index=False, header=False)
    
    iteration += 1
    print(f"{iteration} ML model")

# Save confusion matrices to CSV file
conf_matrix_df = pd.DataFrame(conf_matrix_list, columns=["Loopnr", "TN", "FP", "FN", "TP"])
conf_matrix_df.to_csv(conf_matrix_path, index=False)

# Print the final results
print(f"Results saved: {failed_x_path}, {failed_y_path}, {conf_matrix_path}")
