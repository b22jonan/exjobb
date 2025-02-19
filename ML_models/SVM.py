import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Paths (assuming you already have these set up)
data_x_path = "CSV_files/CodeStates.csv"
data_y_path = "prompting/Qwen/processed_Qwen_Responses.csv"
failed_x_path = "ML_models/results/SVM_Qwen/Student.csv"
failed_y_path = "ML_models/results/SVM_Qwen/LLM.csv"
txt_results_path = "ML_models/results/SVM_Qwen/metrics.txt"

# Load and label datasets
df1 = pd.read_csv(data_x_path, header=None, names=['CodeStateID', 'Code'])
df2 = pd.read_csv(data_y_path, header=None, names=['CodeStateID', 'Code'])

df1 = df1[['CodeStateID', 'Code']].assign(Label=1)  # Dataset 1 (Label 1)
df2 = df2[['CodeStateID', 'Code']].assign(Label=0)  # Dataset 2 (Label 0)

df = pd.concat([df1, df2]).reset_index(drop=True)  # Combine datasets

# Feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Code'])  # Only 'Code' is considered here
y = df['Label']

# Initialize variables to store results across multiple iterations
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Variables to store misclassified samples
misclassified_df1 = pd.DataFrame(columns=df.columns)  # For Label 1 (failed_x_path)
misclassified_df2 = pd.DataFrame(columns=df.columns)  # For Label 0 (failed_y_path)

# Set the number of iterations (you can adjust this number)
num_iterations = 3
iteration = 0

# Create SVM model
model = SVC(kernel='linear', C=1.0)

# While loop to run multiple iterations with different random states
while iteration < num_iterations:
    random_state = np.random.randint(0, 10000)  # Different random state for each iteration
    
    # Train-test split for each iteration
    X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X, y, df.index, test_size=0.2, random_state=random_state)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics for this iteration
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store the metrics for this iteration
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    
    # Identify misclassified samples using .iloc to prevent index errors
    misclassified = df.iloc[test_index].copy()  # Use .iloc to avoid index mismatches
    misclassified = misclassified[y_test.to_numpy() != y_pred]  # Convert y_test to numpy for comparison
    
    # Separate the misclassified samples into two DataFrames based on the 'Label'
    misclassified_df1 = pd.concat([misclassified_df1, misclassified[misclassified['Label'] == 1]], ignore_index=True)
    misclassified_df2 = pd.concat([misclassified_df2, misclassified[misclassified['Label'] == 0]], ignore_index=True)
    
    iteration += 1
    print(f"{iteration} ML model")

# Calculate the mean and standard deviation of each metric
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

mean_precision = np.mean(precisions)
std_precision = np.std(precisions)

mean_recall = np.mean(recalls)
std_recall = np.std(recalls)

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

# Save the final metrics (mean and standard deviation) to a text file
with open(txt_results_path, "w") as f:
    f.write(f"Average Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
    f.write(f"Average Precision: {mean_precision:.4f} ± {std_precision:.4f}\n")
    f.write(f"Average Recall: {mean_recall:.4f} ± {std_recall:.4f}\n")
    f.write(f"Average F1 Score: {mean_f1:.4f} ± {std_f1:.4f}\n")

# Save misclassified samples to CSV files for both datasets
misclassified_df1.to_csv(failed_x_path, index=False)
misclassified_df2.to_csv(failed_y_path, index=False)

# Print the final results
print(f"Average Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Average Precision: {mean_precision:.4f} ± {std_precision:.4f}")
print(f"Average Recall: {mean_recall:.4f} ± {std_recall:.4f}")
print(f"Average F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")