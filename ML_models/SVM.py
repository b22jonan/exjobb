import pandas as pd
import subprocess
import sys
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# File paths
data_y_path = "prompting/ChatGPT4o/processed_responses.csv"
failed_x_path = "ML_models/results/SVM_ChatGPT4o/Student.csv"
failed_y_path = "ML_models/results/SVM_ChatGPT4o/LLM.csv"
txt_results_path = "ML_models/results/SVM_ChatGPT4o/metrics.txt"

# Initialize variables to store results across multiple iterations
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Variables to store misclassified samples
misclassified_df1 = pd.DataFrame()  # For Label 1 (failed_x_path)
misclassified_df2 = pd.DataFrame()  # For Label 0 (failed_y_path)

# Set the number of iterations
num_iterations = 50
iteration = 0

# Create SVM model
model = SVC(kernel='linear', C=1.0)

# Open the results file
with open(txt_results_path, "w") as f:
    # While loop to run multiple iterations with different random states
    while iteration < num_iterations:
        subprocess.run([sys.executable, "scripts/dataset_sampler.py"], check=True)

        df1 = pd.read_csv("CSV_files/Sampled_CodeStates.csv", header=0, names=["CodeStateID", "Code"])
        df2 = pd.read_csv(data_y_path, header=None, names=['CodeStateID','Prompt','Code'])

        # Store ExtraField separately and drop from y_data for training
        y_extra_field = df2[['CodeStateID', 'Prompt']]
        df2 = df2.drop(columns=['Prompt'])

        df1 = df1[['CodeStateID', 'Code']].assign(Label=1)  # Dataset 1 (Label 1)
        df2 = df2[['CodeStateID', 'Code']].assign(Label=0)  # Dataset 2 (Label 0)

        df = pd.concat([df1, df2]).reset_index(drop=True)  # Combine datasets

        # Feature extraction
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df['Code'])
        y = df['Label']

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

        # Save iteration-wise metrics
        f.write(f"Iteration {iteration + 1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}\n")

        # Identify misclassified samples using .iloc to prevent index errors
        misclassified = df.iloc[test_index].copy()
        misclassified = misclassified[y_test.to_numpy() != y_pred]

        failed_x = misclassified[misclassified['Label'] == 0].drop(columns=['Label'])
        failed_y = misclassified[misclassified['Label'] == 1].drop(columns=['Label'])

        failed_x = failed_x.merge(y_extra_field, on="CodeStateID", how="left")

        # Separate the misclassified samples into two DataFrames based on the 'Label'
        misclassified_df1 = pd.concat([misclassified_df1, failed_y], ignore_index=True)
        misclassified_df2 = pd.concat([misclassified_df2, failed_x], ignore_index=True)
        
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

    # Save the final metrics (mean and standard deviation) to the text file
    f.write("\nOverall Metrics:\n")
    f.write(f"Average Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
    f.write(f"Average Precision: {mean_precision:.4f} ± {std_precision:.4f}\n")
    f.write(f"Average Recall: {mean_recall:.4f} ± {std_recall:.4f}\n")
    f.write(f"Average F1 Score: {mean_f1:.4f} ± {std_f1:.4f}\n")

# Save misclassified samples to CSV files for both datasets
misclassified_df1.to_csv(failed_x_path, index=False)
misclassified_df2.to_csv(failed_y_path, index=False)

# Print the final results
print(f"Results saved: {txt_results_path}, {failed_x_path}, {failed_y_path}")
