import os
import pandas as pd
import numpy as np
from decimal import Decimal
import scipy.stats as stats

# Function to calculate both positive and negative metrics
def calculate_metrics(df):
    TP = Decimal(int(df['TP']))
    TN = Decimal(int(df['TN']))
    FP = Decimal(int(df['FP']))
    FN = Decimal(int(df['FN']))
    
    # Positive class metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision_pos = TP / (TP + FP) if (TP + FP) != 0 else Decimal(0)
    recall_pos = TP / (TP + FN) if (TP + FN) != 0 else Decimal(0)
    f1_score_pos = (2 * precision_pos * recall_pos / (precision_pos + recall_pos)) if (precision_pos + recall_pos) != 0 else Decimal(0)

    # Negative class metrics
    precision_neg = TN / (TN + FN) if (TN + FN) != 0 else Decimal(0)
    recall_neg = TN / (TN + FP) if (TN + FP) != 0 else Decimal(0)
    f1_score_neg = (2 * precision_neg * recall_neg / (precision_neg + recall_neg)) if (precision_neg + recall_neg) != 0 else Decimal(0)
    
    return {
        'accuracy': accuracy,
        'precision': precision_pos,
        'recall': recall_pos,
        'f1_score': f1_score_pos,
        'neg_precision': precision_neg,
        'neg_recall': recall_neg,
        'neg_f1_score': f1_score_neg
    }

# Function to load all confusion matrices from the specified directory and subdirectories
def load_confusion_matrices(directory):
    metrics_by_model = {
        'accuracy': {},
        'precision': {},
        'recall': {},
        'f1_score': {},
        'neg_precision': {},
        'neg_recall': {},
        'neg_f1_score': {}
    }
    
    # Traverse all directories and subdirectories using os.walk
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename == "confusion_matrices.csv":
                # Construct the full file path
                file_path = os.path.join(root, filename)
                print(f"Processing file: {file_path}")
                
                # Extract the model name from the path (or filename)
                model_name = os.path.basename(root)  # You can change this logic to get the model name
                
                # Read the confusion matrix CSV file
                df = pd.read_csv(file_path)
                
                # Calculate metrics for each entry (row) in the confusion matrix
                accuracies = []
                precisions = []
                recalls = []
                f1_scores = []
                neg_precisions = []
                neg_recalls = []
                neg_f1_scores = []
                
                for _, row in df.iterrows():
                    metrics = calculate_metrics(row)
                    accuracies.append(float(metrics['accuracy']))
                    precisions.append(float(metrics['precision']))
                    recalls.append(float(metrics['recall']))
                    f1_scores.append(float(metrics['f1_score']))
                    neg_precisions.append(float(metrics['neg_precision']))
                    neg_recalls.append(float(metrics['neg_recall']))
                    neg_f1_scores.append(float(metrics['neg_f1_score']))
                
                # Store the metrics by model
                metrics_by_model['accuracy'][model_name] = accuracies
                metrics_by_model['precision'][model_name] = precisions
                metrics_by_model['recall'][model_name] = recalls
                metrics_by_model['f1_score'][model_name] = f1_scores
                metrics_by_model['neg_precision'][model_name] = neg_precisions
                metrics_by_model['neg_recall'][model_name] = neg_recalls
                metrics_by_model['neg_f1_score'][model_name] = neg_f1_scores
    
    return metrics_by_model

# Function to perform ANOVA on a list of model results
def perform_anova(metrics_by_model):
    # Perform one-way ANOVA for each metric
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'neg_precision', 'neg_recall', 'neg_f1_score']:
        # Prepare the groups for ANOVA (each model's metric values as a list)
        groups = list(metrics_by_model[metric].values())
        
        # Check if we have enough groups for ANOVA
        if len(groups) > 1 and all(len(group) > 1 for group in groups):
            f_statistic, p_value = stats.f_oneway(*groups)
            print(f"{metric.replace('_', ' ').capitalize()} - F-statistic: {f_statistic}, p-value: {p_value:.50e}")
        else:
            print(f"Not enough data for ANOVA on {metric}")

# Define the directory where your confusion matrices are stored
directory = r"ML_models\results"

# Load all confusion matrices and calculate metrics
metrics_by_model = load_confusion_matrices(directory)

# Perform ANOVA on the collected results
perform_anova(metrics_by_model)
