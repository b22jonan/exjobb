import os 
import pandas as pd
import matplotlib.pyplot as plt

def calculate_metrics(TN, FP, FN, TP):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    neg_precision = TN / (TN + FN) if (TN + FN) > 0 else 0
    neg_recall = TN / (TN + FP) if (TN + FP) > 0 else 0
    neg_f1_score = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score, neg_precision, neg_recall, neg_f1_score

def read_and_process_csv_files(results_folder):
    metrics = {"Accuracy": [], "LLM Precision": [], "LLM Recall": [], "LLM F1-score": [], "Student Precision": [], "Student Recall": [], "Student F1-score": []}
    labels = []
    
    subfolders = sorted([f for f in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, f))])
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(results_folder, subfolder)
        csv_file_path = os.path.join(subfolder_path, "confusion_matrices.csv")
        
        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)
            if {'TN', 'FP', 'FN', 'TP'}.issubset(df.columns):  
                df[['Accuracy', 'LLM Precision', 'LLM Recall', 'LLM F1-score', 'Student Precision', 'Student Recall', 'Student F1-score']] = df.apply(
                    lambda row: pd.Series(calculate_metrics(row['TN'], row['FP'], row['FN'], row['TP'])), axis=1
                )
                
                for key in metrics.keys():
                    metrics[key].append(df[key].tolist())
                
                labels.append(subfolder)
    
    return metrics, labels

def plot_boxplot(data, labels, metric_name):
    num_plots = len(data)
    colors = ["darkgreen", "purple", "darkorange", "mediumblue"] * (num_plots // 4)
    segment_lines = [i * 4 for i in range(1, num_plots // 4)]
    
    plt.figure(figsize=(10, 5))
    box = plt.boxplot(data, widths=1, patch_artist=True, medianprops=dict(color="black"))
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    for line_pos in segment_lines:
        plt.axvline(x=line_pos + 0.5, color='black', linewidth=2)
    
    num_segments = num_plots // 4
    segment_names = ["AdaBoost", "LightGBM", "Neural network", "Random Forest", "SVM", "XGBoost"]
    plt.xticks([2.5 + i * 4 for i in range(num_segments)], segment_names)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} for all ML Models/LLMs")
    plt.ylim(0.75, 1.0)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    legend_labels = ["ChatGPT3.5", "ChatGPT4o", "DeepSeek", "Qwen"]
    legend_patches = [plt.Line2D([0], [0], color=color, lw=4) for color in colors[:4]]
    plt.legend(legend_patches, legend_labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.15))
    
    plt.grid(axis='x', linestyle='--', alpha=0)
    plt.show()

if __name__ == "__main__":
    results_folder = "ML_models/results"  # Change this if needed
    metrics, labels = read_and_process_csv_files(results_folder)
    
    for metric_name, data in metrics.items():
        if data:
            plot_boxplot(data, labels, metric_name)
        else:
            print(f"No valid data found for {metric_name}.")
