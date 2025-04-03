import pandas as pd
import numpy as np
import re
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_java_code(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Failed to read {csv_file}: {e}")
        return None

    column_name = None
    for candidate in ["Code", "code", "Extracted_Code"]:
        if candidate in df.columns:
            column_name = candidate
            break

    if column_name is None:
        print(f"Error: No appropriate code column found in {csv_file}.")
        return None

    metrics = {"file": os.path.basename(csv_file), "group": None, "line_counts": [], "word_counts": [],
               "comment_counts": [], "comment_lengths": [], "empty_line_counts": [], "check_counts": []}

    for code in df[column_name].dropna():
        lines = code.split("\n")
        metrics["line_counts"].append(len(lines))
        
        words = code.split()
        metrics["word_counts"].append(len(words))
        
        comments = re.findall(r'//.*', code)
        metrics["comment_counts"].append(len(comments))
        metrics["comment_lengths"].append(sum(len(comment) for comment in comments))
        
        empty_lines = sum(1 for line in lines if not line.strip())
        metrics["empty_line_counts"].append(empty_lines)
        
        metrics["check_counts"].append(len(re.findall(r'\bcheck\b', code, re.IGNORECASE)))
    
    return metrics

def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

results = []

def extract_label(filepath, group):
    if group == "CSEDM":
        return os.path.basename(filepath)
    return os.path.basename(os.path.dirname(filepath))

all_files = [("CSEDM", os.path.join("CSV_files", "CodeStates.csv"))]
all_files.extend([( "LLM Datasets", file) for file in glob.glob(os.path.join("prompting", "*", "*.csv"))])

results_dir = "ML_models/results"
subfolders = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
for folder in subfolders:
    for fname in ["Missclassified_LLM.csv", "misclassified_LLM_all.csv", "Missclassified_Student.csv", "misclassified_Student_all.csv", "Classified_LLM.csv", "Classified_LLM_all.csv", "Classified_Student.csv", "Classified_Student_all.csv"]:
        file_path = os.path.join(folder, fname)
        if os.path.exists(file_path):
            group = " ".join(fname.split("_")[0:2])
            all_files.append((group, file_path))

for group, file_path in all_files:
    metrics = analyze_java_code(file_path)
    if metrics:
        metrics["group"] = group
        results.append(metrics)

charts_dir = "charts"
os.makedirs(charts_dir, exist_ok=True)

metrics_keys = ["line_counts", "word_counts", "comment_counts", "comment_lengths", "empty_line_counts", "check_counts"]
group_colors = {"CSEDM": "blue", "LLM Datasets": "green", "Missclassified LLM": "red", "Missclassified Student": "orange", "Classified LLM": "purple", "Classified Student": "brown"}

for key in metrics_keys:
    plt.figure(figsize=(12, 6))
    data = []
    labels = []
    colors = []
    
    for result in results:
        filtered_data = remove_outliers(result[key])
        if filtered_data:
            data.append(filtered_data)
            labels.append(result["file"])
            colors.append(group_colors.get(result["group"], "gray"))
    
    sns.boxplot(data=data, palette=colors)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha="right")
    plt.ylabel(key.replace("_", " ").title())
    plt.title(f"Box Plot of {key.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, f"{key}_boxplot.png"))
    plt.close()
    print(f"Box plot saved: {os.path.join(charts_dir, f'{key}_boxplot.png')}")
