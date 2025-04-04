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

    column_name = next((col for col in ["Code", "code", "Extracted_Code"] if col in df.columns), None)
    if column_name is None:
        print(f"Error: No appropriate code column found in {csv_file}.")
        return None

    metrics = []
    for code in df[column_name].dropna():
        lines = code.split("\n")
        words = code.split()
        comments = re.findall(r'//.*', code)
        empty_lines = sum(1 for line in lines if not line.strip())
        check_count = len(re.findall(r'\bcheck\b', code, re.IGNORECASE))

        metrics.append({
            "line_count": len(lines),
            "word_count": len(words),
            "comment_count": len(comments),
            "comment_length": sum(len(comment) for comment in comments),
            "empty_line_count": empty_lines,
            "check_count": check_count
        })
    
    return pd.DataFrame(metrics)

def extract_label(filepath, group):
    return os.path.basename(filepath) if group == "CSEDM" else os.path.basename(os.path.dirname(filepath))

all_files = [("CSEDM", os.path.join("CSV_files", "CodeStates.csv"))]
all_files.extend([( "LLM Datasets", file) for file in glob.glob(os.path.join("prompting", "*", "*.csv"))])

results_dir = "ML_models//results"
subfolders = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
for folder in subfolders:
    for group, filenames in [
        ("Missclassified LLM", ["Missclassified_LLM.csv", "misclassified_LLM_all.csv"]),
        ("Missclassified Student", ["Missclassified_Student.csv", "misclassified_Student_all.csv"]),
        ("Classified LLM", ["Classified_LLM.csv", "Classified_LLM_all.csv"]),
        ("Classified Student", ["Classified_Student.csv", "Classified_Student_all.csv"])
    ]:
        for fname in filenames:
            file_path = os.path.join(folder, fname)
            if os.path.exists(file_path):
                all_files.append((group, file_path))

results = []
for group, file_path in all_files:
    df_metrics = analyze_java_code(file_path)
    if df_metrics is not None:
        df_metrics["file"] = extract_label(file_path, group)
        df_metrics["group"] = group
        results.append(df_metrics)

results_df = pd.concat(results, ignore_index=True)
output_csv = os.path.join("scripts", "results_combined.csv")
results_df.to_csv(output_csv, index=False)
print(f"Combined results saved to {output_csv}")

# Define metrics to plot
metrics = ["line_count", "word_count", "comment_count", "comment_length", "empty_line_count", "check_count"]

group_colors = {
    "CSEDM": "blue",
    "LLM Datasets": "green",
    "Missclassified LLM": "red",
    "Missclassified Student": "orange",
    "Classified LLM": "purple",
    "Classified Student": "brown"
}

charts_dir = "charts"
os.makedirs(charts_dir, exist_ok=True)

# Generate violin plots
for metric in metrics:
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x="file", y=metric, hue="group", data=results_df, palette=group_colors, split=True, inner="quartile")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric} distribution across files")
    plt.legend(title="Groups")
    plt.tight_layout()
    
    chart_filename = os.path.join(charts_dir, f"{metric}_violin.png")
    plt.savefig(chart_filename)
    plt.close()
    print(f"Chart saved: {chart_filename}")