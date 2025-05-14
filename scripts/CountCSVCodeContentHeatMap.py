import pandas as pd
import numpy as np
import re
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# --- Analysis Function ---
def sanitize_filename(filename, directory):
    sanitized = re.sub(r'[:]', "standingDots", filename)
    sanitized = re.sub(r'[<]', "leftArrow", sanitized)
    sanitized = re.sub(r'[||]', "standingLines", sanitized)
    return sanitized

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

    line_counts, word_counts, comment_counts = [], [], []
    comment_lengths, empty_line_counts, check_counts = [], [], []

    syntax_counts = {
        "))": [], "<": [], "if(": [], "1]": [], "=": [], "'": [], "(in": [], "h()": [], ":": [],
        "1": [], "2": [], "3": [], "0": [], "&&": [], "(": [], "-": [], "||": []
    }

    for code in df[column_name].dropna():
        lines = code.split("\n")
        line_counts.append(len(lines))
        word_counts.append(len(code.split()))
        comments = re.findall(r'//.*', code)
        comment_counts.append(len(comments))
        comment_lengths.append(sum(len(comment) for comment in comments))
        empty_line_counts.append(sum(1 for line in lines if not line.strip()))
        check_counts.append(len(re.findall(r'\bcheck\b', code, re.IGNORECASE)))
        
        for key in syntax_counts:
            syntax_counts[key].append(len(re.findall(re.escape(key), code)))

    metrics = {
        "avg_line_count": np.mean(line_counts), "std_line_count": np.std(line_counts),
        "avg_word_count": np.mean(word_counts), "std_word_count": np.std(word_counts),
        "avg_comment_count": np.mean(comment_counts), "std_comment_count": np.std(comment_counts),
        "avg_comment_length": np.mean(comment_lengths), "std_comment_length": np.std(comment_lengths),
        "avg_empty_line_count": np.mean(empty_line_counts), "std_empty_line_count": np.std(empty_line_counts),
        "avg_check_count": np.mean(check_counts), "std_check_count": np.std(check_counts)
    }

    for key, counts in syntax_counts.items():
        metrics[f"avg_{key}_count"] = np.mean(counts)
        metrics[f"std_{key}_count"] = np.std(counts)

    return metrics

def extract_label(filepath, group):
    return os.path.basename(filepath) if group == "CSEDM" else os.path.basename(os.path.dirname(filepath))

# --- File Collection ---
results = []
all_files = [("CSEDM", os.path.join("CSV_files", "CodeStates.csv"))]
all_files.extend([("LLM Datasets", file) for file in glob.glob(os.path.join("prompting", "*", "*.csv"))])

results_dir = "ML_models/results"
subfolders = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

file_variants = [
    (["Missclassified_LLM.csv", "Misclassified_LLM_all.csv"], "Missclassified LLM"),
    (["Missclassified_Student.csv", "Misclassified_Student_all.csv"], "Missclassified Student"),
    (["Classified_LLM.csv", "Classified_LLM_all.csv"], "Classified LLM"),
    (["Classified_Student.csv", "Classified_Student_all.csv"], "Classified Student")
]

for folder in subfolders:
    for filenames, label in file_variants:
        for fname in filenames:
            file_path = os.path.join(folder, fname)
            if os.path.exists(file_path):
                all_files.append((label, file_path))
                break

print(f"Total files collected: {len(all_files)}")

# --- Process Files ---
for group, file_path in all_files:
    metrics = analyze_java_code(file_path)
    if metrics:
        filename = extract_label(file_path, group)
        if group in ["CSEDM", "LLM Datasets"]:
            column_key = group
            row_key = "Deepseek"  # Default row for these groups
        else:
            if "_" in filename:
                column_key, row_key = filename.split("_", 1)
            else:
                column_key, row_key = "Unknown", "Unknown"
        metrics.update({
            "file": filename,
            "group": group,
            "column_key": column_key.strip(),
            "row_key": row_key.strip()
        })
        results.append(metrics)

print(f"Total results collected: {len(results)}")

# --- Create DataFrame ---
df = pd.DataFrame(results)

output_csv = os.path.join("scripts", "results_combined.csv")
df.to_csv(output_csv, index=False)
print(f"Combined results saved to {output_csv}")

custom_cmap = LinearSegmentedColormap.from_list(
    "custom_cmap",
    ["cyan", "magenta", "yellow", "black"]
)

# --- Special Column 2 Data ---
special_column_2_files = {
    "deepseek": "prompting/DeepSeek/processed_responses.csv",
    "qwen": "prompting/Qwen/processed_responses.csv",
    "chatgpt4o": "prompting/ChatGPT4o/processed_responses.csv",
    "chatgpt35": "prompting/ChatGPT35/processed_responses.csv"
}

column2_overrides = {}  # row_key (lowercase) → metrics
for row_key, path in special_column_2_files.items():
    metrics = analyze_java_code(path)
    if metrics:
        column2_overrides[row_key] = metrics

# --- Plot Heatmaps ---
charts_dir = "charts"
os.makedirs(charts_dir, exist_ok=True)

row_labels = ["Deepseek", "Qwen", "ChatGPT4o", "ChatGPT35"]
row_mapping = {r.lower(): i for i, r in enumerate(row_labels)}

col_labels = ["CSEDM", "LLM Datasets"]
grouped_models = {
    "Missclassified LLM": ["RandomForest", "NN", "SVM", "XGBoost", "LightGBM", "AdaBoost"],
    "Missclassified Student": ["RandomForest", "NN", "SVM", "XGBoost", "LightGBM", "AdaBoost"],
    "Classified LLM": ["RandomForest", "NN", "SVM", "XGBoost", "LightGBM", "AdaBoost"],
    "Classified Student": ["RandomForest", "NN", "SVM", "XGBoost", "LightGBM", "AdaBoost"]
}
for group, models in grouped_models.items():
    for model in models:
        col_labels.append(f"{group} - {model}")

col_mapping = {c.lower(): i for i, c in enumerate(col_labels)}

metric_keys = [col for col in df.columns if col.startswith("avg_")]

for metric in metric_keys:
    heatmap = np.full((4, 26), np.nan)

    for _, row in df.iterrows():
        row_key = str(row["row_key"]).lower()
        r_idx = row_mapping.get(row_key)
        if pd.isna(row["column_key"]) or pd.isna(row["row_key"]):
            continue

        group = row["group"]
        model = row["column_key"]

        if group == "CSEDM":
            col_name = "CSEDM"
        elif group == "LLM Datasets":
            col_name = "LLM Datasets"
        else:
            col_name = f"{group} - {model}"

        c_idx = col_mapping.get(col_name.lower())
        if r_idx is not None and c_idx is not None:
            heatmap[r_idx, c_idx] = row[metric]

    # Override Column 2 (index 1) values with special CSV data
    for row_label, r_idx in row_mapping.items():
        if row_label in column2_overrides and metric in column2_overrides[row_label]:
            heatmap[r_idx, 1] = column2_overrides[row_label][metric]

    # Plot
    plt.figure(figsize=(12, 4))
    plt.imshow(heatmap, cmap=custom_cmap, aspect="auto")
    plt.colorbar(label=metric)
    plt.xticks(np.arange(26), col_labels, rotation=45)
    plt.yticks(np.arange(4), row_labels)
    plt.title(f"Heatmap of {metric}")
    plt.tight_layout()

    chart_path = os.path.join(charts_dir, f"{sanitize_filename(metric, charts_dir)}_heatmap.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Chart saved: {chart_path}")
