import pandas as pd
import numpy as np
import re
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    if group == "CSEDM":
        return os.path.basename(filepath)
    dirname = os.path.basename(os.path.dirname(filepath))
    parts = dirname.split("_", 1)
    return parts[1].lower() if len(parts) == 2 else dirname.lower()


# Collect all files
results = []
all_files = [("CSEDM", os.path.join("CSV_files", "CodeStates.csv"))]
all_files.extend([( "LLM Datasets", file) for file in glob.glob(os.path.join("prompting", "*", "*.csv"))])

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

# Analyze and label
for group, file_path in all_files:
    metrics = analyze_java_code(file_path)
    if metrics:
        metrics["file"] = extract_label(file_path, group)
        metrics["group"] = group
        results.append(metrics)

print(f"Total results collected: {len(results)}")

results_df = pd.DataFrame(results)

# Group by 'group' and 'file' (second part of folder name) and aggregate
grouped_df = results_df.groupby(["group", "file"], as_index=False).mean(numeric_only=True)

# Save combined results
output_csv = os.path.join("scripts", "results_combined.csv")
grouped_df.to_csv(output_csv, index=False)
print(f"Combined results saved to {output_csv}")

# Plotting
group_colors = {
    "CSEDM": "blue", "LLM Datasets": "green", 
    "Missclassified LLM": "red", "Missclassified Student": "orange",
    "Classified LLM": "purple", "Classified Student": "brown"
}

charts_dir = "charts"
os.makedirs(charts_dir, exist_ok=True)

metrics_pairs = [("avg_line_count", "std_line_count"),
    ("avg_word_count", "std_word_count"),
    ("avg_comment_count", "std_comment_count"),
    ("avg_comment_length", "std_comment_length"),
    ("avg_empty_line_count", "std_empty_line_count"),
    ("avg_check_count", "std_check_count"),
    ("avg_))_count", "std_))_count"),
    ("avg_<_count", "std_<_count"),
    ("avg_if(_count", "std_if(_count"),
    ("avg_1]_count", "std_1]_count"),
    ("avg_=_count", "std_=_count"),
    ("avg_'_count", "std_'_count"),
    ("avg_(in_count", "std_(in_count"),
    ("avg_h()_count", "std_h()_count"),
    ("avg_:_count", "std_:_count"),
    ("avg_1_count", "std_1_count"),
    ("avg_2_count", "std_2_count"),
    ("avg_3_count", "std_3_count"),
    ("avg_0_count", "std_0_count"),
    ("avg_&&_count", "std_&&_count"),
    ("avg_(_count", "std_(_count"),
    ("avg_-_count", "std_-_count"),
    ("avg_||_count", "std_||_count")
]

custom_order = ["CSEDM", "LLM Datasets", "Missclassified LLM", "Missclassified Student", "Classified LLM", "Classified Student"]
grouped_df["group"] = pd.Categorical(grouped_df["group"], categories=custom_order, ordered=True)
grouped_df = grouped_df.sort_values(by=["group", "file"]).reset_index(drop=True)

for avg_metric, std_metric in metrics_pairs:
    plt.figure(figsize=(8, 10))
    x = np.arange(len(grouped_df))
    bar_colors = grouped_df["group"].map(group_colors)

    means = grouped_df[avg_metric].values
    stds = grouped_df[std_metric].values
    lower_errors = np.minimum(means, stds)
    upper_errors = stds
    asymmetric_errors = [lower_errors, upper_errors]

    plt.barh(x, means, xerr=asymmetric_errors, color=bar_colors, capsize=5)
    plt.xlabel(avg_metric)
    plt.title(f"{avg_metric} with asymmetric standard deviation")
    plt.yticks(x, grouped_df["file"])
    plt.ylim(-0.5, len(x) - 0.5)
    plt.gca().invert_yaxis()

    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.05)
    
    legend_patches = [mpatches.Patch(color=color, label=group) for group, color in group_colors.items()]
    plt.legend(handles=legend_patches, title="Groups")

    chart_filename = os.path.join(charts_dir, f"{sanitize_filename(avg_metric, charts_dir)}_chart.png")
    plt.savefig(chart_filename)
    plt.close()
    print(f"Chart saved: {chart_filename}")
