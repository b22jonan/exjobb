import pandas as pd
import numpy as np
import re
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Function to sanitize file names and ensure uniqueness
def sanitize_filename(filename, directory):
    # Replace invalid characters with underscores
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

    # Existing counters
    line_counts, word_counts, comment_counts = [], [], []
    comment_lengths, empty_line_counts, check_counts = [], [], []

    # New counters for the additional metrics
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
        
        # Count occurrences of new syntax patterns in each line
        for key in syntax_counts:
            syntax_counts[key].append(len(re.findall(re.escape(key), code)))

    # Calculate averages and standard deviations for all counts
    metrics = {
        "avg_line_count": np.mean(line_counts), "std_line_count": np.std(line_counts),
        "avg_word_count": np.mean(word_counts), "std_word_count": np.std(word_counts),
        "avg_comment_count": np.mean(comment_counts), "std_comment_count": np.std(comment_counts),
        "avg_comment_length": np.mean(comment_lengths), "std_comment_length": np.std(comment_lengths),
        "avg_empty_line_count": np.mean(empty_line_counts), "std_empty_line_count": np.std(empty_line_counts),
        "avg_check_count": np.mean(check_counts), "std_check_count": np.std(check_counts)
    }

    # Add new metrics with their averages and standard deviations
    for key, counts in syntax_counts.items():
        metrics[f"avg_{key}_count"] = np.mean(counts)
        metrics[f"std_{key}_count"] = np.std(counts)

    return metrics

results = []

def extract_label(filepath, group):
    return os.path.basename(filepath) if group == "CSEDM" else os.path.basename(os.path.dirname(filepath))

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
                break  # Stop after the first match is found

print(f"Total files collected: {len(all_files)}")

for group, file_path in all_files:
    metrics = analyze_java_code(file_path)
    if metrics:
        metrics["file"] = extract_label(file_path, group)
        metrics["group"] = group
        results.append(metrics)

print(f"Total results collected: {len(results)}")

results_df = pd.DataFrame(results)
print(f"Total entries in DataFrame: {len(results_df)}")

output_csv = os.path.join("scripts", "results_combined.csv")
results_df.to_csv(output_csv, index=False)
print(f"Combined results saved to {output_csv}")

group_colors = {
    "CSEDM": "blue", "LLM Datasets": "green", 
    "Missclassified LLM": "red", "Missclassified Student": "orange",
    "Classified LLM": "purple", "Classified Student": "brown"
}

charts_dir = "charts"
os.makedirs(charts_dir, exist_ok=True)

# Existing metrics pairs
metrics_pairs = [("avg_line_count", "std_line_count"),
    ("avg_word_count", "std_word_count"),
    ("avg_comment_count", "std_comment_count"),
    ("avg_comment_length", "std_comment_length"),
    ("avg_empty_line_count", "std_empty_line_count"),
    ("avg_check_count", "std_check_count")]

# New syntax-specific metrics pairs
new_syntax_metrics = [
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

# Combine old and new metrics
metrics_pairs.extend(new_syntax_metrics)

# Sort results by group
custom_order = ["CSEDM", "LLM Datasets", "Missclassified LLM", "Missclassified Student", "Classified LLM", "Classified Student"]
results_df["group"] = pd.Categorical(results_df["group"], categories=custom_order, ordered=True)
results_df = results_df.sort_values(by=["group", "file"]).reset_index(drop=True)

# Plot charts for all metrics
for avg_metric, std_metric in metrics_pairs:
    plt.figure(figsize=(16, 20))
    x = np.arange(len(results_df))
    bar_colors = results_df["group"].map(group_colors)

    # Prepare asymmetric error bars
    means = results_df[avg_metric].values
    stds = results_df[std_metric].values
    lower_errors = np.minimum(means, stds)
    upper_errors = stds
    asymmetric_errors = [lower_errors, upper_errors]  # This is a 2D array: [neg, pos]

    # Plot with asymmetric error bars
    plt.barh(x, means, xerr=asymmetric_errors, color=bar_colors, capsize=5)
    plt.xlabel(avg_metric)
    plt.title(f"{avg_metric} with asymmetric standard deviation")
    plt.yticks(x, results_df["file"])
    plt.ylim(-0.5, len(x) - 0.5)
    plt.gca().invert_yaxis()

    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.05)
    
    # Add legend
    legend_patches = [mpatches.Patch(color=color, label=group) for group, color in group_colors.items()]
    plt.legend(handles=legend_patches, title="Groups")

    # Save chart with sanitized and unique filename
    chart_filename = os.path.join(charts_dir, f"{sanitize_filename(avg_metric, charts_dir)}_chart.png")
    plt.savefig(chart_filename)
    plt.close()
    print(f"Chart saved: {chart_filename}")