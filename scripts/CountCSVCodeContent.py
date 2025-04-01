import pandas as pd
import numpy as np
import re
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def analyze_java_code(csv_file):
    """
    Reads the CSV file, extracts the code column (if present), and calculates:
      - line count (average and std),
      - word count (average and std),
      - comment count (average and std),
      - comment length (average and std),
      - empty line count (average and std),
      - occurrences of the word "check" (average and std).
    Returns a dictionary of computed metrics.
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Failed to read {csv_file}: {e}")
        return None

    # Identify the appropriate column name for the code
    column_name = None
    for candidate in ["Code", "code", "Extracted_Code"]:
        if candidate in df.columns:
            column_name = candidate
            break

    if column_name is None:
        print(f"Error: No appropriate code column found in {csv_file}.")
        return None

    # Initialize lists for metrics
    line_counts = []
    word_counts = []
    comment_counts = []
    comment_lengths = []
    empty_line_counts = []
    check_counts = []  # New metric for counting occurrences of "check"

    # Process each non-null code entry
    for code in df[column_name].dropna():
        lines = code.split("\n")
        line_counts.append(len(lines))
        
        words = code.split()
        word_counts.append(len(words))
        
        comments = re.findall(r'//.*', code)
        comment_counts.append(len(comments))
        comment_lengths.append(sum(len(comment) for comment in comments))
        
        empty_lines = sum(1 for line in lines if not line.strip())
        empty_line_counts.append(empty_lines)

        # Count occurrences of the word "check" (case-insensitive)
        check_counts.append(len(re.findall(r'\bcheck\b', code, re.IGNORECASE)))

    metrics = {
        "avg_line_count": np.mean(line_counts),
        "std_line_count": np.std(line_counts),
        "avg_word_count": np.mean(word_counts),
        "std_word_count": np.std(word_counts),
        "avg_comment_count": np.mean(comment_counts),
        "std_comment_count": np.std(comment_counts),
        "avg_comment_length": np.mean(comment_lengths),
        "std_comment_length": np.std(comment_lengths),
        "avg_empty_line_count": np.mean(empty_line_counts),
        "std_empty_line_count": np.std(empty_line_counts),
        "avg_check_count": np.mean(check_counts),
        "std_check_count": np.std(check_counts)
    }
    return metrics

# Create an empty list to store results for each file
results = []

# Helper function to extract folder name for Group 2, 3, and 4
def extract_label(filepath, group):
    if group == "CSEDM":
        return os.path.basename(filepath)  # Use filename for Group1
    return os.path.basename(os.path.dirname(filepath))  # Use folder name for other groups

# --- Collect files and process them ---
all_files = []

# Group 1: Single CSV file
all_files.append(("CSEDM", os.path.join("CSV_files", "CodeStates.csv")))

# Group 2: Files in the "prompting" folder (4 subfolders)
all_files.extend([("LLM Datasets", file) for file in glob.glob(os.path.join("prompting", "*", "*.csv"))])

# Group 3: LLM CSV files in the "results" folder (24 subfolders)
results_dir = "ML_models//results"
subfolders = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
for folder in subfolders:
    for fname in ["LLM.csv", "misclassified_LLM_all.csv"]:
        file_path = os.path.join(folder, fname)
        if os.path.exists(file_path):
            all_files.append(("Missidientified LLM", file_path))

# Group 4: Student CSV files in the "results" folder (24 subfolders)
for folder in subfolders:
    for fname in ["Student.csv", "misclassified_Student_all.csv"]:
        file_path = os.path.join(folder, fname)
        if os.path.exists(file_path):
            all_files.append(("Missidentified Student", file_path))

# Process each file
for group, file_path in all_files:
    metrics = analyze_java_code(file_path)
    if metrics:
        metrics["file"] = extract_label(file_path, group)  # Use folder name for Group 2, 3, 4
        metrics["group"] = group
        results.append(metrics)

# Convert the list of dictionaries to a DataFrame and save to a CSV file
results_df = pd.DataFrame(results)
output_csv = os.path.join("scripts", "results_combined.csv")
results_df.to_csv(output_csv, index=False)
print(f"Combined results saved to {output_csv}")

# --- Plotting the results ---
metrics_pairs = [
    ("avg_line_count", "std_line_count"),
    ("avg_word_count", "std_word_count"),
    ("avg_comment_count", "std_comment_count"),
    ("avg_comment_length", "std_comment_length"),
    ("avg_empty_line_count", "std_empty_line_count"),
    ("avg_check_count", "std_check_count")  # New metric pair for "check" occurrences
]

# Define colors for each group
group_colors = {
    "CSEDM": "blue",   # 1 file
    "LLM Datasets": "green",  # 4 files
    "Missidientified LLM": "red",    # 24 files (LLM)
    "Missidentified Student": "orange"  # 24 files (Student)
}

# Ensure charts directory exists
charts_dir = "charts"
if not os.path.exists(charts_dir):
    os.makedirs(charts_dir)

# Create bar charts
for avg_metric, std_metric in metrics_pairs:
    plt.figure(figsize=(12, 6))

    # Create x positions for each file
    x = np.arange(len(results_df))

    # Assign colors based on group
    bar_colors = results_df["group"].map(group_colors)

    # Plot bars with error bars
    plt.bar(x, results_df[avg_metric], yerr=results_df[std_metric],
            color=bar_colors, capsize=5)

    plt.ylabel(avg_metric)
    plt.title(f"{avg_metric} with error bars")

    # Set x-axis labels to display folder names (only for Group 2, 3, 4)
    plt.xticks(x, results_df["file"], rotation=45, ha="right")

    plt.tight_layout()

    legend_patches = [mpatches.Patch(color=color, label=group) for group, color in group_colors.items()]
    plt.legend(handles=legend_patches, title="Groups")

    # Save the chart
    chart_filename = os.path.join(charts_dir, f"{avg_metric}_chart.png")
    plt.savefig(chart_filename)
    plt.close()
    print(f"Chart saved: {chart_filename}")
