import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import re

# Load the combined results
results_df = pd.read_csv("scripts/results_combined.csv")

# Define group sets
GROUP1_2 = ["CSEDM", "LLM Datasets"]
GROUP3_6 = ["Missclassified LLM", "Missclassified Student", "Classified LLM", "Classified Student"]

# Define color map
group_colors = {
    "CSEDM": "blue", "LLM Datasets": "green", 
    "Missclassified LLM": "red", "Missclassified Student": "orange",
    "Classified LLM": "purple", "Classified Student": "brown"
}

# Extract folder prefix from file name (assuming it corresponds to first half of folder name)
results_df["folder_prefix"] = results_df["file"].apply(lambda x: x.split("_")[0] if "_" in x else "Other")

# Extract LLM name from file
def extract_llm_name(filename):
    return filename.split("_")[-1].replace(".py", "")  # Adjust if your file format differs

# List of metric pairs
metrics_pairs = [
    ("avg_line_count", "std_line_count"),
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

charts_dir = "charts"
os.makedirs(charts_dir, exist_ok=True)

import urllib.parse

def sanitize_filename(name):
    return urllib.parse.quote(name, safe='')

# Loop through all metrics
for avg_metric, std_metric in metrics_pairs:
    global_max = results_df[avg_metric].max() + results_df[std_metric].max()

    # --- Part 1: Trellis chart for Groups 3–6 ---
    subset = results_df[results_df["group"].isin(GROUP3_6)].copy()
    subset["llm_name"] = subset["file"].apply(extract_llm_name)

    prefixes = sorted(subset["folder_prefix"].unique())
    n_rows = len(prefixes)
    n_cols = len(GROUP3_6)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False, sharex=False)
    fig.subplots_adjust(left=0.2, top=0.9)  # Adjust for row labels and title

    for j, group in enumerate(GROUP3_6):  # columns
        for i, prefix in enumerate(prefixes):  # rows
            ax = axes[i, j]
            data = subset[(subset["folder_prefix"] == prefix) & (subset["group"] == group)]
            if data.empty:
                ax.axis("off")
                continue
            y_pos = np.arange(len(data))
            means = data[avg_metric].values
            stds = data[std_metric].values
            lower_errors = np.minimum(means, stds)
            upper_errors = stds
            ax.barh(y_pos, means, xerr=[lower_errors, upper_errors], color=group_colors[group], capsize=5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(data["llm_name"])
            ax.invert_yaxis()
            ax.set_xlim(0, global_max)

            # Add group name as column title
            if i == 0:
                ax.set_title(group)

            # Add model name as row label
            if j == 0:
                ax.set_ylabel(prefix, rotation=0, labelpad=50, fontsize=10, va='center')

    # Add top-level figure title
    fig.suptitle(
        f"Trellis diagram of ML/LLM misclassification based on {avg_metric}, with asymmetric standard deviation",
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    trellis_filename = os.path.join(charts_dir, f"trellis_{sanitize_filename(avg_metric)}.png")
    plt.savefig(trellis_filename)
    plt.close()

    # --- Part 2: Bar chart for Grops 1 and 2 ---
    subset_g1_g2 = results_df[results_df["group"].isin(GROUP1_2)]
    x = np.arange(len(subset_g1_g2))
    means = subset_g1_g2[avg_metric].values
    stds = subset_g1_g2[std_metric].values
    lower_errors = np.minimum(means, stds)
    upper_errors = stds
    bar_colors = subset_g1_g2["group"].map(group_colors)

    plt.figure(figsize=(12, max(6, len(x) * 0.5)))
    plt.barh(x, means, xerr=[lower_errors, upper_errors], color=bar_colors, capsize=5)
    plt.xlabel(avg_metric)
    plt.title(f"{avg_metric} - Group u1 & 2")
    plt.yticks(x, subset_g1_g2["file"])
    plt.gca().invert_yaxis()
    plt.xlim(0, global_max)


    # Legend
    legend_patches = [mpatches.Patch(color=color, label=group) for group, color in group_colors.items() if group in GROUP1_2]
    plt.legend(handles=legend_patches)

    bar_filename = os.path.join(charts_dir, f"bar_group1_2_{sanitize_filename(avg_metric)}.png")
    plt.tight_layout()
    plt.savefig(bar_filename)
    plt.close()
