import csv
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_prompts(txt_file):
    """Reads the TXT file and splits it into a list of prompts."""
    with open(txt_file, 'r', encoding='utf-8') as f:
        prompts = f.read().split('?')
    return [p.strip() for p in prompts if p.strip()]

def count_prompt_occurrences(txt_file, csv_file):
    prompts = read_prompts(txt_file)
    prompt_counts = {f'Nr{i+1}': 0 for i in range(6)}
    
    # Read CSV and count occurrences
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        csv_prompts = [row['Prompt'].strip() for row in reader if 'Prompt' in row]
    
    for i, prompt in enumerate(prompts):
        nr_index = (i % 6)  # Cycle through Nr1 - Nr6
        count = csv_prompts.count(prompt)
        prompt_counts[f'Nr{nr_index + 1}'] += count
    
    return prompt_counts

def process_all_files(txt_file, folder_path):
    # Initialize a list to store counts for each subfolder
    all_counts = []
    subfolder_names = []

    # Iterate over subfolders
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            csv_file = os.path.join(subfolder_path, "Missclassified_LLM.csv")
            if os.path.exists(csv_file):
                subfolder_names.append(subfolder)
                prompt_counts = count_prompt_occurrences(txt_file, csv_file)
                all_counts.append([prompt_counts[f'Nr{i+1}'] for i in range(6)])
            else:
                csv_file = os.path.join(subfolder_path, "misclassified_LLM_all.csv")
                if os.path.exists(csv_file):
                    subfolder_names.append(subfolder)
                    prompt_counts = count_prompt_occurrences(txt_file, csv_file)
                    all_counts.append([prompt_counts[f'Nr{i+1}'] for i in range(6)])

    # Now sort the subfolder names based on the latter half of the folder name
    subfolder_names_sorted = sorted(subfolder_names, key=lambda x: x.split('_')[1] if len(x.split('_')) > 1 else x)

    # Reorder all_counts to match the sorted subfolder names
    sorted_counts = [all_counts[subfolder_names.index(name)] for name in subfolder_names_sorted]

    return subfolder_names_sorted, sorted_counts

def plot_results(subfolder_names, all_counts, bar_names, colors):
    df = pd.DataFrame(all_counts, columns=[f'Nr{i+1}' for i in range(6)], index=subfolder_names)

    # Plotting the 100% stacked bar chart
    ax = df.div(df.sum(axis=1), axis=0).plot(kind='bar', stacked=True, figsize=(10, 6), color=colors, width=0.8)
    
    # Adding the bar names below the columns
    ax.set_xticklabels(subfolder_names, rotation=45, ha='right')
    ax.set_ylabel('Percentage')
    ax.set_title('Prompt missclasification diagram')
    
    # Adjust layout to provide space for labels
    plt.subplots_adjust(bottom=0.35, left=0.1, right=0.9, top=0.9)

    legend_patches = [plt.Line2D([0], [0], color=color, lw=6) for color in colors[:6]]
    plt.legend(legend_patches, bar_names, loc='center', ncol=4, bbox_to_anchor=(0.5, -0.55))
    
    plt.show()

if __name__ == "__main__":
    txt_file = "Prompts.txt"
    folder_path = "ML_models//results"  # Path to the results folder
    bar_names = ['Copy Paste', 'P.e.r.f.e.c.t', 'Memetic', 'Meta', 'Restraints', 'Translate']  # Custom bar names
    colors = ['#DC143C', '#40E0D0', '#E97451', '#D81B60', '#DAA520', '#009688']  # Custom colors

    subfolder_names, all_counts = process_all_files(txt_file, folder_path)
    plot_results(subfolder_names, all_counts, bar_names, colors)
