import csv
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm

def read_prompts(txt_file):
    """Reads the TXT file and splits it into a list of prompts."""
    with open(txt_file, 'r', encoding='utf-8') as f:
        prompts = f.read().split('?')
    return [p.strip() for p in prompts if p.strip()]

def count_prompt_occurrences(txt_file, csv_file):
    prompts = read_prompts(txt_file)
    prompt_counts = {f'No{i+1}': 0 for i in range(50)}  # 50 bins instead of 6
    
    # Read CSV and count occurrences
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        csv_prompts = [row['Prompt'].strip() for row in reader if 'Prompt' in row]
    
    for i, prompt in enumerate(prompts):
        nr_index = (i // 6) % 50  # Cycle through Nr1 - Nr50
        count = csv_prompts.count(prompt)
        prompt_counts[f'No{nr_index + 1}'] += count
    
    return prompt_counts

def process_all_files(txt_file, folder_path):
    all_counts = []
    subfolder_names = []

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            csv_file = os.path.join(subfolder_path, "Missclassified_LLM.csv")
            if os.path.exists(csv_file):
                subfolder_names.append(subfolder)
                prompt_counts = count_prompt_occurrences(txt_file, csv_file)
                all_counts.append([prompt_counts[f'No{i+1}'] for i in range(50)])
            else:
                csv_file = os.path.join(subfolder_path, "misclassified_LLM_all.csv")
                if os.path.exists(csv_file):
                    subfolder_names.append(subfolder)
                    prompt_counts = count_prompt_occurrences(txt_file, csv_file)
                    all_counts.append([prompt_counts[f'No{i+1}'] for i in range(50)])

    return subfolder_names, all_counts

def plot_heatmap(subfolder_names, all_counts):
    df = pd.DataFrame(all_counts, columns=[f'No {i+1}' for i in range(50)], index=subfolder_names)
    
    # Normalize the data such that the bottom 5% of values are mapped to 0 (black)
    vmin = np.percentile(df.values, 0)  # Bottom 5% value
    vmax = np.max(df.values)  # Maximum value in the dataset
    
    # Use Normalize to scale the data between vmin and vmax
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Create a custom colormap that starts with black and transitions to other colors
    cmap = cm.viridis  
    
    plt.figure(figsize=(14, 8))  # Adjust figure size for better spacing
    
    # Apply colormap with normalization
    heatmap = plt.imshow(df, aspect='auto', cmap=cmap, norm=norm)
    
    # Add colorbar with the correct label and shrink it for better visibility
    cbar = plt.colorbar(heatmap, label='Misclassifications', shrink=0.8)
    
    # Customize ticks and labels
    plt.xticks(ticks=np.arange(50), labels=[f'No {i+1}' for i in range(50)], rotation=90)
    plt.yticks(ticks=np.arange(len(subfolder_names)), labels=subfolder_names, ha='right')
    
    plt.xlabel("Question No")
    plt.ylabel("LLM/ML combo")
    plt.title("Question Misclassification Heatmap")
    
    # Adjust layout for better spacing
    plt.subplots_adjust(left=0.2, right=0.85)
    
    plt.show()

if __name__ == "__main__":
    txt_file = "Prompts.txt"
    folder_path = "ML_models//results"
    
    subfolder_names, all_counts = process_all_files(txt_file, folder_path)
    plot_heatmap(subfolder_names, all_counts)
