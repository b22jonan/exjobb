import csv

def read_prompts(txt_file):
    """Reads the TXT file and splits it into a list of prompts."""
    with open(txt_file, 'r', encoding='utf-8') as f:
        prompts = f.read().split('?')
    return [p.strip() for p in prompts if p.strip()]

def categorize_prompts(txt_file, csv_file, output_csv):
    prompts = read_prompts(txt_file)
    prompt_mapping = {}
    
    # Assign each unique prompt a type between 1 and 6
    for i, prompt in enumerate(prompts):
        prompt_mapping[prompt] = (i % 6) + 1  # Cycle through 1-6
    
    # Read original CSV and add the "promptType" column
    with open(csv_file, 'r', encoding='utf-8') as infile, open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile)
        # Rename "Prompt" to "prompt" and add "PromptType"
        fieldnames = ["prompt" if field == "Prompt" else field for field in reader.fieldnames] + ['PromptType']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in reader:
            prompt = row['Prompt'].strip()
            row['prompt'] = row.pop('Prompt')  # Rename the column
            row['PromptType'] = prompt_mapping.get(prompt, '')  # Assign type or empty if not found
            writer.writerow(row)


if __name__ == "__main__":
    txt_file = "Prompts.txt"
    csv_file =           "ML_models/code_similarity/misclassified_LLM_all_xg_qwen.csv"
    output_csv = "ML_models/code_similarity/updated_misclassified_LLM_all_xg_qwen.csv"
    
    categorize_prompts(txt_file, csv_file, output_csv)
