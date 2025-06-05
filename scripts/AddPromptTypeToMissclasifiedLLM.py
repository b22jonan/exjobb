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

LLMs = ["Qwen", "ChatGPT4o", "ChatGPT35", "DeepSeek"]
MLs = ["RandomForest", "SVM", "LightGBM", "NN", "XGBoost", "AdaBoost"]

if __name__ == "__main__":
    txt_file = "Prompts.txt"
    
    for LLM in LLMs:
        for ML in MLs:
            # Specify the CSV file path based on the model
            csv_file = f"ML_models/results/{ML}_{LLM}/classified_LLM_all.csv"
            output_csv = f"ML_models/code_similarity/csv_files_llm_not_in_use/updated_classified_LLM_{ML}_{LLM}.csv"
            
            categorize_prompts(txt_file, csv_file, output_csv)

