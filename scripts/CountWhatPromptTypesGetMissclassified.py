import csv

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
    
    # Print results
    for key, value in prompt_counts.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    txt_file = "Prompts.txt"
    csv_file = "ML_models//results//NN_ChatGPT4o//LLM.csv"
    count_prompt_occurrences(txt_file, csv_file)
