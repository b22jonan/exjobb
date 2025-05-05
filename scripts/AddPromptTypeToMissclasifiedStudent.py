import csv

def add_prompt_type_column(input_file, output_file):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = list(reader)
        
        if not rows:
            print("The input file is empty.")
            return
        
        # Add the new column name
        rows[0].append("PromptType")
        
        # Add '0' to each row
        for row in rows[1:]:
            row.append("0")
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

    print(f"File '{output_file}' created successfully with 'PromptType' column.")

LLMs = ["Qwen", "ChatGPT4o", "ChatGPT35", "DeepSeek"]
MLs = ["RandomForest", "SVM", "LightGBM", "NN", "XGBoost", "AdaBoost"]

if __name__ == "__main__":
    for LLM in LLMs:
        for ML in MLs:
            # Specify the CSV file path based on the model
            csv_file = f"ML_models/results/{ML}_{LLM}/misclassified_Student_all.csv"
            output_csv = f"ML_models/code_similarity/csv_files_student_not_in_use/updated_misclassified_Student_{ML}_{LLM}.csv"
            
            add_prompt_type_column(csv_file, output_csv)