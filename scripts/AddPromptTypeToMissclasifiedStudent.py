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

add_prompt_type_column('ML_models/results/XGBoost_Qwen/Student.csv', 'ML_models/results/XGBoost_Qwen/Updated_Student.csv')
