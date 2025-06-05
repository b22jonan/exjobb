import csv

def count_rows_in_csv(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        row_count = sum(1 for row in reader)
    return row_count

# Example usage:
file_path = 'ML_models//results//SVM_DeepSeek//misclassified_LLM_all.csv' 
print(f"Number of rows: {count_rows_in_csv(file_path)}")