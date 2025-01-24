import pandas as pd
import os
import subprocess

csv_file_path = "FinalDataset.csv"
df = pd.read_csv(csv_file_path)

assert 'Code' in df.columns, "CSV should have a 'Code' column"
assert 'ProblemID' in df.columns, "CSV should have a 'ProblemID' column"

# Create a file with each code snippet formatted in a way code2vec can understand
with open(os.path.join("code2vec_data", "code_data.txt"), "w", encoding="utf-8") as code_file:
    for index, row in df.iterrows():
        code = row['Code']
        ProblemID = row['ProblemID']
        code_file.write(f"{ProblemID} {code}\n")

training_instructions = f"python -m code2vec.train --train_data {os.path.join('code2vec_data', 'code_data.txt')} --model_dir {'code2vec_data'} --epochs 1"
subprocess.run(training_instructions, shell=True, check=True)

print("Training completed!")