import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the dataset
df = pd.read_csv('FinalDataset.csv')

# Remove duplicates based on the 'Code' column
df = df.drop_duplicates(subset=['Code'])

# Check dataset size after removing duplicates
print(f"Total dataset size after removing duplicates: {len(df)}")

# Split the dataset into train (70%), temp (30% for validation and test)
train, temp = train_test_split(df, test_size=0.3, random_state=42)

# Further split temp into validation (15%) and test (15%)
validation, test = train_test_split(temp, test_size=0.5, random_state=42)

# Check sizes of each split
print(f"Train size: {len(train)}, Validation size: {len(validation)}, Test size: {len(test)}")

# Create directories for train, validation, and test sets
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/validation', exist_ok=True)
os.makedirs('data/test', exist_ok=True)

# Function to save code snippets into their respective directories
def save_snippets(data, folder):
    if data.empty:
        print(f"No data found for {folder}. Skipping...")
        return
    for i, row in data.iterrows():
        filename = f"{folder}/snippet_{i}.java"
        try:
            with open(filename, "w", encoding="utf-8") as f:  # Use utf-8 encoding
                f.write(row['Code'])
        except Exception as e:
            print(f"Error writing file {filename}: {e}")
    print(f"Saved {len(data)} files to {folder}")

# Save each split into its respective directory
save_snippets(train, 'data/train')
save_snippets(validation, 'data/validation')
save_snippets(test, 'data/test')

print("Dataset has been successfully split and saved.")
