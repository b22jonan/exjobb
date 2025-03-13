import pandas as pd
import glob
import os

csv_folder = r'C:\Users\jonat\Desktop\docs\exjobb\ML_models\code_similarity\batch'

csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))

if not csv_files:
    raise FileNotFoundError(f'No CSV files found in directory: {csv_folder}')

merged_dataframes = []

for file in csv_files:
    df = pd.read_csv(file)

    # Determine the correct ID column
    if 'CodeStateID' in df.columns:
        df['ID'] = df['CodeStateID']
    elif 'ID' in df.columns:
        df = df[~df['ID'].isna()]
    else:
        continue  # Skip files with no ID

    # Ensure 'label' exists; create placeholder if missing
    if 'label' not in df.columns or df['label'].isna().all():
        df['label'] = None

    # Check for essential 'Code' column
    if 'Code' not in df.columns:
        continue  # Skip files without 'Code'

    # Select only necessary columns explicitly
    df_cleaned = df[['ID', 'Code', 'label']].copy()

    # Drop rows missing essential information
    df_cleaned.dropna(subset=['Code', 'ID'], inplace=True)

    # Add cleaned DataFrame to list if not empty
    if not df_cleaned.empty:
        merged_dataframes.append(df_cleaned)

if not merged_dataframes:
    raise ValueError('No valid data found in provided CSV files.')

# Concatenate DataFrames
merged_df = pd.concat(merged_dataframes, ignore_index=True)

# Drop duplicates
merged_df.drop_duplicates(inplace=True)

# Save merged file
merged_df.to_csv('merged_file.csv', index=False)

print(f'Successfully merged {len(merged_dataframes)} CSV files into merged_file.csv, duplicates and incomplete rows removed.')
