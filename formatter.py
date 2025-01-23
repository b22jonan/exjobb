import pandas as pd

# File paths for the provided data
code_states_path = 'CodeStates.csv'
main_table_path = 'MainTable.csv'
early_path = 'early.csv'
late_path = 'late.csv'

# Load the necessary datasets
code_states = pd.read_csv(code_states_path)
main_table = pd.read_csv(main_table_path)
early_data = pd.read_csv(early_path)
late_data = pd.read_csv(late_path)

# Merge MainTable with CodeStates to get code snippets
merged_data = main_table.merge(code_states, on='CodeStateID', how='inner')

# Add labels from early and late datasets
# Early data contains labels for the first 30 problems
merged_data = merged_data.merge(
    early_data[['SubjectID', 'ProblemID', 'Label']],
    on=['SubjectID', 'ProblemID'],
    how='left'
)

# Late data contains labels for the final 20 problems
merged_data = merged_data.merge(
    late_data[['SubjectID', 'ProblemID', 'Label']],
    on=['SubjectID', 'ProblemID'],
    how='left',
    suffixes=('_early', '_late')
)

# Combine early and late labels into a single column
merged_data['Label'] = merged_data['Label_early'].combine_first(merged_data['Label_late'])

# Filter relevant columns for the final dataset
final_dataset = merged_data[['Code', 'ProblemID', 'Label']]

# Sort the final dataset by ProblemID
final_dataset = final_dataset.sort_values(by='ProblemID').reset_index(drop=True)

# Save the final dataset
output_path = 'FinalDataset.csv'
final_dataset.to_csv(output_path, index=False)

print(f"Final dataset saved to: {output_path}")
