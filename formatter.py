import pandas as pd

# File paths for the provided data
code_states_path = 'CodeStates.csv'
main_table_path = 'MainTable.csv'

# Load the necessary datasets
code_states = pd.read_csv(code_states_path)
main_table = pd.read_csv(main_table_path)

# Merge MainTable with CodeStates to get code snippets
merged_data = main_table.merge(code_states, on='CodeStateID', how='inner')

# Filter relevant columns for the final dataset
final_dataset = merged_data[['Code', 'ProblemID']]

# Allow specifying how many problems to include (default: first 5 problems)
# The maximum number of problems is around 250, but it may vary slightly depending on the dataset.
num_problems = 20  # Change this value as needed
unique_problems = final_dataset['ProblemID'].drop_duplicates().sort_values().head(num_problems)
final_dataset = final_dataset[final_dataset['ProblemID'].isin(unique_problems)]

# Allow specifying max number of entries per problem
max_entries_per_problem = 500  # Change this value as needed
final_dataset = final_dataset.groupby('ProblemID').apply(lambda x: x.head(max_entries_per_problem)).reset_index(drop=True)

# Sort the final dataset by ProblemID
final_dataset = final_dataset.sort_values(by='ProblemID').reset_index(drop=True)

# Print dataset information
total_entries = len(final_dataset)
problem_counts = final_dataset['ProblemID'].value_counts()
print(f"Total entries in the dataset: {total_entries}")
print("Entries per ProblemID:")
print(problem_counts)

# Save the final dataset
output_path = 'FinalDataset.csv'
final_dataset.to_csv(output_path, index=False)

print(f"Final dataset saved to: {output_path}")
