import pandas as pd

# Load the CSV file with proper handling for multiline entries
file_path = 'CodeStates.csv'  # Adjusted for project root
data = pd.read_csv(file_path, header=None, names=['ID', 'Code'], quoting=3, skip_blank_lines=True, on_bad_lines='skip')

# Split data into entries by looking for rows where 'ID' is not NaN
entries = []
current_entry = []
for _, row in data.iterrows():
    if pd.notna(row['ID']):
        if current_entry:
            entries.append(current_entry)
        current_entry = [row['ID'], row['Code'] if pd.notna(row['Code']) else ""]
    else:
        if current_entry:
            current_entry[1] += f"\n{row['Code']}" if pd.notna(row['Code']) else ""

# Add the last entry
if current_entry:
    entries.append(current_entry)

# Convert entries into a DataFrame
formatted_data = pd.DataFrame(entries, columns=['ID', 'Code'])

# Number of labels and data points per label
num_labels = 10
entries_per_label = 500

# Verify the data length matches the expected total entries
num_entries = len(entries)  # Count the number of processed entries, not rows
if num_entries != num_labels * entries_per_label:
    raise ValueError(f"Expected {num_labels * entries_per_label} entries, but found {num_entries} entries in the dataset.")

# Create labels
labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
for i in range(num_labels):
    label_name = f"Label {i + 1}"
    labels.extend([label_name] * entries_per_label)

# Add the labels to the dataset
formatted_data['Label'] = labels

# Save the reformatted dataset to a new CSV file
output_path = 'Reformatted_CodeStates.csv'  # Adjusted for project root
formatted_data.to_csv(output_path, index=False)

print(f"Reformatted dataset saved to: {output_path}")
