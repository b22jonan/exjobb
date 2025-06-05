import pandas as pd
import re
import time

def extract_function_name(code):
    """Extracts function name from Python or Java-style function signatures."""
    match = re.search(r'def ([a-zA-Z_][a-zA-Z0-9_]*)|public [^ ]+ ([a-zA-Z_][a-zA-Z0-9_]*)', code)
    return match.group(1) if match and match.group(1) else (match.group(2) if match else "Unknown")

def sample_student_dataset(input_file, output_file, num_samples_per_problem=60, total_samples=3000):
    """Samples a reduced student dataset while maintaining problem diversity."""
    # Load student dataset
    data = pd.read_csv(input_file, header=None, names=["CodeStateID", "Code"], dtype=str)
    data.dropna(inplace=True)
    
    # Extract function names for problem grouping
    data["FunctionName"] = data["Code"].apply(extract_function_name)
    
    # Ensure we have at least 50 unique problems
    unique_problems = data["FunctionName"].nunique()
    if unique_problems < 50:
        print(f"Warning: Only {unique_problems} unique problems detected! Check function extraction.")
    
    # Sample up to num_samples_per_problem per function name
    sampled_data = data.groupby("FunctionName", group_keys=False, as_index=False).apply(
        lambda x: x.sample(min(len(x), num_samples_per_problem), random_state=42)
    )
    
    # Ensure total dataset does not exceed total_samples
    sampled_data = sampled_data.sample(n=min(total_samples, len(sampled_data)), random_state=42)
    
    # Drop function name column before saving
    sampled_data.drop(columns=["FunctionName"], inplace=True)
    
    # Attempt to save the dataset with retry mechanism
    for attempt in range(5):  # Retry up to 5 times
        try:
            sampled_data.to_csv(output_file, index=False)
            print(f"Sampled dataset saved to {output_file} with {len(sampled_data)} samples.")
            break  # Exit loop if successful
        except OSError:
            print(f"Write attempt {attempt + 1} failed. Retrying...")
            time.sleep(1)  # Wait before retrying
    else:
        print("Failed to save the sampled dataset after multiple attempts.")

# Run the function to create the reduced dataset
if __name__ == "__main__":
    sample_student_dataset("CSV_files/CodeStates.csv", "CSV_files/Sampled_CodeStates.csv")
