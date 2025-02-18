import re
import pandas as pd

def extract_code(text):
    # Try to extract code within triple backticks first
    code_blocks = re.findall(r'```(?:[a-zA-Z0-9+]*\n)?(.*?)```', text, re.DOTALL)
    
    if code_blocks:
        extracted = '\n'.join(code_blocks).strip()
    else:
        # If no triple backticks, attempt to identify code heuristically
        lines = text.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            # Detect potential start of code (heuristics)
            if re.match(r'^[a-zA-Z0-9_\s]+\(.*\)\s*\{?', line):  # Looks like function/class definition
                in_code_block = True
            
            if in_code_block:
                code_lines.append(line)
        
        extracted = '\n'.join(code_lines).strip() if code_lines else text
    
    return extracted

def process_csv(file_path, output_file):
    df = pd.read_csv(file_path)
    
    # Assuming responses are in a column named 'Response'. Adjust if needed.
    if 'Response' not in df.columns:
        print("Error: 'Response' column not found in CSV file.")
        return
    
    df['Extracted_Code'] = df['Response'].apply(extract_code)
    
    # Debugging: Print first few rows to check what is happening
    print("Before removing language specifier:")
    print(df[['ID', 'Extracted_Code']].head(10))
    
    # Remove language specifier (e.g., "java") if it appears at the start of extracted code
    df['Extracted_Code'] = df['Extracted_Code'].str.replace(r'^[a-zA-Z0-9+]+\s*\n', '', regex=True).str.strip()
    
    # Debugging: Print first few rows after processing
    print("After removing language specifier:")
    print(df[['ID', 'Extracted_Code']].head(10))
    
    # Keep ID and Extracted_Code columns
    columns_to_keep = [col for col in ['ID', 'Extracted_Code'] if col in df.columns]
    df[columns_to_keep].to_csv(output_file, index=False)
    print(f"Processed file saved to {output_file}")

# Process both files
process_csv('prompting/ChatGPT/responses.csv', 'prompting/ChatGPT/processed_responses.csv')
