import re
import pandas as pd

def extract_code(text):
    # First, extract code within triple backticks
    code_blocks = re.findall(r'```(?:[a-zA-Z0-9+]*\n)?(.*?)```', text, re.DOTALL)
    if code_blocks:
        cleaned_blocks = []
        for block in code_blocks:
            # Remove any accidental language specifier lines
            block_clean = re.sub(r'^[a-zA-Z0-9+]+\s*\n', '', block.strip())
            cleaned_blocks.append(block_clean.strip())
        return '\n\n'.join(cleaned_blocks).strip()

    # Fallback heuristic with explanatory detection (ignoring comments)
    lines = text.split('\n')
    code_lines = []
    capturing = False

    # Patterns
    explanatory_pattern = re.compile(r'^\s*(In\s+the|This\s+way|You\s+can|Thus,|Therefore,|So,|Hence,|Finally,|As\s+such|If\s+you|Based\s+on|We\s+can|To\s+solve|Alternatively,|\*|-)')
    comment_pattern = re.compile(r'^\s*(//|/\*|\*|#)')
    code_start_pattern = re.compile(
        r'^\s*(public|private|protected|static|class|def|import\s|\w+\s+\w+\s*\(.*\)\s*\{|[#@]|[\}\{])'
    )

    for line in lines:
        if capturing:
            if explanatory_pattern.match(line.strip()) and not comment_pattern.match(line.strip()):
                # Stop capturing when explanatory text (but not comment) is found
                break
            else:
                code_lines.append(line)
        elif code_start_pattern.match(line.strip()):
            capturing = True
            code_lines.append(line)

    return '\n'.join(code_lines).strip() if code_lines else ''

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
    columns_to_keep = [col for col in ['ID', 'Prompt', 'Extracted_Code'] if col in df.columns]
    df[columns_to_keep].to_csv(output_file, index=False)
    print(f"Processed file saved to {output_file}")

# Process file and save the output
process_csv("prompting/Qwen/responses.csv", "prompting/Qwen/processed_responses.csv")
