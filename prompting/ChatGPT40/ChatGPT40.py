import openai
import csv
from dotenv import load_dotenv
import os
import uuid  # Import uuid to generate unique identifiers
import time  # Import time to add a delay

# Get the API key from the environment
load_dotenv()
my_key = os.getenv('GPT_key')

# Initialize OpenAI API key
openai.api_key = my_key
base_url = 'https://api.openai.com/v1'

# Input and output file names
input_file = 'prompting/prompts.txt'
output_file = 'prompting/Qwen/Qwen_Responses.csv'

# Number of times to ask each query (hardcoded as 3)
num_repeats = 3

# Hardcoded max tokens for the response (500 tokens)
max_tokens_per_query = 500

# Read queries from the text file
with open(input_file, 'r', encoding='utf-8') as f:
    queries = [line.strip() for line in f if line.strip()]

# Prepare CSV file for output
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Unique_ID', 'Query', 'Response'])  # CSV header with Unique_ID

    # Process each query for the specified number of repeats
    for repeat in range(num_repeats):
        print(f"Round {repeat + 1}/{num_repeats}")  # Print which round is being processed
       
        for query in queries:
            messages = [
                {"role": "system", "content": "You are an AI assistant knowledgeable in various topics."},
                {"role": "user", "content": query}
            ]
            
            try:
                # Get response from OpenAI ChatGPT-4 model with max tokens set to 500
                response = openai.ChatCompletion.create(
                    model='gpt-4',  # Use the ChatGPT-4 model
                    messages=messages,
                    max_tokens=max_tokens_per_query  # Limit the response to 500 tokens
                )
                assistant_reply = response['choices'][0]['message']['content']
                
                # Generate a unique identifier
                unique_id = str(uuid.uuid4())
                
                # Write to CSV with the unique identifier
                csv_writer.writerow([unique_id, query, assistant_reply])
                print(f"Processed query: {query}")
                
                # Add a delay between requests to avoid rate limit issues
                time.sleep(1)  # 1-second delay between each query

            except openai.error.RateLimitError as e:
                print(f"Rate limit exceeded. Error: {e}")
                break  # Exit the loop if rate limit is exceeded

print(f"All queries processed {num_repeats} times. Responses saved to {output_file}")