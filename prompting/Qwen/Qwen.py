from dotenv import load_dotenv
import openai
import os
import csv
import uuid
import time

# Load API key from .env file
load_dotenv()
my_key = os.getenv("Qwen_key")  # Keep this as your AI/ML API Gateway key

# Initialize AI/ML API Gateway client
client = openai.Client(
    api_key=my_key,
    base_url='https://api.aimlapi.com/v1',  # Ensure this is the correct endpoint for the AI/ML API Gateway
)

def fetch_prompt(prompt, temperature=0.7, top_p=1.0, max_tokens=500, frequency_penalty=0.0, presence_penalty=0.0):
    """Fetch response from AI/ML API Gateway for a given prompt with configurable parameters."""
    start_time = time.time()
    completion = client.chat.completions.create(
        model="qwen/qvq-72b-preview",  # Change model to GPT-4
        messages=[
            {"role": "system", "content": "You are an AI assistant knowledgeable in various topics."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    end_time = time.time()
    response_time = end_time - start_time
    print(f"Processed prompt in {response_time:.2f} seconds")
    response_id = str(uuid.uuid4())
    return response_id, prompt, completion.choices[0].message['content'], response_time

def read_prompts_from_file(file_path):
    """Read prompts from a file where each query is separated by '?' and return a list of prompts."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Split content by '?' and filter out empty prompts
    prompts = [prompt.strip() for prompt in content.split('?') if prompt.strip()]
    return prompts

def main(input_file, output_file, limit, repeats, start_point=0):
    """Main function to process prompts and save responses, with a limit on the number of prompts processed."""
    prompts = read_prompts_from_file(input_file)
    
    # Apply limit and starting point
    prompts = prompts[start_point:start_point + limit]
    print(f"Processing {len(prompts)} prompts with {repeats} repeats each, starting from index {start_point}...")
    
    response_times = []
    
    # Open the CSV file for writing responses one by one
    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        if os.stat(output_file).st_size == 0:
            csvwriter.writerow(["ID", "Prompt", "Response"])  # Write header row if file is empty

        for index, prompt in enumerate(prompts, start=start_point):
            for i in range(repeats):
                response_id, repeated_prompt, response, response_time = fetch_prompt(prompt)
                
                # Immediately save response to CSV after receiving it
                csvwriter.writerow([response_id, repeated_prompt, response])
                
                response_times.append(response_time)

                if len(response_times) > 1:
                    avg_time = sum(response_times) / len(response_times)
                    remaining_prompts = (len(prompts) * repeats) - len(response_times)
                    estimated_time_remaining = avg_time * remaining_prompts
                    print(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds ({remaining_prompts} prompts remaining).")
    
    print(f"Responses saved to {output_file}")
    print(f"Total prompts processed: {len(prompts) * repeats} with a time of {sum(response_times):.2f} seconds")
    
# Example usage
input_file = "Prompts.txt"  # Input file containing queries separated by '?'
output_file = "prompting/ChatGPT/responses.csv"
main(input_file, output_file, limit=1, repeats=10, start_point=0)