from dotenv import load_dotenv
from openai import OpenAI
import os
import csv
import uuid
import time

# Load API key from .env file
load_dotenv()

# Initialize the OpenAI client with the API key
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
def fetch_prompt(prompt, temperature=0.7, top_p=1.0, max_tokens=500, frequency_penalty=0.0, presence_penalty=0.0):
    """Fetch response from OpenAI API for a given prompt with configurable parameters."""
    start_time = time.time()
    completion = client.chat.completions.create(
        model="deepseek/deepseek-chat:free",
        messages=[
            {"role": "system", "content": "You are DeepSeek, an intelligent and engaging AI assistant. You provide accurate, articulate, and context-aware responses while maintaining a friendly and professional tone."},
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
    return response_id, prompt, completion.choices[0].message.content, response_time

def read_prompts_from_file(file_path):
    """Read prompts from a file where each query is separated by '?' and return a list of prompts."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Split content by '?' and filter out empty prompts
    prompts = [prompt.strip() for prompt in content.split('?') if prompt.strip()]
    return prompts

def main(input_file, output_file, limit, repeats):
    """Main function to process prompts and save responses, with a limit on the number of prompts processed."""
    prompts = read_prompts_from_file(input_file)
    
    # Apply limit
    prompts = prompts[:limit]
    print(f"Processing {len(prompts)} prompts with {repeats} repeats each...")
    
    response_times = []
    responses = []
    
    for prompt in prompts:
        for i in range(repeats):
            response_id, repeated_prompt, response, response_time = fetch_prompt(prompt)
            responses.append((response_id, repeated_prompt, response))
            response_times.append(response_time)
            
            if len(response_times) > 1:
                avg_time = sum(response_times) / len(response_times)
                remaining_prompts = (len(prompts) * repeats) - len(response_times)
                estimated_time_remaining = avg_time * remaining_prompts
                print(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds ({remaining_prompts} prompts remaining).")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save responses to a CSV file
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["ID", "Prompt", "Response"])
        for response_id, prompt, response in responses:
            csvwriter.writerow([response_id, prompt, response])
    
    print(f"Responses saved to {output_file}")
    print(f"Total prompts processed: {len(prompts) * repeats} with a time of {sum(response_times):.2f} seconds, on average {sum(response_times) / len(response_times):.2f} seconds per prompt.")
    
# Example usage
input_file = "Prompts.txt"  # Input file containing queries separated by '?'
output_file = "prompting/DeepSeek/responses.csv"
main(input_file, output_file, limit=1, repeats=10)