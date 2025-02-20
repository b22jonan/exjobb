from dotenv import load_dotenv
from openai import OpenAI
import os
import csv
import uuid
import time

# Load API key from .env file
load_dotenv()

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fetch_prompt(prompt, temperature=0.7, top_p=1.0, max_tokens=500, frequency_penalty=0.0, presence_penalty=0.0):
    """Fetch response from OpenAI API for a given prompt with configurable parameters."""
    start_time = time.time()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are ChatGPT, an intelligent and engaging AI assistant. You provide accurate, articulate, and context-aware responses while maintaining a friendly and professional tone."},
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


def main(input_file, output_file, limit, repeats, start_index=0):
    """Main function to process prompts and save responses, with support for resuming from an exact point."""
    prompts = read_prompts_from_file(input_file)
    
    start_prompt_index = start_index // repeats  # Compute the starting prompt index
    start_repeat_index = start_index % repeats  # Compute the starting repeat index
    
    # Apply limit and start index
    prompts = prompts[start_prompt_index:start_prompt_index + limit]
    print(f"Processing {len(prompts)} prompts with {repeats} repeats each, starting from global index {start_index} (prompt {start_prompt_index}, repeat {start_repeat_index})...")
    
    response_times = []
    
    # Open the CSV file for writing responses one by one
    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        if os.stat(output_file).st_size == 0:
            csvwriter.writerow(["ID", "Prompt", "Response"])  # Write header row if file is empty

        for prompt_index, prompt in enumerate(prompts, start=start_prompt_index):
            repeat_start = start_repeat_index if prompt_index == start_prompt_index else 0  # Ensure repeat resumes correctly
            for repeat_index in range(repeat_start, repeats):
                response_id, repeated_prompt, response, response_time = fetch_prompt(prompt)
                
                # Immediately save response to CSV after receiving it
                csvwriter.writerow([response_id, repeated_prompt, response])
                
                response_times.append(response_time)

                if len(response_times) > 1:
                    avg_time = sum(response_times) / len(response_times)
                    remaining_prompts = ((len(prompts) - (prompt_index - start_prompt_index)) * repeats) - (repeat_index + 1)
                    estimated_time_remaining = avg_time * remaining_prompts
                    print(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds ({remaining_prompts} prompts remaining).")
    
    print(f"Responses saved to {output_file}")
    print(f"Total prompts processed: {((len(prompts) - start_prompt_index) * repeats) - start_repeat_index} with a time of {sum(response_times):.2f} seconds")
    
# Example usage
input_file = "Prompts.txt"  # Input file containing queries separated by '?'
output_file = "prompting/ChatGPT35/responses.csv"
main(input_file, output_file, limit=300, repeats=10)