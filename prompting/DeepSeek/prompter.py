from dotenv import load_dotenv
from openai import OpenAI
import os
import csv
import uuid  # To generate unique IDs

# Load API key from .env file
load_dotenv()

# Initialize the OpenAI client with OpenRouter API settings
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def fetch_prompt(prompt):
    # Call the DeepSeek/OpenRouter API using the client interface
    completion = client.chat.completions.create(
        model="deepseek/deepseek-chat:free", # Use the free model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    # Generate a unique ID for this response
    response_id = str(uuid.uuid4())
    # Return the ID, prompt, and response content
    return response_id, prompt, completion.choices[0].message.content

def main(prompts):
    results = [fetch_prompt(prompt) for prompt in prompts]
    return results

prompts = ["Explain recursion.", "Describe Newton's laws."]  # Add more prompts

responses = main(prompts)

# Ensure the directory exists
os.makedirs("prompting/DeepSeek", exist_ok=True)

# Save the responses to a CSV file
with open("prompting/DeepSeek/responses.csv", "w", newline="", encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header row
    csvwriter.writerow(["ID", "Prompt", "Response"])
    # Write each ID, prompt, and response
    for response_id, prompt, response in responses:
        csvwriter.writerow([response_id, prompt, response])

print("Responses saved to responses.csv")
