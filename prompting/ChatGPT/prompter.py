from dotenv import load_dotenv
from openai import OpenAI
import os

# Load API key from .env file
load_dotenv()

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fetch_prompt(prompt):
    # Call the ChatGPT API using the client interface
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    # Corrected to access the message content correctly
    return completion.choices[0].message.content

def main(prompts):
    results = [fetch_prompt(prompt) for prompt in prompts]
    return results

prompts = ["Explain recursion.", "Describe Newton's laws."]  # Add more prompts

responses = main(prompts)

# Save the responses to a file
os.makedirs("prompting/ChatGPT", exist_ok=True)  # Ensure the directory exists
with open("prompting/ChatGPT/responses.txt", "w") as f:
    for response in responses:
        f.write(response + "\n")

print("Responses saved to responses.txt")
