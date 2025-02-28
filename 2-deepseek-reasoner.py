import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Validate required environment variables
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("DEEPSEEK_API_KEY is not set in the environment variables.")

# Initialize the client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# Create chat completion
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Answer as brief as possible."},
        {"role": "user", "content": "In which US state is the city of Ryolithe?"},
    ],
    stream=False
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print("Reasoning: \n", reasoning_content)
print("------------")
print("Answer: \n", response.choices[0].message.content)