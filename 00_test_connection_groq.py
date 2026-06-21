"""
Step 0 (v2): Confirm your Groq API key works.
Run this BEFORE building anything else.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads .env file in the same folder

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Did you create a .env file with "
        "GROQ_API_KEY=your_key_here in this folder?"
    )

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "In one sentence, what is retrieval-augmented generation?"}
    ],
)

print("API key works. Model responded:\n")
print(response.choices[0].message.content)
