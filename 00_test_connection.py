"""
Step 0: Confirm your Gemini API key works.
Run this BEFORE building anything else. If this fails, nothing downstream will work.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()  # reads .env file in the same folder

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found. Did you create a .env file with "
        "GOOGLE_API_KEY=your_key_here in this folder?"
    )

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash")

response = model.generate_content(
    "In one sentence, what is retrieval-augmented generation?"
)

print("API key works. Model responded:\n")
print(response.text)
