"""
One-off script: lists the Gemini models your API key has access to.
Run this once to pick a model name for generate.py.
"""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("Models available to your key:\n")
for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(f"  {model.name}")