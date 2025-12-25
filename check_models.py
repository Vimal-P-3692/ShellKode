"""Check available Gemini models."""
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

print("Available Gemini models:\n")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"✓ {m.name}")
