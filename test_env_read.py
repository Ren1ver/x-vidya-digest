import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch the API key
api_key = os.getenv("OPENAI_API_KEY")

# Check if it was loaded successfully
if api_key:
    print("✅ OPENAI_API_KEY was successfully read from .env")
    print(f"Value (partially hidden): {api_key[:4]}...{api_key[-4:]}")
else:
    print("❌ OPENAI_API_KEY was NOT found. Please check your .env file and its path.")
