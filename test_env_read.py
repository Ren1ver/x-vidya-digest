import os
from pathlib import Path
from dotenv import load_dotenv

# Load the .env file from the script's folder
env_path = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=env_path)

# Fetch the variable
url = os.getenv("DISCORD_WEBHOOK_URL")

# Debug print
print("DISCORD_WEBHOOK_URL:", url if url else "[NOT FOUND]")
