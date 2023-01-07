import os
from dotenv import load_dotenv

#load sourced environment variables
load_dotenv()

#read environment variable
GPT_KEY = os.getenv('GPT_KEY')

print(GPT_KEY)
