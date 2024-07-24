import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env
API_KEY = os.environ.get('COMET_API_KEY')
print(API_KEY)
print(type(API_KEY))