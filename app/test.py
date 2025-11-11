import google.generativeai as genai
import time
import os
from dotenv import load_dotenv
 
load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

start = time.time()
response = model.generate_content("What is 2+2?")
end = time.time()

print(f"Direct API call: {end-start:.2f}s")