from google import genai
from google.genai.types import GenerateContentConfig
import os

# Initialize with your API key (you have multiple options):

# Option 1: Directly set the API key
GOOGLE_API_KEY = "your-api-key-here"
genai.configure(api_key=GOOGLE_API_KEY)

# Option 2: Using environment variable (make sure GOOGLE_API_KEY is set)
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Option 3: Using dotenv (your current approach)
# from dotenv import load_dotenv
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Now create client
client = genai.Client()

# Example: Analyze job listings from a Naukri search
naukri_url = "https://www.naukri.com/jobs-in-delhi"

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"Extract job titles, companies, and locations from this job search: {naukri_url}",
    config=GenerateContentConfig(
        tools=[{"url_context": {}}],
        temperature=0.1
    )
)

for each in response.candidates[0].content.parts:
    print(each.text)