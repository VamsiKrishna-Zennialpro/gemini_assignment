import os
os.environ['GEMINI_API_KEY']="AIzaSyCUrMSvX2p5J8ysffdkSH9yawZaBPHUaRU"

from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What are decorators in python",
)

print(response.text)