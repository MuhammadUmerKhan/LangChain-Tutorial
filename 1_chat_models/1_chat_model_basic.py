import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini-Pro model
model = ChatGoogleGenerativeAI(model="gemini-pro", api_key = api_key)

result = model.invoke("Hello, How are you?")
print("Full Result")
print(result)
print(result.content)
