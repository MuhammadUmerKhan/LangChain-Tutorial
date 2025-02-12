import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini-Pro model
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

# Only use HumanMessage
messages = [
    HumanMessage(content="Imagine you are a future predictor. Will AI replace our jobs?"),
]

# Invoke the model
result = model.invoke(messages)
print("Full Result:")
print(result)
print("Content Only:")
print(result.content)
