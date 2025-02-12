import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini-Pro model (Fix API key parameter)
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

# Store conversation history
chat_history = []

while True:
    query = input("You: ")
    
    if query.lower() == "exit":
        break

    # Append user message
    chat_history.append(HumanMessage(content=query))

    # Keep only last 5 messages to prevent history overflow
    chat_history = chat_history[-5:]

    # Invoke the model
    result = model.invoke(chat_history)
    response = result.content

    # Append AI response
    chat_history.append(AIMessage(content=response))

    print("Chatbot:", response)

print("Chat ended.")
