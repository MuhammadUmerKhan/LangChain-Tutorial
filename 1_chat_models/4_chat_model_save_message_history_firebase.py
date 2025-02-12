import os
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini-Pro model (Fix API key parameter)
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

PROJECT_ID = "chat-history-1ccb1"
SESSION_ID = "user_session_new"
COLLECTION_NAME = "chat_history"


print("Initializing Firebase Client...")
client = firestore.Client(project=PROJECT_ID)

print("Initializing Firebase Chat Message History...")
chat_history = FirestoreChatMessageHistory(client=client, 
                                           collection=COLLECTION_NAME, 
                                           session_id=SESSION_ID)

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    query = input("You: ")
    
    if query.lower() == "exit":
        break

    chat_history.add_user_message(query)
    
    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)
    
    print("Chatbot:", ai_response.content)