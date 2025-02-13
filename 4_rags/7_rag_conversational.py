import os
import warnings
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env
load_dotenv()
hugging_face_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define the persistent directory for ChromaDB
current_dir = os.getcwd()
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Load embedding model (384 dimensions)
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Chroma vector store
db = Chroma(persist_directory=persistent_directory, embedding_function=huggingface_embeddings)

# Define the retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Define InferenceClient to connect to local TGI server

client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=hugging_face_api_key)

# Function to query the model
def query_model(prompt):
    response = client.text_generation(prompt, max_new_tokens=300, temperature=0.7)
    return response.strip()

# Function to handle chat
def chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        # Retrieve relevant documents
        relevant_documents = retriever.invoke(user_input)

        # Display retrieved documents
        print("\n--- Relevant Documents ---")
        for i, doc in enumerate(relevant_documents, 1):
            print(f"Document {i}:\n{doc.page_content}\n")

        # Combine retrieved context with the user query
        combined_input = (
            "Here are some documents that might help answer the question:\n"
            + user_input
            + "\n\nRelevant Documents:\n"
            + "\n\n".join([doc.page_content for doc in relevant_documents])
            + "\n\nPlease provide an answer based only on the provided documents. "
            "If the answer is not found in the documents, respond with 'I'm not sure'."
        )

        # Convert messages to a string format before passing to the model
        messages_str = "System: You are a helpful assistant.\nUser: " + combined_input

        # Generate response using the local inference client
        result = query_model(messages_str)

        # Print the AI's response
        print("\n--- AI Response ---")
        print(result)

        # Store chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(SystemMessage(content=result))

# Start chat
if __name__ == "__main__":
    chat()
