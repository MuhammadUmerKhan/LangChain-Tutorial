import os
from huggingface_hub import InferenceClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()
hugging_face_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

current_dir = os.getcwd()
presisdent_directory = os.path.join(current_dir, "db", "chroma_db")

# print(presisdent_directory)
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions
)

db = Chroma(
    persist_directory=presisdent_directory,
    embedding_function=huggingface_embeddings
)

query = "How can I learn more about LangChain?"

retreiver = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

relevant_documents = retreiver.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_documents, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

combined_input = (a
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_documents])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=hugging_face_api_key)

# Convert messages to a string format before passing
messages_str = "System: You are a helpful assistant.\nUser: " + combined_input

result = client.text_generation(messages_str, max_new_tokens=300)

# Display the full result and content only
print("\n--- Generated Response ---")
print(result.strip())  # Ensure the response is cleaned
