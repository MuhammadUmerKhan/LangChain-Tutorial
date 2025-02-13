import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
import warnings

warnings.filterwarnings("ignore")

current_dir = os.getcwd()
file_path = os.path.join(current_dir, "books", "odyssey.txt")
presisdent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(presisdent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    print("\n--- Creating embeddings ---")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("\n--- Embedding Created ---")

    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=presisdent_directory)
    print("\n--- Vector store created ---")

else:
    print("Vector store already exists. No need to initialize.")

import shutil
shutil.make_archive('/content/db', 'zip', '/content/db')

from google.colab import files
files.download('/content/db.zip')

"""### 1b_rag_basics"""

current_dir = os.getcwd()
presisdent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory=presisdent_directory, embedding_function=embeddings)

retreiver = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4},
)

query = "Who is Odysseus' wife?"

relevant_docs = retreiver.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

