# -*- coding: utf-8 -*-
"""3_rag_text_splitting_deep_dive.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17HBX70XClpoPbkookD8FZ7Rm-A_u4iVb
"""

# !unzip /content/books.zip
# !unzip /content/db.zip

!pip install -U langchain_community chromadb tiktoken

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
import warnings

warnings.filterwarnings("ignore")

current_dir = os.getcwd()
books_dir = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(books_dir):
    raise FileNotFoundError(f"The file {books_dir} does not exist. Please check the path.")

loader = TextLoader(books_dir)
documents = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vecotr_space(docs, store_name):

  persistan_directory = os.path.join(db_dir, store_name)

  if not os.path.exists(persistan_directory):
    vector_space = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persistan_directory,
    )
    print(f"--- Finished creating vector store {store_name} ---")
  else:
    print(f"Vector store {store_name} already exists. No need to initialize.")

# 1. Character-based Splitting
# Splits text into chunks based on a specified number of characters.
# Useful for consistent chunk sizes regardless of content structure.
print("\n--- Using Character-based Splitting ---")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vecotr_space(char_docs, "chroma_db_char")

# 2. Sentence-based Splitting
# Splits text into chunks based on sentences, ensuring chunks end at sentence boundaries.
# Ideal for maintaining semantic coherence within chunks.
print("\n--- Using Sentence-based Splitting ---")
sentence_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=100)
sentence_docs = sentence_splitter.split_documents(documents)
create_vecotr_space(sentence_docs, "chroma_db_sentence")

# 3. Token-based Splitting
# Splits text into chunks based on tokens (words or subwords), using tokenizers like GPT-2.
# Useful for transformer models with strict token limits.
print("\n--- Using Token-based Splitting ---")
token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
token_docs = token_splitter.split_documents(documents)
create_vecotr_space(token_docs, "chroma_db_token")

# 4. Recursive Character-based Splitting
# Attempts to split text at natural boundaries (sentences, paragraphs) within character limit.
# Balances between maintaining coherence and adhering to character limits.
print("\n--- Using Recursive Character-based Splitting ---")
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
recursive_docs = recursive_splitter.split_documents(documents)
create_vecotr_space(recursive_docs, "chroma_db_recursive")

# 5. Custom Splitting
# Allows creating custom splitting logic based on specific requirements.
# Useful for documents with unique structure that standard splitters can't handle.
print("\n--- Using Custom Splitting ---")
class CustomSplitter(TextSplitter):
  def split_text(self, text):
    return text.split("\n\n")

custom_splitter = CustomSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vecotr_space(custom_docs, "chroma_db_custom")

# Function to query a vector store
def query_vector_store(store_name, query):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory, embedding_function=embeddings
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.1},
        )
        relevant_docs = retriever.invoke(query)
        # Display the relevant results with metadata
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")

query = "How did Juliet die?"

# Query each vector store
query_vector_store("chroma_db_char", query)
query_vector_store("chroma_db_sent", query)
query_vector_store("chroma_db_token", query)
query_vector_store("chroma_db_rec_char", query)
query_vector_store("chroma_db_custom", query)

import shutil
shutil.make_archive('/content/db', 'zip', '/content/db')

from google.colab import files
files.download('/content/db.zip')