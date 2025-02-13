from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
import os
from pathlib import Path

# Load Hugging Face API token
hugging_face_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not hugging_face_api_token:
    raise ValueError("Hugging Face API token is missing. Set it as an environment variable.")

# Define the persistent database directory
persistent_directory = Path("../../4_rags/db/chroma_db_with_metadata").resolve()
if not persistent_directory.exists():
    raise FileNotFoundError(f"The directory {persistent_directory} does not exist. Please check the path.")

# Load vector embeddings and ChromaDB
print("Loading Existing vector store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=str(persistent_directory), embedding_function=embeddings)

# Define Retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define Hugging Face LLM with LangChain's HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=hugging_face_api_token
)

# Contextualization Prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# QA System Prompt
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Define the retrieval-augmented question-answering (RAG) chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Load ReAct prompt from LangChain Hub
react_docstore_prompt = hub.pull("hwchase17/react")

# Define tool for answering questions
tools = [
    Tool(
        name="Answer Question",
        func=lambda input, chat_history=[]: rag_chain.invoke({"input": input, "chat_history": chat_history}),
        description="Useful for when you need to answer questions about the context",
    )
]

# Create ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstore_prompt
)

# Create Agent Executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# Start interactive loop
chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke({"input": query, "chat_history": chat_history})
    print(f"AI: {response['output']}")

    # Update chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))
