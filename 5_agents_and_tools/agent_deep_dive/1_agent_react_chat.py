from functools import lru_cache
from wikipedia import summary
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, create_structured_chat_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
from langchain_huggingface import HuggingFaceEndpoint
import datetime
import os


hugging_face_api_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")  

@lru_cache(maxsize=5)  # Store last 5 queries in memory
def search_wikipedia(query):
    try:
        return summary(query, sentences=2)
    except Exception as e:
        return f"I couldn't find any information on that. Error: {str(e)}"
    
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a topic",
    ),
]

prompt = hub.pull("hwchase17/structured-chat-agent")

llm = HuggingFaceEndpoint(
    endpoint_url="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=hugging_face_api_token,
    max_new_tokens=100,
    temperature=0.7
)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,  
    handle_parsing_errors=True,  
)

initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat Loop to interact with the user
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        print("Goodbye! Have a great day. 👋")
        break

    # Add the user's message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({
    "input": user_input, 
    "chat_history": memory.chat_memory.messages  # Ensure memory is used correctly
    })
    print("Bot:", response["output"])

    # Add the agent's response to the conversation memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))