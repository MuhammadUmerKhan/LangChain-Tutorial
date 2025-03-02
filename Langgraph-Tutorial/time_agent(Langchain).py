import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_groq import ChatGroq  # ✅ Groq API for LLM
import datetime  # ✅ Import for time and date functionality

# ✅ Load environment variables from .env
load_dotenv()

# ✅ Get API key for Groq LLM
grok_api_key = os.getenv("GROK_API_KEY")

# ✅ Initialize the LLM (Groq API)
llm = ChatGroq(
    temperature=0.3,
    groq_api_key=grok_api_key,
    model_name="qwen-2.5-32b"
)

# ✅ Define a tool for fetching the current time in 12-hour format (AM/PM)
def get_time(_):
    """Returns the current time in 12-hour format (HH:MM:SS AM/PM)."""
    return datetime.datetime.now().strftime("⏰ The current time is %I:%M:%S %p.")

# ✅ Define a tool for fetching today's date
def get_date(_):
    """Returns the current date in Day, DD Month YYYY format."""
    return datetime.datetime.now().strftime("📅 Today's date is %A, %d %B %Y.")

# ✅ Create tools for the agent
tools = [
    Tool(name="TimeTool", func=get_time, description="Use this tool to get the current time."),
    Tool(name="DateTool", func=get_date, description="Use this tool to get today's date."),
]

# ✅ Initialize the agent
agent = initialize_agent(
    tools=tools,  # ✅ Give the agent access to time, date, and self-introduction tools
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ✅ Allow the agent to decide when to use tools
    verbose=True  # ✅ Enable detailed logging (optional)
)

# ✅ Simple chatbot loop with agent integration
print("\n🤖 Chatbot is ready! Type 'exit', 'quit', or 'bye' to stop.\n")

while True:
    user_input = input("You: ")  # Take user input
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Chatbot: Goodbye! Have a great day! 👋")
        break  # Exit loop

    # ✅ Get agent response (agent will use LLM or tools if needed)
    response = agent.invoke(user_input)
    
    # ✅ Print chatbot response
    print(f"🤖 Chatbot: {response['output']}\n")
