from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_huggingface import HuggingFaceEndpoint
import datetime
import os

# Set Hugging Face API token
hugging_face_api_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

# Define a tool function that returns the current time
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")  # Ensure clean formatting

# List of tools available to the agent
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time",
    ),
]

# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/react")

# Initialize the Mistral model using Hugging Face API
llm = HuggingFaceEndpoint(
    endpoint_url="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=hugging_face_api_token,
    max_new_tokens=100,  # ✅ Passed explicitly
    temperature=0.7       # ✅ Passed explicitly
)

# Create the ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# Create an agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

# Run the agent with a test query
response = agent_executor.invoke({"input": "What time is it?"})

# Print the response
print("response:", response)