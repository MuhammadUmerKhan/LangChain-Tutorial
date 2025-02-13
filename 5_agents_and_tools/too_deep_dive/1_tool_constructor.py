# Import necessary libraries
from langchain import hub
from langchain_huggingface import HuggingFaceEndpoint
from langchain.agents import AgentExecutor, create_react_agent  # ✅ Use ReAct instead
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool, Tool
import os

# Functions for the tools
def greet_user(name: str) -> str:
    """Greets the user by name."""
    return f"Hello, {name}!"

def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]

def concatenate_strings(a: str, b: str) -> str:
    """Concatenates two strings."""
    return a + b

# Pydantic model for tool arguments
class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="First string")
    b: str = Field(description="Second string")

# Create tools
tools = [
    Tool(name="GreetUser", func=greet_user, description="Greets the user by name."),
    Tool(name="ReverseString", func=reverse_string, description="Reverses the given string."),
    StructuredTool.from_function(
        func=concatenate_strings,
        name="ConcatenateStrings",
        description="Concatenates two strings.",
        args_schema=ConcatenateStringsArgs,
    ),
]

# Load Hugging Face API token
hugging_face_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if not hugging_face_api_token:
    raise ValueError("Hugging Face API token is missing. Set it as an environment variable.")

# Initialize Hugging Face LLM
llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=hugging_face_api_token,
    max_new_tokens=100,
    temperature=0.7,
)

# Load ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

# Create the ReAct agent (compatible with Hugging Face models)
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# Test the agent with sample queries
response = agent_executor.invoke({"input": "Greet Alice"})
print("Response for 'Greet Alice':", response)

response = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("Response for 'Reverse the string hello':", response)

response = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("Response for 'Concatenate hello and world':", response)
