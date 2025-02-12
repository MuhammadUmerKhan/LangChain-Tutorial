import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from huggingface_hub import InferenceClient

# Load API key
load_dotenv()
hugging_face_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hugging_face_api_key:
    raise ValueError("❌ API key not found! Check your .env file.")

# Load Mistral model correctly
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=hugging_face_api_key)

# Define a simple text-based prompt (No structured messages)
prompt_template = ChatPromptTemplate.from_template(
    "You are a comedian who tells jokes about {topic}.\nTell me {count} jokes about {topic}."
)

# Wrap the model in a RunnableLambda
def call_model(input_text):
    prompt = input_text.to_string()  # Convert to a plain text string
    response = client.text_generation(prompt, max_new_tokens=200)  # Use text_generation
    return response

invoke_model = RunnableLambda(call_model)

# Create the chain correctly
chain = prompt_template | invoke_model | StrOutputParser()

# Run the chain with correct input
result = chain.invoke({"topic": "lawyers", "count": 3})

# Output the response
print("\n--- AI Response ---\n", result)
