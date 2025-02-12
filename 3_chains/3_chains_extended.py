import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()
hugging_face_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Validate API key
if not hugging_face_api_key:
    raise ValueError("❌ API key not found! Check your .env file.")

# Initialize Hugging Face model client
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=hugging_face_api_key)

# Define prompt template
prompt_template = ChatPromptTemplate.from_template(
    "You are a comedian who tells jokes about {topic}.\nTell me {count} jokes about {topic}."
)

# Model call function (extracting text properly)
def call_model(input_text):
    prompt = str(input_text)  # Ensure it's a string
    response = client.text_generation(prompt, max_new_tokens=200)
    return response  # Return generated text directly

# Wrap function in RunnableLambda
invoke_model = RunnableLambda(call_model)

# Post-processing functions
upper_case = RunnableLambda(lambda x: x.upper())  # Convert response to uppercase
count_words = RunnableLambda(lambda x: f"Word Count: {len(x.split())}\n{x}")  # Count words

# Build the chain
chain = prompt_template | invoke_model | upper_case | count_words

# Run the chain
result = chain.invoke({"topic": "Lawyer", "count": 3})

# Print final result
print("\n--- AI Response ---\n", result)
