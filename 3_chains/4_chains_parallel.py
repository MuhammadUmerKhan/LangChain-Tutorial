import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()
hugging_face_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Validate API key
if not hugging_face_api_key:
    raise ValueError("❌ API key not found! Check your .env file.")

# Initialize Mistral model client
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=hugging_face_api_key)

# Define a function to call the Mistral model
def call_mistral(prompt):
    response = client.text_generation(prompt, max_new_tokens=300)
    return response.strip()  # Clean response

# Define the model wrapper as a RunnableLambda
mistral_model = RunnableLambda(call_mistral)

# Define the main product review prompt
prompt_template = ChatPromptTemplate.from_template(
    "You are an expert product reviewer.\nList the main features of the product: {product_name}."
)

# Define the pros analysis step
def analyze_pros(features):
    return f"You are an expert product reviewer.\nGiven these features: {features}, list the pros of these features."

# Define the cons analysis step
def analyze_cons(features):
    return f"You are an expert product reviewer.\nGiven these features: {features}, list the cons of these features."

# Combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

# Define branches for pros and cons
pros_branch_chain = RunnableLambda(analyze_pros) | mistral_model | StrOutputParser()
cons_branch_chain = RunnableLambda(analyze_cons) | mistral_model | StrOutputParser()

# Convert ChatPromptTemplate to a string before passing to Mistral
chain = (
    RunnableLambda(lambda x: prompt_template.format(product_name=x["product_name"]))  # Convert template to string
    | mistral_model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# Run the chain
result = chain.invoke({"product_name": "MacBook Pro"})

# Output
print("\n--- AI Response ---\n", result)
