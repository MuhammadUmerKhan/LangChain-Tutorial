import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnableBranch
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
    if hasattr(prompt, "to_string"):
        prompt = prompt.to_string()
    response = client.text_generation(prompt, max_new_tokens=300)
    return response.strip()



positive_feedback_template = ChatPromptTemplate.from_template(
    "You are a helpful assistant.\nGenerate a thank you note for this positive feedback: {feedback}."
)

negative_feedback_template = ChatPromptTemplate.from_template(
    "You are a helpful assistant.\nGenerate a response addressing this negative feedback: {feedback}."
)

neutral_feedback_template = ChatPromptTemplate.from_template(
    "You are a helpful assistant.\nGenerate a request for more details for this neutral feedback: {feedback}."
)

escalate_feedback_template = ChatPromptTemplate.from_template(
    "You are a helpful assistant.\nGenerate a message to escalate this feedback to a human agent: {feedback}."
)

invoke_model = RunnableLambda(call_mistral)

classification_template = ChatPromptTemplate.from_template(
    "You are a helpful assistant.\nClassify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."
)
branches = RunnableBranch(
    (lambda x: x.lower().strip() == "positive", positive_feedback_template | invoke_model | StrOutputParser()),
    (lambda x: x.lower().strip() == "negative", negative_feedback_template | invoke_model | StrOutputParser()),
    (lambda x: x.lower().strip() == "neutral", neutral_feedback_template | invoke_model | StrOutputParser()),
    escalate_feedback_template | invoke_model | StrOutputParser()
)

classification_chain = classification_template | invoke_model | StrOutputParser()
chain = classification_chain | branches 

review = "The product is terrible. It broke after just one use and the quality is very poor."
result = chain.invoke({"feedback":review})
print("\n--- AI Response ---\n", result)