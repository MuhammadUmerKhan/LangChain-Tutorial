import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from huggingface_hub import InferenceClient

# Load API key
load_dotenv()
hugging_face_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hugging_face_api_key:
    raise ValueError("❌ API key not found! Check your .env file.")

# Load Mistral model
model = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=hugging_face_api_key)

# Define prompt template
messages = [
    {"role": "system", "content": "You are a comedian who tells jokes about {topic}."},
    {"role": "user", "content": "Tell me {count} jokes about {topic}."}
]

prompt_template = ChatPromptTemplate.from_messages(messages)

# Create function steps
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))

def call_model(x):
    response = model.chat_completion(messages=[{"role": "user", "content": x.to_string()}])
    return response.choices[0]["message"]["content"]

invoke_model = RunnableLambda(call_model)
parse_output = RunnableLambda(lambda x: x)  # No need to extract `.content`

# ✅ FIXED: Use `RunnableSequence` directly
chain = RunnableSequence(format_prompt, invoke_model, parse_output)

# Invoke chain
response = chain.invoke({"topic": "lawyers", "count": 5})

print("\n--- AI Response ---\n", response)
