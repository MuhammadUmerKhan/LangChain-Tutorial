import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.llms import HuggingFaceHub

# Load API key from .env
load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize Gemini-Pro model (Fix API key parameter)
# model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=api_key
)

# # PART 1: Create a ChatChatPromptTemplate using a template string
# template = "Tell me a joke about {topic}."
# prompt_template = ChatPromptTemplate.from_template(template)

# print("-----Prompt from Template-----")
# prompt = prompt_template.invoke(
#     {"topic":"cat"}
# )
# print(prompt) # messages=[HumanMessage(content='Tell me a joke about cat.', additional_kwargs={}, response_metadata={})]
# result = model.invoke(prompt)
# print(result.content)


#  ----------------------------------------------------------------
# # PART 2: Prompt with Multiple Placeholders
# print("\n----- Prompt with Multiple Placeholders -----\n")
# template_multiple = """You are a helpful assistant.
# Human: Tell me a {adjective} short story about a {animal}.
# Assistant:"""
# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)      # use for Gemini, GPT
# prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"}) # prompt for Gemini and GPT

# prompt_multiple = PromptTemplate(template=template_multiple) # use for LLMs from hugging face
# prompt = prompt_multiple.format(adjective="funny", animal="panda")

# result = llm.invoke(prompt)
# print(result)

#  ----------------------------------------------------------------
# PART 3: Prompt with System and Human Messages (Using Tuples)
# message = [
#     ("system", "You are a commedian who tells joke about {topic}."),
#     ("human", "Tell me a {adjective} story about cats.")
# ]

# prompt_template_system_human = ChatPromptTemplate.from_messages(message)
# prmpt = prompt_template_system_human.invoke({"topic": "cat", "adjective":"funny"})
# print("\n----- Prompt with System and Human Messages (Tuples) -----\n")
# print(prmpt)

prompt_multiple = ChatPromptTemplate.from_messages([
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me a {adjective} story about {animal}.")
])

# Format the prompt with variables
prompt = prompt_multiple.format(topic="cats", adjective="funny", animal="cats")

# Generate response
result = llm.invoke(prompt)
print("\n--- AI Response ---\n", result)

#  ----------------------------------------------------------------
# # Extra Informoation about Part 3.
# This does work:
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}."),
#     HumanMessage(content="Tell me 3 jokes."),
# ]
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "lawyers"})
# print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
# print(prompt)


#  ----------------------------------------------------------------
# This does NOT work:
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}."),
#     HumanMessage(content="Tell me {joke_count} jokes."),
# ]
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
# print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
# print(prompt)