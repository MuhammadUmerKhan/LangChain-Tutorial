from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# PART 1: Create a ChatChatPromptTemplate using a template string
# template = "Tell me a joke about {topic}."
# prompt_template = ChatPromptTemplate.from_template(template)

# print("-----Prompt from Template-----")
# prompt = prompt_template.invoke(
#     {"topic":"cat"}
# )
# print(prompt) # text='Tell me a joke about cat.'


#  ----------------------------------------------------------------
# # PART 2: Prompt with Multiple Placeholders
# template_multiple_placeholders = """You are a helpful assistant
# Human: Tell me a {adjective} story about {animal}.
# Assistant:"""

# prompt_template_multiple_placeholders = ChatPromptTemplate.from_template(template_multiple_placeholders)

# print("-----Prompt with Multiple Placeholders-----")
# prompt = prompt_template_multiple_placeholders.invoke(
#     {"adjective":"funny", "animal":"cat"}
# )
# print("\n----- Prompt with Multiple Placeholders -----\n")
# print(prompt)


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
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)