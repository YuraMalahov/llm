from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser

API = "*****"

chat_model = ChatOpenAI(temperature=0.5, openai_api_key=API)

# *********************************************
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generated 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chain = LLMChain(llm=chat_model, prompt=chat_prompt)

r = chain.run("colors")
print(r)

# >> ['red', 'blue', 'green', 'yellow', 'orange']
